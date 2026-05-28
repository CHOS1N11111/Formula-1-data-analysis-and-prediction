import csv
import json
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter


BASE_DIR = Path(__file__).resolve().parent
FEATURES_PATH = BASE_DIR / "data" / "processed" / "f1_features.csv"
SQLITE_PATH = BASE_DIR / "formula-1-race-data-sqlite" / "Formula1.sqlite"
JOLPICA_RAW_DIR = BASE_DIR / "data" / "raw" / "jolpica"
VIDEO_DIR = BASE_DIR / "outputs" / "videos"
MANIFEST_PATH = VIDEO_DIR / "video_manifest.json"

START_SEASON = 2019
END_SEASON = 2025
CONSTRUCTOR_VIDEO_START_SEASON = 1950
CONSTRUCTOR_VIDEO_END_SEASON = 2025
TOP_N_DRIVERS = 10
TOP_N_CONSTRUCTORS = 10
FPS = 8
DPI = 160

TEAM_COLORS = {
    "Mercedes": "#00A19B",
    "Red Bull": "#1E3A8A",
    "Ferrari": "#DC2626",
    "McLaren": "#F97316",
    "Aston Martin": "#047857",
    "Alpine F1 Team": "#2563EB",
    "Williams": "#38BDF8",
    "Haas F1 Team": "#6B7280",
    "RB F1 Team": "#334155",
    "Audi": "#111827",
    "Cadillac F1 Team": "#B45309",
    "AlphaTauri": "#1D4ED8",
    "Toro Rosso": "#1E40AF",
    "Racing Point": "#EC4899",
    "Renault": "#FACC15",
    "Alfa Romeo": "#991B1B",
}

DEFAULT_COLORS = [
    "#2563EB",
    "#DC2626",
    "#059669",
    "#EA580C",
    "#7C3AED",
    "#0891B2",
    "#DB2777",
    "#4B5563",
    "#CA8A04",
    "#0F766E",
]


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def to_int(value, default=0):
    try:
        if value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def to_float(value, default=0.0):
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def load_modern_history_rows():
    rows = read_csv(FEATURES_PATH)
    return [
        row
        for row in rows
        if START_SEASON <= to_int(row["season"]) <= END_SEASON
    ]


def load_sqlite_constructor_rows():
    if not SQLITE_PATH.exists():
        raise FileNotFoundError(f"Database file not found: {SQLITE_PATH}")

    conn = sqlite3.connect(SQLITE_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            r.year AS season,
            r.round AS round,
            r.name AS race_name,
            r.date AS race_date,
            c.name AS constructor_name,
            rs.points AS points
        FROM results rs
        JOIN races r ON rs.raceId = r.raceId
        JOIN constructors c ON rs.constructorId = c.constructorId
        WHERE r.year BETWEEN 1950 AND 2017
        ORDER BY r.year, r.round
        """
    )
    rows = [
        {
            "season": str(row[0]),
            "round": str(row[1]),
            "race_name": row[2],
            "race_date": row[3],
            "constructor_name": row[4],
            "points": str(row[5]),
        }
        for row in cursor.fetchall()
    ]
    conn.close()
    return rows


def iter_jolpica_races(year, endpoint):
    path = JOLPICA_RAW_DIR / str(year) / f"{endpoint}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing Jolpica source file: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    for page in data.get("pages", []):
        races = page.get("MRData", {}).get("RaceTable", {}).get("Races", [])
        for race in races:
            yield race


def load_jolpica_constructor_rows(year):
    rows = []
    for race in iter_jolpica_races(year, "results"):
        for result in race.get("Results", []):
            constructor = result.get("Constructor", {})
            rows.append(
                {
                    "season": race.get("season", str(year)),
                    "round": race.get("round", ""),
                    "race_name": race.get("raceName", ""),
                    "race_date": race.get("date", ""),
                    "constructor_name": constructor.get("name", ""),
                    "points": result.get("points", "0"),
                }
            )
    return rows


def load_full_constructor_rows():
    rows = load_sqlite_constructor_rows()
    rows.extend(load_jolpica_constructor_rows(2018))

    modern_rows = load_modern_history_rows()
    for row in modern_rows:
        rows.append(
            {
                "season": row["season"],
                "round": row["round"],
                "race_name": row["race_name"],
                "race_date": row["race_date"],
                "constructor_name": row["constructor_name"],
                "points": row["points"],
            }
        )

    return [
        row
        for row in rows
        if CONSTRUCTOR_VIDEO_START_SEASON
        <= to_int(row["season"])
        <= CONSTRUCTOR_VIDEO_END_SEASON
    ]


def group_races(rows):
    grouped = defaultdict(list)
    for row in rows:
        key = (
            to_int(row["season"]),
            to_int(row["round"]),
            row["race_name"],
            row["race_date"],
        )
        grouped[key].append(row)
    return [(key, grouped[key]) for key in sorted(grouped)]


def build_points_frames(rows, entity_type, reset_each_season=True):
    """Build one standings snapshot after each race.

    entity_type is either "constructor" or "driver". Standings reset each season,
    matching Formula 1 championship scoring.
    """
    races = group_races(rows)
    frames = []
    current_season = None
    points = defaultdict(float)

    for (season, round_number, race_name, race_date), race_rows in races:
        if reset_each_season and current_season != season:
            current_season = season
            points = defaultdict(float)
        elif current_season != season:
            current_season = season

        race_points = defaultdict(float)
        for row in race_rows:
            if entity_type == "constructor":
                name = row["constructor_name"]
            else:
                name = row["driver_name"]
            race_points[name] += to_float(row["points"])

        for name, value in race_points.items():
            points[name] += value

        standings = sorted(points.items(), key=lambda item: (-item[1], item[0]))
        frames.append(
            {
                "season": season,
                "round": round_number,
                "race_name": race_name,
                "race_date": race_date,
                "standings": standings,
            }
        )

    return frames


def color_for_name(name, index):
    return TEAM_COLORS.get(name, DEFAULT_COLORS[index % len(DEFAULT_COLORS)])


def draw_frame(ax, frame, title, top_n):
    ax.clear()
    standings = frame["standings"][:top_n]
    standings = list(reversed(standings))
    labels = [name for name, _ in standings]
    values = [points for _, points in standings]
    colors = [color_for_name(name, index) for index, name in enumerate(labels)]

    ax.barh(labels, values, color=colors)
    max_value = max(values) if values else 1
    ax.set_xlim(0, max_value * 1.18)
    ax.set_title(title, fontsize=15, pad=34)
    ax.set_xlabel("Season cumulative points")
    ax.grid(True, axis="x", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    subtitle = (
        f"{frame['season']} Round {frame['round']} - "
        f"{frame['race_name']} ({frame['race_date']})"
    )
    ax.text(
        0.5,
        1.015,
        subtitle,
        transform=ax.transAxes,
        fontsize=10,
        color="#4B5563",
        ha="center",
        va="bottom",
    )

    for y_index, value in enumerate(values):
        ax.text(
            value + max_value * 0.012,
            y_index,
            f"{value:.0f}",
            va="center",
            ha="left",
            fontsize=9,
            color="#374151",
        )


def save_animation(frames, output_path, title, top_n):
    fig, ax = plt.subplots(figsize=(11, 7.4))
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(top=0.80, left=0.19, right=0.96, bottom=0.11)

    def update(frame_index):
        draw_frame(ax, frames[frame_index], title, top_n)

    animation = FuncAnimation(
        fig,
        update,
        frames=len(frames),
        interval=1000 / FPS,
        repeat=False,
    )
    writer = FFMpegWriter(fps=FPS, metadata={"artist": "Formula1_project"})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    animation.save(output_path, writer=writer, dpi=DPI)
    plt.close(fig)


def main():
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    for path in VIDEO_DIR.glob("*.mp4"):
        path.unlink()

    rows = load_modern_history_rows()
    full_constructor_rows = load_full_constructor_rows()
    constructor_frames = build_points_frames(
        full_constructor_rows, "constructor", reset_each_season=True
    )
    constructor_all_time_frames = build_points_frames(
        full_constructor_rows, "constructor", reset_each_season=False
    )
    driver_frames = build_points_frames(rows, "driver", reset_each_season=True)

    constructor_video = VIDEO_DIR / "constructor_points_race_progress_1950_2025.mp4"
    constructor_all_time_video = (
        VIDEO_DIR / "constructor_all_time_points_progress_1950_2025.mp4"
    )
    driver_video = VIDEO_DIR / "driver_points_race_progress_2019_2025.mp4"

    save_animation(
        constructor_frames,
        constructor_video,
        "Constructor Season Points Race-by-Race Progress, 1950-2025",
        TOP_N_CONSTRUCTORS,
    )
    save_animation(
        constructor_all_time_frames,
        constructor_all_time_video,
        "Constructor All-Time Points Race-by-Race Progress, 1950-2025",
        TOP_N_CONSTRUCTORS,
    )
    save_animation(
        driver_frames,
        driver_video,
        "Driver Points Race-by-Race Progress, 2019-2025",
        TOP_N_DRIVERS,
    )

    manifest = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "sources": [
            str(SQLITE_PATH.relative_to(BASE_DIR)),
            str((JOLPICA_RAW_DIR / "2018" / "results.json").relative_to(BASE_DIR)),
            str(FEATURES_PATH.relative_to(BASE_DIR)),
        ],
        "constructor_video_season_range": [
            CONSTRUCTOR_VIDEO_START_SEASON,
            CONSTRUCTOR_VIDEO_END_SEASON,
        ],
        "driver_video_season_range": [START_SEASON, END_SEASON],
        "note": "Videos exclude 2026 data. Constructor videos combine SQLite 1950-2017, Jolpica 2018, and processed 2019-2025 rows.",
        "videos": [
            {
                "filename": constructor_video.name,
                "title": "Constructor Season Points Race-by-Race Progress, 1950-2025",
                "description": "Animated bar chart of constructor season cumulative points after each race. Points reset each season.",
                "frames": len(constructor_frames),
            },
            {
                "filename": constructor_all_time_video.name,
                "title": "Constructor All-Time Points Race-by-Race Progress, 1950-2025",
                "description": "Animated bar chart of all-time constructor cumulative points after each race. Points do not reset by season.",
                "frames": len(constructor_all_time_frames),
            },
            {
                "filename": driver_video.name,
                "title": "Driver Points Race-by-Race Progress, 2019-2025",
                "description": "Animated bar chart of driver cumulative points after each race.",
                "frames": len(driver_frames),
            },
        ],
    }
    write_json(MANIFEST_PATH, manifest)

    print(f"Videos saved to: {VIDEO_DIR}")
    print(f"Created: {constructor_video.name}")
    print(f"Created: {constructor_all_time_video.name}")
    print(f"Created: {driver_video.name}")
    print(f"Manifest saved to: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
