import csv
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = BASE_DIR / "data" / "analysis"
FEATURES_PATH = BASE_DIR / "data" / "processed" / "f1_features.csv"
FIGURE_DIR = BASE_DIR / "outputs" / "figures"
MANIFEST_CSV = FIGURE_DIR / "figure_manifest.csv"
MANIFEST_JSON = FIGURE_DIR / "figure_manifest.json"

FIGURE_DPI = 170

COLORS = {
    "blue": "#2563EB",
    "green": "#059669",
    "red": "#DC2626",
    "orange": "#EA580C",
    "teal": "#0F766E",
    "gray": "#4B5563",
    "yellow": "#CA8A04",
    "pink": "#DB2777",
    "cyan": "#0891B2",
    "slate": "#334155",
}

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
}


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def clear_old_figures():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for path in FIGURE_DIR.glob("*.png"):
        path.unlink()


def to_float(value, default=0.0):
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value, default=0):
    try:
        if value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def average(values):
    clean_values = [value for value in values if value is not None]
    if not clean_values:
        return None
    return sum(clean_values) / len(clean_values)


def pct(numerator, denominator):
    if denominator == 0:
        return None
    return numerator / denominator


def pearson_correlation(pairs):
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    if len(pairs) < 2:
        return None

    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    denominator_x = sum((x - mean_x) ** 2 for x in xs) ** 0.5
    denominator_y = sum((y - mean_y) ** 2 for y in ys) ** 0.5

    if denominator_x == 0 or denominator_y == 0:
        return None
    return numerator / (denominator_x * denominator_y)


def load_modern_history_rows():
    return [
        row
        for row in read_csv(FEATURES_PATH)
        if 2019 <= to_int(row["season"]) <= 2025
    ]


def build_grid_position_summary(rows):
    summary_rows = []
    grids = sorted({to_int(row["grid"]) for row in rows if to_int(row["grid"]) > 0})
    for grid in grids:
        group = [row for row in rows if to_int(row["grid"]) == grid]
        summary_rows.append(
            {
                "grid": grid,
                "records": len(group),
                "avg_finish_position": average([to_float(row["finish_position"]) for row in group]),
                "win_rate": pct(sum(1 for row in group if to_int(row["finish_position"]) == 1), len(group)),
                "podium_rate": pct(sum(to_int(row["is_podium"]) for row in group), len(group)),
                "top10_rate": pct(sum(to_int(row["is_top10"]) for row in group), len(group)),
            }
        )
    return summary_rows


def build_qualifying_position_summary(rows):
    summary_rows = []
    positions = sorted(
        {
            to_int(row["qualifying_position"])
            for row in rows
            if row["qualifying_position"] != ""
        }
    )
    for position in positions:
        group = [row for row in rows if to_int(row["qualifying_position"]) == position]
        summary_rows.append(
            {
                "qualifying_position": position,
                "records": len(group),
                "avg_finish_position": average([to_float(row["finish_position"]) for row in group]),
                "win_rate": pct(sum(1 for row in group if to_int(row["finish_position"]) == 1), len(group)),
                "podium_rate": pct(sum(to_int(row["is_podium"]) for row in group), len(group)),
                "top10_rate": pct(sum(to_int(row["is_top10"]) for row in group), len(group)),
            }
        )
    return summary_rows


def build_grid_finish_metrics(rows):
    valid_grid_rows = [row for row in rows if to_int(row["grid"]) > 0]
    valid_quali_rows = [row for row in rows if row["qualifying_position"] != ""]
    pole_rows = [row for row in valid_grid_rows if to_int(row["grid"]) == 1]
    front3_rows = [row for row in valid_grid_rows if 1 <= to_int(row["grid"]) <= 3]
    front10_rows = [row for row in valid_grid_rows if 1 <= to_int(row["grid"]) <= 10]
    return {
        "grid_finish_correlation": pearson_correlation(
            [(to_float(row["grid"]), to_float(row["finish_position"])) for row in valid_grid_rows]
        ),
        "qualifying_finish_correlation": pearson_correlation(
            [
                (to_float(row["qualifying_position"]), to_float(row["finish_position"]))
                for row in valid_quali_rows
            ]
        ),
        "pole_win_rate": pct(
            sum(1 for row in pole_rows if to_int(row["finish_position"]) == 1),
            len(pole_rows),
        ),
        "front3_podium_rate": pct(
            sum(to_int(row["is_podium"]) for row in front3_rows),
            len(front3_rows),
        ),
        "front10_top10_rate": pct(
            sum(to_int(row["is_top10"]) for row in front10_rows),
            len(front10_rows),
        ),
    }


def build_driver_summary_rows(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["driver_id"], []).append(row)

    summary_rows = []
    for driver_id, group in grouped.items():
        summary_rows.append(
            {
                "driver_id": driver_id,
                "driver_name": group[-1]["driver_name"],
                "total_points": sum(to_float(row["points"]) for row in group),
            }
        )
    return sorted(summary_rows, key=lambda row: row["total_points"], reverse=True)


def build_constructor_summary_rows(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["constructor_id"], []).append(row)

    summary_rows = []
    for constructor_id, group in grouped.items():
        summary_rows.append(
            {
                "constructor_id": constructor_id,
                "constructor_name": group[-1]["constructor_name"],
                "total_points": sum(to_float(row["points"]) for row in group),
            }
        )
    return sorted(summary_rows, key=lambda row: row["total_points"], reverse=True)


def build_circuit_summary_rows(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["circuit_id"], []).append(row)

    summary_rows = []
    for circuit_id, group in grouped.items():
        races = {(row["season"], row["round"]) for row in group}
        pole_rows = [row for row in group if to_int(row["grid"]) == 1]
        if len(races) < 3 or not pole_rows:
            continue
        summary_rows.append(
            {
                "circuit_id": circuit_id,
                "circuit_name": group[-1]["circuit_name"],
                "race_count": len(races),
                "pole_win_rate": pct(
                    sum(1 for row in pole_rows if to_int(row["finish_position"]) == 1),
                    len(pole_rows),
                ),
            }
        )
    return sorted(summary_rows, key=lambda row: row["pole_win_rate"], reverse=True)


def build_constructor_podium_rows(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["constructor_id"], []).append(row)

    summary_rows = []
    for constructor_id, group in grouped.items():
        summary_rows.append(
            {
                "constructor_id": constructor_id,
                "constructor_name": group[-1]["constructor_name"],
                "podiums": sum(to_int(row["is_podium"]) for row in group),
                "wins": sum(1 for row in group if to_int(row["finish_position"]) == 1),
                "top10s": sum(to_int(row["is_top10"]) for row in group),
            }
        )
    return sorted(summary_rows, key=lambda row: row["podiums"], reverse=True)


def build_driver_podium_rows(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["driver_id"], []).append(row)

    summary_rows = []
    for driver_id, group in grouped.items():
        summary_rows.append(
            {
                "driver_id": driver_id,
                "driver_name": group[-1]["driver_name"],
                "podiums": sum(to_int(row["is_podium"]) for row in group),
                "wins": sum(1 for row in group if to_int(row["finish_position"]) == 1),
                "top10s": sum(to_int(row["is_top10"]) for row in group),
            }
        )
    return sorted(summary_rows, key=lambda row: row["podiums"], reverse=True)


def build_status_by_year(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["season"], []).append(row)

    summary_rows = []
    for season, group in sorted(grouped.items()):
        classified_rows = [row for row in group if row["status"] != ""]
        finished = sum(1 for row in classified_rows if row["status"] == "Finished")
        lapped = sum(1 for row in classified_rows if row["status"].startswith("+"))
        other = len(classified_rows) - finished - lapped
        summary_rows.append(
            {
                "season": season,
                "finished_rate": pct(finished, len(classified_rows)),
                "lapped_rate": pct(lapped, len(classified_rows)),
                "other_status_rate": pct(other, len(classified_rows)),
            }
        )
    return summary_rows


def build_circuit_position_change_rows(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["circuit_id"], []).append(row)

    summary_rows = []
    for circuit_id, group in grouped.items():
        races = {(row["season"], row["round"]) for row in group}
        changes = [
            to_int(row["grid"]) - to_int(row["finish_position"])
            for row in group
            if to_int(row["grid"]) > 0
        ]
        if len(races) < 3 or not changes:
            continue
        summary_rows.append(
            {
                "circuit_id": circuit_id,
                "circuit_name": group[-1]["circuit_name"],
                "race_count": len(races),
                "avg_position_change": average(changes),
            }
        )
    return sorted(summary_rows, key=lambda row: row["avg_position_change"], reverse=True)


def build_circuit_grid_importance_rows(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["circuit_id"], []).append(row)

    summary_rows = []
    for circuit_id, group in grouped.items():
        races = {(row["season"], row["round"]) for row in group}
        valid_grid_rows = [row for row in group if to_int(row["grid"]) > 0]
        pole_rows = [row for row in valid_grid_rows if to_int(row["grid"]) == 1]
        front3_rows = [row for row in valid_grid_rows if 1 <= to_int(row["grid"]) <= 3]
        changes = [
            to_int(row["grid"]) - to_int(row["finish_position"])
            for row in valid_grid_rows
        ]
        pairs = [
            (to_float(row["grid"]), to_float(row["finish_position"]))
            for row in valid_grid_rows
        ]
        if len(races) < 3 or not pole_rows or len(pairs) < 10:
            continue
        summary_rows.append(
            {
                "circuit_id": circuit_id,
                "circuit_name": group[-1]["circuit_name"],
                "country": group[-1]["circuit_country"],
                "race_count": len(races),
                "pole_win_rate": pct(
                    sum(1 for row in pole_rows if to_int(row["finish_position"]) == 1),
                    len(pole_rows),
                ),
                "front3_podium_rate": pct(
                    sum(to_int(row["is_podium"]) for row in front3_rows),
                    len(front3_rows),
                ),
                "avg_abs_position_change": average([abs(value) for value in changes]),
                "avg_position_change": average(changes),
                "grid_finish_correlation": pearson_correlation(pairs),
            }
        )
    return summary_rows


def build_driver_consistency_rows(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["driver_id"], []).append(row)

    summary_rows = []
    for driver_id, group in grouped.items():
        if len(group) < 30:
            continue
        finishes = [to_float(row["finish_position"]) for row in group]
        avg_finish = average(finishes)
        variance = average([(finish - avg_finish) ** 2 for finish in finishes])
        summary_rows.append(
            {
                "driver_id": driver_id,
                "driver_name": group[-1]["driver_name"],
                "starts": len(group),
                "avg_finish_position": avg_finish,
                "finish_stddev": variance ** 0.5,
                "avg_points": average([to_float(row["points"]) for row in group]),
                "podium_rate": pct(sum(to_int(row["is_podium"]) for row in group), len(group)),
                "top10_rate": pct(sum(to_int(row["is_top10"]) for row in group), len(group)),
            }
        )
    return summary_rows


def build_constructor_efficiency_rows(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row["constructor_id"], []).append(row)

    summary_rows = []
    for constructor_id, group in grouped.items():
        if len(group) < 30:
            continue
        summary_rows.append(
            {
                "constructor_id": constructor_id,
                "constructor_name": group[-1]["constructor_name"],
                "records": len(group),
                "avg_grid": average([to_float(row["grid"]) for row in group if to_int(row["grid"]) > 0]),
                "avg_finish_position": average([to_float(row["finish_position"]) for row in group]),
                "avg_points": average([to_float(row["points"]) for row in group]),
                "podium_rate": pct(sum(to_int(row["is_podium"]) for row in group), len(group)),
            }
        )
    return summary_rows


def build_feature_correlation_matrix(rows):
    features = [
        "grid",
        "qualifying_position",
        "driver_pre_race_points",
        "driver_pre_race_rank",
        "constructor_pre_race_points",
        "constructor_pre_race_rank",
        "driver_last3_avg_points",
        "driver_last3_avg_finish_position",
        "driver_last3_podium_count",
        "constructor_last3_avg_points",
        "constructor_last3_podium_count",
    ]
    targets = ["finish_position", "is_podium", "is_top10"]
    matrix = []
    for feature in features:
        row_values = []
        for target in targets:
            row_values.append(
                pearson_correlation(
                    [
                        (to_float(row[feature]), to_float(row[target]))
                        for row in rows
                        if row[feature] != "" and row[target] != ""
                    ]
                )
            )
        matrix.append(row_values)
    return features, targets, matrix


def build_position_gain_summary(rows, group_key, name_key, min_records=20):
    grouped = {}
    for row in rows:
        if to_int(row["grid"]) <= 0:
            continue
        grouped.setdefault(row[group_key], []).append(row)

    summary_rows = []
    for item_id, group in grouped.items():
        if len(group) < min_records:
            continue
        changes = [
            to_int(row["grid"]) - to_int(row["finish_position"])
            for row in group
            if to_int(row["grid"]) > 0
        ]
        summary_rows.append(
            {
                group_key: item_id,
                name_key: group[-1][name_key],
                "records": len(changes),
                "avg_position_change": average(changes),
                "positive_change_rate": pct(
                    sum(1 for value in changes if value > 0), len(changes)
                ),
            }
        )
    return sorted(summary_rows, key=lambda row: row["avg_position_change"], reverse=True)


def setup_figure(width=10, height=6):
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    fig, ax = plt.subplots(figsize=(width, height))
    ax.grid(True, axis="y", color="#E5E7EB", linewidth=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def save_current_figure(filename):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / filename
    plt.tight_layout()
    plt.savefig(path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    return path


def add_bar_labels(ax, bars, value_format="{:.0f}", padding=3):
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + padding,
            bar.get_y() + bar.get_height() / 2,
            value_format.format(width),
            va="center",
            ha="left",
            fontsize=8,
            color=COLORS["gray"],
        )


def add_vertical_bar_labels(ax, bars, value_format="{:.0f}", padding=2):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + padding,
            value_format.format(height),
            va="bottom",
            ha="center",
            fontsize=8,
            color=COLORS["gray"],
        )


def add_manifest(manifest, filename, title, source, description):
    manifest.append(
        {
            "filename": filename,
            "title": title,
            "source": source,
            "description": description,
        }
    )


def plot_yearly_records(manifest):
    rows = [
        row
        for row in read_csv(ANALYSIS_DIR / "dataset_overview_by_year.csv")
        if 2019 <= to_int(row["season"]) <= 2025
    ]
    years = [row["season"] for row in rows]
    records = [to_int(row["records"]) for row in rows]
    races = [to_int(row["races"]) for row in rows]

    fig, ax = setup_figure(10, 5.5)
    bars = ax.bar(years, records, color=COLORS["blue"], label="Driver-race records")
    ax2 = ax.twinx()
    ax2.plot(years, races, color=COLORS["orange"], marker="o", linewidth=2.2, label="Races")
    ax.set_title("Modern Historical Dataset Size by Season, 2019-2025")
    ax.set_xlabel("Season")
    ax.set_ylabel("Driver-race records")
    ax2.set_ylabel("Race count")
    ax2.grid(False)
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")
    add_vertical_bar_labels(ax, bars, "{:.0f}", padding=max(records) * 0.01)

    filename = "yearly_records_2019_2025.png"
    save_current_figure(filename)
    add_manifest(
        manifest,
        filename,
        "Modern Dataset Size by Season",
        "dataset_overview_by_year.csv",
        "Shows yearly sample size and race count for the 2019-2025 modern historical analysis dataset.",
    )


def plot_grid_rates(manifest, modern_rows):
    rows = build_grid_position_summary(modern_rows)
    rows = [row for row in rows if 1 <= to_int(row["grid"]) <= 22]
    grids = [to_int(row["grid"]) for row in rows]
    win_rate = [to_float(row["win_rate"]) * 100 for row in rows]
    podium_rate = [to_float(row["podium_rate"]) * 100 for row in rows]
    top10_rate = [to_float(row["top10_rate"]) * 100 for row in rows]

    fig, ax = setup_figure(11, 6)
    ax.plot(grids, win_rate, marker="o", linewidth=2.1, color=COLORS["red"], label="Win rate")
    ax.plot(grids, podium_rate, marker="o", linewidth=2.1, color=COLORS["blue"], label="Podium rate")
    ax.plot(grids, top10_rate, marker="o", linewidth=2.1, color=COLORS["green"], label="Top 10 rate")
    ax.set_title("Race Outcome Rates by Grid Position, 2019-2025")
    ax.set_xlabel("Grid position")
    ax.set_ylabel("Rate (%)")
    ax.set_xticks(grids)
    ax.set_ylim(0, 105)
    ax.legend()

    filename = "grid_position_outcome_rates.png"
    save_current_figure(filename)
    add_manifest(
        manifest,
        filename,
        "Race Outcome Rates by Grid Position",
        "f1_features.csv filtered to 2019-2025",
        "Compares win, podium, and top-10 rates for each starting grid position without 2026 data.",
    )


def plot_qualifying_rates(manifest, modern_rows):
    rows = build_qualifying_position_summary(modern_rows)
    rows = [row for row in rows if 1 <= to_int(row["qualifying_position"]) <= 22]
    positions = [to_int(row["qualifying_position"]) for row in rows]
    podium_rate = [to_float(row["podium_rate"]) * 100 for row in rows]
    top10_rate = [to_float(row["top10_rate"]) * 100 for row in rows]
    avg_finish = [to_float(row["avg_finish_position"]) for row in rows]

    fig, ax = setup_figure(11, 6)
    ax.plot(positions, podium_rate, marker="o", linewidth=2.1, color=COLORS["blue"], label="Podium rate")
    ax.plot(positions, top10_rate, marker="o", linewidth=2.1, color=COLORS["green"], label="Top 10 rate")
    ax2 = ax.twinx()
    ax2.plot(positions, avg_finish, marker="s", linewidth=1.8, color=COLORS["orange"], label="Avg finish")
    ax.set_title("Qualifying Position and Race Outcome, 2019-2025")
    ax.set_xlabel("Qualifying position")
    ax.set_ylabel("Rate (%)")
    ax2.set_ylabel("Average finish position")
    ax.set_xticks(positions)
    ax.set_ylim(0, 105)
    ax2.invert_yaxis()
    ax2.grid(False)
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    filename = "qualifying_position_outcome_rates.png"
    save_current_figure(filename)
    add_manifest(
        manifest,
        filename,
        "Qualifying Position and Race Outcome",
        "f1_features.csv filtered to 2019-2025",
        "Shows how qualifying position relates to podium rate, top-10 rate, and average finish without 2026 data.",
    )


def plot_modern_vs_historical_grid(manifest, modern_rows):
    modern = build_grid_finish_metrics(modern_rows)
    historical = {
        row["metric"].replace("_1950_2017", ""): to_float(row["value"])
        for row in read_csv(ANALYSIS_DIR / "grid_finish_summary_historical.csv")
    }

    metrics = [
        ("pole_win_rate", "Pole win"),
        ("front3_podium_rate", "Front 3 podium"),
        ("front10_top10_rate", "Front 10 top 10"),
    ]
    labels = [label for _, label in metrics]
    modern_values = [modern[key] * 100 for key, _ in metrics]
    historical_values = [historical[key] * 100 for key, _ in metrics]
    x = list(range(len(metrics)))
    width = 0.36

    fig, ax = setup_figure(9, 5.5)
    ax.bar([i - width / 2 for i in x], historical_values, width=width, color=COLORS["gray"], label="1950-2017")
    ax.bar([i + width / 2 for i in x], modern_values, width=width, color=COLORS["blue"], label="2019-2025")
    ax.set_title("Grid Advantage: Historical vs Modern Era")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 100)
    ax.legend()

    filename = "grid_advantage_historical_vs_modern.png"
    save_current_figure(filename)
    add_manifest(
        manifest,
        filename,
        "Grid Advantage: Historical vs Modern Era",
        "f1_features.csv filtered to 2019-2025; grid_finish_summary_historical.csv",
        "Compares grid-position outcome rates between long-run history and the 2019-2025 modern historical dataset.",
    )


def plot_top_driver_points(manifest, modern_rows):
    rows = build_driver_summary_rows(modern_rows)[:10]
    rows = list(reversed(rows))
    labels = [row["driver_name"] for row in rows]
    points = [to_float(row["total_points"]) for row in rows]

    fig, ax = setup_figure(10, 6)
    bars = ax.barh(labels, points, color=COLORS["blue"])
    ax.set_title("Top 10 Drivers by Total Points, 2019-2025")
    ax.set_xlabel("Total points")
    add_bar_labels(ax, bars, "{:.0f}", padding=max(points) * 0.01)

    filename = "top10_driver_points_2019_2025.png"
    save_current_figure(filename)
    add_manifest(
        manifest,
        filename,
        "Top 10 Drivers by Total Points",
        "f1_features.csv filtered to 2019-2025",
        "Ranks the top drivers in the 2019-2025 modern historical dataset by total points.",
    )


def plot_constructor_points(manifest, modern_rows):
    rows = build_constructor_summary_rows(modern_rows)[:10]
    rows = list(reversed(rows))
    labels = [row["constructor_name"] for row in rows]
    points = [to_float(row["total_points"]) for row in rows]
    colors = [TEAM_COLORS.get(label, COLORS["blue"]) for label in labels]

    fig, ax = setup_figure(10, 6)
    bars = ax.barh(labels, points, color=colors)
    ax.set_title("Top Constructors by Total Points, 2019-2025")
    ax.set_xlabel("Total points")
    add_bar_labels(ax, bars, "{:.0f}", padding=max(points) * 0.01)

    filename = "constructor_points_2019_2025.png"
    save_current_figure(filename)
    add_manifest(
        manifest,
        filename,
        "Top Constructors by Total Points",
        "f1_features.csv filtered to 2019-2025",
        "Ranks constructors by total points in the 2019-2025 modern historical dataset.",
    )


def select_top_entities_by_total(rows, entity_key, value_key, top_n):
    totals = {}
    names = {}
    for row in rows:
        entity_id = row[entity_key]
        totals[entity_id] = totals.get(entity_id, 0.0) + to_float(row[value_key])
        names[entity_id] = row.get("driver_name") or row.get("constructor_name") or entity_id
    selected = sorted(totals, key=totals.get, reverse=True)[:top_n]
    return selected, names


def plot_driver_points_trend(manifest):
    rows = [
        row
        for row in read_csv(ANALYSIS_DIR / "driver_points_by_year.csv")
        if 2019 <= to_int(row["season"]) <= 2025
    ]
    selected_ids, names = select_top_entities_by_total(rows, "driver_id", "total_points", 6)
    years = sorted({row["season"] for row in rows})

    fig, ax = setup_figure(11, 6)
    palette = [COLORS["blue"], COLORS["red"], COLORS["green"], COLORS["orange"], COLORS["teal"], COLORS["pink"]]
    for index, driver_id in enumerate(selected_ids):
        by_year = {row["season"]: to_float(row["total_points"]) for row in rows if row["driver_id"] == driver_id}
        values = [by_year.get(year, 0.0) for year in years]
        ax.plot(years, values, marker="o", linewidth=2.2, color=palette[index], label=names[driver_id])

    ax.set_title("Driver Points Trend, Top 6 Modern Drivers")
    ax.set_xlabel("Season")
    ax.set_ylabel("Points")
    ax.legend(ncol=2)

    filename = "driver_points_trend_top6.png"
    save_current_figure(filename)
    add_manifest(
        manifest,
        filename,
        "Driver Points Trend, Top 6 Modern Drivers",
        "driver_points_by_year.csv",
        "Shows season-by-season point trends for the highest-scoring drivers.",
    )


def plot_constructor_points_trend(manifest):
    rows = [
        row
        for row in read_csv(ANALYSIS_DIR / "constructor_points_by_year.csv")
        if 2019 <= to_int(row["season"]) <= 2025
    ]
    selected_ids, names = select_top_entities_by_total(rows, "constructor_id", "total_points", 6)
    years = sorted({row["season"] for row in rows})

    fig, ax = setup_figure(11, 6)
    for constructor_id in selected_ids:
        name = names[constructor_id]
        by_year = {row["season"]: to_float(row["total_points"]) for row in rows if row["constructor_id"] == constructor_id}
        values = [by_year.get(year, 0.0) for year in years]
        ax.plot(years, values, marker="o", linewidth=2.3, color=TEAM_COLORS.get(name, COLORS["blue"]), label=name)

    ax.set_title("Constructor Points Trend, Top 6 Modern Teams")
    ax.set_xlabel("Season")
    ax.set_ylabel("Points")
    ax.legend(ncol=2)

    filename = "constructor_points_trend_top6.png"
    save_current_figure(filename)
    add_manifest(
        manifest,
        filename,
        "Constructor Points Trend, Top 6 Modern Teams",
        "constructor_points_by_year.csv",
        "Shows season-by-season point trends for leading constructors.",
    )


def plot_historical_races_by_year(manifest):
    rows = read_csv(ANALYSIS_DIR / "races_by_year_historical.csv")
    years = [to_int(row["year"]) for row in rows]
    counts = [to_int(row["race_count"]) for row in rows]

    fig, ax = setup_figure(12, 5.8)
    ax.plot(years, counts, color=COLORS["blue"], linewidth=2.1)
    ax.fill_between(years, counts, color=COLORS["blue"], alpha=0.12)
    ax.set_title("Formula 1 Race Count by Year, 1950-2018")
    ax.set_xlabel("Year")
    ax.set_ylabel("Race count")
    ax.set_xlim(min(years), max(years))

    filename = "historical_races_by_year.png"
    save_current_figure(filename)
    add_manifest(
        manifest,
        filename,
        "Formula 1 Race Count by Year",
        "races_by_year_historical.csv",
        "Shows the expansion of the F1 calendar from 1950 to 2018.",
    )


def plot_historical_driver_wins(manifest):
    rows = read_csv(ANALYSIS_DIR / "driver_wins_historical.csv")[:10]
    rows = list(reversed(rows))
    labels = [row["driver_name"] for row in rows]
    wins = [to_float(row["wins"]) for row in rows]

    fig, ax = setup_figure(10, 6)
    bars = ax.barh(labels, wins, color=COLORS["red"])
    ax.set_title("Historical Top 10 Driver Wins, 1950-2017")
    ax.set_xlabel("Wins")
    add_bar_labels(ax, bars, "{:.0f}", padding=max(wins) * 0.01)

    filename = "historical_top10_driver_wins.png"
    save_current_figure(filename)
    add_manifest(
        manifest,
        filename,
        "Historical Top 10 Driver Wins",
        "driver_wins_historical.csv",
        "Ranks the top race-winning drivers in the historical SQLite data.",
    )


def plot_historical_constructor_wins(manifest):
    rows = read_csv(ANALYSIS_DIR / "constructor_wins_historical.csv")[:10]
    rows = list(reversed(rows))
    labels = [row["constructor_name"] for row in rows]
    wins = [to_float(row["wins"]) for row in rows]

    fig, ax = setup_figure(10, 6)
    bars = ax.barh(labels, wins, color=COLORS["green"])
    ax.set_title("Historical Top 10 Constructor Wins, 1950-2017")
    ax.set_xlabel("Wins")
    add_bar_labels(ax, bars, "{:.0f}", padding=max(wins) * 0.01)

    filename = "historical_top10_constructor_wins.png"
    save_current_figure(filename)
    add_manifest(
        manifest,
        filename,
        "Historical Top 10 Constructor Wins",
        "constructor_wins_historical.csv",
        "Ranks the top race-winning constructors in the historical SQLite data.",
    )


def plot_circuit_pole_win_rate(manifest, modern_rows):
    rows = build_circuit_summary_rows(modern_rows)
    rows = [row for row in rows if to_int(row["race_count"]) >= 3 and row["pole_win_rate"] is not None]
    rows = sorted(rows, key=lambda row: to_float(row["pole_win_rate"]), reverse=True)[:12]
    rows = list(reversed(rows))
    labels = [row["circuit_name"] for row in rows]
    rates = [to_float(row["pole_win_rate"]) * 100 for row in rows]

    fig, ax = setup_figure(11, 7)
    bars = ax.barh(labels, rates, color=COLORS["teal"])
    ax.set_title("Circuits with Highest Pole Win Rate, 2019-2025")
    ax.set_xlabel("Pole win rate (%)")
    ax.set_xlim(0, 105)
    add_bar_labels(ax, bars, "{:.1f}", padding=1.5)

    filename = "circuit_pole_win_rate_top12.png"
    save_current_figure(filename)
    add_manifest(
        manifest,
        filename,
        "Circuits with Highest Pole Win Rate",
        "f1_features.csv filtered to 2019-2025",
        "Highlights circuits where pole position most often converted to victory without 2026 data.",
    )


def plot_driver_points_heatmap(manifest):
    rows = [
        row
        for row in read_csv(ANALYSIS_DIR / "driver_points_by_year.csv")
        if 2019 <= to_int(row["season"]) <= 2025
    ]
    selected_ids, names = select_top_entities_by_total(rows, "driver_id", "total_points", 10)
    years = sorted({row["season"] for row in rows})
    matrix = []
    for driver_id in selected_ids:
        by_year = {row["season"]: to_float(row["total_points"]) for row in rows if row["driver_id"] == driver_id}
        matrix.append([by_year.get(year, 0.0) for year in years])

    fig, ax = setup_figure(11, 6.5)
    image = ax.imshow(matrix, aspect="auto", cmap="YlGnBu")
    ax.set_title("Driver Points Heatmap, Top 10 Drivers, 2019-2025")
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years)
    ax.set_yticks(range(len(selected_ids)))
    ax.set_yticklabels([names[driver_id] for driver_id in selected_ids])
    ax.grid(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.03)
    colorbar.set_label("Points")

    filename = "driver_points_heatmap_top10.png"
    save_current_figure(filename)
    add_manifest(
        manifest,
        filename,
        "Driver Points Heatmap, Top 10 Drivers",
        "driver_points_by_year.csv filtered to 2019-2025",
        "Shows year-by-year points for the top modern drivers as a compact heatmap.",
    )


def plot_constructor_points_share(manifest):
    rows = [
        row
        for row in read_csv(ANALYSIS_DIR / "constructor_points_by_year.csv")
        if 2019 <= to_int(row["season"]) <= 2025
    ]
    selected_ids, names = select_top_entities_by_total(rows, "constructor_id", "total_points", 6)
    years = sorted({row["season"] for row in rows})
    series = []
    labels = []
    colors = []
    for constructor_id in selected_ids:
        name = names[constructor_id]
        by_year = {row["season"]: to_float(row["total_points"]) for row in rows if row["constructor_id"] == constructor_id}
        values = [by_year.get(year, 0.0) for year in years]
        series.append(values)
        labels.append(name)
        colors.append(TEAM_COLORS.get(name, COLORS["blue"]))

    fig, ax = setup_figure(11, 6)
    ax.stackplot(years, series, labels=labels, colors=colors, alpha=0.85)
    ax.set_title("Constructor Points Share, Top 6 Teams, 2019-2025")
    ax.set_xlabel("Season")
    ax.set_ylabel("Points")
    ax.legend(loc="upper left", ncol=2)

    filename = "constructor_points_share_top6.png"
    save_current_figure(filename)
    add_manifest(
        manifest,
        filename,
        "Constructor Points Share, Top 6 Teams",
        "constructor_points_by_year.csv filtered to 2019-2025",
        "Shows how leading teams' point totals changed relative to each other over time.",
    )


def plot_constructor_podiums(manifest, modern_rows):
    rows = build_constructor_podium_rows(modern_rows)[:10]
    rows = list(reversed(rows))
    labels = [row["constructor_name"] for row in rows]
    podiums = [row["podiums"] for row in rows]
    wins = [row["wins"] for row in rows]
    colors = [TEAM_COLORS.get(label, COLORS["blue"]) for label in labels]

    fig, ax = setup_figure(10, 6)
    bars = ax.barh(labels, podiums, color=colors, label="Podiums")
    ax.barh(labels, wins, color="#111827", alpha=0.35, label="Wins")
    ax.set_title("Constructor Podiums and Wins, 2019-2025")
    ax.set_xlabel("Count")
    ax.legend()
    add_bar_labels(ax, bars, "{:.0f}", padding=max(podiums) * 0.01)

    filename = "constructor_podiums_wins_2019_2025.png"
    save_current_figure(filename)
    add_manifest(
        manifest,
        filename,
        "Constructor Podiums and Wins",
        "f1_features.csv filtered to 2019-2025",
        "Compares podium counts and wins by constructor in the modern historical period.",
    )


def plot_driver_podiums(manifest, modern_rows):
    rows = build_driver_podium_rows(modern_rows)[:12]
    rows = list(reversed(rows))
    labels = [row["driver_name"] for row in rows]
    podiums = [row["podiums"] for row in rows]
    wins = [row["wins"] for row in rows]

    fig, ax = setup_figure(10, 6.5)
    bars = ax.barh(labels, podiums, color=COLORS["blue"], label="Podiums")
    ax.barh(labels, wins, color=COLORS["red"], alpha=0.55, label="Wins")
    ax.set_title("Driver Podiums and Wins, 2019-2025")
    ax.set_xlabel("Count")
    ax.legend()
    add_bar_labels(ax, bars, "{:.0f}", padding=max(podiums) * 0.01)

    filename = "driver_podiums_wins_2019_2025.png"
    save_current_figure(filename)
    add_manifest(
        manifest,
        filename,
        "Driver Podiums and Wins",
        "f1_features.csv filtered to 2019-2025",
        "Compares podium counts and wins by driver in the modern historical period.",
    )


def plot_status_by_year(manifest, modern_rows):
    rows = build_status_by_year(modern_rows)
    years = [row["season"] for row in rows]
    finished = [row["finished_rate"] * 100 for row in rows]
    lapped = [row["lapped_rate"] * 100 for row in rows]
    other = [row["other_status_rate"] * 100 for row in rows]

    fig, ax = setup_figure(10, 5.8)
    ax.stackplot(
        years,
        finished,
        lapped,
        other,
        labels=["Finished", "Classified laps down", "Other status"],
        colors=[COLORS["green"], COLORS["yellow"], COLORS["red"]],
        alpha=0.85,
    )
    ax.set_title("Race Result Status Mix by Season, 2019-2025")
    ax.set_xlabel("Season")
    ax.set_ylabel("Share of classified records (%)")
    ax.set_ylim(0, 100)
    ax.legend(loc="lower left")

    filename = "status_mix_by_year_2019_2025.png"
    save_current_figure(filename)
    add_manifest(
        manifest,
        filename,
        "Race Result Status Mix by Season",
        "f1_features.csv filtered to 2019-2025",
        "Shows the share of finished, lapped, and other race statuses by season.",
    )


def plot_circuit_position_change(manifest, modern_rows):
    rows = build_circuit_position_change_rows(modern_rows)
    selected = rows[:6] + rows[-6:]
    selected = sorted(selected, key=lambda row: row["avg_position_change"])
    labels = [row["circuit_name"] for row in selected]
    values = [row["avg_position_change"] for row in selected]
    colors = [COLORS["red"] if value < 0 else COLORS["green"] for value in values]

    fig, ax = setup_figure(11, 7)
    bars = ax.barh(labels, values, color=colors)
    ax.axvline(0, color="#111827", linewidth=1)
    ax.set_title("Average Position Change by Circuit, 2019-2025")
    ax.set_xlabel("Average grid position minus finish position")
    ax.text(
        0.01,
        0.02,
        "Positive values indicate average gains from start to finish.",
        transform=ax.transAxes,
        fontsize=8,
        color=COLORS["gray"],
    )
    for bar, value in zip(bars, values):
        offset = 0.08 if value >= 0 else -0.08
        ha = "left" if value >= 0 else "right"
        ax.text(
            value + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            ha=ha,
            fontsize=8,
            color=COLORS["gray"],
        )

    filename = "circuit_position_change_2019_2025.png"
    save_current_figure(filename)
    add_manifest(
        manifest,
        filename,
        "Average Position Change by Circuit",
        "f1_features.csv filtered to 2019-2025",
        "Contrasts circuits where drivers tend to gain or lose positions from start to finish.",
    )


def plot_qualifying_finish_heatmap(manifest, modern_rows):
    max_position = 20
    matrix = [[0 for _ in range(max_position)] for _ in range(max_position)]
    for row in modern_rows:
        qualifying = to_int(row["qualifying_position"], None)
        finish = to_int(row["finish_position"], None)
        if qualifying is None or finish is None:
            continue
        if 1 <= qualifying <= max_position and 1 <= finish <= max_position:
            matrix[finish - 1][qualifying - 1] += 1

    fig, ax = setup_figure(9, 7.5)
    image = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_title("Qualifying Position vs Finish Position Heatmap, 2019-2025")
    ax.set_xlabel("Qualifying position")
    ax.set_ylabel("Finish position")
    ax.set_xticks(range(max_position))
    ax.set_xticklabels(range(1, max_position + 1))
    ax.set_yticks(range(max_position))
    ax.set_yticklabels(range(1, max_position + 1))
    ax.grid(False)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    colorbar.set_label("Record count")

    filename = "qualifying_finish_heatmap_2019_2025.png"
    save_current_figure(filename)
    add_manifest(
        manifest,
        filename,
        "Qualifying Position vs Finish Position Heatmap",
        "f1_features.csv filtered to 2019-2025",
        "Shows the joint distribution of qualifying position and final race position.",
    )


def plot_constructor_rank_bump_chart(manifest):
    rows = [
        row
        for row in read_csv(ANALYSIS_DIR / "constructor_points_by_year.csv")
        if 2019 <= to_int(row["season"]) <= 2025
    ]
    totals = {}
    names = {}
    for row in rows:
        constructor_id = row["constructor_id"]
        totals[constructor_id] = totals.get(constructor_id, 0.0) + to_float(row["total_points"])
        names[constructor_id] = row["constructor_name"]
    selected_ids = sorted(totals, key=totals.get, reverse=True)[:8]
    years = sorted({row["season"] for row in rows})

    ranks_by_year = {}
    for year in years:
        year_rows = [row for row in rows if row["season"] == year]
        year_rows = sorted(year_rows, key=lambda row: to_float(row["total_points"]), reverse=True)
        ranks_by_year[year] = {
            row["constructor_id"]: index
            for index, row in enumerate(year_rows, start=1)
        }

    fig, ax = setup_figure(11, 6.5)
    ax.grid(True, axis="x", color="#E5E7EB", linewidth=0.8)
    for constructor_id in selected_ids:
        name = names[constructor_id]
        ranks = [ranks_by_year[year].get(constructor_id) for year in years]
        valid_points = [(year, rank) for year, rank in zip(years, ranks) if rank is not None]
        if not valid_points:
            continue
        ax.plot(
            years,
            ranks,
            marker="o",
            linewidth=2.4,
            color=TEAM_COLORS.get(name, COLORS["blue"]),
            label=name,
        )
        last_year, last_rank = valid_points[-1]
        ax.text(last_year, last_rank + 0.05, name, fontsize=8, va="center")

    ax.set_title("Constructor Championship Rank Bump Chart, 2019-2025")
    ax.set_xlabel("Season")
    ax.set_ylabel("Annual constructor rank")
    ax.set_ylim(8.5, 0.5)
    ax.set_yticks(range(1, 9))

    filename = "constructor_rank_bump_chart_2019_2025.png"
    save_current_figure(filename)
    add_manifest(
        manifest,
        filename,
        "Constructor Championship Rank Bump Chart",
        "constructor_points_by_year.csv filtered to 2019-2025",
        "Shows how leading constructor ranking changed across seasons.",
    )


def plot_circuit_grid_importance_bubble(manifest, modern_rows):
    rows = build_circuit_grid_importance_rows(modern_rows)
    x_values = [row["pole_win_rate"] * 100 for row in rows]
    y_values = [row["avg_abs_position_change"] for row in rows]
    sizes = [80 + row["race_count"] * 45 for row in rows]
    colors = [row["grid_finish_correlation"] for row in rows]

    fig, ax = setup_figure(11, 7)
    scatter = ax.scatter(
        x_values,
        y_values,
        s=sizes,
        c=colors,
        cmap="viridis",
        alpha=0.75,
        edgecolor="white",
        linewidth=0.9,
    )
    ax.set_title("Circuit Grid Importance Bubble Chart, 2019-2025")
    ax.set_xlabel("Pole win rate (%)")
    ax.set_ylabel("Average absolute position change")
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.03)
    colorbar.set_label("Grid-finish correlation")

    labeled = sorted(rows, key=lambda row: row["pole_win_rate"], reverse=True)[:5]
    labeled += sorted(rows, key=lambda row: row["avg_abs_position_change"], reverse=True)[:5]
    seen = set()
    for row in labeled:
        if row["circuit_id"] in seen:
            continue
        seen.add(row["circuit_id"])
        ax.annotate(
            row["circuit_name"].replace(" Circuit", ""),
            (row["pole_win_rate"] * 100, row["avg_abs_position_change"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
            color=COLORS["slate"],
        )

    filename = "circuit_grid_importance_bubble_2019_2025.png"
    save_current_figure(filename)
    add_manifest(
        manifest,
        filename,
        "Circuit Grid Importance Bubble Chart",
        "f1_features.csv filtered to 2019-2025",
        "Compares circuits by pole win rate, position-change volatility, race count, and grid-finish correlation.",
    )


def plot_driver_consistency_scatter(manifest, modern_rows):
    rows = build_driver_consistency_rows(modern_rows)
    rows = sorted(rows, key=lambda row: row["avg_points"], reverse=True)[:18]
    x_values = [row["avg_finish_position"] for row in rows]
    y_values = [row["finish_stddev"] for row in rows]
    sizes = [90 + row["podium_rate"] * 500 for row in rows]
    colors = [row["avg_points"] for row in rows]

    fig, ax = setup_figure(11, 7)
    scatter = ax.scatter(
        x_values,
        y_values,
        s=sizes,
        c=colors,
        cmap="plasma",
        alpha=0.78,
        edgecolor="white",
        linewidth=0.9,
    )
    ax.set_title("Driver Performance and Consistency, 2019-2025")
    ax.set_xlabel("Average finish position")
    ax.set_ylabel("Finish position standard deviation")
    ax.invert_xaxis()
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.03)
    colorbar.set_label("Average points")

    for row in rows[:10]:
        ax.annotate(
            row["driver_name"].split()[-1],
            (row["avg_finish_position"], row["finish_stddev"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
            color=COLORS["slate"],
        )

    filename = "driver_consistency_scatter_2019_2025.png"
    save_current_figure(filename)
    add_manifest(
        manifest,
        filename,
        "Driver Performance and Consistency",
        "f1_features.csv filtered to 2019-2025",
        "Compares drivers by average finish, finish variability, podium rate, and average points.",
    )


def plot_constructor_efficiency_scatter(manifest, modern_rows):
    rows = build_constructor_efficiency_rows(modern_rows)
    x_values = [row["avg_grid"] for row in rows]
    y_values = [row["avg_finish_position"] for row in rows]
    sizes = [100 + row["podium_rate"] * 700 for row in rows]
    colors = [row["avg_points"] for row in rows]

    fig, ax = setup_figure(10, 7)
    scatter = ax.scatter(
        x_values,
        y_values,
        s=sizes,
        c=colors,
        cmap="YlOrRd",
        alpha=0.78,
        edgecolor="white",
        linewidth=0.9,
    )
    ax.plot([min(x_values), max(x_values)], [min(x_values), max(x_values)], color=COLORS["gray"], linestyle="--", linewidth=1)
    ax.set_title("Constructor Qualifying-to-Race Efficiency, 2019-2025")
    ax.set_xlabel("Average grid position")
    ax.set_ylabel("Average finish position")
    ax.invert_xaxis()
    ax.invert_yaxis()
    colorbar = fig.colorbar(scatter, ax=ax, fraction=0.035, pad=0.03)
    colorbar.set_label("Average points per car")

    for row in rows:
        ax.annotate(
            row["constructor_name"].replace(" F1 Team", ""),
            (row["avg_grid"], row["avg_finish_position"]),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
            color=COLORS["slate"],
        )

    filename = "constructor_qualifying_race_efficiency_2019_2025.png"
    save_current_figure(filename)
    add_manifest(
        manifest,
        filename,
        "Constructor Qualifying-to-Race Efficiency",
        "f1_features.csv filtered to 2019-2025",
        "Compares teams' average grid position with average finish position, sized by podium rate.",
    )


def plot_position_change_distribution(manifest, modern_rows):
    changes = [
        to_int(row["grid"]) - to_int(row["finish_position"])
        for row in modern_rows
        if to_int(row["grid"]) > 0
    ]
    bins = list(range(-20, 21, 2))

    fig, ax = setup_figure(10, 5.8)
    ax.hist(changes, bins=bins, color=COLORS["blue"], alpha=0.82, edgecolor="white")
    ax.axvline(0, color=COLORS["red"], linewidth=1.5)
    ax.set_title("Race Position Change Distribution, 2019-2025")
    ax.set_xlabel("Grid position minus finish position")
    ax.set_ylabel("Record count")
    ax.text(
        0.02,
        0.93,
        "Positive values mean finishing ahead of starting position.",
        transform=ax.transAxes,
        fontsize=8,
        color=COLORS["gray"],
    )

    filename = "position_change_distribution_2019_2025.png"
    save_current_figure(filename)
    add_manifest(
        manifest,
        filename,
        "Race Position Change Distribution",
        "f1_features.csv filtered to 2019-2025",
        "Shows how often drivers gained or lost positions relative to their grid starts.",
    )


def plot_feature_correlation_heatmap(manifest, modern_rows):
    features, targets, matrix = build_feature_correlation_matrix(modern_rows)
    labels = {
        "grid": "Grid",
        "qualifying_position": "Qualifying",
        "driver_pre_race_points": "Driver pre pts",
        "driver_pre_race_rank": "Driver pre rank",
        "constructor_pre_race_points": "Team pre pts",
        "constructor_pre_race_rank": "Team pre rank",
        "driver_last3_avg_points": "Driver L3 pts",
        "driver_last3_avg_finish_position": "Driver L3 finish",
        "driver_last3_podium_count": "Driver L3 podiums",
        "constructor_last3_avg_points": "Team L3 pts",
        "constructor_last3_podium_count": "Team L3 podiums",
        "finish_position": "Finish",
        "is_podium": "Podium",
        "is_top10": "Top 10",
    }

    fig, ax = setup_figure(8.5, 7.5)
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_title("Feature Correlation Heatmap, 2019-2025")
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels([labels[target] for target in targets])
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels([labels[feature] for feature in features])
    ax.grid(False)
    for y_index, row_values in enumerate(matrix):
        for x_index, value in enumerate(row_values):
            ax.text(
                x_index,
                y_index,
                f"{value:.2f}" if value is not None else "",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value is not None and abs(value) > 0.45 else "#111827",
            )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    colorbar.set_label("Pearson correlation")

    filename = "feature_correlation_heatmap_2019_2025.png"
    save_current_figure(filename)
    add_manifest(
        manifest,
        filename,
        "Feature Correlation Heatmap",
        "f1_features.csv filtered to 2019-2025",
        "Summarizes correlations between race features and finish, podium, and top-10 outcomes.",
    )


def plot_constructor_competitiveness(manifest):
    rows = [
        row
        for row in read_csv(ANALYSIS_DIR / "constructor_competitiveness_by_year.csv")
        if 2019 <= to_int(row["season"]) <= 2025
    ]
    years = [row["season"] for row in rows]
    top1 = [to_float(row["top1_points_share"]) * 100 for row in rows]
    top2 = [to_float(row["top2_points_share"]) * 100 for row in rows]
    top3 = [to_float(row["top3_points_share"]) * 100 for row in rows]

    fig, ax = setup_figure(10, 5.8)
    ax.plot(years, top1, marker="o", linewidth=2.2, color=COLORS["red"], label="Top 1 team")
    ax.plot(years, top2, marker="o", linewidth=2.2, color=COLORS["orange"], label="Top 2 teams")
    ax.plot(years, top3, marker="o", linewidth=2.2, color=COLORS["blue"], label="Top 3 teams")
    ax.set_title("Constructor Points Concentration by Season, 2019-2025")
    ax.set_xlabel("Season")
    ax.set_ylabel("Share of total constructor points (%)")
    ax.set_ylim(0, 100)
    ax.legend()

    filename = "constructor_competitiveness_by_year_2019_2025.png"
    save_current_figure(filename)
    add_manifest(
        manifest,
        filename,
        "Constructor Points Concentration by Season",
        "constructor_competitiveness_by_year.csv filtered to 2019-2025",
        "Shows how much of each season's constructor points were captured by the top teams.",
    )


def plot_constructor_hhi(manifest):
    rows = [
        row
        for row in read_csv(ANALYSIS_DIR / "constructor_competitiveness_by_year.csv")
        if 2019 <= to_int(row["season"]) <= 2025
    ]
    years = [row["season"] for row in rows]
    hhi = [to_float(row["hhi"]) for row in rows]
    effective_count = [to_float(row["effective_constructor_count"]) for row in rows]

    fig, ax = setup_figure(10, 5.8)
    ax.plot(years, hhi, marker="o", linewidth=2.2, color=COLORS["red"], label="HHI")
    ax.set_title("Constructor Championship Concentration Index, 2019-2025")
    ax.set_xlabel("Season")
    ax.set_ylabel("HHI")
    ax2 = ax.twinx()
    ax2.plot(
        years,
        effective_count,
        marker="s",
        linewidth=2.0,
        color=COLORS["blue"],
        label="Effective constructor count",
    )
    ax2.set_ylabel("Effective constructor count")
    ax2.grid(False)
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    filename = "constructor_hhi_competitiveness_2019_2025.png"
    save_current_figure(filename)
    add_manifest(
        manifest,
        filename,
        "Constructor Championship Concentration Index",
        "constructor_competitiveness_by_year.csv filtered to 2019-2025",
        "Uses HHI and effective constructor count to measure season-level competitive concentration.",
    )


def plot_pre_race_strength_bins(manifest):
    rows = [
        row
        for row in read_csv(ANALYSIS_DIR / "pre_race_strength_bins.csv")
        if row["variable"] in {"driver_pre_race_rank", "driver_last3_avg_points"}
    ]
    labels_by_variable = {
        "driver_pre_race_rank": "Driver pre-race rank",
        "driver_last3_avg_points": "Driver last-3 average points",
    }

    fig, ax = setup_figure(10, 5.8)
    for variable, color in [
        ("driver_pre_race_rank", COLORS["blue"]),
        ("driver_last3_avg_points", COLORS["green"]),
    ]:
        group = sorted(
            [row for row in rows if row["variable"] == variable],
            key=lambda row: to_int(row["bin_sort"]),
        )
        x_values = [row["bin"] for row in group]
        y_values = [to_float(row["podium_rate"]) * 100 for row in group]
        ax.plot(
            x_values,
            y_values,
            marker="o",
            linewidth=2.2,
            color=color,
            label=labels_by_variable[variable],
        )

    ax.set_title("Pre-Race Strength and Podium Rate, 2019-2025")
    ax.set_xlabel("Strength bin")
    ax.set_ylabel("Podium rate (%)")
    ax.set_ylim(0, 70)
    ax.legend()

    filename = "pre_race_strength_podium_rate_2019_2025.png"
    save_current_figure(filename)
    add_manifest(
        manifest,
        filename,
        "Pre-Race Strength and Podium Rate",
        "pre_race_strength_bins.csv filtered to 2019-2025 variables",
        "Shows whether pre-race standings and recent form separate podium probability.",
    )


def plot_circuit_grid_importance_score(manifest):
    rows = read_csv(ANALYSIS_DIR / "circuit_grid_importance_score.csv")[:12]
    rows = list(reversed(rows))
    labels = [row["circuit_name"] for row in rows]
    scores = [to_float(row["grid_importance_score"]) for row in rows]

    fig, ax = setup_figure(11, 7)
    bars = ax.barh(labels, scores, color=COLORS["teal"])
    ax.set_title("Circuit Grid Importance Score, 2019-2025")
    ax.set_xlabel("Composite grid importance score")
    ax.set_xlim(0, max(scores) * 1.18)
    add_bar_labels(ax, bars, "{:.3f}", padding=max(scores) * 0.01)

    filename = "circuit_grid_importance_score_2019_2025.png"
    save_current_figure(filename)
    add_manifest(
        manifest,
        filename,
        "Circuit Grid Importance Score",
        "circuit_grid_importance_score.csv",
        "Ranks circuits by a composite score using pole win rate, front-3 podium rate, and grid-finish correlation.",
    )


def plot_position_gain_leaders(manifest, modern_rows):
    driver_rows = build_position_gain_summary(modern_rows, "driver_id", "driver_name")[:12]
    constructor_rows = build_position_gain_summary(
        modern_rows, "constructor_id", "constructor_name"
    )[:10]

    driver_rows = list(reversed(driver_rows))
    fig, ax = setup_figure(10, 6.5)
    labels = [row["driver_name"] for row in driver_rows]
    values = [row["avg_position_change"] for row in driver_rows]
    bars = ax.barh(labels, values, color=COLORS["green"])
    ax.axvline(0, color="#111827", linewidth=1)
    ax.set_title("Top Driver Average Position Gains, 2019-2025")
    ax.set_xlabel("Average grid position minus finish position")
    add_bar_labels(ax, bars, "{:.2f}", padding=max(values) * 0.01)
    filename = "top_driver_position_gain_2019_2025.png"
    save_current_figure(filename)
    add_manifest(
        manifest,
        filename,
        "Top Driver Average Position Gains",
        "f1_features.csv filtered to 2019-2025",
        "Ranks drivers by average race position gain from grid to finish.",
    )

    constructor_rows = list(reversed(constructor_rows))
    fig, ax = setup_figure(10, 6)
    labels = [row["constructor_name"] for row in constructor_rows]
    values = [row["avg_position_change"] for row in constructor_rows]
    colors = [TEAM_COLORS.get(label, COLORS["green"]) for label in labels]
    bars = ax.barh(labels, values, color=colors)
    ax.axvline(0, color="#111827", linewidth=1)
    ax.set_title("Top Constructor Average Position Gains, 2019-2025")
    ax.set_xlabel("Average grid position minus finish position")
    add_bar_labels(ax, bars, "{:.2f}", padding=max(values) * 0.01)
    filename = "top_constructor_position_gain_2019_2025.png"
    save_current_figure(filename)
    add_manifest(
        manifest,
        filename,
        "Top Constructor Average Position Gains",
        "f1_features.csv filtered to 2019-2025",
        "Ranks constructors by average race position gain from grid to finish.",
    )


def main():
    clear_old_figures()
    manifest = []
    modern_rows = load_modern_history_rows()
    plot_yearly_records(manifest)
    plot_grid_rates(manifest, modern_rows)
    plot_qualifying_rates(manifest, modern_rows)
    plot_modern_vs_historical_grid(manifest, modern_rows)
    plot_top_driver_points(manifest, modern_rows)
    plot_constructor_points(manifest, modern_rows)
    plot_driver_points_trend(manifest)
    plot_constructor_points_trend(manifest)
    plot_historical_races_by_year(manifest)
    plot_historical_driver_wins(manifest)
    plot_historical_constructor_wins(manifest)
    plot_circuit_pole_win_rate(manifest, modern_rows)
    plot_driver_points_heatmap(manifest)
    plot_constructor_points_share(manifest)
    plot_constructor_podiums(manifest, modern_rows)
    plot_driver_podiums(manifest, modern_rows)
    plot_status_by_year(manifest, modern_rows)
    plot_circuit_position_change(manifest, modern_rows)
    plot_qualifying_finish_heatmap(manifest, modern_rows)
    plot_constructor_rank_bump_chart(manifest)
    plot_circuit_grid_importance_bubble(manifest, modern_rows)
    plot_driver_consistency_scatter(manifest, modern_rows)
    plot_constructor_efficiency_scatter(manifest, modern_rows)
    plot_position_change_distribution(manifest, modern_rows)
    plot_feature_correlation_heatmap(manifest, modern_rows)
    plot_constructor_competitiveness(manifest)
    plot_constructor_hhi(manifest)
    plot_pre_race_strength_bins(manifest)
    plot_circuit_grid_importance_score(manifest)
    plot_position_gain_leaders(manifest, modern_rows)

    write_csv(MANIFEST_CSV, ["filename", "title", "source", "description"], manifest)
    write_json(
        MANIFEST_JSON,
        {
            "built_at": datetime.now().isoformat(timespec="seconds"),
            "figure_dir": str(FIGURE_DIR.relative_to(BASE_DIR)),
            "figure_count": len(manifest),
            "figures": manifest,
        },
    )

    print(f"Figures saved to: {FIGURE_DIR}")
    print(f"Figure count: {len(manifest)}")
    print(f"Manifest saved to: {MANIFEST_CSV}")


if __name__ == "__main__":
    main()
