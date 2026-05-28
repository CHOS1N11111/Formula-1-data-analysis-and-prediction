import csv
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
FEATURES_PATH = BASE_DIR / "data" / "processed" / "f1_features.csv"
SCHEDULE_2026_PATH = BASE_DIR / "data" / "processed" / "f1_2026_schedule.csv"
ANALYSIS_DIR = BASE_DIR / "data" / "analysis"
SUMMARY_PATH = ANALYSIS_DIR / "analysis_summary.json"


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def to_int(value, default=None):
    try:
        if value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def to_float(value, default=None):
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def format_float(value, digits=4):
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return f"{value:.{digits}f}"


def average(values):
    clean_values = [value for value in values if value is not None]
    if not clean_values:
        return None
    return sum(clean_values) / len(clean_values)


def pearson_correlation(x_values, y_values):
    pairs = [
        (x, y)
        for x, y in zip(x_values, y_values)
        if x is not None and y is not None
    ]
    if len(pairs) < 2:
        return None

    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    denominator_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denominator_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))

    if denominator_x == 0 or denominator_y == 0:
        return None
    return numerator / (denominator_x * denominator_y)


def pct(numerator, denominator):
    if denominator == 0:
        return None
    return numerator / denominator


def sort_rows(rows, key, reverse=True):
    return sorted(rows, key=lambda row: to_float(row[key], 0.0), reverse=reverse)


def build_dataset_overview(rows):
    years = sorted({to_int(row["season"]) for row in rows})
    overview = [
        {"metric": "record_count", "value": len(rows)},
        {"metric": "season_min", "value": min(years)},
        {"metric": "season_max", "value": max(years)},
        {"metric": "race_count", "value": len({(row["season"], row["round"]) for row in rows})},
        {"metric": "driver_count", "value": len({row["driver_id"] for row in rows})},
        {"metric": "constructor_count", "value": len({row["constructor_id"] for row in rows})},
        {"metric": "circuit_count", "value": len({row["circuit_id"] for row in rows})},
        {"metric": "podium_records", "value": sum(to_int(row["is_podium"], 0) for row in rows)},
        {"metric": "top10_records", "value": sum(to_int(row["is_top10"], 0) for row in rows)},
    ]

    by_year = []
    for year in years:
        year_rows = [row for row in rows if to_int(row["season"]) == year]
        by_year.append(
            {
                "season": year,
                "records": len(year_rows),
                "races": len({row["round"] for row in year_rows}),
                "drivers": len({row["driver_id"] for row in year_rows}),
                "constructors": len({row["constructor_id"] for row in year_rows}),
                "circuits": len({row["circuit_id"] for row in year_rows}),
                "podium_records": sum(to_int(row["is_podium"], 0) for row in year_rows),
                "top10_records": sum(to_int(row["is_top10"], 0) for row in year_rows),
            }
        )

    return overview, by_year


def build_grid_finish_analysis(rows):
    finish_positions = [to_float(row["finish_position"]) for row in rows]
    grid_positions = [to_float(row["grid"]) for row in rows if to_int(row["grid"], 0) > 0]
    grid_finish_positions = [
        to_float(row["finish_position"]) for row in rows if to_int(row["grid"], 0) > 0
    ]
    qualifying_positions = [to_float(row["qualifying_position"]) for row in rows]

    pole_rows = [row for row in rows if to_int(row["grid"]) == 1]
    front3_rows = [row for row in rows if 1 <= to_int(row["grid"], 0) <= 3]
    front10_rows = [row for row in rows if 1 <= to_int(row["grid"], 0) <= 10]
    valid_quali_rows = [row for row in rows if to_int(row["qualifying_position"]) is not None]

    summary = [
        {
            "metric": "grid_finish_correlation",
            "value": format_float(pearson_correlation(grid_positions, grid_finish_positions)),
        },
        {
            "metric": "qualifying_finish_correlation",
            "value": format_float(
                pearson_correlation(
                    [to_float(row["qualifying_position"]) for row in valid_quali_rows],
                    [to_float(row["finish_position"]) for row in valid_quali_rows],
                )
            ),
        },
        {
            "metric": "pole_win_rate",
            "value": format_float(
                pct(
                    sum(1 for row in pole_rows if to_int(row["finish_position"]) == 1),
                    len(pole_rows),
                )
            ),
        },
        {
            "metric": "front3_podium_rate",
            "value": format_float(
                pct(sum(to_int(row["is_podium"], 0) for row in front3_rows), len(front3_rows))
            ),
        },
        {
            "metric": "front10_top10_rate",
            "value": format_float(
                pct(sum(to_int(row["is_top10"], 0) for row in front10_rows), len(front10_rows))
            ),
        },
        {
            "metric": "average_finish_position",
            "value": format_float(average(finish_positions)),
        },
    ]

    by_grid = []
    grid_values = sorted(
        {to_int(row["grid"]) for row in rows if to_int(row["grid"], 0) > 0}
    )
    for grid in grid_values:
        group = [row for row in rows if to_int(row["grid"]) == grid]
        by_grid.append(
            {
                "grid": grid,
                "records": len(group),
                "avg_finish_position": format_float(
                    average([to_float(row["finish_position"]) for row in group])
                ),
                "win_rate": format_float(
                    pct(sum(1 for row in group if to_int(row["finish_position"]) == 1), len(group))
                ),
                "podium_rate": format_float(
                    pct(sum(to_int(row["is_podium"], 0) for row in group), len(group))
                ),
                "top10_rate": format_float(
                    pct(sum(to_int(row["is_top10"], 0) for row in group), len(group))
                ),
            }
        )

    by_qualifying = []
    qualifying_values = sorted(
        {
            to_int(row["qualifying_position"])
            for row in rows
            if to_int(row["qualifying_position"]) is not None
        }
    )
    for qualifying_position in qualifying_values:
        group = [
            row
            for row in rows
            if to_int(row["qualifying_position"]) == qualifying_position
        ]
        by_qualifying.append(
            {
                "qualifying_position": qualifying_position,
                "records": len(group),
                "avg_finish_position": format_float(
                    average([to_float(row["finish_position"]) for row in group])
                ),
                "win_rate": format_float(
                    pct(sum(1 for row in group if to_int(row["finish_position"]) == 1), len(group))
                ),
                "podium_rate": format_float(
                    pct(sum(to_int(row["is_podium"], 0) for row in group), len(group))
                ),
                "top10_rate": format_float(
                    pct(sum(to_int(row["is_top10"], 0) for row in group), len(group))
                ),
            }
        )

    return summary, by_grid, by_qualifying


def build_driver_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["driver_id"]].append(row)

    summary_rows = []
    for driver_id, group in grouped.items():
        driver_name = group[-1]["driver_name"]
        points = [to_float(row["points"], 0.0) for row in group]
        finishes = [to_float(row["finish_position"]) for row in group]
        grids = [to_float(row["grid"]) for row in group if to_int(row["grid"], 0) > 0]
        qualifying = [
            to_float(row["qualifying_position"])
            for row in group
            if to_int(row["qualifying_position"]) is not None
        ]
        summary_rows.append(
            {
                "driver_id": driver_id,
                "driver_name": driver_name,
                "nationality": group[-1]["driver_nationality"],
                "seasons": len({row["season"] for row in group}),
                "starts": len(group),
                "total_points": format_float(sum(points)),
                "avg_points": format_float(average(points)),
                "wins": sum(1 for row in group if to_int(row["finish_position"]) == 1),
                "podiums": sum(to_int(row["is_podium"], 0) for row in group),
                "top10s": sum(to_int(row["is_top10"], 0) for row in group),
                "avg_finish_position": format_float(average(finishes)),
                "avg_grid": format_float(average(grids)),
                "avg_qualifying_position": format_float(average(qualifying)),
            }
        )

    return sort_rows(summary_rows, "total_points")


def build_driver_points_by_year(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["season"], row["driver_id"])].append(row)

    summary_rows = []
    for (season, driver_id), group in grouped.items():
        points = [to_float(row["points"], 0.0) for row in group]
        summary_rows.append(
            {
                "season": season,
                "driver_id": driver_id,
                "driver_name": group[-1]["driver_name"],
                "constructor_names": ", ".join(
                    sorted({row["constructor_name"] for row in group})
                ),
                "starts": len(group),
                "total_points": format_float(sum(points)),
                "wins": sum(1 for row in group if to_int(row["finish_position"]) == 1),
                "podiums": sum(to_int(row["is_podium"], 0) for row in group),
                "top10s": sum(to_int(row["is_top10"], 0) for row in group),
                "avg_finish_position": format_float(
                    average([to_float(row["finish_position"]) for row in group])
                ),
            }
        )

    return sorted(
        summary_rows,
        key=lambda row: (to_int(row["season"], 0), -to_float(row["total_points"], 0.0)),
    )


def build_constructor_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["constructor_id"]].append(row)

    summary_rows = []
    for constructor_id, group in grouped.items():
        points = [to_float(row["points"], 0.0) for row in group]
        finishes = [to_float(row["finish_position"]) for row in group]
        grids = [to_float(row["grid"]) for row in group if to_int(row["grid"], 0) > 0]
        qualifying = [
            to_float(row["qualifying_position"])
            for row in group
            if to_int(row["qualifying_position"]) is not None
        ]
        summary_rows.append(
            {
                "constructor_id": constructor_id,
                "constructor_name": group[-1]["constructor_name"],
                "nationality": group[-1]["constructor_nationality"],
                "seasons": len({row["season"] for row in group}),
                "records": len(group),
                "total_points": format_float(sum(points)),
                "avg_points_per_car": format_float(average(points)),
                "wins": sum(1 for row in group if to_int(row["finish_position"]) == 1),
                "podiums": sum(to_int(row["is_podium"], 0) for row in group),
                "top10s": sum(to_int(row["is_top10"], 0) for row in group),
                "avg_finish_position": format_float(average(finishes)),
                "avg_grid": format_float(average(grids)),
                "avg_qualifying_position": format_float(average(qualifying)),
            }
        )

    return sort_rows(summary_rows, "total_points")


def build_constructor_points_by_year(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["season"], row["constructor_id"])].append(row)

    summary_rows = []
    for (season, constructor_id), group in grouped.items():
        points = [to_float(row["points"], 0.0) for row in group]
        summary_rows.append(
            {
                "season": season,
                "constructor_id": constructor_id,
                "constructor_name": group[-1]["constructor_name"],
                "records": len(group),
                "total_points": format_float(sum(points)),
                "wins": sum(1 for row in group if to_int(row["finish_position"]) == 1),
                "podiums": sum(to_int(row["is_podium"], 0) for row in group),
                "top10s": sum(to_int(row["is_top10"], 0) for row in group),
                "avg_finish_position": format_float(
                    average([to_float(row["finish_position"]) for row in group])
                ),
            }
        )

    return sorted(
        summary_rows,
        key=lambda row: (to_int(row["season"], 0), -to_float(row["total_points"], 0.0)),
    )


def build_circuit_summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["circuit_id"]].append(row)

    summary_rows = []
    for circuit_id, group in grouped.items():
        races = {(row["season"], row["round"]) for row in group}
        pole_rows = [row for row in group if to_int(row["grid"]) == 1]
        front3_rows = [row for row in group if 1 <= to_int(row["grid"], 0) <= 3]
        winners = {row["constructor_id"] for row in group if to_int(row["finish_position"]) == 1}
        position_changes = [
            to_int(row["grid"]) - to_int(row["finish_position"])
            for row in group
            if to_int(row["grid"], 0) > 0
        ]

        summary_rows.append(
            {
                "circuit_id": circuit_id,
                "circuit_name": group[-1]["circuit_name"],
                "country": group[-1]["circuit_country"],
                "race_count": len(races),
                "records": len(group),
                "avg_position_change": format_float(average(position_changes)),
                "pole_win_rate": format_float(
                    pct(sum(1 for row in pole_rows if to_int(row["finish_position"]) == 1), len(pole_rows))
                ),
                "front3_podium_rate": format_float(
                    pct(sum(to_int(row["is_podium"], 0) for row in front3_rows), len(front3_rows))
                ),
                "winner_constructor_count": len(winners),
            }
        )

    return sorted(summary_rows, key=lambda row: row["circuit_id"])


def build_2026_outputs(rows, schedule_rows):
    current_rows = [row for row in rows if row["season"] == "2026"]

    driver_points = defaultdict(float)
    driver_podiums = defaultdict(int)
    driver_top10s = defaultdict(int)
    driver_names = {}
    for row in current_rows:
        driver_id = row["driver_id"]
        driver_names[driver_id] = row["driver_name"]
        driver_points[driver_id] += to_float(row["points"], 0.0)
        driver_podiums[driver_id] += to_int(row["is_podium"], 0)
        driver_top10s[driver_id] += to_int(row["is_top10"], 0)

    driver_standings = []
    for driver_id, points in driver_points.items():
        driver_standings.append(
            {
                "driver_id": driver_id,
                "driver_name": driver_names[driver_id],
                "points": format_float(points),
                "podiums": driver_podiums[driver_id],
                "top10s": driver_top10s[driver_id],
            }
        )
    driver_standings = sort_rows(driver_standings, "points")
    for index, row in enumerate(driver_standings, start=1):
        row["rank"] = index

    constructor_points = defaultdict(float)
    constructor_podiums = defaultdict(int)
    constructor_top10s = defaultdict(int)
    constructor_names = {}
    for row in current_rows:
        constructor_id = row["constructor_id"]
        constructor_names[constructor_id] = row["constructor_name"]
        constructor_points[constructor_id] += to_float(row["points"], 0.0)
        constructor_podiums[constructor_id] += to_int(row["is_podium"], 0)
        constructor_top10s[constructor_id] += to_int(row["is_top10"], 0)

    constructor_standings = []
    for constructor_id, points in constructor_points.items():
        constructor_standings.append(
            {
                "constructor_id": constructor_id,
                "constructor_name": constructor_names[constructor_id],
                "points": format_float(points),
                "podiums": constructor_podiums[constructor_id],
                "top10s": constructor_top10s[constructor_id],
            }
        )
    constructor_standings = sort_rows(constructor_standings, "points")
    for index, row in enumerate(constructor_standings, start=1):
        row["rank"] = index

    completed_results = sorted(
        current_rows,
        key=lambda row: (to_int(row["round"], 0), to_int(row["finish_position"], 99)),
    )
    remaining_schedule = [row for row in schedule_rows if row["has_result"] == "0"]

    return driver_standings, constructor_standings, completed_results, remaining_schedule


def main():
    rows = read_csv(FEATURES_PATH)
    schedule_rows = read_csv(SCHEDULE_2026_PATH)

    overview, overview_by_year = build_dataset_overview(rows)
    grid_summary, by_grid, by_qualifying = build_grid_finish_analysis(rows)
    driver_summary = build_driver_summary(rows)
    driver_points_by_year = build_driver_points_by_year(rows)
    constructor_summary = build_constructor_summary(rows)
    constructor_points_by_year = build_constructor_points_by_year(rows)
    circuit_summary = build_circuit_summary(rows)
    (
        current_driver_standings,
        current_constructor_standings,
        completed_2026_results,
        remaining_2026_schedule,
    ) = build_2026_outputs(rows, schedule_rows)

    outputs = [
        ("dataset_overview.csv", ["metric", "value"], overview),
        (
            "dataset_overview_by_year.csv",
            ["season", "records", "races", "drivers", "constructors", "circuits", "podium_records", "top10_records"],
            overview_by_year,
        ),
        ("grid_finish_summary.csv", ["metric", "value"], grid_summary),
        (
            "grid_position_summary.csv",
            ["grid", "records", "avg_finish_position", "win_rate", "podium_rate", "top10_rate"],
            by_grid,
        ),
        (
            "qualifying_position_summary.csv",
            ["qualifying_position", "records", "avg_finish_position", "win_rate", "podium_rate", "top10_rate"],
            by_qualifying,
        ),
        (
            "driver_summary.csv",
            ["driver_id", "driver_name", "nationality", "seasons", "starts", "total_points", "avg_points", "wins", "podiums", "top10s", "avg_finish_position", "avg_grid", "avg_qualifying_position"],
            driver_summary,
        ),
        (
            "driver_points_by_year.csv",
            ["season", "driver_id", "driver_name", "constructor_names", "starts", "total_points", "wins", "podiums", "top10s", "avg_finish_position"],
            driver_points_by_year,
        ),
        (
            "constructor_summary.csv",
            ["constructor_id", "constructor_name", "nationality", "seasons", "records", "total_points", "avg_points_per_car", "wins", "podiums", "top10s", "avg_finish_position", "avg_grid", "avg_qualifying_position"],
            constructor_summary,
        ),
        (
            "constructor_points_by_year.csv",
            ["season", "constructor_id", "constructor_name", "records", "total_points", "wins", "podiums", "top10s", "avg_finish_position"],
            constructor_points_by_year,
        ),
        (
            "circuit_summary.csv",
            ["circuit_id", "circuit_name", "country", "race_count", "records", "avg_position_change", "pole_win_rate", "front3_podium_rate", "winner_constructor_count"],
            circuit_summary,
        ),
        (
            "current_2026_driver_standings.csv",
            ["rank", "driver_id", "driver_name", "points", "podiums", "top10s"],
            current_driver_standings,
        ),
        (
            "current_2026_constructor_standings.csv",
            ["rank", "constructor_id", "constructor_name", "points", "podiums", "top10s"],
            current_constructor_standings,
        ),
        (
            "completed_2026_results.csv",
            ["season", "round", "race_name", "race_date", "driver_id", "driver_name", "constructor_id", "constructor_name", "grid", "qualifying_position", "finish_position", "points", "status"],
            completed_2026_results,
        ),
        (
            "remaining_2026_schedule.csv",
            ["season", "round", "race_name", "race_date", "race_time", "circuit_id", "circuit_name", "circuit_locality", "circuit_country", "has_result"],
            remaining_2026_schedule,
        ),
    ]

    for filename, fieldnames, output_rows in outputs:
        write_csv(ANALYSIS_DIR / filename, fieldnames, output_rows)

    summary = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "input_features": str(FEATURES_PATH.relative_to(BASE_DIR)),
        "input_2026_schedule": str(SCHEDULE_2026_PATH.relative_to(BASE_DIR)),
        "output_dir": str(ANALYSIS_DIR.relative_to(BASE_DIR)),
        "output_files": [filename for filename, _, _ in outputs],
        "record_count": len(rows),
        "current_2026_completed_races": len({row["round"] for row in completed_2026_results}),
        "current_2026_remaining_races": len(remaining_2026_schedule),
    }
    write_json(SUMMARY_PATH, summary)

    print(f"Analysis files saved to: {ANALYSIS_DIR}")
    print(f"Summary saved to: {SUMMARY_PATH}")
    print(f"Output files: {len(outputs)}")


if __name__ == "__main__":
    main()
