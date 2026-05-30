"""Generate modern F1 statistical analysis CSV files from engineered features."""

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


def spearman_correlation(x_values, y_values):
    pairs = [
        (x, y)
        for x, y in zip(x_values, y_values)
        if x is not None and y is not None
    ]
    if len(pairs) < 2:
        return None

    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    return pearson_correlation(rank_values(xs), rank_values(ys))


def rank_values(values):
    sorted_values = sorted((value, index) for index, value in enumerate(values))
    ranks = [0.0] * len(values)
    position = 0
    while position < len(sorted_values):
        end = position
        while end + 1 < len(sorted_values) and sorted_values[end + 1][0] == sorted_values[position][0]:
            end += 1
        average_rank = (position + 1 + end + 1) / 2
        for item_index in range(position, end + 1):
            original_index = sorted_values[item_index][1]
            ranks[original_index] = average_rank
        position = end + 1
    return ranks


def pct(numerator, denominator):
    if denominator == 0:
        return None
    return numerator / denominator


def sort_rows(rows, key, reverse=True):
    return sorted(rows, key=lambda row: to_float(row[key], 0.0), reverse=reverse)


def build_dataset_overview(rows):
    """Summarize row counts, seasons, races, drivers, teams, and circuits."""
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
    """Measure how grid and qualifying positions relate to race outcomes."""
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


def build_feature_correlation_summary(rows):
    fields = [
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
        "driver_history_count",
        "constructor_history_count",
        "missing_qualifying",
        "grid_is_zero",
    ]
    targets = ["finish_position", "is_podium", "is_top10"]
    output_rows = []

    for field in fields:
        x_values = [to_float(row[field]) for row in rows]
        for target in targets:
            y_values = [to_float(row[target]) for row in rows]
            output_rows.append(
                {
                    "feature": field,
                    "target": target,
                    "pearson_correlation": format_float(
                        pearson_correlation(x_values, y_values)
                    ),
                    "spearman_correlation": format_float(
                        spearman_correlation(x_values, y_values)
                    ),
                    "records": sum(
                        1
                        for x_value, y_value in zip(x_values, y_values)
                        if x_value is not None and y_value is not None
                    ),
                }
            )

    return output_rows


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


def build_constructor_competitiveness_by_year(constructor_points_by_year):
    grouped = defaultdict(list)
    for row in constructor_points_by_year:
        grouped[row["season"]].append(row)

    output_rows = []
    for season, group in sorted(grouped.items(), key=lambda item: to_int(item[0])):
        sorted_group = sorted(
            group, key=lambda row: to_float(row["total_points"], 0.0), reverse=True
        )
        total_points = sum(to_float(row["total_points"], 0.0) for row in sorted_group)
        shares = [
            to_float(row["total_points"], 0.0) / total_points
            for row in sorted_group
            if total_points > 0
        ]
        hhi = sum(share**2 for share in shares)
        top1 = sum(to_float(row["total_points"], 0.0) for row in sorted_group[:1])
        top2 = sum(to_float(row["total_points"], 0.0) for row in sorted_group[:2])
        top3 = sum(to_float(row["total_points"], 0.0) for row in sorted_group[:3])
        output_rows.append(
            {
                "season": season,
                "constructor_count": len(sorted_group),
                "total_points": format_float(total_points),
                "top_constructor": sorted_group[0]["constructor_name"] if sorted_group else "",
                "top1_points_share": format_float(pct(top1, total_points)),
                "top2_points_share": format_float(pct(top2, total_points)),
                "top3_points_share": format_float(pct(top3, total_points)),
                "hhi": format_float(hhi),
                "effective_constructor_count": format_float(1 / hhi if hhi else None),
            }
        )

    return output_rows


def assign_rank_bin(rank):
    rank = to_int(rank)
    if rank is None:
        return "unknown"
    if rank <= 3:
        return "1-3"
    if rank <= 6:
        return "4-6"
    if rank <= 10:
        return "7-10"
    return "11+"


def assign_points_bin(value):
    value = to_float(value)
    if value is None:
        return "unknown"
    if value == 0:
        return "0"
    if value < 5:
        return "0-5"
    if value < 10:
        return "5-10"
    if value < 15:
        return "10-15"
    return "15+"


def summarize_bin(rows, variable, bin_name, bin_value):
    return {
        "variable": variable,
        "bin": bin_name,
        "bin_sort": bin_value,
        "records": len(rows),
        "podium_rate": format_float(pct(sum(to_int(row["is_podium"], 0) for row in rows), len(rows))),
        "top10_rate": format_float(pct(sum(to_int(row["is_top10"], 0) for row in rows), len(rows))),
        "avg_finish_position": format_float(average([to_float(row["finish_position"]) for row in rows])),
        "avg_points": format_float(average([to_float(row["points"]) for row in rows])),
    }


def build_pre_race_strength_bins(rows):
    configs = [
        ("driver_pre_race_rank", lambda row: assign_rank_bin(row["driver_pre_race_rank"]), {"1-3": 1, "4-6": 2, "7-10": 3, "11+": 4}),
        ("constructor_pre_race_rank", lambda row: assign_rank_bin(row["constructor_pre_race_rank"]), {"1-3": 1, "4-6": 2, "7-10": 3, "11+": 4}),
        ("driver_last3_avg_points", lambda row: assign_points_bin(row["driver_last3_avg_points"]), {"0": 1, "0-5": 2, "5-10": 3, "10-15": 4, "15+": 5}),
        ("constructor_last3_avg_points", lambda row: assign_points_bin(row["constructor_last3_avg_points"]), {"0": 1, "0-5": 2, "5-10": 3, "10-15": 4, "15+": 5}),
    ]
    output_rows = []

    for variable, bin_func, bin_order in configs:
        grouped = defaultdict(list)
        for row in rows:
            grouped[bin_func(row)].append(row)
        for bin_name, group in grouped.items():
            output_rows.append(
                summarize_bin(
                    group,
                    variable,
                    bin_name,
                    bin_order.get(bin_name, 99),
                )
            )

    return sorted(output_rows, key=lambda row: (row["variable"], row["bin_sort"]))


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


def build_circuit_grid_importance_score(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["circuit_id"]].append(row)

    output_rows = []
    for circuit_id, group in grouped.items():
        races = {(row["season"], row["round"]) for row in group}
        valid_grid_rows = [row for row in group if to_int(row["grid"], 0) > 0]
        valid_quali_rows = [row for row in group if row["qualifying_position"] != ""]
        pole_rows = [row for row in valid_grid_rows if to_int(row["grid"]) == 1]
        front3_rows = [row for row in valid_grid_rows if 1 <= to_int(row["grid"]) <= 3]
        if len(races) < 3 or len(valid_grid_rows) < 20 or not pole_rows:
            continue

        grid_corr = pearson_correlation(
            [to_float(row["grid"]) for row in valid_grid_rows],
            [to_float(row["finish_position"]) for row in valid_grid_rows],
        )
        qualifying_corr = pearson_correlation(
            [to_float(row["qualifying_position"]) for row in valid_quali_rows],
            [to_float(row["finish_position"]) for row in valid_quali_rows],
        )
        pole_win_rate = pct(
            sum(1 for row in pole_rows if to_int(row["finish_position"]) == 1),
            len(pole_rows),
        )
        front3_podium_rate = pct(
            sum(to_int(row["is_podium"], 0) for row in front3_rows),
            len(front3_rows),
        )
        avg_abs_position_change = average(
            [
                abs(to_int(row["grid"]) - to_int(row["finish_position"]))
                for row in valid_grid_rows
            ]
        )
        grid_importance_score = (
            0.4 * pole_win_rate
            + 0.3 * front3_podium_rate
            + 0.3 * max(grid_corr or 0, 0)
        )

        output_rows.append(
            {
                "circuit_id": circuit_id,
                "circuit_name": group[-1]["circuit_name"],
                "country": group[-1]["circuit_country"],
                "race_count": len(races),
                "pole_win_rate": format_float(pole_win_rate),
                "front3_podium_rate": format_float(front3_podium_rate),
                "grid_finish_correlation": format_float(grid_corr),
                "qualifying_finish_correlation": format_float(qualifying_corr),
                "avg_abs_position_change": format_float(avg_abs_position_change),
                "grid_importance_score": format_float(grid_importance_score),
            }
        )

    return sorted(
        output_rows,
        key=lambda row: to_float(row["grid_importance_score"], 0.0),
        reverse=True,
    )


def build_position_gain_summary(rows, group_key, name_key, min_records=20):
    grouped = defaultdict(list)
    for row in rows:
        grid = to_int(row["grid"])
        finish = to_int(row["finish_position"])
        if grid is None or finish is None or grid <= 0:
            continue
        grouped[row[group_key]].append(row)

    summary_rows = []
    for item_id, group in grouped.items():
        if len(group) < min_records:
            continue
        changes = [
            to_int(row["grid"]) - to_int(row["finish_position"])
            for row in group
            if to_int(row["grid"], 0) > 0
        ]
        positive_count = sum(1 for value in changes if value > 0)
        large_gain_count = sum(1 for value in changes if value >= 5)
        large_loss_count = sum(1 for value in changes if value <= -5)
        summary_rows.append(
            {
                group_key: item_id,
                name_key: group[-1][name_key],
                "records": len(changes),
                "avg_position_change": format_float(average(changes)),
                "positive_change_rate": format_float(pct(positive_count, len(changes))),
                "large_gain_count": large_gain_count,
                "large_loss_count": large_loss_count,
            }
        )

    return sorted(
        summary_rows,
        key=lambda row: to_float(row["avg_position_change"], 0.0),
        reverse=True,
    )


def classify_status(status):
    normalized = (status or "").strip().lower()
    if normalized == "finished" or "lap" in normalized:
        return "classified"
    if any(keyword in normalized for keyword in ["accident", "collision", "spun"]):
        return "incident"
    return "mechanical_or_other"


def build_reliability_by_year(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["season"]].append(row)

    output_rows = []
    for season, group in sorted(grouped.items(), key=lambda item: to_int(item[0], 0)):
        classified = sum(1 for row in group if classify_status(row["status"]) == "classified")
        incident = sum(1 for row in group if classify_status(row["status"]) == "incident")
        other = len(group) - classified - incident
        output_rows.append(
            {
                "season": season,
                "records": len(group),
                "classified_rate": format_float(pct(classified, len(group))),
                "incident_rate": format_float(pct(incident, len(group))),
                "mechanical_or_other_rate": format_float(pct(other, len(group))),
            }
        )
    return output_rows


def build_reliability_summary(rows, group_key, name_key, min_records=20):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row[group_key]].append(row)

    output_rows = []
    for item_id, group in grouped.items():
        if len(group) < min_records:
            continue
        classified = sum(1 for row in group if classify_status(row["status"]) == "classified")
        incident = sum(1 for row in group if classify_status(row["status"]) == "incident")
        other = len(group) - classified - incident
        output_rows.append(
            {
                group_key: item_id,
                name_key: group[-1][name_key],
                "records": len(group),
                "classified_rate": format_float(pct(classified, len(group))),
                "incident_rate": format_float(pct(incident, len(group))),
                "mechanical_or_other_rate": format_float(pct(other, len(group))),
                "avg_points": format_float(
                    average([to_float(row["points"], 0.0) for row in group])
                ),
            }
        )

    return sorted(
        output_rows,
        key=lambda row: (
            -to_float(row["classified_rate"], 0.0),
            -to_float(row["avg_points"], 0.0),
        ),
    )


def build_teammate_comparison(rows, min_shared_races=8):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["season"], row["constructor_id"], row["round"])].append(row)

    pair_stats = defaultdict(lambda: defaultdict(float))
    for (season, constructor_id, _round), group in grouped.items():
        valid_group = [
            row
            for row in group
            if to_int(row["finish_position"]) is not None
            and to_float(row["points"], 0.0) is not None
        ]
        if len(valid_group) != 2:
            continue

        first, second = sorted(
            valid_group,
            key=lambda row: to_int(row["finish_position"], 99),
        )
        pair_key = (
            season,
            constructor_id,
            first["constructor_name"],
            first["driver_id"],
            first["driver_name"],
            second["driver_id"],
            second["driver_name"],
        )
        reverse_key = (
            season,
            constructor_id,
            first["constructor_name"],
            second["driver_id"],
            second["driver_name"],
            first["driver_id"],
            first["driver_name"],
        )

        pair_stats[pair_key]["shared_races"] += 1
        pair_stats[pair_key]["finish_ahead_count"] += 1
        pair_stats[pair_key]["points"] += to_float(first["points"], 0.0)
        pair_stats[pair_key]["teammate_points"] += to_float(second["points"], 0.0)
        pair_stats[pair_key]["finish_position_sum"] += to_float(first["finish_position"], 0.0)
        pair_stats[pair_key]["teammate_finish_position_sum"] += to_float(second["finish_position"], 0.0)

        pair_stats[reverse_key]["shared_races"] += 1
        pair_stats[reverse_key]["points"] += to_float(second["points"], 0.0)
        pair_stats[reverse_key]["teammate_points"] += to_float(first["points"], 0.0)
        pair_stats[reverse_key]["finish_position_sum"] += to_float(second["finish_position"], 0.0)
        pair_stats[reverse_key]["teammate_finish_position_sum"] += to_float(first["finish_position"], 0.0)

    output_rows = []
    for (
        season,
        constructor_id,
        constructor_name,
        driver_id,
        driver_name,
        teammate_driver_id,
        teammate_driver_name,
    ), stats in pair_stats.items():
        shared_races = int(stats["shared_races"])
        if shared_races < min_shared_races:
            continue
        points = stats["points"]
        teammate_points = stats["teammate_points"]
        output_rows.append(
            {
                "season": season,
                "constructor_id": constructor_id,
                "constructor_name": constructor_name,
                "driver_id": driver_id,
                "driver_name": driver_name,
                "teammate_driver_id": teammate_driver_id,
                "teammate_driver_name": teammate_driver_name,
                "shared_races": shared_races,
                "finish_ahead_rate": format_float(
                    pct(stats["finish_ahead_count"], shared_races)
                ),
                "points": format_float(points),
                "teammate_points": format_float(teammate_points),
                "points_share": format_float(pct(points, points + teammate_points)),
                "avg_finish_position": format_float(
                    stats["finish_position_sum"] / shared_races
                ),
                "teammate_avg_finish_position": format_float(
                    stats["teammate_finish_position_sum"] / shared_races
                ),
            }
        )

    return sorted(
        output_rows,
        key=lambda row: (
            to_int(row["season"], 0),
            row["constructor_name"],
            -to_float(row["points_share"], 0.0),
        ),
    )


def build_circuit_volatility_index(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["circuit_id"]].append(row)

    output_rows = []
    for circuit_id, group in grouped.items():
        races = {(row["season"], row["round"]) for row in group}
        valid_rows = [
            row
            for row in group
            if to_int(row["grid"], 0) > 0 and to_int(row["finish_position"]) is not None
        ]
        if len(races) < 3 or len(valid_rows) < 20:
            continue

        changes = [
            to_int(row["grid"]) - to_int(row["finish_position"])
            for row in valid_rows
        ]
        winner_rows = [row for row in valid_rows if to_int(row["finish_position"]) == 1]
        non_front_row_winners = sum(1 for row in winner_rows if to_int(row["grid"], 99) > 3)
        large_gain_rate = pct(sum(1 for change in changes if change >= 5), len(changes))
        large_loss_rate = pct(sum(1 for change in changes if change <= -5), len(changes))
        volatility_index = (
            0.4 * (large_gain_rate or 0.0)
            + 0.3 * (large_loss_rate or 0.0)
            + 0.3 * pct(non_front_row_winners, len(winner_rows) or 1)
        )

        output_rows.append(
            {
                "circuit_id": circuit_id,
                "circuit_name": group[-1]["circuit_name"],
                "country": group[-1]["circuit_country"],
                "race_count": len(races),
                "avg_abs_position_change": format_float(
                    average([abs(change) for change in changes])
                ),
                "large_gain_rate": format_float(large_gain_rate),
                "large_loss_rate": format_float(large_loss_rate),
                "non_front_row_winner_rate": format_float(
                    pct(non_front_row_winners, len(winner_rows))
                ),
                "volatility_index": format_float(volatility_index),
            }
        )

    return sorted(
        output_rows,
        key=lambda row: to_float(row["volatility_index"], 0.0),
        reverse=True,
    )


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
    analysis_rows = [row for row in rows if to_int(row["season"], 0) <= 2025]

    overview, overview_by_year = build_dataset_overview(analysis_rows)
    grid_summary, by_grid, by_qualifying = build_grid_finish_analysis(analysis_rows)
    feature_correlation_summary = build_feature_correlation_summary(analysis_rows)
    driver_summary = build_driver_summary(analysis_rows)
    driver_points_by_year = build_driver_points_by_year(analysis_rows)
    constructor_summary = build_constructor_summary(analysis_rows)
    constructor_points_by_year = build_constructor_points_by_year(analysis_rows)
    constructor_competitiveness_by_year = build_constructor_competitiveness_by_year(
        constructor_points_by_year
    )
    pre_race_strength_bins = build_pre_race_strength_bins(analysis_rows)
    circuit_summary = build_circuit_summary(analysis_rows)
    circuit_grid_importance_score = build_circuit_grid_importance_score(analysis_rows)
    driver_position_gain_summary = build_position_gain_summary(
        analysis_rows, "driver_id", "driver_name"
    )
    constructor_position_gain_summary = build_position_gain_summary(
        analysis_rows, "constructor_id", "constructor_name"
    )
    reliability_by_year = build_reliability_by_year(analysis_rows)
    driver_reliability_summary = build_reliability_summary(
        analysis_rows, "driver_id", "driver_name"
    )
    constructor_reliability_summary = build_reliability_summary(
        analysis_rows, "constructor_id", "constructor_name"
    )
    teammate_comparison = build_teammate_comparison(analysis_rows)
    circuit_volatility_index = build_circuit_volatility_index(analysis_rows)
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
            "feature_correlation_summary.csv",
            ["feature", "target", "pearson_correlation", "spearman_correlation", "records"],
            feature_correlation_summary,
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
            "constructor_competitiveness_by_year.csv",
            ["season", "constructor_count", "total_points", "top_constructor", "top1_points_share", "top2_points_share", "top3_points_share", "hhi", "effective_constructor_count"],
            constructor_competitiveness_by_year,
        ),
        (
            "pre_race_strength_bins.csv",
            ["variable", "bin", "bin_sort", "records", "podium_rate", "top10_rate", "avg_finish_position", "avg_points"],
            pre_race_strength_bins,
        ),
        (
            "circuit_summary.csv",
            ["circuit_id", "circuit_name", "country", "race_count", "records", "avg_position_change", "pole_win_rate", "front3_podium_rate", "winner_constructor_count"],
            circuit_summary,
        ),
        (
            "circuit_grid_importance_score.csv",
            ["circuit_id", "circuit_name", "country", "race_count", "pole_win_rate", "front3_podium_rate", "grid_finish_correlation", "qualifying_finish_correlation", "avg_abs_position_change", "grid_importance_score"],
            circuit_grid_importance_score,
        ),
        (
            "driver_position_gain_summary.csv",
            ["driver_id", "driver_name", "records", "avg_position_change", "positive_change_rate", "large_gain_count", "large_loss_count"],
            driver_position_gain_summary,
        ),
        (
            "constructor_position_gain_summary.csv",
            ["constructor_id", "constructor_name", "records", "avg_position_change", "positive_change_rate", "large_gain_count", "large_loss_count"],
            constructor_position_gain_summary,
        ),
        (
            "reliability_by_year.csv",
            ["season", "records", "classified_rate", "incident_rate", "mechanical_or_other_rate"],
            reliability_by_year,
        ),
        (
            "driver_reliability_summary.csv",
            ["driver_id", "driver_name", "records", "classified_rate", "incident_rate", "mechanical_or_other_rate", "avg_points"],
            driver_reliability_summary,
        ),
        (
            "constructor_reliability_summary.csv",
            ["constructor_id", "constructor_name", "records", "classified_rate", "incident_rate", "mechanical_or_other_rate", "avg_points"],
            constructor_reliability_summary,
        ),
        (
            "teammate_comparison.csv",
            ["season", "constructor_id", "constructor_name", "driver_id", "driver_name", "teammate_driver_id", "teammate_driver_name", "shared_races", "finish_ahead_rate", "points", "teammate_points", "points_share", "avg_finish_position", "teammate_avg_finish_position"],
            teammate_comparison,
        ),
        (
            "circuit_volatility_index.csv",
            ["circuit_id", "circuit_name", "country", "race_count", "avg_abs_position_change", "large_gain_rate", "large_loss_rate", "non_front_row_winner_rate", "volatility_index"],
            circuit_volatility_index,
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
        "analysis_season_range": "2019-2025",
        "analysis_record_count": len(analysis_rows),
        "all_available_record_count": len(rows),
        "current_2026_completed_races": len({row["round"] for row in completed_2026_results}),
        "current_2026_remaining_races": len(remaining_2026_schedule),
    }
    write_json(SUMMARY_PATH, summary)

    print(f"Analysis files saved to: {ANALYSIS_DIR}")
    print(f"Summary saved to: {SUMMARY_PATH}")
    print(f"Output files: {len(outputs)}")


if __name__ == "__main__":
    main()
