"""Create leakage-safe pre-race features for each driver-race record."""

import csv
import json
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "data" / "processed" / "f1_model_dataset.csv"
OUTPUT_PATH = BASE_DIR / "data" / "processed" / "f1_features.csv"
SUMMARY_PATH = BASE_DIR / "data" / "processed" / "feature_summary.json"

ROLLING_WINDOW = 3
DEFAULT_AVG_FINISH_POSITION = 20.0

FEATURE_FIELDS = [
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
    "driver_has_history",
    "constructor_has_history",
    "missing_qualifying",
    "grid_is_zero",
]


def read_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        return list(reader), reader.fieldnames


def write_rows(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def format_float(value):
    return f"{value:.4f}"


def average(values, default=0.0):
    if not values:
        return default
    return sum(values) / len(values)


def rank_descending(score_by_id):
    """Return competition ranks for higher scores."""
    sorted_items = sorted(score_by_id.items(), key=lambda item: (-item[1], item[0]))
    ranks = {}
    previous_score = None
    previous_rank = 0

    for index, (item_id, score) in enumerate(sorted_items, start=1):
        if previous_score is None or score != previous_score:
            previous_rank = index
        ranks[item_id] = previous_rank
        previous_score = score

    return ranks


def group_by_race(rows):
    races = defaultdict(list)
    for row in rows:
        key = (to_int(row["season"]), to_int(row["round"]))
        races[key].append(row)
    return dict(sorted(races.items()))


def build_features(rows):
    """Add standings, recent-form, and data-quality features in race order."""
    rows_by_race = group_by_race(rows)
    enriched_rows = []

    driver_points_by_season = defaultdict(float)
    constructor_points_by_season = defaultdict(float)
    driver_history = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW))
    constructor_history = defaultdict(lambda: deque(maxlen=ROLLING_WINDOW))

    rows_by_year = defaultdict(int)
    missing_qualifying_count = 0
    grid_zero_count = 0

    for (season, round_number), race_rows in rows_by_race.items():
        driver_ids = sorted({row["driver_id"] for row in race_rows})
        constructor_ids = sorted({row["constructor_id"] for row in race_rows})

        driver_pre_points = {
            driver_id: driver_points_by_season[(season, driver_id)]
            for driver_id in driver_ids
        }
        constructor_pre_points = {
            constructor_id: constructor_points_by_season[(season, constructor_id)]
            for constructor_id in constructor_ids
        }
        driver_ranks = rank_descending(driver_pre_points)
        constructor_ranks = rank_descending(constructor_pre_points)

        race_constructor_points = defaultdict(float)
        race_constructor_has_podium = defaultdict(int)

        for row in race_rows:
            driver_id = row["driver_id"]
            constructor_id = row["constructor_id"]
            driver_hist = list(driver_history[driver_id])
            constructor_hist = list(constructor_history[constructor_id])

            missing_qualifying = 1 if row["qualifying_position"] == "" else 0
            grid_is_zero = 1 if to_int(row["grid"]) == 0 else 0
            missing_qualifying_count += missing_qualifying
            grid_zero_count += grid_is_zero

            row["driver_pre_race_points"] = format_float(driver_pre_points[driver_id])
            row["driver_pre_race_rank"] = driver_ranks[driver_id]
            row["constructor_pre_race_points"] = format_float(
                constructor_pre_points[constructor_id]
            )
            row["constructor_pre_race_rank"] = constructor_ranks[constructor_id]
            row["driver_last3_avg_points"] = format_float(
                average([item["points"] for item in driver_hist])
            )
            row["driver_last3_avg_finish_position"] = format_float(
                average(
                    [item["finish_position"] for item in driver_hist],
                    DEFAULT_AVG_FINISH_POSITION,
                )
            )
            row["driver_last3_podium_count"] = sum(item["is_podium"] for item in driver_hist)
            row["constructor_last3_avg_points"] = format_float(
                average([item["points"] for item in constructor_hist])
            )
            row["constructor_last3_podium_count"] = sum(
                item["has_podium"] for item in constructor_hist
            )
            row["driver_history_count"] = len(driver_hist)
            row["constructor_history_count"] = len(constructor_hist)
            row["driver_has_history"] = 1 if driver_hist else 0
            row["constructor_has_history"] = 1 if constructor_hist else 0
            row["missing_qualifying"] = missing_qualifying
            row["grid_is_zero"] = grid_is_zero

            enriched_rows.append(row)
            rows_by_year[season] += 1

            points = to_float(row["points"])
            is_podium = to_int(row["is_podium"])
            race_constructor_points[constructor_id] += points
            race_constructor_has_podium[constructor_id] = max(
                race_constructor_has_podium[constructor_id], is_podium
            )

        for row in race_rows:
            driver_id = row["driver_id"]
            constructor_id = row["constructor_id"]

            driver_points_by_season[(season, driver_id)] += to_float(row["points"])
            driver_history[driver_id].append(
                {
                    "points": to_float(row["points"]),
                    "finish_position": to_float(
                        row["finish_position"], DEFAULT_AVG_FINISH_POSITION
                    ),
                    "is_podium": to_int(row["is_podium"]),
                }
            )

        for constructor_id in constructor_ids:
            constructor_points_by_season[(season, constructor_id)] += (
                race_constructor_points[constructor_id]
            )
            constructor_history[constructor_id].append(
                {
                    "points": race_constructor_points[constructor_id],
                    "has_podium": race_constructor_has_podium[constructor_id],
                }
            )

    summary = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "input": str(INPUT_PATH.relative_to(BASE_DIR)),
        "output": str(OUTPUT_PATH.relative_to(BASE_DIR)),
        "rows_total": len(enriched_rows),
        "rows_by_year": dict(sorted(rows_by_year.items())),
        "rolling_window": ROLLING_WINDOW,
        "feature_fields": FEATURE_FIELDS,
        "missing_qualifying_rows": missing_qualifying_count,
        "grid_zero_rows": grid_zero_count,
    }

    return enriched_rows, summary


def main():
    rows, base_fields = read_rows(INPUT_PATH)
    enriched_rows, summary = build_features(rows)
    fieldnames = list(base_fields) + FEATURE_FIELDS

    write_rows(OUTPUT_PATH, fieldnames, enriched_rows)
    write_json(SUMMARY_PATH, summary)

    print(f"Feature dataset saved to: {OUTPUT_PATH}")
    print(f"Summary saved to: {SUMMARY_PATH}")
    print(f"Rows: {len(enriched_rows)}")


if __name__ == "__main__":
    main()
