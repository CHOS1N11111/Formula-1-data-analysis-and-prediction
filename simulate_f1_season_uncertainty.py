"""Evaluate points-ranking strategies and season uncertainty simulations.

This post-modeling script reads 2025 Top 10, points, and podium prediction
outputs, compares race-level ranking strategies after mapping them to the
current F1 Grand Prix points table, and tests empirical Top 10 probability
calibration. It writes strategy metrics, calibration outputs, diagnostic
figures, and Monte Carlo season-simulation summaries used before final 2026
championship prediction.
"""

import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "data" / "modeling"
FIGURE_DIR = BASE_DIR / "outputs" / "figures"

POINTS_PREDICTIONS_PATH = MODEL_DIR / "points_predictions_2025.csv"
PODIUM_PREDICTIONS_PATH = MODEL_DIR / "podium_predictions_2025.csv"
MODEL_FIGURE_MANIFEST_PATH = FIGURE_DIR / "model_figure_manifest.json"

F1_POINTS_TABLE = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
SIMULATION_COUNT = 3000
RANDOM_SEED = 42

STRATEGIES = {
    "predicted_points": "Rank by continuous predicted race points.",
    "top10_probability": "Rank by raw Top 10 probability.",
    "calibrated_top10_probability": "Rank by calibrated Top 10 probability.",
    "points_top10_hybrid": "Rank by 70% normalized points and 30% raw Top 10 probability.",
    "points_calibrated_top10_hybrid": "Rank by 70% normalized points and 30% calibrated Top 10 probability.",
    "podium_top10_points_hybrid": "Rank by 50% normalized points, 30% podium probability, and 20% calibrated Top 10 probability.",
}

CALIBRATED_PREDICTION_FIELDS = [
    "season",
    "round",
    "race_name",
    "race_date",
    "driver_id",
    "driver_name",
    "constructor_id",
    "constructor_name",
    "top10_probability",
    "calibrated_top10_probability",
    "actual_is_top10",
]

CALIBRATION_COMPARISON_FIELDS = [
    "probability_bin",
    "bin_sort",
    "records",
    "mean_raw_probability",
    "mean_calibrated_probability",
    "actual_top10_rate",
    "raw_calibration_error",
    "calibrated_calibration_error",
]

CALIBRATION_SUMMARY_FIELDS = [
    "metric",
    "raw_probability",
    "calibrated_probability",
]

STRATEGY_PREDICTION_FIELDS = [
    "strategy",
    "strategy_description",
    "season",
    "round",
    "race_name",
    "race_date",
    "driver_id",
    "driver_name",
    "constructor_id",
    "constructor_name",
    "ranking_score",
    "predicted_rank",
    "rule_mapped_points",
    "actual_points",
    "actual_finish_position",
    "predicted_points",
    "top10_probability",
    "calibrated_top10_probability",
    "podium_probability",
]

STRATEGY_METRIC_FIELDS = [
    "strategy",
    "strategy_description",
    "records",
    "race_count",
    "mae",
    "rmse",
    "mean_top10_precision",
    "mean_top10_recall",
    "mean_top3_precision",
    "winner_hit_rate",
    "exact_top10_set_rate",
    "total_actual_points",
    "total_rule_mapped_points",
]

DRIVER_SIMULATION_FIELDS = [
    "driver_id",
    "driver_name",
    "primary_constructor",
    "actual_points",
    "actual_rank",
    "deterministic_points",
    "deterministic_rank",
    "mean_simulated_points",
    "p05_simulated_points",
    "p25_simulated_points",
    "p50_simulated_points",
    "p75_simulated_points",
    "p95_simulated_points",
    "mean_simulated_rank",
    "champion_probability",
    "top3_probability",
    "top5_probability",
]

CONSTRUCTOR_SIMULATION_FIELDS = [
    "constructor_id",
    "constructor_name",
    "actual_points",
    "actual_rank",
    "deterministic_points",
    "deterministic_rank",
    "mean_simulated_points",
    "p05_simulated_points",
    "p25_simulated_points",
    "p50_simulated_points",
    "p75_simulated_points",
    "p95_simulated_points",
    "mean_simulated_rank",
    "champion_probability",
    "top3_probability",
    "top5_probability",
]

SIMULATION_SUMMARY_FIELDS = [
    "entity_type",
    "simulation_count",
    "best_strategy",
    "predicted_champion",
    "predicted_champion_probability",
    "actual_champion",
    "actual_champion_simulated_probability",
    "deterministic_champion",
]

MODEL_TASK_SUMMARY_FIELDS = [
    "task",
    "target",
    "best_model",
    "best_feature_mode",
    "key_metric",
    "value",
    "source_file",
]


def read_csv(path):
    """Read a CSV file into a list of dictionaries."""
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        return list(csv.DictReader(csv_file))


def write_csv(path, fieldnames, rows):
    """Write rows to a CSV file using stable column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, payload):
    """Write a JSON document with readable indentation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def to_float(value, default=0.0):
    """Convert numeric strings to float while tolerating empty values."""
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value, default=0):
    """Convert numeric strings to int while tolerating empty values."""
    try:
        if value in (None, ""):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def format_float(value):
    """Format model output numbers consistently for CSV files."""
    return f"{float(value):.6f}"


def race_key(row):
    """Return the grouping key for one race."""
    return (row["season"], row["round"])


def clip_probability(value):
    """Clip a probability to a numerically stable range."""
    return min(max(float(value), 1e-6), 1.0 - 1e-6)


def probability_bin_index(probability):
    """Return the 0-9 decile index used by the Top 10 calibration table."""
    return min(9, max(0, int(math.floor(clip_probability(probability) * 10))))


def build_podium_probability_lookup():
    """Load podium probabilities keyed by season, round, and driver."""
    lookup = {}
    if not PODIUM_PREDICTIONS_PATH.exists():
        return lookup
    for row in read_csv(PODIUM_PREDICTIONS_PATH):
        key = (row["season"], row["round"], row["driver_id"])
        lookup[key] = to_float(row.get("podium_probability"))
    return lookup


def load_prediction_rows():
    """Read model predictions and add podium plus calibrated Top 10 fields.

    Calibration is estimated with a leave-one-race-out empirical decile mapping:
    when calibrating one race, that race's actual Top 10 outcomes are excluded
    from the bin statistics. This keeps the 2025 backtest from directly using a
    race's own labels to calibrate that same race.
    """
    podium_lookup = build_podium_probability_lookup()
    raw_rows = read_csv(POINTS_PREDICTIONS_PATH)

    global_bin_counts = defaultdict(lambda: {"records": 0, "actual_top10": 0.0})
    race_bin_counts = defaultdict(lambda: {"records": 0, "actual_top10": 0.0})
    global_actual_top10 = 0.0
    for row in raw_rows:
        bin_index = probability_bin_index(to_float(row["top10_probability"]))
        race_bin_key = (race_key(row), bin_index)
        actual_top10 = to_float(row["actual_is_top10"])
        global_bin_counts[bin_index]["records"] += 1
        global_bin_counts[bin_index]["actual_top10"] += actual_top10
        race_bin_counts[race_bin_key]["records"] += 1
        race_bin_counts[race_bin_key]["actual_top10"] += actual_top10
        global_actual_top10 += actual_top10

    rows = []
    for row in raw_rows:
        raw_probability = to_float(row["top10_probability"])
        bin_index = probability_bin_index(raw_probability)
        bin_counts = global_bin_counts[bin_index]
        excluded_counts = race_bin_counts[(race_key(row), bin_index)]
        heldout_records = bin_counts["records"] - excluded_counts["records"]
        heldout_actual = bin_counts["actual_top10"] - excluded_counts["actual_top10"]
        if heldout_records > 0:
            calibrated_probability = heldout_actual / heldout_records
        else:
            calibrated_probability = global_actual_top10 / max(len(raw_rows), 1)
        merged = dict(row)
        merged["podium_probability"] = podium_lookup.get(
            (row["season"], row["round"], row["driver_id"]), 0.0
        )
        merged["calibrated_top10_probability"] = calibrated_probability
        rows.append(merged)
    return rows


def group_rows_by_race(rows):
    """Group prediction rows by race."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[race_key(row)].append(row)
    return dict(grouped)


def normalized_points(row, max_predicted_points):
    """Normalize predicted points within a race for mixed ranking scores."""
    if max_predicted_points <= 0:
        return 0.0
    return to_float(row["predicted_points"]) / max_predicted_points


def strategy_score(row, strategy, max_predicted_points):
    """Compute the ranking score for a race-level points strategy."""
    points_norm = normalized_points(row, max_predicted_points)
    predicted_points = to_float(row["predicted_points"])
    top10_probability = to_float(row["top10_probability"])
    calibrated_top10_probability = to_float(row["calibrated_top10_probability"])
    podium_probability = to_float(row["podium_probability"])

    if strategy == "predicted_points":
        return predicted_points
    if strategy == "top10_probability":
        return top10_probability
    if strategy == "calibrated_top10_probability":
        return calibrated_top10_probability
    if strategy == "points_top10_hybrid":
        return 0.7 * points_norm + 0.3 * top10_probability
    if strategy == "points_calibrated_top10_hybrid":
        return 0.7 * points_norm + 0.3 * calibrated_top10_probability
    if strategy == "podium_top10_points_hybrid":
        return 0.5 * points_norm + 0.3 * podium_probability + 0.2 * calibrated_top10_probability
    raise ValueError(f"Unknown strategy: {strategy}")


def sorting_key(row):
    """Return deterministic tie-breakers for strategy rankings."""
    grid = to_int(row.get("grid"), 999)
    if grid <= 0:
        grid = 999
    return (
        to_float(row["_ranking_score"]),
        to_float(row["predicted_points"]),
        to_float(row["podium_probability"]),
        to_float(row["top10_probability"]),
        -grid,
    )


def build_strategy_prediction_rows(rows):
    """Map each strategy's race ranking to official F1 points."""
    output_rows = []
    for strategy, description in STRATEGIES.items():
        for _, race_rows in sorted(group_rows_by_race(rows).items(), key=lambda item: (to_int(item[0][0]), to_int(item[0][1]))):
            max_predicted_points = max(to_float(row["predicted_points"]) for row in race_rows)
            scored_rows = []
            for row in race_rows:
                scored = dict(row)
                scored["_ranking_score"] = strategy_score(row, strategy, max_predicted_points)
                scored_rows.append(scored)
            ranked_rows = sorted(scored_rows, key=sorting_key, reverse=True)
            for index, row in enumerate(ranked_rows, start=1):
                mapped_points = F1_POINTS_TABLE[index - 1] if index <= len(F1_POINTS_TABLE) else 0
                output_rows.append(
                    {
                        "strategy": strategy,
                        "strategy_description": description,
                        "season": row["season"],
                        "round": row["round"],
                        "race_name": row["race_name"],
                        "race_date": row["race_date"],
                        "driver_id": row["driver_id"],
                        "driver_name": row["driver_name"],
                        "constructor_id": row["constructor_id"],
                        "constructor_name": row["constructor_name"],
                        "ranking_score": format_float(row["_ranking_score"]),
                        "predicted_rank": index,
                        "rule_mapped_points": format_float(mapped_points),
                        "actual_points": format_float(to_float(row["actual_points"])),
                        "actual_finish_position": row["actual_finish_position"],
                        "predicted_points": format_float(to_float(row["predicted_points"])),
                        "top10_probability": format_float(to_float(row["top10_probability"])),
                        "calibrated_top10_probability": format_float(to_float(row["calibrated_top10_probability"])),
                        "podium_probability": format_float(to_float(row["podium_probability"])),
                    }
                )
    return output_rows


def calculate_strategy_metrics(strategy_rows):
    """Calculate error and race-level ranking metrics for each strategy."""
    rows_by_strategy = defaultdict(list)
    for row in strategy_rows:
        rows_by_strategy[row["strategy"]].append(row)

    metric_rows = []
    for strategy, rows in rows_by_strategy.items():
        actual_values = np.array([to_float(row["actual_points"]) for row in rows])
        predicted_values = np.array([to_float(row["rule_mapped_points"]) for row in rows])
        errors = predicted_values - actual_values
        race_groups = group_rows_by_race(rows)
        top10_precision_values = []
        top10_recall_values = []
        top3_precision_values = []
        winner_hit_values = []
        exact_top10_values = []

        for race_rows in race_groups.values():
            predicted_top10 = {row["driver_id"] for row in race_rows if to_float(row["rule_mapped_points"]) > 0}
            actual_top10 = {row["driver_id"] for row in race_rows if to_float(row["actual_points"]) > 0}
            top10_hits = len(predicted_top10 & actual_top10)
            top10_precision_values.append(top10_hits / max(len(predicted_top10), 1))
            top10_recall_values.append(top10_hits / max(len(actual_top10), 1))
            exact_top10_values.append(1.0 if predicted_top10 == actual_top10 else 0.0)

            predicted_top3 = {
                row["driver_id"] for row in race_rows if 1 <= to_int(row["predicted_rank"]) <= 3
            }
            actual_top3 = {
                row["driver_id"] for row in race_rows if 1 <= to_int(row["actual_finish_position"]) <= 3
            }
            top3_precision_values.append(len(predicted_top3 & actual_top3) / 3.0)

            predicted_winner = min(race_rows, key=lambda row: to_int(row["predicted_rank"]))
            actual_winner = min(race_rows, key=lambda row: to_int(row["actual_finish_position"], 999))
            winner_hit_values.append(
                1.0 if predicted_winner["driver_id"] == actual_winner["driver_id"] else 0.0
            )

        metric_rows.append(
            {
                "strategy": strategy,
                "strategy_description": STRATEGIES[strategy],
                "records": len(rows),
                "race_count": len(race_groups),
                "mae": format_float(np.mean(np.abs(errors))),
                "rmse": format_float(math.sqrt(np.mean(errors ** 2))),
                "mean_top10_precision": format_float(np.mean(top10_precision_values)),
                "mean_top10_recall": format_float(np.mean(top10_recall_values)),
                "mean_top3_precision": format_float(np.mean(top3_precision_values)),
                "winner_hit_rate": format_float(np.mean(winner_hit_values)),
                "exact_top10_set_rate": format_float(np.mean(exact_top10_values)),
                "total_actual_points": format_float(np.sum(actual_values)),
                "total_rule_mapped_points": format_float(np.sum(predicted_values)),
            }
        )
    return sorted(
        metric_rows,
        key=lambda row: (
            to_float(row["mae"]),
            to_float(row["rmse"]),
            -to_float(row["mean_top10_precision"]),
        ),
    )


def calculate_brier_score(rows, probability_field):
    """Calculate the Brier score for Top 10 probabilities."""
    values = []
    for row in rows:
        probability = to_float(row[probability_field])
        target = to_float(row["actual_is_top10"])
        values.append((probability - target) ** 2)
    return float(np.mean(values)) if values else 0.0


def build_calibration_comparison_rows(rows):
    """Compare raw and calibrated Top 10 probability bins."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[probability_bin_index(to_float(row["top10_probability"]))].append(row)

    output_rows = []
    for bin_index in range(10):
        bin_rows = grouped.get(bin_index, [])
        if not bin_rows:
            continue
        raw_mean = np.mean([to_float(row["top10_probability"]) for row in bin_rows])
        calibrated_mean = np.mean([to_float(row["calibrated_top10_probability"]) for row in bin_rows])
        actual_rate = np.mean([to_float(row["actual_is_top10"]) for row in bin_rows])
        output_rows.append(
            {
                "probability_bin": f"{bin_index / 10:.1f}-{(bin_index + 1) / 10:.1f}",
                "bin_sort": bin_index,
                "records": len(bin_rows),
                "mean_raw_probability": format_float(raw_mean),
                "mean_calibrated_probability": format_float(calibrated_mean),
                "actual_top10_rate": format_float(actual_rate),
                "raw_calibration_error": format_float(abs(raw_mean - actual_rate)),
                "calibrated_calibration_error": format_float(abs(calibrated_mean - actual_rate)),
            }
        )
    return output_rows


def calculate_probability_ece(rows, probability_field):
    """Calculate standard ECE by binning the selected probability field."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[probability_bin_index(to_float(row[probability_field]))].append(row)

    total_records = len(rows)
    if total_records == 0:
        return 0.0

    weighted_error = 0.0
    for bin_rows in grouped.values():
        mean_probability = np.mean([to_float(row[probability_field]) for row in bin_rows])
        actual_rate = np.mean([to_float(row["actual_is_top10"]) for row in bin_rows])
        weighted_error += len(bin_rows) * abs(mean_probability - actual_rate)
    return weighted_error / total_records


def build_calibration_summary_rows(rows, comparison_rows):
    """Summarize raw versus calibrated Top 10 probability quality."""
    return [
        {
            "metric": "brier_score",
            "raw_probability": format_float(calculate_brier_score(rows, "top10_probability")),
            "calibrated_probability": format_float(calculate_brier_score(rows, "calibrated_top10_probability")),
        },
        {
            "metric": "expected_calibration_error",
            "raw_probability": format_float(calculate_probability_ece(rows, "top10_probability")),
            "calibrated_probability": format_float(calculate_probability_ece(rows, "calibrated_top10_probability")),
        },
    ]


def build_calibrated_prediction_rows(rows):
    """Build per-row calibrated Top 10 probability output."""
    output_rows = []
    for row in rows:
        output_rows.append(
            {
                "season": row["season"],
                "round": row["round"],
                "race_name": row["race_name"],
                "race_date": row["race_date"],
                "driver_id": row["driver_id"],
                "driver_name": row["driver_name"],
                "constructor_id": row["constructor_id"],
                "constructor_name": row["constructor_name"],
                "top10_probability": format_float(to_float(row["top10_probability"])),
                "calibrated_top10_probability": format_float(to_float(row["calibrated_top10_probability"])),
                "actual_is_top10": row["actual_is_top10"],
            }
        )
    return output_rows


def normalize_values(values):
    """Normalize values to 0-1 for stochastic ranking scores."""
    values = np.asarray(values, dtype=float)
    value_range = float(values.max() - values.min()) if len(values) else 0.0
    if value_range <= 1e-12:
        return np.zeros_like(values)
    return (values - values.min()) / value_range


def stochastic_race_points(race_rows, strategy, rng):
    """Sample one race outcome using calibrated Top 10 probabilities."""
    max_predicted_points = max(to_float(row["predicted_points"]) for row in race_rows)
    top10_utilities = []
    ranking_scores = []
    for row in race_rows:
        probability = clip_probability(to_float(row["calibrated_top10_probability"]))
        top10_utilities.append(math.log(probability / (1.0 - probability)))
        ranking_scores.append(strategy_score(row, strategy, max_predicted_points))

    top10_utilities = np.asarray(top10_utilities, dtype=float)
    ranking_scores = normalize_values(ranking_scores)

    top10_noise = rng.gumbel(loc=0.0, scale=0.65, size=len(race_rows))
    selected_indices = np.argsort(top10_utilities + top10_noise)[-len(F1_POINTS_TABLE):]
    selected_set = set(int(index) for index in selected_indices)

    rank_noise = rng.gumbel(loc=0.0, scale=0.16, size=len(race_rows))
    selected_rank_scores = [
        (ranking_scores[index] + rank_noise[index], index)
        for index in selected_set
    ]
    selected_rank_scores.sort(reverse=True)

    points_by_index = {index: 0.0 for index in range(len(race_rows))}
    for rank_index, (_, row_index) in enumerate(selected_rank_scores, start=1):
        points_by_index[row_index] = float(F1_POINTS_TABLE[rank_index - 1])
    return points_by_index


def rank_from_points(points_by_id):
    """Return one-based rank positions from descending points."""
    sorted_items = sorted(points_by_id.items(), key=lambda item: (-item[1], item[0]))
    return {entity_id: rank for rank, (entity_id, _) in enumerate(sorted_items, start=1)}


def summarize_entity_simulation(
    entity_ids,
    entity_names,
    actual_points,
    deterministic_points,
    simulated_points,
    field_id,
):
    """Summarize simulated points, ranks, and title probabilities."""
    actual_ranks = rank_from_points(actual_points)
    deterministic_ranks = rank_from_points(deterministic_points)
    champion_indices = np.argmax(simulated_points, axis=1)
    simulated_ranks = np.zeros_like(simulated_points, dtype=float)
    for sim_index, row in enumerate(simulated_points):
        order = np.argsort(-row)
        for rank, entity_index in enumerate(order, start=1):
            simulated_ranks[sim_index, entity_index] = rank

    rows = []
    for entity_index, entity_id in enumerate(entity_ids):
        points_distribution = simulated_points[:, entity_index]
        rank_distribution = simulated_ranks[:, entity_index]
        rows.append(
            {
                field_id: entity_id,
                **entity_names[entity_id],
                "actual_points": format_float(actual_points.get(entity_id, 0.0)),
                "actual_rank": actual_ranks.get(entity_id, ""),
                "deterministic_points": format_float(deterministic_points.get(entity_id, 0.0)),
                "deterministic_rank": deterministic_ranks.get(entity_id, ""),
                "mean_simulated_points": format_float(np.mean(points_distribution)),
                "p05_simulated_points": format_float(np.percentile(points_distribution, 5)),
                "p25_simulated_points": format_float(np.percentile(points_distribution, 25)),
                "p50_simulated_points": format_float(np.percentile(points_distribution, 50)),
                "p75_simulated_points": format_float(np.percentile(points_distribution, 75)),
                "p95_simulated_points": format_float(np.percentile(points_distribution, 95)),
                "mean_simulated_rank": format_float(np.mean(rank_distribution)),
                "champion_probability": format_float(np.mean(champion_indices == entity_index)),
                "top3_probability": format_float(np.mean(rank_distribution <= 3)),
                "top5_probability": format_float(np.mean(rank_distribution <= 5)),
            }
        )
    return sorted(rows, key=lambda row: (-to_float(row["mean_simulated_points"]), row[field_id]))


def build_deterministic_points(strategy_rows, best_strategy):
    """Sum deterministic rule-mapped strategy points by driver and constructor."""
    driver_points = defaultdict(float)
    constructor_points = defaultdict(float)
    for row in strategy_rows:
        if row["strategy"] != best_strategy:
            continue
        points = to_float(row["rule_mapped_points"])
        driver_points[row["driver_id"]] += points
        constructor_points[row["constructor_id"]] += points
    return dict(driver_points), dict(constructor_points)


def simulate_season(rows, strategy_rows, best_strategy):
    """Run Monte Carlo season simulations for driver and constructor standings."""
    rng = np.random.default_rng(RANDOM_SEED)
    race_groups = group_rows_by_race(rows)

    driver_ids = sorted({row["driver_id"] for row in rows})
    constructor_ids = sorted({row["constructor_id"] for row in rows})
    driver_index = {driver_id: index for index, driver_id in enumerate(driver_ids)}
    constructor_index = {constructor_id: index for index, constructor_id in enumerate(constructor_ids)}

    driver_names = {}
    constructor_names = {}
    constructor_counter_by_driver = defaultdict(Counter)
    actual_driver_points = defaultdict(float)
    actual_constructor_points = defaultdict(float)
    for row in rows:
        driver_names[row["driver_id"]] = {"driver_name": row["driver_name"]}
        constructor_names[row["constructor_id"]] = {"constructor_name": row["constructor_name"]}
        constructor_counter_by_driver[row["driver_id"]][row["constructor_name"]] += 1
        actual_points = to_float(row["actual_points"])
        actual_driver_points[row["driver_id"]] += actual_points
        actual_constructor_points[row["constructor_id"]] += actual_points

    for driver_id, name_row in driver_names.items():
        primary_constructor = constructor_counter_by_driver[driver_id].most_common(1)[0][0]
        name_row["primary_constructor"] = primary_constructor

    deterministic_driver_points, deterministic_constructor_points = build_deterministic_points(
        strategy_rows, best_strategy
    )

    driver_simulated_points = np.zeros((SIMULATION_COUNT, len(driver_ids)), dtype=float)
    constructor_simulated_points = np.zeros((SIMULATION_COUNT, len(constructor_ids)), dtype=float)

    sorted_races = sorted(race_groups.items(), key=lambda item: (to_int(item[0][0]), to_int(item[0][1])))
    for sim_index in range(SIMULATION_COUNT):
        for _, race_rows in sorted_races:
            points_by_index = stochastic_race_points(race_rows, best_strategy, rng)
            for row_index, points in points_by_index.items():
                row = race_rows[row_index]
                driver_simulated_points[sim_index, driver_index[row["driver_id"]]] += points
                constructor_simulated_points[sim_index, constructor_index[row["constructor_id"]]] += points

    driver_rows = summarize_entity_simulation(
        driver_ids,
        driver_names,
        dict(actual_driver_points),
        deterministic_driver_points,
        driver_simulated_points,
        "driver_id",
    )
    constructor_rows = summarize_entity_simulation(
        constructor_ids,
        constructor_names,
        dict(actual_constructor_points),
        deterministic_constructor_points,
        constructor_simulated_points,
        "constructor_id",
    )
    return driver_rows, constructor_rows


def build_simulation_summary_rows(driver_rows, constructor_rows, best_strategy):
    """Build compact championship uncertainty summary rows."""
    rows = []
    for entity_type, standings, id_field, name_field in [
        ("driver", driver_rows, "driver_id", "driver_name"),
        ("constructor", constructor_rows, "constructor_id", "constructor_name"),
    ]:
        predicted_champion = max(standings, key=lambda row: to_float(row["champion_probability"]))
        actual_champion = min(standings, key=lambda row: to_int(row["actual_rank"], 999))
        deterministic_champion = min(standings, key=lambda row: to_int(row["deterministic_rank"], 999))
        rows.append(
            {
                "entity_type": entity_type,
                "simulation_count": SIMULATION_COUNT,
                "best_strategy": best_strategy,
                "predicted_champion": predicted_champion[name_field],
                "predicted_champion_probability": predicted_champion["champion_probability"],
                "actual_champion": actual_champion[name_field],
                "actual_champion_simulated_probability": actual_champion["champion_probability"],
                "deterministic_champion": deterministic_champion[name_field],
            }
        )
    return rows


def update_model_task_summary(best_strategy_row):
    """Update the compact task summary with the best points post-processing strategy."""
    summary_path = MODEL_DIR / "model_task_summary.csv"
    if not summary_path.exists():
        return

    rows = read_csv(summary_path)
    existing_points_model = "catboost_regressor"
    for row in rows:
        if row.get("task") == "points" and row.get("best_model"):
            existing_points_model = row["best_model"].split(" + ")[0]

    updated_rows = []
    points_row_written = False
    for row in rows:
        if row.get("task") == "points":
            updated_rows.append(
                {
                    "task": "points",
                    "target": "current_rule_points",
                    "best_model": f"{existing_points_model} + {best_strategy_row['strategy']}",
                    "best_feature_mode": "post_qualifying",
                    "key_metric": "strategy_rule_mapped_MAE",
                    "value": best_strategy_row["mae"],
                    "source_file": "rule_mapped_strategy_metrics_2025.csv",
                }
            )
            points_row_written = True
        else:
            updated_rows.append(row)

    if not points_row_written:
        updated_rows.append(
            {
                "task": "points",
                "target": "current_rule_points",
                "best_model": f"{existing_points_model} + {best_strategy_row['strategy']}",
                "best_feature_mode": "post_qualifying",
                "key_metric": "strategy_rule_mapped_MAE",
                "value": best_strategy_row["mae"],
                "source_file": "rule_mapped_strategy_metrics_2025.csv",
            }
        )
    write_csv(summary_path, MODEL_TASK_SUMMARY_FIELDS, updated_rows)


def save_strategy_comparison_chart(metric_rows):
    """Save a bar chart comparing rule-mapped ranking strategies."""
    labels = [row["strategy"].replace("_", "\n") for row in metric_rows]
    values = [to_float(row["mae"]) for row in metric_rows]
    colors = ["#059669" if index == 0 else "#2563EB" for index in range(len(metric_rows))]

    fig, ax = plt.subplots(figsize=(11.4, 5.8))
    bars = ax.bar(range(len(metric_rows)), values, color=colors)
    ax.set_xticks(range(len(metric_rows)), labels=labels, rotation=25, ha="right")
    ax.set_ylabel("MAE")
    ax.set_title("F1 Rule-Mapped Points Strategy Comparison, 2025")
    y_min = max(0.0, min(values) - 0.06)
    y_max = max(values) + 0.06
    ax.set_ylim(y_min, y_max)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    output_path = FIGURE_DIR / "rule_mapped_strategy_comparison_2025.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_calibration_comparison_chart(rows):
    """Save a Top 10 probability calibration chart before and after mapping."""
    labels = [row["probability_bin"] for row in rows]
    x_values = np.arange(len(rows))
    raw_values = [to_float(row["mean_raw_probability"]) for row in rows]
    calibrated_values = [to_float(row["mean_calibrated_probability"]) for row in rows]
    actual_values = [to_float(row["actual_top10_rate"]) for row in rows]

    fig, ax = plt.subplots(figsize=(9.2, 5.8))
    ax.plot(x_values, actual_values, marker="o", linewidth=2.0, color="#111827", label="Actual Top 10 rate")
    ax.plot(x_values, raw_values, marker="o", linewidth=1.8, color="#2563EB", label="Raw probability")
    ax.plot(x_values, calibrated_values, marker="o", linewidth=1.8, color="#EA580C", label="Calibrated probability")
    ax.set_xticks(x_values, labels=labels, rotation=35, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability / rate")
    ax.set_title("Top 10 Probability Calibration Applied to Simulation, 2025")
    ax.grid(True, color="#E5E7EB", linewidth=0.8)
    ax.legend()
    fig.tight_layout()
    output_path = FIGURE_DIR / "top10_calibrated_probability_curve_2025.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_uncertainty_chart(rows, label_field, filename, title):
    """Save a points uncertainty interval chart for top simulated standings."""
    top_rows = rows[:10]
    labels = [row[label_field] for row in reversed(top_rows)]
    means = np.array([to_float(row["mean_simulated_points"]) for row in reversed(top_rows)])
    lower = means - np.array([to_float(row["p05_simulated_points"]) for row in reversed(top_rows)])
    upper = np.array([to_float(row["p95_simulated_points"]) for row in reversed(top_rows)]) - means
    actual = [to_float(row["actual_points"]) for row in reversed(top_rows)]

    fig, ax = plt.subplots(figsize=(9.6, 6.2))
    y_values = np.arange(len(top_rows))
    ax.barh(y_values, means, xerr=[lower, upper], color="#2563EB", alpha=0.78, label="Simulated mean and 5-95% interval")
    ax.scatter(actual, y_values, color="#DC2626", marker="D", s=42, label="Actual points")
    ax.set_yticks(y_values, labels=labels)
    ax.set_xlabel("Season points")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    output_path = FIGURE_DIR / filename
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_champion_probability_chart(driver_rows, constructor_rows):
    """Save champion probability bars from the Monte Carlo simulation."""
    driver_top = [row for row in driver_rows if to_float(row["champion_probability"]) > 0][:8]
    constructor_top = [row for row in constructor_rows if to_float(row["champion_probability"]) > 0][:8]

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.4))
    for ax, rows, label_field, title, color in [
        (axes[0], driver_top, "driver_name", "Driver Champion Probability", "#2563EB"),
        (axes[1], constructor_top, "constructor_name", "Constructor Champion Probability", "#EA580C"),
    ]:
        labels = [row[label_field] for row in reversed(rows)]
        values = [to_float(row["champion_probability"]) for row in reversed(rows)]
        ax.barh(labels, values, color=color)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        ax.set_title(title)
        for index, value in enumerate(values):
            ax.text(value + 0.01, index, f"{value:.2f}", va="center", fontsize=8)
    fig.suptitle("Monte Carlo Championship Uncertainty, 2025 Backtest")
    fig.tight_layout()
    output_path = FIGURE_DIR / "season_simulation_champion_probabilities_2025.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def update_model_figure_manifest(source_files, figure_paths):
    """Merge the new simulation figures into the model figure manifest."""
    manifest = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source_files": [],
        "figures": [],
    }
    if MODEL_FIGURE_MANIFEST_PATH.exists():
        try:
            manifest = json.loads(MODEL_FIGURE_MANIFEST_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    existing_sources = list(manifest.get("source_files", []))
    existing_figures = list(manifest.get("figures", []))
    for source_file in source_files:
        if source_file not in existing_sources:
            existing_sources.append(source_file)
    for figure_path in figure_paths:
        relative_path = str(figure_path.relative_to(BASE_DIR))
        if relative_path not in existing_figures:
            existing_figures.append(relative_path)

    manifest["built_at"] = datetime.now(timezone.utc).isoformat()
    manifest["source_files"] = existing_sources
    manifest["figures"] = existing_figures
    write_json(MODEL_FIGURE_MANIFEST_PATH, manifest)


def main():
    """Run strategy comparison, calibration application, and season simulation."""
    rows = load_prediction_rows()
    calibrated_prediction_rows = build_calibrated_prediction_rows(rows)
    calibration_comparison_rows = build_calibration_comparison_rows(rows)
    calibration_summary_rows = build_calibration_summary_rows(rows, calibration_comparison_rows)
    strategy_prediction_rows = build_strategy_prediction_rows(rows)
    strategy_metric_rows = calculate_strategy_metrics(strategy_prediction_rows)
    best_strategy = strategy_metric_rows[0]["strategy"]
    update_model_task_summary(strategy_metric_rows[0])
    driver_simulation_rows, constructor_simulation_rows = simulate_season(
        rows, strategy_prediction_rows, best_strategy
    )
    simulation_summary_rows = build_simulation_summary_rows(
        driver_simulation_rows, constructor_simulation_rows, best_strategy
    )

    write_csv(
        MODEL_DIR / "top10_calibrated_predictions_2025.csv",
        CALIBRATED_PREDICTION_FIELDS,
        calibrated_prediction_rows,
    )
    write_csv(
        MODEL_DIR / "top10_calibrated_probability_bins_2025.csv",
        CALIBRATION_COMPARISON_FIELDS,
        calibration_comparison_rows,
    )
    write_csv(
        MODEL_DIR / "top10_calibrated_probability_summary_2025.csv",
        CALIBRATION_SUMMARY_FIELDS,
        calibration_summary_rows,
    )
    write_csv(
        MODEL_DIR / "rule_mapped_strategy_predictions_2025.csv",
        STRATEGY_PREDICTION_FIELDS,
        strategy_prediction_rows,
    )
    write_csv(
        MODEL_DIR / "rule_mapped_strategy_metrics_2025.csv",
        STRATEGY_METRIC_FIELDS,
        strategy_metric_rows,
    )
    write_csv(
        MODEL_DIR / "season_simulation_driver_standings_2025.csv",
        DRIVER_SIMULATION_FIELDS,
        driver_simulation_rows,
    )
    write_csv(
        MODEL_DIR / "season_simulation_constructor_standings_2025.csv",
        CONSTRUCTOR_SIMULATION_FIELDS,
        constructor_simulation_rows,
    )
    write_csv(
        MODEL_DIR / "season_simulation_summary_2025.csv",
        SIMULATION_SUMMARY_FIELDS,
        simulation_summary_rows,
    )

    strategy_chart = save_strategy_comparison_chart(strategy_metric_rows)
    calibration_chart = save_calibration_comparison_chart(calibration_comparison_rows)
    driver_uncertainty_chart = save_uncertainty_chart(
        driver_simulation_rows,
        "driver_name",
        "season_simulation_driver_uncertainty_2025.png",
        "Driver Points Uncertainty from Monte Carlo Simulation, 2025",
    )
    constructor_uncertainty_chart = save_uncertainty_chart(
        constructor_simulation_rows,
        "constructor_name",
        "season_simulation_constructor_uncertainty_2025.png",
        "Constructor Points Uncertainty from Monte Carlo Simulation, 2025",
    )
    champion_probability_chart = save_champion_probability_chart(
        driver_simulation_rows, constructor_simulation_rows
    )

    summary = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "input_files": [
            str(POINTS_PREDICTIONS_PATH.relative_to(BASE_DIR)),
            str(PODIUM_PREDICTIONS_PATH.relative_to(BASE_DIR)),
        ],
        "simulation_count": SIMULATION_COUNT,
        "random_seed": RANDOM_SEED,
        "best_rule_mapped_strategy": best_strategy,
        "strategy_count": len(STRATEGIES),
        "calibration_method": "Leave-one-race-out empirical decile mapping from 2025 Top 10 predictions.",
        "simulation_method": "Monte Carlo race simulation using calibrated Top 10 probabilities for scorer selection and the best rule-mapped strategy for points ranking.",
        "outputs": [
            "top10_calibrated_predictions_2025.csv",
            "top10_calibrated_probability_bins_2025.csv",
            "top10_calibrated_probability_summary_2025.csv",
            "rule_mapped_strategy_predictions_2025.csv",
            "rule_mapped_strategy_metrics_2025.csv",
            "season_simulation_driver_standings_2025.csv",
            "season_simulation_constructor_standings_2025.csv",
            "season_simulation_summary_2025.csv",
            "model_task_summary.csv",
        ],
        "figures": [
            str(strategy_chart.relative_to(BASE_DIR)),
            str(calibration_chart.relative_to(BASE_DIR)),
            str(driver_uncertainty_chart.relative_to(BASE_DIR)),
            str(constructor_uncertainty_chart.relative_to(BASE_DIR)),
            str(champion_probability_chart.relative_to(BASE_DIR)),
        ],
        "note": "This is a 2025 backtest and preparation layer for later 2026 championship prediction. It does not predict the final 2026 champions yet.",
    }
    write_json(MODEL_DIR / "season_simulation_summary_2025.json", summary)

    update_model_figure_manifest(
        [
            "data\\modeling\\rule_mapped_strategy_metrics_2025.csv",
            "data\\modeling\\top10_calibrated_probability_summary_2025.csv",
            "data\\modeling\\season_simulation_summary_2025.csv",
        ],
        [
            strategy_chart,
            calibration_chart,
            driver_uncertainty_chart,
            constructor_uncertainty_chart,
            champion_probability_chart,
        ],
    )

    print(f"Best rule-mapped strategy: {best_strategy}")
    print(f"Season uncertainty outputs saved to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
