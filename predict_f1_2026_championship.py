"""Predict 2026 Formula 1 driver and constructor championship probabilities.

This script is the final application layer of the project. It combines the
completed 2026 race results with pre-race Top 10 and race-points models, predicts
the remaining 2026 races, maps each simulated race to the current F1 Grand Prix
points table, and runs Monte Carlo simulations to estimate championship
probabilities and points uncertainty.
"""

import csv
import json
import warnings
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", message="X does not have valid feature names")

from train_f1_points_model import (
    F1_POINTS_TABLE,
    TOP10_TARGET,
    build_points_models,
    build_top10_models,
    build_x,
    build_binary_y,
    build_points_y,
    clamp_points,
)
from train_f1_podium_model import (
    BASE_DIR,
    MODEL_DIR,
    add_circuit_history_features,
    build_feature_dict,
    format_float,
    get_training_features_path,
    read_csv,
    to_float,
    to_int,
    write_csv,
    write_json,
)


DATA_DIR = BASE_DIR / "data"
ANALYSIS_DIR = DATA_DIR / "analysis"
FIGURE_DIR = BASE_DIR / "outputs" / "figures"

SCHEDULE_PATH = ANALYSIS_DIR / "remaining_2026_schedule.csv"
TOP10_METRICS_PATH = MODEL_DIR / "top10_model_metrics.csv"
POINTS_METRICS_PATH = MODEL_DIR / "points_model_metrics.csv"
MODEL_FIGURE_MANIFEST_PATH = FIGURE_DIR / "model_figure_manifest.json"

DRIVER_OUTPUT_PATH = MODEL_DIR / "season_prediction_driver_standings_2026.csv"
CONSTRUCTOR_OUTPUT_PATH = MODEL_DIR / "season_prediction_constructor_standings_2026.csv"
RACE_PREDICTION_OUTPUT_PATH = MODEL_DIR / "season_prediction_race_points_2026.csv"
SUMMARY_OUTPUT_PATH = MODEL_DIR / "season_prediction_summary_2026.csv"
SUMMARY_JSON_PATH = MODEL_DIR / "season_prediction_summary_2026.json"
DRIVER_SCENARIO_OUTPUT_PATH = MODEL_DIR / "season_prediction_driver_standings_2026_by_model.csv"
CONSTRUCTOR_SCENARIO_OUTPUT_PATH = MODEL_DIR / "season_prediction_constructor_standings_2026_by_model.csv"
RACE_SCENARIO_OUTPUT_PATH = MODEL_DIR / "season_prediction_race_points_2026_by_model.csv"
SCENARIO_SUMMARY_OUTPUT_PATH = MODEL_DIR / "season_prediction_model_scenarios_2026.csv"
SCENARIO_DIAGNOSTICS_OUTPUT_PATH = MODEL_DIR / "season_prediction_model_scenario_diagnostics_2026.csv"
TOP10_CALIBRATION_OUTPUT_PATH = MODEL_DIR / "season_prediction_top10_calibration_2026.csv"

SIMULATION_COUNT = 5000
RANDOM_SEED = 42
FEATURE_MODE = "pre_race"
MODEL_SCENARIO_COUNT = 3
FUTURE_FEATURE_FEEDBACK_WEIGHT = 0.50
CURRENT_SEASON_ONLINE_REPEAT = 1
CURRENT_SEASON_FORM_BOOST_ALPHA = 0.0
FINAL_TRAIN_START_SEASON = 2003
FINAL_TRAIN_END_SEASON = 2025
TOP10_CALIBRATION_TRAIN_END_SEASON = 2024
TOP10_CALIBRATION_SEASON = 2025
TOP10_CALIBRATION_BIN_COUNT = 10
RANKING_STRATEGY = "points_calibrated_top10_hybrid"
RANKING_POINTS_WEIGHT = 0.7
RANKING_TOP10_WEIGHT = 0.3

DRIVER_OUTPUT_FIELDS = [
    "driver_id",
    "driver_name",
    "primary_constructor",
    "current_points",
    "deterministic_projected_points",
    "deterministic_projected_rank",
    "mean_projected_points",
    "p05_projected_points",
    "p25_projected_points",
    "p50_projected_points",
    "p75_projected_points",
    "p95_projected_points",
    "mean_projected_rank",
    "champion_probability",
    "top3_probability",
    "top5_probability",
]

CONSTRUCTOR_OUTPUT_FIELDS = [
    "constructor_id",
    "constructor_name",
    "current_points",
    "deterministic_projected_points",
    "deterministic_projected_rank",
    "mean_projected_points",
    "p05_projected_points",
    "p25_projected_points",
    "p50_projected_points",
    "p75_projected_points",
    "p95_projected_points",
    "mean_projected_rank",
    "champion_probability",
    "top3_probability",
    "top5_probability",
]

RACE_PREDICTION_FIELDS = [
    "season",
    "round",
    "race_name",
    "race_date",
    "driver_id",
    "driver_name",
    "constructor_id",
    "constructor_name",
    "predicted_points",
    "top10_probability",
    "calibrated_top10_probability",
    "ranking_score",
    "deterministic_rank",
    "deterministic_rule_mapped_points",
]

SUMMARY_FIELDS = [
    "entity_type",
    "simulation_count",
    "feature_mode",
    "top10_model",
    "points_model",
    "predicted_champion",
    "predicted_champion_probability",
    "runner_up",
    "runner_up_probability",
    "champion_probability_margin",
    "deterministic_champion",
]

SCENARIO_PREFIX_FIELDS = [
    "scenario_rank",
    "top10_model_rank",
    "top10_model",
    "points_model_rank",
    "points_model",
]

SCENARIO_SUMMARY_FIELDS = [
    "scenario_rank",
    "entity_type",
    "simulation_count",
    "feature_mode",
    "top10_model_rank",
    "top10_model",
    "top10_metric_name",
    "top10_metric_value",
    "points_model_rank",
    "points_model",
    "points_metric_name",
    "points_metric_value",
    "predicted_champion",
    "predicted_champion_probability",
    "runner_up",
    "runner_up_probability",
    "champion_probability_margin",
    "deterministic_champion",
]

DRIVER_SCENARIO_FIELDS = SCENARIO_PREFIX_FIELDS + DRIVER_OUTPUT_FIELDS
CONSTRUCTOR_SCENARIO_FIELDS = SCENARIO_PREFIX_FIELDS + CONSTRUCTOR_OUTPUT_FIELDS
RACE_SCENARIO_FIELDS = SCENARIO_PREFIX_FIELDS + RACE_PREDICTION_FIELDS
SCENARIO_DIAGNOSTICS_FIELDS = [
    "scenario_rank",
    "top10_model",
    "points_model",
    "remaining_race_count",
    "deterministic_driver_winner_count",
    "deterministic_driver_winner_distribution",
    "driver_champion",
    "driver_champion_probability",
    "driver_runner_up",
    "driver_runner_up_probability",
    "driver_probability_margin",
    "constructor_champion",
    "constructor_champion_probability",
    "constructor_runner_up",
    "constructor_runner_up_probability",
    "constructor_probability_margin",
]
TOP10_CALIBRATION_FIELDS = [
    "scenario_rank",
    "top10_model",
    "calibration_train_seasons",
    "calibration_season",
    "probability_bin",
    "bin_sort",
    "records",
    "mean_raw_probability",
    "actual_top10_rate",
    "calibrated_top10_probability",
    "calibration_error",
]


def load_csv(path):
    """Read a CSV file using UTF-8 with optional BOM support."""
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def select_top_model_rows(metrics_path, metric_name, lower_is_better=False, limit=MODEL_SCENARIO_COUNT):
    """Select top-ranked pre-race model rows from an existing model metrics CSV."""
    rows = [
        row
        for row in load_csv(metrics_path)
        if row.get("feature_mode") == FEATURE_MODE
    ]
    if lower_is_better:
        ordered_rows = sorted(rows, key=lambda row: to_float(row[metric_name]))
    else:
        ordered_rows = sorted(rows, key=lambda row: to_float(row[metric_name]), reverse=True)
    return ordered_rows[:limit]


def probability_bin_index(probability):
    """Map a probability to a decile bin index."""
    clipped = max(0.0, min(0.999999, float(probability)))
    return int(clipped * TOP10_CALIBRATION_BIN_COUNT)


def probability_bin_label(bin_index):
    """Return a stable text label for a probability decile."""
    lower = bin_index / TOP10_CALIBRATION_BIN_COUNT
    upper = (bin_index + 1) / TOP10_CALIBRATION_BIN_COUNT
    return f"{lower:.1f}-{upper:.1f}"


def build_top10_calibration_map(calibration_rows, probabilities):
    """Build empirical 2025 Top 10 probability calibration by decile."""
    bin_stats = {
        bin_index: {"records": 0, "probability_sum": 0.0, "actual_top10_sum": 0.0}
        for bin_index in range(TOP10_CALIBRATION_BIN_COUNT)
    }
    for row, probability in zip(calibration_rows, probabilities):
        bin_index = probability_bin_index(probability)
        bin_stats[bin_index]["records"] += 1
        bin_stats[bin_index]["probability_sum"] += float(probability)
        bin_stats[bin_index]["actual_top10_sum"] += to_float(row[TOP10_TARGET])

    calibration_map = {}
    output_rows = []
    for bin_index in range(TOP10_CALIBRATION_BIN_COUNT):
        stats = bin_stats[bin_index]
        records = stats["records"]
        if records > 0:
            mean_raw_probability = stats["probability_sum"] / records
            actual_top10_rate = stats["actual_top10_sum"] / records
            calibration_map[bin_index] = actual_top10_rate
            output_rows.append(
                {
                    "probability_bin": probability_bin_label(bin_index),
                    "bin_sort": bin_index,
                    "records": records,
                    "mean_raw_probability": format_float(mean_raw_probability),
                    "actual_top10_rate": format_float(actual_top10_rate),
                    "calibrated_top10_probability": format_float(actual_top10_rate),
                    "calibration_error": format_float(
                        abs(mean_raw_probability - actual_top10_rate)
                    ),
                }
            )
        else:
            output_rows.append(
                {
                    "probability_bin": probability_bin_label(bin_index),
                    "bin_sort": bin_index,
                    "records": records,
                    "mean_raw_probability": "",
                    "actual_top10_rate": "",
                    "calibrated_top10_probability": "",
                    "calibration_error": "",
                }
            )
    return calibration_map, output_rows


def train_top10_probability_calibration(rows, top10_model_name, train_end_season, calibration_season):
    """Train a holdout Top 10 model and build an empirical calibration map."""
    train_rows = [
        row for row in rows
        if FINAL_TRAIN_START_SEASON <= to_int(row["season"]) <= train_end_season
    ]
    calibration_rows = [
        row for row in rows
        if to_int(row["season"]) == calibration_season
    ]
    if not train_rows or not calibration_rows:
        return {}, []

    calibration_model = build_top10_models()[top10_model_name]
    calibration_model.fit(
        build_x(train_rows, FEATURE_MODE),
        build_binary_y(train_rows, TOP10_TARGET),
    )
    probabilities = calibration_model.predict_proba(
        build_x(calibration_rows, FEATURE_MODE)
    )[:, 1]
    return build_top10_calibration_map(calibration_rows, probabilities)


def calibrate_top10_probability(probability, calibration_map):
    """Apply decile calibration, falling back to the raw probability if a bin is empty."""
    return float(calibration_map.get(probability_bin_index(probability), probability))


def add_scenario_fields(rows, scenario):
    """Attach model-scenario metadata to output rows."""
    enriched_rows = []
    for row in rows:
        enriched_rows.append(
            {
                "scenario_rank": scenario["scenario_rank"],
                "top10_model_rank": scenario["top10_model_rank"],
                "top10_model": scenario["top10_model"],
                "points_model_rank": scenario["points_model_rank"],
                "points_model": scenario["points_model"],
                **row,
            }
        )
    return enriched_rows


def rank_from_points(points_by_id):
    """Rank entities by points with deterministic ID tie-breaking."""
    ordered = sorted(points_by_id.items(), key=lambda item: (-item[1], item[0]))
    return {entity_id: index + 1 for index, (entity_id, _) in enumerate(ordered)}


def build_current_2026_state(rows):
    """Build rolling driver and constructor state from completed 2026 rows."""
    completed_rows = [
        row for row in rows
        if to_int(row["season"]) == 2026 and to_int(row.get("finish_position"), 0) > 0
    ]
    completed_rows = sorted(completed_rows, key=lambda row: (to_int(row["round"]), row["driver_id"]))
    completed_by_race = group_by_race(completed_rows)

    driver_points = defaultdict(float)
    constructor_points = defaultdict(float)
    driver_finishes = defaultdict(lambda: deque(maxlen=3))
    driver_recent_points = defaultdict(lambda: deque(maxlen=3))
    driver_recent_podiums = defaultdict(lambda: deque(maxlen=3))
    constructor_recent_points = defaultdict(lambda: deque(maxlen=3))
    constructor_recent_podiums = defaultdict(lambda: deque(maxlen=3))
    driver_history_count = defaultdict(int)
    constructor_history_count = defaultdict(int)
    driver_meta = {}
    constructor_meta = {}

    for race_rows in completed_by_race.values():
        race_constructor_points = defaultdict(float)
        race_constructor_has_podium = defaultdict(int)
        race_constructor_ids = set()

        for row in race_rows:
            driver_id = row["driver_id"]
            constructor_id = row["constructor_id"]
            points = current_rule_points_from_finish(row)
            finish_position = to_int(row["finish_position"], 20)
            is_podium = 1 if 1 <= finish_position <= 3 else 0

            driver_points[driver_id] += points
            constructor_points[constructor_id] += points
            driver_recent_points[driver_id].append(points)
            driver_finishes[driver_id].append(finish_position)
            driver_recent_podiums[driver_id].append(is_podium)
            driver_history_count[driver_id] += 1
            race_constructor_points[constructor_id] += points
            race_constructor_has_podium[constructor_id] = max(
                race_constructor_has_podium[constructor_id], is_podium
            )
            race_constructor_ids.add(constructor_id)
            driver_meta[driver_id] = {
                "driver_id": driver_id,
                "driver_code": row.get("driver_code", ""),
                "driver_name": row["driver_name"],
                "driver_nationality": row.get("driver_nationality", ""),
                "constructor_id": constructor_id,
                "constructor_name": row["constructor_name"],
                "constructor_nationality": row.get("constructor_nationality", ""),
            }
            constructor_meta[constructor_id] = {
                "constructor_id": constructor_id,
                "constructor_name": row["constructor_name"],
            }

        for constructor_id in race_constructor_ids:
            constructor_recent_points[constructor_id].append(
                race_constructor_points[constructor_id]
            )
            constructor_recent_podiums[constructor_id].append(
                race_constructor_has_podium[constructor_id]
            )
            constructor_history_count[constructor_id] += 1

    return {
        "driver_points": driver_points,
        "constructor_points": constructor_points,
        "driver_recent_points": driver_recent_points,
        "driver_finishes": driver_finishes,
        "driver_recent_podiums": driver_recent_podiums,
        "constructor_recent_points": constructor_recent_points,
        "constructor_recent_podiums": constructor_recent_podiums,
        "driver_history_count": driver_history_count,
        "constructor_history_count": constructor_history_count,
        "driver_meta": driver_meta,
        "constructor_meta": constructor_meta,
    }


def current_rule_points_from_finish(row):
    """Return official current-rules Grand Prix points from finish position."""
    finish_position = to_int(row.get("finish_position"), 0)
    if 1 <= finish_position <= len(F1_POINTS_TABLE):
        return float(F1_POINTS_TABLE[finish_position - 1])
    return 0.0


def average(values, default=0.0):
    """Return the arithmetic mean for a small rolling sequence."""
    values = list(values)
    if not values:
        return default
    return sum(values) / len(values)


def build_driver_pool(state):
    """Return the active 2026 driver pool inferred from completed 2026 rows."""
    return sorted(state["driver_meta"].values(), key=lambda row: row["driver_id"])


def clone_prediction_state(state):
    """Copy mutable season state before building future-race features."""
    return {
        "driver_points": defaultdict(float, state["driver_points"]),
        "constructor_points": defaultdict(float, state["constructor_points"]),
        "driver_recent_points": defaultdict(
            lambda: deque(maxlen=3),
            {
                key: deque(value, maxlen=3)
                for key, value in state["driver_recent_points"].items()
            },
        ),
        "driver_finishes": defaultdict(
            lambda: deque(maxlen=3),
            {
                key: deque(value, maxlen=3)
                for key, value in state["driver_finishes"].items()
            },
        ),
        "driver_recent_podiums": defaultdict(
            lambda: deque(maxlen=3),
            {
                key: deque(value, maxlen=3)
                for key, value in state["driver_recent_podiums"].items()
            },
        ),
        "constructor_recent_points": defaultdict(
            lambda: deque(maxlen=3),
            {
                key: deque(value, maxlen=3)
                for key, value in state["constructor_recent_points"].items()
            },
        ),
        "constructor_recent_podiums": defaultdict(
            lambda: deque(maxlen=3),
            {
                key: deque(value, maxlen=3)
                for key, value in state["constructor_recent_podiums"].items()
            },
        ),
        "driver_history_count": defaultdict(int, state["driver_history_count"]),
        "constructor_history_count": defaultdict(int, state["constructor_history_count"]),
        "driver_meta": state["driver_meta"],
        "constructor_meta": state["constructor_meta"],
    }


def build_future_race_rows_for_race(race, driver_pool, state):
    """Create feature rows for one remaining 2026 race from current projected state."""
    future_rows = []
    driver_ranks = rank_from_points(state["driver_points"])
    constructor_ranks = rank_from_points(state["constructor_points"])
    for driver in driver_pool:
        driver_id = driver["driver_id"]
        constructor_id = driver["constructor_id"]
        row = {
            "season": race["season"],
            "round": race["round"],
            "race_name": race["race_name"],
            "race_date": race["race_date"],
            "race_time": race.get("race_time", ""),
            "circuit_id": race["circuit_id"],
            "circuit_name": race["circuit_name"],
            "circuit_locality": race.get("circuit_locality", ""),
            "circuit_country": race.get("circuit_country", ""),
            "driver_id": driver_id,
            "driver_code": driver.get("driver_code", ""),
            "driver_name": driver["driver_name"],
            "driver_nationality": driver.get("driver_nationality", ""),
            "constructor_id": constructor_id,
            "constructor_name": driver["constructor_name"],
            "constructor_nationality": driver.get("constructor_nationality", ""),
            "grid": "0",
            "qualifying_position": "0",
            "finish_position": "0",
            "points": "0.0000",
            "laps": "0",
            "status": "scheduled",
            "is_podium": "0",
            "is_top10": "0",
            "data_source": "2026 projection",
            "driver_pre_race_points": format_float(state["driver_points"][driver_id]),
            "driver_pre_race_rank": driver_ranks.get(driver_id, len(driver_ranks) + 1),
            "constructor_pre_race_points": format_float(
                state["constructor_points"][constructor_id]
            ),
            "constructor_pre_race_rank": constructor_ranks.get(
                constructor_id, len(constructor_ranks) + 1
            ),
            "driver_last3_avg_points": format_float(
                average(state["driver_recent_points"][driver_id])
            ),
            "driver_last3_avg_finish_position": format_float(
                average(state["driver_finishes"][driver_id], default=20.0)
            ),
            "driver_last3_podium_count": sum(state["driver_recent_podiums"][driver_id]),
            "constructor_last3_avg_points": format_float(
                average(state["constructor_recent_points"][constructor_id])
            ),
            "constructor_last3_podium_count": sum(
                state["constructor_recent_podiums"][constructor_id]
            ),
            "driver_history_count": state["driver_history_count"][driver_id],
            "constructor_history_count": state["constructor_history_count"][constructor_id],
            "driver_has_history": 1 if state["driver_history_count"][driver_id] > 0 else 0,
            "constructor_has_history": (
                1 if state["constructor_history_count"][constructor_id] > 0 else 0
            ),
            "missing_qualifying": 1,
            "grid_is_zero": 1,
        }
        future_rows.append(row)
    return future_rows


def train_selected_models(rows, top10_model_name, points_model_name):
    """Train the selected pre-race Top 10 and points models on seasons through 2025."""
    train_rows = [
        row for row in rows
        if FINAL_TRAIN_START_SEASON <= to_int(row["season"]) <= FINAL_TRAIN_END_SEASON
    ]
    current_season_rows = [
        row for row in rows
        if to_int(row["season"]) == 2026 and to_int(row.get("finish_position"), 0) > 0
    ]
    train_rows = train_rows + current_season_rows * CURRENT_SEASON_ONLINE_REPEAT
    calibration_map, calibration_rows = train_top10_probability_calibration(
        rows,
        top10_model_name,
        TOP10_CALIBRATION_TRAIN_END_SEASON,
        TOP10_CALIBRATION_SEASON,
    )
    top10_models = build_top10_models()
    points_models = build_points_models()
    top10_model = top10_models[top10_model_name]
    points_model = points_models[points_model_name]

    top10_model.fit(build_x(train_rows, FEATURE_MODE), build_binary_y(train_rows, TOP10_TARGET))
    points_model.fit(build_x(train_rows, FEATURE_MODE), build_points_y(train_rows))
    return top10_model, points_model, calibration_map, calibration_rows


def predict_future_races(future_rows, top10_model, points_model, calibration_map):
    """Predict Top 10 probabilities and expected points for future 2026 race rows."""
    x_rows = [build_feature_dict(row, FEATURE_MODE) for row in future_rows]
    top10_probabilities = top10_model.predict_proba(x_rows)[:, 1]
    predicted_points = [clamp_points(value) for value in points_model.predict(x_rows)]
    max_points = max(predicted_points) if predicted_points else 1.0

    output_rows = []
    for row, top10_probability, predicted_point in zip(
        future_rows, top10_probabilities, predicted_points
    ):
        calibrated_probability = calibrate_top10_probability(
            top10_probability, calibration_map
        )
        normalized_points = predicted_point / max_points if max_points else 0.0
        ranking_score = (
            RANKING_POINTS_WEIGHT * normalized_points
            + RANKING_TOP10_WEIGHT * calibrated_probability
        )
        enriched = dict(row)
        enriched["top10_probability"] = float(top10_probability)
        enriched["calibrated_top10_probability"] = float(calibrated_probability)
        enriched["predicted_points"] = float(predicted_point)
        enriched["ranking_score"] = float(ranking_score)
        output_rows.append(enriched)
    return output_rows


def predict_one_future_race(race, driver_pool, state, history_rows, top10_model, points_model, calibration_map):
    """Build and predict one future race using the current projected state."""
    raw_rows = build_future_race_rows_for_race(race, driver_pool, state)
    enriched_rows = add_circuit_history_features(history_rows + raw_rows)[len(history_rows):]
    return predict_future_races(enriched_rows, top10_model, points_model, calibration_map)


def group_by_race(rows):
    """Group prediction rows by season and round."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(to_int(row["season"]), to_int(row["round"]))].append(row)
    return dict(sorted(grouped.items()))


def build_deterministic_race_predictions(prediction_rows):
    """Assign deterministic rule-mapped points for each future race."""
    output_rows = []
    for _, race_rows in group_by_race(prediction_rows).items():
        ordered = sorted(
            race_rows,
            key=lambda row: (
                -to_float(row["ranking_score"]),
                -to_float(row["predicted_points"]),
                -to_float(row["calibrated_top10_probability"]),
                -to_float(row["top10_probability"]),
                row["driver_id"],
            ),
        )
        for index, row in enumerate(ordered):
            mapped_points = F1_POINTS_TABLE[index] if index < len(F1_POINTS_TABLE) else 0
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
                    "predicted_points": format_float(row["predicted_points"]),
                    "top10_probability": format_float(row["top10_probability"]),
                    "calibrated_top10_probability": format_float(
                        row["calibrated_top10_probability"]
                    ),
                    "ranking_score": format_float(row["ranking_score"]),
                    "deterministic_rank": index + 1,
                    "deterministic_rule_mapped_points": mapped_points,
                }
            )
    return output_rows


def update_state_after_damped_projected_race(state, race_rows, feedback_weight):
    """Apply damped projected race outcomes to future-race feature state."""
    race_constructor_points = defaultdict(float)
    race_constructor_has_podium = defaultdict(int)
    race_constructor_ids = set()
    for row in race_rows:
        driver_id = row["driver_id"]
        constructor_id = row["constructor_id"]
        full_points = to_float(row["deterministic_rule_mapped_points"])
        damped_points = full_points * feedback_weight
        rank = to_int(row["deterministic_rank"], 20)
        damped_finish = 20.0 - feedback_weight * (20.0 - rank)
        is_podium_signal = feedback_weight if 1 <= rank <= 3 else 0.0

        state["driver_points"][driver_id] += damped_points
        state["constructor_points"][constructor_id] += damped_points
        state["driver_recent_points"][driver_id].append(damped_points)
        state["driver_finishes"][driver_id].append(damped_finish)
        state["driver_recent_podiums"][driver_id].append(is_podium_signal)
        state["driver_history_count"][driver_id] += feedback_weight
        race_constructor_points[constructor_id] += damped_points
        race_constructor_has_podium[constructor_id] = max(
            race_constructor_has_podium[constructor_id], is_podium_signal
        )
        race_constructor_ids.add(constructor_id)

    for constructor_id in race_constructor_ids:
        state["constructor_recent_points"][constructor_id].append(
            race_constructor_points[constructor_id]
        )
        state["constructor_recent_podiums"][constructor_id].append(
            race_constructor_has_podium[constructor_id]
        )
        state["constructor_history_count"][constructor_id] += feedback_weight


def build_projected_history_rows(race_predictions, race_deterministic_rows, feedback_weight):
    """Create damped projected history rows for circuit-history features."""
    deterministic_by_driver = {
        row["driver_id"]: row for row in race_deterministic_rows
    }
    projected_history_rows = []
    for prediction_row in race_predictions:
        deterministic_row = deterministic_by_driver[prediction_row["driver_id"]]
        projected_row = dict(prediction_row)
        rank = to_int(deterministic_row["deterministic_rank"], 20)
        points = to_float(deterministic_row["deterministic_rule_mapped_points"])
        projected_row["finish_position"] = 20.0 - feedback_weight * (20.0 - rank)
        projected_row["points"] = points * feedback_weight
        projected_row["is_podium"] = feedback_weight if rank <= 3 else 0
        projected_row["is_top10"] = feedback_weight if rank <= 10 else 0
        projected_history_rows.append(projected_row)
    return projected_history_rows


def build_future_predictions_with_damped_feedback(schedule_rows, driver_pool, state, history_rows, top10_model, points_model, calibration_map):
    """Predict remaining races while softly feeding projected trends into later features."""
    rolling_state = clone_prediction_state(state)
    rolling_history_rows = list(history_rows)
    prediction_rows = []
    deterministic_rows = []
    for race in sorted(schedule_rows, key=lambda row: to_int(row["round"])):
        race_predictions = predict_one_future_race(
            race,
            driver_pool,
            rolling_state,
            rolling_history_rows,
            top10_model,
            points_model,
            calibration_map,
        )
        race_deterministic_rows = build_deterministic_race_predictions(race_predictions)
        prediction_rows.extend(race_predictions)
        deterministic_rows.extend(race_deterministic_rows)
        rolling_history_rows.extend(
            build_projected_history_rows(
                race_predictions,
                race_deterministic_rows,
                FUTURE_FEATURE_FEEDBACK_WEIGHT,
            )
        )
        update_state_after_damped_projected_race(
            rolling_state,
            race_deterministic_rows,
            FUTURE_FEATURE_FEEDBACK_WEIGHT,
        )
    return prediction_rows, deterministic_rows


def stochastic_race_points(race_rows, rng):
    """Sample one race result from predicted score and probability uncertainty."""
    scores = np.array([to_float(row["ranking_score"]) for row in race_rows], dtype=float)
    probabilities = np.array(
        [
            max(0.02, min(0.98, to_float(row.get("calibrated_top10_probability", row["top10_probability"]))))
            for row in race_rows
        ],
        dtype=float,
    )
    noise_scale = max(0.12, float(np.std(scores)) * 1.35)
    utilities = scores + rng.normal(0.0, noise_scale, size=len(race_rows))
    utilities += rng.normal(0.0, 0.18, size=len(race_rows)) * probabilities
    ordered_indices = list(np.argsort(utilities)[::-1])

    points_by_driver = {}
    for rank_index, row_index in enumerate(ordered_indices, start=1):
        points = F1_POINTS_TABLE[rank_index - 1] if rank_index <= len(F1_POINTS_TABLE) else 0
        points_by_driver[race_rows[row_index]["driver_id"]] = float(points)
    return points_by_driver


def build_season_random_effects(driver_ids, constructor_ids, rng):
    """Create simulation-level random form effects for drivers and constructors."""
    driver_effects = {
        driver_id: float(rng.normal(0.0, 0.11)) for driver_id in driver_ids
    }
    constructor_effects = {
        constructor_id: float(rng.normal(0.0, 0.08)) for constructor_id in constructor_ids
    }
    return driver_effects, constructor_effects


def apply_season_random_effects(race_rows, driver_effects, constructor_effects):
    """Adjust race ranking scores with simulation-level form uncertainty."""
    adjusted_rows = []
    for row in race_rows:
        adjusted_row = dict(row)
        adjusted_row["ranking_score"] = (
            to_float(row["ranking_score"])
            + driver_effects.get(row["driver_id"], 0.0)
            + constructor_effects.get(row["constructor_id"], 0.0)
        )
        adjusted_rows.append(adjusted_row)
    return adjusted_rows


def deterministic_projected_points(current_driver_points, current_constructor_points, race_rows):
    """Build deterministic projected points from current standings and ranked races."""
    driver_points = dict(current_driver_points)
    constructor_points = dict(current_constructor_points)
    for row in race_rows:
        points = to_float(row["deterministic_rule_mapped_points"])
        driver_id = row["driver_id"]
        constructor_id = row["constructor_id"]
        driver_points[driver_id] = driver_points.get(driver_id, 0.0) + points
        constructor_points[constructor_id] = constructor_points.get(constructor_id, 0.0) + points
    return driver_points, constructor_points


def run_monte_carlo(prediction_rows, current_driver_points, current_constructor_points, state):
    """Run season-level Monte Carlo simulation for remaining 2026 races."""
    rng = np.random.default_rng(RANDOM_SEED)
    race_groups = group_by_race(prediction_rows)
    driver_ids = sorted(state["driver_meta"])
    constructor_ids = sorted(state["constructor_meta"])
    driver_samples = defaultdict(list)
    constructor_samples = defaultdict(list)
    driver_rank_samples = defaultdict(list)
    constructor_rank_samples = defaultdict(list)

    for _ in range(SIMULATION_COUNT):
        driver_points = {driver_id: current_driver_points.get(driver_id, 0.0) for driver_id in driver_ids}
        constructor_points = {
            constructor_id: current_constructor_points.get(constructor_id, 0.0)
            for constructor_id in constructor_ids
        }
        driver_effects, constructor_effects = build_season_random_effects(
            driver_ids, constructor_ids, rng
        )
        for race_rows in race_groups.values():
            adjusted_race_rows = apply_season_random_effects(
                race_rows, driver_effects, constructor_effects
            )
            race_points = stochastic_race_points(adjusted_race_rows, rng)
            for row in race_rows:
                driver_id = row["driver_id"]
                constructor_id = row["constructor_id"]
                points = race_points.get(driver_id, 0.0)
                driver_points[driver_id] += points
                constructor_points[constructor_id] += points

        driver_ranks = rank_from_points(driver_points)
        constructor_ranks = rank_from_points(constructor_points)
        for driver_id in driver_ids:
            driver_samples[driver_id].append(driver_points[driver_id])
            driver_rank_samples[driver_id].append(driver_ranks[driver_id])
        for constructor_id in constructor_ids:
            constructor_samples[constructor_id].append(constructor_points[constructor_id])
            constructor_rank_samples[constructor_id].append(constructor_ranks[constructor_id])

    return driver_samples, constructor_samples, driver_rank_samples, constructor_rank_samples


def summarize_driver_predictions(samples, rank_samples, deterministic_points, state):
    """Summarize driver points distributions and championship probabilities."""
    deterministic_ranks = rank_from_points(deterministic_points)
    rows = []
    for driver_id, values in samples.items():
        ranks = rank_samples[driver_id]
        values_array = np.array(values, dtype=float)
        ranks_array = np.array(ranks, dtype=float)
        meta = state["driver_meta"][driver_id]
        rows.append(
            {
                "driver_id": driver_id,
                "driver_name": meta["driver_name"],
                "primary_constructor": meta["constructor_name"],
                "current_points": format_float(state["driver_points"][driver_id]),
                "deterministic_projected_points": format_float(deterministic_points[driver_id]),
                "deterministic_projected_rank": deterministic_ranks[driver_id],
                "mean_projected_points": format_float(float(np.mean(values_array))),
                "p05_projected_points": format_float(float(np.percentile(values_array, 5))),
                "p25_projected_points": format_float(float(np.percentile(values_array, 25))),
                "p50_projected_points": format_float(float(np.percentile(values_array, 50))),
                "p75_projected_points": format_float(float(np.percentile(values_array, 75))),
                "p95_projected_points": format_float(float(np.percentile(values_array, 95))),
                "mean_projected_rank": format_float(float(np.mean(ranks_array))),
                "champion_probability": format_float(float(np.mean(ranks_array == 1))),
                "top3_probability": format_float(float(np.mean(ranks_array <= 3))),
                "top5_probability": format_float(float(np.mean(ranks_array <= 5))),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            -to_float(row["champion_probability"]),
            -to_float(row["mean_projected_points"]),
            row["driver_id"],
        ),
    )


def summarize_constructor_predictions(samples, rank_samples, deterministic_points, state):
    """Summarize constructor points distributions and championship probabilities."""
    deterministic_ranks = rank_from_points(deterministic_points)
    rows = []
    for constructor_id, values in samples.items():
        ranks = rank_samples[constructor_id]
        values_array = np.array(values, dtype=float)
        ranks_array = np.array(ranks, dtype=float)
        meta = state["constructor_meta"][constructor_id]
        rows.append(
            {
                "constructor_id": constructor_id,
                "constructor_name": meta["constructor_name"],
                "current_points": format_float(state["constructor_points"][constructor_id]),
                "deterministic_projected_points": format_float(deterministic_points[constructor_id]),
                "deterministic_projected_rank": deterministic_ranks[constructor_id],
                "mean_projected_points": format_float(float(np.mean(values_array))),
                "p05_projected_points": format_float(float(np.percentile(values_array, 5))),
                "p25_projected_points": format_float(float(np.percentile(values_array, 25))),
                "p50_projected_points": format_float(float(np.percentile(values_array, 50))),
                "p75_projected_points": format_float(float(np.percentile(values_array, 75))),
                "p95_projected_points": format_float(float(np.percentile(values_array, 95))),
                "mean_projected_rank": format_float(float(np.mean(ranks_array))),
                "champion_probability": format_float(float(np.mean(ranks_array == 1))),
                "top3_probability": format_float(float(np.mean(ranks_array <= 3))),
                "top5_probability": format_float(float(np.mean(ranks_array <= 5))),
            }
        )
    return sorted(
        rows,
        key=lambda row: (
            -to_float(row["champion_probability"]),
            -to_float(row["mean_projected_points"]),
            row["constructor_id"],
        ),
    )


def save_champion_probability_chart(rows, label_field, probability_field, filename, title):
    """Save a horizontal bar chart of champion probabilities."""
    top_rows = [row for row in rows if to_float(row[probability_field]) > 0][:8]
    if not top_rows:
        top_rows = rows[:8]
    top_rows = list(reversed(top_rows))
    labels = [row[label_field] for row in top_rows]
    values = [to_float(row[probability_field]) for row in top_rows]

    fig, ax = plt.subplots(figsize=(10, 5.6))
    bars = ax.barh(labels, values, color="#2563EB")
    ax.set_xlim(0, max(1.0, max(values) * 1.08 if values else 1.0))
    ax.set_xlabel("Champion probability")
    ax.set_title(title)
    for bar, value in zip(bars, values):
        label = "<0.01" if 0 < value < 0.005 else f"{value:.2f}"
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            fontsize=8,
        )
    fig.tight_layout()
    output_path = FIGURE_DIR / filename
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_points_uncertainty_chart(rows, label_field, filename, title):
    """Save a points interval chart for projected final standings."""
    top_rows = rows[:10]
    labels = [row[label_field] for row in reversed(top_rows)]
    means = np.array([to_float(row["mean_projected_points"]) for row in reversed(top_rows)])
    p05 = np.array([to_float(row["p05_projected_points"]) for row in reversed(top_rows)])
    p95 = np.array([to_float(row["p95_projected_points"]) for row in reversed(top_rows)])
    current = np.array([to_float(row["current_points"]) for row in reversed(top_rows)])
    xerr = np.vstack([means - p05, p95 - means])

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.barh(labels, means, color="#60A5FA", label="Mean projected points")
    ax.errorbar(means, labels, xerr=xerr, fmt="none", ecolor="#374151", capsize=3)
    ax.scatter(current, labels, color="#DC2626", marker="D", s=24, label="Current points")
    ax.set_xlabel("Season points")
    ax.set_title(title)
    upper_limit = max(float(np.max(p95)) if len(p95) else 0.0, float(np.max(means)) if len(means) else 0.0)
    tick_max = int(np.ceil(upper_limit / 100.0) * 100)
    if tick_max >= 100:
        ax.set_xticks(np.arange(0, tick_max + 100, 100))
    ax.grid(axis="x", color="#D1D5DB", linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right")
    fig.tight_layout()
    output_path = FIGURE_DIR / filename
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_model_scenario_comparison_chart(scenario_summary_rows):
    """Save a chart comparing champion probabilities across model scenarios."""
    labels = []
    driver_values = []
    constructor_values = []
    for scenario_rank in sorted({to_int(row["scenario_rank"]) for row in scenario_summary_rows}):
        scenario_rows = [
            row for row in scenario_summary_rows
            if to_int(row["scenario_rank"]) == scenario_rank
        ]
        driver_row = next(row for row in scenario_rows if row["entity_type"] == "driver")
        constructor_row = next(row for row in scenario_rows if row["entity_type"] == "constructor")
        labels.append(
            f"S{scenario_rank}\n"
            f"{driver_row['top10_model']}\n"
            f"{driver_row['points_model']}"
        )
        driver_values.append(to_float(driver_row["predicted_champion_probability"]))
        constructor_values.append(to_float(constructor_row["predicted_champion_probability"]))

    x_positions = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5.8))
    driver_bars = ax.bar(
        x_positions - width / 2,
        driver_values,
        width,
        label="Driver champion",
        color="#2563EB",
    )
    constructor_bars = ax.bar(
        x_positions + width / 2,
        constructor_values,
        width,
        label="Constructor champion",
        color="#16A34A",
    )
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Champion probability")
    ax.set_title("2026 Champion Probability by Top-3 Model Scenario")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), borderaxespad=0)
    for bars in [driver_bars, constructor_bars]:
        for bar in bars:
            value = bar.get_height()
            label = "<0.01" if 0 < value < 0.005 else f"{value:.2f}"
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.02,
                label,
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.subplots_adjust(bottom=0.27, left=0.09, right=0.78, top=0.90)
    output_path = FIGURE_DIR / "season_prediction_model_scenarios_2026.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def update_model_figure_manifest(source_files, figure_paths):
    """Append 2026 prediction figures to the model figure manifest."""
    try:
        if MODEL_FIGURE_MANIFEST_PATH.exists():
            payload = json.loads(MODEL_FIGURE_MANIFEST_PATH.read_text(encoding="utf-8"))
        else:
            payload = {"source_files": [], "figures": []}
    except json.JSONDecodeError:
        payload = {"source_files": [], "figures": []}

    def manifest_path_text(path_like):
        """Convert path-like values to stable project-relative manifest text."""
        if isinstance(path_like, Path):
            try:
                return str(path_like.relative_to(BASE_DIR))
            except ValueError:
                return str(path_like)
        return str(path_like)

    existing_sources = list(payload.get("source_files", []))
    existing_figures = list(payload.get("figures", []))
    for source_file in source_files:
        source_text = manifest_path_text(source_file)
        if source_text not in existing_sources:
            existing_sources.append(source_text)
    for figure_path in figure_paths:
        figure_text = manifest_path_text(figure_path)
        if figure_text not in existing_figures:
            existing_figures.append(figure_text)
    payload["built_at"] = datetime.now(timezone.utc).isoformat()
    payload["source_files"] = existing_sources
    payload["figures"] = existing_figures
    write_json(MODEL_FIGURE_MANIFEST_PATH, payload)


def build_summary_rows(driver_rows, constructor_rows, scenario):
    """Build compact champion prediction summary rows."""
    driver_champion = driver_rows[0]
    driver_runner_up = driver_rows[1] if len(driver_rows) > 1 else driver_rows[0]
    constructor_champion = constructor_rows[0]
    constructor_runner_up = constructor_rows[1] if len(constructor_rows) > 1 else constructor_rows[0]
    deterministic_driver = min(driver_rows, key=lambda row: to_int(row["deterministic_projected_rank"]))
    deterministic_constructor = min(
        constructor_rows, key=lambda row: to_int(row["deterministic_projected_rank"])
    )
    return [
        {
            "entity_type": "driver",
            "simulation_count": SIMULATION_COUNT,
            "feature_mode": FEATURE_MODE,
            "top10_model": scenario["top10_model"],
            "points_model": scenario["points_model"],
            "predicted_champion": driver_champion["driver_name"],
            "predicted_champion_probability": driver_champion["champion_probability"],
            "runner_up": driver_runner_up["driver_name"],
            "runner_up_probability": driver_runner_up["champion_probability"],
            "champion_probability_margin": format_float(
                to_float(driver_champion["champion_probability"])
                - to_float(driver_runner_up["champion_probability"])
            ),
            "deterministic_champion": deterministic_driver["driver_name"],
        },
        {
            "entity_type": "constructor",
            "simulation_count": SIMULATION_COUNT,
            "feature_mode": FEATURE_MODE,
            "top10_model": scenario["top10_model"],
            "points_model": scenario["points_model"],
            "predicted_champion": constructor_champion["constructor_name"],
            "predicted_champion_probability": constructor_champion["champion_probability"],
            "runner_up": constructor_runner_up["constructor_name"],
            "runner_up_probability": constructor_runner_up["champion_probability"],
            "champion_probability_margin": format_float(
                to_float(constructor_champion["champion_probability"])
                - to_float(constructor_runner_up["champion_probability"])
            ),
            "deterministic_champion": deterministic_constructor["constructor_name"],
        },
    ]


def build_scenario_summary_rows(driver_rows, constructor_rows, scenario):
    """Build model-scenario champion summary rows with metric metadata."""
    base_rows = build_summary_rows(driver_rows, constructor_rows, scenario)
    return [
        {
            "scenario_rank": scenario["scenario_rank"],
            "top10_model_rank": scenario["top10_model_rank"],
            "top10_metric_name": "f1",
            "top10_metric_value": format_float(scenario["top10_metric_value"], digits=6),
            "points_model_rank": scenario["points_model_rank"],
            "points_metric_name": "mae",
            "points_metric_value": format_float(scenario["points_metric_value"], digits=6),
            **row,
        }
        for row in base_rows
    ]


def build_scenario_diagnostics_row(scenario, deterministic_race_rows, driver_rows, constructor_rows):
    """Build diagnostics that explain scenario-level certainty and deterministic ranking."""
    deterministic_winners = [
        row["driver_name"]
        for row in deterministic_race_rows
        if to_int(row["deterministic_rank"]) == 1
    ]
    winner_distribution = {
        name: deterministic_winners.count(name)
        for name in sorted(set(deterministic_winners))
    }
    driver_champion = driver_rows[0]
    driver_runner_up = driver_rows[1] if len(driver_rows) > 1 else driver_rows[0]
    constructor_champion = constructor_rows[0]
    constructor_runner_up = (
        constructor_rows[1] if len(constructor_rows) > 1 else constructor_rows[0]
    )
    return {
        "scenario_rank": scenario["scenario_rank"],
        "top10_model": scenario["top10_model"],
        "points_model": scenario["points_model"],
        "remaining_race_count": len(group_by_race(deterministic_race_rows)),
        "deterministic_driver_winner_count": len(winner_distribution),
        "deterministic_driver_winner_distribution": json.dumps(
            winner_distribution, ensure_ascii=False, sort_keys=True
        ),
        "driver_champion": driver_champion["driver_name"],
        "driver_champion_probability": driver_champion["champion_probability"],
        "driver_runner_up": driver_runner_up["driver_name"],
        "driver_runner_up_probability": driver_runner_up["champion_probability"],
        "driver_probability_margin": format_float(
            to_float(driver_champion["champion_probability"])
            - to_float(driver_runner_up["champion_probability"])
        ),
        "constructor_champion": constructor_champion["constructor_name"],
        "constructor_champion_probability": constructor_champion["champion_probability"],
        "constructor_runner_up": constructor_runner_up["constructor_name"],
        "constructor_runner_up_probability": constructor_runner_up["champion_probability"],
        "constructor_probability_margin": format_float(
            to_float(constructor_champion["champion_probability"])
            - to_float(constructor_runner_up["champion_probability"])
        ),
    }


def add_calibration_scenario_fields(calibration_rows, scenario):
    """Attach scenario metadata to Top 10 calibration rows."""
    return [
        {
            "scenario_rank": scenario["scenario_rank"],
            "top10_model": scenario["top10_model"],
            "calibration_train_seasons": (
                f"{FINAL_TRAIN_START_SEASON}-{TOP10_CALIBRATION_TRAIN_END_SEASON}"
            ),
            "calibration_season": TOP10_CALIBRATION_SEASON,
            **row,
        }
        for row in calibration_rows
    ]


def main():
    """Train pre-race models, simulate 2026, and write championship predictions."""
    feature_rows = add_circuit_history_features(read_csv(get_training_features_path()))
    state = build_current_2026_state(feature_rows)
    driver_pool = build_driver_pool(state)
    schedule_rows = load_csv(SCHEDULE_PATH)

    top10_metric_rows = select_top_model_rows(TOP10_METRICS_PATH, "f1")
    points_metric_rows = select_top_model_rows(
        POINTS_METRICS_PATH, "mae", lower_is_better=True
    )
    scenario_count = min(len(top10_metric_rows), len(points_metric_rows), MODEL_SCENARIO_COUNT)
    scenarios = []
    for index in range(scenario_count):
        top10_metric_row = top10_metric_rows[index]
        points_metric_row = points_metric_rows[index]
        scenarios.append(
            {
                "scenario_rank": index + 1,
                "top10_model_rank": index + 1,
                "top10_model": top10_metric_row["model"],
                "top10_metric_row": top10_metric_row,
                "top10_metric_value": to_float(top10_metric_row["f1"]),
                "points_model_rank": index + 1,
                "points_model": points_metric_row["model"],
                "points_metric_row": points_metric_row,
                "points_metric_value": to_float(points_metric_row["mae"]),
            }
        )

    current_driver_points = dict(state["driver_points"])
    current_constructor_points = dict(state["constructor_points"])
    scenario_driver_rows = []
    scenario_constructor_rows = []
    scenario_race_rows = []
    scenario_summary_rows = []
    scenario_diagnostics_rows = []
    scenario_calibration_rows = []
    scenario_results = []
    primary_result = None

    for scenario in scenarios:
        print(
            "Running scenario "
            f"{scenario['scenario_rank']}: "
            f"{scenario['top10_model']} + {scenario['points_model']}"
        )
        top10_model, points_model, calibration_map, calibration_rows = train_selected_models(
            feature_rows, scenario["top10_model"], scenario["points_model"]
        )
        prediction_rows, deterministic_race_rows = build_future_predictions_with_damped_feedback(
            schedule_rows,
            driver_pool,
            state,
            feature_rows,
            top10_model,
            points_model,
            calibration_map,
        )
        deterministic_driver_points, deterministic_constructor_points = deterministic_projected_points(
            current_driver_points, current_constructor_points, deterministic_race_rows
        )
        driver_samples, constructor_samples, driver_rank_samples, constructor_rank_samples = run_monte_carlo(
            prediction_rows, current_driver_points, current_constructor_points, state
        )
        driver_rows = summarize_driver_predictions(
            driver_samples, driver_rank_samples, deterministic_driver_points, state
        )
        constructor_rows = summarize_constructor_predictions(
            constructor_samples,
            constructor_rank_samples,
            deterministic_constructor_points,
            state,
        )
        summary_rows = build_summary_rows(driver_rows, constructor_rows, scenario)
        scenario_driver_rows.extend(add_scenario_fields(driver_rows, scenario))
        scenario_constructor_rows.extend(add_scenario_fields(constructor_rows, scenario))
        scenario_race_rows.extend(add_scenario_fields(deterministic_race_rows, scenario))
        scenario_summary_rows.extend(
            build_scenario_summary_rows(driver_rows, constructor_rows, scenario)
        )
        scenario_diagnostics_rows.append(
            build_scenario_diagnostics_row(
                scenario, deterministic_race_rows, driver_rows, constructor_rows
            )
        )
        scenario_calibration_rows.extend(
            add_calibration_scenario_fields(calibration_rows, scenario)
        )
        scenario_results.append(
            {
                "scenario": scenario,
                "driver_rows": driver_rows,
                "constructor_rows": constructor_rows,
            }
        )

        if scenario["scenario_rank"] == 1:
            primary_result = {
                "scenario": scenario,
                "driver_rows": driver_rows,
                "constructor_rows": constructor_rows,
                "deterministic_race_rows": deterministic_race_rows,
                "summary_rows": summary_rows,
            }

    if primary_result is None:
        raise RuntimeError("No model scenarios were available for prediction.")

    primary_scenario = primary_result["scenario"]
    driver_rows = primary_result["driver_rows"]
    constructor_rows = primary_result["constructor_rows"]
    deterministic_race_rows = primary_result["deterministic_race_rows"]
    summary_rows = primary_result["summary_rows"]

    write_csv(DRIVER_OUTPUT_PATH, DRIVER_OUTPUT_FIELDS, driver_rows)
    write_csv(CONSTRUCTOR_OUTPUT_PATH, CONSTRUCTOR_OUTPUT_FIELDS, constructor_rows)
    write_csv(RACE_PREDICTION_OUTPUT_PATH, RACE_PREDICTION_FIELDS, deterministic_race_rows)
    write_csv(SUMMARY_OUTPUT_PATH, SUMMARY_FIELDS, summary_rows)
    write_csv(DRIVER_SCENARIO_OUTPUT_PATH, DRIVER_SCENARIO_FIELDS, scenario_driver_rows)
    write_csv(
        CONSTRUCTOR_SCENARIO_OUTPUT_PATH,
        CONSTRUCTOR_SCENARIO_FIELDS,
        scenario_constructor_rows,
    )
    write_csv(RACE_SCENARIO_OUTPUT_PATH, RACE_SCENARIO_FIELDS, scenario_race_rows)
    write_csv(SCENARIO_SUMMARY_OUTPUT_PATH, SCENARIO_SUMMARY_FIELDS, scenario_summary_rows)
    write_csv(
        TOP10_CALIBRATION_OUTPUT_PATH,
        TOP10_CALIBRATION_FIELDS,
        scenario_calibration_rows,
    )
    write_csv(
        SCENARIO_DIAGNOSTICS_OUTPUT_PATH,
        SCENARIO_DIAGNOSTICS_FIELDS,
        scenario_diagnostics_rows,
    )

    figure_paths = [
        save_champion_probability_chart(
            driver_rows,
            "driver_name",
            "champion_probability",
            "season_prediction_driver_champion_2026.png",
            "2026 Driver Champion Probability",
        ),
        save_champion_probability_chart(
            constructor_rows,
            "constructor_name",
            "champion_probability",
            "season_prediction_constructor_champion_2026.png",
            "2026 Constructor Champion Probability",
        ),
        save_points_uncertainty_chart(
            driver_rows,
            "driver_name",
            "season_prediction_driver_points_uncertainty_2026.png",
            "2026 Driver Points Projection Uncertainty",
        ),
        save_points_uncertainty_chart(
            constructor_rows,
            "constructor_name",
            "season_prediction_constructor_points_uncertainty_2026.png",
            "2026 Constructor Points Projection Uncertainty",
        ),
        save_model_scenario_comparison_chart(scenario_summary_rows),
    ]
    for result in scenario_results:
        scenario = result["scenario"]
        scenario_rank = scenario["scenario_rank"]
        scenario_label = (
            f"Scenario {scenario_rank}: "
            f"{scenario['top10_model']} + {scenario['points_model']}"
        )
        figure_paths.extend(
            [
                save_champion_probability_chart(
                    result["driver_rows"],
                    "driver_name",
                    "champion_probability",
                    f"season_prediction_s{scenario_rank}_driver_champion_2026.png",
                    f"2026 Driver Champion Probability - {scenario_label}",
                ),
                save_champion_probability_chart(
                    result["constructor_rows"],
                    "constructor_name",
                    "champion_probability",
                    f"season_prediction_s{scenario_rank}_constructor_champion_2026.png",
                    f"2026 Constructor Champion Probability - {scenario_label}",
                ),
                save_points_uncertainty_chart(
                    result["driver_rows"],
                    "driver_name",
                    f"season_prediction_s{scenario_rank}_driver_points_uncertainty_2026.png",
                    f"2026 Driver Points Projection - {scenario_label}",
                ),
                save_points_uncertainty_chart(
                    result["constructor_rows"],
                    "constructor_name",
                    f"season_prediction_s{scenario_rank}_constructor_points_uncertainty_2026.png",
                    f"2026 Constructor Points Projection - {scenario_label}",
                ),
            ]
        )
    update_model_figure_manifest(
        [
            str(DRIVER_OUTPUT_PATH.relative_to(BASE_DIR)),
            str(CONSTRUCTOR_OUTPUT_PATH.relative_to(BASE_DIR)),
            str(RACE_PREDICTION_OUTPUT_PATH.relative_to(BASE_DIR)),
            str(SUMMARY_OUTPUT_PATH.relative_to(BASE_DIR)),
            str(DRIVER_SCENARIO_OUTPUT_PATH.relative_to(BASE_DIR)),
            str(CONSTRUCTOR_SCENARIO_OUTPUT_PATH.relative_to(BASE_DIR)),
            str(RACE_SCENARIO_OUTPUT_PATH.relative_to(BASE_DIR)),
            str(SCENARIO_SUMMARY_OUTPUT_PATH.relative_to(BASE_DIR)),
            str(TOP10_CALIBRATION_OUTPUT_PATH.relative_to(BASE_DIR)),
            str(SCENARIO_DIAGNOSTICS_OUTPUT_PATH.relative_to(BASE_DIR)),
        ],
        figure_paths,
    )

    write_json(
        SUMMARY_JSON_PATH,
        {
            "built_at": datetime.now(timezone.utc).isoformat(),
            "feature_mode": FEATURE_MODE,
            "simulation_count": SIMULATION_COUNT,
            "random_seed": RANDOM_SEED,
            "points_rule": F1_POINTS_TABLE,
            "top10_model": primary_scenario["top10_model"],
            "top10_model_2025_metric": primary_scenario["top10_metric_row"],
            "points_model": primary_scenario["points_model"],
            "points_model_2025_metric": primary_scenario["points_metric_row"],
            "ranking_strategy": RANKING_STRATEGY,
            "ranking_points_weight": RANKING_POINTS_WEIGHT,
            "ranking_top10_weight": RANKING_TOP10_WEIGHT,
            "top10_calibration": {
                "method": "Empirical decile mapping from a 2025 holdout season.",
                "train_seasons": (
                    f"{FINAL_TRAIN_START_SEASON}-{TOP10_CALIBRATION_TRAIN_END_SEASON}"
                ),
                "calibration_season": TOP10_CALIBRATION_SEASON,
                "bin_count": TOP10_CALIBRATION_BIN_COUNT,
            },
            "model_scenarios": [
                {
                    "scenario_rank": scenario["scenario_rank"],
                    "top10_model_rank": scenario["top10_model_rank"],
                    "top10_model": scenario["top10_model"],
                    "top10_f1": scenario["top10_metric_value"],
                    "points_model_rank": scenario["points_model_rank"],
                    "points_model": scenario["points_model"],
                    "points_mae": scenario["points_metric_value"],
                }
                for scenario in scenarios
            ],
            "future_feature_feedback_weight": FUTURE_FEATURE_FEEDBACK_WEIGHT,
            "current_season_online_repeat": CURRENT_SEASON_ONLINE_REPEAT,
            "current_season_form_boost_alpha": CURRENT_SEASON_FORM_BOOST_ALPHA,
            "remaining_race_count": len(schedule_rows),
            "driver_count": len(driver_pool),
            "outputs": [
                str(DRIVER_OUTPUT_PATH.relative_to(BASE_DIR)),
                str(CONSTRUCTOR_OUTPUT_PATH.relative_to(BASE_DIR)),
                str(RACE_PREDICTION_OUTPUT_PATH.relative_to(BASE_DIR)),
                str(SUMMARY_OUTPUT_PATH.relative_to(BASE_DIR)),
                str(DRIVER_SCENARIO_OUTPUT_PATH.relative_to(BASE_DIR)),
                str(CONSTRUCTOR_SCENARIO_OUTPUT_PATH.relative_to(BASE_DIR)),
                str(RACE_SCENARIO_OUTPUT_PATH.relative_to(BASE_DIR)),
                str(SCENARIO_SUMMARY_OUTPUT_PATH.relative_to(BASE_DIR)),
                str(TOP10_CALIBRATION_OUTPUT_PATH.relative_to(BASE_DIR)),
                str(SCENARIO_DIAGNOSTICS_OUTPUT_PATH.relative_to(BASE_DIR)),
            ],
            "figures": [str(path.relative_to(BASE_DIR)) for path in figure_paths],
            "notes": [
                "Completed 2026 races are converted from finishing positions to the current Grand Prix points table as the starting point.",
                "Remaining 2026 predictions use damped feedback: projected future results partially update later pre-race features rather than being written back at full weight.",
                "The feedback weight is selected by tune_f1_feedback_weight.py using 2022-2025 historical backtests and average combined driver/constructor points MAE.",
                "Completed 2026 rows are repeated once in final model training. This conservative online-training setting was selected by tune_f1_current_form_boost.py because it improves short-history driver MAE without using an unstable explicit ranking boost.",
                "Race ranking uses the 2025-best rule-mapped strategy: 70% normalized predicted points plus 30% calibrated Top 10 probability.",
                "Monte Carlo simulations sample from pre-race prediction signals generated for the remaining races.",
                "Remaining 2026 races use pre-race features only; qualifying and grid data are not used.",
                "Sprint points and fastest-lap bonus points are outside this project scope.",
                "Each simulated race maps the ranked top 10 to the current Grand Prix points table.",
                "The primary output uses the best-ranked pre-race Top 10 and points models, while by-model outputs compare the top three ranked model pairs.",
            ],
        },
    )

    print("2026 championship prediction completed.")
    print(f"Driver champion: {summary_rows[0]['predicted_champion']} ({summary_rows[0]['predicted_champion_probability']})")
    print(f"Constructor champion: {summary_rows[1]['predicted_champion']} ({summary_rows[1]['predicted_champion_probability']})")
    print(f"Outputs saved to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
