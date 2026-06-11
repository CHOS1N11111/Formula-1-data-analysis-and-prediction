"""Tune future-feature feedback weight with historical backtests.

The final 2026 predictor can feed projected future race outcomes into later
pre-race features. This script reads historical season rows, tests candidate
feedback weights on 2022-2025 seasons, and writes backtest metrics comparing
projected final standings with actual standings under the current F1 Grand Prix
points table.
"""

import csv
from collections import defaultdict
from datetime import datetime, timezone

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from predict_f1_2026_championship import (
    BASE_DIR,
    CURRENT_SEASON_ONLINE_REPEAT,
    FEATURE_MODE,
    FIGURE_DIR,
    MODEL_DIR,
    build_current_2026_state,
    build_deterministic_race_predictions,
    build_points_models,
    build_projected_history_rows,
    build_top10_models,
    build_x,
    build_binary_y,
    build_points_y,
    clone_prediction_state,
    current_rule_points_from_finish,
    format_float,
    get_training_features_path,
    predict_one_future_race,
    read_csv,
    train_top10_probability_calibration,
    to_float,
    to_int,
    update_state_after_damped_projected_race,
    update_model_figure_manifest,
    write_csv,
    write_json,
)


OUTPUT_DETAIL_PATH = MODEL_DIR / "feedback_weight_tuning_metrics.csv"
OUTPUT_SUMMARY_PATH = MODEL_DIR / "feedback_weight_tuning_summary.csv"
OUTPUT_JSON_PATH = MODEL_DIR / "feedback_weight_tuning_summary.json"
OUTPUT_FIGURE_PATH = FIGURE_DIR / "feedback_weight_tuning_2022_2025.png"

BACKTEST_SEASONS = [2022, 2023, 2024, 2025]
KNOWN_RACE_COUNT = 5
CANDIDATE_WEIGHTS = [0.0, 0.2, 0.35, 0.5, 0.75, 1.0]
TOP10_MODEL_NAME = "xgboost_classifier"
POINTS_MODEL_NAME = "mlp_regressor"
CALIBRATION_SEASONS = {
    2022: {"train_end_season": 2020, "calibration_season": 2021},
    2023: {"train_end_season": 2021, "calibration_season": 2022},
    2024: {"train_end_season": 2022, "calibration_season": 2023},
    2025: {"train_end_season": 2023, "calibration_season": 2024},
}

DETAIL_FIELDS = [
    "season",
    "known_race_count",
    "remaining_race_count",
    "feedback_weight",
    "driver_points_mae",
    "constructor_points_mae",
    "combined_points_mae",
    "driver_champion_actual",
    "driver_champion_predicted",
    "driver_champion_hit",
    "constructor_champion_actual",
    "constructor_champion_predicted",
    "constructor_champion_hit",
    "driver_top3_overlap",
    "constructor_top3_overlap",
    "calibration_train_end_season",
    "calibration_season",
]

SUMMARY_FIELDS = [
    "feedback_weight",
    "season_count",
    "avg_driver_points_mae",
    "avg_constructor_points_mae",
    "avg_combined_points_mae",
    "driver_champion_hit_rate",
    "constructor_champion_hit_rate",
    "avg_driver_top3_overlap",
    "avg_constructor_top3_overlap",
]


def train_backtest_models(rows, test_season):
    """Train models with the same current-season online strategy as 2026."""
    train_rows = [
        row for row in rows
        if 2003 <= to_int(row["season"]) < test_season
    ]
    known_rows = [
        row for row in rows
        if to_int(row["season"]) == test_season
        and to_int(row["round"]) <= KNOWN_RACE_COUNT
    ]
    train_rows = train_rows + known_rows * CURRENT_SEASON_ONLINE_REPEAT
    top10_model = build_top10_models()[TOP10_MODEL_NAME]
    points_model = build_points_models()[POINTS_MODEL_NAME]
    top10_model.fit(build_x(train_rows, FEATURE_MODE), build_binary_y(train_rows, "is_top10"))
    points_model.fit(build_x(train_rows, FEATURE_MODE), build_points_y(train_rows))
    calibration_config = CALIBRATION_SEASONS[test_season]
    calibration_map, _ = train_top10_probability_calibration(
        rows,
        TOP10_MODEL_NAME,
        calibration_config["train_end_season"],
        calibration_config["calibration_season"],
    )
    return top10_model, points_model, calibration_map, calibration_config


def build_season_state(rows, season, known_race_count):
    """Build known-season state from the first N races of a historical season."""
    known_rows = [
        row for row in rows
        if to_int(row["season"]) == season and to_int(row["round"]) <= known_race_count
    ]
    state_rows = []
    for row in known_rows:
        state_row = dict(row)
        state_row["season"] = "2026"
        state_rows.append(state_row)
    return build_current_2026_state(state_rows)


def build_driver_pool_from_state(state):
    """Return active driver metadata inferred from known-season state."""
    return sorted(state["driver_meta"].values(), key=lambda row: row["driver_id"])


def build_remaining_schedule(rows, season, known_race_count):
    """Build remaining-race schedule rows for a historical season."""
    schedule_rows = []
    seen = set()
    for row in rows:
        if to_int(row["season"]) != season or to_int(row["round"]) <= known_race_count:
            continue
        key = (row["season"], row["round"])
        if key in seen:
            continue
        seen.add(key)
        schedule_rows.append(
            {
                "season": row["season"],
                "round": row["round"],
                "race_name": row["race_name"],
                "race_date": row["race_date"],
                "race_time": row.get("race_time", ""),
                "circuit_id": row["circuit_id"],
                "circuit_name": row["circuit_name"],
                "circuit_locality": row.get("circuit_locality", ""),
                "circuit_country": row.get("circuit_country", ""),
            }
        )
    return sorted(schedule_rows, key=lambda row: to_int(row["round"]))


def predict_with_feedback_weight(schedule_rows, driver_pool, state, history_rows, top10_model, points_model, calibration_map, feedback_weight):
    """Predict remaining races using a candidate feedback weight."""
    rolling_state = clone_prediction_state(state)
    rolling_history_rows = list(history_rows)
    deterministic_rows = []
    for race in schedule_rows:
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
        deterministic_rows.extend(race_deterministic_rows)
        rolling_history_rows.extend(
            build_projected_history_rows(
                race_predictions, race_deterministic_rows, feedback_weight
            )
        )
        update_state_after_damped_projected_race(
            rolling_state, race_deterministic_rows, feedback_weight
        )
    return deterministic_rows


def actual_final_points(rows, season):
    """Calculate actual final driver and constructor points using current rules."""
    driver_points = defaultdict(float)
    constructor_points = defaultdict(float)
    driver_names = {}
    constructor_names = {}
    for row in rows:
        if to_int(row["season"]) != season:
            continue
        points = current_rule_points_from_finish(row)
        driver_points[row["driver_id"]] += points
        constructor_points[row["constructor_id"]] += points
        driver_names[row["driver_id"]] = row["driver_name"]
        constructor_names[row["constructor_id"]] = row["constructor_name"]
    return driver_points, constructor_points, driver_names, constructor_names


def projected_final_points(state, deterministic_rows):
    """Combine known state points with predicted remaining-race points."""
    driver_points = defaultdict(float, state["driver_points"])
    constructor_points = defaultdict(float, state["constructor_points"])
    for row in deterministic_rows:
        points = to_float(row["deterministic_rule_mapped_points"])
        driver_points[row["driver_id"]] += points
        constructor_points[row["constructor_id"]] += points
    return driver_points, constructor_points


def points_mae(actual_points, predicted_points):
    """Compute MAE on the union of actual and predicted entities."""
    entity_ids = sorted(set(actual_points) | set(predicted_points))
    if not entity_ids:
        return 0.0
    errors = [
        abs(actual_points.get(entity_id, 0.0) - predicted_points.get(entity_id, 0.0))
        for entity_id in entity_ids
    ]
    return float(np.mean(errors))


def champion_id(points):
    """Return the champion ID from a points dictionary."""
    return sorted(points.items(), key=lambda item: (-item[1], item[0]))[0][0]


def top_n_ids(points, n=3):
    """Return the top N IDs from a points dictionary."""
    return [
        entity_id for entity_id, _ in sorted(points.items(), key=lambda item: (-item[1], item[0]))[:n]
    ]


def build_backtest_history_rows(rows, season, known_race_count):
    """Keep historical context limited to past seasons and known races only."""
    return [
        row for row in rows
        if to_int(row["season"]) < season
        or (
            to_int(row["season"]) == season
            and to_int(row["round"]) <= known_race_count
        )
    ]


def evaluate_weight(rows, season, feedback_weight, top10_model, points_model, calibration_map, calibration_config):
    """Evaluate one feedback weight on one backtest season."""
    state = build_season_state(rows, season, KNOWN_RACE_COUNT)
    driver_pool = build_driver_pool_from_state(state)
    schedule_rows = build_remaining_schedule(rows, season, KNOWN_RACE_COUNT)
    history_rows = build_backtest_history_rows(rows, season, KNOWN_RACE_COUNT)
    deterministic_rows = predict_with_feedback_weight(
        schedule_rows,
        driver_pool,
        state,
        history_rows,
        top10_model,
        points_model,
        calibration_map,
        feedback_weight,
    )
    actual_driver, actual_constructor, driver_names, constructor_names = actual_final_points(rows, season)
    predicted_driver, predicted_constructor = projected_final_points(state, deterministic_rows)

    actual_driver_champion = champion_id(actual_driver)
    predicted_driver_champion = champion_id(predicted_driver)
    actual_constructor_champion = champion_id(actual_constructor)
    predicted_constructor_champion = champion_id(predicted_constructor)

    driver_mae = points_mae(actual_driver, predicted_driver)
    constructor_mae = points_mae(actual_constructor, predicted_constructor)
    return {
        "season": season,
        "known_race_count": KNOWN_RACE_COUNT,
        "remaining_race_count": len(schedule_rows),
        "feedback_weight": format_float(feedback_weight, digits=2),
        "driver_points_mae": format_float(driver_mae),
        "constructor_points_mae": format_float(constructor_mae),
        "combined_points_mae": format_float((driver_mae + constructor_mae) / 2),
        "driver_champion_actual": driver_names.get(actual_driver_champion, actual_driver_champion),
        "driver_champion_predicted": driver_names.get(predicted_driver_champion, predicted_driver_champion),
        "driver_champion_hit": 1 if actual_driver_champion == predicted_driver_champion else 0,
        "constructor_champion_actual": constructor_names.get(actual_constructor_champion, actual_constructor_champion),
        "constructor_champion_predicted": constructor_names.get(predicted_constructor_champion, predicted_constructor_champion),
        "constructor_champion_hit": 1 if actual_constructor_champion == predicted_constructor_champion else 0,
        "driver_top3_overlap": len(set(top_n_ids(actual_driver)) & set(top_n_ids(predicted_driver))),
        "constructor_top3_overlap": len(set(top_n_ids(actual_constructor)) & set(top_n_ids(predicted_constructor))),
        "calibration_train_end_season": calibration_config["train_end_season"],
        "calibration_season": calibration_config["calibration_season"],
    }


def summarize_metrics(detail_rows):
    """Aggregate feedback-weight metrics across backtest seasons."""
    summary_rows = []
    for weight in CANDIDATE_WEIGHTS:
        weight_text = format_float(weight, digits=2)
        rows = [row for row in detail_rows if row["feedback_weight"] == weight_text]
        summary_rows.append(
            {
                "feedback_weight": weight_text,
                "season_count": len(rows),
                "avg_driver_points_mae": format_float(np.mean([to_float(row["driver_points_mae"]) for row in rows])),
                "avg_constructor_points_mae": format_float(np.mean([to_float(row["constructor_points_mae"]) for row in rows])),
                "avg_combined_points_mae": format_float(np.mean([to_float(row["combined_points_mae"]) for row in rows])),
                "driver_champion_hit_rate": format_float(np.mean([to_int(row["driver_champion_hit"]) for row in rows])),
                "constructor_champion_hit_rate": format_float(np.mean([to_int(row["constructor_champion_hit"]) for row in rows])),
                "avg_driver_top3_overlap": format_float(np.mean([to_int(row["driver_top3_overlap"]) for row in rows])),
                "avg_constructor_top3_overlap": format_float(np.mean([to_int(row["constructor_top3_overlap"]) for row in rows])),
            }
        )
    return sorted(summary_rows, key=lambda row: to_float(row["avg_combined_points_mae"]))


def save_tuning_chart(summary_rows):
    """Save feedback-weight tuning chart for report use."""
    rows = sorted(summary_rows, key=lambda row: to_float(row["feedback_weight"]))
    weights = [to_float(row["feedback_weight"]) for row in rows]
    driver_mae = [to_float(row["avg_driver_points_mae"]) for row in rows]
    constructor_mae = [to_float(row["avg_constructor_points_mae"]) for row in rows]
    combined_mae = [to_float(row["avg_combined_points_mae"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    ax.plot(weights, driver_mae, marker="o", label="Driver points MAE")
    ax.plot(weights, constructor_mae, marker="o", label="Constructor points MAE")
    ax.plot(weights, combined_mae, marker="o", linewidth=2.5, label="Combined MAE")
    ax.set_xlabel("Future feature feedback weight")
    ax.set_ylabel("Mean absolute error")
    ax.set_title("Feedback Weight Backtest, 2022-2025")
    ax.grid(True, color="#D1D5DB", linewidth=0.8, alpha=0.9)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_FIGURE_PATH, dpi=180)
    plt.close(fig)


def main():
    """Run feedback-weight tuning backtests and write outputs."""
    rows = read_csv(get_training_features_path())
    detail_rows = []
    for season in BACKTEST_SEASONS:
        top10_model, points_model, calibration_map, calibration_config = train_backtest_models(rows, season)
        for weight in CANDIDATE_WEIGHTS:
            print(f"Backtesting season={season}, feedback_weight={weight}")
            detail_rows.append(
                evaluate_weight(
                    rows,
                    season,
                    weight,
                    top10_model,
                    points_model,
                    calibration_map,
                    calibration_config,
                )
            )

    summary_rows = summarize_metrics(detail_rows)
    best_row = summary_rows[0]
    write_csv(OUTPUT_DETAIL_PATH, DETAIL_FIELDS, detail_rows)
    write_csv(OUTPUT_SUMMARY_PATH, SUMMARY_FIELDS, summary_rows)
    save_tuning_chart(summary_rows)
    update_model_figure_manifest(
        [OUTPUT_DETAIL_PATH, OUTPUT_SUMMARY_PATH, OUTPUT_JSON_PATH],
        [OUTPUT_FIGURE_PATH],
    )
    write_json(
        OUTPUT_JSON_PATH,
        {
            "built_at": datetime.now(timezone.utc).isoformat(),
            "backtest_seasons": BACKTEST_SEASONS,
            "known_race_count": KNOWN_RACE_COUNT,
            "candidate_weights": CANDIDATE_WEIGHTS,
            "top10_model": TOP10_MODEL_NAME,
            "points_model": POINTS_MODEL_NAME,
            "current_season_online_repeat": CURRENT_SEASON_ONLINE_REPEAT,
            "top10_calibration_method": "Empirical decile mapping using the season immediately before each backtest season.",
            "best_feedback_weight": best_row["feedback_weight"],
            "selection_metric": "avg_combined_points_mae",
            "best_row": best_row,
            "outputs": [
                str(OUTPUT_DETAIL_PATH.relative_to(BASE_DIR)),
                str(OUTPUT_SUMMARY_PATH.relative_to(BASE_DIR)),
                str(OUTPUT_JSON_PATH.relative_to(BASE_DIR)),
                str(OUTPUT_FIGURE_PATH.relative_to(BASE_DIR)),
            ],
            "notes": [
                "Lower combined MAE is preferred.",
                "If several weights are close, use the lower weight to reduce positive-feedback amplification.",
                "Model training repeats the known current-season rows using the same online-repeat setting as the final 2026 predictor.",
                "Backtest features use only past seasons plus the first known races of the test season, avoiding future same-season leakage.",
            ],
        },
    )
    print(f"Best feedback weight: {best_row['feedback_weight']}")
    print(f"Summary saved to: {OUTPUT_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
