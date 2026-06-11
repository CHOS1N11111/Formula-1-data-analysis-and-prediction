"""Backtest current-season form handling for short-history drivers.

This experiment reads historical season data and tests whether the 2026
championship predictor should adjust for drivers with limited historical samples
but strong current-season form. It compares baseline prediction, online
current-season training, form-based ranking boosts, and their combinations on
2022-2025 backtests before any rule is considered for the final predictor.
"""

from collections import defaultdict
from datetime import datetime, timezone

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from predict_f1_2026_championship import (
    BASE_DIR,
    FEATURE_MODE,
    FIGURE_DIR,
    FINAL_TRAIN_START_SEASON,
    FUTURE_FEATURE_FEEDBACK_WEIGHT,
    MODEL_DIR,
    average,
    build_binary_y,
    build_current_2026_state,
    build_deterministic_race_predictions,
    build_points_models,
    build_projected_history_rows,
    build_top10_models,
    build_x,
    clone_prediction_state,
    current_rule_points_from_finish,
    format_float,
    get_training_features_path,
    predict_one_future_race,
    rank_from_points,
    read_csv,
    to_float,
    to_int,
    train_top10_probability_calibration,
    update_model_figure_manifest,
    update_state_after_damped_projected_race,
    write_csv,
    write_json,
)


DETAIL_OUTPUT_PATH = MODEL_DIR / "current_form_boost_backtest_metrics.csv"
SUMMARY_OUTPUT_PATH = MODEL_DIR / "current_form_boost_backtest_summary.csv"
FOCUS_OUTPUT_PATH = MODEL_DIR / "current_form_boost_backtest_driver_focus.csv"
JSON_OUTPUT_PATH = MODEL_DIR / "current_form_boost_backtest_summary.json"
FIGURE_OUTPUT_PATH = FIGURE_DIR / "current_form_boost_backtest_2022_2025.png"

BACKTEST_SEASONS = [2022, 2023, 2024, 2025]
KNOWN_RACE_COUNT = 5
TOP10_MODEL_NAME = "xgboost_classifier"
POINTS_MODEL_NAME = "mlp_regressor"
ONLINE_REPEAT_OPTIONS = [0, 1, 3]
FORM_BOOST_ALPHA_OPTIONS = [0.0, 0.03, 0.06, 0.10]
SHORT_HISTORY_SAMPLE_THRESHOLD = 60
STRONG_FORM_SCORE_THRESHOLD = 0.45
FORM_RANK_CUTOFF = 10

CALIBRATION_SEASONS = {
    2022: {"train_end_season": 2020, "calibration_season": 2021},
    2023: {"train_end_season": 2021, "calibration_season": 2022},
    2024: {"train_end_season": 2022, "calibration_season": 2023},
    2025: {"train_end_season": 2023, "calibration_season": 2024},
}

DETAIL_FIELDS = [
    "experiment_id",
    "season",
    "known_race_count",
    "remaining_race_count",
    "online_repeat_count",
    "form_boost_alpha",
    "driver_points_mae",
    "constructor_points_mae",
    "combined_points_mae",
    "short_history_driver_count",
    "short_history_driver_mae",
    "short_form_driver_count",
    "short_form_driver_mae",
    "driver_champion_actual",
    "driver_champion_predicted",
    "driver_champion_hit",
    "constructor_champion_actual",
    "constructor_champion_predicted",
    "constructor_champion_hit",
    "driver_top3_overlap",
    "constructor_top3_overlap",
]

SUMMARY_FIELDS = [
    "experiment_id",
    "online_repeat_count",
    "form_boost_alpha",
    "adoption_candidate",
    "adoption_reason",
    "season_count",
    "avg_driver_points_mae",
    "avg_constructor_points_mae",
    "avg_combined_points_mae",
    "combined_mae_improvement_vs_baseline",
    "avg_short_history_driver_mae",
    "short_history_mae_improvement_vs_baseline",
    "avg_short_form_driver_mae",
    "short_form_mae_improvement_vs_baseline",
    "driver_champion_hit_rate",
    "constructor_champion_hit_rate",
    "avg_driver_top3_overlap",
    "avg_constructor_top3_overlap",
]

FOCUS_FIELDS = [
    "experiment_id",
    "season",
    "driver_id",
    "driver_name",
    "constructor_name",
    "prior_sample_count",
    "known_current_points",
    "known_current_rank",
    "current_form_score",
    "short_history_factor",
    "is_short_history_driver",
    "is_short_form_driver",
    "actual_final_points",
    "predicted_final_points",
    "absolute_error",
    "actual_final_rank",
    "predicted_final_rank",
]


def experiment_id(online_repeat_count, form_boost_alpha):
    """Create a compact ID for one backtest configuration."""
    if online_repeat_count == 0 and form_boost_alpha == 0:
        return "baseline"
    return f"online{online_repeat_count}_boost{form_boost_alpha:.2f}"


def build_known_rows(rows, season):
    """Return the first known races for one historical backtest season."""
    return [
        row for row in rows
        if to_int(row["season"]) == season and to_int(row["round"]) <= KNOWN_RACE_COUNT
    ]


def build_training_rows(rows, season, online_repeat_count):
    """Build model training rows with optional repeated current-season known rows."""
    historical_rows = [
        row for row in rows
        if FINAL_TRAIN_START_SEASON <= to_int(row["season"]) < season
    ]
    known_rows = build_known_rows(rows, season)
    return historical_rows + known_rows * online_repeat_count


def train_backtest_models(rows, season, online_repeat_count):
    """Train Top 10 and points models for one online-training setting."""
    train_rows = build_training_rows(rows, season, online_repeat_count)
    top10_model = build_top10_models()[TOP10_MODEL_NAME]
    points_model = build_points_models()[POINTS_MODEL_NAME]
    top10_model.fit(build_x(train_rows, FEATURE_MODE), build_binary_y(train_rows, "is_top10"))
    points_model.fit(build_x(train_rows, FEATURE_MODE), build_points_y_for_current_rules(train_rows))
    return top10_model, points_model


def build_points_y_for_current_rules(rows):
    """Build current-rules point labels for rows used in this experiment."""
    return [current_rule_points_from_finish(row) for row in rows]


def build_prior_sample_counts(rows, season):
    """Count each driver's available historical samples before the target season."""
    counts = defaultdict(int)
    for row in rows:
        row_season = to_int(row["season"])
        if FINAL_TRAIN_START_SEASON <= row_season < season:
            counts[row["driver_id"]] += 1
    return counts


def build_season_state(rows, season):
    """Build rolling current-season state from the first known races."""
    state_rows = []
    for row in build_known_rows(rows, season):
        state_row = dict(row)
        state_row["season"] = "2026"
        state_rows.append(state_row)
    return build_current_2026_state(state_rows)


def build_driver_pool_from_state(state):
    """Return active drivers inferred from the known races."""
    return sorted(state["driver_meta"].values(), key=lambda row: row["driver_id"])


def build_remaining_schedule(rows, season):
    """Return one schedule row per remaining race in a historical season."""
    seen = set()
    schedule_rows = []
    for row in rows:
        if to_int(row["season"]) != season or to_int(row["round"]) <= KNOWN_RACE_COUNT:
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


def build_backtest_history_rows(rows, season):
    """Keep feature history limited to past seasons and known current-season races."""
    return [
        row for row in rows
        if to_int(row["season"]) < season
        or (
            to_int(row["season"]) == season
            and to_int(row["round"]) <= KNOWN_RACE_COUNT
        )
    ]


def short_history_factor(prior_sample_count):
    """Return a 0-1 factor for drivers with limited pre-season sample history."""
    return max(
        0.0,
        min(
            1.0,
            (SHORT_HISTORY_SAMPLE_THRESHOLD - prior_sample_count)
            / SHORT_HISTORY_SAMPLE_THRESHOLD,
        ),
    )


def form_score_from_values(pre_race_rank, last3_avg_points, last3_avg_finish, last3_podium_count):
    """Score current form using only information available before the next race."""
    rank_score = max(0.0, (FORM_RANK_CUTOFF + 1 - pre_race_rank) / FORM_RANK_CUTOFF)
    points_score = min(1.0, last3_avg_points / 18.0)
    finish_score = max(0.0, min(1.0, (20.0 - last3_avg_finish) / 19.0))
    podium_score = min(1.0, last3_podium_count / 3.0)
    return (
        0.35 * rank_score
        + 0.35 * points_score
        + 0.10 * finish_score
        + 0.20 * podium_score
    )


def form_score_from_prediction_row(row):
    """Calculate form score from a future-race feature row."""
    return form_score_from_values(
        to_int(row["driver_pre_race_rank"], 99),
        to_float(row["driver_last3_avg_points"]),
        to_float(row["driver_last3_avg_finish_position"], 20.0),
        to_float(row["driver_last3_podium_count"]),
    )


def apply_current_form_boost(prediction_rows, prior_sample_counts, form_boost_alpha):
    """Boost ranking score for short-history drivers with strong current form."""
    if form_boost_alpha <= 0:
        return prediction_rows
    boosted_rows = []
    for row in prediction_rows:
        prior_count = prior_sample_counts.get(row["driver_id"], 0)
        boost = (
            form_boost_alpha
            * short_history_factor(prior_count)
            * form_score_from_prediction_row(row)
        )
        boosted_row = dict(row)
        boosted_row["ranking_score"] = to_float(row["ranking_score"]) + boost
        boosted_row["current_form_boost"] = boost
        boosted_rows.append(boosted_row)
    return boosted_rows


def predict_remaining_races(
    schedule_rows,
    driver_pool,
    state,
    history_rows,
    top10_model,
    points_model,
    calibration_map,
    prior_sample_counts,
    form_boost_alpha,
):
    """Predict remaining races with optional current-form ranking boost."""
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
        race_predictions = apply_current_form_boost(
            race_predictions, prior_sample_counts, form_boost_alpha
        )
        race_deterministic_rows = build_deterministic_race_predictions(race_predictions)
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
    return deterministic_rows


def actual_final_points(rows, season):
    """Calculate actual final driver and constructor points under current rules."""
    driver_points = defaultdict(float)
    constructor_points = defaultdict(float)
    for row in rows:
        if to_int(row["season"]) != season:
            continue
        points = current_rule_points_from_finish(row)
        driver_points[row["driver_id"]] += points
        constructor_points[row["constructor_id"]] += points
    return driver_points, constructor_points


def projected_final_points(state, deterministic_rows):
    """Combine known-season points with predicted remaining-race points."""
    driver_points = defaultdict(float, state["driver_points"])
    constructor_points = defaultdict(float, state["constructor_points"])
    for row in deterministic_rows:
        points = to_float(row["deterministic_rule_mapped_points"])
        driver_points[row["driver_id"]] += points
        constructor_points[row["constructor_id"]] += points
    return driver_points, constructor_points


def points_mae(actual_points, predicted_points, entity_ids=None):
    """Compute points MAE for either all entities or a selected subset."""
    if entity_ids is None:
        entity_ids = sorted(set(actual_points) | set(predicted_points))
    else:
        entity_ids = list(entity_ids)
    if not entity_ids:
        return None
    errors = [
        abs(actual_points.get(entity_id, 0.0) - predicted_points.get(entity_id, 0.0))
        for entity_id in entity_ids
    ]
    return float(np.mean(errors))


def champion_id(points):
    """Return the top entity ID by points."""
    return sorted(points.items(), key=lambda item: (-item[1], item[0]))[0][0]


def top_n_ids(points, n=3):
    """Return top-N entity IDs by points."""
    return [
        entity_id
        for entity_id, _ in sorted(points.items(), key=lambda item: (-item[1], item[0]))[:n]
    ]


def form_driver_sets(state, prior_sample_counts):
    """Identify short-history and short-history-plus-strong-form active drivers."""
    ranks = rank_from_points(state["driver_points"])
    short_history_driver_ids = []
    short_form_driver_ids = []
    for driver_id in state["driver_meta"]:
        prior_count = prior_sample_counts.get(driver_id, 0)
        form_score = form_score_from_values(
            ranks.get(driver_id, 99),
            average(state["driver_recent_points"][driver_id]),
            average(state["driver_finishes"][driver_id], default=20.0),
            average(state["driver_recent_podiums"][driver_id]),
        )
        if short_history_factor(prior_count) > 0:
            short_history_driver_ids.append(driver_id)
        if short_history_factor(prior_count) > 0 and form_score >= STRONG_FORM_SCORE_THRESHOLD:
            short_form_driver_ids.append(driver_id)
    return short_history_driver_ids, short_form_driver_ids


def build_focus_rows(experiment, season, state, prior_sample_counts, actual_points, predicted_points):
    """Build per-driver diagnostics for active short-history drivers."""
    actual_ranks = rank_from_points(actual_points)
    predicted_ranks = rank_from_points(predicted_points)
    known_ranks = rank_from_points(state["driver_points"])
    focus_rows = []
    for driver_id, meta in sorted(state["driver_meta"].items()):
        prior_count = prior_sample_counts.get(driver_id, 0)
        factor = short_history_factor(prior_count)
        if factor <= 0:
            continue
        form_score = form_score_from_values(
            known_ranks.get(driver_id, 99),
            average(state["driver_recent_points"][driver_id]),
            average(state["driver_finishes"][driver_id], default=20.0),
            average(state["driver_recent_podiums"][driver_id]),
        )
        is_short_form_driver = 1 if form_score >= STRONG_FORM_SCORE_THRESHOLD else 0
        focus_rows.append(
            {
                "experiment_id": experiment["experiment_id"],
                "season": season,
                "driver_id": driver_id,
                "driver_name": meta["driver_name"],
                "constructor_name": meta["constructor_name"],
                "prior_sample_count": prior_count,
                "known_current_points": format_float(state["driver_points"][driver_id]),
                "known_current_rank": known_ranks.get(driver_id, ""),
                "current_form_score": format_float(form_score),
                "short_history_factor": format_float(factor),
                "is_short_history_driver": 1,
                "is_short_form_driver": is_short_form_driver,
                "actual_final_points": format_float(actual_points.get(driver_id, 0.0)),
                "predicted_final_points": format_float(predicted_points.get(driver_id, 0.0)),
                "absolute_error": format_float(
                    abs(actual_points.get(driver_id, 0.0) - predicted_points.get(driver_id, 0.0))
                ),
                "actual_final_rank": actual_ranks.get(driver_id, ""),
                "predicted_final_rank": predicted_ranks.get(driver_id, ""),
            }
        )
    return focus_rows


def evaluate_configuration(
    rows,
    season,
    experiment,
    top10_model,
    points_model,
    calibration_map,
):
    """Evaluate one experiment configuration on one historical season."""
    state = build_season_state(rows, season)
    driver_pool = build_driver_pool_from_state(state)
    schedule_rows = build_remaining_schedule(rows, season)
    history_rows = build_backtest_history_rows(rows, season)
    prior_counts = build_prior_sample_counts(rows, season)
    deterministic_rows = predict_remaining_races(
        schedule_rows,
        driver_pool,
        state,
        history_rows,
        top10_model,
        points_model,
        calibration_map,
        prior_counts,
        experiment["form_boost_alpha"],
    )

    actual_driver_points, actual_constructor_points = actual_final_points(rows, season)
    predicted_driver_points, predicted_constructor_points = projected_final_points(
        state, deterministic_rows
    )
    short_history_driver_ids, short_form_driver_ids = form_driver_sets(state, prior_counts)
    driver_mae = points_mae(actual_driver_points, predicted_driver_points)
    constructor_mae = points_mae(actual_constructor_points, predicted_constructor_points)
    short_history_mae = points_mae(
        actual_driver_points, predicted_driver_points, short_history_driver_ids
    )
    short_form_mae = points_mae(
        actual_driver_points, predicted_driver_points, short_form_driver_ids
    )
    actual_driver_champion = champion_id(actual_driver_points)
    predicted_driver_champion = champion_id(predicted_driver_points)
    actual_constructor_champion = champion_id(actual_constructor_points)
    predicted_constructor_champion = champion_id(predicted_constructor_points)

    detail_row = {
        "experiment_id": experiment["experiment_id"],
        "season": season,
        "known_race_count": KNOWN_RACE_COUNT,
        "remaining_race_count": len(schedule_rows),
        "online_repeat_count": experiment["online_repeat_count"],
        "form_boost_alpha": format_float(experiment["form_boost_alpha"], digits=4),
        "driver_points_mae": format_float(driver_mae),
        "constructor_points_mae": format_float(constructor_mae),
        "combined_points_mae": format_float((driver_mae + constructor_mae) / 2.0),
        "short_history_driver_count": len(short_history_driver_ids),
        "short_history_driver_mae": (
            "" if short_history_mae is None else format_float(short_history_mae)
        ),
        "short_form_driver_count": len(short_form_driver_ids),
        "short_form_driver_mae": (
            "" if short_form_mae is None else format_float(short_form_mae)
        ),
        "driver_champion_actual": actual_driver_champion,
        "driver_champion_predicted": predicted_driver_champion,
        "driver_champion_hit": int(actual_driver_champion == predicted_driver_champion),
        "constructor_champion_actual": actual_constructor_champion,
        "constructor_champion_predicted": predicted_constructor_champion,
        "constructor_champion_hit": int(
            actual_constructor_champion == predicted_constructor_champion
        ),
        "driver_top3_overlap": len(
            set(top_n_ids(actual_driver_points)) & set(top_n_ids(predicted_driver_points))
        ),
        "constructor_top3_overlap": len(
            set(top_n_ids(actual_constructor_points))
            & set(top_n_ids(predicted_constructor_points))
        ),
    }
    focus_rows = build_focus_rows(
        experiment,
        season,
        state,
        prior_counts,
        actual_driver_points,
        predicted_driver_points,
    )
    return detail_row, focus_rows


def build_experiments():
    """Build all online-training and form-boost experiment configurations."""
    experiments = []
    for online_repeat_count in ONLINE_REPEAT_OPTIONS:
        for form_boost_alpha in FORM_BOOST_ALPHA_OPTIONS:
            experiments.append(
                {
                    "experiment_id": experiment_id(online_repeat_count, form_boost_alpha),
                    "online_repeat_count": online_repeat_count,
                    "form_boost_alpha": form_boost_alpha,
                }
            )
    return experiments


def safe_mean(values):
    """Average non-empty numeric values, returning None for empty lists."""
    values = [value for value in values if value is not None]
    if not values:
        return None
    return float(np.mean(values))


def numeric_or_none(row, field):
    """Convert a metric field to float, preserving blanks as None."""
    value = row.get(field, "")
    if value == "":
        return None
    return to_float(value)


def summarize_detail_rows(detail_rows):
    """Summarize all season-level backtest rows by experiment."""
    grouped = defaultdict(list)
    for row in detail_rows:
        grouped[row["experiment_id"]].append(row)

    raw_rows = []
    for experiment_id_value, rows in grouped.items():
        first = rows[0]
        raw_rows.append(
            {
                "experiment_id": experiment_id_value,
                "online_repeat_count": first["online_repeat_count"],
                "form_boost_alpha": first["form_boost_alpha"],
                "season_count": len(rows),
                "avg_driver_points_mae": safe_mean(
                    [numeric_or_none(row, "driver_points_mae") for row in rows]
                ),
                "avg_constructor_points_mae": safe_mean(
                    [numeric_or_none(row, "constructor_points_mae") for row in rows]
                ),
                "avg_combined_points_mae": safe_mean(
                    [numeric_or_none(row, "combined_points_mae") for row in rows]
                ),
                "avg_short_history_driver_mae": safe_mean(
                    [numeric_or_none(row, "short_history_driver_mae") for row in rows]
                ),
                "avg_short_form_driver_mae": safe_mean(
                    [numeric_or_none(row, "short_form_driver_mae") for row in rows]
                ),
                "driver_champion_hit_rate": safe_mean(
                    [numeric_or_none(row, "driver_champion_hit") for row in rows]
                ),
                "constructor_champion_hit_rate": safe_mean(
                    [numeric_or_none(row, "constructor_champion_hit") for row in rows]
                ),
                "avg_driver_top3_overlap": safe_mean(
                    [numeric_or_none(row, "driver_top3_overlap") for row in rows]
                ),
                "avg_constructor_top3_overlap": safe_mean(
                    [numeric_or_none(row, "constructor_top3_overlap") for row in rows]
                ),
            }
        )

    baseline = next(row for row in raw_rows if row["experiment_id"] == "baseline")
    output_rows = []
    for row in raw_rows:
        output_rows.append(
            {
                "experiment_id": row["experiment_id"],
                "online_repeat_count": row["online_repeat_count"],
                "form_boost_alpha": row["form_boost_alpha"],
                "adoption_candidate": 0,
                "adoption_reason": "",
                "season_count": row["season_count"],
                "avg_driver_points_mae": format_float(row["avg_driver_points_mae"]),
                "avg_constructor_points_mae": format_float(row["avg_constructor_points_mae"]),
                "avg_combined_points_mae": format_float(row["avg_combined_points_mae"]),
                "combined_mae_improvement_vs_baseline": format_float(
                    baseline["avg_combined_points_mae"] - row["avg_combined_points_mae"]
                ),
                "avg_short_history_driver_mae": (
                    "" if row["avg_short_history_driver_mae"] is None
                    else format_float(row["avg_short_history_driver_mae"])
                ),
                "short_history_mae_improvement_vs_baseline": (
                    "" if row["avg_short_history_driver_mae"] is None
                    or baseline["avg_short_history_driver_mae"] is None
                    else format_float(
                        baseline["avg_short_history_driver_mae"]
                        - row["avg_short_history_driver_mae"]
                    )
                ),
                "avg_short_form_driver_mae": (
                    "" if row["avg_short_form_driver_mae"] is None
                    else format_float(row["avg_short_form_driver_mae"])
                ),
                "short_form_mae_improvement_vs_baseline": (
                    "" if row["avg_short_form_driver_mae"] is None
                    or baseline["avg_short_form_driver_mae"] is None
                    else format_float(
                        baseline["avg_short_form_driver_mae"]
                        - row["avg_short_form_driver_mae"]
                    )
                ),
                "driver_champion_hit_rate": format_float(row["driver_champion_hit_rate"]),
                "constructor_champion_hit_rate": format_float(
                    row["constructor_champion_hit_rate"]
                ),
                "avg_driver_top3_overlap": format_float(row["avg_driver_top3_overlap"]),
                "avg_constructor_top3_overlap": format_float(
                    row["avg_constructor_top3_overlap"]
                ),
            }
        )
    recommendation = select_recommended_experiment(output_rows)
    for row in output_rows:
        if row["experiment_id"] == recommendation["experiment_id"]:
            row["adoption_candidate"] = 1
            row["adoption_reason"] = recommendation["adoption_reason"]
    return sorted(output_rows, key=lambda row: to_float(row["avg_combined_points_mae"]))


def select_recommended_experiment(summary_rows):
    """Choose a conservative configuration for the official 2026 predictor."""
    candidates = [
        row for row in summary_rows
        if to_int(row["online_repeat_count"]) > 0
        and to_float(row["form_boost_alpha"]) == 0.0
        and to_float(row["short_history_mae_improvement_vs_baseline"]) > 0
        and to_float(row["combined_mae_improvement_vs_baseline"]) >= -0.75
    ]
    if not candidates:
        baseline = next(row for row in summary_rows if row["experiment_id"] == "baseline")
        return {
            **baseline,
            "adoption_reason": "No non-baseline option met the conservative adoption constraints.",
        }
    best = sorted(
        candidates,
        key=lambda row: (
            -to_float(row["short_history_mae_improvement_vs_baseline"]),
            -to_float(row["combined_mae_improvement_vs_baseline"]),
        ),
    )[0]
    return {
        **best,
        "adoption_reason": (
            "Conservative choice: improves short-history driver MAE, keeps overall "
            "combined MAE within 0.75 points of baseline, and avoids explicit "
            "ranking-score boost because boost variants were unstable on short-form samples."
        ),
    }


def save_summary_figure(summary_rows):
    """Save comparison chart for overall and short-form backtest MAE."""
    ordered_rows = list(reversed(summary_rows))
    labels = [row["experiment_id"] for row in ordered_rows]
    combined_values = [to_float(row["avg_combined_points_mae"]) for row in ordered_rows]
    short_form_values = [
        np.nan if row["avg_short_form_driver_mae"] == ""
        else to_float(row["avg_short_form_driver_mae"])
        for row in ordered_rows
    ]
    baseline_combined = next(
        to_float(row["avg_combined_points_mae"])
        for row in summary_rows
        if row["experiment_id"] == "baseline"
    )
    baseline_short_form = next(
        row["avg_short_form_driver_mae"]
        for row in summary_rows
        if row["experiment_id"] == "baseline"
    )
    baseline_short_form = (
        None if baseline_short_form == "" else to_float(baseline_short_form)
    )
    colors = [
        "#F59E0B" if to_int(row["adoption_candidate"]) == 1 else "#2563EB"
        for row in ordered_rows
    ]
    short_colors = [
        "#F59E0B" if to_int(row["adoption_candidate"]) == 1 else "#16A34A"
        for row in ordered_rows
    ]

    y_positions = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.5), sharey=True)
    axes[0].barh(y_positions, combined_values, color=colors)
    axes[0].axvline(baseline_combined, color="#111827", linestyle="--", linewidth=1)
    axes[0].set_title("Average Combined Points MAE")
    axes[0].set_xlabel("MAE, lower is better")
    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(labels, fontsize=8)

    axes[1].barh(y_positions, short_form_values, color=short_colors)
    if baseline_short_form is not None:
        axes[1].axvline(baseline_short_form, color="#111827", linestyle="--", linewidth=1)
    axes[1].set_title("Short-History Strong-Form Driver MAE")
    axes[1].set_xlabel("MAE, lower is better")
    axes[1].grid(axis="x", color="#E5E7EB", linewidth=0.8)
    axes[0].grid(axis="x", color="#E5E7EB", linewidth=0.8)
    axes[1].text(
        0.98,
        0.04,
        "Orange = official conservative adoption",
        transform=axes[1].transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color="#92400E",
    )
    fig.suptitle("Current-Season Form Boost Backtest, 2022-2025", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIGURE_OUTPUT_PATH, dpi=180)
    plt.close(fig)
    return FIGURE_OUTPUT_PATH


def main():
    """Run current-form boost experiments and write all backtest outputs."""
    rows = read_csv(get_training_features_path())
    experiments = build_experiments()
    detail_rows = []
    focus_rows = []

    for season in BACKTEST_SEASONS:
        calibration_config = CALIBRATION_SEASONS[season]
        calibration_map, _ = train_top10_probability_calibration(
            rows,
            TOP10_MODEL_NAME,
            calibration_config["train_end_season"],
            calibration_config["calibration_season"],
        )
        for online_repeat_count in ONLINE_REPEAT_OPTIONS:
            print(
                "Training "
                f"{season} backtest, online_repeat={online_repeat_count}"
            )
            top10_model, points_model = train_backtest_models(
                rows, season, online_repeat_count
            )
            for form_boost_alpha in FORM_BOOST_ALPHA_OPTIONS:
                experiment = {
                    "experiment_id": experiment_id(online_repeat_count, form_boost_alpha),
                    "online_repeat_count": online_repeat_count,
                    "form_boost_alpha": form_boost_alpha,
                }
                detail_row, experiment_focus_rows = evaluate_configuration(
                    rows,
                    season,
                    experiment,
                    top10_model,
                    points_model,
                    calibration_map,
                )
                detail_rows.append(detail_row)
                focus_rows.extend(experiment_focus_rows)

    summary_rows = summarize_detail_rows(detail_rows)
    recommended_row = next(row for row in summary_rows if to_int(row["adoption_candidate"]) == 1)
    figure_path = save_summary_figure(summary_rows)

    write_csv(DETAIL_OUTPUT_PATH, DETAIL_FIELDS, detail_rows)
    write_csv(SUMMARY_OUTPUT_PATH, SUMMARY_FIELDS, summary_rows)
    write_csv(FOCUS_OUTPUT_PATH, FOCUS_FIELDS, focus_rows)
    best_row = summary_rows[0]
    write_json(
        JSON_OUTPUT_PATH,
        {
            "built_at": datetime.now(timezone.utc).isoformat(),
            "backtest_seasons": BACKTEST_SEASONS,
            "known_race_count": KNOWN_RACE_COUNT,
            "top10_model": TOP10_MODEL_NAME,
            "points_model": POINTS_MODEL_NAME,
            "feedback_weight": FUTURE_FEATURE_FEEDBACK_WEIGHT,
            "short_history_sample_threshold": SHORT_HISTORY_SAMPLE_THRESHOLD,
            "strong_form_score_threshold": STRONG_FORM_SCORE_THRESHOLD,
            "best_experiment": best_row,
            "recommended_experiment": recommended_row,
            "outputs": [
                str(DETAIL_OUTPUT_PATH.relative_to(BASE_DIR)),
                str(SUMMARY_OUTPUT_PATH.relative_to(BASE_DIR)),
                str(FOCUS_OUTPUT_PATH.relative_to(BASE_DIR)),
                str(JSON_OUTPUT_PATH.relative_to(BASE_DIR)),
            ],
            "figures": [str(figure_path.relative_to(BASE_DIR))],
            "notes": [
                "Positive improvement fields mean lower MAE than the baseline.",
                "Online repeat count repeats the known current-season rows in model training.",
                "Form boost is a generic ranking-score adjustment for short-history drivers with strong pre-race current form.",
                "The experiment does not change the official 2026 predictor unless the selected configuration is adopted separately.",
            ],
        },
    )
    update_model_figure_manifest(
        [
            DETAIL_OUTPUT_PATH,
            SUMMARY_OUTPUT_PATH,
            FOCUS_OUTPUT_PATH,
            JSON_OUTPUT_PATH,
        ],
        [figure_path],
    )

    print("Current-season form boost backtest completed.")
    print(
        "Best experiment: "
        f"{best_row['experiment_id']} "
        f"(combined MAE={best_row['avg_combined_points_mae']}, "
        f"short-form MAE={best_row['avg_short_form_driver_mae']})"
    )
    print(
        "Recommended adoption: "
        f"{recommended_row['experiment_id']} "
        f"(short-history improvement="
        f"{recommended_row['short_history_mae_improvement_vs_baseline']})"
    )
    print(f"Outputs saved to: {MODEL_DIR}")


if __name__ == "__main__":
    main()
