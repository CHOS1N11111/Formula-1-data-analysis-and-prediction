"""Backtest F1 model combinations with full season-level metrics.

This script evaluates Top 10 and race-points model combinations on historical
season rollouts. It does not change the final 2026 prediction scenarios. The
goal is to provide an additional selection view based on full-season standings
quality rather than one-race model metrics alone.
"""

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
    FUTURE_FEATURE_FEEDBACK_WEIGHT,
    MODEL_DIR,
    TOP10_METRICS_PATH,
    POINTS_METRICS_PATH,
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
    select_top_model_rows,
    to_float,
    to_int,
    train_top10_probability_calibration,
    update_model_figure_manifest,
    update_state_after_damped_projected_race,
    write_csv,
    write_json,
)


BACKTEST_SEASONS = [2022, 2023, 2024, 2025]
KNOWN_RACE_COUNT = 5
TOP10_CANDIDATE_COUNT = 3
POINTS_CANDIDATE_COUNT = 5
CALIBRATION_SEASONS = {
    2022: {"train_end_season": 2020, "calibration_season": 2021},
    2023: {"train_end_season": 2021, "calibration_season": 2022},
    2024: {"train_end_season": 2022, "calibration_season": 2023},
    2025: {"train_end_season": 2023, "calibration_season": 2024},
}

DETAIL_OUTPUT_PATH = MODEL_DIR / "season_model_scenario_backtest_metrics.csv"
SUMMARY_OUTPUT_PATH = MODEL_DIR / "season_model_scenario_backtest_summary.csv"
SUMMARY_JSON_PATH = MODEL_DIR / "season_model_scenario_backtest_summary.json"
FIGURE_OUTPUT_PATH = FIGURE_DIR / "season_model_scenario_backtest.png"

DETAIL_FIELDS = [
    "top10_model_rank",
    "top10_model",
    "points_model_rank",
    "points_model",
    "season",
    "known_race_count",
    "remaining_race_count",
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
    "deterministic_driver_winner_count",
    "max_driver_winner_share",
    "winner_distribution",
    "avg_winner_runner_up_score_gap",
    "calibration_train_end_season",
    "calibration_season",
]

SUMMARY_FIELDS = [
    "scenario_backtest_rank",
    "top10_model_rank",
    "top10_model",
    "top10_2025_f1",
    "points_model_rank",
    "points_model",
    "points_2025_mae",
    "season_count",
    "avg_driver_points_mae",
    "avg_constructor_points_mae",
    "avg_combined_points_mae",
    "driver_champion_hit_rate",
    "constructor_champion_hit_rate",
    "avg_driver_top3_overlap",
    "avg_constructor_top3_overlap",
    "avg_deterministic_driver_winner_count",
    "avg_max_driver_winner_share",
    "avg_winner_runner_up_score_gap",
    "concentration_penalty",
    "selection_score",
    "recommended_role",
]


def build_candidate_scenarios():
    """Build Top 10 and points-model candidate combinations from 2025 metrics."""
    top10_rows = select_top_model_rows(
        TOP10_METRICS_PATH,
        "f1",
        lower_is_better=False,
        limit=TOP10_CANDIDATE_COUNT,
    )
    points_rows = select_top_model_rows(
        POINTS_METRICS_PATH,
        "mae",
        lower_is_better=True,
        limit=POINTS_CANDIDATE_COUNT,
    )
    scenarios = []
    for top10_index, top10_row in enumerate(top10_rows, start=1):
        for points_index, points_row in enumerate(points_rows, start=1):
            scenarios.append(
                {
                    "top10_model_rank": top10_index,
                    "top10_model": top10_row["model"],
                    "top10_2025_f1": to_float(top10_row["f1"]),
                    "points_model_rank": points_index,
                    "points_model": points_row["model"],
                    "points_2025_mae": to_float(points_row["mae"]),
                }
            )
    return scenarios


def train_backtest_models(rows, season, scenario):
    """Train one candidate model pair for a historical season rollout."""
    train_rows = [
        row
        for row in rows
        if 2003 <= to_int(row["season"]) < season
    ]
    known_rows = [
        row
        for row in rows
        if to_int(row["season"]) == season
        and to_int(row["round"]) <= KNOWN_RACE_COUNT
    ]
    train_rows = train_rows + known_rows * CURRENT_SEASON_ONLINE_REPEAT

    top10_model = build_top10_models()[scenario["top10_model"]]
    points_model = build_points_models()[scenario["points_model"]]
    top10_model.fit(build_x(train_rows, FEATURE_MODE), build_binary_y(train_rows, "is_top10"))
    points_model.fit(build_x(train_rows, FEATURE_MODE), build_points_y(train_rows))

    calibration_config = CALIBRATION_SEASONS[season]
    calibration_map, _ = train_top10_probability_calibration(
        rows,
        scenario["top10_model"],
        calibration_config["train_end_season"],
        calibration_config["calibration_season"],
    )
    return top10_model, points_model, calibration_map, calibration_config


def build_season_state(rows, season):
    """Build known-season state from the first backtest races."""
    known_rows = [
        row
        for row in rows
        if to_int(row["season"]) == season
        and to_int(row["round"]) <= KNOWN_RACE_COUNT
    ]
    state_rows = []
    for row in known_rows:
        state_row = dict(row)
        state_row["season"] = "2026"
        state_rows.append(state_row)
    return build_current_2026_state(state_rows)


def build_driver_pool_from_state(state):
    """Return active driver metadata inferred from known-season races."""
    return sorted(state["driver_meta"].values(), key=lambda row: row["driver_id"])


def build_remaining_schedule(rows, season):
    """Create remaining-race schedule rows for a historical season."""
    schedule_rows = []
    seen = set()
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
    """Keep historical context limited to past seasons and known races."""
    return [
        row
        for row in rows
        if to_int(row["season"]) < season
        or (
            to_int(row["season"]) == season
            and to_int(row["round"]) <= KNOWN_RACE_COUNT
        )
    ]


def predict_backtest_remaining_races(schedule_rows, driver_pool, state, history_rows, top10_model, points_model, calibration_map):
    """Predict remaining races for one season backtest."""
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
    """Calculate actual final points under the current F1 points rule."""
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
    """Combine known points with predicted remaining-race points."""
    driver_points = defaultdict(float, state["driver_points"])
    constructor_points = defaultdict(float, state["constructor_points"])
    for row in deterministic_rows:
        points = to_float(row["deterministic_rule_mapped_points"])
        driver_points[row["driver_id"]] += points
        constructor_points[row["constructor_id"]] += points
    return driver_points, constructor_points


def points_mae(actual_points, predicted_points):
    """Compute points MAE across the union of actual and predicted entities."""
    entity_ids = sorted(set(actual_points) | set(predicted_points))
    if not entity_ids:
        return 0.0
    errors = [
        abs(actual_points.get(entity_id, 0.0) - predicted_points.get(entity_id, 0.0))
        for entity_id in entity_ids
    ]
    return float(np.mean(errors))


def champion_id(points):
    """Return champion ID from a points dictionary."""
    return sorted(points.items(), key=lambda item: (-item[1], item[0]))[0][0]


def top_n_ids(points, n=3):
    """Return top-N entity IDs from a points dictionary."""
    return [
        entity_id
        for entity_id, _ in sorted(points.items(), key=lambda item: (-item[1], item[0]))[:n]
    ]


def winner_diagnostics(deterministic_rows):
    """Summarize deterministic winner diversity and score margins."""
    grouped = defaultdict(list)
    for row in deterministic_rows:
        grouped[(to_int(row["season"]), to_int(row["round"]))].append(row)

    winners = []
    gaps = []
    for race_rows in grouped.values():
        sorted_rows = sorted(race_rows, key=lambda row: to_int(row["deterministic_rank"]))
        if not sorted_rows:
            continue
        winners.append(sorted_rows[0]["driver_name"])
        if len(sorted_rows) > 1:
            gaps.append(to_float(sorted_rows[0].get("winner_runner_up_score_gap", 0)))

    distribution = {
        driver_name: winners.count(driver_name)
        for driver_name in sorted(set(winners))
    }
    max_share = max(distribution.values()) / len(winners) if winners else 0.0
    avg_gap = float(np.mean(gaps)) if gaps else 0.0
    return distribution, len(distribution), max_share, avg_gap


def evaluate_scenario_season(rows, scenario, season):
    """Evaluate one model combination on one historical season."""
    top10_model, points_model, calibration_map, calibration_config = train_backtest_models(
        rows,
        season,
        scenario,
    )
    state = build_season_state(rows, season)
    driver_pool = build_driver_pool_from_state(state)
    schedule_rows = build_remaining_schedule(rows, season)
    history_rows = build_backtest_history_rows(rows, season)
    deterministic_rows = predict_backtest_remaining_races(
        schedule_rows,
        driver_pool,
        state,
        history_rows,
        top10_model,
        points_model,
        calibration_map,
    )

    actual_driver, actual_constructor, driver_names, constructor_names = actual_final_points(rows, season)
    predicted_driver, predicted_constructor = projected_final_points(state, deterministic_rows)
    actual_driver_champion = champion_id(actual_driver)
    predicted_driver_champion = champion_id(predicted_driver)
    actual_constructor_champion = champion_id(actual_constructor)
    predicted_constructor_champion = champion_id(predicted_constructor)
    winner_distribution, winner_count, max_share, avg_gap = winner_diagnostics(deterministic_rows)

    driver_mae = points_mae(actual_driver, predicted_driver)
    constructor_mae = points_mae(actual_constructor, predicted_constructor)
    return {
        "top10_model_rank": scenario["top10_model_rank"],
        "top10_model": scenario["top10_model"],
        "points_model_rank": scenario["points_model_rank"],
        "points_model": scenario["points_model"],
        "season": season,
        "known_race_count": KNOWN_RACE_COUNT,
        "remaining_race_count": len(schedule_rows),
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
        "deterministic_driver_winner_count": winner_count,
        "max_driver_winner_share": format_float(max_share),
        "winner_distribution": str(winner_distribution),
        "avg_winner_runner_up_score_gap": format_float(avg_gap),
        "calibration_train_end_season": calibration_config["train_end_season"],
        "calibration_season": calibration_config["calibration_season"],
    }


def normalize(values, lower_is_better=False):
    """Normalize numeric values to a 0-1 score."""
    values = list(values)
    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    if abs(maximum - minimum) < 1e-12:
        return [1.0 for _ in values]
    if lower_is_better:
        return [(maximum - value) / (maximum - minimum) for value in values]
    return [(value - minimum) / (maximum - minimum) for value in values]


def summarize_scenarios(detail_rows, scenarios):
    """Aggregate season-level metrics into model-combination rankings."""
    summary_rows = []
    scenario_lookup = {
        (scenario["top10_model"], scenario["points_model"]): scenario
        for scenario in scenarios
    }
    grouped = defaultdict(list)
    for row in detail_rows:
        grouped[(row["top10_model"], row["points_model"])].append(row)

    raw_rows = []
    for key, rows in grouped.items():
        scenario = scenario_lookup[key]
        raw_rows.append(
            {
                "top10_model_rank": scenario["top10_model_rank"],
                "top10_model": scenario["top10_model"],
                "top10_2025_f1": scenario["top10_2025_f1"],
                "points_model_rank": scenario["points_model_rank"],
                "points_model": scenario["points_model"],
                "points_2025_mae": scenario["points_2025_mae"],
                "season_count": len(rows),
                "avg_driver_points_mae": float(np.mean([to_float(row["driver_points_mae"]) for row in rows])),
                "avg_constructor_points_mae": float(np.mean([to_float(row["constructor_points_mae"]) for row in rows])),
                "avg_combined_points_mae": float(np.mean([to_float(row["combined_points_mae"]) for row in rows])),
                "driver_champion_hit_rate": float(np.mean([to_int(row["driver_champion_hit"]) for row in rows])),
                "constructor_champion_hit_rate": float(np.mean([to_int(row["constructor_champion_hit"]) for row in rows])),
                "avg_driver_top3_overlap": float(np.mean([to_int(row["driver_top3_overlap"]) for row in rows])),
                "avg_constructor_top3_overlap": float(np.mean([to_int(row["constructor_top3_overlap"]) for row in rows])),
                "avg_deterministic_driver_winner_count": float(np.mean([to_int(row["deterministic_driver_winner_count"]) for row in rows])),
                "avg_max_driver_winner_share": float(np.mean([to_float(row["max_driver_winner_share"]) for row in rows])),
                "avg_winner_runner_up_score_gap": float(np.mean([to_float(row["avg_winner_runner_up_score_gap"]) for row in rows])),
            }
        )

    mae_scores = normalize([row["avg_combined_points_mae"] for row in raw_rows], lower_is_better=True)
    driver_champion_scores = normalize([row["driver_champion_hit_rate"] for row in raw_rows])
    constructor_champion_scores = normalize([row["constructor_champion_hit_rate"] for row in raw_rows])
    top3_scores = normalize(
        [
            (row["avg_driver_top3_overlap"] + row["avg_constructor_top3_overlap"]) / 2
            for row in raw_rows
        ]
    )
    concentration_scores = normalize(
        [row["avg_max_driver_winner_share"] for row in raw_rows],
        lower_is_better=True,
    )

    for index, row in enumerate(raw_rows):
        concentration_penalty = row["avg_max_driver_winner_share"]
        selection_score = (
            0.40 * mae_scores[index]
            + 0.20 * driver_champion_scores[index]
            + 0.15 * constructor_champion_scores[index]
            + 0.15 * top3_scores[index]
            + 0.10 * concentration_scores[index]
        )
        recommended_role = "candidate"
        if concentration_penalty >= 0.85:
            recommended_role = "sensitivity_only_high_concentration"
        summary_rows.append(
            {
                "scenario_backtest_rank": 0,
                "top10_model_rank": row["top10_model_rank"],
                "top10_model": row["top10_model"],
                "top10_2025_f1": format_float(row["top10_2025_f1"]),
                "points_model_rank": row["points_model_rank"],
                "points_model": row["points_model"],
                "points_2025_mae": format_float(row["points_2025_mae"]),
                "season_count": row["season_count"],
                "avg_driver_points_mae": format_float(row["avg_driver_points_mae"]),
                "avg_constructor_points_mae": format_float(row["avg_constructor_points_mae"]),
                "avg_combined_points_mae": format_float(row["avg_combined_points_mae"]),
                "driver_champion_hit_rate": format_float(row["driver_champion_hit_rate"]),
                "constructor_champion_hit_rate": format_float(row["constructor_champion_hit_rate"]),
                "avg_driver_top3_overlap": format_float(row["avg_driver_top3_overlap"]),
                "avg_constructor_top3_overlap": format_float(row["avg_constructor_top3_overlap"]),
                "avg_deterministic_driver_winner_count": format_float(row["avg_deterministic_driver_winner_count"]),
                "avg_max_driver_winner_share": format_float(row["avg_max_driver_winner_share"]),
                "avg_winner_runner_up_score_gap": format_float(row["avg_winner_runner_up_score_gap"]),
                "concentration_penalty": format_float(concentration_penalty),
                "selection_score": format_float(selection_score),
                "recommended_role": recommended_role,
            }
        )

    ordered_rows = sorted(
        summary_rows,
        key=lambda row: (
            -to_float(row["selection_score"]),
            to_float(row["avg_combined_points_mae"]),
            to_float(row["concentration_penalty"]),
            row["top10_model"],
            row["points_model"],
        ),
    )
    for rank, row in enumerate(ordered_rows, start=1):
        row["scenario_backtest_rank"] = rank
    return ordered_rows


def save_backtest_chart(summary_rows):
    """Save a compact chart of the best model-combination backtests."""
    top_rows = summary_rows[:10]
    labels = [
        f"{row['top10_model']} +\n{row['points_model']}"
        for row in reversed(top_rows)
    ]
    mae_values = [to_float(row["avg_combined_points_mae"]) for row in reversed(top_rows)]
    score_values = [to_float(row["selection_score"]) for row in reversed(top_rows)]

    fig, ax1 = plt.subplots(figsize=(11, 6.5))
    positions = np.arange(len(labels))
    ax1.barh(positions, mae_values, color="#60A5FA", label="Combined points MAE")
    ax1.set_yticks(positions)
    ax1.set_yticklabels(labels, fontsize=8)
    ax1.set_xlabel("Average combined points MAE")
    ax1.grid(axis="x", color="#D1D5DB", linewidth=0.8, alpha=0.9)
    ax1.set_axisbelow(True)

    ax2 = ax1.twiny()
    ax2.plot(score_values, positions, color="#DC2626", marker="o", label="Selection score")
    ax2.set_xlabel("Selection score")
    ax2.set_xlim(0, max(1.0, max(score_values) * 1.08 if score_values else 1.0))
    ax1.set_title("Season-Level Model Combination Backtest, 2022-2025")

    lines, labels_left = ax1.get_legend_handles_labels()
    lines2, labels_right = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels_left + labels_right, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURE_OUTPUT_PATH, dpi=180)
    plt.close(fig)


def main():
    """Run season-level model-combination backtests and write outputs."""
    rows = read_csv(get_training_features_path())
    scenarios = build_candidate_scenarios()
    detail_rows = []
    for scenario in scenarios:
        for season in BACKTEST_SEASONS:
            print(
                "Backtesting "
                f"{scenario['top10_model']} + {scenario['points_model']} "
                f"on {season}"
            )
            detail_rows.append(evaluate_scenario_season(rows, scenario, season))

    summary_rows = summarize_scenarios(detail_rows, scenarios)
    best_row = summary_rows[0]
    usable_rows = [
        row for row in summary_rows
        if row["recommended_role"] == "candidate"
    ]
    best_usable_row = usable_rows[0] if usable_rows else None
    write_csv(DETAIL_OUTPUT_PATH, DETAIL_FIELDS, detail_rows)
    write_csv(SUMMARY_OUTPUT_PATH, SUMMARY_FIELDS, summary_rows)
    save_backtest_chart(summary_rows)
    update_model_figure_manifest(
        [
            DETAIL_OUTPUT_PATH,
            SUMMARY_OUTPUT_PATH,
            SUMMARY_JSON_PATH,
        ],
        [FIGURE_OUTPUT_PATH],
    )
    write_json(
        SUMMARY_JSON_PATH,
        {
            "built_at": datetime.now(timezone.utc).isoformat(),
            "backtest_seasons": BACKTEST_SEASONS,
            "known_race_count": KNOWN_RACE_COUNT,
            "top10_candidate_count": TOP10_CANDIDATE_COUNT,
            "points_candidate_count": POINTS_CANDIDATE_COUNT,
            "feature_mode": FEATURE_MODE,
            "future_feature_feedback_weight": FUTURE_FEATURE_FEEDBACK_WEIGHT,
            "current_season_online_repeat": CURRENT_SEASON_ONLINE_REPEAT,
            "selection_score_weights": {
                "combined_points_mae": 0.40,
                "driver_champion_hit_rate": 0.20,
                "constructor_champion_hit_rate": 0.15,
                "top3_overlap": 0.15,
                "winner_concentration": 0.10,
            },
            "best_overall_row": best_row,
            "best_usable_candidate_row": best_usable_row,
            "outputs": [
                str(DETAIL_OUTPUT_PATH.relative_to(BASE_DIR)),
                str(SUMMARY_OUTPUT_PATH.relative_to(BASE_DIR)),
                str(SUMMARY_JSON_PATH.relative_to(BASE_DIR)),
                str(FIGURE_OUTPUT_PATH.relative_to(BASE_DIR)),
            ],
            "notes": [
                "This script does not modify the final S1/S2/S3 prediction scenarios.",
                "It evaluates candidate model combinations by rolling out complete historical seasons after the first five known races.",
                "Lower combined points MAE and lower winner concentration are preferred.",
                "Backtest features use only past seasons plus known races from the test season.",
            ],
        },
    )
    print(f"Best season-level model combination: {best_row['top10_model']} + {best_row['points_model']}")
    if best_usable_row:
        print(
            "Best usable candidate: "
            f"{best_usable_row['top10_model']} + {best_usable_row['points_model']}"
        )
    print(f"Summary saved to: {SUMMARY_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
