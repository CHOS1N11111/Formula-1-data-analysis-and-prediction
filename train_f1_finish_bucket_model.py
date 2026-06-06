"""Train finish-bucket classifiers for high-points-zone F1 prediction.

This script is an experimental extension that does not replace the existing
Top 10 or race-points models. It predicts finish-position buckets such as win,
podium, top five, points, and no-points, then converts bucket probabilities into
expected F1 points for diagnostics.
"""

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train_f1_points_model import (
    BASE_DIR,
    F1_POINTS_TABLE,
    FEATURE_MODES,
    FIGURE_DIR,
    MODEL_DIR,
    TEST_SEASON,
    TRAIN_END_SEASON,
    TRAIN_START_SEASON,
    add_circuit_history_features,
    build_feature_dict,
    format_float,
    get_training_features_path,
    read_csv,
    split_rows,
    to_int,
    write_csv,
    write_json,
)


OUTPUT_METRICS_PATH = MODEL_DIR / "finish_bucket_model_metrics.csv"
OUTPUT_PREDICTIONS_PATH = MODEL_DIR / "finish_bucket_predictions_2025.csv"
OUTPUT_RACE_DIAGNOSTICS_PATH = MODEL_DIR / "finish_bucket_race_diagnostics_2025.csv"
OUTPUT_SUMMARY_PATH = MODEL_DIR / "finish_bucket_model_summary.json"

BUCKET_LABELS = [
    "win",
    "podium_no_win",
    "top5_no_podium",
    "points_no_top5",
    "no_points",
]
BUCKET_EXPECTED_POINTS = {
    "win": 25.0,
    "podium_no_win": 16.5,
    "top5_no_podium": 11.0,
    "points_no_top5": 4.2,
    "no_points": 0.0,
}

METRIC_FIELDS = [
    "feature_mode",
    "model",
    "train_seasons",
    "test_season",
    "train_records",
    "test_records",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "log_loss",
    "expected_points_mae",
    "winner_hit_rate",
    "avg_bucket_rank_of_actual_bucket",
]

PREDICTION_FIELDS = [
    "season",
    "round",
    "race_name",
    "driver_id",
    "driver_name",
    "constructor_name",
    "actual_finish_position",
    "actual_bucket",
    "predicted_bucket",
    "bucket_rank_of_actual_bucket",
    "prob_win",
    "prob_podium_no_win",
    "prob_top5_no_podium",
    "prob_points_no_top5",
    "prob_no_points",
    "expected_points_from_buckets",
    "actual_points",
]

RACE_DIAGNOSTIC_FIELDS = [
    "season",
    "round",
    "race_name",
    "feature_mode",
    "model",
    "predicted_winner",
    "actual_winner",
    "winner_hit",
    "predicted_winner_prob_win",
    "runner_up_by_prob_win",
    "winner_runner_up_prob_gap",
    "predicted_winner_expected_points",
]


def finish_bucket(row):
    """Map a finish position to an interpretable high-points-zone bucket."""
    position = to_int(row.get("finish_position", 0))
    if position == 1:
        return "win"
    if 2 <= position <= 3:
        return "podium_no_win"
    if 4 <= position <= 5:
        return "top5_no_podium"
    if 6 <= position <= 10:
        return "points_no_top5"
    return "no_points"


def current_rule_points_from_finish(row):
    """Map finish position to the current full Grand Prix points table."""
    position = to_int(row.get("finish_position", 0))
    if 1 <= position <= len(F1_POINTS_TABLE):
        return float(F1_POINTS_TABLE[position - 1])
    return 0.0


def build_bucket_y(rows):
    """Build finish bucket target labels."""
    return np.array([finish_bucket(row) for row in rows])


def build_models():
    """Create finish-bucket multiclass classifiers."""
    return {
        "logistic_regression_multinomial": Pipeline(
            [
                ("vectorizer", DictVectorizer(sparse=False)),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        solver="lbfgs",
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting_classifier": Pipeline(
            [
                ("vectorizer", DictVectorizer(sparse=False)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=250,
                        learning_rate=0.05,
                        l2_regularization=0.05,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def bucket_probabilities(model, x_rows):
    """Return probability dictionaries aligned to all bucket labels."""
    probabilities = model.predict_proba(x_rows)
    classes = list(model.classes_)
    rows = []
    for values in probabilities:
        row = {label: 0.0 for label in BUCKET_LABELS}
        for label, value in zip(classes, values):
            row[label] = float(value)
        rows.append(row)
    return rows


def expected_points(probability_row):
    """Convert finish-bucket probabilities into expected F1 points."""
    return sum(
        probability_row[label] * BUCKET_EXPECTED_POINTS[label]
        for label in BUCKET_LABELS
    )


def bucket_rank(probability_row, actual_bucket):
    """Return the probability rank of the actual finish bucket."""
    ordered = sorted(BUCKET_LABELS, key=lambda label: probability_row[label], reverse=True)
    return ordered.index(actual_bucket) + 1


def race_winner_diagnostics(prediction_rows, feature_mode, model_name):
    """Summarize winner prediction quality race by race."""
    grouped = defaultdict(list)
    for row in prediction_rows:
        grouped[(row["season"], row["round"])].append(row)

    diagnostics = []
    for _, race_rows in sorted(grouped.items(), key=lambda item: (int(item[0][0]), int(item[0][1]))):
        ordered = sorted(race_rows, key=lambda row: float(row["prob_win"]), reverse=True)
        predicted = ordered[0]
        runner_up = ordered[1] if len(ordered) > 1 else {}
        actual_winner = next(
            (row for row in race_rows if to_int(row["actual_finish_position"]) == 1),
            {},
        )
        diagnostics.append(
            {
                "season": predicted["season"],
                "round": predicted["round"],
                "race_name": predicted["race_name"],
                "feature_mode": feature_mode,
                "model": model_name,
                "predicted_winner": predicted["driver_name"],
                "actual_winner": actual_winner.get("driver_name", ""),
                "winner_hit": int(predicted["driver_id"] == actual_winner.get("driver_id", "")),
                "predicted_winner_prob_win": predicted["prob_win"],
                "runner_up_by_prob_win": runner_up.get("driver_name", ""),
                "winner_runner_up_prob_gap": format_float(
                    float(predicted["prob_win"]) - float(runner_up.get("prob_win", 0.0))
                ),
                "predicted_winner_expected_points": predicted["expected_points_from_buckets"],
            }
        )
    return diagnostics


def evaluate_model(feature_mode, model_name, model, train_rows, test_rows):
    """Train and evaluate one finish-bucket classifier."""
    train_x = [build_feature_dict(row, feature_mode) for row in train_rows]
    test_x = [build_feature_dict(row, feature_mode) for row in test_rows]
    train_y = build_bucket_y(train_rows)
    test_y = build_bucket_y(test_rows)
    model.fit(train_x, train_y)

    predicted_labels = model.predict(test_x)
    probability_rows = bucket_probabilities(model, test_x)
    expected_values = [expected_points(row) for row in probability_rows]
    actual_points = [current_rule_points_from_finish(row) for row in test_rows]
    actual_buckets = [finish_bucket(row) for row in test_rows]
    actual_bucket_ranks = [
        bucket_rank(probability_row, actual_bucket)
        for probability_row, actual_bucket in zip(probability_rows, actual_buckets)
    ]

    prediction_rows = []
    for row, predicted_label, probability_row, expected_value, actual_point, actual_bucket, actual_rank in zip(
        test_rows,
        predicted_labels,
        probability_rows,
        expected_values,
        actual_points,
        actual_buckets,
        actual_bucket_ranks,
    ):
        prediction_rows.append(
            {
                "season": row["season"],
                "round": row["round"],
                "race_name": row["race_name"],
                "driver_id": row["driver_id"],
                "driver_name": row["driver_name"],
                "constructor_name": row["constructor_name"],
                "actual_finish_position": row["finish_position"],
                "actual_bucket": actual_bucket,
                "predicted_bucket": predicted_label,
                "bucket_rank_of_actual_bucket": actual_rank,
                "prob_win": format_float(probability_row["win"]),
                "prob_podium_no_win": format_float(probability_row["podium_no_win"]),
                "prob_top5_no_podium": format_float(probability_row["top5_no_podium"]),
                "prob_points_no_top5": format_float(probability_row["points_no_top5"]),
                "prob_no_points": format_float(probability_row["no_points"]),
                "expected_points_from_buckets": format_float(expected_value),
                "actual_points": format_float(actual_point),
            }
        )

    race_diagnostics = race_winner_diagnostics(prediction_rows, feature_mode, model_name)
    winner_hit_rate = (
        sum(int(row["winner_hit"]) for row in race_diagnostics) / len(race_diagnostics)
        if race_diagnostics
        else 0.0
    )

    metric_row = {
        "feature_mode": feature_mode,
        "model": model_name,
        "train_seasons": f"{TRAIN_START_SEASON}-{TRAIN_END_SEASON}",
        "test_season": TEST_SEASON,
        "train_records": len(train_rows),
        "test_records": len(test_rows),
        "accuracy": format_float(accuracy_score(test_y, predicted_labels)),
        "macro_f1": format_float(f1_score(test_y, predicted_labels, average="macro")),
        "weighted_f1": format_float(f1_score(test_y, predicted_labels, average="weighted")),
        "log_loss": format_float(log_loss(test_y, model.predict_proba(test_x), labels=model.classes_)),
        "expected_points_mae": format_float(mean_absolute_error(actual_points, expected_values)),
        "winner_hit_rate": format_float(winner_hit_rate),
        "avg_bucket_rank_of_actual_bucket": format_float(
            sum(actual_bucket_ranks) / len(actual_bucket_ranks)
        ),
    }
    return metric_row, prediction_rows, race_diagnostics


def main():
    """Run finish-bucket training and 2025 diagnostics."""
    feature_rows = add_circuit_history_features(read_csv(get_training_features_path()))
    split_result = split_rows(feature_rows)
    train_rows, test_rows = split_result[0], split_result[1]
    metric_rows = []
    all_prediction_rows = []
    all_race_diagnostic_rows = []

    for feature_mode in FEATURE_MODES:
        for model_name, model in build_models().items():
            print(f"Training finish bucket model: {feature_mode}, {model_name}")
            metric_row, prediction_rows, race_diagnostics = evaluate_model(
                feature_mode,
                model_name,
                model,
                train_rows,
                test_rows,
            )
            metric_rows.append(metric_row)
            for row in prediction_rows:
                row["feature_mode"] = feature_mode
                row["model"] = model_name
            all_prediction_rows.extend(prediction_rows)
            all_race_diagnostic_rows.extend(race_diagnostics)

    metric_rows.sort(
        key=lambda row: (
            -float(row["winner_hit_rate"]),
            -float(row["weighted_f1"]),
            float(row["expected_points_mae"]),
        )
    )
    write_csv(OUTPUT_METRICS_PATH, METRIC_FIELDS, metric_rows)
    write_csv(
        OUTPUT_PREDICTIONS_PATH,
        ["feature_mode", "model"] + PREDICTION_FIELDS,
        all_prediction_rows,
    )
    write_csv(OUTPUT_RACE_DIAGNOSTICS_PATH, RACE_DIAGNOSTIC_FIELDS, all_race_diagnostic_rows)
    write_json(
        OUTPUT_SUMMARY_PATH,
        {
            "built_at": datetime.now(timezone.utc).isoformat(),
            "target": "finish_bucket",
            "bucket_labels": BUCKET_LABELS,
            "bucket_expected_points": BUCKET_EXPECTED_POINTS,
            "best_model": metric_rows[0]["model"],
            "best_feature_mode": metric_rows[0]["feature_mode"],
            "best_winner_hit_rate": metric_rows[0]["winner_hit_rate"],
            "best_expected_points_mae": metric_rows[0]["expected_points_mae"],
            "outputs": [
                str(OUTPUT_METRICS_PATH.relative_to(BASE_DIR)),
                str(OUTPUT_PREDICTIONS_PATH.relative_to(BASE_DIR)),
                str(OUTPUT_RACE_DIAGNOSTICS_PATH.relative_to(BASE_DIR)),
            ],
            "notes": [
                "This is an additive experiment and does not replace existing Top 10 or race-points models.",
                "Finish buckets provide a high-points-zone signal that may later be tested in final race ranking.",
            ],
        },
    )
    print(f"Finish bucket metrics saved to: {OUTPUT_METRICS_PATH}")


if __name__ == "__main__":
    main()
