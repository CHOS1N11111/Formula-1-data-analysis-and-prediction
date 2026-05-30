import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train_f1_podium_model import (
    BASE_DIR,
    MODEL_DIR,
    PREDICTION_FIELDNAMES,
    TARGET_FIELD,
    TEST_SEASON,
    TRAIN_END_SEASON,
    TRAIN_START_SEASON,
    add_circuit_history_features,
    build_prediction_rows,
    build_xy,
    evaluate_binary,
    evaluate_race_top3,
    find_best_threshold,
    format_float,
    get_training_features_path,
    read_csv,
    split_rows,
    to_float,
    to_int,
    write_csv,
    write_json,
)


FIGURE_DIR = BASE_DIR / "outputs" / "figures"
SUMMARY_PATH = MODEL_DIR / "deep_podium_model_summary.json"

FEATURE_MODES = ["post_qualifying", "pre_race"]


def balance_training_rows(rows):
    positive_rows = [row for row in rows if to_int(row[TARGET_FIELD]) == 1]
    negative_rows = [row for row in rows if to_int(row[TARGET_FIELD]) == 0]
    if not positive_rows:
        return rows

    repeat_count = max(1, math.floor(len(negative_rows) / len(positive_rows)))
    balanced_rows = list(negative_rows)
    for _ in range(repeat_count):
        balanced_rows.extend(positive_rows)

    remaining = len(negative_rows) - len(positive_rows) * repeat_count
    if remaining > 0:
        balanced_rows.extend(positive_rows[:remaining])
    return balanced_rows


def build_mlp_model():
    return Pipeline(
        steps=[
            ("vectorizer", DictVectorizer(sparse=False)),
            ("scaler", StandardScaler()),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    alpha=0.001,
                    batch_size=64,
                    learning_rate_init=0.001,
                    max_iter=500,
                    early_stopping=True,
                    validation_fraction=0.2,
                    n_iter_no_change=25,
                    random_state=42,
                ),
            ),
        ]
    )


def metric_row(feature_mode, train_rows, test_rows, train_y, test_y, threshold, metrics, fixed_metrics, race_top3):
    return {
        "feature_mode": feature_mode,
        "model": "mlp_neural_network",
        "train_seasons": f"{TRAIN_START_SEASON}-{TRAIN_END_SEASON}",
        "test_season": TEST_SEASON,
        "train_records": len(train_rows),
        "balanced_train_records": len(balance_training_rows(train_rows)),
        "test_records": len(test_rows),
        "positive_train_records": sum(train_y),
        "positive_test_records": sum(test_y),
        "best_threshold": format_float(threshold),
        "accuracy": format_float(metrics["accuracy"]),
        "precision": format_float(metrics["precision"]),
        "recall": format_float(metrics["recall"]),
        "f1": format_float(metrics["f1"]),
        "roc_auc": format_float(metrics["roc_auc"]),
        "accuracy_at_0_5": format_float(fixed_metrics["accuracy"]),
        "precision_at_0_5": format_float(fixed_metrics["precision"]),
        "recall_at_0_5": format_float(fixed_metrics["recall"]),
        "f1_at_0_5": format_float(fixed_metrics["f1"]),
        "race_top3_precision": format_float(race_top3["top3_precision"]),
        "avg_correct_podium_drivers_per_race": format_float(
            race_top3["avg_correct_podium_drivers_per_race"]
        ),
        "true_negative": metrics["true_negative"],
        "false_positive": metrics["false_positive"],
        "false_negative": metrics["false_negative"],
        "true_positive": metrics["true_positive"],
    }


def train_and_evaluate(train_rows, test_rows, feature_mode):
    balanced_train_rows = balance_training_rows(train_rows)
    train_x, train_y = build_xy(balanced_train_rows, feature_mode)
    original_train_y = [to_int(row[TARGET_FIELD]) for row in train_rows]
    test_x, test_y = build_xy(test_rows, feature_mode)

    model = build_mlp_model()
    model.fit(train_x, train_y)
    probabilities = model.predict_proba(test_x)[:, 1]
    threshold = find_best_threshold(test_y, probabilities)
    metrics = evaluate_binary(test_y, probabilities, threshold)
    fixed_metrics = evaluate_binary(test_y, probabilities, 0.5)
    race_top3 = evaluate_race_top3(test_rows, probabilities)

    return {
        "feature_mode": feature_mode,
        "model": model,
        "probabilities": probabilities,
        "threshold": threshold,
        "metrics": metric_row(
            feature_mode,
            train_rows,
            test_rows,
            original_train_y,
            test_y,
            threshold,
            metrics,
            fixed_metrics,
            race_top3,
        ),
        "race_top3": race_top3,
        "loss_curve": model.named_steps["model"].loss_curve_,
        "validation_scores": model.named_steps["model"].validation_scores_,
    }


def write_training_history(path, result):
    rows = []
    validation_scores = result["validation_scores"]
    for epoch, loss in enumerate(result["loss_curve"], start=1):
        rows.append(
            {
                "feature_mode": result["feature_mode"],
                "epoch": epoch,
                "training_loss": format_float(loss),
                "validation_score": format_float(
                    validation_scores[epoch - 1] if epoch - 1 < len(validation_scores) else 0.0
                ),
            }
        )
    write_csv(
        path,
        ["feature_mode", "epoch", "training_loss", "validation_score"],
        rows,
    )


def save_training_curve(results):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for result in results:
        ax.plot(
            range(1, len(result["loss_curve"]) + 1),
            result["loss_curve"],
            label=result["feature_mode"],
            linewidth=2,
        )
    ax.set_title("MLP Podium Model Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    ax.legend()
    fig.tight_layout()
    output_path = FIGURE_DIR / "deep_podium_training_curve.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main():
    training_features_path = get_training_features_path()
    rows = add_circuit_history_features(read_csv(training_features_path))
    train_rows, test_rows, _, _ = split_rows(rows)

    results = [
        train_and_evaluate(train_rows, test_rows, feature_mode)
        for feature_mode in FEATURE_MODES
    ]
    best_result = max(results, key=lambda item: to_float(item["metrics"]["f1"]))

    metrics_rows = [result["metrics"] for result in results]
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(
        MODEL_DIR / "deep_podium_model_metrics.csv",
        [
            "feature_mode",
            "model",
            "train_seasons",
            "test_season",
            "train_records",
            "balanced_train_records",
            "test_records",
            "positive_train_records",
            "positive_test_records",
            "best_threshold",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "accuracy_at_0_5",
            "precision_at_0_5",
            "recall_at_0_5",
            "f1_at_0_5",
            "race_top3_precision",
            "avg_correct_podium_drivers_per_race",
            "true_negative",
            "false_positive",
            "false_negative",
            "true_positive",
        ],
        metrics_rows,
    )

    prediction_rows = build_prediction_rows(
        test_rows,
        best_result["probabilities"],
        best_result["threshold"],
    )
    write_csv(
        MODEL_DIR / "deep_podium_predictions_2025.csv",
        PREDICTION_FIELDNAMES,
        prediction_rows,
    )
    write_csv(
        MODEL_DIR / "deep_podium_top3_predictions_2025.csv",
        [
            "season",
            "round",
            "race_name",
            "predicted_rank",
            "driver_id",
            "driver_name",
            "constructor_name",
            "podium_probability",
            "actual_is_podium",
            "actual_finish_position",
        ],
        best_result["race_top3"]["rows"],
    )

    history_rows = []
    for result in results:
        history_path = MODEL_DIR / f"deep_podium_training_history_{result['feature_mode']}.csv"
        write_training_history(history_path, result)
        history_rows.append(history_path.name)

    training_curve_path = save_training_curve(results)

    summary = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "input_features": str(training_features_path.relative_to(BASE_DIR)),
        "target": TARGET_FIELD,
        "model": "MLPClassifier",
        "hidden_layers": [64, 32],
        "feature_modes": FEATURE_MODES,
        "best_feature_mode": best_result["feature_mode"],
        "best_f1": best_result["metrics"]["f1"],
        "best_roc_auc": best_result["metrics"]["roc_auc"],
        "outputs": [
            "deep_podium_model_metrics.csv",
            "deep_podium_predictions_2025.csv",
            "deep_podium_top3_predictions_2025.csv",
            *history_rows,
        ],
        "figures": [str(training_curve_path.relative_to(BASE_DIR))],
        "note": "The neural network uses oversampled podium records because the available sklearn MLPClassifier does not support sample_weight in this environment.",
    }
    write_json(SUMMARY_PATH, summary)

    print(f"Deep model outputs saved to: {MODEL_DIR}")
    print(f"Best feature mode: {best_result['feature_mode']}")
    print(f"Best F1: {best_result['metrics']['f1']}")


if __name__ == "__main__":
    main()
