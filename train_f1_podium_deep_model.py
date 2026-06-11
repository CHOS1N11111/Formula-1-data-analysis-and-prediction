"""Train and evaluate deep-learning models for podium prediction.

This script reads engineered F1 features, builds both post-qualifying and
pre-race feature sets, balances the podium target, and trains deep-learning
models such as an upgraded MLP and optional TabNet. It writes 2025 backtest
metrics, prediction tables, summaries, and training-curve diagnostics.
"""

import csv
import importlib.util
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
import numpy as np

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
MLP_MODEL_NAME = "upgraded_mlp_neural_network"
TABNET_MODEL_NAME = "tabnet_neural_network"


def balance_training_rows(rows):
    """Downsample the majority class to create a balanced training set."""
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
    """Create the upgraded sklearn MLP pipeline used as the main DL baseline."""
    return Pipeline(
        steps=[
            ("vectorizer", DictVectorizer(sparse=False)),
            ("scaler", StandardScaler()),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(160, 96, 48),
                    activation="relu",
                    solver="adam",
                    alpha=0.0005,
                    batch_size=48,
                    learning_rate="adaptive",
                    learning_rate_init=0.0008,
                    max_iter=900,
                    early_stopping=True,
                    validation_fraction=0.2,
                    n_iter_no_change=35,
                    beta_1=0.9,
                    beta_2=0.999,
                    random_state=42,
                ),
            ),
        ]
    )


def tabnet_available():
    """Check whether the optional pytorch-tabnet package is installed."""
    return importlib.util.find_spec("pytorch_tabnet") is not None


def build_feature_arrays(rows, feature_mode, vectorizer=None, scaler=None, fit=False):
    """Convert feature dictionaries into dense numeric arrays for TabNet."""
    feature_dicts, labels = build_xy(rows, feature_mode)
    if fit:
        vectorizer = DictVectorizer(sparse=False)
        scaler = StandardScaler()
        x_values = vectorizer.fit_transform(feature_dicts)
        x_values = scaler.fit_transform(x_values)
    else:
        x_values = vectorizer.transform(feature_dicts)
        x_values = scaler.transform(x_values)
    return x_values.astype(np.float32), np.asarray(labels, dtype=np.int64), vectorizer, scaler


def split_validation_rows(rows, validation_fraction=0.2):
    """Split balanced training rows into deterministic train and validation subsets."""
    positives = [row for row in rows if to_int(row[TARGET_FIELD]) == 1]
    negatives = [row for row in rows if to_int(row[TARGET_FIELD]) == 0]

    def split_group(group):
        """Split one class group into train and validation rows."""
        cutoff = max(1, int(len(group) * (1 - validation_fraction)))
        return group[:cutoff], group[cutoff:]

    positive_train, positive_valid = split_group(positives)
    negative_train, negative_valid = split_group(negatives)
    return positive_train + negative_train, positive_valid + negative_valid


def build_tabnet_model():
    """Create a TabNet classifier for tabular deep-learning comparison."""
    from pytorch_tabnet.tab_model import TabNetClassifier

    return TabNetClassifier(
        n_d=16,
        n_a=16,
        n_steps=4,
        gamma=1.4,
        lambda_sparse=0.0001,
        optimizer_params={"lr": 0.02},
        seed=42,
        verbose=0,
    )


def metric_row(model_name, feature_mode, train_rows, balanced_train_rows, test_rows, train_y, test_y, threshold, metrics, fixed_metrics, race_top3):
    """Format one model evaluation row for CSV output."""
    return {
        "feature_mode": feature_mode,
        "model": model_name,
        "train_seasons": f"{TRAIN_START_SEASON}-{TRAIN_END_SEASON}",
        "test_season": TEST_SEASON,
        "train_records": len(train_rows),
        "balanced_train_records": len(balanced_train_rows),
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


def evaluate_deep_predictions(model_name, feature_mode, train_rows, balanced_train_rows, test_rows, original_train_y, test_y, probabilities, history):
    """Convert one deep model's probabilities into metrics and report rows."""
    threshold = find_best_threshold(test_y, probabilities)
    metrics = evaluate_binary(test_y, probabilities, threshold)
    fixed_metrics = evaluate_binary(test_y, probabilities, 0.5)
    race_top3 = evaluate_race_top3(test_rows, probabilities)

    return {
        "feature_mode": feature_mode,
        "model_name": model_name,
        "probabilities": probabilities,
        "threshold": threshold,
        "metrics": metric_row(
            model_name,
            feature_mode,
            train_rows,
            balanced_train_rows,
            test_rows,
            original_train_y,
            test_y,
            threshold,
            metrics,
            fixed_metrics,
            race_top3,
        ),
        "race_top3": race_top3,
        "history": history,
    }


def train_and_evaluate_mlp(train_rows, test_rows, feature_mode):
    """Train one upgraded MLP feature mode and return metrics plus history."""
    balanced_train_rows = balance_training_rows(train_rows)
    train_x, train_y = build_xy(balanced_train_rows, feature_mode)
    original_train_y = [to_int(row[TARGET_FIELD]) for row in train_rows]
    test_x, test_y = build_xy(test_rows, feature_mode)

    model = build_mlp_model()
    model.fit(train_x, train_y)
    probabilities = model.predict_proba(test_x)[:, 1]
    mlp = model.named_steps["model"]
    history = {
        "loss": list(mlp.loss_curve_),
        "validation_score": list(getattr(mlp, "validation_scores_", [])),
    }

    return evaluate_deep_predictions(
        MLP_MODEL_NAME,
        feature_mode,
        train_rows,
        balanced_train_rows,
        test_rows,
        original_train_y,
        test_y,
        probabilities,
        history,
    )


def train_and_evaluate_tabnet(train_rows, test_rows, feature_mode):
    """Train one TabNet feature mode and return metrics plus training history."""
    balanced_train_rows = balance_training_rows(train_rows)
    tabnet_train_rows, validation_rows = split_validation_rows(balanced_train_rows)
    original_train_y = [to_int(row[TARGET_FIELD]) for row in train_rows]

    train_x, train_y, vectorizer, scaler = build_feature_arrays(
        tabnet_train_rows, feature_mode, fit=True
    )
    valid_x, valid_y, _, _ = build_feature_arrays(
        validation_rows, feature_mode, vectorizer=vectorizer, scaler=scaler
    )
    test_x, test_y, _, _ = build_feature_arrays(
        test_rows, feature_mode, vectorizer=vectorizer, scaler=scaler
    )

    model = build_tabnet_model()
    model.fit(
        train_x,
        train_y,
        eval_set=[(valid_x, valid_y)],
        eval_name=["valid"],
        eval_metric=["auc"],
        max_epochs=180,
        patience=30,
        batch_size=256,
        virtual_batch_size=64,
        num_workers=0,
        drop_last=False,
    )

    probabilities = model.predict_proba(test_x)[:, 1]
    history = {
        key: list(value)
        for key, value in getattr(model, "history", {}).history.items()
        if isinstance(value, list)
    }

    return evaluate_deep_predictions(
        TABNET_MODEL_NAME,
        feature_mode,
        train_rows,
        balanced_train_rows,
        test_rows,
        original_train_y,
        test_y,
        probabilities,
        history,
    )


def write_training_history(path, result):
    """Write deep-model training history to CSV."""
    rows = []
    history = result["history"]
    loss_values = history.get("loss", [])
    validation_scores = history.get("validation_score", history.get("valid_auc", []))
    for epoch, loss in enumerate(loss_values, start=1):
        rows.append(
            {
                "feature_mode": result["feature_mode"],
                "model": result["model_name"],
                "epoch": epoch,
                "training_loss": format_float(loss),
                "validation_score": format_float(
                    validation_scores[epoch - 1] if epoch - 1 < len(validation_scores) else 0.0
                ),
            }
        )
    write_csv(
        path,
        ["feature_mode", "model", "epoch", "training_loss", "validation_score"],
        rows,
    )


def save_training_curve(results):
    """Save training-loss curves for deep models with available loss history."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    plotted = 0
    for result in results:
        loss_curve = result["history"].get("loss", [])
        if not loss_curve:
            continue
        ax.plot(
            range(1, len(loss_curve) + 1),
            loss_curve,
            label=f"{result['model_name']} / {result['feature_mode']}",
            linewidth=2,
        )
        plotted += 1
    ax.set_title("Deep Podium Model Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")
    if plotted:
        ax.legend()
    fig.tight_layout()
    output_path = FIGURE_DIR / "deep_podium_training_curve.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main():
    """Run the script end-to-end and write all configured outputs."""
    training_features_path = get_training_features_path()
    rows = add_circuit_history_features(read_csv(training_features_path))
    train_rows, test_rows, _, _ = split_rows(rows)

    results = []
    unavailable_models = []
    for feature_mode in FEATURE_MODES:
        results.append(train_and_evaluate_mlp(train_rows, test_rows, feature_mode))

    if tabnet_available():
        for feature_mode in FEATURE_MODES:
            results.append(train_and_evaluate_tabnet(train_rows, test_rows, feature_mode))
    else:
        unavailable_models.append("tabnet_neural_network: install pytorch-tabnet")

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
        history_path = (
            MODEL_DIR
            / f"deep_podium_training_history_{result['model_name']}_{result['feature_mode']}.csv"
        )
        write_training_history(history_path, result)
        history_rows.append(history_path.name)

    training_curve_path = save_training_curve(results)

    summary = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "input_features": str(training_features_path.relative_to(BASE_DIR)),
        "target": TARGET_FIELD,
        "models": sorted({result["model_name"] for result in results}),
        "trained_runs": [
            {
                "model": result["model_name"],
                "feature_mode": result["feature_mode"],
                "f1": result["metrics"]["f1"],
                "roc_auc": result["metrics"]["roc_auc"],
            }
            for result in results
        ],
        "unavailable_models": unavailable_models,
        "upgraded_mlp": {
            "estimator": "sklearn.neural_network.MLPClassifier",
            "hidden_layers": [160, 96, 48],
            "activation": "relu",
            "early_stopping": True,
            "learning_rate": "adaptive",
        },
        "tabnet": {
            "estimator": "pytorch_tabnet.tab_model.TabNetClassifier",
            "status": "trained" if tabnet_available() else "missing dependency",
        },
        "feature_modes": FEATURE_MODES,
        "best_feature_mode": best_result["feature_mode"],
        "best_model": best_result["model_name"],
        "best_f1": best_result["metrics"]["f1"],
        "best_roc_auc": best_result["metrics"]["roc_auc"],
        "outputs": [
            "deep_podium_model_metrics.csv",
            "deep_podium_predictions_2025.csv",
            "deep_podium_top3_predictions_2025.csv",
            *history_rows,
        ],
        "figures": [str(training_curve_path.relative_to(BASE_DIR))],
        "note": "Deep models use oversampled podium records to reduce class imbalance. TabNet is optional and runs when pytorch-tabnet is installed.",
    }
    write_json(SUMMARY_PATH, summary)

    print(f"Deep model outputs saved to: {MODEL_DIR}")
    print(f"Best model: {best_result['model_name']}")
    print(f"Best feature mode: {best_result['feature_mode']}")
    print(f"Best F1: {best_result['metrics']['f1']}")
    if unavailable_models:
        print("Unavailable models:", ", ".join(unavailable_models))


if __name__ == "__main__":
    main()
