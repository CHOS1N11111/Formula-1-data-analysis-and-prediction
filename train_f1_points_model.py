"""Train Top 10 and race-points models as the bridge toward 2026 championship prediction.

This script extends the current podium-prediction workflow from a single binary target
to points-related targets. It trains Top 10 classifiers and points regressors, evaluates
them on the 2025 season, writes prediction tables, and saves figures that can later feed
season-level driver and constructor championship simulations.
"""

import json
import math
from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train_f1_podium_model import (
    BASE_DIR,
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
    to_float,
    to_int,
    write_csv,
    write_json,
)


FIGURE_DIR = BASE_DIR / "outputs" / "figures"
SUMMARY_PATH = MODEL_DIR / "points_model_summary.json"

FEATURE_MODES = ["post_qualifying", "pre_race"]
TOP10_TARGET = "is_top10"
POINTS_TARGET = "points"

TOP10_METRIC_FIELDS = [
    "feature_mode",
    "model",
    "train_seasons",
    "test_season",
    "train_records",
    "test_records",
    "positive_train_records",
    "positive_test_records",
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
]

POINTS_METRIC_FIELDS = [
    "feature_mode",
    "model",
    "train_seasons",
    "test_season",
    "train_records",
    "test_records",
    "mae",
    "rmse",
    "r2",
    "mean_actual_points",
    "mean_predicted_points",
    "total_actual_points",
    "total_predicted_points",
]

POINTS_PREDICTION_FIELDS = [
    "season",
    "round",
    "race_name",
    "race_date",
    "driver_id",
    "driver_name",
    "constructor_id",
    "constructor_name",
    "grid",
    "qualifying_position",
    "top10_probability",
    "predicted_is_top10",
    "predicted_points",
    "actual_is_top10",
    "actual_points",
    "actual_finish_position",
]


def build_top10_models():
    """Create classifiers that estimate whether a driver finishes in the points."""
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("vectorizer", DictVectorizer(sparse=False)),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("vectorizer", DictVectorizer(sparse=False)),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=10,
                        min_samples_leaf=4,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("vectorizer", DictVectorizer(sparse=False)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=300,
                        learning_rate=0.04,
                        max_leaf_nodes=18,
                        l2_regularization=0.1,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def build_points_models():
    """Create regressors that estimate each driver's race points."""
    return {
        "ridge_regression": Pipeline(
            steps=[
                ("vectorizer", DictVectorizer(sparse=False)),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=5.0)),
            ]
        ),
        "random_forest_regressor": Pipeline(
            steps=[
                ("vectorizer", DictVectorizer(sparse=False)),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=500,
                        max_depth=10,
                        min_samples_leaf=4,
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "extra_trees_regressor": Pipeline(
            steps=[
                ("vectorizer", DictVectorizer(sparse=False)),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=500,
                        max_depth=10,
                        min_samples_leaf=4,
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting_regressor": Pipeline(
            steps=[
                ("vectorizer", DictVectorizer(sparse=False)),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=300,
                        learning_rate=0.04,
                        max_leaf_nodes=18,
                        l2_regularization=0.1,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def build_x(rows, feature_mode):
    """Build feature dictionaries for one prediction mode."""
    return [build_feature_dict(row, feature_mode) for row in rows]


def build_binary_y(rows, target_field):
    """Read a binary target column from row dictionaries."""
    return [to_int(row[target_field]) for row in rows]


def build_points_y(rows):
    """Read race points as a regression target."""
    return [to_float(row[POINTS_TARGET]) for row in rows]


def clamp_points(value):
    """Keep predicted points in a plausible non-negative Formula 1 range."""
    if math.isnan(value):
        return 0.0
    return max(0.0, min(30.0, value))


def evaluate_top10(y_true, probabilities, threshold=0.5):
    """Evaluate a Top 10 classifier from predicted probabilities."""
    predictions = [1 if probability >= threshold else 0 for probability in probabilities]
    return {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
    }


def evaluate_points(y_true, predictions):
    """Evaluate point predictions with regression metrics."""
    clipped_predictions = [clamp_points(value) for value in predictions]
    return {
        "mae": mean_absolute_error(y_true, clipped_predictions),
        "rmse": math.sqrt(mean_squared_error(y_true, clipped_predictions)),
        "r2": r2_score(y_true, clipped_predictions),
        "mean_actual_points": sum(y_true) / len(y_true),
        "mean_predicted_points": sum(clipped_predictions) / len(clipped_predictions),
        "total_actual_points": sum(y_true),
        "total_predicted_points": sum(clipped_predictions),
    }


def top10_metric_row(feature_mode, model_name, train_rows, test_rows, train_y, test_y, metrics):
    """Format one Top 10 classification result row."""
    return {
        "feature_mode": feature_mode,
        "model": model_name,
        "train_seasons": f"{TRAIN_START_SEASON}-{TRAIN_END_SEASON}",
        "test_season": TEST_SEASON,
        "train_records": len(train_rows),
        "test_records": len(test_rows),
        "positive_train_records": sum(train_y),
        "positive_test_records": sum(test_y),
        "accuracy": format_float(metrics["accuracy"]),
        "precision": format_float(metrics["precision"]),
        "recall": format_float(metrics["recall"]),
        "f1": format_float(metrics["f1"]),
        "roc_auc": format_float(metrics["roc_auc"]),
    }


def points_metric_row(feature_mode, model_name, train_rows, test_rows, metrics):
    """Format one race-points regression result row."""
    return {
        "feature_mode": feature_mode,
        "model": model_name,
        "train_seasons": f"{TRAIN_START_SEASON}-{TRAIN_END_SEASON}",
        "test_season": TEST_SEASON,
        "train_records": len(train_rows),
        "test_records": len(test_rows),
        "mae": format_float(metrics["mae"]),
        "rmse": format_float(metrics["rmse"]),
        "r2": format_float(metrics["r2"]),
        "mean_actual_points": format_float(metrics["mean_actual_points"]),
        "mean_predicted_points": format_float(metrics["mean_predicted_points"]),
        "total_actual_points": format_float(metrics["total_actual_points"]),
        "total_predicted_points": format_float(metrics["total_predicted_points"]),
    }


def train_top10_models(train_rows, test_rows, feature_mode):
    """Train all Top 10 classifiers for one feature mode."""
    train_x = build_x(train_rows, feature_mode)
    test_x = build_x(test_rows, feature_mode)
    train_y = build_binary_y(train_rows, TOP10_TARGET)
    test_y = build_binary_y(test_rows, TOP10_TARGET)

    metric_rows = []
    fitted_models = {}
    for model_name, model in build_top10_models().items():
        model.fit(train_x, train_y)
        probabilities = model.predict_proba(test_x)[:, 1]
        metrics = evaluate_top10(test_y, probabilities)
        metric_rows.append(
            top10_metric_row(
                feature_mode,
                model_name,
                train_rows,
                test_rows,
                train_y,
                test_y,
                metrics,
            )
        )
        fitted_models[model_name] = {
            "model": model,
            "probabilities": probabilities,
            "metrics": metrics,
        }

    best_row = max(metric_rows, key=lambda row: to_float(row["f1"]))
    return metric_rows, fitted_models, best_row


def train_points_models(train_rows, test_rows, feature_mode):
    """Train all race-points regressors for one feature mode."""
    train_x = build_x(train_rows, feature_mode)
    test_x = build_x(test_rows, feature_mode)
    train_y = build_points_y(train_rows)
    test_y = build_points_y(test_rows)

    metric_rows = []
    fitted_models = {}
    for model_name, model in build_points_models().items():
        model.fit(train_x, train_y)
        predictions = [clamp_points(value) for value in model.predict(test_x)]
        metrics = evaluate_points(test_y, predictions)
        metric_rows.append(
            points_metric_row(feature_mode, model_name, train_rows, test_rows, metrics)
        )
        fitted_models[model_name] = {
            "model": model,
            "predictions": predictions,
            "metrics": metrics,
        }

    best_row = min(metric_rows, key=lambda row: to_float(row["mae"]))
    return metric_rows, fitted_models, best_row


def build_prediction_rows(test_rows, top10_probabilities, points_predictions):
    """Create a combined Top 10 probability and expected-points prediction table."""
    output_rows = []
    for row, top10_probability, predicted_points in zip(
        test_rows, top10_probabilities, points_predictions
    ):
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
                "grid": row["grid"],
                "qualifying_position": row["qualifying_position"],
                "top10_probability": format_float(top10_probability),
                "predicted_is_top10": 1 if top10_probability >= 0.5 else 0,
                "predicted_points": format_float(clamp_points(predicted_points)),
                "actual_is_top10": row["is_top10"],
                "actual_points": row["points"],
                "actual_finish_position": row["finish_position"],
            }
        )
    return sorted(
        output_rows,
        key=lambda item: (
            to_int(item["season"]),
            to_int(item["round"]),
            -to_float(item["predicted_points"]),
        ),
    )


def save_top10_chart(rows):
    """Save a chart comparing Top 10 model F1 scores."""
    sorted_rows = sorted(rows, key=lambda row: to_float(row["f1"]), reverse=True)
    labels = [f"{row['feature_mode']}\n{row['model']}" for row in sorted_rows]
    values = [to_float(row["f1"]) for row in sorted_rows]
    colors = ["#2563EB" if row["feature_mode"] == "post_qualifying" else "#059669" for row in sorted_rows]

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    bars = ax.bar(range(len(sorted_rows)), values, color=colors)
    ax.set_xticks(range(len(sorted_rows)), labels=labels, rotation=35, ha="right")
    ax.set_ylim(0, max(values) * 1.15)
    ax.set_ylabel("F1 score")
    ax.set_title("Top 10 Classification Model Comparison, 2025")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    output_path = FIGURE_DIR / "top10_model_comparison_2025.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_points_chart(rows):
    """Save a chart comparing points-regression MAE values."""
    sorted_rows = sorted(rows, key=lambda row: to_float(row["mae"]))
    labels = [f"{row['feature_mode']}\n{row['model']}" for row in sorted_rows]
    values = [to_float(row["mae"]) for row in sorted_rows]
    colors = ["#7C3AED" if row["feature_mode"] == "post_qualifying" else "#0891B2" for row in sorted_rows]

    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    bars = ax.bar(range(len(sorted_rows)), values, color=colors)
    ax.set_xticks(range(len(sorted_rows)), labels=labels, rotation=35, ha="right")
    ax.set_ylabel("Mean absolute error")
    ax.set_title("Race Points Regression Model Comparison, 2025")
    ax.set_ylim(0, max(values) * 1.2)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.03, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    output_path = FIGURE_DIR / "points_model_comparison_2025.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_actual_vs_predicted_chart(rows):
    """Save a scatter plot comparing actual and predicted race points."""
    actual = [to_float(row["actual_points"]) for row in rows]
    predicted = [to_float(row["predicted_points"]) for row in rows]
    max_value = max(actual + predicted) if rows else 1.0

    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    ax.scatter(actual, predicted, alpha=0.65, s=28, color="#2563EB", edgecolor="white", linewidth=0.4)
    ax.plot([0, max_value], [0, max_value], color="#DC2626", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Actual points")
    ax.set_ylabel("Predicted points")
    ax.set_title("Actual vs Predicted Race Points, 2025")
    ax.grid(True, color="#E5E7EB", linewidth=0.8)
    fig.tight_layout()
    output_path = FIGURE_DIR / "points_actual_vs_predicted_2025.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main():
    """Run Top 10 and points prediction training, evaluation, and output writing."""
    training_features_path = get_training_features_path()
    rows = add_circuit_history_features(read_csv(training_features_path))
    train_rows, test_rows, final_train_rows, completed_2026_rows = split_rows(rows)

    top10_metric_rows = []
    points_metric_rows = []
    top10_fitted_by_mode = {}
    points_fitted_by_mode = {}
    top10_best_rows = []
    points_best_rows = []

    for feature_mode in FEATURE_MODES:
        top10_metrics, top10_fitted, top10_best = train_top10_models(
            train_rows, test_rows, feature_mode
        )
        points_metrics, points_fitted, points_best = train_points_models(
            train_rows, test_rows, feature_mode
        )
        top10_metric_rows.extend(top10_metrics)
        points_metric_rows.extend(points_metrics)
        top10_fitted_by_mode[feature_mode] = top10_fitted
        points_fitted_by_mode[feature_mode] = points_fitted
        top10_best_rows.append(top10_best)
        points_best_rows.append(points_best)

    best_top10_row = max(top10_best_rows, key=lambda row: to_float(row["f1"]))
    best_points_row = min(points_best_rows, key=lambda row: to_float(row["mae"]))
    best_top10_info = top10_fitted_by_mode[best_top10_row["feature_mode"]][best_top10_row["model"]]
    best_points_info = points_fitted_by_mode[best_points_row["feature_mode"]][best_points_row["model"]]

    prediction_rows = build_prediction_rows(
        test_rows,
        best_top10_info["probabilities"],
        best_points_info["predictions"],
    )

    final_top10_model = build_top10_models()[best_top10_row["model"]]
    final_points_model = build_points_models()[best_points_row["model"]]
    final_top10_x = build_x(final_train_rows, best_top10_row["feature_mode"])
    final_top10_y = build_binary_y(final_train_rows, TOP10_TARGET)
    final_points_x = build_x(final_train_rows, best_points_row["feature_mode"])
    final_points_y = build_points_y(final_train_rows)
    final_top10_model.fit(final_top10_x, final_top10_y)
    final_points_model.fit(final_points_x, final_points_y)

    completed_2026_predictions = []
    if completed_2026_rows:
        completed_top10_x = build_x(completed_2026_rows, best_top10_row["feature_mode"])
        completed_points_x = build_x(completed_2026_rows, best_points_row["feature_mode"])
        completed_2026_predictions = build_prediction_rows(
            completed_2026_rows,
            final_top10_model.predict_proba(completed_top10_x)[:, 1],
            [clamp_points(value) for value in final_points_model.predict(completed_points_x)],
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(MODEL_DIR / "top10_model_metrics.csv", TOP10_METRIC_FIELDS, top10_metric_rows)
    write_csv(MODEL_DIR / "points_model_metrics.csv", POINTS_METRIC_FIELDS, points_metric_rows)
    write_csv(
        MODEL_DIR / "points_predictions_2025.csv",
        POINTS_PREDICTION_FIELDS,
        prediction_rows,
    )
    write_csv(
        MODEL_DIR / "points_completed_2026_predictions.csv",
        POINTS_PREDICTION_FIELDS,
        completed_2026_predictions,
    )

    top10_chart = save_top10_chart(top10_metric_rows)
    points_chart = save_points_chart(points_metric_rows)
    actual_vs_predicted_chart = save_actual_vs_predicted_chart(prediction_rows)

    summary = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "input_features": str(training_features_path.relative_to(BASE_DIR)),
        "train_seasons": f"{TRAIN_START_SEASON}-{TRAIN_END_SEASON}",
        "test_season": TEST_SEASON,
        "feature_modes": FEATURE_MODES,
        "top10_target": TOP10_TARGET,
        "points_target": POINTS_TARGET,
        "best_top10_model": {
            "feature_mode": best_top10_row["feature_mode"],
            "model": best_top10_row["model"],
            "f1": best_top10_row["f1"],
            "roc_auc": best_top10_row["roc_auc"],
        },
        "best_points_model": {
            "feature_mode": best_points_row["feature_mode"],
            "model": best_points_row["model"],
            "mae": best_points_row["mae"],
            "rmse": best_points_row["rmse"],
            "r2": best_points_row["r2"],
        },
        "outputs": [
            "top10_model_metrics.csv",
            "points_model_metrics.csv",
            "points_predictions_2025.csv",
            "points_completed_2026_predictions.csv",
        ],
        "figures": [
            str(top10_chart.relative_to(BASE_DIR)),
            str(points_chart.relative_to(BASE_DIR)),
            str(actual_vs_predicted_chart.relative_to(BASE_DIR)),
        ],
        "note": "These models estimate Top 10 probability and expected race points. They are intended as the next input layer for 2026 season-level driver and constructor championship simulation.",
    }
    write_json(SUMMARY_PATH, summary)

    print(f"Top10 and points model outputs saved to: {MODEL_DIR}")
    print(f"Best Top10 model: {best_top10_row['feature_mode']} / {best_top10_row['model']} / F1={best_top10_row['f1']}")
    print(f"Best points model: {best_points_row['feature_mode']} / {best_points_row['model']} / MAE={best_points_row['mae']}")


if __name__ == "__main__":
    main()
