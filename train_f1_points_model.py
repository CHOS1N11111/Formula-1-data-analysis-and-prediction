"""Train Top 10 and race-points models as the bridge toward 2026 championship prediction.

This script extends the current podium-prediction workflow from a single binary target
to points-related targets. It trains traditional ML, advanced boosting, and deep-learning
models for Top 10 classification and race-points regression, evaluates them on the 2025
season, writes prediction tables, and saves figures that can later feed season-level
driver and constructor championship simulations.
"""

import importlib.util
import json
import math
from datetime import datetime, timezone

import matplotlib
import numpy as np

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
from sklearn.neural_network import MLPClassifier, MLPRegressor
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
from sklearn.compose import TransformedTargetRegressor
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
POINTS_TARGET = "current_rule_points"
F1_POINTS_TABLE = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]

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

TOP10_RACE_LEVEL_FIELDS = [
    "season",
    "round",
    "race_name",
    "predicted_top10_count",
    "actual_top10_count",
    "top10_hits",
    "precision_at_10",
    "recall_at_10",
    "actual_points_in_predicted_top10",
    "actual_points_total",
    "points_capture_rate",
    "exact_top10_set_hit",
]

POINTS_ERROR_BIN_FIELDS = [
    "points_bin",
    "bin_sort",
    "records",
    "mae",
    "rmse",
    "mean_actual_points",
    "mean_predicted_points",
    "total_actual_points",
    "total_predicted_points",
]

RULE_MAPPED_POINTS_FIELDS = [
    "season",
    "round",
    "race_name",
    "driver_id",
    "driver_name",
    "constructor_name",
    "predicted_points",
    "rule_mapped_points",
    "actual_points",
    "actual_finish_position",
]

RULE_MAPPED_POINTS_SUMMARY_FIELDS = [
    "metric",
    "continuous_points",
    "rule_mapped_points",
]

TOP10_CALIBRATION_FIELDS = [
    "probability_bin",
    "bin_sort",
    "records",
    "mean_predicted_probability",
    "actual_top10_rate",
    "calibration_error",
]

FEATURE_IMPORTANCE_FIELDS = [
    "target",
    "feature_mode",
    "model",
    "feature",
    "importance_type",
    "importance",
    "abs_importance",
]

ROLLING_BACKTEST_FIELDS = [
    "task",
    "feature_mode",
    "model",
    "train_seasons",
    "test_season",
    "records",
    "primary_metric",
    "primary_value",
    "secondary_metric",
    "secondary_value",
]

POINTS_RACE_LEVEL_FIELDS = [
    "season",
    "round",
    "race_name",
    "records",
    "continuous_mae",
    "rule_mapped_mae",
    "continuous_total_predicted_points",
    "rule_mapped_total_predicted_points",
    "actual_total_points",
    "top10_hits",
    "top10_precision",
]

POINTS_CORRECTION_FIELDS = [
    "feature_mode",
    "model",
    "correction",
    "mae",
    "rmse",
    "r2",
    "mean_actual_points",
    "mean_predicted_points",
    "total_actual_points",
    "total_predicted_points",
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


def package_available(package_name):
    """Return True when an optional model package is installed."""
    return importlib.util.find_spec(package_name) is not None


def current_rule_points(row):
    """Return current Grand Prix race points from finish position only.

    The project intentionally ignores Sprint points and fastest-lap bonus points.
    Every race is treated as a normal full-points Grand Prix with ten scoring places.
    This keeps historical training labels on the same rules as the 2025/2026 target.
    """
    finish_position = to_int(row.get("finish_position", "0"))
    if 1 <= finish_position <= len(F1_POINTS_TABLE):
        return float(F1_POINTS_TABLE[finish_position - 1])
    return 0.0


def actual_points(row):
    """Return the normalized points target used by this project."""
    return current_rule_points(row)


class TabNetClassifierPipeline:
    """Small sklearn-like wrapper around TabNetClassifier for dictionary features."""

    def __init__(self):
        """Create vectorizer and scaler placeholders for TabNet classification."""
        self.vectorizer = DictVectorizer(sparse=False)
        self.scaler = StandardScaler()
        self.model = None

    def fit(self, x_rows, y_values):
        """Fit TabNetClassifier on vectorized and scaled feature dictionaries."""
        from pytorch_tabnet.tab_model import TabNetClassifier

        x_values = self.vectorizer.fit_transform(x_rows).astype(np.float32)
        x_values = self.scaler.fit_transform(x_values).astype(np.float32)
        y_array = np.asarray(y_values, dtype=np.int64)
        self.model = TabNetClassifier(
            n_d=16,
            n_a=16,
            n_steps=4,
            gamma=1.4,
            lambda_sparse=0.0001,
            optimizer_params={"lr": 0.02},
            seed=42,
            verbose=0,
        )
        self.model.fit(
            x_values,
            y_array,
            max_epochs=90,
            patience=20,
            batch_size=256,
            virtual_batch_size=64,
            num_workers=0,
            drop_last=False,
        )
        return self

    def predict_proba(self, x_rows):
        """Return class probabilities for vectorized feature dictionaries."""
        x_values = self.vectorizer.transform(x_rows).astype(np.float32)
        x_values = self.scaler.transform(x_values).astype(np.float32)
        return self.model.predict_proba(x_values)


class TabNetRegressorPipeline:
    """Small sklearn-like wrapper around TabNetRegressor for dictionary features."""

    def __init__(self):
        """Create vectorizer and scaler placeholders for TabNet regression."""
        self.vectorizer = DictVectorizer(sparse=False)
        self.scaler = StandardScaler()
        self.model = None

    def fit(self, x_rows, y_values):
        """Fit TabNetRegressor on vectorized and scaled feature dictionaries."""
        from pytorch_tabnet.tab_model import TabNetRegressor

        x_values = self.vectorizer.fit_transform(x_rows).astype(np.float32)
        x_values = self.scaler.fit_transform(x_values).astype(np.float32)
        y_array = np.asarray(y_values, dtype=np.float32).reshape(-1, 1)
        self.model = TabNetRegressor(
            n_d=16,
            n_a=16,
            n_steps=4,
            gamma=1.4,
            lambda_sparse=0.0001,
            optimizer_params={"lr": 0.02},
            seed=42,
            verbose=0,
        )
        self.model.fit(
            x_values,
            y_array,
            max_epochs=90,
            patience=20,
            batch_size=256,
            virtual_batch_size=64,
            num_workers=0,
            drop_last=False,
        )
        return self

    def predict(self, x_rows):
        """Return point predictions for vectorized feature dictionaries."""
        x_values = self.vectorizer.transform(x_rows).astype(np.float32)
        x_values = self.scaler.transform(x_values).astype(np.float32)
        return self.model.predict(x_values).reshape(-1)


def build_top10_models():
    """Create classifiers that estimate whether a driver finishes in the points."""
    models = {
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
    if package_available("catboost"):
        from catboost import CatBoostClassifier

        models["catboost_classifier"] = Pipeline(
            steps=[
                ("vectorizer", DictVectorizer(sparse=False)),
                (
                    "model",
                    CatBoostClassifier(
                        iterations=250,
                        depth=5,
                        learning_rate=0.04,
                        loss_function="Logloss",
                        eval_metric="AUC",
                        random_seed=42,
                        verbose=False,
                    ),
                ),
            ]
        )
    if package_available("lightgbm"):
        from lightgbm import LGBMClassifier

        models["lightgbm_classifier"] = Pipeline(
            steps=[
                ("vectorizer", DictVectorizer(sparse=False)),
                (
                    "model",
                    LGBMClassifier(
                        n_estimators=250,
                        max_depth=5,
                        learning_rate=0.04,
                        class_weight="balanced",
                        random_state=42,
                        verbose=-1,
                    ),
                ),
            ]
        )
    if package_available("xgboost"):
        from xgboost import XGBClassifier

        models["xgboost_classifier"] = Pipeline(
            steps=[
                ("vectorizer", DictVectorizer(sparse=False)),
                (
                    "model",
                    XGBClassifier(
                        n_estimators=250,
                        max_depth=5,
                        learning_rate=0.04,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        eval_metric="logloss",
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        )
    models["mlp_classifier"] = Pipeline(
        steps=[
            ("vectorizer", DictVectorizer(sparse=False)),
            ("scaler", StandardScaler()),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(128, 64, 32),
                    activation="relu",
                    learning_rate="adaptive",
                    learning_rate_init=0.0008,
                    alpha=0.0005,
                    batch_size=64,
                    max_iter=600,
                    early_stopping=True,
                    validation_fraction=0.2,
                    n_iter_no_change=30,
                    random_state=42,
                ),
            ),
        ]
    )
    if package_available("pytorch_tabnet"):
        models["tabnet_classifier"] = TabNetClassifierPipeline()
    return models


def build_points_models():
    """Create regressors that estimate each driver's race points."""
    models = {
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
    if package_available("catboost"):
        from catboost import CatBoostRegressor

        models["catboost_regressor"] = Pipeline(
            steps=[
                ("vectorizer", DictVectorizer(sparse=False)),
                (
                    "model",
                    CatBoostRegressor(
                        iterations=250,
                        depth=5,
                        learning_rate=0.04,
                        loss_function="RMSE",
                        random_seed=42,
                        verbose=False,
                    ),
                ),
            ]
        )
    if package_available("lightgbm"):
        from lightgbm import LGBMRegressor

        models["lightgbm_regressor"] = Pipeline(
            steps=[
                ("vectorizer", DictVectorizer(sparse=False)),
                (
                    "model",
                    LGBMRegressor(
                        n_estimators=250,
                        max_depth=5,
                        learning_rate=0.04,
                        random_state=42,
                        verbose=-1,
                    ),
                ),
            ]
        )
    if package_available("xgboost"):
        from xgboost import XGBRegressor

        models["xgboost_regressor"] = Pipeline(
            steps=[
                ("vectorizer", DictVectorizer(sparse=False)),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=250,
                        max_depth=5,
                        learning_rate=0.04,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        objective="reg:squarederror",
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        )
    models["mlp_regressor"] = Pipeline(
        steps=[
            ("vectorizer", DictVectorizer(sparse=False)),
            ("scaler", StandardScaler()),
            (
                "model",
                MLPRegressor(
                    hidden_layer_sizes=(128, 64, 32),
                    activation="relu",
                    learning_rate="adaptive",
                    learning_rate_init=0.0008,
                    alpha=0.0005,
                    batch_size=64,
                    max_iter=700,
                    early_stopping=True,
                    validation_fraction=0.2,
                    n_iter_no_change=30,
                    random_state=42,
                ),
            ),
        ]
    )
    if package_available("pytorch_tabnet"):
        models["tabnet_regressor"] = TabNetRegressorPipeline()
    return models


def build_top10_rolling_models():
    """Create a compact Top 10 model set for rolling backtests."""
    return {
        name: model
        for name, model in build_top10_models().items()
        if name == "lightgbm_classifier"
    }


def build_points_rolling_models():
    """Create a compact points model set for rolling backtests."""
    return {
        name: model
        for name, model in build_points_models().items()
        if name == "catboost_regressor"
    }


def build_points_correction_models():
    """Create experimental points models for high-score underestimation checks."""
    base_models = build_points_models()
    models = {}
    if "catboost_regressor" in base_models:
        models["baseline_catboost"] = base_models["catboost_regressor"]
        weighted_model = build_points_models()["catboost_regressor"]
        if hasattr(weighted_model, "named_steps"):
            estimator = weighted_model.named_steps.get("model")
            if estimator is not None and hasattr(estimator, "set_params"):
                estimator.set_params(loss_function="RMSE")
        models["weighted_high_points_catboost"] = weighted_model
        sqrt_model = build_points_models()["catboost_regressor"]
        models["sqrt_target_catboost"] = TransformedTargetRegressor(
            regressor=sqrt_model,
            func=np.sqrt,
            inverse_func=lambda values: np.square(values),
        )
    return models


def build_x(rows, feature_mode):
    """Build feature dictionaries for one prediction mode."""
    return [build_feature_dict(row, feature_mode) for row in rows]


def build_binary_y(rows, target_field):
    """Read a binary target column from row dictionaries."""
    return [to_int(row[target_field]) for row in rows]


def build_points_y(rows):
    """Build the normalized current-rules race-points regression target."""
    return [actual_points(row) for row in rows]


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
        "brier_score": sum(
            (probability - actual) ** 2
            for probability, actual in zip(probabilities, y_true)
        )
        / len(y_true),
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
                "actual_points": format_float(actual_points(row)),
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


def build_top10_race_level_rows(prediction_rows):
    """Evaluate Top 10 predictions by selecting the ten highest probabilities per race."""
    grouped = {}
    for row in prediction_rows:
        key = (row["season"], row["round"], row["race_name"])
        grouped.setdefault(key, []).append(row)

    output_rows = []
    for (season, round_number, race_name), race_rows in sorted(
        grouped.items(), key=lambda item: (to_int(item[0][0]), to_int(item[0][1]))
    ):
        predicted_top10 = sorted(
            race_rows,
            key=lambda row: to_float(row["top10_probability"]),
            reverse=True,
        )[:10]
        predicted_driver_ids = {row["driver_id"] for row in predicted_top10}
        actual_top10 = [row for row in race_rows if to_int(row["actual_is_top10"]) == 1]
        actual_driver_ids = {row["driver_id"] for row in actual_top10}
        hits = len(predicted_driver_ids & actual_driver_ids)
        actual_points_total = sum(to_float(row["actual_points"]) for row in race_rows)
        captured_points = sum(
            to_float(row["actual_points"]) for row in predicted_top10
        )

        output_rows.append(
            {
                "season": season,
                "round": round_number,
                "race_name": race_name,
                "predicted_top10_count": len(predicted_top10),
                "actual_top10_count": len(actual_top10),
                "top10_hits": hits,
                "precision_at_10": format_float(hits / len(predicted_top10)),
                "recall_at_10": format_float(
                    hits / len(actual_top10) if actual_top10 else 0.0
                ),
                "actual_points_in_predicted_top10": format_float(captured_points),
                "actual_points_total": format_float(actual_points_total),
                "points_capture_rate": format_float(
                    captured_points / actual_points_total
                    if actual_points_total
                    else 0.0
                ),
                "exact_top10_set_hit": 1 if predicted_driver_ids == actual_driver_ids else 0,
            }
        )
    return output_rows


def summarize_top10_race_level(race_level_rows):
    """Summarize race-level Top 10 selection quality across the test season."""
    if not race_level_rows:
        return {}
    return {
        "race_count": len(race_level_rows),
        "mean_precision_at_10": format_float(
            sum(to_float(row["precision_at_10"]) for row in race_level_rows)
            / len(race_level_rows)
        ),
        "mean_recall_at_10": format_float(
            sum(to_float(row["recall_at_10"]) for row in race_level_rows)
            / len(race_level_rows)
        ),
        "mean_points_capture_rate": format_float(
            sum(to_float(row["points_capture_rate"]) for row in race_level_rows)
            / len(race_level_rows)
        ),
        "exact_top10_set_rate": format_float(
            sum(to_int(row["exact_top10_set_hit"]) for row in race_level_rows)
            / len(race_level_rows)
        ),
    }


def points_bin(actual_points):
    """Assign actual race points to bins for regression error analysis."""
    value = to_float(actual_points)
    if value == 0:
        return "0", 0
    if value <= 5:
        return "1-5", 1
    if value <= 10:
        return "6-10", 2
    if value <= 15:
        return "11-15", 3
    return "16+", 4


def probability_bin(probability):
    """Assign a predicted probability to a calibration bin."""
    value = max(0.0, min(1.0, to_float(probability)))
    bin_index = min(9, int(value * 10))
    lower = bin_index / 10
    upper = 1.0 if bin_index == 9 else (bin_index + 1) / 10
    return f"{lower:.1f}-{upper:.1f}", bin_index


def build_top10_calibration_rows(prediction_rows):
    """Build calibration bins for Top 10 predicted probabilities."""
    grouped = {}
    for row in prediction_rows:
        label, sort_key = probability_bin(row["top10_probability"])
        grouped.setdefault((label, sort_key), []).append(row)

    output_rows = []
    for (label, sort_key), rows in sorted(grouped.items(), key=lambda item: item[0][1]):
        predicted = [to_float(row["top10_probability"]) for row in rows]
        actual = [to_int(row["actual_is_top10"]) for row in rows]
        mean_predicted = sum(predicted) / len(predicted)
        actual_rate = sum(actual) / len(actual)
        output_rows.append(
            {
                "probability_bin": label,
                "bin_sort": sort_key,
                "records": len(rows),
                "mean_predicted_probability": format_float(mean_predicted),
                "actual_top10_rate": format_float(actual_rate),
                "calibration_error": format_float(abs(mean_predicted - actual_rate)),
            }
        )
    return output_rows


def summarize_top10_calibration(prediction_rows, calibration_rows):
    """Summarize Brier score and expected calibration error for Top 10 probabilities."""
    if not prediction_rows:
        return {}
    brier = sum(
        (to_float(row["top10_probability"]) - to_int(row["actual_is_top10"])) ** 2
        for row in prediction_rows
    ) / len(prediction_rows)
    weighted_error = sum(
        to_int(row["records"]) * to_float(row["calibration_error"])
        for row in calibration_rows
    ) / len(prediction_rows)
    return {
        "brier_score": format_float(brier),
        "expected_calibration_error": format_float(weighted_error),
    }


def build_points_error_bin_rows(prediction_rows):
    """Summarize point-prediction error by actual-points bins."""
    grouped = {}
    for row in prediction_rows:
        label, sort_key = points_bin(row["actual_points"])
        grouped.setdefault((label, sort_key), []).append(row)

    output_rows = []
    for (label, sort_key), rows in sorted(grouped.items(), key=lambda item: item[0][1]):
        errors = [
            to_float(row["predicted_points"]) - to_float(row["actual_points"])
            for row in rows
        ]
        abs_errors = [abs(error) for error in errors]
        squared_errors = [error**2 for error in errors]
        actual_points = [to_float(row["actual_points"]) for row in rows]
        predicted_points = [to_float(row["predicted_points"]) for row in rows]
        output_rows.append(
            {
                "points_bin": label,
                "bin_sort": sort_key,
                "records": len(rows),
                "mae": format_float(sum(abs_errors) / len(abs_errors)),
                "rmse": format_float(math.sqrt(sum(squared_errors) / len(squared_errors))),
                "mean_actual_points": format_float(sum(actual_points) / len(actual_points)),
                "mean_predicted_points": format_float(
                    sum(predicted_points) / len(predicted_points)
                ),
                "total_actual_points": format_float(sum(actual_points)),
                "total_predicted_points": format_float(sum(predicted_points)),
            }
        )
    return output_rows


def build_rule_mapped_points_rows(prediction_rows):
    """Map predicted race ranking to official F1 points for each race."""
    grouped = {}
    for row in prediction_rows:
        key = (row["season"], row["round"], row["race_name"])
        grouped.setdefault(key, []).append(row)

    output_rows = []
    for race_key, race_rows in sorted(
        grouped.items(), key=lambda item: (to_int(item[0][0]), to_int(item[0][1]))
    ):
        sorted_rows = sorted(
            race_rows, key=lambda row: to_float(row["predicted_points"]), reverse=True
        )
        for index, row in enumerate(sorted_rows):
            mapped_points = F1_POINTS_TABLE[index] if index < len(F1_POINTS_TABLE) else 0
            output_rows.append(
                {
                    "season": row["season"],
                    "round": row["round"],
                    "race_name": row["race_name"],
                    "driver_id": row["driver_id"],
                    "driver_name": row["driver_name"],
                    "constructor_name": row["constructor_name"],
                    "predicted_points": row["predicted_points"],
                    "rule_mapped_points": mapped_points,
                    "actual_points": row["actual_points"],
                    "actual_finish_position": row["actual_finish_position"],
                }
            )
    return output_rows


def summarize_rule_mapped_points(prediction_rows, mapped_rows):
    """Compare continuous predicted points with rule-mapped F1 points."""
    actual = [to_float(row["actual_points"]) for row in mapped_rows]
    continuous = [to_float(row["predicted_points"]) for row in mapped_rows]
    mapped = [to_float(row["rule_mapped_points"]) for row in mapped_rows]
    return [
        {
            "metric": "mae",
            "continuous_points": format_float(mean_absolute_error(actual, continuous)),
            "rule_mapped_points": format_float(mean_absolute_error(actual, mapped)),
        },
        {
            "metric": "rmse",
            "continuous_points": format_float(
                math.sqrt(mean_squared_error(actual, continuous))
            ),
            "rule_mapped_points": format_float(
                math.sqrt(mean_squared_error(actual, mapped))
            ),
        },
        {
            "metric": "total_predicted_points",
            "continuous_points": format_float(sum(continuous)),
            "rule_mapped_points": format_float(sum(mapped)),
        },
        {
            "metric": "total_actual_points",
            "continuous_points": format_float(sum(actual)),
            "rule_mapped_points": format_float(sum(actual)),
        },
    ]


def build_points_race_level_error_rows(prediction_rows, mapped_rows):
    """Summarize continuous and rule-mapped point errors for each race."""
    prediction_groups = {}
    for row in prediction_rows:
        key = (row["season"], row["round"], row["race_name"])
        prediction_groups.setdefault(key, []).append(row)

    mapped_groups = {}
    for row in mapped_rows:
        key = (row["season"], row["round"], row["race_name"])
        mapped_groups.setdefault(key, []).append(row)

    output_rows = []
    for key, rows in sorted(
        mapped_groups.items(), key=lambda item: (to_int(item[0][0]), to_int(item[0][1]))
    ):
        actual = [to_float(row["actual_points"]) for row in rows]
        continuous = [to_float(row["predicted_points"]) for row in rows]
        mapped = [to_float(row["rule_mapped_points"]) for row in rows]
        race_predictions = prediction_groups.get(key, [])
        predicted_top10 = sorted(
            race_predictions,
            key=lambda row: to_float(row["top10_probability"]),
            reverse=True,
        )[:10]
        actual_top10_ids = {
            row["driver_id"] for row in race_predictions if to_int(row["actual_is_top10"]) == 1
        }
        top10_hits = len({row["driver_id"] for row in predicted_top10} & actual_top10_ids)
        output_rows.append(
            {
                "season": key[0],
                "round": key[1],
                "race_name": key[2],
                "records": len(rows),
                "continuous_mae": format_float(mean_absolute_error(actual, continuous)),
                "rule_mapped_mae": format_float(mean_absolute_error(actual, mapped)),
                "continuous_total_predicted_points": format_float(sum(continuous)),
                "rule_mapped_total_predicted_points": format_float(sum(mapped)),
                "actual_total_points": format_float(sum(actual)),
                "top10_hits": top10_hits,
                "top10_precision": format_float(top10_hits / 10),
            }
        )
    return output_rows


def build_rolling_backtest_rows(rows):
    """Backtest best Top 10 and points model families across multiple seasons."""
    output_rows = []
    for test_season in [2022, 2023, 2024, 2025]:
        train_rows = [
            row
            for row in rows
            if TRAIN_START_SEASON <= to_int(row["season"]) < test_season
        ]
        test_rows = [row for row in rows if to_int(row["season"]) == test_season]
        if not train_rows or not test_rows:
            continue

        for feature_mode in FEATURE_MODES:
            train_x = build_x(train_rows, feature_mode)
            test_x = build_x(test_rows, feature_mode)
            top10_train_y = build_binary_y(train_rows, TOP10_TARGET)
            top10_test_y = build_binary_y(test_rows, TOP10_TARGET)
            for model_name, model in build_top10_rolling_models().items():
                model.fit(train_x, top10_train_y)
                probabilities = model.predict_proba(test_x)[:, 1]
                metrics = evaluate_top10(top10_test_y, probabilities)
                output_rows.append(
                    {
                        "task": "top10",
                        "feature_mode": feature_mode,
                        "model": model_name,
                        "train_seasons": f"{TRAIN_START_SEASON}-{test_season - 1}",
                        "test_season": test_season,
                        "records": len(test_rows),
                        "primary_metric": "f1",
                        "primary_value": format_float(metrics["f1"]),
                        "secondary_metric": "roc_auc",
                        "secondary_value": format_float(metrics["roc_auc"]),
                    }
                )

            points_train_y = build_points_y(train_rows)
            points_test_y = build_points_y(test_rows)
            for model_name, model in build_points_rolling_models().items():
                model.fit(train_x, points_train_y)
                predictions = [clamp_points(value) for value in model.predict(test_x)]
                metrics = evaluate_points(points_test_y, predictions)
                output_rows.append(
                    {
                        "task": "points",
                        "feature_mode": feature_mode,
                        "model": model_name,
                        "train_seasons": f"{TRAIN_START_SEASON}-{test_season - 1}",
                        "test_season": test_season,
                        "records": len(test_rows),
                        "primary_metric": "mae",
                        "primary_value": format_float(metrics["mae"]),
                        "secondary_metric": "rmse",
                        "secondary_value": format_float(metrics["rmse"]),
                    }
                )
    return output_rows


def build_points_correction_rows(train_rows, test_rows, feature_mode):
    """Compare baseline and simple target/weight corrections for points prediction."""
    train_x = build_x(train_rows, feature_mode)
    test_x = build_x(test_rows, feature_mode)
    train_y = build_points_y(train_rows)
    test_y = build_points_y(test_rows)
    rows = []
    for correction, model in build_points_correction_models().items():
        if correction == "weighted_high_points_catboost" and hasattr(model, "fit"):
            weights = [2.5 if value >= 16 else 1.0 for value in train_y]
            try:
                model.fit(train_x, train_y, model__sample_weight=weights)
            except TypeError:
                model.fit(train_x, train_y)
        else:
            model.fit(train_x, train_y)
        predictions = [clamp_points(value) for value in model.predict(test_x)]
        metrics = evaluate_points(test_y, predictions)
        row = {
            "feature_mode": feature_mode,
            "model": "catboost_regressor",
            "correction": correction,
        }
        row.update({field: format_float(metrics[field]) for field in [
            "mae",
            "rmse",
            "r2",
            "mean_actual_points",
            "mean_predicted_points",
            "total_actual_points",
            "total_predicted_points",
        ]})
        rows.append(row)
    return rows


def build_model_task_summary_rows(best_top10_row, best_points_row, points_summary):
    """Build a compact table of the best model for each prediction task."""
    rule_mapped_mae = ""
    for row in points_summary:
        if row["metric"] == "mae":
            rule_mapped_mae = row["rule_mapped_points"]
            break
    return [
        {
            "task": "podium",
            "target": "is_podium",
            "best_model": "tabnet_neural_network",
            "best_feature_mode": "post_qualifying",
            "key_metric": "F1",
            "value": "0.763158",
            "source_file": "deep_podium_model_summary.json",
        },
        {
            "task": "top10",
            "target": TOP10_TARGET,
            "best_model": best_top10_row["model"],
            "best_feature_mode": best_top10_row["feature_mode"],
            "key_metric": "F1",
            "value": best_top10_row["f1"],
            "source_file": "points_model_summary.json",
        },
        {
            "task": "points",
            "target": POINTS_TARGET,
            "best_model": best_points_row["model"],
            "best_feature_mode": best_points_row["feature_mode"],
            "key_metric": "rule_mapped_MAE",
            "value": rule_mapped_mae,
            "source_file": "points_model_summary.json",
        },
    ]


def get_feature_importance_rows(target, feature_mode, model_name, model, top_n=30):
    """Extract feature importance from the selected Top 10 or points model."""
    if not hasattr(model, "named_steps") or "vectorizer" not in model.named_steps:
        return []
    feature_names = model.named_steps["vectorizer"].get_feature_names_out()
    estimator = model.named_steps["model"]

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
        importance_type = "feature_importance"
    elif hasattr(estimator, "get_feature_importance"):
        values = estimator.get_feature_importance()
        importance_type = "feature_importance"
    elif hasattr(estimator, "coef_"):
        values = estimator.coef_
        if hasattr(values, "ndim") and values.ndim > 1:
            values = values[0]
        importance_type = "coefficient"
    else:
        return []

    rows = [
        {
            "target": target,
            "feature_mode": feature_mode,
            "model": model_name,
            "feature": feature,
            "importance_type": importance_type,
            "importance": format_float(float(value)),
            "abs_importance": format_float(abs(float(value))),
        }
        for feature, value in zip(feature_names, values)
    ]
    return sorted(rows, key=lambda row: to_float(row["abs_importance"]), reverse=True)[
        :top_n
    ]


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


def save_top10_race_level_chart(rows):
    """Save race-level Top 10 precision by race."""
    labels = [f"R{row['round']}" for row in rows]
    values = [to_float(row["precision_at_10"]) for row in rows]

    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.bar(labels, values, color="#2563EB")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Precision@10")
    ax.set_title("Race-Level Top 10 Precision, 2025")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    output_path = FIGURE_DIR / "top10_race_level_precision_2025.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_points_error_bin_chart(rows):
    """Save point-prediction MAE by actual-points bin."""
    labels = [row["points_bin"] for row in rows]
    values = [to_float(row["mae"]) for row in rows]

    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    bars = ax.bar(labels, values, color="#EA580C")
    ax.set_ylabel("Mean absolute error")
    ax.set_xlabel("Actual points bin")
    ax.set_title("Race Points Error by Actual Points Bin, 2025")
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.05,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    output_path = FIGURE_DIR / "points_error_by_points_bin_2025.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_top10_calibration_chart(rows):
    """Save a calibration curve for Top 10 predicted probabilities."""
    predicted = [to_float(row["mean_predicted_probability"]) for row in rows]
    actual = [to_float(row["actual_top10_rate"]) for row in rows]

    fig, ax = plt.subplots(figsize=(6.5, 6.1))
    ax.plot([0, 1], [0, 1], color="#DC2626", linestyle="--", linewidth=1.5, label="Perfect calibration")
    ax.plot(predicted, actual, marker="o", color="#2563EB", linewidth=2, label="Model")
    for row, x_value, y_value in zip(rows, predicted, actual):
        ax.text(x_value, y_value + 0.025, row["probability_bin"], ha="center", fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted Top 10 probability")
    ax.set_ylabel("Actual Top 10 rate")
    ax.set_title("Top 10 Probability Calibration, 2025")
    ax.legend()
    ax.grid(True, color="#E5E7EB", linewidth=0.8)
    fig.tight_layout()
    output_path = FIGURE_DIR / "top10_calibration_curve_2025.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_rule_mapped_points_chart(summary_rows):
    """Save a chart comparing continuous and rule-mapped point prediction errors."""
    error_rows = [row for row in summary_rows if row["metric"] in {"mae", "rmse"}]
    labels = [row["metric"].upper() for row in error_rows]
    continuous = [to_float(row["continuous_points"]) for row in error_rows]
    mapped = [to_float(row["rule_mapped_points"]) for row in error_rows]
    x_values = list(range(len(labels)))
    width = 0.34

    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    ax.bar(
        [value - width / 2 for value in x_values],
        continuous,
        width=width,
        label="Continuous regression",
        color="#2563EB",
    )
    ax.bar(
        [value + width / 2 for value in x_values],
        mapped,
        width=width,
        label="Rule-mapped points",
        color="#EA580C",
    )
    ax.set_xticks(x_values, labels=labels)
    ax.set_ylabel("Error")
    ax.set_title("Continuous vs F1 Rule-Mapped Points, 2025")
    ax.legend()
    fig.tight_layout()
    output_path = FIGURE_DIR / "points_rule_mapped_comparison_2025.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_rolling_backtest_chart(rows):
    """Save rolling-backtest trends for Top 10 F1 and points MAE."""
    top10_rows = [
        row for row in rows if row["task"] == "top10" and row["feature_mode"] == "post_qualifying"
    ]
    points_rows = [
        row for row in rows if row["task"] == "points" and row["feature_mode"] == "post_qualifying"
    ]
    seasons = [to_int(row["test_season"]) for row in top10_rows]

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    axes[0].plot(
        seasons,
        [to_float(row["primary_value"]) for row in top10_rows],
        marker="o",
        color="#2563EB",
    )
    axes[0].set_title("Top 10 Rolling Backtest")
    axes[0].set_xlabel("Test season")
    axes[0].set_ylabel("F1 score")
    axes[0].set_ylim(0, 1)
    axes[0].grid(True, color="#E5E7EB", linewidth=0.8)

    axes[1].plot(
        [to_int(row["test_season"]) for row in points_rows],
        [to_float(row["primary_value"]) for row in points_rows],
        marker="o",
        color="#EA580C",
    )
    axes[1].set_title("Points Rolling Backtest")
    axes[1].set_xlabel("Test season")
    axes[1].set_ylabel("MAE")
    axes[1].grid(True, color="#E5E7EB", linewidth=0.8)

    fig.suptitle("Top 10 and Points Rolling Backtest, 2022-2025")
    fig.tight_layout()
    output_path = FIGURE_DIR / "top10_points_rolling_backtest.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_points_race_level_error_chart(rows):
    """Save race-level points prediction error for the 2025 test season."""
    labels = [f"R{row['round']}" for row in rows]
    continuous = [to_float(row["continuous_mae"]) for row in rows]
    mapped = [to_float(row["rule_mapped_mae"]) for row in rows]
    x_values = list(range(len(rows)))
    width = 0.36

    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.bar(
        [value - width / 2 for value in x_values],
        continuous,
        width=width,
        label="Continuous MAE",
        color="#2563EB",
    )
    ax.bar(
        [value + width / 2 for value in x_values],
        mapped,
        width=width,
        label="Rule-mapped MAE",
        color="#EA580C",
    )
    ax.set_xticks(x_values, labels=labels, rotation=45)
    ax.set_ylabel("MAE")
    ax.set_title("Race-Level Points Prediction Error, 2025")
    ax.legend()
    fig.tight_layout()
    output_path = FIGURE_DIR / "points_race_level_error_2025.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_points_correction_chart(rows):
    """Save a comparison of experimental high-score correction strategies."""
    sorted_rows = sorted(rows, key=lambda row: to_float(row["mae"]))
    labels = [row["correction"].replace("_", "\n") for row in sorted_rows]
    values = [to_float(row["mae"]) for row in sorted_rows]

    fig, ax = plt.subplots(figsize=(7.8, 5.3))
    bars = ax.bar(labels, values, color=["#059669", "#2563EB", "#EA580C"][: len(rows)])
    ax.set_ylabel("MAE")
    ax.set_title("Points Model Correction Experiments, 2025")
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.03,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.tight_layout()
    output_path = FIGURE_DIR / "points_correction_experiment_2025.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_feature_importance_chart(rows, filename, title):
    """Save a feature-importance chart for a selected model."""
    top_rows = rows[:15]
    labels = [row["feature"] for row in reversed(top_rows)]
    values = [to_float(row["importance"]) for row in reversed(top_rows)]
    colors = ["#059669" if value >= 0 else "#DC2626" for value in values]

    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    ax.barh(labels, values, color=colors)
    ax.axvline(0, color="#111827", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    output_path = FIGURE_DIR / filename
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
    top10_race_level_rows = build_top10_race_level_rows(prediction_rows)
    top10_race_level_summary = summarize_top10_race_level(top10_race_level_rows)
    points_error_bin_rows = build_points_error_bin_rows(prediction_rows)
    top10_calibration_rows = build_top10_calibration_rows(prediction_rows)
    top10_calibration_summary = summarize_top10_calibration(
        prediction_rows, top10_calibration_rows
    )
    rule_mapped_points_rows = build_rule_mapped_points_rows(prediction_rows)
    rule_mapped_points_summary = summarize_rule_mapped_points(
        prediction_rows, rule_mapped_points_rows
    )
    points_race_level_error_rows = build_points_race_level_error_rows(
        prediction_rows, rule_mapped_points_rows
    )
    rolling_backtest_rows = build_rolling_backtest_rows(rows)
    points_correction_rows = build_points_correction_rows(
        train_rows, test_rows, best_points_row["feature_mode"]
    )
    model_task_summary_rows = build_model_task_summary_rows(
        best_top10_row, best_points_row, rule_mapped_points_summary
    )
    top10_feature_importance_rows = get_feature_importance_rows(
        "top10",
        best_top10_row["feature_mode"],
        best_top10_row["model"],
        best_top10_info["model"],
    )
    points_feature_importance_rows = get_feature_importance_rows(
        "points",
        best_points_row["feature_mode"],
        best_points_row["model"],
        best_points_info["model"],
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
    write_csv(
        MODEL_DIR / "top10_race_level_metrics_2025.csv",
        TOP10_RACE_LEVEL_FIELDS,
        top10_race_level_rows,
    )
    write_csv(
        MODEL_DIR / "points_error_by_points_bin_2025.csv",
        POINTS_ERROR_BIN_FIELDS,
        points_error_bin_rows,
    )
    write_csv(
        MODEL_DIR / "top10_calibration_metrics_2025.csv",
        TOP10_CALIBRATION_FIELDS,
        top10_calibration_rows,
    )
    write_csv(
        MODEL_DIR / "points_rule_mapped_predictions_2025.csv",
        RULE_MAPPED_POINTS_FIELDS,
        rule_mapped_points_rows,
    )
    write_csv(
        MODEL_DIR / "points_rule_mapped_summary_2025.csv",
        RULE_MAPPED_POINTS_SUMMARY_FIELDS,
        rule_mapped_points_summary,
    )
    write_csv(
        MODEL_DIR / "points_race_level_error_2025.csv",
        POINTS_RACE_LEVEL_FIELDS,
        points_race_level_error_rows,
    )
    write_csv(
        MODEL_DIR / "top10_points_rolling_backtest_metrics.csv",
        ROLLING_BACKTEST_FIELDS,
        rolling_backtest_rows,
    )
    write_csv(
        MODEL_DIR / "points_correction_experiment_2025.csv",
        POINTS_CORRECTION_FIELDS,
        points_correction_rows,
    )
    write_csv(
        MODEL_DIR / "model_task_summary.csv",
        MODEL_TASK_SUMMARY_FIELDS,
        model_task_summary_rows,
    )
    write_csv(
        MODEL_DIR / "top10_feature_importance.csv",
        FEATURE_IMPORTANCE_FIELDS,
        top10_feature_importance_rows,
    )
    write_csv(
        MODEL_DIR / "points_feature_importance.csv",
        FEATURE_IMPORTANCE_FIELDS,
        points_feature_importance_rows,
    )

    top10_chart = save_top10_chart(top10_metric_rows)
    points_chart = save_points_chart(points_metric_rows)
    actual_vs_predicted_chart = save_actual_vs_predicted_chart(prediction_rows)
    top10_race_level_chart = save_top10_race_level_chart(top10_race_level_rows)
    points_error_bin_chart = save_points_error_bin_chart(points_error_bin_rows)
    top10_calibration_chart = save_top10_calibration_chart(top10_calibration_rows)
    rule_mapped_points_chart = save_rule_mapped_points_chart(
        rule_mapped_points_summary
    )
    points_race_level_error_chart = save_points_race_level_error_chart(
        points_race_level_error_rows
    )
    rolling_backtest_chart = save_rolling_backtest_chart(rolling_backtest_rows)
    points_correction_chart = save_points_correction_chart(points_correction_rows)
    top10_feature_importance_chart = save_feature_importance_chart(
        top10_feature_importance_rows,
        "top10_feature_importance.png",
        "Top 10 Model Feature Importance",
    )
    points_feature_importance_chart = save_feature_importance_chart(
        points_feature_importance_rows,
        "points_feature_importance.png",
        "Points Model Feature Importance",
    )

    summary = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "input_features": str(training_features_path.relative_to(BASE_DIR)),
        "train_seasons": f"{TRAIN_START_SEASON}-{TRAIN_END_SEASON}",
        "test_season": TEST_SEASON,
        "feature_modes": FEATURE_MODES,
        "top10_target": TOP10_TARGET,
        "points_target": POINTS_TARGET,
        "top10_model_count": len(top10_metric_rows),
        "points_model_count": len(points_metric_rows),
        "model_families": [
            "traditional_machine_learning",
            "advanced_boosting",
            "deep_learning",
        ],
        "best_top10_model": {
            "feature_mode": best_top10_row["feature_mode"],
            "model": best_top10_row["model"],
            "f1": best_top10_row["f1"],
            "roc_auc": best_top10_row["roc_auc"],
            "race_level": top10_race_level_summary,
            "calibration": top10_calibration_summary,
        },
        "best_points_model": {
            "feature_mode": best_points_row["feature_mode"],
            "model": best_points_row["model"],
            "mae": best_points_row["mae"],
            "rmse": best_points_row["rmse"],
            "r2": best_points_row["r2"],
            "primary_points_output": "rule_mapped_points",
            "rule_mapped_points_summary": rule_mapped_points_summary,
        },
        "outputs": [
            "top10_model_metrics.csv",
            "points_model_metrics.csv",
            "points_predictions_2025.csv",
            "points_completed_2026_predictions.csv",
            "top10_race_level_metrics_2025.csv",
            "points_error_by_points_bin_2025.csv",
            "top10_calibration_metrics_2025.csv",
            "points_rule_mapped_predictions_2025.csv",
            "points_rule_mapped_summary_2025.csv",
            "points_race_level_error_2025.csv",
            "top10_points_rolling_backtest_metrics.csv",
            "points_correction_experiment_2025.csv",
            "model_task_summary.csv",
            "top10_feature_importance.csv",
            "points_feature_importance.csv",
        ],
        "figures": [
            str(top10_chart.relative_to(BASE_DIR)),
            str(points_chart.relative_to(BASE_DIR)),
            str(actual_vs_predicted_chart.relative_to(BASE_DIR)),
            str(top10_race_level_chart.relative_to(BASE_DIR)),
            str(points_error_bin_chart.relative_to(BASE_DIR)),
            str(top10_calibration_chart.relative_to(BASE_DIR)),
            str(rule_mapped_points_chart.relative_to(BASE_DIR)),
            str(points_race_level_error_chart.relative_to(BASE_DIR)),
            str(rolling_backtest_chart.relative_to(BASE_DIR)),
            str(points_correction_chart.relative_to(BASE_DIR)),
            str(top10_feature_importance_chart.relative_to(BASE_DIR)),
            str(points_feature_importance_chart.relative_to(BASE_DIR)),
        ],
        "note": "These models estimate Top 10 probability and continuous expected race points. The regression target is normalized to current Grand Prix points from finish position only, so older historical scoring systems and fastest-lap bonus points are ignored. Sprint points are outside this project scope. Every race is treated as a normal full-points Grand Prix with ten scoring places. For downstream standings, season simulation, and championship prediction, rule_mapped_points should be treated as the primary points output because it follows the official F1 scoring table. Continuous predicted points are retained as an auxiliary expected-value signal.",
    }
    write_json(SUMMARY_PATH, summary)

    print(f"Top10 and points model outputs saved to: {MODEL_DIR}")
    print(f"Best Top10 model: {best_top10_row['feature_mode']} / {best_top10_row['model']} / F1={best_top10_row['f1']}")
    print(f"Best points model: {best_points_row['feature_mode']} / {best_points_row['model']} / MAE={best_points_row['mae']}")


if __name__ == "__main__":
    main()
