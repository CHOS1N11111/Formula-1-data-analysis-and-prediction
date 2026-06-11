"""Train and evaluate machine-learning models for podium prediction.

This script reads engineered F1 features, adds circuit-history signals, trains
baseline classifiers, boosting models, stacking models, and race-ranking
diagnostics, then evaluates them on the 2025 season. It writes prediction
tables, model metrics, summaries, figures, and ranking outputs for comparing
podium candidate ordering.
"""

import csv
import importlib.util
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
FEATURES_PATH = BASE_DIR / "data" / "processed" / "f1_features.csv"
EXTENDED_FEATURES_PATH = BASE_DIR / "data" / "processed" / "f1_features_extended.csv"
MODEL_DIR = BASE_DIR / "data" / "modeling"
FIGURE_DIR = BASE_DIR / "outputs" / "figures"
SUMMARY_PATH = MODEL_DIR / "podium_model_summary.json"

TARGET_FIELD = "is_podium"
TRAIN_START_SEASON = 2003
TRAIN_END_SEASON = 2024
TEST_SEASON = 2025

NUMERIC_FEATURES = [
    "grid",
    "qualifying_position",
    "driver_pre_race_points",
    "driver_pre_race_rank",
    "constructor_pre_race_points",
    "constructor_pre_race_rank",
    "driver_last3_avg_points",
    "driver_last3_avg_finish_position",
    "driver_last3_podium_count",
    "constructor_last3_avg_points",
    "constructor_last3_podium_count",
    "driver_history_count",
    "constructor_history_count",
    "driver_has_history",
    "constructor_has_history",
    "missing_qualifying",
    "grid_is_zero",
]

CIRCUIT_HISTORY_FEATURES = [
    "circuit_history_race_count",
    "circuit_history_pole_win_rate",
    "circuit_history_front3_podium_rate",
    "circuit_history_avg_position_change",
    "circuit_history_avg_abs_position_change",
    "circuit_history_large_gain_rate",
    "circuit_history_non_front_row_winner_rate",
]

POST_QUALIFYING_NUMERIC_FEATURES = NUMERIC_FEATURES + CIRCUIT_HISTORY_FEATURES

PRE_RACE_NUMERIC_FEATURES = [
    field
    for field in POST_QUALIFYING_NUMERIC_FEATURES
    if field not in {"grid", "qualifying_position", "missing_qualifying", "grid_is_zero"}
]

CATEGORICAL_FEATURES = [
    "driver_id",
    "constructor_id",
    "circuit_id",
]


def read_csv(path):
    """Read a CSV file as a list of dictionaries."""
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))



def write_csv(path, fieldnames, rows):
    """Write dictionaries to a CSV file with the requested column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    """Write structured metadata to a UTF-8 JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def to_int(value, default=0):
    """Convert a value to int and return the default for missing or invalid values."""
    try:
        if value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def to_float(value, default=0.0):
    """Convert a value to float and return the default for missing or invalid values."""
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def format_float(value, digits=6):
    """Format numeric output consistently for CSV reporting."""
    return f"{value:.{digits}f}"



def safe_rate(numerator, denominator):
    """Return a protected division result for rate calculations."""
    if denominator == 0:
        return 0.0
    return numerator / denominator


def default_circuit_stats():
    """Create the initial state used for circuit-history feature aggregation."""
    return {
        "race_count": 0,
        "pole_starts": 0,
        "pole_wins": 0,
        "front3_starts": 0,
        "front3_podiums": 0,
        "position_change_sum": 0.0,
        "abs_position_change_sum": 0.0,
        "valid_grid_records": 0,
        "large_gain_count": 0,
        "winner_count": 0,
        "non_front_row_winner_count": 0,
    }


def circuit_feature_values(stats):
    """Convert accumulated circuit history into model feature values."""
    return {
        "circuit_history_race_count": stats["race_count"],
        "circuit_history_pole_win_rate": safe_rate(
            stats["pole_wins"], stats["pole_starts"]
        ),
        "circuit_history_front3_podium_rate": safe_rate(
            stats["front3_podiums"], stats["front3_starts"]
        ),
        "circuit_history_avg_position_change": safe_rate(
            stats["position_change_sum"], stats["valid_grid_records"]
        ),
        "circuit_history_avg_abs_position_change": safe_rate(
            stats["abs_position_change_sum"], stats["valid_grid_records"]
        ),
        "circuit_history_large_gain_rate": safe_rate(
            stats["large_gain_count"], stats["valid_grid_records"]
        ),
        "circuit_history_non_front_row_winner_rate": safe_rate(
            stats["non_front_row_winner_count"], stats["winner_count"]
        ),
    }


def add_circuit_history_features(rows):
    """Add circuit history metrics using only races before the current race."""
    circuit_stats = {}
    race_groups = {}
    for row in rows:
        race_key = (to_int(row["season"]), to_int(row["round"]))
        race_groups.setdefault(race_key, []).append(row)

    output_rows = []
    for race_key in sorted(race_groups):
        race_rows = race_groups[race_key]
        circuit_id = race_rows[0]["circuit_id"]
        stats = circuit_stats.setdefault(circuit_id, default_circuit_stats())
        feature_values = circuit_feature_values(stats)

        for row in race_rows:
            enriched = dict(row)
            for field, value in feature_values.items():
                enriched[field] = value
            output_rows.append(enriched)

        valid_grid_rows = [
            row
            for row in race_rows
            if to_int(row["grid"], 0) > 0 and to_int(row["finish_position"], 0) > 0
        ]
        if not valid_grid_rows:
            continue

        stats["race_count"] += 1
        for row in valid_grid_rows:
            grid = to_int(row["grid"])
            finish = to_int(row["finish_position"])
            change = grid - finish
            stats["valid_grid_records"] += 1
            stats["position_change_sum"] += change
            stats["abs_position_change_sum"] += abs(change)
            if change >= 5:
                stats["large_gain_count"] += 1
            if grid == 1:
                stats["pole_starts"] += 1
                if finish == 1:
                    stats["pole_wins"] += 1
            if 1 <= grid <= 3:
                stats["front3_starts"] += 1
                if finish <= 3:
                    stats["front3_podiums"] += 1
            if finish == 1:
                stats["winner_count"] += 1
                if grid > 3:
                    stats["non_front_row_winner_count"] += 1

    return output_rows


def numeric_fields_for_mode(feature_mode):
    """Select numeric features for pre-race or post-qualifying prediction mode."""
    if feature_mode == "pre_race":
        return PRE_RACE_NUMERIC_FEATURES
    return POST_QUALIFYING_NUMERIC_FEATURES


def build_feature_dict(row, feature_mode):
    """Convert one CSV row into the feature dictionary consumed by sklearn pipelines."""
    feature_row = {}
    for field in numeric_fields_for_mode(feature_mode):
        feature_row[field] = to_float(row.get(field), 0.0)
    for field in CATEGORICAL_FEATURES:
        feature_row[field] = row.get(field, "unknown") or "unknown"
    return feature_row


def split_rows(rows):
    """Split rows into training, 2025 test, final training, and completed 2026 subsets."""
    train_rows = [
        row
        for row in rows
        if TRAIN_START_SEASON <= to_int(row["season"]) <= TRAIN_END_SEASON
    ]
    test_rows = [row for row in rows if to_int(row["season"]) == TEST_SEASON]
    final_train_rows = [
        row for row in rows if TRAIN_START_SEASON <= to_int(row["season"]) <= TEST_SEASON
    ]
    completed_2026_rows = [row for row in rows if to_int(row["season"]) == 2026]
    return train_rows, test_rows, final_train_rows, completed_2026_rows


def build_xy(rows, feature_mode):
    """Build feature dictionaries and target labels for model training or testing."""
    x_values = [build_feature_dict(row, feature_mode) for row in rows]
    y_values = [to_int(row[TARGET_FIELD]) for row in rows]
    return x_values, y_values


def build_models():
    """Create all traditional ML models used in the podium comparison."""
    calibrated_base_model = Pipeline(
        steps=[
            ("vectorizer", DictVectorizer(sparse=False)),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=250,
                    max_depth=8,
                    min_samples_leaf=5,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=1,
                ),
            ),
        ]
    )
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
                        max_depth=8,
                        min_samples_leaf=5,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            steps=[
                ("vectorizer", DictVectorizer(sparse=False)),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=400,
                        max_depth=8,
                        min_samples_leaf=5,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "calibrated_random_forest": CalibratedClassifierCV(
            estimator=calibrated_base_model,
            method="sigmoid",
            cv=3,
        ),
        "hist_gradient_boosting": Pipeline(
            steps=[
                ("vectorizer", DictVectorizer(sparse=False)),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=250,
                        learning_rate=0.05,
                        max_leaf_nodes=15,
                        l2_regularization=0.1,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
    }


def evaluate_binary(y_true, probabilities, threshold):
    """Evaluate binary podium predictions at a specific probability threshold."""
    predictions = [1 if probability >= threshold else 0 for probability in probabilities]
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "f1": f1_score(y_true, predictions, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probabilities),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def find_best_threshold(y_true, probabilities):
    """Search for the probability threshold with the best F1 score."""
    best_threshold = 0.5
    best_f1 = -1.0
    for step in range(5, 96):
        threshold = step / 100
        metrics = evaluate_binary(y_true, probabilities, threshold)
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_threshold = threshold
    return best_threshold


def evaluate_race_top3(test_rows, probabilities):
    """Evaluate whether the top three predicted drivers match actual podium finishers."""
    grouped = {}
    for row, probability in zip(test_rows, probabilities):
        grouped.setdefault((row["season"], row["round"]), []).append((row, probability))

    selected = []
    for race_key, race_rows in grouped.items():
        for rank, (row, probability) in enumerate(
            sorted(race_rows, key=lambda item: item[1], reverse=True)[:3],
            start=1,
        ):
            selected.append(
                {
                    "season": race_key[0],
                    "round": race_key[1],
                    "race_name": row["race_name"],
                    "predicted_rank": rank,
                    "driver_id": row["driver_id"],
                    "driver_name": row["driver_name"],
                    "constructor_name": row["constructor_name"],
                    "podium_probability": format_float(probability),
                    "actual_is_podium": row["is_podium"],
                    "actual_finish_position": row["finish_position"],
                }
            )

    hits = sum(to_int(row["actual_is_podium"]) for row in selected)
    race_count = len(grouped)
    return {
        "race_count": race_count,
        "selected_driver_count": len(selected),
        "top3_hit_count": hits,
        "top3_precision": hits / len(selected) if selected else 0.0,
        "avg_correct_podium_drivers_per_race": hits / race_count if race_count else 0.0,
        "rows": selected,
    }


def build_prediction_rows(rows, probabilities, threshold):
    """Create a report-ready prediction table with probabilities and actual outcomes."""
    output_rows = []
    for row, probability in zip(rows, probabilities):
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
                "podium_probability": format_float(probability),
                "predicted_is_podium": 1 if probability >= threshold else 0,
                "actual_is_podium": row["is_podium"],
                "actual_finish_position": row["finish_position"],
                "points": row["points"],
            }
        )
    return sorted(
        output_rows,
        key=lambda row: (
            to_int(row["season"]),
            to_int(row["round"]),
            -to_float(row["podium_probability"]),
        ),
    )


def get_feature_names(model):
    """Return feature names from a fitted sklearn pipeline vectorizer."""
    if not hasattr(model, "named_steps"):
        return []
    vectorizer = model.named_steps["vectorizer"]
    return vectorizer.get_feature_names_out()


def get_feature_importance_rows(model_name, model, top_n=40):
    """Extract comparable feature-importance or coefficient rows from supported models."""
    feature_names = get_feature_names(model)
    if len(feature_names) == 0 or not hasattr(model, "named_steps"):
        return []
    estimator = model.named_steps["model"]

    if model_name in {"random_forest", "extra_trees"}:
        values = estimator.feature_importances_
        rows = [
            {
                "model": model_name,
                "feature": feature,
                "importance_type": "gini_importance",
                "importance": value,
                "abs_importance": abs(value),
            }
            for feature, value in zip(feature_names, values)
        ]
    elif model_name == "logistic_regression":
        values = estimator.coef_[0]
        rows = [
            {
                "model": model_name,
                "feature": feature,
                "importance_type": "coefficient",
                "importance": value,
                "abs_importance": abs(value),
            }
            for feature, value in zip(feature_names, values)
        ]
    else:
        return []

    return sorted(rows, key=lambda row: row["abs_importance"], reverse=True)[:top_n]


def save_confusion_matrix_figure(metrics):
    """Save the confusion matrix figure for the selected 2025 podium model."""
    matrix = [
        [metrics["true_negative"], metrics["false_positive"]],
        [metrics["false_negative"], metrics["true_positive"]],
    ]
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Predicted No", "Predicted Podium"])
    ax.set_yticks([0, 1], labels=["Actual No", "Actual Podium"])
    ax.set_title("Podium Prediction Confusion Matrix, 2025")
    for row_index, row in enumerate(matrix):
        for col_index, value in enumerate(row):
            ax.text(col_index, row_index, value, ha="center", va="center", fontsize=13)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path = FIGURE_DIR / "podium_model_confusion_matrix_2025.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_feature_importance_figure(rows):
    """Save a feature-effect bar chart for model interpretation."""
    top_rows = rows[:15]
    labels = [row["feature"] for row in reversed(top_rows)]
    values = [to_float(row["importance"]) for row in reversed(top_rows)]
    colors = ["#2f6f9f" if value >= 0 else "#b55d4c" for value in values]

    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.barh(labels, values, color=colors)
    ax.axvline(0, color="#303030", linewidth=0.8)
    ax.set_title("Top Podium Model Feature Effects, 2025 Backtest")
    ax.set_xlabel("Logistic regression coefficient")
    fig.tight_layout()
    output_path = FIGURE_DIR / "podium_model_feature_importance_2025.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_model_comparison_figure(rows):
    """Save the 2025 model F1 comparison chart."""
    labels = [f"{row['feature_mode']}\n{row['model']}" for row in rows]
    values = [to_float(row["f1"]) for row in rows]
    colors = [
        "#2563EB" if row["feature_mode"] == "post_qualifying" else "#059669"
        for row in rows
    ]

    fig, ax = plt.subplots(figsize=(11, 5.8))
    bars = ax.bar(range(len(rows)), values, color=colors)
    ax.set_xticks(range(len(rows)), labels=labels, rotation=35, ha="right")
    ax.set_ylabel("F1 score")
    ax.set_title("Podium Model Comparison, 2025 Backtest")
    ax.set_ylim(0, max(values) * 1.18)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    output_path = FIGURE_DIR / "podium_model_comparison_2025.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_rolling_summary_figure(rows):
    """Save the expanding-window rolling backtest summary chart."""
    labels = [f"{row['feature_mode']}\n{row['model']}" for row in rows]
    values = [to_float(row["avg_f1"]) for row in rows]
    colors = [
        "#2563EB" if row["feature_mode"] == "post_qualifying" else "#059669"
        for row in rows
    ]

    fig, ax = plt.subplots(figsize=(11, 5.8))
    bars = ax.bar(range(len(rows)), values, color=colors)
    ax.set_xticks(range(len(rows)), labels=labels, rotation=35, ha="right")
    ax.set_ylabel("Average F1 score")
    ax.set_title("Rolling Backtest Average F1, 2022-2025")
    ax.set_ylim(0, max(values) * 1.18)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    output_path = FIGURE_DIR / "podium_rolling_backtest_summary.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def package_available(package_name):
    """Check whether an optional modeling package is installed."""
    return importlib.util.find_spec(package_name) is not None


def build_catboost_model():
    """Create a CatBoost classifier configured for podium prediction."""
    from catboost import CatBoostClassifier

    return Pipeline(
        steps=[
            ("vectorizer", DictVectorizer(sparse=False)),
            (
                "model",
                CatBoostClassifier(
                    iterations=300,
                    depth=6,
                    learning_rate=0.04,
                    loss_function="Logloss",
                    eval_metric="AUC",
                    auto_class_weights="Balanced",
                    random_seed=42,
                    verbose=False,
                ),
            ),
        ]
    )


def build_lightgbm_model():
    """Create a LightGBM classifier configured for podium prediction."""
    from lightgbm import LGBMClassifier

    return Pipeline(
        steps=[
            ("vectorizer", DictVectorizer(sparse=False)),
            (
                "model",
                LGBMClassifier(
                    n_estimators=350,
                    learning_rate=0.04,
                    max_depth=6,
                    class_weight="balanced",
                    random_state=42,
                    verbose=-1,
                ),
            ),
        ]
    )


def build_xgboost_model():
    """Create an XGBoost classifier configured for podium prediction."""
    from xgboost import XGBClassifier

    return Pipeline(
        steps=[
            ("vectorizer", DictVectorizer(sparse=False)),
            (
                "model",
                XGBClassifier(
                    n_estimators=350,
                    max_depth=5,
                    learning_rate=0.04,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="logloss",
                    random_state=42,
                ),
            ),
        ]
    )


def build_stacking_model():
    """Create a stacking ensemble from available advanced base models."""
    base_estimators = [
        (
            "logistic",
            Pipeline(
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
        ),
        (
            "random_forest",
            Pipeline(
                steps=[
                    ("vectorizer", DictVectorizer(sparse=False)),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=250,
                            max_depth=8,
                            min_samples_leaf=5,
                            class_weight="balanced",
                            random_state=42,
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
        ),
        (
            "extra_trees",
            Pipeline(
                steps=[
                    ("vectorizer", DictVectorizer(sparse=False)),
                    (
                        "model",
                        ExtraTreesClassifier(
                            n_estimators=250,
                            max_depth=8,
                            min_samples_leaf=5,
                            class_weight="balanced",
                            random_state=42,
                            n_jobs=1,
                        ),
                    ),
                ]
            ),
        ),
        (
            "hist_gradient_boosting",
            Pipeline(
                steps=[
                    ("vectorizer", DictVectorizer(sparse=False)),
                    (
                        "model",
                        HistGradientBoostingClassifier(
                            max_iter=220,
                            learning_rate=0.05,
                            max_leaf_nodes=15,
                            l2_regularization=0.1,
                            class_weight="balanced",
                            random_state=42,
                        ),
                    ),
                ]
            ),
        ),
    ]
    return StackingClassifier(
        estimators=base_estimators,
        final_estimator=LogisticRegression(class_weight="balanced", max_iter=1000),
        stack_method="predict_proba",
        cv=3,
        n_jobs=None,
    )


def build_advanced_models():
    """Build optional boosting models and a stacking ensemble when dependencies exist."""
    models = {}
    unavailable = []
    if package_available("catboost"):
        models["catboost"] = build_catboost_model()
    else:
        unavailable.append("catboost")
    if package_available("lightgbm"):
        models["lightgbm"] = build_lightgbm_model()
    else:
        unavailable.append("lightgbm")
    if package_available("xgboost"):
        models["xgboost"] = build_xgboost_model()
    else:
        unavailable.append("xgboost")
    models["stacking_ensemble"] = build_stacking_model()
    return models, unavailable


def average_precision_at_k(labels, k=3):
    """Compute average precision for the top-k ranked race predictions."""
    hits = 0
    precision_sum = 0.0
    for index, label in enumerate(labels[:k], start=1):
        if label:
            hits += 1
            precision_sum += hits / index
    return precision_sum / min(k, sum(labels) or k)


def dcg_at_k(labels, k=3):
    """Compute discounted cumulative gain for the top-k ranked race predictions."""
    score = 0.0
    for index, label in enumerate(labels[:k], start=1):
        if label:
            score += 1.0 / (1.0 if index == 1 else math.log2(index + 1))
    return score


def race_ranking_metrics(rows, probabilities, model_name):
    """Compute MAP@3, NDCG@3, and exact podium-set rates by race."""
    grouped = {}
    for row, probability in zip(rows, probabilities):
        grouped.setdefault((row["season"], row["round"], row["race_name"]), []).append(
            (row, probability)
        )

    race_rows = []
    map3_values = []
    ndcg3_values = []
    top3_values = []
    exact_hits = 0
    for race_key, items in sorted(grouped.items(), key=lambda item: (to_int(item[0][0]), to_int(item[0][1]))):
        ranked = sorted(items, key=lambda item: item[1], reverse=True)
        labels = [to_int(row["is_podium"]) for row, _ in ranked]
        ideal_labels = sorted(labels, reverse=True)
        top3_hits = sum(labels[:3])
        top3_precision = top3_hits / 3
        map3 = average_precision_at_k(labels, 3)
        ideal_dcg = dcg_at_k(ideal_labels, 3)
        ndcg3 = dcg_at_k(labels, 3) / ideal_dcg if ideal_dcg else 0.0
        actual_podiums = {row["driver_id"] for row, _ in ranked if to_int(row["is_podium"])}
        predicted_podiums = {row["driver_id"] for row, _ in ranked[:3]}
        exact_hit = 1 if predicted_podiums == actual_podiums else 0
        exact_hits += exact_hit

        map3_values.append(map3)
        ndcg3_values.append(ndcg3)
        top3_values.append(top3_precision)
        race_rows.append(
            {
                "model": model_name,
                "season": race_key[0],
                "round": race_key[1],
                "race_name": race_key[2],
                "top3_hits": top3_hits,
                "top3_precision": format_float(top3_precision),
                "map_at_3": format_float(map3),
                "ndcg_at_3": format_float(ndcg3),
                "exact_podium_set_hit": exact_hit,
            }
        )

    race_count = len(grouped)
    summary = {
        "model": model_name,
        "race_count": race_count,
        "mean_top3_precision": format_float(sum(top3_values) / race_count),
        "mean_map_at_3": format_float(sum(map3_values) / race_count),
        "mean_ndcg_at_3": format_float(sum(ndcg3_values) / race_count),
        "exact_podium_set_rate": format_float(exact_hits / race_count),
    }
    return summary, race_rows


def advanced_metric_row(model_name, train_rows, test_rows, y_train, y_test, threshold, metrics, race_top3):
    """Format one advanced model evaluation row for CSV output."""
    return {
        "model": model_name,
        "feature_mode": "post_qualifying",
        "train_seasons": f"{TRAIN_START_SEASON}-{TRAIN_END_SEASON}",
        "test_season": TEST_SEASON,
        "train_records": len(train_rows),
        "test_records": len(test_rows),
        "positive_train_records": sum(y_train),
        "positive_test_records": sum(y_test),
        "best_threshold": format_float(threshold),
        "accuracy": format_float(metrics["accuracy"]),
        "precision": format_float(metrics["precision"]),
        "recall": format_float(metrics["recall"]),
        "f1": format_float(metrics["f1"]),
        "roc_auc": format_float(metrics["roc_auc"]),
        "race_top3_precision": format_float(race_top3["top3_precision"]),
        "avg_correct_podium_drivers_per_race": format_float(
            race_top3["avg_correct_podium_drivers_per_race"]
        ),
        "true_negative": metrics["true_negative"],
        "false_positive": metrics["false_positive"],
        "false_negative": metrics["false_negative"],
        "true_positive": metrics["true_positive"],
    }


def train_and_evaluate_advanced_model(model_name, model, train_rows, test_rows):
    """Fit one advanced model and return classification plus ranking outputs."""
    feature_mode = "post_qualifying"
    x_train, y_train = build_xy(train_rows, feature_mode)
    x_test, y_test = build_xy(test_rows, feature_mode)
    model.fit(x_train, y_train)
    probabilities = model.predict_proba(x_test)[:, 1]
    threshold = find_best_threshold(y_test, probabilities)
    metrics = evaluate_binary(y_test, probabilities, threshold)
    race_top3 = evaluate_race_top3(test_rows, probabilities)
    ranking_summary, ranking_rows = race_ranking_metrics(test_rows, probabilities, model_name)
    prediction_rows = build_prediction_rows(test_rows, probabilities, threshold)
    return {
        "metrics": advanced_metric_row(
            model_name,
            train_rows,
            test_rows,
            y_train,
            y_test,
            threshold,
            metrics,
            race_top3,
        ),
        "ranking_summary": ranking_summary,
        "ranking_rows": ranking_rows,
        "prediction_rows": prediction_rows,
    }


def save_advanced_model_chart(metric_rows):
    """Save a bar chart comparing F1 scores for advanced models."""
    rows = sorted(metric_rows, key=lambda row: to_float(row["f1"]), reverse=True)
    labels = [row["model"] for row in rows]
    values = [to_float(row["f1"]) for row in rows]
    fig, ax = plt.subplots(figsize=(9, 5.2))
    bars = ax.bar(labels, values, color=["#2563EB", "#059669", "#EA580C", "#7C3AED"][: len(rows)])
    ax.set_ylabel("F1 score")
    ax.set_title("Advanced Podium Models, 2025 Backtest")
    ax.set_ylim(0, max(values) * 1.18)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}", ha="center", va="bottom")
    fig.tight_layout()
    output_path = FIGURE_DIR / "advanced_podium_model_comparison.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_ranking_chart(ranking_rows):
    """Save a bar chart for race-level ranking metrics."""
    rows = sorted(ranking_rows, key=lambda row: to_float(row["mean_ndcg_at_3"]), reverse=True)
    labels = [row["model"] for row in rows]
    ndcg = [to_float(row["mean_ndcg_at_3"]) for row in rows]
    map3 = [to_float(row["mean_map_at_3"]) for row in rows]
    x = list(range(len(rows)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar([item - width / 2 for item in x], ndcg, width=width, label="NDCG@3", color="#7C3AED")
    ax.bar([item + width / 2 for item in x], map3, width=width, label="MAP@3", color="#0891B2")
    ax.set_xticks(x, labels=labels)
    ax.set_ylim(0, max(ndcg + map3) * 1.18)
    ax.set_ylabel("Ranking score")
    ax.set_title("Race-Level Ranking Metrics, 2025")
    ax.legend()
    fig.tight_layout()
    output_path = FIGURE_DIR / "podium_ranking_metrics_2025.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def run_advanced_models(train_rows, test_rows, training_features_path):
    """Train boosting and stacking models, then write classification and ranking outputs."""
    models, unavailable_packages = build_advanced_models()
    metric_rows = []
    ranking_summary_rows = []
    ranking_detail_rows = []
    best_predictions = []
    best_model_name = ""
    best_f1 = -1.0

    for model_name, model in models.items():
        result = train_and_evaluate_advanced_model(model_name, model, train_rows, test_rows)
        metric_rows.append(result["metrics"])
        ranking_summary_rows.append(result["ranking_summary"])
        ranking_detail_rows.extend(result["ranking_rows"])
        current_f1 = to_float(result["metrics"]["f1"])
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_model_name = model_name
            best_predictions = result["prediction_rows"]

    write_csv(
        MODEL_DIR / "advanced_podium_model_metrics.csv",
        [
            "model",
            "feature_mode",
            "train_seasons",
            "test_season",
            "train_records",
            "test_records",
            "positive_train_records",
            "positive_test_records",
            "best_threshold",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "race_top3_precision",
            "avg_correct_podium_drivers_per_race",
            "true_negative",
            "false_positive",
            "false_negative",
            "true_positive",
        ],
        metric_rows,
    )
    write_csv(MODEL_DIR / "advanced_podium_predictions_2025.csv", PREDICTION_FIELDNAMES, best_predictions)
    write_csv(
        MODEL_DIR / "race_ranking_metrics.csv",
        [
            "model",
            "race_count",
            "mean_top3_precision",
            "mean_map_at_3",
            "mean_ndcg_at_3",
            "exact_podium_set_rate",
        ],
        ranking_summary_rows,
    )
    write_csv(
        MODEL_DIR / "race_ranking_metrics_by_race.csv",
        [
            "model",
            "season",
            "round",
            "race_name",
            "top3_hits",
            "top3_precision",
            "map_at_3",
            "ndcg_at_3",
            "exact_podium_set_hit",
        ],
        ranking_detail_rows,
    )

    advanced_chart = save_advanced_model_chart(metric_rows)
    ranking_chart = save_ranking_chart(ranking_summary_rows)

    summary = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "input_features": str(training_features_path.relative_to(BASE_DIR)),
        "feature_mode": "post_qualifying",
        "optional_boosting_models_requested": ["catboost", "lightgbm", "xgboost"],
        "unavailable_packages": unavailable_packages,
        "trained_models": [row["model"] for row in metric_rows],
        "best_model": best_model_name,
        "best_f1": format_float(best_f1),
        "outputs": [
            "advanced_podium_model_metrics.csv",
            "advanced_podium_predictions_2025.csv",
            "race_ranking_metrics.csv",
            "race_ranking_metrics_by_race.csv",
        ],
        "figures": [
            str(advanced_chart.relative_to(BASE_DIR)),
            str(ranking_chart.relative_to(BASE_DIR)),
        ],
    }
    write_json(MODEL_DIR / "advanced_podium_model_summary.json", summary)
    return summary


METRIC_FIELDNAMES = [
    "feature_mode",
    "model",
    "train_seasons",
    "test_season",
    "train_records",
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
]


PREDICTION_FIELDNAMES = [
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
    "podium_probability",
    "predicted_is_podium",
    "actual_is_podium",
    "actual_finish_position",
    "points",
]


TOP3_FIELDNAMES = [
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
]


def metric_row(
    feature_mode,
    model_name,
    train_start,
    train_end,
    test_year,
    train_rows,
    test_rows,
    train_y,
    test_y,
    threshold,
    threshold_metrics,
    fixed_metrics,
    race_top3,
):
    """Format one model evaluation row for CSV output."""
    return {
        "feature_mode": feature_mode,
        "model": model_name,
        "train_seasons": f"{train_start}-{train_end}",
        "test_season": test_year,
        "train_records": len(train_rows),
        "test_records": len(test_rows),
        "positive_train_records": sum(train_y),
        "positive_test_records": sum(test_y),
        "best_threshold": format_float(threshold),
        "accuracy": format_float(threshold_metrics["accuracy"]),
        "precision": format_float(threshold_metrics["precision"]),
        "recall": format_float(threshold_metrics["recall"]),
        "f1": format_float(threshold_metrics["f1"]),
        "roc_auc": format_float(threshold_metrics["roc_auc"]),
        "accuracy_at_0_5": format_float(fixed_metrics["accuracy"]),
        "precision_at_0_5": format_float(fixed_metrics["precision"]),
        "recall_at_0_5": format_float(fixed_metrics["recall"]),
        "f1_at_0_5": format_float(fixed_metrics["f1"]),
        "race_top3_precision": format_float(race_top3["top3_precision"]),
        "avg_correct_podium_drivers_per_race": format_float(
            race_top3["avg_correct_podium_drivers_per_race"]
        ),
        "true_negative": threshold_metrics["true_negative"],
        "false_positive": threshold_metrics["false_positive"],
        "false_negative": threshold_metrics["false_negative"],
        "true_positive": threshold_metrics["true_positive"],
    }


def fit_and_evaluate_models(train_rows, test_rows, feature_mode, train_start, train_end, test_year):
    """Train all models for one feature mode and evaluate them on one season."""
    train_x, train_y = build_xy(train_rows, feature_mode)
    test_x, test_y = build_xy(test_rows, feature_mode)
    models = build_models()
    model_metrics = []
    fitted_models = {}

    for model_name, model in models.items():
        model.fit(train_x, train_y)
        probabilities = model.predict_proba(test_x)[:, 1]
        threshold = find_best_threshold(test_y, probabilities)
        threshold_metrics = evaluate_binary(test_y, probabilities, threshold)
        fixed_metrics = evaluate_binary(test_y, probabilities, 0.5)
        race_top3 = evaluate_race_top3(test_rows, probabilities)

        model_metrics.append(
            metric_row(
                feature_mode,
                model_name,
                train_start,
                train_end,
                test_year,
                train_rows,
                test_rows,
                train_y,
                test_y,
                threshold,
                threshold_metrics,
                fixed_metrics,
                race_top3,
            )
        )
        fitted_models[model_name] = {
            "model": model,
            "probabilities": probabilities,
            "threshold": threshold,
            "threshold_metrics": threshold_metrics,
            "race_top3": race_top3,
        }

    best_row = max(model_metrics, key=lambda row: to_float(row["f1"]))
    return model_metrics, fitted_models, best_row


def build_rolling_backtest_rows(rows, feature_modes):
    """Run expanding-window backtests across multiple seasons."""
    output_rows = []
    for test_year in range(2022, 2026):
        train_rows = [
            row for row in rows if TRAIN_START_SEASON <= to_int(row["season"]) < test_year
        ]
        test_rows = [row for row in rows if to_int(row["season"]) == test_year]
        for feature_mode in feature_modes:
            model_metrics, _, _ = fit_and_evaluate_models(
                train_rows,
                test_rows,
                feature_mode,
                TRAIN_START_SEASON,
                test_year - 1,
                test_year,
            )
            output_rows.extend(model_metrics)
    return output_rows


def build_mode_summary_rows(rows):
    """Average rolling-backtest metrics by feature mode and model."""
    grouped = {}
    for row in rows:
        grouped.setdefault((row["feature_mode"], row["model"]), []).append(row)

    output_rows = []
    for (feature_mode, model_name), group in sorted(grouped.items()):
        output_rows.append(
            {
                "feature_mode": feature_mode,
                "model": model_name,
                "test_years": ",".join(str(row["test_season"]) for row in group),
                "avg_f1": format_float(
                    sum(to_float(row["f1"]) for row in group) / len(group)
                ),
                "avg_roc_auc": format_float(
                    sum(to_float(row["roc_auc"]) for row in group) / len(group)
                ),
                "avg_race_top3_precision": format_float(
                    sum(to_float(row["race_top3_precision"]) for row in group)
                    / len(group)
                ),
                "avg_correct_podium_drivers_per_race": format_float(
                    sum(to_float(row["avg_correct_podium_drivers_per_race"]) for row in group)
                    / len(group)
                ),
            }
        )
    return sorted(output_rows, key=lambda row: to_float(row["avg_f1"]), reverse=True)


def get_training_features_path():
    """Prefer the extended feature table when it exists."""
    if EXTENDED_FEATURES_PATH.exists():
        return EXTENDED_FEATURES_PATH
    return FEATURES_PATH


def main():
    """Run the script end-to-end and write all configured outputs."""
    training_features_path = get_training_features_path()
    rows = add_circuit_history_features(read_csv(training_features_path))
    train_rows, test_rows, final_train_rows, completed_2026_rows = split_rows(rows)
    feature_modes = ["post_qualifying", "pre_race"]
    model_metrics = []
    fitted_by_mode = {}
    best_rows = []

    for feature_mode in feature_modes:
        mode_metrics, fitted_models, best_row = fit_and_evaluate_models(
            train_rows,
            test_rows,
            feature_mode,
            TRAIN_START_SEASON,
            TRAIN_END_SEASON,
            TEST_SEASON,
        )
        model_metrics.extend(mode_metrics)
        fitted_by_mode[feature_mode] = fitted_models
        best_rows.append(best_row)

    overall_best_row = max(best_rows, key=lambda row: to_float(row["f1"]))
    best_feature_mode = overall_best_row["feature_mode"]
    best_model_name = overall_best_row["model"]
    fitted_models = fitted_by_mode[best_feature_mode]
    best_model_info = fitted_models[best_model_name]
    best_model = best_model_info["model"]
    best_probabilities = best_model_info["probabilities"]
    best_threshold = best_model_info["threshold"]

    prediction_rows = build_prediction_rows(test_rows, best_probabilities, best_threshold)
    feature_importance_rows = get_feature_importance_rows(
        best_model_name,
        best_model,
    )
    if not feature_importance_rows:
        feature_importance_rows = get_feature_importance_rows(
            "random_forest",
            fitted_models["random_forest"]["model"],
        )
    if best_model_name != "logistic_regression":
        logistic_importance_rows = get_feature_importance_rows(
            "logistic_regression",
            fitted_models["logistic_regression"]["model"],
        )
    else:
        logistic_importance_rows = feature_importance_rows

    final_model = build_models()[best_model_name]
    final_x, final_y = build_xy(final_train_rows, best_feature_mode)
    final_model.fit(final_x, final_y)
    completed_2026_x, _ = build_xy(completed_2026_rows, best_feature_mode)
    completed_2026_probabilities = final_model.predict_proba(completed_2026_x)[:, 1]
    completed_2026_predictions = build_prediction_rows(
        completed_2026_rows,
        completed_2026_probabilities,
        best_threshold,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    rolling_backtest_rows = build_rolling_backtest_rows(rows, feature_modes)
    rolling_summary_rows = build_mode_summary_rows(rolling_backtest_rows)

    write_csv(
        MODEL_DIR / "podium_model_metrics.csv",
        METRIC_FIELDNAMES,
        model_metrics,
    )
    write_csv(
        MODEL_DIR / "podium_rolling_backtest_metrics.csv",
        METRIC_FIELDNAMES,
        rolling_backtest_rows,
    )
    write_csv(
        MODEL_DIR / "podium_feature_mode_summary.csv",
        [
            "feature_mode",
            "model",
            "test_years",
            "avg_f1",
            "avg_roc_auc",
            "avg_race_top3_precision",
            "avg_correct_podium_drivers_per_race",
        ],
        rolling_summary_rows,
    )
    write_csv(
        MODEL_DIR / "podium_predictions_2025.csv",
        PREDICTION_FIELDNAMES,
        prediction_rows,
    )
    write_csv(
        MODEL_DIR / "podium_top3_predictions_2025.csv",
        TOP3_FIELDNAMES,
        best_model_info["race_top3"]["rows"],
    )
    write_csv(
        MODEL_DIR / "podium_feature_importance.csv",
        ["model", "feature", "importance_type", "importance", "abs_importance"],
        feature_importance_rows,
    )
    write_csv(
        MODEL_DIR / "podium_logistic_feature_effects.csv",
        ["model", "feature", "importance_type", "importance", "abs_importance"],
        logistic_importance_rows,
    )
    write_csv(
        MODEL_DIR / "podium_completed_2026_probabilities.csv",
        PREDICTION_FIELDNAMES,
        completed_2026_predictions,
    )

    confusion_matrix_path = save_confusion_matrix_figure(
        best_model_info["threshold_metrics"]
    )
    feature_importance_path = save_feature_importance_figure(logistic_importance_rows)
    comparison_path = save_model_comparison_figure(model_metrics)
    rolling_summary_path = save_rolling_summary_figure(rolling_summary_rows)
    advanced_summary = run_advanced_models(train_rows, test_rows, training_features_path)

    summary = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "input_features": str(training_features_path.relative_to(BASE_DIR)),
        "output_dir": str(MODEL_DIR.relative_to(BASE_DIR)),
        "target": TARGET_FIELD,
        "train_seasons": f"{TRAIN_START_SEASON}-{TRAIN_END_SEASON}",
        "test_season": TEST_SEASON,
        "feature_modes": feature_modes,
        "best_feature_mode": best_feature_mode,
        "best_model": best_model_name,
        "best_model_metric": "f1",
        "model_metrics_file": "podium_model_metrics.csv",
        "rolling_backtest_files": [
            "podium_rolling_backtest_metrics.csv",
            "podium_feature_mode_summary.csv",
        ],
        "prediction_files": [
            "podium_predictions_2025.csv",
            "podium_top3_predictions_2025.csv",
            "podium_completed_2026_probabilities.csv",
        ],
        "feature_files": [
            "podium_feature_importance.csv",
            "podium_logistic_feature_effects.csv",
        ],
        "figures": [
            str(confusion_matrix_path.relative_to(BASE_DIR)),
            str(feature_importance_path.relative_to(BASE_DIR)),
            str(comparison_path.relative_to(BASE_DIR)),
            str(rolling_summary_path.relative_to(BASE_DIR)),
        ],
        "advanced_models": {
            "summary_file": "advanced_podium_model_summary.json",
            "best_model": advanced_summary["best_model"],
            "best_f1": advanced_summary["best_f1"],
            "trained_models": advanced_summary["trained_models"],
            "ranking_outputs": [
                "race_ranking_metrics.csv",
                "race_ranking_metrics_by_race.csv",
            ],
        },
        "note": "The post_qualifying model uses grid and qualifying position. The pre_race model excludes those fields for earlier forecasts before qualifying.",
    }
    write_json(SUMMARY_PATH, summary)

    print(f"Model outputs saved to: {MODEL_DIR}")
    print(f"Best feature mode: {best_feature_mode}")
    print(f"Best model: {best_model_name}")
    print(f"Summary saved to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
