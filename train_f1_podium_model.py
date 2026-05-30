import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
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
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def to_int(value, default=0):
    try:
        if value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def to_float(value, default=0.0):
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def format_float(value, digits=6):
    return f"{value:.{digits}f}"


def safe_rate(numerator, denominator):
    if denominator == 0:
        return 0.0
    return numerator / denominator


def default_circuit_stats():
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
    if feature_mode == "pre_race":
        return PRE_RACE_NUMERIC_FEATURES
    return POST_QUALIFYING_NUMERIC_FEATURES


def build_feature_dict(row, feature_mode):
    feature_row = {}
    for field in numeric_fields_for_mode(feature_mode):
        feature_row[field] = to_float(row.get(field), 0.0)
    for field in CATEGORICAL_FEATURES:
        feature_row[field] = row.get(field, "unknown") or "unknown"
    return feature_row


def split_rows(rows):
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
    x_values = [build_feature_dict(row, feature_mode) for row in rows]
    y_values = [to_int(row[TARGET_FIELD]) for row in rows]
    return x_values, y_values


def build_models():
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
    if not hasattr(model, "named_steps"):
        return []
    vectorizer = model.named_steps["vectorizer"]
    return vectorizer.get_feature_names_out()


def get_feature_importance_rows(model_name, model, top_n=40):
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
    if EXTENDED_FEATURES_PATH.exists():
        return EXTENDED_FEATURES_PATH
    return FEATURES_PATH


def main():
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
        "note": "The post_qualifying model uses grid and qualifying position. The pre_race model excludes those fields for earlier forecasts before qualifying.",
    }
    write_json(SUMMARY_PATH, summary)

    print(f"Model outputs saved to: {MODEL_DIR}")
    print(f"Best feature mode: {best_feature_mode}")
    print(f"Best model: {best_model_name}")
    print(f"Summary saved to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
