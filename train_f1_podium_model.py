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
MODEL_DIR = BASE_DIR / "data" / "modeling"
FIGURE_DIR = BASE_DIR / "outputs" / "figures"
SUMMARY_PATH = MODEL_DIR / "podium_model_summary.json"

TARGET_FIELD = "is_podium"
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


def build_feature_dict(row):
    feature_row = {}
    for field in NUMERIC_FEATURES:
        feature_row[field] = to_float(row.get(field), 0.0)
    for field in CATEGORICAL_FEATURES:
        feature_row[field] = row.get(field, "unknown") or "unknown"
    return feature_row


def split_rows(rows):
    train_rows = [
        row
        for row in rows
        if 2019 <= to_int(row["season"]) <= TRAIN_END_SEASON
    ]
    test_rows = [row for row in rows if to_int(row["season"]) == TEST_SEASON]
    final_train_rows = [
        row for row in rows if 2019 <= to_int(row["season"]) <= TEST_SEASON
    ]
    completed_2026_rows = [row for row in rows if to_int(row["season"]) == 2026]
    return train_rows, test_rows, final_train_rows, completed_2026_rows


def build_xy(rows):
    x_values = [build_feature_dict(row) for row in rows]
    y_values = [to_int(row[TARGET_FIELD]) for row in rows]
    return x_values, y_values


def build_models():
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
    vectorizer = model.named_steps["vectorizer"]
    return vectorizer.get_feature_names_out()


def get_feature_importance_rows(model_name, model, top_n=40):
    feature_names = get_feature_names(model)
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


def main():
    rows = read_csv(FEATURES_PATH)
    train_rows, test_rows, final_train_rows, completed_2026_rows = split_rows(rows)
    train_x, train_y = build_xy(train_rows)
    test_x, test_y = build_xy(test_rows)

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
            {
                "model": model_name,
                "train_seasons": f"2019-{TRAIN_END_SEASON}",
                "test_season": TEST_SEASON,
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
        )
        fitted_models[model_name] = {
            "model": model,
            "probabilities": probabilities,
            "threshold": threshold,
            "threshold_metrics": threshold_metrics,
            "race_top3": race_top3,
        }

    best_model_name = max(model_metrics, key=lambda row: to_float(row["f1"]))["model"]
    best_model_info = fitted_models[best_model_name]
    best_model = best_model_info["model"]
    best_probabilities = best_model_info["probabilities"]
    best_threshold = best_model_info["threshold"]

    prediction_rows = build_prediction_rows(test_rows, best_probabilities, best_threshold)
    feature_importance_rows = get_feature_importance_rows(
        best_model_name,
        best_model,
    )
    if best_model_name != "logistic_regression":
        logistic_importance_rows = get_feature_importance_rows(
            "logistic_regression",
            fitted_models["logistic_regression"]["model"],
        )
    else:
        logistic_importance_rows = feature_importance_rows

    final_model = build_models()[best_model_name]
    final_x, final_y = build_xy(final_train_rows)
    final_model.fit(final_x, final_y)
    completed_2026_x, _ = build_xy(completed_2026_rows)
    completed_2026_probabilities = final_model.predict_proba(completed_2026_x)[:, 1]
    completed_2026_predictions = build_prediction_rows(
        completed_2026_rows,
        completed_2026_probabilities,
        best_threshold,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(
        MODEL_DIR / "podium_model_metrics.csv",
        [
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
        ],
        model_metrics,
    )
    write_csv(
        MODEL_DIR / "podium_predictions_2025.csv",
        [
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
        ],
        prediction_rows,
    )
    write_csv(
        MODEL_DIR / "podium_top3_predictions_2025.csv",
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
        [
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
        ],
        completed_2026_predictions,
    )

    confusion_matrix_path = save_confusion_matrix_figure(
        best_model_info["threshold_metrics"]
    )
    feature_importance_path = save_feature_importance_figure(logistic_importance_rows)

    summary = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "input_features": str(FEATURES_PATH.relative_to(BASE_DIR)),
        "output_dir": str(MODEL_DIR.relative_to(BASE_DIR)),
        "target": TARGET_FIELD,
        "train_seasons": f"2019-{TRAIN_END_SEASON}",
        "test_season": TEST_SEASON,
        "best_model": best_model_name,
        "best_model_metric": "f1",
        "model_metrics_file": "podium_model_metrics.csv",
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
        ],
        "note": "2026 future races are not predicted yet because future qualifying/grid positions are unknown in the current feature table.",
    }
    write_json(SUMMARY_PATH, summary)

    print(f"Model outputs saved to: {MODEL_DIR}")
    print(f"Best model: {best_model_name}")
    print(f"Summary saved to: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
