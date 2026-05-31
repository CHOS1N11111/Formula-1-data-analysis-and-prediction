"""Generate comparison figures for traditional ML and MLP podium models."""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "data" / "modeling"
FIGURE_DIR = BASE_DIR / "outputs" / "figures"
MANIFEST_PATH = FIGURE_DIR / "model_figure_manifest.json"

ML_METRICS_PATH = MODEL_DIR / "podium_model_metrics.csv"
DL_METRICS_PATH = MODEL_DIR / "deep_podium_model_metrics.csv"
ROLLING_SUMMARY_PATH = MODEL_DIR / "podium_feature_mode_summary.csv"


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def to_float(value, default=0.0):
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def model_label(row):
    prefix = "DL" if row.get("model") == "mlp_neural_network" else "ML"
    return f"{prefix}: {row['feature_mode']}\n{row['model']}"


def combined_metric_rows():
    rows = []
    for row in read_csv(ML_METRICS_PATH):
        enriched = dict(row)
        enriched["model_family"] = "Traditional ML"
        rows.append(enriched)
    for row in read_csv(DL_METRICS_PATH):
        enriched = dict(row)
        enriched["model_family"] = "Deep Learning"
        rows.append(enriched)
    return rows


def save_grouped_metric_chart(rows, metric, title, filename, ylabel):
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            row["feature_mode"],
            0 if row["model_family"] == "Traditional ML" else 1,
            row["model"],
        ),
    )
    labels = [model_label(row) for row in sorted_rows]
    values = [to_float(row[metric]) for row in sorted_rows]
    colors = [
        "#2563EB" if row["model_family"] == "Traditional ML" else "#DC2626"
        for row in sorted_rows
    ]

    fig, ax = plt.subplots(figsize=(13, 6.5))
    bars = ax.bar(range(len(sorted_rows)), values, color=colors)
    ax.set_xticks(range(len(sorted_rows)), labels=labels, rotation=40, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
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
    output_path = FIGURE_DIR / filename
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_best_model_chart(rows):
    best_by_family_mode = {}
    for row in rows:
        key = (row["model_family"], row["feature_mode"])
        if key not in best_by_family_mode or to_float(row["f1"]) > to_float(
            best_by_family_mode[key]["f1"]
        ):
            best_by_family_mode[key] = row

    best_rows = sorted(best_by_family_mode.values(), key=lambda row: row["feature_mode"])
    labels = [model_label(row) for row in best_rows]
    f1_values = [to_float(row["f1"]) for row in best_rows]
    auc_values = [to_float(row["roc_auc"]) for row in best_rows]

    x = list(range(len(best_rows)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 6))
    f1_bars = ax.bar(
        [item - width / 2 for item in x],
        f1_values,
        width=width,
        label="F1",
        color="#2563EB",
    )
    auc_bars = ax.bar(
        [item + width / 2 for item in x],
        auc_values,
        width=width,
        label="ROC-AUC",
        color="#059669",
    )
    ax.set_xticks(x, labels=labels, rotation=25, ha="right")
    ax.set_ylim(0, max(auc_values) * 1.16)
    ax.set_ylabel("Score")
    ax.set_title("Best ML vs DL Podium Models by Feature Mode")
    ax.legend()
    for bars in [f1_bars, auc_bars]:
        for bar in bars:
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    fig.tight_layout()
    output_path = FIGURE_DIR / "podium_best_ml_vs_dl_models.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_rolling_vs_2025_chart(rows):
    rolling_rows = read_csv(ROLLING_SUMMARY_PATH)
    rolling_lookup = {
        (row["feature_mode"], row["model"]): row for row in rolling_rows
    }
    comparable_rows = [
        row
        for row in rows
        if row["model_family"] == "Traditional ML"
        and (row["feature_mode"], row["model"]) in rolling_lookup
    ]
    comparable_rows = sorted(comparable_rows, key=lambda row: to_float(row["f1"]), reverse=True)

    labels = [f"{row['feature_mode']}\n{row['model']}" for row in comparable_rows]
    f1_2025 = [to_float(row["f1"]) for row in comparable_rows]
    f1_rolling = [
        to_float(rolling_lookup[(row["feature_mode"], row["model"])]["avg_f1"])
        for row in comparable_rows
    ]

    x = list(range(len(comparable_rows)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.bar(
        [item - width / 2 for item in x],
        f1_2025,
        width=width,
        label="2025 F1",
        color="#7C3AED",
    )
    ax.bar(
        [item + width / 2 for item in x],
        f1_rolling,
        width=width,
        label="Rolling average F1",
        color="#0891B2",
    )
    ax.set_xticks(x, labels=labels, rotation=35, ha="right")
    ax.set_ylim(0, max(f1_2025 + f1_rolling) * 1.18)
    ax.set_ylabel("F1 score")
    ax.set_title("Traditional ML: 2025 Backtest vs Rolling Average")
    ax.legend()
    fig.tight_layout()
    output_path = FIGURE_DIR / "podium_ml_2025_vs_rolling_f1.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def main():
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    rows = combined_metric_rows()
    figures = [
        save_grouped_metric_chart(
            rows,
            "f1",
            "Podium Prediction F1: Traditional ML vs MLP",
            "podium_ml_dl_f1_comparison.png",
            "F1 score",
        ),
        save_grouped_metric_chart(
            rows,
            "roc_auc",
            "Podium Prediction ROC-AUC: Traditional ML vs MLP",
            "podium_ml_dl_roc_auc_comparison.png",
            "ROC-AUC",
        ),
        save_grouped_metric_chart(
            rows,
            "race_top3_precision",
            "Race-Level Top 3 Precision: Traditional ML vs MLP",
            "podium_ml_dl_top3_precision.png",
            "Top 3 precision",
        ),
        save_best_model_chart(rows),
        save_rolling_vs_2025_chart(rows),
    ]

    write_json(
        MANIFEST_PATH,
        {
            "built_at": datetime.now(timezone.utc).isoformat(),
            "source_files": [
                str(ML_METRICS_PATH.relative_to(BASE_DIR)),
                str(DL_METRICS_PATH.relative_to(BASE_DIR)),
                str(ROLLING_SUMMARY_PATH.relative_to(BASE_DIR)),
            ],
            "figures": [str(path.relative_to(BASE_DIR)) for path in figures],
        },
    )

    print(f"Model result figures saved to: {FIGURE_DIR}")
    print(f"Figure count: {len(figures)}")


if __name__ == "__main__":
    main()
