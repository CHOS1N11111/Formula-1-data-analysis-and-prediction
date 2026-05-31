"""Generate comparison figures for all Formula 1 podium prediction model families.

The script combines traditional ML, advanced boosting/stacking, and deep-learning metric files, then creates unified F1, ROC-AUC, Top3 Precision, heatmap, best-family, rolling-backtest, and ranking visualizations."""

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
ADVANCED_METRICS_PATH = MODEL_DIR / "advanced_podium_model_metrics.csv"
DL_METRICS_PATH = MODEL_DIR / "deep_podium_model_metrics.csv"
ROLLING_SUMMARY_PATH = MODEL_DIR / "podium_feature_mode_summary.csv"
RANKING_METRICS_PATH = MODEL_DIR / "race_ranking_metrics.csv"

FAMILY_COLORS = {
    "Traditional ML": "#2563EB",
    "Advanced ML": "#EA580C",
    "Deep Learning": "#DC2626",
}


def read_csv(path):
    """Read a CSV file as a list of dictionaries."""
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))



def write_json(path, data):
    """Write structured metadata to a UTF-8 JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def to_float(value, default=0.0):
    """Convert a value to float and return the default for missing or invalid values."""
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def model_label(row):
    """Build a compact chart label that includes model family and feature mode."""
    family_prefix = {
        "Traditional ML": "ML",
        "Advanced ML": "Boost",
        "Deep Learning": "DL",
    }.get(row["model_family"], row["model_family"])
    return f"{family_prefix}: {row['feature_mode']}\n{row['model']}"


def combined_metric_rows():
    """Combine model metric rows from traditional ML, advanced ML, and DL outputs."""
    rows = []
    for row in read_csv(ML_METRICS_PATH):
        enriched = dict(row)
        enriched["model_family"] = "Traditional ML"
        rows.append(enriched)
    if ADVANCED_METRICS_PATH.exists():
        for row in read_csv(ADVANCED_METRICS_PATH):
            enriched = dict(row)
            enriched["model_family"] = "Advanced ML"
            rows.append(enriched)
    for row in read_csv(DL_METRICS_PATH):
        enriched = dict(row)
        enriched["model_family"] = "Deep Learning"
        rows.append(enriched)
    return rows


def save_grouped_metric_chart(rows, metric, title, filename, ylabel):
    """Save one grouped bar chart for a selected metric across all models."""
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            row["feature_mode"],
            family_sort_key(row["model_family"]),
            row["model"],
        ),
    )
    labels = [model_label(row) for row in sorted_rows]
    values = [to_float(row[metric]) for row in sorted_rows]
    colors = [FAMILY_COLORS[row["model_family"]] for row in sorted_rows]

    fig, ax = plt.subplots(figsize=(16, 7.4))
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


def family_sort_key(model_family):
    """Return a stable ordering for model families in charts."""
    order = {"Traditional ML": 0, "Advanced ML": 1, "Deep Learning": 2}
    return order.get(model_family, 99)



def save_best_model_chart(rows):
    """Save the best-per-family model comparison chart."""
    best_by_family_mode = {}
    for row in rows:
        key = (row["model_family"], row["feature_mode"])
        if key not in best_by_family_mode or to_float(row["f1"]) > to_float(
            best_by_family_mode[key]["f1"]
        ):
            best_by_family_mode[key] = row

    best_rows = sorted(
        best_by_family_mode.values(),
        key=lambda row: (row["feature_mode"], family_sort_key(row["model_family"])),
    )
    labels = [model_label(row) for row in best_rows]
    f1_values = [to_float(row["f1"]) for row in best_rows]
    auc_values = [to_float(row["roc_auc"]) for row in best_rows]

    x = list(range(len(best_rows)))
    width = 0.36
    fig, ax = plt.subplots(figsize=(12, 6.5))
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
    ax.set_title("Best Podium Models by Family and Feature Mode")
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
    output_path = FIGURE_DIR / "podium_best_models_by_family.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_all_model_heatmap(rows):
    """Save a compact heatmap comparing all model families across metrics."""
    metric_fields = [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1"),
        ("roc_auc", "ROC-AUC"),
        ("race_top3_precision", "Top3 Precision"),
    ]
    sorted_rows = sorted(
        rows,
        key=lambda row: (
            row["feature_mode"],
            family_sort_key(row["model_family"]),
            -to_float(row["f1"]),
        ),
    )
    values = [
        [to_float(row[field]) for field, _ in metric_fields]
        for row in sorted_rows
    ]
    labels = [model_label(row).replace("\n", " | ") for row in sorted_rows]

    fig, ax = plt.subplots(figsize=(11.5, 8.5))
    image = ax.imshow(values, cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(metric_fields)), labels=[label for _, label in metric_fields])
    ax.set_yticks(range(len(sorted_rows)), labels=labels)
    ax.set_title("All Podium Models: Metric Heatmap, 2025 Backtest")
    for row_index, row_values in enumerate(values):
        for col_index, value in enumerate(row_values):
            ax.text(
                col_index,
                row_index,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color="#111827" if value < 0.72 else "white",
            )
    fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    fig.tight_layout()
    output_path = FIGURE_DIR / "podium_all_model_metric_heatmap.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_advanced_ranking_chart():
    """Save ranking-oriented metrics for boosting and stacking models."""
    if not RANKING_METRICS_PATH.exists():
        return None
    rows = sorted(
        read_csv(RANKING_METRICS_PATH),
        key=lambda row: to_float(row["mean_ndcg_at_3"]),
        reverse=True,
    )
    if not rows:
        return None

    labels = [row["model"] for row in rows]
    metrics = [
        ("mean_top3_precision", "Top3 Precision", "#2563EB"),
        ("mean_map_at_3", "MAP@3", "#059669"),
        ("mean_ndcg_at_3", "NDCG@3", "#7C3AED"),
        ("exact_podium_set_rate", "Exact Set Rate", "#EA580C"),
    ]
    x = list(range(len(rows)))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10.5, 6))
    for metric_index, (field, label, color) in enumerate(metrics):
        offset = (metric_index - 1.5) * width
        values = [to_float(row[field]) for row in rows]
        ax.bar(
            [item + offset for item in x],
            values,
            width=width,
            label=label,
            color=color,
        )

    ax.set_xticks(x, labels=labels)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Advanced Models: Race-Level Ranking Metrics, 2025")
    ax.legend(ncol=2)
    fig.tight_layout()
    output_path = FIGURE_DIR / "podium_advanced_ranking_metrics_all.png"
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_rolling_vs_2025_chart(rows):
    """Compare 2025 F1 scores with rolling-backtest averages."""
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
    """Run the script end-to-end and write all configured outputs."""
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    rows = combined_metric_rows()
    figures = [
        save_grouped_metric_chart(
            rows,
            "f1",
            "Podium Prediction F1: All Model Families",
            "podium_all_models_f1_comparison.png",
            "F1 score",
        ),
        save_grouped_metric_chart(
            rows,
            "roc_auc",
            "Podium Prediction ROC-AUC: All Model Families",
            "podium_all_models_roc_auc_comparison.png",
            "ROC-AUC",
        ),
        save_grouped_metric_chart(
            rows,
            "race_top3_precision",
            "Race-Level Top 3 Precision: All Model Families",
            "podium_all_models_top3_precision.png",
            "Top 3 precision",
        ),
        save_best_model_chart(rows),
        save_all_model_heatmap(rows),
        save_advanced_ranking_chart(),
        save_rolling_vs_2025_chart(rows),
    ]
    figures = [path for path in figures if path is not None]

    write_json(
        MANIFEST_PATH,
        {
            "built_at": datetime.now(timezone.utc).isoformat(),
            "source_files": [
                str(ML_METRICS_PATH.relative_to(BASE_DIR)),
                str(ADVANCED_METRICS_PATH.relative_to(BASE_DIR)),
                str(DL_METRICS_PATH.relative_to(BASE_DIR)),
                str(ROLLING_SUMMARY_PATH.relative_to(BASE_DIR)),
                str(RANKING_METRICS_PATH.relative_to(BASE_DIR)),
            ],
            "figures": [str(path.relative_to(BASE_DIR)) for path in figures],
        },
    )

    print(f"Model result figures saved to: {FIGURE_DIR}")
    print(f"Figure count: {len(figures)}")


if __name__ == "__main__":
    main()
