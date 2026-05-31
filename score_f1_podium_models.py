"""Compute one composite score for every F1 podium prediction model."""

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

ML_METRICS_PATH = MODEL_DIR / "podium_model_metrics.csv"
ADVANCED_METRICS_PATH = MODEL_DIR / "advanced_podium_model_metrics.csv"
DL_METRICS_PATH = MODEL_DIR / "deep_podium_model_metrics.csv"
RANKING_METRICS_PATH = MODEL_DIR / "race_ranking_metrics.csv"

SCORE_OUTPUT_PATH = MODEL_DIR / "podium_model_composite_scores.csv"
SUMMARY_OUTPUT_PATH = MODEL_DIR / "podium_model_composite_score_summary.json"
FIGURE_OUTPUT_PATH = FIGURE_DIR / "podium_model_composite_score.png"

# Common metrics are used for every model family so ML, advanced ML, and DL are comparable.
COMMON_SCORE_WEIGHTS = {
    "f1": 0.35,
    "roc_auc": 0.25,
    "race_top3_precision": 0.20,
    "precision": 0.10,
    "recall": 0.10,
}

# Ranking metrics are reported as an extra diagnostic for advanced models only.
RANKING_SCORE_WEIGHTS = {
    "mean_top3_precision": 0.35,
    "mean_map_at_3": 0.25,
    "mean_ndcg_at_3": 0.25,
    "exact_podium_set_rate": 0.15,
}


def read_csv(path):
    if not path.exists():
        return []
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


def to_float(value, default=0.0):
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def format_float(value, digits=6):
    return f"{value:.{digits}f}"


def load_model_rows():
    """Load all available model metrics and attach a model-family label."""
    rows = []
    for row in read_csv(ML_METRICS_PATH):
        enriched = dict(row)
        enriched["model_family"] = "Traditional ML"
        rows.append(enriched)
    for row in read_csv(ADVANCED_METRICS_PATH):
        enriched = dict(row)
        enriched["model_family"] = "Advanced ML"
        rows.append(enriched)
    for row in read_csv(DL_METRICS_PATH):
        enriched = dict(row)
        enriched["model_family"] = "Deep Learning"
        rows.append(enriched)
    return rows


def load_ranking_scores():
    """Build optional race-ranking scores for models that have ranking outputs."""
    ranking_scores = {}
    for row in read_csv(RANKING_METRICS_PATH):
        score = sum(
            to_float(row[field]) * weight
            for field, weight in RANKING_SCORE_WEIGHTS.items()
        )
        ranking_scores[row["model"]] = {
            "ranking_score": score,
            "mean_top3_precision": to_float(row["mean_top3_precision"]),
            "mean_map_at_3": to_float(row["mean_map_at_3"]),
            "mean_ndcg_at_3": to_float(row["mean_ndcg_at_3"]),
            "exact_podium_set_rate": to_float(row["exact_podium_set_rate"]),
        }
    return ranking_scores


def composite_score(row):
    """Calculate a 0-100 score from metrics shared by all model families."""
    return 100 * sum(
        to_float(row[field]) * weight
        for field, weight in COMMON_SCORE_WEIGHTS.items()
    )


def score_rows(rows, ranking_scores):
    output_rows = []
    for row in rows:
        ranking = ranking_scores.get(row["model"], {})
        common_score = composite_score(row)
        output_rows.append(
            {
                "rank": 0,
                "model_family": row["model_family"],
                "feature_mode": row["feature_mode"],
                "model": row["model"],
                "composite_score_100": format_float(common_score),
                "f1": row["f1"],
                "roc_auc": row["roc_auc"],
                "race_top3_precision": row["race_top3_precision"],
                "precision": row["precision"],
                "recall": row["recall"],
                "ranking_score_100": format_float(
                    ranking["ranking_score"] * 100
                )
                if ranking
                else "",
                "mean_map_at_3": format_float(ranking.get("mean_map_at_3", 0.0))
                if ranking
                else "",
                "mean_ndcg_at_3": format_float(ranking.get("mean_ndcg_at_3", 0.0))
                if ranking
                else "",
                "exact_podium_set_rate": format_float(
                    ranking.get("exact_podium_set_rate", 0.0)
                )
                if ranking
                else "",
            }
        )

    sorted_rows = sorted(
        output_rows,
        key=lambda item: to_float(item["composite_score_100"]),
        reverse=True,
    )
    for rank, row in enumerate(sorted_rows, start=1):
        row["rank"] = rank
    return sorted_rows


def save_score_figure(rows):
    """Save a horizontal ranking chart for the composite model scores."""
    top_rows = list(reversed(rows))
    labels = [
        f"{row['rank']}. {row['model_family']} | {row['feature_mode']} | {row['model']}"
        for row in top_rows
    ]
    values = [to_float(row["composite_score_100"]) for row in top_rows]
    color_map = {
        "Traditional ML": "#2563EB",
        "Advanced ML": "#EA580C",
        "Deep Learning": "#DC2626",
    }
    colors = [color_map[row["model_family"]] for row in top_rows]

    fig_height = max(6.5, len(top_rows) * 0.42)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    bars = ax.barh(labels, values, color=colors)
    ax.set_xlim(0, max(values) * 1.12)
    ax.set_xlabel("Composite score out of 100")
    ax.set_title("F1 Podium Prediction Model Composite Scores")
    for bar, value in zip(bars, values):
        ax.text(
            value + 0.4,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}",
            va="center",
            fontsize=8,
        )
    fig.tight_layout()
    FIGURE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_OUTPUT_PATH, dpi=180)
    plt.close(fig)
    return FIGURE_OUTPUT_PATH


def main():
    rows = load_model_rows()
    if not rows:
        raise FileNotFoundError("No model metric files were found in data/modeling.")

    ranking_scores = load_ranking_scores()
    scored_rows = score_rows(rows, ranking_scores)
    score_figure = save_score_figure(scored_rows)

    write_csv(
        SCORE_OUTPUT_PATH,
        [
            "rank",
            "model_family",
            "feature_mode",
            "model",
            "composite_score_100",
            "f1",
            "roc_auc",
            "race_top3_precision",
            "precision",
            "recall",
            "ranking_score_100",
            "mean_map_at_3",
            "mean_ndcg_at_3",
            "exact_podium_set_rate",
        ],
        scored_rows,
    )

    best_row = scored_rows[0]
    write_json(
        SUMMARY_OUTPUT_PATH,
        {
            "built_at": datetime.now(timezone.utc).isoformat(),
            "score_formula": COMMON_SCORE_WEIGHTS,
            "score_explanation": "Composite score = weighted common metrics * 100. Ranking metrics are reported separately because only advanced models have MAP@3 and NDCG@3.",
            "model_count": len(scored_rows),
            "best_model": {
                "rank": best_row["rank"],
                "model_family": best_row["model_family"],
                "feature_mode": best_row["feature_mode"],
                "model": best_row["model"],
                "composite_score_100": best_row["composite_score_100"],
            },
            "outputs": [
                str(SCORE_OUTPUT_PATH.relative_to(BASE_DIR)),
                str(score_figure.relative_to(BASE_DIR)),
            ],
        },
    )

    print(f"Model score table saved to: {SCORE_OUTPUT_PATH}")
    print(f"Model score figure saved to: {score_figure}")
    print(
        "Best model: "
        f"{best_row['model_family']} / {best_row['feature_mode']} / {best_row['model']} "
        f"with score {best_row['composite_score_100']}"
    )


if __name__ == "__main__":
    main()
