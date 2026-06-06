"""Score and select robust 2026 championship prediction scenarios.

The final prediction keeps several model scenarios for sensitivity analysis.
This script ranks those scenarios with both validation performance and
race-level behavior diagnostics, so extremely concentrated winner patterns are
flagged instead of treated as equally reliable.
"""

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "data" / "modeling"

SCENARIO_SUMMARY_PATH = MODEL_DIR / "season_prediction_model_scenarios_2026.csv"
SCENARIO_DIAGNOSTICS_PATH = MODEL_DIR / "season_prediction_model_scenario_diagnostics_2026.csv"
RACE_SIGNAL_DIAGNOSTICS_PATH = MODEL_DIR / "season_prediction_race_signal_diagnostics_2026.csv"
CIRCUIT_ARCHETYPE_DIAGNOSTICS_PATH = MODEL_DIR / "season_prediction_circuit_archetype_diagnostics_2026.csv"

OUTPUT_PATH = MODEL_DIR / "season_prediction_scenario_selection_2026.csv"
SUMMARY_JSON_PATH = MODEL_DIR / "season_prediction_scenario_selection_summary_2026.json"

OUTPUT_FIELDS = [
    "scenario_rank",
    "top10_model",
    "points_model",
    "top10_f1",
    "points_mae",
    "performance_score",
    "winner_diversity_score",
    "winner_concentration_penalty",
    "champion_margin_penalty",
    "ranking_gap_penalty",
    "archetype_concentration_penalty",
    "scenario_selection_score",
    "recommended_role",
    "winner_distribution",
    "deterministic_driver_winner_count",
    "driver_champion",
    "driver_champion_probability",
    "constructor_champion",
    "constructor_champion_probability",
    "avg_winner_runner_up_score_gap",
    "max_archetype_winner_share",
]


def read_csv(path):
    """Read a UTF-8 CSV file into dictionaries."""
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path, fieldnames, rows):
    """Write dictionaries to CSV with a stable schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    """Write JSON metadata for the scenario selection step."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def to_float(value, default=0.0):
    """Convert a CSV value to float with a safe fallback."""
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value, default=0):
    """Convert a CSV value to int with a safe fallback."""
    try:
        if value == "":
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def format_float(value, digits=6):
    """Format numeric outputs consistently."""
    return f"{value:.{digits}f}"


def parse_distribution(raw_value):
    """Parse JSON winner distribution text from scenario diagnostics."""
    try:
        return json.loads(raw_value)
    except (TypeError, json.JSONDecodeError):
        return {}


def normalize(value, minimum, maximum, reverse=False):
    """Min-max normalize a value, optionally reversing the direction."""
    if maximum <= minimum:
        return 1.0
    score = (value - minimum) / (maximum - minimum)
    score = max(0.0, min(1.0, score))
    return 1.0 - score if reverse else score


def scenario_metric_rows():
    """Merge scenario-level driver and constructor rows into one record per scenario."""
    summary_rows = read_csv(SCENARIO_SUMMARY_PATH)
    by_scenario = defaultdict(dict)
    for row in summary_rows:
        scenario = row["scenario_rank"]
        entity_type = row["entity_type"]
        if entity_type == "driver":
            by_scenario[scenario].update(
                {
                    "scenario_rank": scenario,
                    "top10_model": row["top10_model"],
                    "points_model": row["points_model"],
                    "top10_f1": to_float(row["top10_metric_value"]),
                    "points_mae": to_float(row["points_metric_value"]),
                    "driver_champion": row["predicted_champion"],
                    "driver_champion_probability": to_float(row["predicted_champion_probability"]),
                    "driver_margin": to_float(row["champion_probability_margin"]),
                }
            )
        elif entity_type == "constructor":
            by_scenario[scenario].update(
                {
                    "constructor_champion": row["predicted_champion"],
                    "constructor_champion_probability": to_float(row["predicted_champion_probability"]),
                    "constructor_margin": to_float(row["champion_probability_margin"]),
                }
            )
    return by_scenario


def race_signal_stats():
    """Aggregate ranking-score gap diagnostics by scenario."""
    stats = defaultdict(list)
    for row in read_csv(RACE_SIGNAL_DIAGNOSTICS_PATH):
        stats[row["scenario_rank"]].append(to_float(row["winner_runner_up_score_gap"]))
    return {
        scenario: {
            "avg_gap": sum(values) / len(values) if values else 0.0,
            "max_gap": max(values) if values else 0.0,
        }
        for scenario, values in stats.items()
    }


def archetype_stats():
    """Calculate the strongest within-archetype winner share for each scenario."""
    stats = defaultdict(list)
    for row in read_csv(CIRCUIT_ARCHETYPE_DIAGNOSTICS_PATH):
        distribution = parse_distribution(row["winner_distribution"])
        race_count = to_int(row["race_count"])
        max_share = max(distribution.values()) / race_count if race_count and distribution else 0.0
        stats[row["scenario_rank"]].append(max_share)
    return {
        scenario: max(values) if values else 0.0
        for scenario, values in stats.items()
    }


def build_selection_rows():
    """Score scenarios using performance and robustness diagnostics."""
    scenarios = scenario_metric_rows()
    diagnostics = {row["scenario_rank"]: row for row in read_csv(SCENARIO_DIAGNOSTICS_PATH)}
    gap_stats = race_signal_stats()
    archetype_max_share = archetype_stats()

    top10_values = [row["top10_f1"] for row in scenarios.values()]
    mae_values = [row["points_mae"] for row in scenarios.values()]
    min_top10, max_top10 = min(top10_values), max(top10_values)
    min_mae, max_mae = min(mae_values), max(mae_values)

    rows = []
    for scenario, metrics in scenarios.items():
        diagnostic = diagnostics[scenario]
        winner_distribution = parse_distribution(diagnostic["deterministic_driver_winner_distribution"])
        remaining_races = to_int(diagnostic["remaining_race_count"])
        max_winner_count = max(winner_distribution.values()) if winner_distribution else 0
        winner_share = max_winner_count / remaining_races if remaining_races else 0.0
        winner_count = to_int(diagnostic["deterministic_driver_winner_count"])
        max_archetype_share = archetype_max_share.get(scenario, 0.0)

        top10_score = normalize(metrics["top10_f1"], min_top10, max_top10)
        points_score = normalize(metrics["points_mae"], min_mae, max_mae, reverse=True)
        performance_score = 0.5 * top10_score + 0.5 * points_score

        winner_diversity_score = min(1.0, winner_count / 3.0)
        winner_concentration_penalty = winner_share
        champion_margin_penalty = 0.5 * (
            metrics.get("driver_margin", 0.0) + metrics.get("constructor_margin", 0.0)
        )
        ranking_gap_penalty = min(1.0, gap_stats.get(scenario, {}).get("avg_gap", 0.0) / 0.15)
        archetype_concentration_penalty = max_archetype_share

        selection_score = (
            0.45 * performance_score
            + 0.25 * winner_diversity_score
            - 0.15 * winner_concentration_penalty
            - 0.08 * champion_margin_penalty
            - 0.04 * ranking_gap_penalty
            - 0.03 * archetype_concentration_penalty
        )

        rows.append(
            {
                "scenario_rank": scenario,
                "top10_model": metrics["top10_model"],
                "points_model": metrics["points_model"],
                "top10_f1": format_float(metrics["top10_f1"]),
                "points_mae": format_float(metrics["points_mae"]),
                "performance_score": format_float(performance_score),
                "winner_diversity_score": format_float(winner_diversity_score),
                "winner_concentration_penalty": format_float(winner_concentration_penalty),
                "champion_margin_penalty": format_float(champion_margin_penalty),
                "ranking_gap_penalty": format_float(ranking_gap_penalty),
                "archetype_concentration_penalty": format_float(archetype_concentration_penalty),
                "scenario_selection_score": format_float(selection_score),
                "recommended_role": "",
                "winner_distribution": json.dumps(winner_distribution, ensure_ascii=False),
                "deterministic_driver_winner_count": winner_count,
                "driver_champion": metrics["driver_champion"],
                "driver_champion_probability": format_float(metrics["driver_champion_probability"]),
                "constructor_champion": metrics["constructor_champion"],
                "constructor_champion_probability": format_float(metrics["constructor_champion_probability"]),
                "avg_winner_runner_up_score_gap": format_float(gap_stats.get(scenario, {}).get("avg_gap", 0.0)),
                "max_archetype_winner_share": format_float(max_archetype_share),
            }
        )

    rows.sort(key=lambda row: to_float(row["scenario_selection_score"]), reverse=True)
    for index, row in enumerate(rows):
        if index == 0:
            row["recommended_role"] = "primary_recommended"
        elif to_float(row["winner_concentration_penalty"]) >= 0.95:
            row["recommended_role"] = "sensitivity_only_extreme_concentration"
        else:
            row["recommended_role"] = "secondary_sensitivity"
    return rows


def main():
    """Build and save scenario selection scores."""
    rows = build_selection_rows()
    write_csv(OUTPUT_PATH, OUTPUT_FIELDS, rows)
    write_json(
        SUMMARY_JSON_PATH,
        {
            "built_at": datetime.now(timezone.utc).isoformat(),
            "inputs": [
                str(SCENARIO_SUMMARY_PATH.relative_to(BASE_DIR)),
                str(SCENARIO_DIAGNOSTICS_PATH.relative_to(BASE_DIR)),
                str(RACE_SIGNAL_DIAGNOSTICS_PATH.relative_to(BASE_DIR)),
                str(CIRCUIT_ARCHETYPE_DIAGNOSTICS_PATH.relative_to(BASE_DIR)),
            ],
            "outputs": [
                str(OUTPUT_PATH.relative_to(BASE_DIR)),
            ],
            "selection_method": {
                "performance_score": "0.5 * normalized Top10 F1 + 0.5 * inverse normalized points MAE",
                "final_score": "0.45 performance + 0.25 winner diversity - concentration, margin, ranking-gap, and archetype-concentration penalties",
            },
            "recommended_primary_scenario": rows[0]["scenario_rank"] if rows else "",
            "notes": [
                "Scenario selection does not change model predictions; it labels which scenario should be treated as the primary forecast.",
                "Scenarios with deterministic winner concentration near 1.0 are kept as sensitivity-only views.",
            ],
        },
    )
    print(f"Scenario selection saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
