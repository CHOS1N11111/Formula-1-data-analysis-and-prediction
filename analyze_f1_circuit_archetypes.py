"""Classify Formula 1 circuits into interpretable archetypes.

This script uses existing 2019-2025 circuit analysis outputs to assign each
track a rule-based archetype such as qualifying-heavy, overtaking-friendly, or
high-volatility. It then joins the archetypes to the 2026 championship race
signal diagnostics so model behavior can be checked by circuit type.
"""

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
ANALYSIS_DIR = BASE_DIR / "data" / "analysis"
MODELING_DIR = BASE_DIR / "data" / "modeling"

GRID_IMPORTANCE_PATH = ANALYSIS_DIR / "circuit_grid_importance_score.csv"
VOLATILITY_PATH = ANALYSIS_DIR / "circuit_volatility_index.csv"
SUMMARY_PATH = ANALYSIS_DIR / "circuit_summary.csv"
SIGNAL_DIAGNOSTICS_PATH = MODELING_DIR / "season_prediction_race_signal_diagnostics_2026.csv"

ARCHETYPE_OUTPUT_PATH = ANALYSIS_DIR / "circuit_archetypes_2019_2025.csv"
PREDICTION_DIAGNOSTICS_OUTPUT_PATH = MODELING_DIR / "season_prediction_circuit_archetype_diagnostics_2026.csv"
SUMMARY_JSON_PATH = ANALYSIS_DIR / "circuit_archetype_summary.json"

ARCHETYPE_FIELDS = [
    "circuit_id",
    "circuit_name",
    "country",
    "race_count",
    "pole_win_rate",
    "front3_podium_rate",
    "grid_importance_score",
    "avg_abs_position_change",
    "large_gain_rate",
    "non_front_row_winner_rate",
    "volatility_index",
    "circuit_archetype",
    "archetype_reason",
]

PREDICTION_DIAGNOSTIC_FIELDS = [
    "scenario_rank",
    "top10_model",
    "points_model",
    "circuit_archetype",
    "race_count",
    "winner_distribution",
    "avg_winner_runner_up_score_gap",
    "min_winner_runner_up_score_gap",
    "max_winner_runner_up_score_gap",
    "avg_winner_runner_up_predicted_points_gap",
    "avg_winner_runner_up_calibrated_top10_gap",
]


def read_csv(path):
    """Read a UTF-8 CSV file into dictionaries."""
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path, fieldnames, rows):
    """Write dictionaries to CSV with stable field order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    """Write JSON metadata for report reproducibility."""
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


def format_float(value, digits=6):
    """Format numeric values consistently for CSV outputs."""
    return f"{value:.{digits}f}"


def merge_circuit_metrics():
    """Merge existing circuit analysis tables by circuit_id."""
    circuits = {}
    for row in read_csv(SUMMARY_PATH):
        circuits[row["circuit_id"]] = {
            "circuit_id": row["circuit_id"],
            "circuit_name": row["circuit_name"],
            "country": row["country"],
            "race_count": row["race_count"],
            "pole_win_rate": row["pole_win_rate"],
            "front3_podium_rate": row["front3_podium_rate"],
            "grid_importance_score": "",
            "avg_abs_position_change": "",
            "large_gain_rate": "",
            "non_front_row_winner_rate": "",
            "volatility_index": "",
        }

    for row in read_csv(GRID_IMPORTANCE_PATH):
        circuit = circuits.setdefault(row["circuit_id"], {})
        circuit.update(
            {
                "circuit_id": row["circuit_id"],
                "circuit_name": row["circuit_name"],
                "country": row["country"],
                "race_count": row["race_count"],
                "pole_win_rate": row["pole_win_rate"],
                "front3_podium_rate": row["front3_podium_rate"],
                "grid_importance_score": row["grid_importance_score"],
                "avg_abs_position_change": row["avg_abs_position_change"],
            }
        )

    for row in read_csv(VOLATILITY_PATH):
        circuit = circuits.setdefault(row["circuit_id"], {})
        circuit.update(
            {
                "circuit_id": row["circuit_id"],
                "circuit_name": row["circuit_name"],
                "country": row["country"],
                "race_count": row["race_count"],
                "avg_abs_position_change": row["avg_abs_position_change"],
                "large_gain_rate": row["large_gain_rate"],
                "non_front_row_winner_rate": row["non_front_row_winner_rate"],
                "volatility_index": row["volatility_index"],
            }
        )

    return circuits


def classify_circuit(row):
    """Assign a rule-based circuit archetype from historical metrics."""
    grid_importance = to_float(row.get("grid_importance_score"))
    pole_win_rate = to_float(row.get("pole_win_rate"))
    front3_podium_rate = to_float(row.get("front3_podium_rate"))
    large_gain_rate = to_float(row.get("large_gain_rate"))
    non_front_row_winner_rate = to_float(row.get("non_front_row_winner_rate"))
    volatility_index = to_float(row.get("volatility_index"))
    avg_abs_change = to_float(row.get("avg_abs_position_change"))

    if volatility_index >= 0.20 or non_front_row_winner_rate >= 0.40:
        return (
            "high_volatility",
            "High volatility or frequent non-front-row winners.",
        )
    if large_gain_rate >= 0.13 or avg_abs_change >= 3.70:
        return (
            "overtaking_friendly",
            "Large position changes or frequent large gains.",
        )
    if grid_importance >= 0.70 or (pole_win_rate >= 0.70 and front3_podium_rate >= 0.70):
        return (
            "qualifying_heavy",
            "Grid position, pole wins, and front-three starts are historically important.",
        )
    return (
        "balanced",
        "No single qualifying, overtaking, or volatility metric dominates.",
    )


def build_archetype_rows(circuits):
    """Create sorted circuit archetype rows."""
    rows = []
    for circuit_id, circuit in circuits.items():
        archetype, reason = classify_circuit(circuit)
        row = dict(circuit)
        row["circuit_archetype"] = archetype
        row["archetype_reason"] = reason
        rows.append(row)
    return sorted(rows, key=lambda row: (row["circuit_archetype"], row["circuit_id"]))


def infer_circuit_from_race_name(race_name, archetype_by_circuit):
    """Map future race names to circuit ids when diagnostics lack circuit_id."""
    race_name = race_name.lower()
    aliases = {
        "monaco": "monaco",
        "barcelona": "catalunya",
        "austrian": "red_bull_ring",
        "british": "silverstone",
        "belgian": "spa",
        "hungarian": "hungaroring",
        "dutch": "zandvoort",
        "italian": "monza",
        "spanish": "jarama",
        "azerbaijan": "baku",
        "singapore": "marina_bay",
        "united states": "americas",
        "mexico city": "rodriguez",
        "sao paulo": "interlagos",
        "las vegas": "vegas",
        "qatar": "losail",
        "abu dhabi": "yas_marina",
    }
    for token, circuit_id in aliases.items():
        if token in race_name and circuit_id in archetype_by_circuit:
            return circuit_id
    return ""


def build_prediction_diagnostics(signal_rows, archetype_rows):
    """Aggregate 2026 prediction behavior by scenario and circuit archetype."""
    archetype_by_circuit = {row["circuit_id"]: row for row in archetype_rows}
    grouped = defaultdict(list)
    for row in signal_rows:
        circuit_id = infer_circuit_from_race_name(row["race_name"], archetype_by_circuit)
        archetype = archetype_by_circuit.get(circuit_id, {}).get("circuit_archetype", "unknown")
        key = (row["scenario_rank"], row["top10_model"], row["points_model"], archetype)
        enriched = dict(row)
        enriched["circuit_archetype"] = archetype
        grouped[key].append(enriched)

    output_rows = []
    for (scenario_rank, top10_model, points_model, archetype), rows in sorted(grouped.items()):
        winner_counts = Counter(row["winner_driver"] for row in rows)
        score_gaps = [to_float(row["winner_runner_up_score_gap"]) for row in rows]
        points_gaps = [to_float(row["winner_runner_up_predicted_points_gap"]) for row in rows]
        top10_gaps = [to_float(row["winner_runner_up_calibrated_top10_gap"]) for row in rows]
        output_rows.append(
            {
                "scenario_rank": scenario_rank,
                "top10_model": top10_model,
                "points_model": points_model,
                "circuit_archetype": archetype,
                "race_count": len(rows),
                "winner_distribution": json.dumps(dict(winner_counts), ensure_ascii=False),
                "avg_winner_runner_up_score_gap": format_float(sum(score_gaps) / len(score_gaps)),
                "min_winner_runner_up_score_gap": format_float(min(score_gaps)),
                "max_winner_runner_up_score_gap": format_float(max(score_gaps)),
                "avg_winner_runner_up_predicted_points_gap": format_float(sum(points_gaps) / len(points_gaps)),
                "avg_winner_runner_up_calibrated_top10_gap": format_float(sum(top10_gaps) / len(top10_gaps)),
            }
        )
    return output_rows


def main():
    """Build circuit archetype tables and 2026 prediction diagnostics."""
    circuits = merge_circuit_metrics()
    archetype_rows = build_archetype_rows(circuits)
    signal_rows = read_csv(SIGNAL_DIAGNOSTICS_PATH)
    prediction_rows = build_prediction_diagnostics(signal_rows, archetype_rows)

    write_csv(ARCHETYPE_OUTPUT_PATH, ARCHETYPE_FIELDS, archetype_rows)
    write_csv(PREDICTION_DIAGNOSTICS_OUTPUT_PATH, PREDICTION_DIAGNOSTIC_FIELDS, prediction_rows)
    write_json(
        SUMMARY_JSON_PATH,
        {
            "built_at": datetime.now(timezone.utc).isoformat(),
            "inputs": [
                str(GRID_IMPORTANCE_PATH.relative_to(BASE_DIR)),
                str(VOLATILITY_PATH.relative_to(BASE_DIR)),
                str(SUMMARY_PATH.relative_to(BASE_DIR)),
                str(SIGNAL_DIAGNOSTICS_PATH.relative_to(BASE_DIR)),
            ],
            "outputs": [
                str(ARCHETYPE_OUTPUT_PATH.relative_to(BASE_DIR)),
                str(PREDICTION_DIAGNOSTICS_OUTPUT_PATH.relative_to(BASE_DIR)),
            ],
            "archetype_counts": dict(Counter(row["circuit_archetype"] for row in archetype_rows)),
            "notes": [
                "Circuit archetypes are rule-based and intended for diagnostics, not direct manual prediction correction.",
                "2026 prediction diagnostics show whether model winner concentration changes by circuit type.",
            ],
        },
    )
    print(f"Circuit archetypes saved to: {ARCHETYPE_OUTPUT_PATH}")
    print(f"Prediction diagnostics saved to: {PREDICTION_DIAGNOSTICS_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
