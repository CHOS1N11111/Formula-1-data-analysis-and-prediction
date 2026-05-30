import csv
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from build_f1_features import FEATURE_FIELDS, build_features


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "formula-1-race-data-sqlite" / "Formula1.sqlite"
MODERN_DATASET_PATH = BASE_DIR / "data" / "processed" / "f1_model_dataset.csv"
EXTENDED_DATASET_PATH = BASE_DIR / "data" / "processed" / "f1_model_dataset_extended.csv"
EXTENDED_FEATURES_PATH = BASE_DIR / "data" / "processed" / "f1_features_extended.csv"
SUMMARY_PATH = BASE_DIR / "data" / "processed" / "extended_feature_summary.json"

HISTORICAL_START_YEAR = 2003
HISTORICAL_END_YEAR = 2017


BASE_FIELDS = [
    "season",
    "round",
    "race_name",
    "race_date",
    "race_time",
    "circuit_id",
    "circuit_name",
    "circuit_locality",
    "circuit_country",
    "driver_id",
    "driver_code",
    "driver_name",
    "driver_nationality",
    "constructor_id",
    "constructor_name",
    "constructor_nationality",
    "grid",
    "qualifying_position",
    "finish_position",
    "points",
    "laps",
    "status",
    "is_podium",
    "is_top10",
    "data_source",
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


def clean_text(value):
    if value is None:
        return ""
    value = str(value).replace("\x00", "").strip()
    if value in {"\\N", "N/A", "None"}:
        return ""
    return value


def fetch_historical_rows():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"SQLite database not found: {DB_PATH}")

    query = """
        SELECT
            r.year AS season,
            r.round AS round,
            r.name AS race_name,
            r.date AS race_date,
            r.time AS race_time,
            ci.circuitRef AS circuit_id,
            ci.name AS circuit_name,
            ci.location AS circuit_locality,
            ci.country AS circuit_country,
            d.driverRef AS driver_id,
            d.code AS driver_code,
            TRIM(d.forename || ' ' || d.surname) AS driver_name,
            d.nationality AS driver_nationality,
            c.constructorRef AS constructor_id,
            c.name AS constructor_name,
            c.nationality AS constructor_nationality,
            rs.grid AS grid,
            q.position AS qualifying_position,
            rs.positionOrder AS finish_position,
            rs.points AS points,
            rs.laps AS laps,
            s.status AS status
        FROM results rs
        JOIN races r ON rs.raceId = r.raceId
        JOIN drivers d ON rs.driverId = d.driverId
        JOIN constructors c ON rs.constructorId = c.constructorId
        JOIN circuits ci ON r.circuitId = ci.circuitId
        JOIN qualifying q ON q.raceId = rs.raceId AND q.driverId = rs.driverId
        LEFT JOIN status s ON rs.statusId = s.statusId
        WHERE r.year BETWEEN ? AND ?
        ORDER BY r.year, r.round, rs.positionOrder
    """

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = []
    for row in conn.execute(query, (HISTORICAL_START_YEAR, HISTORICAL_END_YEAR)):
        output_row = {field: row[field] for field in row.keys()}
        finish_position = int(output_row["finish_position"])
        output_row["season"] = str(output_row["season"])
        output_row["round"] = str(output_row["round"])
        output_row["race_time"] = clean_text(output_row["race_time"])
        output_row["driver_code"] = clean_text(output_row["driver_code"])
        output_row["points"] = str(float(output_row["points"] or 0.0))
        output_row["laps"] = str(output_row["laps"] or "")
        output_row["is_podium"] = "1" if finish_position <= 3 else "0"
        output_row["is_top10"] = "1" if finish_position <= 10 else "0"
        output_row["data_source"] = "Formula1.sqlite"
        rows.append(output_row)
    conn.close()
    return rows


def build_extended_dataset():
    historical_rows = fetch_historical_rows()
    modern_rows = read_csv(MODERN_DATASET_PATH)
    selected_modern_rows = [
        row for row in modern_rows if int(row["season"]) >= 2019
    ]
    all_rows = historical_rows + selected_modern_rows
    all_rows = sorted(
        all_rows,
        key=lambda row: (
            int(row["season"]),
            int(row["round"]),
            int(float(row["finish_position"])),
        ),
    )
    return historical_rows, selected_modern_rows, all_rows


def count_by_year(rows):
    counts = {}
    for row in rows:
        counts[row["season"]] = counts.get(row["season"], 0) + 1
    return dict(sorted(counts.items(), key=lambda item: int(item[0])))


def main():
    historical_rows, modern_rows, all_rows = build_extended_dataset()
    enriched_rows, feature_summary = build_features([dict(row) for row in all_rows])

    write_csv(EXTENDED_DATASET_PATH, BASE_FIELDS, all_rows)
    write_csv(EXTENDED_FEATURES_PATH, BASE_FIELDS + FEATURE_FIELDS, enriched_rows)

    summary = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "historical_source": str(DB_PATH.relative_to(BASE_DIR)),
        "modern_source": str(MODERN_DATASET_PATH.relative_to(BASE_DIR)),
        "historical_year_range": [HISTORICAL_START_YEAR, HISTORICAL_END_YEAR],
        "modern_year_min": 2019,
        "output_dataset": str(EXTENDED_DATASET_PATH.relative_to(BASE_DIR)),
        "output_features": str(EXTENDED_FEATURES_PATH.relative_to(BASE_DIR)),
        "historical_rows": len(historical_rows),
        "modern_rows": len(modern_rows),
        "total_rows": len(enriched_rows),
        "rows_by_year": count_by_year(enriched_rows),
        "feature_summary": feature_summary,
        "note": "Historical data starts at 2003 because SQLite qualifying coverage becomes consistently usable from 2003 onward.",
    }
    write_json(SUMMARY_PATH, summary)

    print(f"Extended dataset saved to: {EXTENDED_DATASET_PATH}")
    print(f"Extended features saved to: {EXTENDED_FEATURES_PATH}")
    print(f"Rows: {len(enriched_rows)}")


if __name__ == "__main__":
    main()
