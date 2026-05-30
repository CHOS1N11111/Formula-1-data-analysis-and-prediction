"""Generate historical background analysis CSV files from Formula1.sqlite."""

import csv
import json
import math
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "formula-1-race-data-sqlite" / "Formula1.sqlite"
ANALYSIS_DIR = BASE_DIR / "data" / "analysis"
SUMMARY_PATH = ANALYSIS_DIR / "analysis_historical_summary.json"


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def format_float(value, digits=4):
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return f"{value:.{digits}f}"


def pearson_correlation(pairs):
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    if len(pairs) < 2:
        return None

    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    denominator_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    denominator_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))

    if denominator_x == 0 or denominator_y == 0:
        return None
    return numerator / (denominator_x * denominator_y)


def fetch_all(cursor, query):
    cursor.execute(query)
    columns = [description[0] for description in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def build_overview(cursor):
    """Summarize historical database coverage and entity counts."""
    rows = []

    cursor.execute("SELECT MIN(year), MAX(year), COUNT(*) FROM races")
    min_year, max_year, race_count = cursor.fetchone()
    rows.extend(
        [
            {"metric": "race_year_min", "value": min_year},
            {"metric": "race_year_max", "value": max_year},
            {"metric": "race_count", "value": race_count},
        ]
    )

    cursor.execute(
        """
        SELECT MIN(r.year), MAX(r.year), COUNT(*)
        FROM results rs
        JOIN races r ON rs.raceId = r.raceId
        """
    )
    result_min_year, result_max_year, result_count = cursor.fetchone()
    rows.extend(
        [
            {"metric": "result_year_min", "value": result_min_year},
            {"metric": "result_year_max", "value": result_max_year},
            {"metric": "result_count", "value": result_count},
        ]
    )

    cursor.execute(
        """
        SELECT MIN(r.year), MAX(r.year), COUNT(*)
        FROM qualifying q
        JOIN races r ON q.raceId = r.raceId
        """
    )
    qualifying_min_year, qualifying_max_year, qualifying_count = cursor.fetchone()
    rows.extend(
        [
            {"metric": "qualifying_year_min", "value": qualifying_min_year},
            {"metric": "qualifying_year_max", "value": qualifying_max_year},
            {"metric": "qualifying_count", "value": qualifying_count},
        ]
    )

    cursor.execute("SELECT COUNT(*) FROM drivers")
    rows.append({"metric": "driver_count", "value": cursor.fetchone()[0]})

    cursor.execute("SELECT COUNT(*) FROM constructors")
    rows.append({"metric": "constructor_count", "value": cursor.fetchone()[0]})

    cursor.execute("SELECT COUNT(*) FROM circuits")
    rows.append({"metric": "circuit_count", "value": cursor.fetchone()[0]})

    return rows


def build_races_by_year(cursor):
    return fetch_all(
        cursor,
        """
        SELECT
            year,
            COUNT(*) AS race_count
        FROM races
        GROUP BY year
        ORDER BY year
        """,
    )


def build_driver_wins(cursor):
    return fetch_all(
        cursor,
        """
        SELECT
            d.driverRef AS driver_id,
            TRIM(d.forename || ' ' || d.surname) AS driver_name,
            d.nationality,
            COUNT(*) AS wins,
            MIN(r.year) AS first_win_year,
            MAX(r.year) AS last_win_year
        FROM results rs
        JOIN races r ON rs.raceId = r.raceId
        JOIN drivers d ON rs.driverId = d.driverId
        WHERE rs.positionOrder = 1
        GROUP BY d.driverId
        ORDER BY wins DESC, driver_name
        LIMIT 30
        """,
    )


def build_constructor_wins(cursor):
    return fetch_all(
        cursor,
        """
        SELECT
            c.constructorRef AS constructor_id,
            c.name AS constructor_name,
            c.nationality,
            COUNT(*) AS wins,
            MIN(r.year) AS first_win_year,
            MAX(r.year) AS last_win_year
        FROM results rs
        JOIN races r ON rs.raceId = r.raceId
        JOIN constructors c ON rs.constructorId = c.constructorId
        WHERE rs.positionOrder = 1
        GROUP BY c.constructorId
        ORDER BY wins DESC, constructor_name
        LIMIT 30
        """,
    )


def build_grid_historical_summary(cursor):
    rows = fetch_all(
        cursor,
        """
        SELECT
            rs.grid,
            rs.positionOrder
        FROM results rs
        JOIN races r ON rs.raceId = r.raceId
        WHERE rs.grid > 0
          AND rs.positionOrder IS NOT NULL
        """,
    )
    pairs = [(row["grid"], row["positionOrder"]) for row in rows]

    pole_rows = [row for row in rows if row["grid"] == 1]
    front3_rows = [row for row in rows if 1 <= row["grid"] <= 3]
    front10_rows = [row for row in rows if 1 <= row["grid"] <= 10]

    return [
        {
            "metric": "grid_finish_correlation_1950_2017",
            "value": format_float(pearson_correlation(pairs)),
        },
        {
            "metric": "pole_win_rate_1950_2017",
            "value": format_float(
                sum(1 for row in pole_rows if row["positionOrder"] == 1) / len(pole_rows)
            ),
        },
        {
            "metric": "front3_podium_rate_1950_2017",
            "value": format_float(
                sum(1 for row in front3_rows if row["positionOrder"] <= 3)
                / len(front3_rows)
            ),
        },
        {
            "metric": "front10_top10_rate_1950_2017",
            "value": format_float(
                sum(1 for row in front10_rows if row["positionOrder"] <= 10)
                / len(front10_rows)
            ),
        },
    ]


def build_grid_by_position(cursor):
    rows = fetch_all(
        cursor,
        """
        SELECT
            grid,
            COUNT(*) AS records,
            AVG(positionOrder) AS avg_finish_position,
            AVG(CASE WHEN positionOrder = 1 THEN 1.0 ELSE 0.0 END) AS win_rate,
            AVG(CASE WHEN positionOrder <= 3 THEN 1.0 ELSE 0.0 END) AS podium_rate,
            AVG(CASE WHEN positionOrder <= 10 THEN 1.0 ELSE 0.0 END) AS top10_rate
        FROM results
        WHERE grid > 0
          AND positionOrder IS NOT NULL
        GROUP BY grid
        ORDER BY grid
        """
    )

    for row in rows:
        row["avg_finish_position"] = format_float(row["avg_finish_position"])
        row["win_rate"] = format_float(row["win_rate"])
        row["podium_rate"] = format_float(row["podium_rate"])
        row["top10_rate"] = format_float(row["top10_rate"])

    return rows


def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database file not found: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    outputs = [
        (
            "dataset_overview_historical.csv",
            ["metric", "value"],
            build_overview(cursor),
        ),
        (
            "races_by_year_historical.csv",
            ["year", "race_count"],
            build_races_by_year(cursor),
        ),
        (
            "driver_wins_historical.csv",
            ["driver_id", "driver_name", "nationality", "wins", "first_win_year", "last_win_year"],
            build_driver_wins(cursor),
        ),
        (
            "constructor_wins_historical.csv",
            ["constructor_id", "constructor_name", "nationality", "wins", "first_win_year", "last_win_year"],
            build_constructor_wins(cursor),
        ),
        (
            "grid_finish_summary_historical.csv",
            ["metric", "value"],
            build_grid_historical_summary(cursor),
        ),
        (
            "grid_position_summary_historical.csv",
            ["grid", "records", "avg_finish_position", "win_rate", "podium_rate", "top10_rate"],
            build_grid_by_position(cursor),
        ),
    ]

    for filename, fieldnames, rows in outputs:
        write_csv(ANALYSIS_DIR / filename, fieldnames, rows)

    summary = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source_database": str(DB_PATH.relative_to(BASE_DIR)),
        "output_dir": str(ANALYSIS_DIR.relative_to(BASE_DIR)),
        "output_files": [filename for filename, _, _ in outputs],
    }
    write_json(SUMMARY_PATH, summary)

    conn.close()

    print(f"Historical analysis files saved to: {ANALYSIS_DIR}")
    print(f"Summary saved to: {SUMMARY_PATH}")
    print(f"Output files: {len(outputs)}")


if __name__ == "__main__":
    main()
