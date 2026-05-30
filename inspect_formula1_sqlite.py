"""Inspect the local Formula1.sqlite database tables, columns, and sample rows."""

import sqlite3
import sys
from pathlib import Path


DB_PATH = Path(__file__).resolve().parent / "formula-1-race-data-sqlite" / "Formula1.sqlite"


def print_rows(cursor, table_name, limit=10):
    """Print a small sample of rows from one SQLite table."""
    cursor.execute(f'SELECT * FROM "{table_name}" LIMIT {limit}')
    rows = cursor.fetchall()
    columns = [description[0] for description in cursor.description]

    print(f"\n{table_name} first {limit} rows")
    print("-" * 80)
    print(" | ".join(columns))
    for row in rows:
        print(" | ".join(str(value) for value in row))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if not DB_PATH.exists():
        raise FileNotFoundError(f"Database file not found: {DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [row[0] for row in cursor.fetchall()]

    print("\nTables")
    print("-" * 80)

    for table in tables:
        cursor.execute(f'SELECT COUNT(*) FROM "{table}"')
        row_count = cursor.fetchone()[0]

        cursor.execute(f'PRAGMA table_info("{table}")')
        columns = [row[1] for row in cursor.fetchall()]

        print(f"{table}: {row_count} rows")
        print("  columns:", ", ".join(columns))

    sample_tables = [
        "drivers",
        "constructors",
        "circuits",
        "races",
        "results",
        "qualifying",
        "laptimes",
        "pitstops",
        "driver_standings",
        "constructor_standings",
    ]

    for table in sample_tables:
        print_rows(cursor, table)

    conn.close()


if __name__ == "__main__":
    main()
