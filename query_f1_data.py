"""Run interactive SQL-style queries against all available Formula 1 data.

This utility turns project CSV outputs, JSON metadata/raw files, historical
SQLite tables, and selected joined historical views into an in-memory SQLite
database. It is intended for exploratory data analysis: users can list available
tables, inspect schemas, run read-only SQL SELECT queries, or use a small
key-value search helper for common driver/team queries.
"""

import csv
import json
import re
import shlex
import sqlite3
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SQLITE_DIR = BASE_DIR / "formula-1-race-data-sqlite"
HISTORICAL_SQLITE_PATH = SQLITE_DIR / "Formula1.sqlite"
QUERY_FILE_ROOTS = [
    (DATA_DIR, []),
    (BASE_DIR / "formula-1-fantasy-2021", ["formula-1-fantasy-2021"]),
    (BASE_DIR / "formula1-data-1950-2022", ["formula1-data-1950-2022"]),
]

FRIENDLY_TABLE_ALIASES = {
    "features": DATA_DIR / "processed" / "f1_features.csv",
    "features_extended": DATA_DIR / "processed" / "f1_features_extended.csv",
    "model_dataset": DATA_DIR / "processed" / "f1_model_dataset.csv",
    "schedule_2026": DATA_DIR / "processed" / "f1_2026_schedule.csv",
    "driver_summary": DATA_DIR / "analysis" / "driver_summary.csv",
    "constructor_summary": DATA_DIR / "analysis" / "constructor_summary.csv",
    "circuit_summary": DATA_DIR / "analysis" / "circuit_summary.csv",
    "completed_2026_results": DATA_DIR / "analysis" / "completed_2026_results.csv",
    "remaining_2026_schedule": DATA_DIR / "analysis" / "remaining_2026_schedule.csv",
    "season_prediction_driver_2026": DATA_DIR / "modeling" / "season_prediction_driver_standings_2026.csv",
    "season_prediction_constructor_2026": DATA_DIR / "modeling" / "season_prediction_constructor_standings_2026.csv",
    "season_prediction_races_2026": DATA_DIR / "modeling" / "season_prediction_race_points_2026.csv",
}

HISTORICAL_SQLITE_TABLES = {
    "historical_races": """
        SELECT
            r.year AS season,
            r.round AS round,
            r.name AS race_name,
            r.date AS race_date,
            c.circuitRef AS circuit_id,
            c.name AS circuit_name,
            c.location AS circuit_locality,
            c.country AS circuit_country
        FROM races r
        JOIN circuits c ON r.circuitId = c.circuitId
        ORDER BY r.year, r.round
    """,
    "historical_results": """
        SELECT
            r.year AS season,
            r.round AS round,
            r.name AS race_name,
            r.date AS race_date,
            c.circuitRef AS circuit_id,
            c.name AS circuit_name,
            c.location AS circuit_locality,
            c.country AS circuit_country,
            d.driverRef AS driver_id,
            TRIM(COALESCE(d.forename, '') || ' ' || COALESCE(d.surname, '')) AS driver_name,
            d.nationality AS driver_nationality,
            cons.constructorRef AS constructor_id,
            cons.name AS constructor_name,
            cons.nationality AS constructor_nationality,
            rs.grid AS grid,
            rs.positionOrder AS finish_position,
            rs.points AS points,
            rs.laps AS laps,
            s.status AS status
        FROM results rs
        JOIN races r ON rs.raceId = r.raceId
        JOIN drivers d ON rs.driverId = d.driverId
        JOIN constructors cons ON rs.constructorId = cons.constructorId
        JOIN circuits c ON r.circuitId = c.circuitId
        LEFT JOIN status s ON rs.statusId = s.statusId
        ORDER BY r.year, r.round, rs.positionOrder
    """,
}

HELP_TEXT = """
Commands:
  help
      Show this help message.
  tables [keyword]
      List queryable tables, optionally filtered by keyword.
  schema <table>
      Show table columns.
  find [table=<table>] [season=2025] [driver=hamilton] [constructor=mercedes] [race=monaco] [limit=20]
      Run a quick filtered search. Default table is features_extended.
  SELECT ... FROM ... LIMIT ...
      Run a read-only SQL query.
  exit
      Leave the query console.

Examples:
  tables
  tables modeling
  schema features_extended
  schema csv_modeling_points_model_metrics
  schema source_formula1_results
  schema json_raw_jolpica_2026_races
  find driver=antonelli season=2026
  find table=historical_results driver=senna season=1991 limit=5
  find table=season_prediction_races_2026 race=Monaco limit=10
  SELECT season, race_name, driver_name, constructor_name, finish_position
  FROM features_extended
  WHERE season = '2025' AND driver_name LIKE '%Hamilton%'
  LIMIT 10;
  SELECT season, race_name, driver_name, constructor_name, finish_position
  FROM historical_results
  WHERE season = '1991' AND driver_name LIKE '%Senna%'
  LIMIT 10;
"""


def normalize_identifier(value):
    """Convert a file path part or source column name into a safe identifier."""
    normalized = re.sub(r"[^A-Za-z0-9_]+", "_", str(value)).strip("_").lower()
    if not normalized:
        normalized = "column"
    if normalized[0].isdigit():
        normalized = f"column_{normalized}"
    return normalized


def normalize_fieldnames(fieldnames):
    """Normalize source field names and keep them unique for SQLite queries."""
    normalized = []
    seen = {}
    for index, field in enumerate(fieldnames):
        candidate = normalize_identifier(field or f"column_{index + 1}")
        count = seen.get(candidate, 0) + 1
        seen[candidate] = count
        if count > 1:
            candidate = f"{candidate}_{count}"
        normalized.append(candidate)
    return normalized


def table_name_from_path(path):
    """Build a deterministic query table name from a data file path."""
    root = None
    root_prefix_parts = []
    for candidate_root, candidate_prefix_parts in QUERY_FILE_ROOTS:
        try:
            relative = path.relative_to(candidate_root)
            root = candidate_root
            root_prefix_parts = candidate_prefix_parts
            break
        except ValueError:
            continue
    if root is None:
        raise ValueError(f"File is outside configured query roots: {path}")

    prefix = path.suffix.lower().lstrip(".")
    stem_parts = list(relative.with_suffix("").parts)
    if root == DATA_DIR:
        name_parts = [prefix, *stem_parts]
    else:
        name_parts = [prefix, *root_prefix_parts, *stem_parts]
    return normalize_identifier("_".join(name_parts))


def sqlite_prefix_from_path(path):
    """Build a deterministic table prefix for one source SQLite database."""
    if path.name.lower() == "formula1.sqlite":
        return "source_formula1"
    return normalize_identifier(f"source_{path.stem}")


def read_csv_rows(path):
    """Read one CSV file with common encoding fallbacks."""
    if not path.exists():
        raise FileNotFoundError(f"Missing CSV file: {path}")

    last_error = None
    for encoding in ("utf-8-sig", "utf-8", "gbk", "cp1252", "latin-1"):
        try:
            with path.open("r", encoding=encoding, newline="") as file:
                reader = csv.DictReader(file)
                original_fieldnames = reader.fieldnames or []
                fieldnames = normalize_fieldnames(original_fieldnames)
                rows = []
                for row in reader:
                    rows.append(
                        {
                            fieldnames[index]: "" if row.get(original, "") is None else row.get(original, "")
                            for index, original in enumerate(original_fieldnames)
                        }
                    )
                return fieldnames, rows
        except UnicodeDecodeError as exc:
            last_error = exc

    raise UnicodeDecodeError(
        last_error.encoding,
        last_error.object,
        last_error.start,
        last_error.end,
        f"Unable to decode {path} with supported encodings",
    )


def flatten_json(value, path="$"):
    """Yield one row for each scalar value found inside a nested JSON object."""
    if isinstance(value, dict):
        for key, child in value.items():
            yield from flatten_json(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from flatten_json(child, f"{path}[{index}]")
    else:
        key = path.rsplit(".", 1)[-1].split("[", 1)[0]
        yield {
            "json_path": path,
            "key": key,
            "value": "" if value is None else str(value),
            "value_type": type(value).__name__,
        }


def read_json_rows(path):
    """Read one JSON file as a queryable key-value path table."""
    if not path.exists():
        raise FileNotFoundError(f"Missing JSON file: {path}")

    with path.open("r", encoding="utf-8-sig") as file:
        data = json.load(file)

    fieldnames = ["json_path", "key", "value", "value_type"]
    return fieldnames, list(flatten_json(data))


def read_sqlite_rows(path, sql):
    """Read one source SQLite query and return field names plus text rows."""
    if not path.exists():
        return [], []

    source_connection = sqlite3.connect(path)
    source_connection.row_factory = sqlite3.Row
    try:
        cursor = source_connection.execute(sql)
        rows = cursor.fetchall()
        original_fieldnames = [description[0] for description in cursor.description]
        fieldnames = normalize_fieldnames(original_fieldnames)
    finally:
        source_connection.close()

    converted_rows = []
    for row in rows:
        converted_rows.append(
            {
                fieldnames[index]: "" if row[original] is None else str(row[original])
                for index, original in enumerate(original_fieldnames)
            }
        )
    return fieldnames, converted_rows


def quote_source_identifier(identifier):
    """Quote a source SQLite identifier without changing its original spelling."""
    return '"' + str(identifier).replace('"', '""') + '"'


def list_source_sqlite_tables(path):
    """Return user tables from one source SQLite database."""
    if not path.exists():
        return []

    source_connection = sqlite3.connect(path)
    try:
        cursor = source_connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        )
        return [row[0] for row in cursor.fetchall()]
    finally:
        source_connection.close()


def read_source_sqlite_table(path, source_table):
    """Read one raw source SQLite table into rows with normalized columns."""
    sql = f"SELECT * FROM {quote_source_identifier(source_table)}"
    return read_sqlite_rows(path, sql)


def create_table(connection, table_name, fieldnames, rows):
    """Create one in-memory SQLite table and insert rows as text values."""
    quoted_table = quote_identifier(table_name)
    columns_sql = ", ".join(f"{quote_identifier(field)} TEXT" for field in fieldnames)
    connection.execute(f"CREATE TABLE {quoted_table} ({columns_sql})")

    if not rows:
        return

    placeholders = ", ".join("?" for _ in fieldnames)
    insert_sql = f"INSERT INTO {quoted_table} VALUES ({placeholders})"
    values = [[row.get(field, "") for field in fieldnames] for row in rows]
    connection.executemany(insert_sql, values)


def quote_identifier(identifier):
    """Quote a SQLite identifier after validating its table or column syntax."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", identifier):
        raise ValueError(f"Unsafe identifier: {identifier}")
    return f'"{identifier}"'


def add_table(connection, table_metadata, table_name, fieldnames, rows, source_path, source_type):
    """Create a table and record its source metadata for the query console."""
    if table_name in table_metadata:
        raise ValueError(f"Duplicate query table name: {table_name}")
    create_table(connection, table_name, fieldnames, rows)
    table_metadata[table_name] = {
        "path": source_path,
        "columns": fieldnames,
        "row_count": len(rows),
        "type": source_type,
    }


def add_alias_view(connection, table_metadata, alias_name, target_name):
    """Add a short view alias for a longer auto-discovered table name."""
    if alias_name in table_metadata or target_name not in table_metadata:
        return

    connection.execute(f"CREATE VIEW {quote_identifier(alias_name)} AS SELECT * FROM {quote_identifier(target_name)}")
    target_metadata = table_metadata[target_name]
    table_metadata[alias_name] = {
        "path": target_metadata["path"],
        "columns": target_metadata["columns"],
        "row_count": target_metadata["row_count"],
        "type": f"alias:{target_name}",
    }


def discover_data_files():
    """Find every CSV and JSON file in project data source directories."""
    files = []
    for root, _prefix_parts in QUERY_FILE_ROOTS:
        if root.exists():
            files.extend(path for path in root.rglob("*") if path.suffix.lower() in {".csv", ".json"})
    return sorted(files)


def load_data_files(connection, table_metadata):
    """Load all CSV and JSON project data files into the query database."""
    path_to_table = {}

    for path in discover_data_files():
        table_name = table_name_from_path(path)
        if path.suffix.lower() == ".csv":
            fieldnames, rows = read_csv_rows(path)
            source_type = "csv"
        else:
            fieldnames, rows = read_json_rows(path)
            source_type = "json"

        if not fieldnames:
            continue
        add_table(connection, table_metadata, table_name, fieldnames, rows, path, source_type)
        path_to_table[path.resolve()] = table_name

    for alias_name, path in FRIENDLY_TABLE_ALIASES.items():
        target_name = path_to_table.get(path.resolve())
        if target_name:
            add_alias_view(connection, table_metadata, alias_name, target_name)


def load_source_sqlite_tables(connection, table_metadata):
    """Load every table from available local SQLite source databases."""
    if not SQLITE_DIR.exists():
        return

    for sqlite_path in sorted(SQLITE_DIR.glob("*.sqlite")):
        table_prefix = sqlite_prefix_from_path(sqlite_path)
        for source_table in list_source_sqlite_tables(sqlite_path):
            table_name = normalize_identifier(f"{table_prefix}_{source_table}")
            fieldnames, rows = read_source_sqlite_table(sqlite_path, source_table)
            if not fieldnames:
                continue
            add_table(connection, table_metadata, table_name, fieldnames, rows, sqlite_path, "sqlite")


def load_historical_join_tables(connection, table_metadata):
    """Load historical joined convenience tables from the main SQLite database."""
    if not HISTORICAL_SQLITE_PATH.exists():
        return

    for table_name, sql in HISTORICAL_SQLITE_TABLES.items():
        fieldnames, rows = read_sqlite_rows(HISTORICAL_SQLITE_PATH, sql)
        if not fieldnames:
            continue
        add_table(connection, table_metadata, table_name, fieldnames, rows, HISTORICAL_SQLITE_PATH, "sqlite_join")


def add_data_catalog_table(connection, table_metadata):
    """Expose the loaded table inventory as a queryable catalog table."""
    catalog_rows = []
    for table_name, metadata in sorted(table_metadata.items()):
        catalog_rows.append(
            {
                "table_name": table_name,
                "row_count": str(metadata["row_count"]),
                "source_type": metadata["type"],
                "source_path": str(metadata["path"].relative_to(BASE_DIR)),
                "columns": ", ".join(metadata["columns"]),
            }
        )

    add_table(
        connection,
        table_metadata,
        "data_catalog",
        ["table_name", "row_count", "source_type", "source_path", "columns"],
        catalog_rows,
        BASE_DIR,
        "catalog",
    )


def build_database():
    """Load configured CSV and historical SQLite data into memory."""
    connection = sqlite3.connect(":memory:")
    table_metadata = {}

    load_data_files(connection, table_metadata)
    load_source_sqlite_tables(connection, table_metadata)
    load_historical_join_tables(connection, table_metadata)
    add_data_catalog_table(connection, table_metadata)

    return connection, table_metadata


def print_table_metadata(table_metadata, keyword=None):
    """Print available tables with optional keyword filtering."""
    normalized_keyword = keyword.lower() if keyword else None
    rows = []
    for table_name, metadata in sorted(table_metadata.items()):
        relative_path = str(metadata["path"].relative_to(BASE_DIR))
        searchable = f"{table_name} {metadata['type']} {relative_path}".lower()
        if normalized_keyword and normalized_keyword not in searchable:
            continue
        rows.append((table_name, metadata, relative_path))

    title = "Available tables" if not keyword else f"Available tables matching '{keyword}'"
    print(f"\n{title}")
    print("-" * 80)
    for table_name, metadata, relative_path in rows:
        print(f"{table_name:58s} rows={metadata['row_count']:7d} type={metadata['type']:20s} source={relative_path}")
    print(f"\nTables shown: {len(rows)} / {len(table_metadata)}")


def print_schema(table_metadata, table_name):
    """Print the source path and columns for one loaded table."""
    if table_name not in table_metadata:
        print(f"Unknown table: {table_name}")
        return

    metadata = table_metadata[table_name]
    print(f"\n{table_name}")
    print("-" * 80)
    print(f"source: {metadata['path'].relative_to(BASE_DIR)}")
    print(f"rows:   {metadata['row_count']}")
    print("columns:")
    for column in metadata["columns"]:
        print(f"  - {column}")


def is_read_only_query(sql):
    """Return True when the SQL command is a single read-only SELECT query."""
    stripped = sql.strip().rstrip(";").strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    blocked = ("insert", "update", "delete", "drop", "alter", "create", "replace", "pragma", "attach")
    return lowered.startswith("select") and not any(re.search(rf"\b{word}\b", lowered) for word in blocked)


def print_query_rows(cursor, rows):
    """Print SQL query output in a compact aligned text table."""
    columns = [description[0] for description in cursor.description]
    if not rows:
        print("(no rows)")
        return

    widths = []
    for index, column in enumerate(columns):
        max_value_width = max(len(str(row[index])) for row in rows)
        widths.append(min(max(len(column), max_value_width), 28))

    header = " | ".join(column[:widths[index]].ljust(widths[index]) for index, column in enumerate(columns))
    print(header)
    print("-" * len(header))
    for row in rows:
        cells = [str(value)[:widths[index]].ljust(widths[index]) for index, value in enumerate(row)]
        print(" | ".join(cells))


def run_select(connection, sql):
    """Execute a read-only SQL query and print up to its returned rows."""
    if not is_read_only_query(sql):
        print("Only read-only SELECT queries are allowed.")
        return

    try:
        cursor = connection.execute(sql)
        rows = cursor.fetchall()
    except sqlite3.Error as exc:
        print(f"SQL error: {exc}")
        return

    print_query_rows(cursor, rows)
    print(f"\nRows returned: {len(rows)}")


def parse_key_value_arguments(command):
    """Parse simple key=value command arguments for the find helper."""
    pairs = {}
    for token in shlex.split(command)[1:]:
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        pairs[key.strip().lower()] = value.strip().strip('"').strip("'")
    return pairs


def run_find(connection, table_metadata, command):
    """Run a common filtered search without requiring users to write SQL."""
    args = parse_key_value_arguments(command)
    table_name = args.get("table", "features_extended")
    if table_name not in table_metadata:
        print(f"Unknown table: {table_name}")
        return

    limit = args.get("limit", "20")
    if not limit.isdigit():
        limit = "20"

    columns = table_metadata[table_name]["columns"]
    filters = []
    params = []

    filter_map = {
        "season": ("season", "="),
        "round": ("round", "="),
        "driver": ("driver_name", "LIKE"),
        "driver_id": ("driver_id", "="),
        "constructor": ("constructor_name", "LIKE"),
        "constructor_id": ("constructor_id", "="),
        "race": ("race_name", "LIKE"),
        "circuit": ("circuit_name", "LIKE"),
    }

    for key, (column, operator) in filter_map.items():
        if key not in args or column not in columns:
            continue
        value = args[key]
        if operator == "LIKE":
            value = f"%{value}%"
        filters.append(f"{quote_identifier(column)} {operator} ?")
        params.append(value)

    if not filters:
        print("No usable filters were provided. Try: find driver=hamilton season=2025")
        return

    preferred_columns = [
        "season",
        "round",
        "race_name",
        "driver_name",
        "constructor_name",
        "finish_position",
        "points",
        "predicted_points",
        "deterministic_rule_mapped_points",
        "champion_probability",
    ]
    selected_columns = [column for column in preferred_columns if column in columns]
    if not selected_columns:
        selected_columns = columns[:8]

    select_sql = ", ".join(quote_identifier(column) for column in selected_columns)
    where_sql = " AND ".join(filters)
    sql = f"SELECT {select_sql} FROM {quote_identifier(table_name)} WHERE {where_sql} LIMIT {int(limit)}"

    try:
        cursor = connection.execute(sql, params)
        rows = cursor.fetchall()
    except sqlite3.Error as exc:
        print(f"Search error: {exc}")
        return

    print_query_rows(cursor, rows)
    print(f"\nRows returned: {len(rows)}")


def interactive_loop(connection, table_metadata):
    """Run the terminal query loop until the user exits."""
    print("Formula 1 data query console. Type 'help' for commands, 'exit' to quit.")
    print_table_metadata(table_metadata)

    while True:
        try:
            command = input("\nf1-sql> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting query console.")
            return

        if not command:
            continue
        lowered = command.lower()
        if lowered in {"exit", "quit", "q"}:
            print("Exiting query console.")
            return
        if lowered == "help":
            print(HELP_TEXT)
            continue
        if lowered == "tables" or lowered.startswith("tables "):
            keyword = command.split(maxsplit=1)[1].strip() if " " in command else None
            print_table_metadata(table_metadata, keyword)
            continue
        if lowered.startswith("schema "):
            print_schema(table_metadata, command.split(maxsplit=1)[1].strip())
            continue
        if lowered.startswith("find "):
            run_find(connection, table_metadata, command)
            continue

        run_select(connection, command)


def main():
    """Build the in-memory database and start the query console."""
    if len(sys.argv) > 1 and sys.argv[1] in {"--help", "-h"}:
        print(HELP_TEXT)
        return

    connection, table_metadata = build_database()
    try:
        if len(sys.argv) > 1 and sys.argv[1] == "--tables":
            keyword = sys.argv[2] if len(sys.argv) > 2 else None
            print_table_metadata(table_metadata, keyword)
            return
        if len(sys.argv) > 2 and sys.argv[1] == "--schema":
            print_schema(table_metadata, sys.argv[2])
            return
        if len(sys.argv) > 2 and sys.argv[1] == "--query":
            run_select(connection, " ".join(sys.argv[2:]))
            return
        if len(sys.argv) > 2 and sys.argv[1] == "--find":
            run_find(connection, table_metadata, "find " + " ".join(sys.argv[2:]))
            return
        interactive_loop(connection, table_metadata)
    finally:
        connection.close()


if __name__ == "__main__":
    main()
