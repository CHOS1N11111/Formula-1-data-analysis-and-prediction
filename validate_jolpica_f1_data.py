"""Validate downloaded Jolpica-F1 JSON files against API-declared totals."""

import json
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent / "data" / "raw" / "jolpica"
YEARS = range(2019, 2027)


ENDPOINT_COLLECTIONS = {
    "races": ("RaceTable", "Races"),
    "results": ("RaceTable", "Races", "Results"),
    "qualifying": ("RaceTable", "Races", "QualifyingResults"),
    "drivers": ("DriverTable", "Drivers"),
    "constructors": ("ConstructorTable", "Constructors"),
    "driverStandings": ("StandingsTable", "StandingsLists", "DriverStandings"),
    "constructorStandings": ("StandingsTable", "StandingsLists", "ConstructorStandings"),
}


def load_json(path):
    """Load one saved Jolpica-F1 JSON file."""
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def count_records_in_page(page, endpoint):
    mr_data = page["MRData"]
    keys = ENDPOINT_COLLECTIONS[endpoint]

    if endpoint in {"results", "qualifying"}:
        table = mr_data[keys[0]]
        races = table.get(keys[1], [])
        return sum(len(race.get(keys[2], [])) for race in races)

    if endpoint in {"driverStandings", "constructorStandings"}:
        table = mr_data[keys[0]]
        standings_lists = table.get(keys[1], [])
        return sum(len(item.get(keys[2], [])) for item in standings_lists)

    table = mr_data[keys[0]]
    return len(table.get(keys[1], []))


def validate_file(year, endpoint):
    path = DATA_DIR / str(year) / f"{endpoint}.json"
    if not path.exists():
        return {
            "year": year,
            "endpoint": endpoint,
            "status": "missing",
            "declared_total": None,
            "actual_total": 0,
            "pages": 0,
        }

    data = load_json(path)
    pages = data.get("pages", [])
    declared_total = int(data.get("total", 0))
    actual_total = sum(count_records_in_page(page, endpoint) for page in pages)
    status = "ok" if actual_total == declared_total else "incomplete"

    return {
        "year": year,
        "endpoint": endpoint,
        "status": status,
        "declared_total": declared_total,
        "actual_total": actual_total,
        "pages": len(pages),
    }


def main():
    has_problem = False

    for year in YEARS:
        for endpoint in ENDPOINT_COLLECTIONS:
            result = validate_file(year, endpoint)
            status = result["status"]
            if status != "ok":
                has_problem = True

            print(
                f"{result['year']} {result['endpoint']}: "
                f"{status}, declared={result['declared_total']}, "
                f"actual={result['actual_total']}, pages={result['pages']}"
            )

    if has_problem:
        raise SystemExit("Validation failed: some files are missing or incomplete.")

    print("Validation passed: all downloaded Jolpica-F1 files match declared totals.")


if __name__ == "__main__":
    main()
