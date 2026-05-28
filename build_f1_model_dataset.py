import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


RAW_DIR = Path(__file__).resolve().parent / "data" / "raw" / "jolpica"
PROCESSED_DIR = Path(__file__).resolve().parent / "data" / "processed"
MODEL_DATASET_PATH = PROCESSED_DIR / "f1_model_dataset.csv"
SCHEDULE_2026_PATH = PROCESSED_DIR / "f1_2026_schedule.csv"
SUMMARY_PATH = PROCESSED_DIR / "build_summary.json"

YEARS = range(2019, 2027)


MODEL_FIELDS = [
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

SCHEDULE_FIELDS = [
    "season",
    "round",
    "race_name",
    "race_date",
    "race_time",
    "circuit_id",
    "circuit_name",
    "circuit_locality",
    "circuit_country",
    "has_result",
]


def load_endpoint(year, endpoint):
    path = RAW_DIR / str(year) / f"{endpoint}.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing source file: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def iter_races(data):
    for page in data.get("pages", []):
        race_table = page.get("MRData", {}).get("RaceTable", {})
        for race in race_table.get("Races", []):
            yield race


def to_int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def bool_to_int(value):
    if value is None:
        return ""
    return 1 if value else 0


def get_driver_name(driver):
    given_name = driver.get("givenName", "")
    family_name = driver.get("familyName", "")
    return f"{given_name} {family_name}".strip()


def get_race_base_fields(race):
    circuit = race.get("Circuit", {})
    location = circuit.get("Location", {})

    return {
        "season": to_int(race.get("season")),
        "round": to_int(race.get("round")),
        "race_name": race.get("raceName", ""),
        "race_date": race.get("date", ""),
        "race_time": race.get("time", ""),
        "circuit_id": circuit.get("circuitId", ""),
        "circuit_name": circuit.get("circuitName", ""),
        "circuit_locality": location.get("locality", ""),
        "circuit_country": location.get("country", ""),
    }


def build_qualifying_lookup(year):
    """Map (season, round, driver_id) to qualifying position."""
    lookup = {}
    data = load_endpoint(year, "qualifying")

    for race in iter_races(data):
        season = to_int(race.get("season"))
        round_number = to_int(race.get("round"))

        for result in race.get("QualifyingResults", []):
            driver = result.get("Driver", {})
            driver_id = driver.get("driverId", "")
            key = (season, round_number, driver_id)
            lookup[key] = to_int(result.get("position"))

    return lookup


def build_model_rows():
    rows = []
    rows_by_year = defaultdict(int)
    completed_rounds_by_year = defaultdict(set)

    for year in YEARS:
        qualifying_lookup = build_qualifying_lookup(year)
        results_data = load_endpoint(year, "results")

        for race in iter_races(results_data):
            race_fields = get_race_base_fields(race)
            season = race_fields["season"]
            round_number = race_fields["round"]
            completed_rounds_by_year[season].add(round_number)

            for result in race.get("Results", []):
                driver = result.get("Driver", {})
                constructor = result.get("Constructor", {})
                driver_id = driver.get("driverId", "")
                finish_position = to_int(result.get("position"))

                row = {
                    **race_fields,
                    "driver_id": driver_id,
                    "driver_code": driver.get("code", ""),
                    "driver_name": get_driver_name(driver),
                    "driver_nationality": driver.get("nationality", ""),
                    "constructor_id": constructor.get("constructorId", ""),
                    "constructor_name": constructor.get("name", ""),
                    "constructor_nationality": constructor.get("nationality", ""),
                    "grid": to_int(result.get("grid")),
                    "qualifying_position": qualifying_lookup.get(
                        (season, round_number, driver_id), ""
                    ),
                    "finish_position": finish_position if finish_position is not None else "",
                    "points": to_float(result.get("points")),
                    "laps": to_int(result.get("laps")),
                    "status": result.get("status", ""),
                    "is_podium": bool_to_int(
                        finish_position is not None and finish_position <= 3
                    ),
                    "is_top10": bool_to_int(
                        finish_position is not None and finish_position <= 10
                    ),
                    "data_source": "Jolpica-F1",
                }
                rows.append(row)
                rows_by_year[season] += 1

    return rows, rows_by_year, completed_rounds_by_year


def build_2026_schedule(completed_rounds_by_year):
    rows = []
    data = load_endpoint(2026, "races")
    completed_rounds = completed_rounds_by_year.get(2026, set())

    for race in iter_races(data):
        row = get_race_base_fields(race)
        row["has_result"] = bool_to_int(row["round"] in completed_rounds)
        rows.append(row)

    return rows


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


def main():
    model_rows, rows_by_year, completed_rounds_by_year = build_model_rows()
    schedule_2026_rows = build_2026_schedule(completed_rounds_by_year)

    write_csv(MODEL_DATASET_PATH, MODEL_FIELDS, model_rows)
    write_csv(SCHEDULE_2026_PATH, SCHEDULE_FIELDS, schedule_2026_rows)

    summary = {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "source": "data/raw/jolpica",
        "model_dataset": str(MODEL_DATASET_PATH.relative_to(Path(__file__).resolve().parent)),
        "schedule_2026": str(SCHEDULE_2026_PATH.relative_to(Path(__file__).resolve().parent)),
        "rows_total": len(model_rows),
        "rows_by_year": dict(sorted(rows_by_year.items())),
        "completed_rounds_by_year": {
            str(year): sorted(rounds)
            for year, rounds in sorted(completed_rounds_by_year.items())
        },
        "fields": MODEL_FIELDS,
    }
    write_json(SUMMARY_PATH, summary)

    print(f"Model dataset saved to: {MODEL_DATASET_PATH}")
    print(f"2026 schedule saved to: {SCHEDULE_2026_PATH}")
    print(f"Summary saved to: {SUMMARY_PATH}")
    print(f"Rows: {len(model_rows)}")


if __name__ == "__main__":
    main()
