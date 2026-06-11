"""Download paginated Jolpica-F1 JSON data for modern seasons.

This setup script calls the Jolpica-F1 API for races, results, qualifying,
standings, drivers, and constructors from 2019 to 2026. It downloads paginated
responses and writes the raw JSON files unchanged under data/raw/jolpica so
validation and processing scripts can rebuild downstream datasets reproducibly.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen


BASE_URL = "https://api.jolpi.ca/ergast/f1"
OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "raw" / "jolpica"

# 2019-2025 provide complete modern historical seasons; 2026 provides the current
# season schedule, completed results, and standings used by the final predictor.
YEARS = range(2019, 2027)

ENDPOINTS = [
    "races",
    "results",
    "qualifying",
    "driverStandings",
    "constructorStandings",
    "drivers",
    "constructors",
]

PAGE_LIMIT = 1000
REQUEST_SLEEP_SECONDS = 0.25


def fetch_json(url, timeout=30):
    """Request one JSON page from the Jolpica-F1 API with clear error messages."""
    try:
        with urlopen(url, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} while requesting {url}") from exc
    except URLError as exc:
        raise RuntimeError(f"Network error while requesting {url}: {exc.reason}") from exc


def get_total_count(data):
    """Extract the API-declared total number of records from a response."""
    race_table = data.get("MRData", {})
    total = race_table.get("total", 0)
    return int(total)


def get_returned_limit(data):
    """Extract the API page limit used for pagination."""
    race_table = data.get("MRData", {})
    limit = race_table.get("limit", PAGE_LIMIT)
    return int(limit)


def build_url(year, endpoint, offset):
    """Build one paginated Jolpica-F1 endpoint URL."""
    query = urlencode({"limit": PAGE_LIMIT, "offset": offset})
    return f"{BASE_URL}/{year}/{endpoint}.json?{query}"



def download_endpoint(year, endpoint):
    """Download every page for one season and endpoint."""
    all_pages = []
    offset = 0
    total = None

    while total is None or offset < total:
        url = build_url(year, endpoint, offset)
        print(f"Downloading {year} {endpoint}, offset={offset}")
        data = fetch_json(url)

        if total is None:
            total = get_total_count(data)

        all_pages.append(data)
        offset += get_returned_limit(data)
        time.sleep(REQUEST_SLEEP_SECONDS)

        if total == 0:
            break

    return {
        "source": "Jolpica-F1 Ergast-compatible API",
        "base_url": BASE_URL,
        "year": year,
        "endpoint": endpoint,
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
        "total": total if total is not None else 0,
        "pages": all_pages,
    }


def save_json(data, path):
    """Save one JSON object to disk using UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def main():
    """Run the script end-to-end and write all configured outputs."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = []
    for year in YEARS:
        year_dir = OUTPUT_DIR / str(year)
        for endpoint in ENDPOINTS:
            output_path = year_dir / f"{endpoint}.json"
            data = download_endpoint(year, endpoint)
            save_json(data, output_path)

            summary.append(
                {
                    "year": year,
                    "endpoint": endpoint,
                    "total": data["total"],
                    "file": str(output_path.relative_to(Path(__file__).resolve().parent)),
                }
            )

    save_json(
        {
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "years": [min(YEARS), max(YEARS)],
            "endpoints": ENDPOINTS,
            "records": summary,
        },
        OUTPUT_DIR / "download_summary.json",
    )

    print(f"Finished. Raw JSON files saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
