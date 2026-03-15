import json
import time
from pathlib import Path
from datetime import datetime

import requests
import yaml

# =========================
# LOAD CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parents[2]
LOCATIONS_FILE = BASE_DIR / "configs" / "locations.yaml"
PIPELINE_FILE = BASE_DIR / "configs" / "pipeline.yaml"

with open(LOCATIONS_FILE, "r", encoding="utf-8") as f:
    locations_config = yaml.safe_load(f)
LOCATIONS = locations_config.get("locations", [])

with open(PIPELINE_FILE, "r", encoding="utf-8") as f:
    pipeline_config = yaml.safe_load(f)

SOURCE_BASE_URL = pipeline_config["source"]["base_url"]
COMMUNITY = pipeline_config["request"].get("community", "AG")
FORMAT = pipeline_config["request"].get("format", "JSON")
TIME_STANDARD = pipeline_config["request"].get("time_standard", "UTC")
DAILY_PARAMS = pipeline_config["parameters"].get("daily", [])

# =========================
# STORAGE
# =========================
BRONZE_ROOT = (BASE_DIR / "data" / "bronze").resolve()
BRONZE_ROOT.mkdir(parents=True, exist_ok=True)
print("BRONZE_ROOT =", BRONZE_ROOT)

# =========================
# DATE RANGE
# =========================
DATE_START = datetime.strptime(pipeline_config["date_range"]["start_date"], "%Y%m%d")
DATE_END = datetime.strptime(pipeline_config["date_range"]["end_date"], "%Y%m%d")

# =========================
# CONSTANTS
# =========================
TARGET_COLUMN = "PRECTOTCORR"
METADATA_COLUMNS = ["DATE", "LOCATION", "LATITUDE", "LONGITUDE"]
DATASET_COLUMNS = METADATA_COLUMNS + DAILY_PARAMS

# =========================
# HELPER FUNCTIONS
# =========================
def validate_config():
    if not LOCATIONS:
        raise ValueError("No locations found in locations.yaml")

    if not DAILY_PARAMS:
        raise ValueError("No daily parameters found in pipeline.yaml")

    if TARGET_COLUMN not in DAILY_PARAMS:
        raise ValueError(f"Target '{TARGET_COLUMN}' must exist in daily parameters")

    if len(DAILY_PARAMS) > 20:
        raise ValueError(
            f"NASA POWER Daily Point API allows up to 20 parameters per request, "
            f"but got {len(DAILY_PARAMS)}"
        )

def year_ranges(start, end):
    for y in range(start.year, end.year + 1):
        y_start = datetime(y, 1, 1)
        y_end = datetime(y, 12, 31)

        if y_start < start:
            y_start = start
        if y_end > end:
            y_end = end

        yield y_start, y_end

def build_request_url_and_params(lat, lon, start, end, params, temporal):
    url = f"{SOURCE_BASE_URL}/{temporal}/point"
    query_params = {
        "start": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
        "latitude": lat,
        "longitude": lon,
        "community": COMMUNITY,
        "parameters": ",".join(params),
        "format": FORMAT,
        "time-standard": TIME_STANDARD,
    }
    return url, query_params

def call_nasa_api(lat, lon, start, end, params, temporal):
    url, query_params = build_request_url_and_params(lat, lon, start, end, params, temporal)

    for attempt in range(3):
        try:
            resp = requests.get(url, params=query_params, timeout=60)
            resp.raise_for_status()
            return resp.json()

        except requests.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else None
            response_text = e.response.text[:500] if e.response is not None else ""

            if status_code == 404:
                print(f"No data for {temporal} {lat},{lon}, skipping.")
                return None

            print(f"Attempt {attempt + 1} failed with HTTP error {status_code}: {e}")
            if response_text:
                print(f"Response preview: {response_text}")
            time.sleep(10)

        except requests.RequestException as e:
            print(f"Attempt {attempt + 1} failed with request error: {e}")
            time.sleep(10)

        except json.JSONDecodeError as e:
            print(f"Attempt {attempt + 1} failed with JSON decode error: {e}")
            time.sleep(10)

        except Exception as e:
            print(f"Attempt {attempt + 1} failed with unexpected error: {e}")
            time.sleep(10)

    return None

def build_output_payload(loc, start, end, data):
    """
    Gói thêm metadata location/dataset để về sau dễ chuyển sang bảng cột:
    DATE, LOCATION, LATITUDE, LONGITUDE, + 15 biến daily.
    """
    return {
        "metadata": {
            "source": "NASA POWER",
            "temporal": "daily",
            "community": COMMUNITY,
            "format": FORMAT,
            "time_standard": TIME_STANDARD,
            "start_date": start.strftime("%Y%m%d"),
            "end_date": end.strftime("%Y%m%d"),
            "location_id": loc["location_id"],
            "location": loc["province"],
            "file_name": loc.get("file_name"),
            "latitude": loc["lat"],
            "longitude": loc["lon"],
            "target_column": TARGET_COLUMN,
            "feature_columns": [p for p in DAILY_PARAMS if p != TARGET_COLUMN],
            "dataset_columns": DATASET_COLUMNS,
            "requested_parameters": DAILY_PARAMS,
        },
        "nasa_power_response": data,
    }

# =========================
# MAIN CRAWL FUNCTION
# =========================
def crawl_all_locations():
    validate_config()

    for loc in LOCATIONS:
        loc_id = loc["location_id"]
        lat = loc["lat"]
        lon = loc["lon"]
        province = loc["province"].replace(" ", "_")
        file_name = loc.get("file_name", f"{loc_id}_{province}")

        loc_folder = BRONZE_ROOT / file_name
        loc_folder.mkdir(parents=True, exist_ok=True)

        print(f"Start crawling DAILY {province} ({loc_id}) -> folder {loc_folder}")

        for y_start, y_end in year_ranges(DATE_START, DATE_END):
            data_daily = call_nasa_api(lat, lon, y_start, y_end, DAILY_PARAMS, "daily")

            if data_daily:
                output_payload = build_output_payload(loc, y_start, y_end, data_daily)
                year_file = loc_folder / f"{y_start.year}_daily.json"

                with open(year_file, "w", encoding="utf-8") as f:
                    json.dump(output_payload, f, ensure_ascii=False, indent=2)

                print(f"Saved daily: {year_file}")

            time.sleep(2)

# =========================
# RUN
# =========================
if __name__ == "__main__":
    crawl_all_locations()