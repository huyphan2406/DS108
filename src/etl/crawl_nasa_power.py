import json
import time
from pathlib import Path
from datetime import datetime
import requests
import yaml

# =========================
# LOAD CONFIG
# =========================
BASE_DIR = Path(__file__).resolve().parents[2]  # lùi 2 cấp từ file script
LOCATIONS_FILE = BASE_DIR / "configs" / "locations.yaml"
PIPELINE_FILE = BASE_DIR / "configs" / "pipeline.yaml"

with open(LOCATIONS_FILE, "r", encoding="utf-8") as f:
    locations_config = yaml.safe_load(f)
LOCATIONS = locations_config.get("locations", [])

with open(PIPELINE_FILE, "r", encoding="utf-8") as f:
    pipeline_config = yaml.safe_load(f)

SOURCE_BASE_URL = pipeline_config["source"]["base_url"]
COMMUNITY = pipeline_config["request"].get("community", "RE")
FORMAT = pipeline_config["request"].get("format", "JSON")
HOUR_PARAMS = pipeline_config["parameters"].get("hourly", [])

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
DATE_START_HOURLY = DATE_START

# =========================
# HELPER FUNCTIONS
# =========================
def year_ranges(start, end):
    for y in range(start.year, end.year + 1):
        y_start = datetime(y, 1, 1)
        y_end = datetime(y, 12, 31)
        if y_start < start:
            y_start = start
        if y_end > end:
            y_end = end
        yield y_start, y_end

def call_nasa_api(lat, lon, start, end, params, temporal):
    url = (
        f"{SOURCE_BASE_URL}/{temporal}/point?"
        f"start={start.strftime('%Y%m%d')}&end={end.strftime('%Y%m%d')}"
        f"&latitude={lat}&longitude={lon}"
        f"&community={COMMUNITY}"
        f"&parameters={','.join(params)}"
        f"&format={FORMAT}"
    )
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as e:
            status_code = e.response.status_code if e.response is not None else None
            if status_code == 404:
                print(f"No data for {temporal} {lat},{lon}, skipping.")
                return None
            print(f"Attempt {attempt + 1} failed with HTTP error {status_code}: {e}")
            time.sleep(10)
        except requests.RequestException as e:
            print(f"Attempt {attempt + 1} failed with request error: {e}")
            time.sleep(10)
        except Exception as e:
            print(f"Attempt {attempt + 1} failed with unexpected error: {e}")
            time.sleep(10)
    return None

# =========================
# MAIN CRAWL FUNCTION
# =========================
def crawl_all_locations():
    if not LOCATIONS:
        print("No locations found in locations.yaml")
        return
    if not HOUR_PARAMS:
        print("No hourly parameters found in pipeline.yaml")
        return
    for loc in LOCATIONS:
        loc_id = loc["location_id"]
        lat = loc["lat"]
        lon = loc["lon"]
        province = loc["province"].replace(" ", "_")
        file_name = loc.get("file_name", f"{loc_id}_{province}")
        loc_folder = BRONZE_ROOT / file_name
        loc_folder.mkdir(parents=True, exist_ok=True)

        print(f"Start crawling {province} ({loc_id}) -> folder {loc_folder}")

        for y_start, y_end in year_ranges(DATE_START_HOURLY, DATE_END):
            data_hourly = call_nasa_api(lat, lon, y_start, y_end, HOUR_PARAMS, "hourly")

            if data_hourly:
                year_file = loc_folder / f"{y_start.year}_hourly.json"
                with open(year_file, "w", encoding="utf-8") as f:
                    json.dump(data_hourly, f, ensure_ascii=False)

                print(f"Saved hourly: {year_file}")

            time.sleep(2)

# =========================
# RUN
# =========================
if __name__ == "__main__":
    crawl_all_locations()