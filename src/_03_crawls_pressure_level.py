"""Step 3: Download ERA5 era5_pressure_level GRIB files by year."""

import cdsapi
from pathlib import Path
import time

# CONFIGURATION

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET = "reanalysis-era5-pressure-levels"
OUTPUT_BASE_DIR = BASE_DIR / "data" / "raw" / "era5_pressure_level"

YEARS = range(2015, 2025)

MONTHS = [
    "01", "02", "03", "04",
    "05", "06", "07", "08",
    "09", "10", "11", "12"
]

DAYS = [
    "01", "02", "03", "04", "05", "06", "07", "08", "09", "10",
    "11", "12", "13", "14", "15", "16", "17", "18", "19", "20",
    "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31"
]

TIMES = ["00:00", "06:00", "12:00", "18:00"]

PRESSURE_LEVELS = ["500", "850"]

# Vietnam bounding box: North, West, South, East
AREA = [24, 102, 8, 110]

VARIABLES = [
    "geopotential",
    "specific_humidity",
    "temperature",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
]

# ERA5 PRESSURE-LEVEL CRAWLER

def build_request(year: int) -> dict:
    return {
        "product_type": ["reanalysis"],
        "variable": VARIABLES,
        "year": [str(year)],
        "month": MONTHS,
        "day": DAYS,
        "time": TIMES,
        "pressure_level": PRESSURE_LEVELS,
        "data_format": "grib",
        "download_format": "unarchived",
        "area": AREA,
    }

def download_pressure_year(client: cdsapi.Client, year: int) -> None:
    output_dir = OUTPUT_BASE_DIR / str(year)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "data.grib"

    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"✅ {year}: File đã tồn tại, bỏ qua: {output_path}")
        return

    request = build_request(year)

    print(f"\n⬇️ Đang tải ERA5 era5_pressure_level năm {year}...")
    print(f"Output: {output_path}")

    try:
        client.retrieve(DATASET, request).download(str(output_path))

        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"✅ {year}: Tải thành công.")
        else:
            print(f"⚠️ {year}: File tải về rỗng hoặc không tồn tại, cần kiểm tra lại.")

    except Exception as e:
        print(f"❌ {year}: Lỗi khi tải ERA5 era5_pressure_level: {e}")

def crawl_pressure() -> None:
    print("=== START ERA5 PRESSURE-LEVEL CRAWLING ===")
    print(f"Output base directory: {OUTPUT_BASE_DIR}")
    print(f"Years: {YEARS.start}–{YEARS.stop - 1}")
    print(f"Pressure levels: {PRESSURE_LEVELS}")
    print(f"Variables: {len(VARIABLES)}")
    print(f"Area: {AREA}")

    client = cdsapi.Client()

    for year in YEARS:
        download_pressure_year(client, year)
        time.sleep(1)

    print("\n✅ ERA5 era5_pressure_level crawling completed.")

if __name__ == "__main__":
    crawl_pressure()
