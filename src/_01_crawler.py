"""Step 1: Crawl NOAA GSOD raw station data and save Bronze artifacts."""

import pandas as pd
import requests
import io
import time
import json
import warnings
from pathlib import Path
from datetime import datetime, timezone

warnings.filterwarnings("ignore")

# CONFIGURATION

# NOAA NCEI station structure: USAF (6 digits) + WBAN (5 digits)
TARGET_STATIONS = {
    "48825099999": "HA DONG (Đại diện Bắc Bộ)",
    "48845099999": "VINH (Đại diện Bắc Trung Bộ)",
    "48863099999": "QUANG NGAI (Đại diện miền Trung)",
    "48877099999": "NHA TRANG (Đại diện Nam Trung Bộ)",
    "48914099999": "CA MAU (Đại diện Nam Bộ)"
}

YEARS = range(2015, 2025)
BASE_URL = "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/"
REQUEST_TIMEOUT = 15
REQUEST_DELAY = 0.2

# Get base directory
BASE_DIR = Path(__file__).resolve().parent.parent

OUTPUT_DIR = BASE_DIR / "data" / "raw" / "gsod"
OUTPUT_FILE = OUTPUT_DIR / "bronze_data.csv"
OUTPUT_METADATA_FILE = OUTPUT_DIR / "bronze_metadata.json"
OUTPUT_STATION_SUMMARY_FILE = OUTPUT_DIR / "bronze_station_summary.csv"

# 1. DATA CRAWLING

def _fetch_station_data(
    stn_id: str,
    stn_name: str,
    session: requests.Session
) -> tuple[list[pd.DataFrame], int]:
    print(f"📍 Đang xử lý trạm: {stn_name} (Mã: {stn_id})")

    station_data = []
    success_years = 0

    for year in YEARS:
        url = f"{BASE_URL}{year}/{stn_id}.csv"

        try:
            res = session.get(url, timeout=REQUEST_TIMEOUT)

            if res.status_code == 200:
                df_tmp = pd.read_csv(io.StringIO(res.text))

                if "STATION" in df_tmp.columns:
                    df_tmp["STATION"] = df_tmp["STATION"].astype(str)

                df_tmp["station_name"] = stn_name
                df_tmp["source_year"] = year
                df_tmp["source_url"] = url

                station_data.append(df_tmp)
                success_years += 1

                print(f"   + Năm {year}: Tải THÀNH CÔNG")
            else:
                print(f"   - Năm {year}: Không có dữ liệu trên server (HTTP {res.status_code})")

        except Exception as e:
            print(f"   - Năm {year}: Lỗi kết nối mạng ({e})")

        time.sleep(REQUEST_DELAY)

    print(f"-> Tóm tắt trạm {stn_name}: Tải thành công {success_years}/{len(YEARS)} năm.\n")

    return station_data, success_years

def _build_station_summary(final_df: pd.DataFrame) -> pd.DataFrame:
    if "DATE" in final_df.columns:
        date_col = "DATE"
    else:
        date_col = None

    summary_rows = []

    for station_id, group in final_df.groupby("STATION"):
        row = {
            "STATION": station_id,
            "station_name": group["station_name"].iloc[0] if "station_name" in group.columns else None,
            "n_rows": len(group),
            "n_years": group["source_year"].nunique() if "source_year" in group.columns else None,
        }

        if date_col is not None:
            row["start_date"] = group[date_col].min()
            row["end_date"] = group[date_col].max()
            row["n_unique_days"] = group[date_col].nunique()

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)

def _save_metadata(
    output_path: Path,
    station_summary: pd.DataFrame,
    total_rows: int,
    total_columns: int
) -> None:
    metadata = {
        "dataset_layer": "bronze/raw",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "NOAA NCEI Global Summary of the Day",
        "base_url": BASE_URL,
        "years": [YEARS.start, YEARS.stop - 1],
        "target_stations": TARGET_STATIONS,
        "output_file": str(output_path),
        "n_rows": int(total_rows),
        "n_columns": int(total_columns),
        "n_stations": int(len(station_summary)),
        "note": (
            "This is raw Bronze data downloaded from NOAA GSOD. "
            "No deep cleaning, imputation, unit conversion or feature engineering is performed here."
        )
    }

    with open(OUTPUT_METADATA_FILE, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

def _merge_and_save_data(all_data: list[pd.DataFrame], output_path: Path) -> None:
    print("--- 🛠️ ĐANG GỘP DỮ LIỆU (RAW DATA) ---")

    final_df = pd.concat(all_data, ignore_index=True)

    if "DATE" in final_df.columns:
        final_df["DATE"] = pd.to_datetime(final_df["DATE"], errors="coerce")
        final_df = final_df.sort_values(["STATION", "DATE"]).reset_index(drop=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    final_df.to_csv(output_path, index=False)

    station_summary = _build_station_summary(final_df)
    station_summary.to_csv(OUTPUT_STATION_SUMMARY_FILE, index=False)

    _save_metadata(
        output_path=output_path,
        station_summary=station_summary,
        total_rows=len(final_df),
        total_columns=len(final_df.columns)
    )

    print("✅ HOÀN THÀNH XUẤT SẮC!")
    print(f"- Bronze data: {output_path}")
    print(f"- Station summary: {OUTPUT_STATION_SUMMARY_FILE}")
    print(f"- Metadata: {OUTPUT_METADATA_FILE}")
    print(f"- Tổng số dòng (records): {len(final_df):,}")
    print(f"- Tổng số cột (features): {len(final_df.columns)}")

# MAIN PIPELINE

def crawl_weather_data(output_path: str | Path = None) -> None:
    if output_path is None:
        output_path = OUTPUT_FILE

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== 🛰️ BẮT ĐẦU CRAWL DỮ LIỆU BRONZE TỪ NOAA ===")
    print("- Mục tiêu: 5 trạm đại diện")
    print(f"- Giai đoạn: {YEARS.start} đến {YEARS.stop - 1}")
    print(f"- Output folder: {OUTPUT_DIR}\n")

    all_data = []
    station_download_summary = []

    session = requests.Session()

    for stn_id, stn_name in TARGET_STATIONS.items():
        station_data, success_years = _fetch_station_data(stn_id, stn_name, session)
        all_data.extend(station_data)

        station_download_summary.append({
            "STATION": stn_id,
            "station_name": stn_name,
            "success_years": success_years,
            "expected_years": len(YEARS)
        })

    if all_data:
        _merge_and_save_data(all_data, output_path)
    else:
        print("\n❌ LỖI: Không có dữ liệu nào được tải về. Hãy kiểm tra lại kết nối mạng!")

if __name__ == "__main__":
    crawl_weather_data()
