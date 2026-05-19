"""
01_crawler.py

Bronze/raw data collection pipeline for NOAA GSOD weather data.

Mục tiêu của bước này:
1. Tự động tải dữ liệu GSOD từ NOAA NCEI cho 5 trạm đại diện Việt Nam.
2. Lưu dữ liệu ở tầng Bronze/raw: KHÔNG làm sạch giá trị khí tượng, KHÔNG đổi đơn vị,
   KHÔNG impute, KHÔNG loại outlier.
3. Chỉ chuẩn hóa tối thiểu để phục vụ tái lập:
   - Ép STATION về string để tránh mất số 0.
   - Parse DATE để kiểm tra/sắp xếp thời gian.
   - Thêm các cột provenance bắt đầu bằng "_source_*".
4. Lưu metadata đầy đủ: thời điểm crawl, URL nguồn, số dòng từng trạm, năm tải thiếu,
   độ bao phủ theo trạm và cảnh báo chất lượng.
5. Có retry/backoff cho lỗi mạng hoặc lỗi server tạm thời.
"""

from __future__ import annotations

import io
import json
import time
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

warnings.filterwarnings("ignore")


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DIR = BASE_DIR / "data" / "raw" / "gsod"
OUTPUT_FILE = "bronze_data.csv"
METADATA_FILE = "bronze_metadata.json"
STATION_SUMMARY_FILE = "bronze_station_summary.csv"

BASE_URL = "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/"
YEARS = list(range(2015, 2025))  # 2015-2024

REQUEST_TIMEOUT = 20
REQUEST_DELAY = 0.25

MAX_RETRIES = 3
BACKOFF_FACTOR = 1.5
MAX_BACKOFF_SECONDS = 12

RETRY_STATUS_CODES = {429, 500, 502, 503, 504}

# Điều kiện tối thiểu để xem dữ liệu "đủ" cho đồ án.
# Với 10 năm, 0.9 nghĩa là mỗi trạm nên có ít nhất 9/10 file năm.
MIN_STATION_YEAR_COVERAGE = 0.90
MIN_TOTAL_YEAR_COVERAGE = 0.90


@dataclass(frozen=True)
class StationConfig:
    """
    Cấu hình trạm đại diện.

    station_id:
        Mã NOAA GSOD dạng USAF(6) + WBAN(5).
    station_name:
        Tên trạm.
    representative_region:
        Vùng khí hậu/khu vực được trạm đại diện.
    rationale:
        Lý do chọn trạm để đưa vào báo cáo.
    """

    station_id: str
    station_name: str
    representative_region: str
    rationale: str


TARGET_STATIONS: list[StationConfig] = [
    StationConfig(
        station_id="48825099999",
        station_name="HA DONG",
        representative_region="Bắc Bộ",
        rationale=(
            "Đại diện khu vực Bắc Bộ, nơi có đặc trưng mùa đông lạnh hơn, "
            "mưa mùa hạ và chịu ảnh hưởng của gió mùa Đông Bắc."
        ),
    ),
    StationConfig(
        station_id="48845099999",
        station_name="VINH",
        representative_region="Bắc Trung Bộ",
        rationale=(
            "Đại diện Bắc Trung Bộ, khu vực chuyển tiếp giữa khí hậu Bắc Bộ "
            "và Trung Bộ, thường chịu ảnh hưởng của gió mùa, bão và mưa lớn."
        ),
    ),
    StationConfig(
        station_id="48863099999",
        station_name="QUANG NGAI",
        representative_region="Trung Bộ",
        rationale=(
            "Đại diện khu vực duyên hải miền Trung, nơi có mưa tập trung theo mùa "
            "và chịu ảnh hưởng mạnh của bão/áp thấp nhiệt đới."
        ),
    ),
    StationConfig(
        station_id="48877099999",
        station_name="NHA TRANG",
        representative_region="Nam Trung Bộ",
        rationale=(
            "Đại diện Nam Trung Bộ, khu vực khô hơn tương đối so với Trung Bộ "
            "và có chế độ mưa chịu ảnh hưởng biển rõ rệt."
        ),
    ),
    StationConfig(
        station_id="48914099999",
        station_name="CA MAU",
        representative_region="Nam Bộ",
        rationale=(
            "Đại diện Nam Bộ, khu vực có tính chất nhiệt đới gió mùa cận xích đạo, "
            "mùa mưa và mùa khô phân hóa rõ."
        ),
    ),
]


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ensure_output_dirs(*paths: Path) -> None:
    """Create parent folders for all output files if they do not exist."""
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def build_noaa_url(year: int, station_id: str) -> str:
    """Build NOAA GSOD CSV URL for a station-year pair."""
    return f"{BASE_URL}{year}/{station_id}.csv"


def _safe_sleep(seconds: float) -> None:
    """Sleep only when seconds > 0."""
    if seconds > 0:
        time.sleep(seconds)


def request_with_retry(
    session: requests.Session,
    url: str,
    timeout: int = REQUEST_TIMEOUT,
    max_retries: int = MAX_RETRIES,
    backoff_factor: float = BACKOFF_FACTOR,
) -> tuple[requests.Response | None, str | None]:
    """
    Request a URL with retry/backoff.

    Không retry lỗi 404 vì 404 thường nghĩa là file station-year không tồn tại.
    Có retry cho lỗi tạm thời như 429/5xx hoặc lỗi kết nối.
    """
    last_error: str | None = None

    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(url, timeout=timeout)

            if response.status_code == 200:
                return response, None

            if response.status_code == 404:
                return response, "not_found_404"

            if response.status_code in RETRY_STATUS_CODES:
                last_error = f"http_{response.status_code}"
            else:
                return response, f"http_{response.status_code}"

        except requests.RequestException as exc:
            last_error = f"request_exception: {type(exc).__name__}: {exc}"

        if attempt < max_retries:
            sleep_seconds = min(
                backoff_factor * (2 ** (attempt - 1)),
                MAX_BACKOFF_SECONDS,
            )
            print(
                f"      Retry {attempt}/{max_retries - 1} sau {sleep_seconds:.1f}s "
                f"do lỗi: {last_error}"
            )
            _safe_sleep(sleep_seconds)

    return None, last_error


def read_station_year_csv(
    csv_text: str,
    station: StationConfig,
    year: int,
    source_url: str,
    crawled_at_utc: str,
) -> pd.DataFrame:
    """
    Read NOAA CSV text into DataFrame.

    Đây vẫn là Bronze/raw layer:
    - Không sửa mã lỗi như 999.9.
    - Không đổi đơn vị.
    - Không impute.
    - Không loại outlier.
    - Chỉ thêm provenance để truy vết nguồn dữ liệu.
    """
    df = pd.read_csv(io.StringIO(csv_text), low_memory=False)

    if "STATION" in df.columns:
        df["STATION"] = df["STATION"].astype(str)
    else:
        df["STATION"] = station.station_id

    # Provenance columns: giúp truy vết nguồn, không phải làm sạch dữ liệu khí tượng.
    df["_source_url"] = source_url
    df["_source_year"] = year
    df["_source_station_id"] = station.station_id
    df["_source_station_name"] = station.station_name
    df["_representative_region"] = station.representative_region
    df["_crawl_timestamp_utc"] = crawled_at_utc

    return df


def summarize_dataframe_dates(df: pd.DataFrame) -> dict[str, Any]:
    """Return date-level quality summary for the merged bronze dataframe."""
    if "DATE" not in df.columns:
        return {
            "has_DATE_column": False,
            "invalid_DATE_count": None,
            "min_DATE": None,
            "max_DATE": None,
            "duplicate_station_date_count": None,
        }

    parsed_dates = pd.to_datetime(df["DATE"], errors="coerce")
    invalid_date_count = int(parsed_dates.isna().sum())

    duplicate_count = None
    if "STATION" in df.columns:
        tmp = pd.DataFrame({"STATION": df["STATION"].astype(str), "DATE": parsed_dates})
        duplicate_count = int(tmp.duplicated(subset=["STATION", "DATE"]).sum())

    return {
        "has_DATE_column": True,
        "invalid_DATE_count": invalid_date_count,
        "min_DATE": None if parsed_dates.dropna().empty else str(parsed_dates.min().date()),
        "max_DATE": None if parsed_dates.dropna().empty else str(parsed_dates.max().date()),
        "duplicate_station_date_count": duplicate_count,
    }


def sort_bronze_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort raw data by station and date.

    Lưu ý: Đây là chuẩn hóa tối thiểu phục vụ reproducibility.
    Không thay đổi giá trị khí tượng.
    """
    if "DATE" not in df.columns:
        return df.reset_index(drop=True)

    df = df.copy()
    df["_DATE_SORT_KEY"] = pd.to_datetime(df["DATE"], errors="coerce")

    sort_cols = []
    if "STATION" in df.columns:
        sort_cols.append("STATION")
    sort_cols.append("_DATE_SORT_KEY")

    df = df.sort_values(sort_cols).drop(columns=["_DATE_SORT_KEY"]).reset_index(drop=True)
    return df


# ============================================================================
# REPRESENTATIVENESS AND QUALITY CHECKS
# ============================================================================

def validate_station_configuration(stations: list[StationConfig]) -> list[str]:
    """
    Check whether the station set is representative at configuration level.

    Hàm này không thể chứng minh tuyệt đối tính đại diện khí hậu,
    nhưng giúp đảm bảo thiết kế mẫu có đủ 5 vùng chính và không trùng mã trạm.
    """
    warnings_list: list[str] = []

    required_regions = {"Bắc Bộ", "Bắc Trung Bộ", "Trung Bộ", "Nam Trung Bộ", "Nam Bộ"}
    actual_regions = {s.representative_region for s in stations}

    missing_regions = sorted(required_regions - actual_regions)
    if missing_regions:
        warnings_list.append(
            "Thiếu vùng đại diện trong cấu hình trạm: " + ", ".join(missing_regions)
        )

    station_ids = [s.station_id for s in stations]
    if len(station_ids) != len(set(station_ids)):
        warnings_list.append("Có station_id bị trùng trong TARGET_STATIONS.")

    if len(stations) < 5:
        warnings_list.append("Số trạm ít hơn 5, có thể chưa đủ đại diện không gian.")

    return warnings_list


def build_station_summary(
    station_logs: list[dict[str, Any]],
    years: list[int],
) -> pd.DataFrame:
    """Create station-level summary from fetch logs."""
    rows = []

    for log in station_logs:
        successful_years = sorted(log["successful_years"])
        failed_years = sorted(log["failed_years"])
        requested_year_count = len(years)
        success_year_count = len(successful_years)
        coverage_ratio = success_year_count / requested_year_count if requested_year_count else 0

        rows.append(
            {
                "station_id": log["station_id"],
                "station_name": log["station_name"],
                "representative_region": log["representative_region"],
                "rationale": log["rationale"],
                "requested_year_count": requested_year_count,
                "success_year_count": success_year_count,
                "missing_year_count": len(failed_years),
                "successful_years": ",".join(map(str, successful_years)),
                "missing_years": ",".join(map(str, failed_years)),
                "row_count": log["row_count"],
                "coverage_ratio": round(coverage_ratio, 4),
                "status": (
                    "PASS"
                    if coverage_ratio >= MIN_STATION_YEAR_COVERAGE and log["row_count"] > 0
                    else "WARNING"
                ),
            }
        )

    return pd.DataFrame(rows)


def build_quality_report(
    final_df: pd.DataFrame,
    station_summary: pd.DataFrame,
    config_warnings: list[str],
) -> dict[str, Any]:
    """Build final quality report for metadata."""
    total_requested = len(TARGET_STATIONS) * len(YEARS)
    total_success = int(station_summary["success_year_count"].sum())
    total_missing = int(station_summary["missing_year_count"].sum())
    total_coverage = total_success / total_requested if total_requested else 0

    quality_warnings = list(config_warnings)

    if total_coverage < MIN_TOTAL_YEAR_COVERAGE:
        quality_warnings.append(
            f"Độ bao phủ tổng thể thấp: {total_coverage:.2%} "
            f"< ngưỡng {MIN_TOTAL_YEAR_COVERAGE:.2%}."
        )

    weak_stations = station_summary[station_summary["status"] != "PASS"]
    for _, row in weak_stations.iterrows():
        quality_warnings.append(
            f"Trạm {row['station_name']} ({row['station_id']}) chưa đạt ngưỡng bao phủ: "
            f"{row['coverage_ratio']:.2%}; thiếu năm: {row['missing_years'] or 'không rõ'}."
        )

    date_summary = summarize_dataframe_dates(final_df)
    if date_summary["invalid_DATE_count"] not in (None, 0):
        quality_warnings.append(
            f"Có {date_summary['invalid_DATE_count']} dòng DATE không parse được."
        )

    if date_summary["duplicate_station_date_count"] not in (None, 0):
        quality_warnings.append(
            f"Có {date_summary['duplicate_station_date_count']} dòng trùng STATION-DATE trong Bronze."
        )

    return {
        "total_requested_station_year_files": total_requested,
        "total_success_station_year_files": total_success,
        "total_missing_station_year_files": total_missing,
        "total_coverage_ratio": round(total_coverage, 4),
        "date_summary": date_summary,
        "quality_warnings": quality_warnings,
        "quality_status": "PASS" if not quality_warnings else "WARNING",
    }


def save_json(data: dict[str, Any], path: Path) -> None:
    """Save dictionary as UTF-8 JSON."""
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# ============================================================================
# DATA CRAWLING
# ============================================================================

def fetch_station_data(
    station: StationConfig,
    years: list[int],
    session: requests.Session,
    crawled_at_utc: str,
) -> tuple[list[pd.DataFrame], dict[str, Any]]:
    """
    Fetch weather data for one station across multiple years.
    """
    print(
        f"\n📍 Đang xử lý trạm: {station.station_name} "
        f"({station.station_id}) - Đại diện: {station.representative_region}"
    )

    station_data: list[pd.DataFrame] = []
    successful_years: list[int] = []
    failed_years: list[int] = []
    year_logs: list[dict[str, Any]] = []
    row_count = 0

    for year in years:
        url = build_noaa_url(year, station.station_id)

        print(f"   - Năm {year}: {url}")
        response, error = request_with_retry(session, url)

        if response is not None and response.status_code == 200:
            try:
                df_year = read_station_year_csv(
                    response.text,
                    station=station,
                    year=year,
                    source_url=url,
                    crawled_at_utc=crawled_at_utc,
                )

                current_rows = len(df_year)
                if current_rows == 0:
                    failed_years.append(year)
                    year_logs.append(
                        {
                            "year": year,
                            "url": url,
                            "status": "empty_file",
                            "http_status": response.status_code,
                            "row_count": 0,
                            "error": "CSV file was downloaded but empty.",
                        }
                    )
                    print("      ⚠️ File tải được nhưng không có dòng dữ liệu.")
                else:
                    station_data.append(df_year)
                    successful_years.append(year)
                    row_count += current_rows
                    year_logs.append(
                        {
                            "year": year,
                            "url": url,
                            "status": "success",
                            "http_status": response.status_code,
                            "row_count": current_rows,
                            "error": None,
                        }
                    )
                    print(f"      ✅ Thành công: {current_rows:,} dòng")

            except Exception as exc:
                failed_years.append(year)
                year_logs.append(
                    {
                        "year": year,
                        "url": url,
                        "status": "parse_error",
                        "http_status": response.status_code,
                        "row_count": 0,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(f"      ❌ Lỗi parse CSV: {exc}")

        else:
            failed_years.append(year)
            http_status = None if response is None else response.status_code
            year_logs.append(
                {
                    "year": year,
                    "url": url,
                    "status": "failed",
                    "http_status": http_status,
                    "row_count": 0,
                    "error": error,
                }
            )
            print(f"      ❌ Không tải được. HTTP={http_status}, error={error}")

        _safe_sleep(REQUEST_DELAY)

    station_log = {
        "station_id": station.station_id,
        "station_name": station.station_name,
        "representative_region": station.representative_region,
        "rationale": station.rationale,
        "successful_years": successful_years,
        "failed_years": failed_years,
        "row_count": row_count,
        "year_logs": year_logs,
    }

    print(
        f"   -> Tóm tắt {station.station_name}: "
        f"{len(successful_years)}/{len(years)} năm, {row_count:,} dòng."
    )

    return station_data, station_log


def merge_and_save_bronze(
    all_data: list[pd.DataFrame],
    output_path: Path,
) -> pd.DataFrame:
    """
    Merge all downloaded raw files and save Bronze CSV.
    """
    print("\n--- 🛠️ GỘP DỮ LIỆU BRONZE/RAW ---")

    if not all_data:
        raise RuntimeError("Không có dữ liệu nào được tải về. Không thể tạo Bronze dataset.")

    final_df = pd.concat(all_data, ignore_index=True)
    final_df = sort_bronze_data(final_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_path, index=False)

    print(f"✅ Đã lưu Bronze dataset: {output_path}")
    print(f"   - Tổng số dòng: {len(final_df):,}")
    print(f"   - Tổng số cột: {len(final_df.columns):,}")

    return final_df


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def crawl_weather_data(
    output_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
    station_summary_path: str | Path | None = None,
    strict_quality: bool = True,
) -> dict[str, Any]:
    """
    Main crawling pipeline for NOAA GSOD Bronze data.

    Parameters
    ----------
    output_path:
        Path to save Bronze CSV.
    metadata_path:
        Path to save JSON metadata.
    station_summary_path:
        Path to save station-level summary CSV.
    strict_quality:
        Nếu True, pipeline sẽ raise RuntimeError khi độ bao phủ không đạt ngưỡng.
        Nếu False, vẫn lưu dữ liệu và metadata nhưng chỉ cảnh báo.
    """
    if output_path is None:
        output_path = RAW_DIR / OUTPUT_FILE
    else:
        output_path = Path(output_path)

    if metadata_path is None:
        metadata_path = RAW_DIR / METADATA_FILE
    else:
        metadata_path = Path(metadata_path)

    if station_summary_path is None:
        station_summary_path = RAW_DIR / STATION_SUMMARY_FILE
    else:
        station_summary_path = Path(station_summary_path)

    ensure_output_dirs(output_path, metadata_path, station_summary_path)

    crawl_started_at_utc = utc_now_iso()

    print("=== 🛰️ BẮT ĐẦU CRAWL DỮ LIỆU NOAA GSOD - BRONZE LAYER ===")
    print(f"- Layer: Bronze/raw")
    print(f"- Nguồn: NOAA NCEI GSOD")
    print(f"- Số trạm: {len(TARGET_STATIONS)}")
    print(f"- Giai đoạn: {min(YEARS)}-{max(YEARS)}")
    print(f"- Output data: {output_path}")
    print(f"- Output metadata: {metadata_path}")
    print(f"- Output station summary: {station_summary_path}")

    config_warnings = validate_station_configuration(TARGET_STATIONS)

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "DS108-rainfall-dataset-builder/1.0 "
                "(educational project; NOAA GSOD data collection)"
            )
        }
    )

    all_data: list[pd.DataFrame] = []
    station_logs: list[dict[str, Any]] = []

    for station in TARGET_STATIONS:
        station_data, station_log = fetch_station_data(
            station=station,
            years=YEARS,
            session=session,
            crawled_at_utc=crawl_started_at_utc,
        )
        all_data.extend(station_data)
        station_logs.append(station_log)

    final_df = merge_and_save_bronze(all_data, output_path)

    station_summary = build_station_summary(station_logs, YEARS)
    station_summary.to_csv(station_summary_path, index=False)

    quality_report = build_quality_report(
        final_df=final_df,
        station_summary=station_summary,
        config_warnings=config_warnings,
    )

    crawl_finished_at_utc = utc_now_iso()

    metadata = {
        "dataset_name": "Vietnam NOAA GSOD Bronze Weather Dataset",
        "layer": "bronze/raw",
        "description": (
            "Raw NOAA GSOD station observations collected automatically. "
            "This layer intentionally does not clean rogue values, convert units, "
            "impute missing values, or remove outliers. Only minimal reproducibility "
            "operations are applied: station ID casting, date sorting, and provenance columns."
        ),
        "source": {
            "provider": "NOAA National Centers for Environmental Information (NCEI)",
            "dataset": "Global Summary of the Day (GSOD)",
            "base_url": BASE_URL,
            "url_pattern": "{base_url}/{year}/{station_id}.csv",
        },
        "crawl_started_at_utc": crawl_started_at_utc,
        "crawl_finished_at_utc": crawl_finished_at_utc,
        "years_requested": YEARS,
        "stations": [asdict(station) for station in TARGET_STATIONS],
        "request_config": {
            "request_timeout_seconds": REQUEST_TIMEOUT,
            "request_delay_seconds": REQUEST_DELAY,
            "max_retries": MAX_RETRIES,
            "backoff_factor": BACKOFF_FACTOR,
            "retry_status_codes": sorted(RETRY_STATUS_CODES),
        },
        "outputs": {
            "bronze_csv": str(output_path),
            "metadata_json": str(metadata_path),
            "station_summary_csv": str(station_summary_path),
        },
        "station_logs": station_logs,
        "quality_report": quality_report,
        "methodological_notes": [
            "5 stations are selected to cover major Vietnam climate regions: Bắc Bộ, Bắc Trung Bộ, Trung Bộ, Nam Trung Bộ, Nam Bộ.",
            "This step is a data collection/bronze step, not a cleaning step.",
            "No manual editing is allowed; every transformation must be reproducible via code.",
            "DATE parsing is used only for validation and deterministic sorting.",
            "Physical cleaning, unit conversion, duplicate handling, imputation and feature engineering must be performed in later Silver/Gold layers.",
        ],
    }

    save_json(metadata, metadata_path)

    print("\n--- 📊 QUALITY REPORT ---")
    print(f"- Tổng độ bao phủ station-year: {quality_report['total_coverage_ratio']:.2%}")
    print(f"- Trạng thái: {quality_report['quality_status']}")

    if quality_report["quality_warnings"]:
        print("- Cảnh báo:")
        for warning in quality_report["quality_warnings"]:
            print(f"  + {warning}")
    else:
        print("- Không phát hiện cảnh báo chất lượng ở bước Bronze.")

    print("\n✅ HOÀN THÀNH BƯỚC 1 - BRONZE DATA COLLECTION")
    print(f"- Data: {output_path}")
    print(f"- Metadata: {metadata_path}")
    print(f"- Station summary: {station_summary_path}")

    if strict_quality and quality_report["quality_status"] != "PASS":
        raise RuntimeError(
            "Bronze data quality check chưa PASS. "
            "Hãy xem bronze_metadata.json và bronze_station_summary.csv để kiểm tra."
        )

    return metadata


if __name__ == "__main__":
    crawl_weather_data(strict_quality=True)
