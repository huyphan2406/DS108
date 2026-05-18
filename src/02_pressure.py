"""
ERA5 pressure-level processing module for the Bronze/Clean pipeline.

Purpose
-------
This script processes ERA5 pressure-level GRIB files into daily tabular Parquet
files for Vietnam. It extracts two meteorologically meaningful pressure levels:

- 850 hPa: lower troposphere, useful for low-level moisture and wind transport.
- 500 hPa: middle troposphere, useful for atmospheric instability and large-scale
  circulation related to rainfall formation.

Main upgrades compared with the earlier `_02_presure.py` version
-----------------------------------------------------------------
1. Uses the professional spelling `pressure` in paths and outputs.
2. Automatically creates output/report directories.
3. Supports a fallback from the old raw folder name `presure/` to `pressure/`.
4. Produces quality reports after processing:
   - number of rows
   - number of unique days
   - number of grid cells
   - duplicate count by (time, latitude, longitude)
   - missing rate per variable
   - physical sanity checks
5. Adds data assertions:
   - no duplicate records by (time, latitude, longitude)
   - z_500 must be greater than z_850
   - t_850 is expected to be greater than t_500 for most records
6. Saves metadata and quality reports for reproducibility.

Important note
--------------
This step is still a clean/intermediate pressure-level processing stage. It does
not impute missing values, does not remove meteorological outliers blindly, and
does not use target information. It only standardizes ERA5 pressure-level data
for later data integration and feature engineering.
"""

from __future__ import annotations

import gc
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIGURATION
# =============================================================================

# Pressure levels selected for rainfall-related atmospheric structure.
PRESSURE_LEVELS: Tuple[int, int] = (500, 850)

# Vietnam bounding box. ERA5 latitude is often stored from North to South, so the
# slicing helper below automatically handles both descending and ascending order.
VIETNAM_LAT_BOUNDS: Tuple[float, float] = (8.0, 24.0)     # South, North
VIETNAM_LON_BOUNDS: Tuple[float, float] = (102.0, 110.0)  # West, East

LEVEL_NAME = "isobaricInhPa"
RESAMPLE_FREQUENCY = "1D"
GRAVITY = 9.80665

# Project paths. Prefer the corrected folder name `pressure`.
BASE_DIR = Path(__file__).resolve().parent.parent if "__file__" in globals() else Path.cwd()
BASE_PRESSURE_RAW = BASE_DIR / "data" / "raw" / "pressure"
LEGACY_BASE_PRESSURE_RAW = BASE_DIR / "data" / "raw" / "presure"  # old misspelling fallback
BASE_PRESSURE_CLEAN = BASE_DIR / "data" / "clean" / "pressure"
REPORT_DIR = BASE_DIR / "reports" / "data_quality" / "pressure"

OUTPUT_PRESSURE_FINAL = BASE_DIR / "data" / "clean" / "ERA5_pressure_final.parquet"
OUTPUT_METADATA_JSON = REPORT_DIR / "pressure_metadata.json"
OUTPUT_QUALITY_SUMMARY_CSV = REPORT_DIR / "pressure_quality_summary.csv"
OUTPUT_MISSING_RATE_CSV = REPORT_DIR / "pressure_missing_rate_by_column.csv"

# Raw GRIB folder names. Keep this aligned with your local downloaded folders.
PERIODS: Tuple[str, ...] = ("15-17", "18-20", "21-23", "24-25")

# Aggregation rules when converting hourly/sub-daily ERA5 data to daily data.
# Mean is appropriate here because these variables represent atmospheric state.
RESAMPLE_AGG_COLS: Dict[str, str] = {
    "z_500": "mean",
    "t_500": "mean",
    "q_500": "mean",
    "u_500": "mean",
    "v_500": "mean",
    "w_500": "mean",
    "z_850": "mean",
    "t_850": "mean",
    "q_850": "mean",
    "u_850": "mean",
    "v_850": "mean",
    "w_850": "mean",
}

# Data assertion thresholds.
# t_850 is expected to be warmer than t_500 in most observations, but not every
# single row must obey this absolutely due to real atmospheric variability and
# possible numerical/metadata issues. A small violation rate is therefore allowed.
MAX_TEMPERATURE_STRUCTURE_VIOLATION_RATE = 0.05

# If True, severe quality problems raise an exception. If False, they are only
# reported as warnings in metadata/quality reports.
DEFAULT_STRICT_ASSERTIONS = True


# =============================================================================
# PATH AND UTILITY FUNCTIONS
# =============================================================================

def _ensure_directories() -> None:
    """Create required output and report directories if they do not exist."""
    BASE_PRESSURE_CLEAN.mkdir(parents=True, exist_ok=True)
    OUTPUT_PRESSURE_FINAL.parent.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def _resolve_raw_base_dir() -> Path:
    """
    Resolve the raw ERA5 pressure-level directory.

    The previous project used the misspelled folder name `presure`. This function
    prefers the corrected `pressure` path, but falls back to `presure` to keep the
    script runnable on your current project structure.
    """
    if BASE_PRESSURE_RAW.exists():
        return BASE_PRESSURE_RAW
    if LEGACY_BASE_PRESSURE_RAW.exists():
        print(
            "[WARNING] Đang dùng thư mục raw cũ bị sai chính tả: "
            f"{LEGACY_BASE_PRESSURE_RAW}. Nên đổi thành: {BASE_PRESSURE_RAW}"
        )
        return LEGACY_BASE_PRESSURE_RAW
    raise FileNotFoundError(
        "Không tìm thấy thư mục dữ liệu ERA5 pressure-level. "
        f"Đã thử: {BASE_PRESSURE_RAW} và {LEGACY_BASE_PRESSURE_RAW}"
    )


def _now_utc_iso() -> str:
    """Return current UTC time as ISO string for metadata provenance."""
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any) -> float | None:
    """Convert NumPy/pandas values to JSON-friendly floats."""
    if pd.isna(value):
        return None
    return float(value)


# =============================================================================
# GRIB PROCESSING FUNCTIONS
# =============================================================================

def _select_vietnam_region(ds: xr.Dataset) -> xr.Dataset:
    """
    Select Vietnam bounding box and handle both ascending/descending latitudes.
    """
    lat_min, lat_max = VIETNAM_LAT_BOUNDS
    lon_min, lon_max = VIETNAM_LON_BOUNDS

    lat_values = ds["latitude"].values
    is_descending = lat_values[0] > lat_values[-1]
    lat_slice = slice(lat_max, lat_min) if is_descending else slice(lat_min, lat_max)

    return ds.sel(latitude=lat_slice, longitude=slice(lon_min, lon_max))


def _assert_required_levels_exist(ds: xr.Dataset, levels: Iterable[int]) -> None:
    """Ensure required pressure levels are available in the GRIB file."""
    if LEVEL_NAME not in ds.coords and LEVEL_NAME not in ds.dims:
        raise KeyError(f"Không tìm thấy pressure-level coordinate: {LEVEL_NAME}")

    available_levels = set(pd.Series(ds[LEVEL_NAME].values).astype(int).tolist())
    missing_levels = [level for level in levels if level not in available_levels]
    if missing_levels:
        raise ValueError(
            f"GRIB thiếu các tầng áp suất bắt buộc {missing_levels}. "
            f"Các tầng hiện có: {sorted(available_levels)}"
        )


def _extract_pressure_level(ds: xr.Dataset, level: int) -> xr.Dataset:
    """
    Extract a specific pressure level and append the level suffix to variables.
    """
    ds_level = ds.sel({LEVEL_NAME: level})
    if LEVEL_NAME in ds_level.coords or LEVEL_NAME in ds_level.dims:
        ds_level = ds_level.drop_vars(LEVEL_NAME, errors="ignore")
    rename_map = {var: f"{var}_{level}" for var in ds_level.data_vars}
    return ds_level.rename(rename_map)


def _downcast_float64_to_float32(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast float64 columns to float32 to reduce memory usage."""
    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].astype("float32")
    return df


def _convert_temperature_kelvin_to_celsius(df: pd.DataFrame) -> pd.DataFrame:
    """Convert ERA5 temperature variables from Kelvin to Celsius."""
    for col in ["t_500", "t_850"]:
        if col in df.columns:
            df[col] = df[col] - 273.15
    return df


def _convert_geopotential_to_meters(df: pd.DataFrame) -> pd.DataFrame:
    """Convert ERA5 geopotential from m^2/s^2 to geopotential height in meters."""
    for col in ["z_500", "z_850"]:
        if col in df.columns:
            df[col] = df[col] / GRAVITY
    return df


def _get_common_merge_keys(df_left: pd.DataFrame, df_right: pd.DataFrame) -> List[str]:
    """Find common GRIB coordinate/index columns for horizontal merge."""
    preferred_keys = ["time", "latitude", "longitude", "number", "step", "valid_time"]
    return [key for key in preferred_keys if key in df_left.columns and key in df_right.columns]


def _resample_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pressure-level data to daily frequency by grid cell."""
    available_agg_cols = {col: agg for col, agg in RESAMPLE_AGG_COLS.items() if col in df.columns}
    if not available_agg_cols:
        raise ValueError("Không có biến pressure-level nào để resample.")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    if df["time"].isna().any():
        raise ValueError("Cột time có giá trị không parse được sang datetime.")

    daily = (
        df.set_index("time")
        .groupby(["latitude", "longitude"])
        .resample(RESAMPLE_FREQUENCY)
        .agg(available_agg_cols)
        .reset_index()
    )

    return daily.sort_values(["time", "latitude", "longitude"]).reset_index(drop=True)


def process_single_grib_file(grib_path: Path, period: str | None = None) -> pd.DataFrame:
    """
    Process one ERA5 pressure-level GRIB file into daily tabular data.

    Steps:
    1. Open GRIB with xarray/cfgrib.
    2. Filter to Vietnam bounding box.
    3. Extract 500 hPa and 850 hPa levels.
    4. Merge the two levels horizontally.
    5. Convert units: temperature K→°C, geopotential→m.
    6. Aggregate to daily frequency.
    7. Run quality assertions and return the processed dataframe.
    """
    if not grib_path.exists():
        raise FileNotFoundError(f"Không tìm thấy GRIB file: {grib_path}")

    label = f"period={period}" if period else grib_path.name
    print(f"\n--- Đang xử lý ERA5 pressure-level ({label}): {grib_path}")

    ds = xr.open_dataset(grib_path, engine="cfgrib")
    _assert_required_levels_exist(ds, PRESSURE_LEVELS)
    ds = _select_vietnam_region(ds)

    ds_500 = _extract_pressure_level(ds, 500)
    ds_850 = _extract_pressure_level(ds, 850)

    df_500 = _downcast_float64_to_float32(ds_500.to_dataframe().reset_index())
    df_850 = _downcast_float64_to_float32(ds_850.to_dataframe().reset_index())

    merge_keys = _get_common_merge_keys(df_500, df_850)
    if not {"time", "latitude", "longitude"}.issubset(set(merge_keys)):
        raise ValueError(f"Không đủ khóa merge cơ bản. Merge keys hiện có: {merge_keys}")

    df = pd.merge(df_500, df_850, on=merge_keys, how="inner")

    df = _convert_temperature_kelvin_to_celsius(df)
    df = _convert_geopotential_to_meters(df)
    df = _resample_to_daily(df)
    df = _downcast_float64_to_float32(df)

    if period is not None:
        df["source_period"] = period

    print(f"--- Hoàn thành xử lý {label}. Shape: {df.shape}")
    return df


# =============================================================================
# QUALITY REPORT AND DATA ASSERTIONS
# =============================================================================

def build_quality_report(df: pd.DataFrame, dataset_name: str) -> Dict[str, Any]:
    """
    Build a compact quality report for processed ERA5 pressure-level data.
    """
    required_key_cols = ["time", "latitude", "longitude"]
    missing_key_cols = [col for col in required_key_cols if col not in df.columns]
    if missing_key_cols:
        raise KeyError(f"Thiếu key columns bắt buộc: {missing_key_cols}")

    tmp = df.copy()
    tmp["time"] = pd.to_datetime(tmp["time"], errors="coerce")

    duplicate_count = int(tmp.duplicated(subset=required_key_cols).sum())
    n_days = int(tmp["time"].dt.date.nunique())
    n_grids = int(tmp[["latitude", "longitude"]].drop_duplicates().shape[0])

    z_violation_count = None
    z_violation_rate = None
    if {"z_500", "z_850"}.issubset(tmp.columns):
        valid_z = tmp[["z_500", "z_850"]].dropna()
        if len(valid_z) > 0:
            z_violation_count = int((valid_z["z_500"] <= valid_z["z_850"]).sum())
            z_violation_rate = z_violation_count / len(valid_z)

    temp_violation_count = None
    temp_violation_rate = None
    if {"t_500", "t_850"}.issubset(tmp.columns):
        valid_t = tmp[["t_500", "t_850"]].dropna()
        if len(valid_t) > 0:
            temp_violation_count = int((valid_t["t_850"] <= valid_t["t_500"]).sum())
            temp_violation_rate = temp_violation_count / len(valid_t)

    missing_rates = tmp.isna().mean().sort_values(ascending=False)

    report = {
        "dataset_name": dataset_name,
        "generated_at_utc": _now_utc_iso(),
        "n_rows": int(len(tmp)),
        "n_columns": int(tmp.shape[1]),
        "n_days": n_days,
        "start_date": str(tmp["time"].min().date()) if tmp["time"].notna().any() else None,
        "end_date": str(tmp["time"].max().date()) if tmp["time"].notna().any() else None,
        "n_grid_cells": n_grids,
        "duplicate_count_by_time_lat_lon": duplicate_count,
        "max_missing_rate": _safe_float(missing_rates.max()),
        "mean_missing_rate": _safe_float(missing_rates.mean()),
        "z_500_greater_than_z_850_violation_count": z_violation_count,
        "z_500_greater_than_z_850_violation_rate": _safe_float(z_violation_rate),
        "t_850_greater_than_t_500_violation_count": temp_violation_count,
        "t_850_greater_than_t_500_violation_rate": _safe_float(temp_violation_rate),
    }

    return report


def build_missing_rate_table(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    """Return missing rate table per column."""
    return (
        df.isna()
        .mean()
        .rename("missing_rate")
        .reset_index()
        .rename(columns={"index": "column"})
        .assign(dataset_name=dataset_name, missing_percent=lambda x: x["missing_rate"] * 100)
        [["dataset_name", "column", "missing_rate", "missing_percent"]]
        .sort_values(["missing_rate", "column"], ascending=[False, True])
        .reset_index(drop=True)
    )


def run_data_assertions(
    df: pd.DataFrame,
    dataset_name: str,
    strict: bool = DEFAULT_STRICT_ASSERTIONS,
) -> Dict[str, Any]:
    """
    Run sanity checks on processed pressure-level data.

    Assertions:
    - No duplicate rows by (time, latitude, longitude).
    - Geopotential height at 500 hPa must be above 850 hPa.
    - Temperature at 850 hPa should be warmer than at 500 hPa for most rows.

    Returns a dictionary of assertion results. Raises AssertionError when strict
    mode is enabled and a severe issue is found.
    """
    report = build_quality_report(df, dataset_name)
    errors: List[str] = []
    warnings_list: List[str] = []

    duplicate_count = report["duplicate_count_by_time_lat_lon"]
    if duplicate_count > 0:
        errors.append(
            f"Có {duplicate_count} dòng trùng theo (time, latitude, longitude)."
        )

    z_violation_count = report["z_500_greater_than_z_850_violation_count"]
    if z_violation_count is not None and z_violation_count > 0:
        errors.append(
            f"Có {z_violation_count} dòng vi phạm điều kiện vật lý z_500 > z_850."
        )

    temp_violation_rate = report["t_850_greater_than_t_500_violation_rate"]
    temp_violation_count = report["t_850_greater_than_t_500_violation_count"]
    if temp_violation_rate is not None and temp_violation_rate > MAX_TEMPERATURE_STRUCTURE_VIOLATION_RATE:
        warnings_list.append(
            "Tỷ lệ dòng có t_850 <= t_500 là "
            f"{temp_violation_rate:.2%} ({temp_violation_count} dòng), "
            "cao hơn ngưỡng kỳ vọng. Cần kiểm tra dữ liệu hoặc bối cảnh khí tượng."
        )

    max_missing_rate = report["max_missing_rate"]
    if max_missing_rate is not None and max_missing_rate > 0:
        warnings_list.append(
            f"Dataset còn missing values. Max missing rate = {max_missing_rate:.2%}."
        )

    assertion_result = {
        "dataset_name": dataset_name,
        "passed": len(errors) == 0,
        "strict": strict,
        "errors": errors,
        "warnings": warnings_list,
        "quality_report": report,
    }

    for message in warnings_list:
        print(f"[WARNING] {dataset_name}: {message}")

    if errors:
        message = f"[ASSERTION FAILED] {dataset_name}: " + " | ".join(errors)
        if strict:
            raise AssertionError(message)
        print(message)

    print(
        f"[QUALITY] {dataset_name}: rows={report['n_rows']:,}, "
        f"days={report['n_days']:,}, grids={report['n_grid_cells']:,}, "
        f"duplicates={report['duplicate_count_by_time_lat_lon']}, "
        f"max_missing={report['max_missing_rate']:.2%}"
        if report["max_missing_rate"] is not None
        else f"[QUALITY] {dataset_name}: quality report created."
    )

    return assertion_result


def save_quality_outputs(
    quality_reports: List[Dict[str, Any]],
    missing_rate_tables: List[pd.DataFrame],
    metadata: Dict[str, Any],
) -> None:
    """Save metadata, quality summary and missing-rate table."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    flat_reports = [item["quality_report"] for item in quality_reports]
    pd.DataFrame(flat_reports).to_csv(OUTPUT_QUALITY_SUMMARY_CSV, index=False)

    if missing_rate_tables:
        pd.concat(missing_rate_tables, ignore_index=True).to_csv(
            OUTPUT_MISSING_RATE_CSV,
            index=False,
        )

    metadata_to_save = {
        **metadata,
        "quality_assertions": quality_reports,
    }
    with open(OUTPUT_METADATA_JSON, "w", encoding="utf-8") as file:
        json.dump(metadata_to_save, file, indent=2, ensure_ascii=False, default=str)

    print(f"\n[SAVED] Quality summary: {OUTPUT_QUALITY_SUMMARY_CSV}")
    print(f"[SAVED] Missing-rate table: {OUTPUT_MISSING_RATE_CSV}")
    print(f"[SAVED] Metadata JSON: {OUTPUT_METADATA_JSON}")


# =============================================================================
# CONCATENATION AND MAIN PIPELINE
# =============================================================================

def concatenate_period_files(period_paths: List[Path]) -> pd.DataFrame:
    """Concatenate processed period-level parquet files into the final dataset."""
    print("\n--- Bắt đầu gộp các file pressure-level theo thời kỳ...")
    chunks: List[pd.DataFrame] = []

    for path in tqdm(period_paths, desc="Concatenating pressure parquet files"):
        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file parquet tạm: {path}")
        chunks.append(pd.read_parquet(path))

    df_final = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()

    df_final["time"] = pd.to_datetime(df_final["time"], errors="coerce")
    df_final = df_final.sort_values(["time", "latitude", "longitude"]).reset_index(drop=True)

    return df_final


def process_pressure_levels(strict_assertions: bool = DEFAULT_STRICT_ASSERTIONS) -> pd.DataFrame:
    """
    Main ERA5 pressure-level pipeline.

    Outputs
    -------
    - data/clean/pressure/era5_<period>.parquet
    - data/clean/ERA5_pressure_final.parquet
    - reports/data_quality/pressure/pressure_quality_summary.csv
    - reports/data_quality/pressure/pressure_missing_rate_by_column.csv
    - reports/data_quality/pressure/pressure_metadata.json
    """
    _ensure_directories()
    raw_base = _resolve_raw_base_dir()

    print("\n=== BẮT ĐẦU XỬ LÝ ERA5 PRESSURE-LEVEL DATA ===")
    print(f"Raw base directory:   {raw_base}")
    print(f"Clean output folder:  {BASE_PRESSURE_CLEAN}")
    print(f"Final output file:    {OUTPUT_PRESSURE_FINAL}")

    metadata: Dict[str, Any] = {
        "layer": "clean/intermediate pressure-level layer",
        "created_at_utc": _now_utc_iso(),
        "raw_base_dir": str(raw_base),
        "clean_output_dir": str(BASE_PRESSURE_CLEAN),
        "final_output_file": str(OUTPUT_PRESSURE_FINAL),
        "periods": list(PERIODS),
        "pressure_levels_hpa": list(PRESSURE_LEVELS),
        "geographic_bounds": {
            "latitude_south_north": list(VIETNAM_LAT_BOUNDS),
            "longitude_west_east": list(VIETNAM_LON_BOUNDS),
        },
        "resample_frequency": RESAMPLE_FREQUENCY,
        "aggregation_rules": RESAMPLE_AGG_COLS,
        "unit_conversions": {
            "temperature": "Kelvin to Celsius: t_C = t_K - 273.15",
            "geopotential": f"m^2/s^2 to geopotential height meters: z_m = z / {GRAVITY}",
        },
        "methodological_note": (
            "This stage extracts and standardizes ERA5 pressure-level atmospheric "
            "variables. It does not impute, scale, or use any target variable; "
            "therefore it does not introduce target leakage."
        ),
    }

    quality_results: List[Dict[str, Any]] = []
    missing_rate_tables: List[pd.DataFrame] = []
    period_output_paths: List[Path] = []

    for period in tqdm(PERIODS, desc="GRIB to pressure parquet"):
        grib_path = raw_base / period / "data.grib"
        df_period = process_single_grib_file(grib_path, period=period)

        dataset_name = f"pressure_{period}"
        quality_result = run_data_assertions(
            df_period,
            dataset_name=dataset_name,
            strict=strict_assertions,
        )
        quality_results.append(quality_result)
        missing_rate_tables.append(build_missing_rate_table(df_period, dataset_name))

        output_path = BASE_PRESSURE_CLEAN / f"era5_{period}.parquet"
        df_period.to_parquet(output_path, index=False)
        period_output_paths.append(output_path)
        print(f"[SAVED] Intermediate parquet: {output_path}")

        del df_period
        gc.collect()

    df_final = concatenate_period_files(period_output_paths)

    final_quality_result = run_data_assertions(
        df_final,
        dataset_name="pressure_final",
        strict=strict_assertions,
    )
    quality_results.append(final_quality_result)
    missing_rate_tables.append(build_missing_rate_table(df_final, "pressure_final"))

    df_final.to_parquet(OUTPUT_PRESSURE_FINAL, index=False)
    print(f"\n[SUCCESS] Đã lưu ERA5 pressure-level final: {OUTPUT_PRESSURE_FINAL}")
    print(f"Tổng số dòng: {len(df_final):,}")
    print(f"Tổng số cột: {df_final.shape[1]:,}")

    save_quality_outputs(quality_results, missing_rate_tables, metadata)

    return df_final


if __name__ == "__main__":
    # Keep strict_assertions=True for final submission so that serious data issues
    # are caught early. Use False only during debugging when you want the script to
    # continue and inspect quality reports manually.
    process_pressure_levels(strict_assertions=True)
