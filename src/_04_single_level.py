"""
ERA5 single-level data processing module.

This module processes ERA5 single-level weather variables from GRIB format:
- Loads GRIB files containing surface meteorological variables
- Filters data to Vietnam geographic region
- Extracts and converts meteorological units (K→°C, Pa→hPa, etc.)
- Resamples hourly data to daily frequency with appropriate aggregations
- Concatenates multiple yearly files into final output
"""

import gc
import xarray as xr
import pandas as pd
from pathlib import Path
from typing import List

# ============================================================================
# CONFIGURATION
# ============================================================================

VIETNAM_LAT_SLICE = (24, 8)          # (North, South)
VIETNAM_LON_SLICE = (102, 110)       # (West, East)
RESAMPLE_FREQUENCY = "1D"
GRAVITY = 9.80665

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
BASE_SINGLE_RAW = BASE_DIR / "data" / "raw" / "single"
BASE_SINGLE_CLEAN = BASE_DIR / "data" / "clean" / "single_level"
OUTPUT_SINGLE_FINAL = BASE_DIR / "data" / "clean" / "ERA5_single_level.parquet"

# Columns to drop after processing
COLS_TO_DROP = ["number", "step", "surface", "valid_time"]

# Temperature columns to convert from K to °C
TEMP_COLS_K_TO_C = ["d2m", "t2m", "sst"]

# Pressure columns to convert from Pa to hPa
PRESSURE_COLS_PA_TO_HPA = ["msl", "sp"]

# Aggregation columns for resampling
# tp is processed separately because it has valid_time/step structure.
RESAMPLE_AGG_COLS = {
    "t2m": "mean",
    "d2m": "mean",
    "sst": "mean",
    "u10": "mean",
    "v10": "mean",
    "sp": "mean",
    "msl": "mean",
    "z": "mean",
    "lsm": "first",
}


# ============================================================================
# 1. PROCESSING SINGLE LEVEL GRIB FILE
# ============================================================================

def _ensure_directories() -> None:
    """Create required output directories."""
    BASE_SINGLE_CLEAN.mkdir(parents=True, exist_ok=True)
    OUTPUT_SINGLE_FINAL.parent.mkdir(parents=True, exist_ok=True)


def _drop_identifier_columns(df: pd.DataFrame, cols: List[str] = None) -> pd.DataFrame:
    """Remove GRIB identifier columns that are not needed for analysis."""
    if cols is None:
        cols = COLS_TO_DROP
    return df.drop(columns=cols, errors="ignore")


def _downsample_to_float32(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast float64 columns to float32 to save memory."""
    floats = df.select_dtypes(include=["float64"]).columns
    df[floats] = df[floats].astype("float32")
    return df


def _convert_temperature_kelvin_to_celsius(df: pd.DataFrame, cols: List[str] = None) -> pd.DataFrame:
    """Convert temperature columns from Kelvin to Celsius."""
    if cols is None:
        cols = TEMP_COLS_K_TO_C

    for col in cols:
        if col in df.columns:
            df[col] = df[col] - 273.15

    return df


def _convert_pressure_pa_to_hpa(df: pd.DataFrame, cols: List[str] = None) -> pd.DataFrame:
    """Convert pressure columns from Pa to hPa."""
    if cols is None:
        cols = PRESSURE_COLS_PA_TO_HPA

    for col in cols:
        if col in df.columns:
            df[col] = df[col] / 100

    return df


def _convert_geopotential_to_meters(df: pd.DataFrame) -> pd.DataFrame:
    """Convert geopotential from m²/s² to geopotential height in meters."""
    if "z" in df.columns:
        df["z"] = df["z"] / GRAVITY
    return df


def _convert_precipitation_m_to_mm(df: pd.DataFrame) -> pd.DataFrame:
    """Convert precipitation from meters to millimeters."""
    if "tp" in df.columns:
        df["tp"] = df["tp"] * 1000
    return df


def _extract_total_precipitation(path: str | Path) -> pd.DataFrame:
    """
    Extract total precipitation from GRIB file separately.

    ERA5 total precipitation often has valid_time/step metadata, so it is
    handled separately and aggregated to daily total precipitation.
    """
    path = Path(path)

    ds = xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {"shortName": "tp"},
            "errors": "ignore",
        },
    )

    ds_filter = ds.sel(
        latitude=slice(*VIETNAM_LAT_SLICE),
        longitude=slice(*VIETNAM_LON_SLICE),
    )

    tp = ds_filter.to_dataframe().reset_index()

    if "tp" not in tp.columns:
        raise ValueError(f"Không tìm thấy biến tp trong file: {path}")

    # valid_time is the real timestamp for precipitation.
    if "valid_time" in tp.columns:
        if "time" in tp.columns:
            tp = tp.rename(columns={"time": "source_time", "valid_time": "time"})
        else:
            tp = tp.rename(columns={"valid_time": "time"})

    tp["time"] = pd.to_datetime(tp["time"], errors="coerce")
    if tp["time"].isna().any():
        raise ValueError(f"Cột time của tp có giá trị không parse được trong file: {path}")

    tp = _convert_precipitation_m_to_mm(tp)

    # Collapse duplicate valid_time records before daily summation.
    tp = (
        tp.groupby(["time", "latitude", "longitude"], as_index=False)["tp"]
        .max()
        .sort_values(["time", "latitude", "longitude"])
        .reset_index(drop=True)
    )

    tp_daily = (
        tp.set_index("time")
        .groupby(["latitude", "longitude"])
        .resample(RESAMPLE_FREQUENCY)
        .agg({"tp": "sum"})
        .reset_index()
    )

    return tp_daily


def process_single_level(year: str) -> pd.DataFrame:
    """
    Process single level GRIB file:
    1. Load non-precipitation instant variables.
    2. Extract total precipitation separately.
    3. Convert units.
    4. Resample to daily frequency.
    5. Merge daily base variables with daily precipitation.
    """
    _ensure_directories()

    year = str(year)
    path = BASE_SINGLE_RAW / year / "data.grib"

    print(f"\n--- Đang xử lý Single Level năm {year} từ: {path}")

    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy GRIB file: {path}")

    # Load non-precipitation instant variables.
    ds = xr.open_dataset(
        path,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {"stepType": "instant"},
            "errors": "ignore",
        },
    )

    ds_filter = ds.sel(
        latitude=slice(*VIETNAM_LAT_SLICE),
        longitude=slice(*VIETNAM_LON_SLICE),
    )

    df = ds_filter.to_dataframe().reset_index()
    df = _drop_identifier_columns(df)
    df = _downsample_to_float32(df)

    # Convert units for base variables.
    df = _convert_temperature_kelvin_to_celsius(df)
    df = _convert_pressure_pa_to_hpa(df)
    df = _convert_geopotential_to_meters(df)

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    if df["time"].isna().any():
        raise ValueError(f"Cột time có giá trị không parse được trong file: {path}")

    # Use only columns that actually exist, because sst may be absent.
    available_agg_cols = {
        col: agg for col, agg in RESAMPLE_AGG_COLS.items()
        if col in df.columns
    }

    if not available_agg_cols:
        raise ValueError(f"Không có biến single-level nào để resample trong file: {path}")

    base_daily = (
        df.set_index("time")
        .groupby(["latitude", "longitude"])
        .resample(RESAMPLE_FREQUENCY)
        .agg(available_agg_cols)
        .reset_index()
    )

    # Extract and aggregate precipitation separately.
    tp_daily = _extract_total_precipitation(path)

    # Merge daily base variables and daily precipitation.
    df_daily = pd.merge(
        base_daily,
        tp_daily,
        on=["time", "latitude", "longitude"],
        how="outer",
    )

    df_daily = (
        df_daily
        .sort_values(["time", "latitude", "longitude"])
        .reset_index(drop=True)
    )

    df_daily = _downsample_to_float32(df_daily)

    output_path = BASE_SINGLE_CLEAN / f"single_{year}.parquet"
    df_daily.to_parquet(output_path, index=False)

    print(f"--- Đã lưu dữ liệu Single Level sạch tại: {output_path}")
    print(f"--- Kích thước: {df_daily.shape}")

    del ds, ds_filter, df, base_daily, tp_daily
    gc.collect()

    return df_daily


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def process_single_levels() -> None:
    """
    Main pipeline for processing ERA5 single-level data.

    Steps:
    1. Process each year from 2015 to 2024 separately.
    2. Save each as intermediate parquet file.
    3. Concatenate all years into final output.
    """
    _ensure_directories()

    print("\n=== BẮT ĐẦU XỬ LÝ SINGLE LEVEL ===")

    df = process_single_level("2015")

    for year in range(2016, 2025):
        df_i = process_single_level(str(year))

        print("--- Đang kết hợp các năm Single Level...")
        df = pd.concat([df, df_i], ignore_index=True)

        del df_i
        gc.collect()

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.sort_values(["time", "latitude", "longitude"]).reset_index(drop=True)

    df.to_parquet(OUTPUT_SINGLE_FINAL, index=False)

    print(f"[SUCCESS] Đã lưu file Single Level tổng hợp tại: {OUTPUT_SINGLE_FINAL}")
    print(f"[SUCCESS] Shape final: {df.shape}")

    del df
    gc.collect()


if __name__ == "__main__":
    process_single_levels()