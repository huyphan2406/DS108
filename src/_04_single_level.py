"""Step 4: Process ERA5 single-level GRIB files to daily parquet.

This script processes only instant single-level ERA5 variables.
ERA5 total precipitation is intentionally excluded because rainfall target
construction is handled from station data in the downstream pipeline.
"""

import gc
import xarray as xr
import pandas as pd
from pathlib import Path
from typing import List

# CONFIGURATION

VIETNAM_LAT_SLICE = (24, 8)          # (North, South)
VIETNAM_LON_SLICE = (102, 110)       # (West, East)
RESAMPLE_FREQUENCY = "1D"
GRAVITY = 9.80665

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
BASE_SINGLE_RAW = BASE_DIR / "data" / "raw" / "era5_single_level"
BASE_SINGLE_CLEAN = BASE_DIR / "data" / "processed" / "era5_single_level"
OUTPUT_SINGLE_FINAL = BASE_DIR / "data" / "processed" / "ERA5_single_level.parquet"

# Columns to drop after processing
COLS_TO_DROP = ["number", "step", "surface", "valid_time"]

# Temperature columns to convert from K to °C
TEMP_COLS_K_TO_C = ["d2m", "t2m", "sst"]

# Pressure columns to convert from Pa to hPa
PRESSURE_COLS_PA_TO_HPA = ["msl", "sp"]

# Aggregation columns for daily resampling
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

# 1. PROCESSING SINGLE LEVEL GRIB FILE

def _ensure_directories() -> None:
    BASE_SINGLE_CLEAN.mkdir(parents=True, exist_ok=True)
    OUTPUT_SINGLE_FINAL.parent.mkdir(parents=True, exist_ok=True)

def _drop_identifier_columns(df: pd.DataFrame, cols: List[str] = None) -> pd.DataFrame:
    if cols is None:
        cols = COLS_TO_DROP
    return df.drop(columns=cols, errors="ignore")

def _downsample_to_float32(df: pd.DataFrame) -> pd.DataFrame:
    floats = df.select_dtypes(include=["float64"]).columns
    df[floats] = df[floats].astype("float32")
    return df

def _convert_temperature_kelvin_to_celsius(df: pd.DataFrame, cols: List[str] = None) -> pd.DataFrame:
    if cols is None:
        cols = TEMP_COLS_K_TO_C

    for col in cols:
        if col in df.columns:
            df[col] = df[col] - 273.15

    return df

def _convert_pressure_pa_to_hpa(df: pd.DataFrame, cols: List[str] = None) -> pd.DataFrame:
    if cols is None:
        cols = PRESSURE_COLS_PA_TO_HPA

    for col in cols:
        if col in df.columns:
            df[col] = df[col] / 100

    return df

def _convert_geopotential_to_meters(df: pd.DataFrame) -> pd.DataFrame:
    if "z" in df.columns:
        df["z"] = df["z"] / GRAVITY
    return df

def process_single_level(year: str) -> pd.DataFrame:
    _ensure_directories()

    year = str(year)
    path = BASE_SINGLE_RAW / year / "data.grib"

    print(f"\n--- Đang xử lý Single Level năm {year} từ: {path}")

    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy GRIB file: {path}")

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

    df_daily = (
        df.set_index("time")
        .groupby(["latitude", "longitude"])
        .resample(RESAMPLE_FREQUENCY)
        .agg(available_agg_cols)
        .reset_index()
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

    del ds, ds_filter, df
    gc.collect()

    return df_daily

# MAIN PIPELINE

def process_single_levels() -> None:
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
