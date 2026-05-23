import xarray as xr
import pandas as pd
import gc
from tqdm.auto import tqdm
from typing import List
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================
PRESSURE_LEVELS = [500, 850]
VIETNAM_LAT_SLICE = (24, 8)       # (North, South)
VIETNAM_LON_SLICE = (102, 110)    # (West, East)
LEVEL_NAME = "isobaricInhPa"
RESAMPLE_FREQUENCY = "1D"
GRAVITY = 9.80665

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
BASE_PRESSURE_RAW = BASE_DIR / "data" / "raw" / "pressure"
BASE_PRESSURE_CLEAN = BASE_DIR / "data" / "clean" / "pressure"
OUTPUT_PRESSURE_FINAL = BASE_DIR / "data" / "clean" / "ERA5_pressure_final.parquet"

# Aggregation columns for resampling
RESAMPLE_AGG_COLS = {
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


# ============================================================================
# 1. PROCESSING SINGLE GRIB FILE
# ============================================================================

def _ensure_directories() -> None:
    """Create required output directories."""
    BASE_PRESSURE_CLEAN.mkdir(parents=True, exist_ok=True)
    OUTPUT_PRESSURE_FINAL.parent.mkdir(parents=True, exist_ok=True)


def _extract_pressure_levels(ds: xr.Dataset, level: int) -> xr.Dataset:
    """
    Extract a specific pressure level from dataset and rename variables.
    """
    ds_level = ds.sel({LEVEL_NAME: level})

    if LEVEL_NAME in ds_level.coords or LEVEL_NAME in ds_level.dims:
        ds_level = ds_level.drop_vars(LEVEL_NAME, errors="ignore")

    ds_level = ds_level.rename({v: f"{v}_{level}" for v in ds_level.data_vars})
    return ds_level


def _convert_temperature_kelvin_to_celsius(df: pd.DataFrame, temp_cols: List[str]) -> pd.DataFrame:
    """Convert temperature columns from Kelvin to Celsius."""
    for col in temp_cols:
        if col in df.columns:
            df[col] = df[col] - 273.15
    return df


def _convert_geopotential_to_meters(df: pd.DataFrame, z_cols: List[str]) -> pd.DataFrame:
    """Convert geopotential from m²/s² to geopotential height in meters."""
    for col in z_cols:
        if col in df.columns:
            df[col] = df[col] / GRAVITY
    return df


def _downsample_to_float32(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast float64 columns to float32 to save memory."""
    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].astype("float32")
    return df


def _get_common_merge_keys(df_left: pd.DataFrame, df_right: pd.DataFrame) -> List[str]:
    """Find common coordinate columns for merging pressure levels."""
    preferred_keys = ["time", "latitude", "longitude", "number", "step", "valid_time"]
    return [key for key in preferred_keys if key in df_left.columns and key in df_right.columns]


def merge_pressure_levels(grib_path: str | Path) -> pd.DataFrame:
    """
    Process pressure level GRIB file:
    1. Load GRIB file and filter Vietnam region.
    2. Extract 500 hPa and 850 hPa levels.
    3. Merge pressure levels horizontally.
    4. Convert units.
    5. Resample to daily frequency.
    """
    grib_path = Path(grib_path)

    if not grib_path.exists():
        raise FileNotFoundError(f"Không tìm thấy GRIB file: {grib_path}")

    print(f"\n--- Đang đọc và lọc tọa độ từ: {grib_path}")

    ds = xr.open_dataset(
        grib_path,
        engine="cfgrib",
        backend_kwargs={"errors": "ignore"},
    )

    if LEVEL_NAME not in ds.coords and LEVEL_NAME not in ds.dims:
        raise KeyError(f"Không tìm thấy pressure-level coordinate: {LEVEL_NAME}")

    available_levels = set(pd.Series(ds[LEVEL_NAME].values).astype(int).tolist())
    missing_levels = [lvl for lvl in PRESSURE_LEVELS if lvl not in available_levels]

    if missing_levels:
        raise ValueError(
            f"GRIB thiếu tầng áp suất {missing_levels}. "
            f"Các tầng hiện có: {sorted(available_levels)}"
        )

    # Filter to Vietnam region
    ds = ds.sel(
        latitude=slice(*VIETNAM_LAT_SLICE),
        longitude=slice(*VIETNAM_LON_SLICE),
    )

    # Extract pressure levels
    print("--- Trích xuất tầng 500 hPa và 850 hPa...")
    ds_500 = _extract_pressure_levels(ds, 500)
    ds_850 = _extract_pressure_levels(ds, 850)

    # Convert to DataFrames
    df_500 = ds_500.to_dataframe().reset_index()
    df_850 = ds_850.to_dataframe().reset_index()

    df_500 = _downsample_to_float32(df_500)
    df_850 = _downsample_to_float32(df_850)

    # Merge pressure levels horizontally
    print("--- Đang gộp dữ liệu tầng áp suất...")

    merge_keys = _get_common_merge_keys(df_500, df_850)

    if not {"time", "latitude", "longitude"}.issubset(set(merge_keys)):
        raise ValueError(f"Không đủ khóa merge cơ bản. Merge keys hiện có: {merge_keys}")

    df_final = pd.merge(
        df_500,
        df_850,
        on=merge_keys,
        how="inner",
    )

    # Convert units
    df_final = _convert_temperature_kelvin_to_celsius(df_final, ["t_500", "t_850"])
    df_final = _convert_geopotential_to_meters(df_final, ["z_500", "z_850"])

    # Resample to daily frequency
    df_final["time"] = pd.to_datetime(df_final["time"], errors="coerce")

    if df_final["time"].isna().any():
        raise ValueError(f"Cột time có giá trị không parse được trong file: {grib_path}")

    available_agg_cols = {
        col: agg for col, agg in RESAMPLE_AGG_COLS.items()
        if col in df_final.columns
    }

    if not available_agg_cols:
        raise ValueError(f"Không có biến pressure-level nào để resample trong file: {grib_path}")

    df_final = (
        df_final.set_index("time")
        .groupby(["latitude", "longitude"])
        .resample(RESAMPLE_FREQUENCY)
        .agg(available_agg_cols)
        .reset_index()
    )

    df_final = (
        df_final
        .sort_values(["time", "latitude", "longitude"])
        .reset_index(drop=True)
    )

    df_final = _downsample_to_float32(df_final)

    print(f"--- Hoàn thành merge_pressure. Kích thước bảng: {df_final.shape}")

    del ds, ds_500, ds_850, df_500, df_850
    gc.collect()

    return df_final


# ============================================================================
# 2. CONCATENATING MULTIPLE FILES
# ============================================================================

def concatenate_pressure_files(base_path: str | Path, folders: List[int]) -> pd.DataFrame:
    """
    Concatenate pressure level parquet files from multiple years.
    """
    print("\n--- Bắt đầu gộp các file Parquet tổng hợp...")

    base_path = Path(base_path)

    if not base_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file parquet base: {base_path}")

    all_chunks = [pd.read_parquet(base_path)]

    for folder in tqdm(folders, desc="Processing folders"):
        path = BASE_PRESSURE_CLEAN / f"era5_{folder}.parquet"

        if not path.exists():
            raise FileNotFoundError(f"Không tìm thấy file parquet: {path}")

        df_temp = pd.read_parquet(path)
        all_chunks.append(df_temp)

        del df_temp
        gc.collect()

    df_final = pd.concat(all_chunks, ignore_index=True)

    del all_chunks
    gc.collect()

    df_final["time"] = pd.to_datetime(df_final["time"], errors="coerce")
    df_final = (
        df_final
        .sort_values(["time", "latitude", "longitude"])
        .reset_index(drop=True)
    )

    OUTPUT_PRESSURE_FINAL.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_parquet(OUTPUT_PRESSURE_FINAL, index=False)

    print(f"\n[SUCCESS] Đã lưu file tổng: {OUTPUT_PRESSURE_FINAL}")
    print(f"--- Tổng số dòng: {len(df_final):,}")
    print(df_final.info())

    return df_final


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def process_pressure_levels() -> None:
    """
    Main pipeline for processing ERA5 pressure level data.

    Steps:
    1. Process individual yearly GRIB files from 2015 to 2024.
    2. Save each as intermediate parquet file.
    3. Concatenate all yearly files into final output.
    """
    _ensure_directories()

    print("\n=== BẮT ĐẦU XỬ LÝ PRESSURE LEVELS ===")

    # Process each year
    for period in tqdm(range(2015, 2025), desc="GRIB to Parquet"):
        grib_path = BASE_PRESSURE_RAW / str(period) / "data.grib"
        df = merge_pressure_levels(grib_path)

        output_path = BASE_PRESSURE_CLEAN / f"era5_{period}.parquet"
        df.to_parquet(output_path, index=False)

        print(f"-> Đã lưu file tạm: {output_path}")

        del df
        gc.collect()

    # Concatenate all yearly files
    base_path = BASE_PRESSURE_CLEAN / "era5_2015.parquet"
    folders = list(range(2016, 2025))

    concatenate_pressure_files(base_path, folders)


if __name__ == "__main__":
    process_pressure_levels()