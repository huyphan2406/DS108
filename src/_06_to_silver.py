"""Step 6: Build Silver station-day dataset from GSOD and ERA5."""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# CONFIGURATION

BASE_DIR = Path(__file__).resolve().parent.parent

STATIONARY_COLS = ["DATE", "LATITUDE", "LONGITUDE", "ELEVATION"]
WEATHER_COLS = ["TEMP", "PRCP", "WDSP", "DEWP", "STP", "SLP", "VISIB"]
ALL_COLS = ["STATION"] + STATIONARY_COLS + WEATHER_COLS

ROGUE_MAPPING = {
    "PRCP": 99.99,
    "VISIB": 999.9,
    "WDSP": 999.9,
    "TEMP": 9999.9,
    "DEWP": 9999.9,
    "STP": 9999.9,
    "SLP": 9999.9,
}

UNIT_CONVERSIONS = {
    "TEMP": {"factor": 5 / 9, "offset": -32, "unit": "°C from °F"},
    "DEWP": {"factor": 5 / 9, "offset": -32, "unit": "°C from °F"},
    "PRCP": {"factor": 25.4, "offset": 0, "unit": "mm from inches"},
    "WDSP": {"factor": 0.514444, "offset": 0, "unit": "m/s from knots"},
    "VISIB": {"factor": 1.60934, "offset": 0, "unit": "km from miles"},
}

PRESSURE_MIN, PRESSURE_MAX = 800, 1100

DATA_START_DATE = "2015-01-01"
DATA_END_DATE = "2024-12-31"
GRID_RESOLUTION = 0.25

GSOD_RAW_PATH = BASE_DIR / "data" / "raw" / "gsod" / "bronze_data.csv"
GSOD_RAW_FALLBACK_PATH = BASE_DIR / "data" / "raw" / "bronze_data.csv"

ERA5_SINGLE_PATH = BASE_DIR / "data" / "clean" / "ERA5_single_level.parquet"
ERA5_PRESSURE_PATH = BASE_DIR / "data" / "clean" / "ERA5_pressure_final.parquet"

OUTPUT_SILVER_PATH = BASE_DIR / "data" / "clean" / "silver_data.csv"

ERA5_TO_GSOD_MAPPING = {
    "t2m": "TEMP",
    "tp": "PRCP",
    "d2m": "DEWP",
    "sp": "STP",
    "msl": "SLP",
}

# Chỉ dùng cho df_missing trước khi merge ERA5 lần 2.
# Drop các biến ERA5 feature ở đây để tránh sinh u10_era5, v10_era5, z_era5, lsm_era5.
ERA5_COLS_TO_DROP = ["sst", "u10", "v10", "z", "lsm"]

STATIONARY_FILL_COLS = ["STATION", "LATITUDE", "LONGITUDE", "ELEVATION"]

# Drop only ERA5 support columns after they are used for filling.
# Keep u10, v10, z, lsm because they are useful features later.
FINAL_DROPS = ["t2m", "d2m", "tp", "msl", "sp", "sst"]

# 1. GSOD CLEANING

def _resolve_path(primary_path: Path, fallback_path: Path | None = None) -> Path:
    if primary_path.exists():
        return primary_path

    if fallback_path is not None and fallback_path.exists():
        print(f"[WARNING] Dùng fallback path: {fallback_path}")
        return fallback_path

    raise FileNotFoundError(f"Không tìm thấy file: {primary_path}")

def _ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    for col in ALL_COLS:
        if col not in df.columns:
            df[col] = np.nan
    return df[ALL_COLS]

def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df["STATION"] = df["STATION"].astype(str)
    df = df.dropna(subset=["STATION", "DATE"])
    return df.drop_duplicates(subset=["STATION", "DATE"]).reset_index(drop=True)

def _remove_rogue_values(df: pd.DataFrame) -> pd.DataFrame:
    for col, error_val in ROGUE_MAPPING.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[np.isclose(df[col], error_val, atol=0.001), col] = np.nan
    return df

def _validate_pressure(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["STP", "SLP"]:
        if col in df.columns:
            df.loc[~df[col].between(PRESSURE_MIN, PRESSURE_MAX), col] = np.nan
    return df

def _convert_to_metric(df: pd.DataFrame) -> pd.DataFrame:
    for col, conversion in UNIT_CONVERSIONS.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = (df[col] + conversion["offset"]) * conversion["factor"]
    return df

def gsod(path: str | Path | None = None) -> pd.DataFrame:
    if path is None:
        path = _resolve_path(GSOD_RAW_PATH, GSOD_RAW_FALLBACK_PATH)
    else:
        path = Path(path)

    print("--- 📥 BƯỚC 1: Đọc, Gạn lọc & Loại bỏ dòng trùng lặp ---")

    df = pd.read_csv(path)
    df = _ensure_required_columns(df)
    df = _remove_duplicates(df)

    print("--- ⚖️ BƯỚC 2: Xóa rác, Sửa lỗi GSOD & Đổi đơn vị Metric ---")

    df = _remove_rogue_values(df)
    df = _validate_pressure(df)
    df = _convert_to_metric(df)

    return df

# 2. FILL MISSING USING ERA5 SINGLE-LEVEL

def _load_era5_single_level() -> pd.DataFrame:
    if not ERA5_SINGLE_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy ERA5 single-level file: {ERA5_SINGLE_PATH}")

    df_era5 = pd.read_parquet(ERA5_SINGLE_PATH)
    df_era5["time"] = pd.to_datetime(df_era5["time"], errors="coerce")

    if df_era5["time"].isna().any():
        raise ValueError("ERA5 single-level có giá trị time không parse được.")

    # Làm tròn để merge ổn định hơn với tọa độ dạng float.
    if "latitude" in df_era5.columns:
        df_era5["latitude"] = pd.to_numeric(df_era5["latitude"], errors="coerce").round(2)
    if "longitude" in df_era5.columns:
        df_era5["longitude"] = pd.to_numeric(df_era5["longitude"], errors="coerce").round(2)

    return df_era5

def _quantize_coordinates(df: pd.DataFrame, resolution: float = GRID_RESOLUTION) -> pd.DataFrame:
    df["latitude"] = ((pd.to_numeric(df["LATITUDE"], errors="coerce") / resolution).round() * resolution).round(2)
    df["longitude"] = ((pd.to_numeric(df["LONGITUDE"], errors="coerce") / resolution).round() * resolution).round(2)
    return df

def _find_missing_dates(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    full_dates = pd.date_range(start=start_date, end=end_date, freq="D")
    missing_records = []

    df_era5 = _load_era5_single_level()

    for (lat, lon), group in df.groupby(["latitude", "longitude"]):
        existing_dates = pd.to_datetime(group["time"], errors="coerce").dropna().drop_duplicates()
        missing_dates = full_dates.difference(existing_dates)

        df_era5_filtered = df_era5[
            (df_era5["latitude"] == lat)
            & (df_era5["longitude"] == lon)
            & (df_era5["time"].isin(missing_dates))
        ].copy()

        if not df_era5_filtered.empty:
            missing_records.append(df_era5_filtered)

        print(f"({lat}, {lon}): {len(missing_dates)} missing days")

    if missing_records:
        return pd.concat(missing_records, ignore_index=True)

    return pd.DataFrame(columns=df_era5.columns)

def _prepare_era5_data(df_era5: pd.DataFrame) -> pd.DataFrame:
    if df_era5.empty:
        return df_era5

    df_era5 = df_era5.rename(columns=ERA5_TO_GSOD_MAPPING)
    df_era5 = df_era5.drop(columns=ERA5_COLS_TO_DROP, errors="ignore")
    return df_era5

def _forward_fill_stationary(df: pd.DataFrame, cols=None) -> pd.DataFrame:
    if cols is None:
        cols = STATIONARY_FILL_COLS

    for col in cols:
        if col in df.columns:
            df[col] = df.groupby(["latitude", "longitude"])[col].transform(
                lambda x: x.ffill().bfill()
            )
    return df

def _interpolate_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    if "tp" in df.columns and "PRCP" in df.columns:
        df["PRCP"] = df["PRCP"].fillna(df["tp"])

    if "t2m" in df.columns and "TEMP" in df.columns:
        df["TEMP"] = df["TEMP"].fillna(df["t2m"])

    if "d2m" in df.columns and "DEWP" in df.columns:
        df["DEWP"] = df["DEWP"].fillna(df["d2m"])

    if "msl" in df.columns and "SLP" in df.columns:
        df["SLP"] = df["SLP"].fillna(df["msl"])

    if "sp" in df.columns and "STP" in df.columns:
        df["STP"] = df["STP"].fillna(df["sp"])

    if "VISIB" in df.columns:
        df["VISIB"] = df.groupby(["latitude", "longitude"])["VISIB"].transform(
            lambda x: x.interpolate(method="linear", limit_direction="both")
        )

    if {"WDSP", "u10", "v10"}.issubset(df.columns):
        u10 = pd.to_numeric(df["u10"], errors="coerce")
        v10 = pd.to_numeric(df["v10"], errors="coerce")
        era5_wind_speed = np.sqrt(u10 ** 2 + v10 ** 2)
        df["WDSP"] = pd.to_numeric(df["WDSP"], errors="coerce").fillna(era5_wind_speed)

    return df

def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    print("--- 📥 BƯỚC 3: Tìm dữ liệu bị thiếu và điền từ ERA5 ---")

    df = df.rename(columns={"DATE": "time"})
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])

    df = _quantize_coordinates(df)

    df_missing = _find_missing_dates(df, DATA_START_DATE, DATA_END_DATE)
    df_missing = _prepare_era5_data(df_missing)

    df = pd.concat([df, df_missing], axis=0, ignore_index=True)
    df = df.sort_values(by=["latitude", "longitude", "time"]).reset_index(drop=True)

    df_era5 = _load_era5_single_level()

    df = pd.merge(
        df,
        df_era5,
        how="left",
        on=["latitude", "longitude", "time"],
        suffixes=("", "_era5"),
    )

    print("--- ⚖️ BƯỚC 4: Điền giá trị và nội suy ---")

    df = _forward_fill_stationary(df)
    df = _interpolate_missing_values(df)

    # Nếu vẫn còn cột *_era5 do overlap ngoài ý muốn thì bỏ để tránh schema rối.
    era5_suffix_cols = [col for col in df.columns if col.endswith("_era5")]
    if era5_suffix_cols:
        df = df.drop(columns=era5_suffix_cols, errors="ignore")

    df = df.drop(columns=FINAL_DROPS, errors="ignore")

    return df

# 3. MERGE PRESSURE DATA

def _merge_pressure_data(df: pd.DataFrame) -> pd.DataFrame:
    if not ERA5_PRESSURE_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy ERA5 pressure-level file: {ERA5_PRESSURE_PATH}")

    pressure = pd.read_parquet(ERA5_PRESSURE_PATH)
    pressure["time"] = pd.to_datetime(pressure["time"], errors="coerce")

    if "latitude" in pressure.columns:
        pressure["latitude"] = pd.to_numeric(pressure["latitude"], errors="coerce").round(2)
    if "longitude" in pressure.columns:
        pressure["longitude"] = pd.to_numeric(pressure["longitude"], errors="coerce").round(2)

    df = pd.merge(
        df,
        pressure,
        how="left",
        on=["latitude", "longitude", "time"],
    )

    return df

def merge_file(df: pd.DataFrame) -> pd.DataFrame:
    print("--- 📥 BƯỚC 5: Gộp dữ liệu Áp suất ERA5 ---")

    df = _merge_pressure_data(df)

    print("--- ⚖️ BƯỚC 6: Làm sạch & Làm tròn giá trị ---")

    df = df.drop(columns=["latitude", "longitude"], errors="ignore")

    if "STATION" in df.columns:
        df["STATION"] = df["STATION"].astype(str)

    df = df.sort_values(["STATION", "time"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["STATION", "time"]).reset_index(drop=True)

    df = df.round(4)

    return df

# 4. FINAL PIPELINE

def silver_data_pipeline() -> None:
    df = gsod()
    df = fill_missing(df)
    df = merge_file(df)

    OUTPUT_SILVER_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_SILVER_PATH, index=False)

    print(f"✅ Data saved to {OUTPUT_SILVER_PATH}")
    print(f"Shape: {df.shape}")

    if {"STATION", "time"}.issubset(df.columns):
        print(f"Duplicate STATION-time: {df.duplicated(subset=['STATION', 'time']).sum()}")

if __name__ == "__main__":
    silver_data_pipeline()
