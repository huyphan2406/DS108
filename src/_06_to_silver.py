"""Step 6: Build Silver station-day dataset from GSOD and ERA5."""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# =============================================================================
# CONFIGURATION
# =============================================================================
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
GRID_RESOLUTION = 0.25

GSOD_RAW_PATH = BASE_DIR / "data" / "raw" / "gsod" / "bronze_data.csv"
GSOD_RAW_FALLBACK_PATH = BASE_DIR / "data" / "raw" / "bronze_data.csv"

ERA5_SINGLE_PATH = BASE_DIR / "data" / "processed" / "components" / "ERA5_single_level.parquet"
ERA5_PRESSURE_PATH = BASE_DIR / "data" / "processed" / "components" / "ERA5_pressure_final.parquet"

OUTPUT_SILVER_PATH = BASE_DIR / "data" / "processed" / "silver" / "silver_data.csv"

# ERA5 single-level variables used as covariates or support variables.
ERA5_SINGLE_FEATURE_COLS = [
    "time",
    "latitude",
    "longitude",
    "t2m",
    "d2m",
    "sst",
    "u10",
    "v10",
    "sp",
    "msl",
    "z",
    "lsm",
]

STATIONARY_FILL_COLS = ["LATITUDE", "LONGITUDE", "ELEVATION"]

# Drop ERA5 support columns after they are used for filling GSOD variables.
# Keep u10, v10, z, and lsm because they are useful predictors later.
FINAL_DROPS = ["t2m", "d2m", "msl", "sp", "sst"]


# =============================================================================
# 1. GSOD CLEANING
# =============================================================================
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


# =============================================================================
# 2. MERGE ERA5 SINGLE-LEVEL FEATURES AND FILL NON-TARGET GAPS
# =============================================================================
def _load_era5_single_level() -> pd.DataFrame:
    if not ERA5_SINGLE_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy ERA5 single-level file: {ERA5_SINGLE_PATH}")

    df_era5 = pd.read_parquet(ERA5_SINGLE_PATH)

    required_cols = {"time", "latitude", "longitude"}
    missing_cols = sorted(required_cols - set(df_era5.columns))
    if missing_cols:
        raise ValueError(f"ERA5 single-level thiếu các cột bắt buộc: {missing_cols}")

    available_cols = [col for col in ERA5_SINGLE_FEATURE_COLS if col in df_era5.columns]
    df_era5 = df_era5[available_cols].copy()

    df_era5["time"] = pd.to_datetime(df_era5["time"], errors="coerce")
    if df_era5["time"].isna().any():
        raise ValueError("ERA5 single-level có giá trị time không parse được.")

    df_era5["latitude"] = pd.to_numeric(df_era5["latitude"], errors="coerce").round(2)
    df_era5["longitude"] = pd.to_numeric(df_era5["longitude"], errors="coerce").round(2)
    df_era5 = df_era5.dropna(subset=["time", "latitude", "longitude"])
    df_era5 = df_era5.drop_duplicates(subset=["time", "latitude", "longitude"]).reset_index(drop=True)

    return df_era5


def _quantize_coordinates(df: pd.DataFrame, resolution: float = GRID_RESOLUTION) -> pd.DataFrame:
    df["latitude"] = ((pd.to_numeric(df["LATITUDE"], errors="coerce") / resolution).round() * resolution).round(2)
    df["longitude"] = ((pd.to_numeric(df["LONGITUDE"], errors="coerce") / resolution).round() * resolution).round(2)
    return df


def _station_group_keys(df: pd.DataFrame) -> list[str]:
    if "STATION" in df.columns and df["STATION"].notna().any():
        return ["STATION"]
    return ["latitude", "longitude"]


def _forward_fill_stationary(df: pd.DataFrame, cols: list[str] | None = None) -> pd.DataFrame:
    if cols is None:
        cols = STATIONARY_FILL_COLS

    group_keys = _station_group_keys(df)
    for col in cols:
        if col in df.columns:
            df[col] = df.groupby(group_keys)[col].transform(lambda x: x.ffill().bfill())
    return df

def _fill_visibility(df: pd.DataFrame) -> pd.DataFrame:
    if "VISIB" not in df.columns:
        return df

    group_keys = _station_group_keys(df)

    df = df.sort_values(group_keys + ["time"]).reset_index(drop=True)
    df["VISIB"] = pd.to_numeric(df["VISIB"], errors="coerce")
    df["_month"] = pd.to_datetime(df["time"], errors="coerce").dt.month

    # 1. Nội suy tuyến tính cho 1 khoảng thời gian ngắn(2 ngày)
    df["VISIB"] = df.groupby(group_keys)["VISIB"].transform(
        lambda x: x.interpolate(
            method="linear",
            limit=2,
            limit_area="inside"
        )
    )

    # 2. Station-month median fallback
    df["VISIB"] = df["VISIB"].fillna(
        df.groupby(group_keys + ["_month"])["VISIB"].transform("median")
    )

    # 3. Station median fallback
    df["VISIB"] = df["VISIB"].fillna(
        df.groupby(group_keys)["VISIB"].transform("median")
    )

    # 4. Global median fallback
    global_median = df["VISIB"].median()
    if pd.isna(global_median):
        raise ValueError("VISIB is entirely missing; cannot impute visibility values.")

    df["VISIB"] = df["VISIB"].fillna(global_median)

    return df.drop(columns=["_month"], errors="ignore")

def _interpolate_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    if "t2m" in df.columns and "TEMP" in df.columns:
        df["TEMP"] = df["TEMP"].fillna(df["t2m"])

    if "d2m" in df.columns and "DEWP" in df.columns:
        df["DEWP"] = df["DEWP"].fillna(df["d2m"])

    if "msl" in df.columns and "SLP" in df.columns:
        df["SLP"] = df["SLP"].fillna(df["msl"])

    if "sp" in df.columns and "STP" in df.columns:
        df["STP"] = df["STP"].fillna(df["sp"])

    # Riêng biến VISIB thì được nội suy tuyến tính do đặc trưng nào tương tự để thay vào
    df = _fill_visibility(df)

    # Còn biến WDSP thì được xử lý missing bằng công thức
    if {"WDSP", "u10", "v10"}.issubset(df.columns):
        u10 = pd.to_numeric(df["u10"], errors="coerce")
        v10 = pd.to_numeric(df["v10"], errors="coerce")
        era5_wind_speed = np.sqrt(u10 ** 2 + v10 ** 2)
        df["WDSP"] = pd.to_numeric(df["WDSP"], errors="coerce").fillna(era5_wind_speed)

    return df


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    print("--- 📥 BƯỚC 3: Gộp ERA5 single-level vào các ngày GSOD hiện có ---")

    df = df.rename(columns={"DATE": "time"})
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"])
    df = _quantize_coordinates(df)

    df_era5 = _load_era5_single_level()

    df = pd.merge(
        df,
        df_era5,
        how="left",
        on=["latitude", "longitude", "time"],
        suffixes=("", "_era5"),
    )

    print("--- ⚖️ BƯỚC 4: Điền giá trị thiếu cho biến khí tượng không phải target ---")

    df = _forward_fill_stationary(df)
    df = _interpolate_missing_values(df)

    era5_suffix_cols = [col for col in df.columns if col.endswith("_era5")]
    if era5_suffix_cols:
        df = df.drop(columns=era5_suffix_cols, errors="ignore")

    df = df.drop(columns=FINAL_DROPS, errors="ignore")

    return df


# =============================================================================
# 3. MERGE ERA5 PRESSURE-LEVEL DATA
# =============================================================================
def _merge_pressure_data(df: pd.DataFrame) -> pd.DataFrame:
    if not ERA5_PRESSURE_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy ERA5 pressure-level file: {ERA5_PRESSURE_PATH}")

    pressure = pd.read_parquet(ERA5_PRESSURE_PATH)

    required_cols = {"time", "latitude", "longitude"}
    missing_cols = sorted(required_cols - set(pressure.columns))
    if missing_cols:
        raise ValueError(f"ERA5 pressure-level thiếu các cột bắt buộc: {missing_cols}")

    pressure["time"] = pd.to_datetime(pressure["time"], errors="coerce")
    pressure["latitude"] = pd.to_numeric(pressure["latitude"], errors="coerce").round(2)
    pressure["longitude"] = pd.to_numeric(pressure["longitude"], errors="coerce").round(2)
    pressure = pressure.dropna(subset=["time", "latitude", "longitude"])
    pressure = pressure.drop_duplicates(subset=["time", "latitude", "longitude"]).reset_index(drop=True)

    df = pd.merge(
        df,
        pressure,
        how="left",
        on=["latitude", "longitude", "time"],
    )

    return df


def merge_file(df: pd.DataFrame) -> pd.DataFrame:
    print("--- 📥 BƯỚC 5: Gộp dữ liệu áp suất ERA5 ---")

    df = _merge_pressure_data(df)

    print("--- ⚖️ BƯỚC 6: Làm sạch & làm tròn giá trị ---")

    df = df.drop(columns=["latitude", "longitude"], errors="ignore")

    if "STATION" in df.columns:
        df["STATION"] = df["STATION"].astype(str)

    df = df.sort_values(["STATION", "time"]).reset_index(drop=True)
    df = df.drop_duplicates(subset=["STATION", "time"]).reset_index(drop=True)
    df = df.round(4)

    return df


# =============================================================================
# 4. FINAL PIPELINE
# =============================================================================
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
