"""Step 7: Create meteorological features and rainfall occurrence label."""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List

# CONFIGURATION

BASE_DIR = Path(__file__).resolve().parent.parent

PRESSURE_LEVELS = [500, 850]
WINDOW_SIZE = 3
RAIN_LABEL_THRESHOLD_MM = 0.1

TARGET_1STAGE = "target_1stage_prcp_mm"
TARGET_STAGE1 = "target_stage1_rain_label"
TARGET_STAGE2 = "target_stage2_prcp_log1p"

INPUT_PATH = BASE_DIR / "data" / "processed" / "silver" / "silver_data.csv"
OUTPUT_PATH = BASE_DIR / "data" / "features" / "feature_engineered_data.csv"

LAG_FEATURES = [
    "PRCP",
    "u_850",
    "v_850",
    "TEMP",
    "DEWP",
    "SLP",
    "VISIB",
    "dew_point_depression",
    "moisture_flux_850",
]

LAG_PERIODS = [1, 2]

DYNAMIC_FEATURES = [
    "moist_convection_850",
    "dew_point_depression",
    "moisture_flux_850",
    "TEMP",
    "DEWP",
    "SLP",
    "VISIB",
]

GROUPBY_COLS = ["STATION"]

PRCP_PROXY_COL = "PRCP_impute"

# UTILITY FUNCTIONS

def _get_groupby_cols(df: pd.DataFrame) -> List[str]:
    if "STATION" in df.columns:
        return ["STATION"]

    if {"LATITUDE", "LONGITUDE"}.issubset(df.columns):
        return ["LATITUDE", "LONGITUDE"]

    raise KeyError("Không tìm thấy STATION hoặc LATITUDE/LONGITUDE để tạo lag/rolling features.")

def _safe_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def _build_prcp_past_fill(df: pd.DataFrame, group_cols: List[str]) -> pd.Series:
    """
    Tạo past_fill cho PRCP_impute nhưng không dùng tương lai.

    Thứ tự ưu tiên:
    1. ffill tối đa 2 ngày theo từng trạm
    2. median lịch sử theo trạm + tháng
    3. median lịch sử theo trạm
    4. nếu vẫn thiếu thì dùng 0.0

    Lưu ý:
    - Chỉ dùng để tạo PRCP_impute.
    - Không thay thế PRCP thật.
    - Không dùng bfill để tránh leakage từ tương lai.
    """
    df = df.copy()
    df["PRCP"] = pd.to_numeric(df["PRCP"], errors="coerce")
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["_month"] = df["time"].dt.month

    # 1. Fill ngắn hạn bằng giá trị quá khứ gần nhất, tối đa 2 ngày
    past_fill = df.groupby(group_cols)["PRCP"].ffill(limit=2)

    # 2. Median lịch sử theo trạm + tháng, chỉ dùng dữ liệu trước dòng hiện tại
    station_month_median = (
        df.groupby(group_cols + ["_month"])["PRCP"]
        .transform(lambda x: x.shift(1).expanding(min_periods=1).median())
    )

    past_fill = past_fill.fillna(station_month_median)

    # 3. Median lịch sử theo trạm, chỉ dùng dữ liệu trước dòng hiện tại
    station_median = (
        df.groupby(group_cols)["PRCP"]
        .transform(lambda x: x.shift(1).expanding(min_periods=1).median())
    )

    past_fill = past_fill.fillna(station_median)

    # 4. Fallback cuối cùng: 0.0
    past_fill = past_fill.fillna(0.0)

    return past_fill

def create_prcp_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    PRCP gốc là target thật, không fill.
    PRCP_impute chỉ có giá trị tại các dòng PRCP gốc bị missing.
    PRCP_impute chỉ dùng để tạo PRCP_lag và PRCP_past rolling.
    """
    if "PRCP" not in df.columns:
        return df

    group_cols = _get_groupby_cols(df)
    df = df.sort_values(group_cols + ["time"]).reset_index(drop=True)

    df["PRCP"] = pd.to_numeric(df["PRCP"], errors="coerce")

    past_fill = _build_prcp_past_fill(df, group_cols)

    df[PRCP_PROXY_COL] = np.where(
        df["PRCP"].isna(),
        past_fill,
        np.nan
    )

    return df

# 1. TEMPORAL FEATURES

def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).reset_index(drop=True)

    day_of_year = df["time"].dt.dayofyear
    month = df["time"].dt.month

    df["day_sin"] = np.sin(2 * np.pi * day_of_year / 365)
    df["day_cos"] = np.cos(2 * np.pi * day_of_year / 365)

    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    return df

# 2. THERMODYNAMIC FEATURES

def create_thermodynamic_features(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = _get_groupby_cols(df)
    df = df.sort_values(group_cols + ["time"]).reset_index(drop=True)

    # Moist convection at multiple era5_pressure_level levels
    for lvl in PRESSURE_LEVELS:
        q_col = f"q_{lvl}"
        w_col = f"w_{lvl}"

        if q_col in df.columns and w_col in df.columns:
            df[f"moist_convection_{lvl}"] = df[q_col] * df[w_col]

    # Lapse rate: temperature decrease with height
    required_cols = {"t_850", "t_500", "z_500", "z_850"}
    if required_cols.issubset(df.columns):
        denominator = df["z_500"] - df["z_850"]
        df["lapse_rate_850_500"] = np.where(
            denominator != 0,
            (df["t_850"] - df["t_500"]) / denominator,
            np.nan,
        )

    # Dew point depression: air dryness measure
    if {"TEMP", "DEWP"}.issubset(df.columns):
        df["dew_point_depression"] = df["TEMP"] - df["DEWP"]

    return df

# 3. WIND & DYNAMIC FEATURES

def create_wind_features(df: pd.DataFrame) -> pd.DataFrame:
    group_cols = _get_groupby_cols(df)
    df = df.sort_values(group_cols + ["time"]).reset_index(drop=True)

    # Wind shear: vertical wind difference
    if {"u_850", "u_500"}.issubset(df.columns):
        df["du_850_500"] = df["u_850"] - df["u_500"]

    if {"v_850", "v_500"}.issubset(df.columns):
        df["dv_850_500"] = df["v_850"] - df["v_500"]

    if {"du_850_500", "dv_850_500"}.issubset(df.columns):
        df["wind_shear_mag"] = np.sqrt(df["du_850_500"] ** 2 + df["dv_850_500"] ** 2)

    # Ageostrophic signal: time derivative of wind field
    if "u_850" in df.columns:
        df["du_dt_850"] = df.groupby(group_cols)["u_850"].diff()

    if "v_850" in df.columns:
        df["dv_dt_850"] = df.groupby(group_cols)["v_850"].diff()

    if {"du_dt_850", "dv_dt_850"}.issubset(df.columns):
        df["ageostrophic_signal"] = np.sqrt(df["du_dt_850"] ** 2 + df["dv_dt_850"] ** 2)

    return df

# 4. GEOMETRIC FEATURES

def create_geometric_features(df: pd.DataFrame) -> pd.DataFrame:
    if {"z_500", "z_850"}.issubset(df.columns):
        df["thickness_500_850"] = df["z_500"] - df["z_850"]

    if {"ELEVATION", "z"}.issubset(df.columns):
        df["elevation_diff"] = df["ELEVATION"] - df["z"]

    return df

# 5. FLUX FEATURES

def create_flux_features(df: pd.DataFrame) -> pd.DataFrame:
    if {"q_850", "u_850", "v_850"}.issubset(df.columns):
        df["moisture_flux_850"] = df["q_850"] * np.sqrt(df["u_850"] ** 2 + df["v_850"] ** 2)

    return df

# 6. TEMPORAL DYNAMICS FEATURES (Lag & Rolling Statistics)

def create_lag_features(
    df: pd.DataFrame,
    features: List[str] = None,
    lags: List[int] = None,
) -> pd.DataFrame:
    if features is None:
        features = LAG_FEATURES
    if lags is None:
        lags = LAG_PERIODS

    group_cols = _get_groupby_cols(df)
    df = df.sort_values(group_cols + ["time"]).reset_index(drop=True)

    if "PRCP" in df.columns:
        if PRCP_PROXY_COL in df.columns:
            prcp_source = df["PRCP"].combine_first(df[PRCP_PROXY_COL])
        else:
            prcp_source = df["PRCP"]
    else:
        prcp_source = None

    for feature in features:
        if feature == "PRCP":
            if prcp_source is None:
                continue

            for lag in lags:
                df[f"{feature}_lag_{lag}"] = (
                    prcp_source
                    .groupby([df[col] for col in group_cols])
                    .shift(lag)
                )

        else:
            if feature not in df.columns:
                continue

            for lag in lags:
                df[f"{feature}_lag_{lag}"] = (
                    df.groupby(group_cols)[feature]
                    .shift(lag)
                )

    return df

def create_rolling_statistics(
    df: pd.DataFrame,
    features: List[str] = None,
    window: int = None,
) -> pd.DataFrame:
    if features is None:
        features = DYNAMIC_FEATURES
    if window is None:
        window = WINDOW_SIZE

    group_cols = _get_groupby_cols(df)
    df = df.sort_values(group_cols + ["time"]).reset_index(drop=True)

    for col in features:
        if col == "PRCP":
            continue

        if col in df.columns:
            df[f"{col}_{window}d_mean"] = (
                df.groupby(group_cols)[col]
                .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).mean())
            )

    if "moisture_flux_850" in df.columns:
        df[f"moisture_flux_{window}d_sum"] = (
            df.groupby(group_cols)["moisture_flux_850"]
            .transform(lambda x: x.shift(1).rolling(window=window, min_periods=1).sum())
        )

    if "PRCP" in df.columns:
        if PRCP_PROXY_COL in df.columns:
            prcp_source = df["PRCP"].combine_first(df[PRCP_PROXY_COL])
        else:
            prcp_source = df["PRCP"]

        prcp_grouped = prcp_source.groupby([df[col] for col in group_cols])

        df[f"PRCP_past_{window}d_mean"] = prcp_grouped.transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )

        df[f"PRCP_past_{window}d_sum"] = prcp_grouped.transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).sum()
        )

    return df

# 7. TARGET ENCODING

def prepare_target(df: pd.DataFrame) -> pd.DataFrame:
    df["PRCP"] = pd.to_numeric(df["PRCP"], errors="coerce")

    df[TARGET_1STAGE] = df["PRCP"]

    df[TARGET_STAGE1] = np.where(
        df["PRCP"].notna(),
        (df["PRCP"] >= RAIN_LABEL_THRESHOLD_MM).astype("int8"),
        np.nan,
    )

    df[TARGET_STAGE2] = np.where(
        df["PRCP"].notna(),
        np.log1p(df["PRCP"].astype(float)),
        np.nan,
    )

    return df

# MAIN PIPELINE

def feature_engineering(
    input_path: str | Path = None,
    output_path: str | Path = None,
) -> None:
    if input_path is None:
        input_path = INPUT_PATH

    if output_path is None:
        output_path = OUTPUT_PATH

    input_path = Path(input_path)
    output_path = Path(output_path)

    print("🔄 Loading data...")

    if not input_path.exists():
        raise FileNotFoundError(f"Không tìm thấy Silver data: {input_path}")

    df = pd.read_csv(input_path)

    if "STATION" in df.columns:
        df["STATION"] = df["STATION"].astype(str)

    numeric_candidates = [
        "TEMP", "DEWP", "SLP", "STP", "VISIB", "WDSP", "PRCP",
        "LATITUDE", "LONGITUDE", "ELEVATION",
        "z", "lsm",
        "z_500", "z_850",
        "t_500", "t_850",
        "q_500", "q_850",
        "u_500", "u_850",
        "v_500", "v_850",
        "w_500", "w_850",
    ]

    df = _safe_numeric(df, numeric_candidates)

    print("⏰ Creating temporal features...")
    df = create_temporal_features(df)

    print("🌡️  Creating thermodynamic features...")
    df = create_thermodynamic_features(df)

    print("💨 Creating wind & dynamic features...")
    df = create_wind_features(df)

    print("📏 Creating geometric features...")
    df = create_geometric_features(df)

    print("🌊 Creating flux features...")
    df = create_flux_features(df)

    print("🌧️ Creating PRCP proxy for lag/rolling only...")
    df = create_prcp_proxy(df)

    print("⏳ Creating temporal dynamics (lag & rolling)...")
    df = create_lag_features(df)
    df = create_rolling_statistics(df)

    print("🎯 Preparing rainfall targets...")
    df = prepare_target(df)

    sort_cols = ["STATION", "time"] if "STATION" in df.columns else ["time"]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"✅ Saving {df.shape[1]} columns to {output_path}...")
    df.to_csv(output_path, index=False)

    print(f"✨ Feature engineering complete! Shape: {df.shape}")
    print("Target columns created: target_1stage_prcp_mm, target_stage1_rain_label, target_stage2_prcp_log1p")
if __name__ == "__main__":
    feature_engineering()
