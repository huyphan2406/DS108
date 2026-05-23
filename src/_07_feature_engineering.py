"""
Feature engineering module for rainfall prediction.

This module creates advanced meteorological features from Silver weather data:
- Temporal features: cyclical day/month encoding for seasonal patterns
- Thermodynamic features: moist convection, lapse rate, dew point depression
- Wind/dynamic features: wind shear magnitude, ageostrophic signal
- Geometric features: atmospheric layer thickness, elevation relationships
- Flux features: moisture transport
- Temporal dynamics: lagged features and rolling statistics
- Targets for two-stage rainfall modeling:
  Stage 1: PRCP_label = rain/no-rain classification target
  Stage 2: PRCP = rainfall amount in mm for regression target
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent

PRESSURE_LEVELS = [500, 850]
WINDOW_SIZE = 3
PRECIPITATION_THRESHOLD = 0.1

INPUT_PATH = BASE_DIR / "data" / "clean" / "silver_data.csv"
OUTPUT_PATH = BASE_DIR / "data" / "feature_engineering" / "feature_engineered_data.csv"

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


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _get_groupby_cols(df: pd.DataFrame) -> List[str]:
    """
    Use STATION for temporal features if available.
    Fallback to LATITUDE/LONGITUDE if STATION does not exist.
    """
    if "STATION" in df.columns:
        return ["STATION"]

    if {"LATITUDE", "LONGITUDE"}.issubset(df.columns):
        return ["LATITUDE", "LONGITUDE"]

    raise KeyError("Không tìm thấy STATION hoặc LATITUDE/LONGITUDE để tạo lag/rolling features.")


def _safe_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Convert selected columns to numeric if they exist."""
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _drop_temporal_nans(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with NaN created by lag/diff features.
    These NaNs usually occur at the first few days of each station.
    """
    temporal_cols = [
        col for col in df.columns
        if "_lag_" in col
        or "_past_" in col
        or col in ["du_dt_850", "dv_dt_850", "ageostrophic_signal"]
    ]

    temporal_cols = [col for col in temporal_cols if col in df.columns]

    if temporal_cols:
        before = len(df)
        df = df.dropna(subset=temporal_cols).reset_index(drop=True)
        after = len(df)
        print(f"🧹 Dropped {before - after} rows with temporal NaNs from lag/diff features.")

    return df


# ============================================================================
# 1. TEMPORAL FEATURES
# ============================================================================

def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create cyclical time-based features to capture seasonal patterns.
    Uses sine/cosine encoding for day-of-year and month.
    """
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).reset_index(drop=True)

    day_of_year = df["time"].dt.dayofyear
    month = df["time"].dt.month

    df["day_sin"] = np.sin(2 * np.pi * day_of_year / 365)
    df["day_cos"] = np.cos(2 * np.pi * day_of_year / 365)

    df["month_sin"] = np.sin(2 * np.pi * month / 12)
    df["month_cos"] = np.cos(2 * np.pi * month / 12)

    return df


# ============================================================================
# 2. THERMODYNAMIC FEATURES
# ============================================================================

def create_thermodynamic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features representing atmospheric thermodynamic properties and processes.
    Includes moist convection, lapse rate, and moisture-temperature relationship.
    """
    group_cols = _get_groupby_cols(df)
    df = df.sort_values(group_cols + ["time"]).reset_index(drop=True)

    # Moist convection at multiple pressure levels
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


# ============================================================================
# 3. WIND & DYNAMIC FEATURES
# ============================================================================

def create_wind_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features representing atmospheric wind shear and ageostrophic motion.
    These indicate atmospheric instability and imbalance.
    """
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


# ============================================================================
# 4. GEOMETRIC FEATURES
# ============================================================================

def create_geometric_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features representing vertical atmospheric structure and elevation.
    Includes geopotential thickness and surface-upper level relationships.
    """
    if {"z_500", "z_850"}.issubset(df.columns):
        df["thickness_500_850"] = df["z_500"] - df["z_850"]

    if {"ELEVATION", "z"}.issubset(df.columns):
        df["elevation_diff"] = df["ELEVATION"] - df["z"]

    return df


# ============================================================================
# 5. FLUX FEATURES
# ============================================================================

def create_flux_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create features representing atmospheric moisture transport.
    Moisture flux indicates water vapor movement potential.
    """
    if {"q_850", "u_850", "v_850"}.issubset(df.columns):
        df["moisture_flux_850"] = df["q_850"] * np.sqrt(df["u_850"] ** 2 + df["v_850"] ** 2)

    return df


# ============================================================================
# 6. TEMPORAL DYNAMICS FEATURES (Lag & Rolling Statistics)
# ============================================================================

def create_lag_features(
    df: pd.DataFrame,
    features: List[str] = None,
    lags: List[int] = None,
) -> pd.DataFrame:
    """
    Create lagged features to capture temporal dependencies.

    Important:
    - PRCP_lag_1 / PRCP_lag_2 are allowed because they use past rainfall.
    - Current-day PRCP is kept only as target, not as model input.
    """
    if features is None:
        features = LAG_FEATURES
    if lags is None:
        lags = LAG_PERIODS

    group_cols = _get_groupby_cols(df)
    df = df.sort_values(group_cols + ["time"]).reset_index(drop=True)

    for feature in features:
        if feature not in df.columns:
            continue

        for lag in lags:
            df[f"{feature}_lag_{lag}"] = df.groupby(group_cols)[feature].shift(lag)

    return df


def create_rolling_statistics(
    df: pd.DataFrame,
    features: List[str] = None,
    window: int = None,
) -> pd.DataFrame:
    """
    Create rolling window statistics to capture multi-day patterns.

    Important:
    - Rolling statistics for PRCP use past-only values via shift(1),
      so they do not leak current-day rainfall target.
    """
    if features is None:
        features = DYNAMIC_FEATURES
    if window is None:
        window = WINDOW_SIZE

    group_cols = _get_groupby_cols(df)
    df = df.sort_values(group_cols + ["time"]).reset_index(drop=True)

    # Rolling mean for feature persistence
    for col in features:
        if col in df.columns:
            df[f"{col}_{window}d_mean"] = df.groupby(group_cols)[col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )

    # Rolling sum for moisture flux accumulation
    if "moisture_flux_850" in df.columns:
        df[f"moisture_flux_{window}d_sum"] = df.groupby(group_cols)["moisture_flux_850"].transform(
            lambda x: x.rolling(window=window, min_periods=1).sum()
        )

    # Past-only rolling rainfall history
    if "PRCP" in df.columns:
        df[f"PRCP_past_{window}d_mean"] = df.groupby(group_cols)["PRCP"].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )

        df[f"PRCP_past_{window}d_sum"] = df.groupby(group_cols)["PRCP"].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).sum()
        )

    return df


# ============================================================================
# 7. TARGET ENCODING
# ============================================================================

def prepare_target(df: pd.DataFrame, threshold: float = None) -> pd.DataFrame:
    """
    Prepare targets for two-stage rainfall modeling.

    Stage 1:
    - PRCP_label: binary rain/no-rain target.
      PRCP_label = 1 if PRCP > threshold, otherwise 0.

    Stage 2:
    - PRCP: rainfall amount in mm, used as regression target.
    - PRCP_log1p: log1p(PRCP), optional regression target for skewed rainfall.

    Notes:
    - PRCP is kept in the final file because it is the regression target.
    - During model training, PRCP / PRCP_label / PRCP_log1p must be dropped from X.
    """
    if threshold is None:
        threshold = PRECIPITATION_THRESHOLD

    if "PRCP" not in df.columns:
        raise KeyError("Không tìm thấy cột PRCP để tạo target.")

    df["PRCP"] = pd.to_numeric(df["PRCP"], errors="coerce").fillna(0)
    df["PRCP"] = df["PRCP"].clip(lower=0)

    df["PRCP_label"] = (df["PRCP"] > threshold).astype(int)
    df["PRCP_log1p"] = np.log1p(df["PRCP"])

    return df


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def feature_engineering(
    input_path: str | Path = None,
    output_path: str | Path = None,
) -> None:
    """
    Main feature engineering pipeline.

    Steps:
    1. Load Silver data
    2. Create temporal features
    3. Create thermodynamic features
    4. Create wind/dynamic features
    5. Create geometric features
    6. Create flux features
    7. Create lagged and rolling features
    8. Prepare two-stage targets
    9. Save engineered dataset
    """
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

    print("⏳ Creating temporal dynamics (lag & rolling)...")
    df = create_lag_features(df)
    df = create_rolling_statistics(df)

    print("🎯 Preparing target variables...")
    df = prepare_target(df)

    df = _drop_temporal_nans(df)

    sort_cols = ["STATION", "time"] if "STATION" in df.columns else ["time"]
    df = df.sort_values(sort_cols).reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"✅ Saving {df.shape[1]} columns to {output_path}...")
    df.to_csv(output_path, index=False)

    print(f"✨ Feature engineering complete! Shape: {df.shape}")
    print("Targets created: PRCP, PRCP_label, PRCP_log1p")


if __name__ == "__main__":
    feature_engineering()