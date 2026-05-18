"""
Leakage-aware feature engineering module for rainfall classification/forecasting.

This file replaces `_06_feature_engineering.py`.

Main design decisions:
1. The prediction task is defined explicitly with TASK_MODE:
   - "same_day_classification": classify whether day t is rainy using day-t
     meteorological state features. This matches the current project pipeline.
   - "next_day_forecast": use features available up to day t to predict rain
     on day t+1. In this mode rolling statistics are shifted by 1 day so they
     only use past information.
2. Target is created ONLY in this step as `rain_target`.
   The continuous precipitation amount is preserved as `PRCP_mm` for audit, but
   it is excluded from model-ready features to avoid leakage.
3. Rolling features are leakage-aware:
   - same-day classification: rolling may include current day.
   - next-day forecast: rolling uses x.shift(1).rolling(...), i.e. past-only.
4. Temporal NaNs from lag/diff/rolling are handled explicitly:
   default policy is `flag_and_keep`, which adds missingness indicators and
   leaves NaN values for models such as LightGBM that can handle missing values.
5. The script produces two outputs:
   - feature_engineered_data.csv: full audit-friendly table.
   - model_ready_data.csv: numeric model-ready table with `time`, `STATION`,
     and `rain_target`, excluding leakage columns and text/source columns.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

# Choose one:
# - "same_day_classification": classify rain occurrence on the same day.
# - "next_day_forecast": predict rain occurrence for the next day.
TASK_MODE = "same_day_classification"

VALID_TASK_MODES = {"same_day_classification", "next_day_forecast"}

PRECIPITATION_THRESHOLD_MM = 0.1

PRESSURE_LEVELS = [500, 850]
WINDOW_SIZE = 3
LAG_PERIODS = [1, 2]

# Base features for lag creation. Only available columns will be used.
LAG_FEATURES = [
    "u_850",
    "v_850",
    "q_850",
    "TEMP",
    "DEWP",
    "SLP",
    "PRCP_mm",
]

# Features for rolling statistics. Only available columns will be used.
DYNAMIC_FEATURES = [
    "moist_convection_850",
    "dew_point_depression",
    "TEMP",
    "DEWP",
    "SLP",
    "moisture_flux_850",
]

# How to handle NaNs created by lag/diff/rolling/target shifting:
# - "flag_and_keep": add *_is_missing indicators and keep NaNs.
# - "drop_temporal_na_rows": drop rows with NaN in temporal features/target.
# - "group_median_fill": fill temporal feature NaNs using group-wise median
#   and add missing indicators. Use carefully; for strict modeling, fitting
#   imputation parameters inside train folds is better.
TEMPORAL_NAN_POLICY = "flag_and_keep"
VALID_TEMPORAL_NAN_POLICIES = {"flag_and_keep", "drop_temporal_na_rows", "group_median_fill"}

# If True, model_ready_data.csv keeps only numeric features plus time/STATION/target.
NUMERIC_MODEL_READY_ONLY = True

# Columns that must never be used as model features.
LEAKAGE_AND_ID_COLS = {
    "PRCP",
    "PRCP_mm",
    "target_prcp_mm",
    "rain_target",
    "target_time",
    "DATE",
}

# Non-numeric source/audit columns retained in full output but dropped from model-ready.
SOURCE_COL_SUFFIX = "_source"


# ============================================================================
# PATH HELPERS
# ============================================================================

def find_project_root(start: Optional[Path] = None) -> Path:
    """
    Find project root by walking upward until a `data` directory is found.
    Works whether this script is placed in project root or in src/.
    """
    if start is None:
        start = Path(__file__).resolve().parent

    candidates = [start, *start.parents]
    for path in candidates:
        if (path / "data").exists():
            return path

    return Path(__file__).resolve().parent


BASE_DIR = find_project_root()
INPUT_PATH = BASE_DIR / "data" / "clean" / "silver_data_ver2.csv"
FEATURE_DIR = BASE_DIR / "data" / "feature_engineering"
FULL_OUTPUT_PATH = FEATURE_DIR / "feature_engineered_data.csv"
MODEL_READY_OUTPUT_PATH = FEATURE_DIR / "model_ready_data.csv"
REPORT_DIR = BASE_DIR / "reports" / "data_quality" / "feature_engineering"


def ensure_directories() -> None:
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _safe_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index)
    return pd.to_numeric(df[col], errors="coerce")


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def infer_groupby_columns(df: pd.DataFrame) -> List[str]:
    """
    Prefer station-level grouping. Fall back to coordinates if STATION is absent.
    """
    if "STATION" in df.columns:
        return ["STATION"]

    coord_cols = [c for c in ["LATITUDE", "LONGITUDE"] if c in df.columns]
    if len(coord_cols) == 2:
        return coord_cols

    grid_cols = [c for c in ["latitude", "longitude"] if c in df.columns]
    if len(grid_cols) == 2:
        return grid_cols

    warnings.warn(
        "No STATION or coordinate columns found. Temporal features will be "
        "computed globally, which is not ideal for panel weather data."
    )
    return []


def sort_panel(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    sort_cols = [*group_cols, "time"] if group_cols else ["time"]
    return df.sort_values(sort_cols).reset_index(drop=True)


def validate_config(task_mode: str, temporal_nan_policy: str) -> None:
    if task_mode not in VALID_TASK_MODES:
        raise ValueError(f"TASK_MODE must be one of {VALID_TASK_MODES}, got {task_mode}.")
    if temporal_nan_policy not in VALID_TEMPORAL_NAN_POLICIES:
        raise ValueError(
            f"TEMPORAL_NAN_POLICY must be one of {VALID_TEMPORAL_NAN_POLICIES}, "
            f"got {temporal_nan_policy}."
        )


# ============================================================================
# DATA LOADING AND BASIC PREPARATION
# ============================================================================

def load_silver_data(input_path: Path = INPUT_PATH) -> pd.DataFrame:
    """Load the Silver dataset and standardize time / precipitation columns."""
    if not input_path.exists():
        raise FileNotFoundError(f"Silver dataset not found: {input_path}")

    df = pd.read_csv(input_path)
    if "time" not in df.columns:
        raise ValueError("Input silver dataset must contain a `time` column.")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).reset_index(drop=True)

    if "PRCP" not in df.columns and "PRCP_mm" not in df.columns:
        raise ValueError("Input data must contain PRCP or PRCP_mm to create target.")

    if "PRCP_mm" not in df.columns:
        df["PRCP_mm"] = pd.to_numeric(df["PRCP"], errors="coerce")
    else:
        df["PRCP_mm"] = pd.to_numeric(df["PRCP_mm"], errors="coerce")

    group_cols = infer_groupby_columns(df)
    df = sort_panel(df, group_cols)

    duplicate_keys = [*group_cols, "time"] if group_cols else ["time"]
    duplicate_count = int(df.duplicated(subset=duplicate_keys).sum())
    if duplicate_count > 0:
        raise AssertionError(
            f"Input silver data has {duplicate_count:,} duplicate rows by {duplicate_keys}."
        )

    return df


# ============================================================================
# FEATURE REGISTRY
# ============================================================================

class FeatureRegistry:
    """Track created and skipped features for reporting."""

    def __init__(self) -> None:
        self.rows: List[Dict[str, object]] = []

    def created(self, name: str, category: str, description: str, leakage_note: str = "") -> None:
        self.rows.append({
            "feature": name,
            "category": category,
            "status": "created",
            "description": description,
            "leakage_note": leakage_note,
        })

    def skipped(self, name: str, category: str, reason: str) -> None:
        self.rows.append({
            "feature": name,
            "category": category,
            "status": "skipped",
            "description": reason,
            "leakage_note": "",
        })

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


# ============================================================================
# FEATURE CREATION
# ============================================================================

def create_temporal_features(df: pd.DataFrame, registry: FeatureRegistry) -> pd.DataFrame:
    """
    Create calendar and cyclical seasonal features.
    """
    out = df.copy()
    dayofyear = out["time"].dt.dayofyear

    out["day_sin"] = np.sin(2 * np.pi * dayofyear / 365.25)
    out["day_cos"] = np.cos(2 * np.pi * dayofyear / 365.25)
    registry.created("day_sin", "temporal", "Cyclical encoding of day-of-year using sine.")
    registry.created("day_cos", "temporal", "Cyclical encoding of day-of-year using cosine.")

    out["month"] = out["time"].dt.month
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    registry.created("month", "temporal", "Calendar month.")
    registry.created("month_sin", "temporal", "Cyclical encoding of month using sine.")
    registry.created("month_cos", "temporal", "Cyclical encoding of month using cosine.")

    out["is_monsoon_season"] = out["month"].isin([5, 6, 7, 8, 9, 10]).astype(int)
    registry.created(
        "is_monsoon_season",
        "temporal",
        "Binary indicator for approximate Vietnamese rainy/monsoon season."
    )

    return out


def create_thermodynamic_features(df: pd.DataFrame, registry: FeatureRegistry) -> pd.DataFrame:
    """
    Create atmospheric thermodynamic features where required columns exist.
    """
    out = df.copy()

    for lvl in PRESSURE_LEVELS:
        q_col = f"q_{lvl}"
        w_col = f"w_{lvl}"
        feature = f"moist_convection_{lvl}"

        if {q_col, w_col}.issubset(out.columns):
            out[feature] = _safe_numeric(out, q_col) * _safe_numeric(out, w_col)
            registry.created(
                feature,
                "thermodynamic",
                f"Specific humidity times vertical velocity at {lvl} hPa."
            )
        else:
            registry.skipped(
                feature,
                "thermodynamic",
                f"Missing required columns: {q_col}, {w_col}."
            )

    required = {"t_850", "t_500", "z_500", "z_850"}
    if required.issubset(out.columns):
        out["lapse_rate_850_500"] = _safe_divide(
            _safe_numeric(out, "t_850") - _safe_numeric(out, "t_500"),
            _safe_numeric(out, "z_500") - _safe_numeric(out, "z_850"),
        )
        registry.created(
            "lapse_rate_850_500",
            "thermodynamic",
            "Temperature difference between 850 hPa and 500 hPa divided by layer thickness."
        )
    else:
        registry.skipped(
            "lapse_rate_850_500",
            "thermodynamic",
            f"Missing required columns: {sorted(required - set(out.columns))}."
        )

    if {"TEMP", "DEWP"}.issubset(out.columns):
        out["dew_point_depression"] = _safe_numeric(out, "TEMP") - _safe_numeric(out, "DEWP")
        registry.created(
            "dew_point_depression",
            "thermodynamic",
            "Surface temperature minus dew point; lower values imply moister air."
        )
    else:
        registry.skipped(
            "dew_point_depression",
            "thermodynamic",
            "Missing TEMP and/or DEWP."
        )

    return out


def create_wind_features(
    df: pd.DataFrame,
    group_cols: List[str],
    registry: FeatureRegistry
) -> pd.DataFrame:
    """
    Create wind shear and time-difference dynamic features.
    """
    out = sort_panel(df, group_cols)

    required = {"u_850", "u_500", "v_850", "v_500"}
    if required.issubset(out.columns):
        out["du_850_500"] = _safe_numeric(out, "u_850") - _safe_numeric(out, "u_500")
        out["dv_850_500"] = _safe_numeric(out, "v_850") - _safe_numeric(out, "v_500")
        out["wind_shear_mag"] = np.sqrt(out["du_850_500"] ** 2 + out["dv_850_500"] ** 2)

        registry.created("du_850_500", "wind_dynamic", "Vertical u-wind difference: 850 hPa minus 500 hPa.")
        registry.created("dv_850_500", "wind_dynamic", "Vertical v-wind difference: 850 hPa minus 500 hPa.")
        registry.created("wind_shear_mag", "wind_dynamic", "Magnitude of vertical wind shear between 850 hPa and 500 hPa.")
    else:
        registry.skipped(
            "wind_shear_mag",
            "wind_dynamic",
            f"Missing required columns: {sorted(required - set(out.columns))}."
        )

    for wind_col in ["u_850", "v_850"]:
        diff_col = f"d{wind_col}_dt"
        if wind_col in out.columns:
            if group_cols:
                out[diff_col] = out.groupby(group_cols)[wind_col].diff()
            else:
                out[diff_col] = out[wind_col].diff()
            registry.created(
                diff_col,
                "wind_dynamic",
                f"First temporal difference of {wind_col}; uses current minus previous day."
            )
        else:
            registry.skipped(diff_col, "wind_dynamic", f"Missing required column: {wind_col}.")

    if {"du_850_dt", "dv_850_dt"}.issubset(out.columns):
        out["ageostrophic_signal"] = np.sqrt(out["du_850_dt"] ** 2 + out["dv_850_dt"] ** 2)
        registry.created(
            "ageostrophic_signal",
            "wind_dynamic",
            "Magnitude of day-to-day change in 850 hPa wind components."
        )
    else:
        registry.skipped(
            "ageostrophic_signal",
            "wind_dynamic",
            "Missing du_850_dt and/or dv_850_dt."
        )

    return out


def create_geometric_features(df: pd.DataFrame, registry: FeatureRegistry) -> pd.DataFrame:
    """
    Create vertical geometry features.
    """
    out = df.copy()

    if {"z_500", "z_850"}.issubset(out.columns):
        out["thickness_500_850"] = _safe_numeric(out, "z_500") - _safe_numeric(out, "z_850")
        registry.created(
            "thickness_500_850",
            "geometric",
            "Geopotential height difference between 500 hPa and 850 hPa."
        )
    else:
        registry.skipped(
            "thickness_500_850",
            "geometric",
            "Missing z_500 and/or z_850."
        )

    if {"ELEVATION", "z"}.issubset(out.columns):
        out["elevation_diff"] = _safe_numeric(out, "ELEVATION") - _safe_numeric(out, "z")
        registry.created(
            "elevation_diff",
            "geometric",
            "Station elevation minus ERA5 surface geopotential height."
        )
    else:
        registry.skipped(
            "elevation_diff",
            "geometric",
            "Missing ELEVATION and/or z."
        )

    return out


def create_flux_features(df: pd.DataFrame, registry: FeatureRegistry) -> pd.DataFrame:
    """
    Create moisture transport features.
    """
    out = df.copy()

    if {"q_850", "u_850", "v_850"}.issubset(out.columns):
        out["moisture_flux_850"] = (
            _safe_numeric(out, "q_850")
            * np.sqrt(_safe_numeric(out, "u_850") ** 2 + _safe_numeric(out, "v_850") ** 2)
        )
        registry.created(
            "moisture_flux_850",
            "flux",
            "850 hPa specific humidity multiplied by horizontal wind magnitude."
        )
    else:
        registry.skipped(
            "moisture_flux_850",
            "flux",
            "Missing q_850 and/or u_850/v_850."
        )

    return out


def create_lag_features(
    df: pd.DataFrame,
    group_cols: List[str],
    registry: FeatureRegistry,
    features: Optional[List[str]] = None,
    lags: Optional[List[int]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create lagged features. Lags always use past values only.
    """
    if features is None:
        features = LAG_FEATURES
    if lags is None:
        lags = LAG_PERIODS

    out = sort_panel(df, group_cols)
    created_cols: List[str] = []

    for feature in features:
        if feature not in out.columns:
            registry.skipped(f"{feature}_lag", "temporal_dynamics", f"Missing source column: {feature}.")
            continue

        for lag in lags:
            new_col = f"{feature}_lag_{lag}"
            if group_cols:
                out[new_col] = out.groupby(group_cols)[feature].shift(lag)
            else:
                out[new_col] = out[feature].shift(lag)

            created_cols.append(new_col)
            registry.created(
                new_col,
                "temporal_dynamics",
                f"{lag}-day lag of {feature}.",
                "Past-only by construction."
            )

    return out, created_cols


def create_rolling_statistics(
    df: pd.DataFrame,
    group_cols: List[str],
    registry: FeatureRegistry,
    task_mode: str,
    features: Optional[List[str]] = None,
    window: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create rolling statistics.

    For same-day classification:
        rolling(t) = mean/sum over [t-window+1, ..., t]
    For next-day forecast:
        rolling(t) = mean/sum over [t-window, ..., t-1]
        implemented as x.shift(1).rolling(...).
    """
    if features is None:
        features = DYNAMIC_FEATURES
    if window is None:
        window = WINDOW_SIZE

    out = sort_panel(df, group_cols)
    created_cols: List[str] = []

    use_past_only = task_mode == "next_day_forecast"
    leakage_note = (
        "Past-only rolling with shift(1), safe for next-day forecasting."
        if use_past_only
        else "Includes current day, valid for same-day classification but not for strict future forecasting."
    )

    def rolling_mean(s: pd.Series) -> pd.Series:
        base = s.shift(1) if use_past_only else s
        return base.rolling(window=window, min_periods=1).mean()

    def rolling_sum(s: pd.Series) -> pd.Series:
        base = s.shift(1) if use_past_only else s
        return base.rolling(window=window, min_periods=1).sum()

    for col in features:
        if col not in out.columns:
            registry.skipped(f"{col}_{window}d_mean", "temporal_dynamics", f"Missing source column: {col}.")
            continue

        new_col = f"{col}_{window}d_mean"
        if group_cols:
            out[new_col] = out.groupby(group_cols)[col].transform(rolling_mean)
        else:
            out[new_col] = rolling_mean(out[col])

        created_cols.append(new_col)
        registry.created(
            new_col,
            "temporal_dynamics",
            f"{window}-day rolling mean of {col}.",
            leakage_note
        )

    if "moisture_flux_850" in out.columns:
        new_col = f"moisture_flux_{window}d_sum"
        if group_cols:
            out[new_col] = out.groupby(group_cols)["moisture_flux_850"].transform(rolling_sum)
        else:
            out[new_col] = rolling_sum(out["moisture_flux_850"])

        created_cols.append(new_col)
        registry.created(
            new_col,
            "temporal_dynamics",
            f"{window}-day rolling sum of moisture_flux_850.",
            leakage_note
        )
    else:
        registry.skipped(
            f"moisture_flux_{window}d_sum",
            "temporal_dynamics",
            "Missing source column: moisture_flux_850."
        )

    return out, created_cols


# ============================================================================
# TARGET CREATION
# ============================================================================

def create_target(
    df: pd.DataFrame,
    group_cols: List[str],
    registry: FeatureRegistry,
    task_mode: str = TASK_MODE,
    threshold_mm: float = PRECIPITATION_THRESHOLD_MM,
) -> pd.DataFrame:
    """
    Create the binary rainfall target exactly once.

    same_day_classification:
        rain_target(t) = 1 if PRCP_mm(t) > threshold else 0

    next_day_forecast:
        rain_target(t) = 1 if PRCP_mm(t+1) > threshold else 0
        target_prcp_mm and target_time are added for audit.
    """
    out = sort_panel(df, group_cols)

    if "PRCP_mm" not in out.columns:
        raise ValueError("PRCP_mm is required to create rain_target.")

    if task_mode == "same_day_classification":
        out["target_prcp_mm"] = out["PRCP_mm"]
        out["target_time"] = out["time"]
        registry.created(
            "rain_target",
            "target",
            f"Same-day binary target: PRCP_mm > {threshold_mm} mm."
        )

    elif task_mode == "next_day_forecast":
        if group_cols:
            out["target_prcp_mm"] = out.groupby(group_cols)["PRCP_mm"].shift(-1)
            out["target_time"] = out.groupby(group_cols)["time"].shift(-1)
        else:
            out["target_prcp_mm"] = out["PRCP_mm"].shift(-1)
            out["target_time"] = out["time"].shift(-1)

        registry.created(
            "rain_target",
            "target",
            f"Next-day binary target: PRCP_mm(t+1) > {threshold_mm} mm."
        )

    else:
        raise ValueError(f"Unsupported task_mode: {task_mode}")

    out["rain_target"] = np.where(
        out["target_prcp_mm"].notna(),
        (out["target_prcp_mm"] > threshold_mm).astype(int),
        np.nan,
    )

    # Drop rows without target. For next-day forecast, this removes the last
    # record of each group. For same-day classification, it removes rows whose
    # PRCP_mm is missing even after the Silver layer.
    before = len(out)
    out = out.dropna(subset=["rain_target"]).copy()
    after = len(out)

    out["rain_target"] = out["rain_target"].astype(int)

    if before != after:
        warnings.warn(f"Dropped {before - after:,} rows with missing target.")

    return out


# ============================================================================
# NAN HANDLING AND VALIDATION
# ============================================================================

def report_temporal_nans(df: pd.DataFrame, temporal_cols: List[str]) -> pd.DataFrame:
    rows = []
    for col in temporal_cols:
        if col not in df.columns:
            continue
        rows.append({
            "column": col,
            "missing_count": int(df[col].isna().sum()),
            "missing_rate": float(df[col].isna().mean()),
        })
    report = pd.DataFrame(rows).sort_values("missing_rate", ascending=False)
    report.to_csv(REPORT_DIR / "temporal_nan_report.csv", index=False)
    return report


def handle_temporal_nans(
    df: pd.DataFrame,
    group_cols: List[str],
    temporal_cols: List[str],
    policy: str = TEMPORAL_NAN_POLICY,
) -> pd.DataFrame:
    """
    Handle NaNs created by lag/diff/rolling features.
    """
    out = df.copy()
    temporal_cols = [col for col in temporal_cols if col in out.columns]

    for col in temporal_cols:
        if out[col].isna().any():
            out[f"{col}_is_missing"] = out[col].isna().astype(int)

    if policy == "flag_and_keep":
        return out

    if policy == "drop_temporal_na_rows":
        before = len(out)
        out = out.dropna(subset=temporal_cols).reset_index(drop=True)
        after = len(out)
        print(f"Dropped {before - after:,} rows with temporal NaNs.")
        return out

    if policy == "group_median_fill":
        for col in temporal_cols:
            if not out[col].isna().any():
                continue

            if group_cols:
                out[col] = out.groupby(group_cols)[col].transform(
                    lambda x: x.fillna(x.median())
                )

            # Fallback for all-NaN groups.
            global_median = out[col].median()
            out[col] = out[col].fillna(global_median)

        return out

    raise ValueError(f"Unsupported temporal NaN policy: {policy}")


def validate_feature_engineered_data(
    df: pd.DataFrame,
    group_cols: List[str],
    task_mode: str,
) -> None:
    """Final sanity checks for engineered data."""
    if "rain_target" not in df.columns:
        raise AssertionError("rain_target was not created.")

    target_values = set(df["rain_target"].dropna().unique().tolist())
    if not target_values.issubset({0, 1}):
        raise AssertionError(f"rain_target must be binary 0/1. Found: {target_values}")

    duplicate_keys = [*group_cols, "time"] if group_cols else ["time"]
    duplicate_count = int(df.duplicated(subset=duplicate_keys).sum())
    if duplicate_count > 0:
        raise AssertionError(
            f"Feature engineered data has {duplicate_count:,} duplicate rows by {duplicate_keys}."
        )

    if task_mode == "next_day_forecast":
        if "target_time" not in df.columns:
            raise AssertionError("next_day_forecast mode must create target_time.")

        if not (pd.to_datetime(df["target_time"]) > pd.to_datetime(df["time"])).all():
            raise AssertionError("For next_day_forecast, every target_time must be after feature time.")

    # Ensure we did not overwrite PRCP into binary labels.
    if "PRCP" in df.columns:
        unique_prcp = df["PRCP"].dropna().unique()
        if len(unique_prcp) <= 2 and set(unique_prcp).issubset({0, 1}):
            warnings.warn(
                "PRCP appears to be binary. In the improved design, PRCP should "
                "remain continuous if present, and rain_target is the binary label."
            )


# ============================================================================
# MODEL-READY OUTPUT
# ============================================================================

def create_model_ready_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a model-ready table that excludes leakage columns and non-numeric
    source/audit columns.

    It intentionally keeps `time` for time-based splitting and `STATION` for
    grouped diagnostics, but model training should drop both as identifiers.
    """
    out = df.copy()

    # Drop leakage columns but keep target.
    drop_cols = [
        col for col in out.columns
        if col in LEAKAGE_AND_ID_COLS and col != "rain_target"
    ]

    # Drop text/source/audit columns from model-ready.
    if NUMERIC_MODEL_READY_ONLY:
        object_cols = out.select_dtypes(include=["object", "category"]).columns.tolist()
        object_drop = [c for c in object_cols if c not in ["STATION"]]
        drop_cols.extend(object_drop)

        source_cols = [c for c in out.columns if c.endswith(SOURCE_COL_SUFFIX)]
        drop_cols.extend(source_cols)

    drop_cols = sorted(set(c for c in drop_cols if c in out.columns))
    out = out.drop(columns=drop_cols, errors="ignore")

    # Keep only numeric + time + STATION + target
    if NUMERIC_MODEL_READY_ONLY:
        keep_cols = []
        for col in out.columns:
            if col in {"time", "STATION", "rain_target"}:
                keep_cols.append(col)
            elif pd.api.types.is_numeric_dtype(out[col]):
                keep_cols.append(col)

        out = out[keep_cols].copy()

    # Sort stable
    sort_cols = [c for c in ["STATION", "time"] if c in out.columns]
    out = out.sort_values(sort_cols).reset_index(drop=True)

    # Save dropped columns rationale
    pd.DataFrame({
        "dropped_column": drop_cols,
        "reason": [
            "Excluded from model-ready data because it is a leakage, ID, source, or non-numeric audit column."
            for _ in drop_cols
        ],
    }).to_csv(REPORT_DIR / "model_ready_dropped_columns.csv", index=False)

    return out


# ============================================================================
# REPORTS AND METADATA
# ============================================================================

def save_reports_and_metadata(
    full_df: pd.DataFrame,
    model_ready_df: pd.DataFrame,
    registry: FeatureRegistry,
    group_cols: List[str],
    temporal_cols: List[str],
    task_mode: str,
    input_path: Path,
    full_output_path: Path,
    model_ready_output_path: Path,
) -> None:
    ensure_directories()

    registry_df = registry.to_frame()
    registry_df.to_csv(REPORT_DIR / "feature_registry.csv", index=False)

    # Target distribution
    target_dist = (
        full_df["rain_target"]
        .value_counts(dropna=False)
        .rename_axis("rain_target")
        .reset_index(name="count")
    )
    target_dist["rate"] = target_dist["count"] / len(full_df)
    target_dist.to_csv(REPORT_DIR / "target_distribution.csv", index=False)

    # Missing report
    missing_report = (
        full_df.isna()
        .mean()
        .rename("missing_rate")
        .reset_index()
        .rename(columns={"index": "column"})
        .sort_values("missing_rate", ascending=False)
    )
    missing_report.to_csv(REPORT_DIR / "feature_missing_rate.csv", index=False)

    # Model-ready schema
    schema = pd.DataFrame({
        "column": model_ready_df.columns,
        "dtype": [str(model_ready_df[c].dtype) for c in model_ready_df.columns],
        "missing_rate": [float(model_ready_df[c].isna().mean()) for c in model_ready_df.columns],
    })
    schema.to_csv(REPORT_DIR / "model_ready_schema.csv", index=False)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "full_output_path": str(full_output_path),
        "model_ready_output_path": str(model_ready_output_path),
        "task_mode": task_mode,
        "task_definition": (
            "Classify whether the same day is rainy from same-day meteorological state."
            if task_mode == "same_day_classification"
            else "Predict whether the next day is rainy using information available up to the current day."
        ),
        "rain_threshold_mm": PRECIPITATION_THRESHOLD_MM,
        "groupby_columns": group_cols,
        "window_size": WINDOW_SIZE,
        "lag_periods": LAG_PERIODS,
        "temporal_nan_policy": TEMPORAL_NAN_POLICY,
        "rolling_policy": (
            "Rolling features include current day because task_mode is same_day_classification."
            if task_mode == "same_day_classification"
            else "Rolling features use shift(1) before rolling to avoid future/current-target leakage."
        ),
        "target_policy": (
            "Target is created only in step 6 as `rain_target`; step 7 must not recreate PRCP."
        ),
        "model_input_note": (
            "Step 7 should read data/feature_engineering/model_ready_data.csv "
            "or data/feature_engineering/feature_engineered_data.csv with leakage columns dropped. "
            "It should use `rain_target` as target and should not binarize PRCP again."
        ),
        "n_rows_full": int(len(full_df)),
        "n_columns_full": int(full_df.shape[1]),
        "n_rows_model_ready": int(len(model_ready_df)),
        "n_columns_model_ready": int(model_ready_df.shape[1]),
        "reports": {
            "feature_registry": str(REPORT_DIR / "feature_registry.csv"),
            "target_distribution": str(REPORT_DIR / "target_distribution.csv"),
            "feature_missing_rate": str(REPORT_DIR / "feature_missing_rate.csv"),
            "temporal_nan_report": str(REPORT_DIR / "temporal_nan_report.csv"),
            "model_ready_schema": str(REPORT_DIR / "model_ready_schema.csv"),
            "model_ready_dropped_columns": str(REPORT_DIR / "model_ready_dropped_columns.csv"),
        },
    }

    _write_json(metadata, REPORT_DIR / "feature_engineering_metadata.json")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def feature_engineering(
    input_path: Path | str = INPUT_PATH,
    full_output_path: Path | str = FULL_OUTPUT_PATH,
    model_ready_output_path: Path | str = MODEL_READY_OUTPUT_PATH,
    task_mode: str = TASK_MODE,
    temporal_nan_policy: str = TEMPORAL_NAN_POLICY,
) -> pd.DataFrame:
    """
    End-to-end feature engineering pipeline.

    Returns the full audit-friendly feature-engineered dataframe.
    """
    validate_config(task_mode, temporal_nan_policy)
    ensure_directories()

    input_path = Path(input_path)
    full_output_path = Path(full_output_path)
    model_ready_output_path = Path(model_ready_output_path)

    print("\n=== STEP 6: LEAKAGE-AWARE FEATURE ENGINEERING ===")
    print(f"Task mode: {task_mode}")
    print(f"Input: {input_path}")

    df = load_silver_data(input_path)
    group_cols = infer_groupby_columns(df)
    registry = FeatureRegistry()

    # Feature creation
    df = create_temporal_features(df, registry)
    df = create_thermodynamic_features(df, registry)
    df = create_wind_features(df, group_cols, registry)
    df = create_geometric_features(df, registry)
    df = create_flux_features(df, registry)

    temporal_cols: List[str] = []
    df, lag_cols = create_lag_features(df, group_cols, registry)
    temporal_cols.extend(lag_cols)

    df, rolling_cols = create_rolling_statistics(df, group_cols, registry, task_mode=task_mode)
    temporal_cols.extend(rolling_cols)

    # Include diff-based dynamic features in temporal NaN report.
    temporal_cols.extend([c for c in ["du_850_dt", "dv_850_dt", "ageostrophic_signal"] if c in df.columns])
    temporal_cols = sorted(set(temporal_cols))

    # Target creation must happen after PRCP_mm is preserved.
    df = create_target(df, group_cols, registry, task_mode=task_mode)

    # Temporal NaN handling
    report_temporal_nans(df, temporal_cols)
    df = handle_temporal_nans(df, group_cols, temporal_cols, policy=temporal_nan_policy)

    validate_feature_engineered_data(df, group_cols, task_mode)

    # Full audit-friendly output
    full_output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(full_output_path, index=False)

    # Model-ready output
    model_ready_df = create_model_ready_dataset(df)
    model_ready_output_path.parent.mkdir(parents=True, exist_ok=True)
    model_ready_df.to_csv(model_ready_output_path, index=False)

    save_reports_and_metadata(
        full_df=df,
        model_ready_df=model_ready_df,
        registry=registry,
        group_cols=group_cols,
        temporal_cols=temporal_cols,
        task_mode=task_mode,
        input_path=input_path,
        full_output_path=full_output_path,
        model_ready_output_path=model_ready_output_path,
    )

    print(f"\n[SUCCESS] Full feature-engineered data saved to: {full_output_path}")
    print(f"          Shape: {df.shape}")
    print(f"[SUCCESS] Model-ready data saved to: {model_ready_output_path}")
    print(f"          Shape: {model_ready_df.shape}")
    print(f"[SUCCESS] Reports saved to: {REPORT_DIR}")

    return df


if __name__ == "__main__":
    feature_engineering(
        input_path=INPUT_PATH,
        full_output_path=FULL_OUTPUT_PATH,
        model_ready_output_path=MODEL_READY_OUTPUT_PATH,
        task_mode=TASK_MODE,
        temporal_nan_policy=TEMPORAL_NAN_POLICY,
    )
