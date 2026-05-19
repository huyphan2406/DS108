"""
Step 6 - Single-output Leakage-aware Feature Engineering for DS108 Rainfall Project.

Goal
----
Build ONE clean feature file for Step 7:

    data/feature_engineering/feature_engineered_data.csv

This file is already model-ready:
- keeps: time, STATION, rain_target, engineered numeric features
- drops: PRCP, PRCP_mm, target_prcp_mm, target_time, raw month/YEAR/MONTH
- avoids creating many 0/1 missing-indicator columns
- by default, only `rain_target` is allowed to be a binary 0/1 column

Problem framing
---------------
The project estimates rain probability:

    P(rain = 1 | meteorological features)

Then the model converts probability into Rain / No-rain using a decision
threshold optimized in Step 7.

Default task mode:
    same_day_classification
    = classify whether day t is rainy using meteorological state features at day t.

Input:
    data/clean/silver_data.csv

Output:
    data/feature_engineering/feature_engineered_data.csv
    reports/data_quality/feature_engineering/
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

TASK_MODE = "same_day_classification"
VALID_TASK_MODES = {"same_day_classification", "next_day_forecast"}

PRECIPITATION_THRESHOLD_MM = 0.1

PRESSURE_LEVELS = [500, 850]
WINDOW_SIZE = 3
LAG_PERIODS = [1, 2]

# Keep this False to avoid extra 0/1 features.
CREATE_MONSOON_BINARY_FEATURE = False

# Keep this False to avoid many *_is_missing binary columns.
ADD_MISSING_INDICATORS = False

# Cleanest default: remove the first 1-2 days per station caused by lag/diff NaNs.
TEMPORAL_NAN_POLICY = "drop_temporal_na_rows"
VALID_TEMPORAL_NAN_POLICIES = {
    "keep_nan",
    "drop_temporal_na_rows",
    "group_median_fill",
}

# Remove constant features and binary predictor columns.
# This keeps the final dataset easier to explain.
DROP_CONSTANT_FEATURES = True
DROP_BINARY_PREDICTOR_FEATURES = True

# Optional: location can be useful for the same-station validation setup.
# Set False if your instructor worries about station memorization.
KEEP_LOCATION_FEATURES = True

LOCATION_COLS = {"LATITUDE", "LONGITUDE", "ELEVATION"}

# Past-only lag features. PRCP_mm_lag_1/2 are allowed because they use previous
# days only, not current target-day rainfall.
LAG_FEATURES = [
    "u_850",
    "v_850",
    "q_850",
    "TEMP",
    "DEWP",
    "SLP",
    "PRCP_mm",
]

DYNAMIC_FEATURES = [
    "moist_convection_850",
    "dew_point_depression",
    "TEMP",
    "DEWP",
    "SLP",
    "moisture_flux_850",
]

# Columns used for target creation / audit but never used as model features.
LEAKAGE_AND_AUDIT_COLS = {
    "PRCP",
    "PRCP_mm",
    "target_prcp_mm",
    "target_time",
    "DATE",
    "YEAR",
    "MONTH",
    "month",
}

# Final file still keeps these non-feature columns for splitting/diagnostics.
ALWAYS_KEEP_COLS = {"time", "STATION", "rain_target"}


# =============================================================================
# PATH HELPERS
# =============================================================================

def find_project_root(start: Optional[Path] = None) -> Path:
    """Find project root by walking upward until a `data` directory is found."""
    if start is None:
        start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

    candidates = [start, *start.parents]
    for path in candidates:
        if (path / "data").exists():
            return path

    return start


BASE_DIR = find_project_root()

DATA_DIR = BASE_DIR / "data"
CLEAN_DIR = DATA_DIR / "clean"
FEATURE_DIR = DATA_DIR / "feature_engineering"
REPORT_ROOT_DIR = BASE_DIR / "reports" / "data_quality"

INPUT_PATH = CLEAN_DIR / "silver_data.csv"
FEATURE_OUTPUT_PATH = FEATURE_DIR / "feature_engineered_data.csv"
REPORT_DIR = REPORT_ROOT_DIR / "feature_engineering"


def ensure_directories() -> None:
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# BASIC UTILITIES
# =============================================================================

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


def validate_config(task_mode: str, temporal_nan_policy: str) -> None:
    if task_mode not in VALID_TASK_MODES:
        raise ValueError(f"TASK_MODE must be one of {VALID_TASK_MODES}, got {task_mode}.")

    if temporal_nan_policy not in VALID_TEMPORAL_NAN_POLICIES:
        raise ValueError(
            f"TEMPORAL_NAN_POLICY must be one of {VALID_TEMPORAL_NAN_POLICIES}, "
            f"got {temporal_nan_policy}."
        )


def infer_groupby_columns(df: pd.DataFrame) -> List[str]:
    """Prefer station-level grouping."""
    if "STATION" in df.columns:
        return ["STATION"]

    coord_cols = [c for c in ["LATITUDE", "LONGITUDE"] if c in df.columns]
    if len(coord_cols) == 2:
        return coord_cols

    warnings.warn(
        "No STATION or LATITUDE/LONGITUDE columns found. Temporal features will "
        "be computed globally, which is not ideal for panel weather data."
    )
    return []


def sort_panel(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    sort_cols = [*group_cols, "time"] if group_cols else ["time"]
    return df.sort_values(sort_cols).reset_index(drop=True)


def is_binary_series(series: pd.Series) -> bool:
    """Return True if a non-empty series has only values in {0, 1}."""
    s = series.dropna()
    if s.empty:
        return False

    unique_vals = set(pd.unique(s).tolist())
    return unique_vals.issubset({0, 1}) and len(unique_vals) <= 2


def drop_all_missing_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    all_missing_cols = [col for col in df.columns if df[col].isna().all()]
    if all_missing_cols:
        print(f"Dropping all-missing columns: {all_missing_cols}")
        df = df.drop(columns=all_missing_cols)
    return df, all_missing_cols


# =============================================================================
# DATA LOADING
# =============================================================================

def load_silver_data(input_path: Path = INPUT_PATH) -> pd.DataFrame:
    """Load clean Silver dataset and prepare core columns."""
    if not input_path.exists():
        raise FileNotFoundError(f"Silver dataset not found: {input_path}")

    df = pd.read_csv(input_path)

    if "time" not in df.columns:
        raise ValueError("Input Silver dataset must contain a `time` column.")

    if "STATION" in df.columns:
        df["STATION"] = df["STATION"].astype(str)

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).reset_index(drop=True)

    if "PRCP" not in df.columns and "PRCP_mm" not in df.columns:
        raise ValueError("Input data must contain PRCP or PRCP_mm to create rain_target.")

    # Preserve continuous precipitation amount for target creation and lag features.
    if "PRCP_mm" not in df.columns:
        df["PRCP_mm"] = pd.to_numeric(df["PRCP"], errors="coerce")
    else:
        df["PRCP_mm"] = pd.to_numeric(df["PRCP_mm"], errors="coerce")

    df, _ = drop_all_missing_columns(df)

    group_cols = infer_groupby_columns(df)
    df = sort_panel(df, group_cols)

    duplicate_keys = [*group_cols, "time"] if group_cols else ["time"]
    duplicate_count = int(df.duplicated(subset=duplicate_keys).sum())
    if duplicate_count > 0:
        raise AssertionError(
            f"Input Silver data has {duplicate_count:,} duplicate rows by {duplicate_keys}."
        )

    return df


# =============================================================================
# FEATURE REGISTRY
# =============================================================================

class FeatureRegistry:
    """Track created and skipped features for reports."""

    def __init__(self) -> None:
        self.rows: List[Dict[str, object]] = []

    def created(
        self,
        name: str,
        category: str,
        description: str,
        leakage_note: str = "",
    ) -> None:
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

    def dropped(self, name: str, category: str, reason: str) -> None:
        self.rows.append({
            "feature": name,
            "category": category,
            "status": "dropped",
            "description": reason,
            "leakage_note": "",
        })

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


# =============================================================================
# FEATURE CREATION
# =============================================================================

def create_temporal_features(
    df: pd.DataFrame,
    registry: FeatureRegistry,
    create_monsoon_binary: bool = CREATE_MONSOON_BINARY_FEATURE,
) -> pd.DataFrame:
    """Create seasonality features using cyclical encodings."""
    out = df.copy()

    dayofyear = out["time"].dt.dayofyear
    out["day_sin"] = np.sin(2 * np.pi * dayofyear / 365.25)
    out["day_cos"] = np.cos(2 * np.pi * dayofyear / 365.25)
    registry.created("day_sin", "temporal", "Cyclical day-of-year sine encoding.")
    registry.created("day_cos", "temporal", "Cyclical day-of-year cosine encoding.")

    out["month"] = out["time"].dt.month
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    registry.created("month", "temporal", "Calendar month for audit; dropped from final features.")
    registry.created("month_sin", "temporal", "Cyclical month sine encoding.")
    registry.created("month_cos", "temporal", "Cyclical month cosine encoding.")

    if create_monsoon_binary:
        out["is_monsoon_season"] = out["month"].isin([5, 6, 7, 8, 9, 10]).astype(int)
        registry.created(
            "is_monsoon_season",
            "temporal",
            "Binary indicator for approximate Vietnamese rainy/monsoon season.",
        )
    else:
        registry.skipped(
            "is_monsoon_season",
            "temporal",
            "Disabled by default to avoid extra 0/1 predictors.",
        )

    return out


def create_thermodynamic_features(df: pd.DataFrame, registry: FeatureRegistry) -> pd.DataFrame:
    """Create thermodynamic features."""
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
                f"Specific humidity × vertical velocity at {lvl} hPa.",
            )
        else:
            registry.skipped(
                feature,
                "thermodynamic",
                f"Missing required columns: {q_col}, {w_col}.",
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
            "Temperature difference between 850 and 500 hPa divided by layer thickness.",
        )
    else:
        registry.skipped(
            "lapse_rate_850_500",
            "thermodynamic",
            f"Missing required columns: {sorted(required - set(out.columns))}.",
        )

    if {"TEMP", "DEWP"}.issubset(out.columns):
        out["dew_point_depression"] = _safe_numeric(out, "TEMP") - _safe_numeric(out, "DEWP")
        registry.created(
            "dew_point_depression",
            "thermodynamic",
            "Surface temperature minus dew point; lower value implies moister air.",
        )
    else:
        registry.skipped("dew_point_depression", "thermodynamic", "Missing TEMP and/or DEWP.")

    return out


def create_wind_features(
    df: pd.DataFrame,
    group_cols: List[str],
    registry: FeatureRegistry,
) -> Tuple[pd.DataFrame, List[str]]:
    """Create wind shear and dynamic features."""
    out = sort_panel(df, group_cols)
    temporal_cols: List[str] = []

    required = {"u_850", "u_500", "v_850", "v_500"}
    if required.issubset(out.columns):
        out["du_850_500"] = _safe_numeric(out, "u_850") - _safe_numeric(out, "u_500")
        out["dv_850_500"] = _safe_numeric(out, "v_850") - _safe_numeric(out, "v_500")
        out["wind_shear_mag"] = np.sqrt(out["du_850_500"] ** 2 + out["dv_850_500"] ** 2)

        registry.created("du_850_500", "wind_dynamic", "Vertical u-wind difference: 850 - 500 hPa.")
        registry.created("dv_850_500", "wind_dynamic", "Vertical v-wind difference: 850 - 500 hPa.")
        registry.created("wind_shear_mag", "wind_dynamic", "Magnitude of 850-500 hPa vertical wind shear.")
    else:
        registry.skipped(
            "wind_shear_mag",
            "wind_dynamic",
            f"Missing required columns: {sorted(required - set(out.columns))}.",
        )

    for wind_col in ["u_850", "v_850"]:
        diff_col = f"{wind_col}_diff_1d"
        if wind_col in out.columns:
            if group_cols:
                out[diff_col] = out.groupby(group_cols)[wind_col].diff()
            else:
                out[diff_col] = out[wind_col].diff()

            temporal_cols.append(diff_col)
            registry.created(
                diff_col,
                "wind_dynamic",
                f"One-day difference of {wind_col}: current day minus previous day.",
            )
        else:
            registry.skipped(diff_col, "wind_dynamic", f"Missing required column: {wind_col}.")

    if {"u_850_diff_1d", "v_850_diff_1d"}.issubset(out.columns):
        out["ageostrophic_signal"] = np.sqrt(
            out["u_850_diff_1d"] ** 2 + out["v_850_diff_1d"] ** 2
        )
        temporal_cols.append("ageostrophic_signal")
        registry.created(
            "ageostrophic_signal",
            "wind_dynamic",
            "Magnitude of one-day change in 850 hPa wind components.",
        )
    else:
        registry.skipped(
            "ageostrophic_signal",
            "wind_dynamic",
            "Missing u_850_diff_1d and/or v_850_diff_1d.",
        )

    return out, temporal_cols


def create_geometric_features(df: pd.DataFrame, registry: FeatureRegistry) -> pd.DataFrame:
    """Create geometric/terrain features."""
    out = df.copy()

    if {"z_500", "z_850"}.issubset(out.columns):
        out["thickness_500_850"] = _safe_numeric(out, "z_500") - _safe_numeric(out, "z_850")
        registry.created(
            "thickness_500_850",
            "geometric",
            "Geopotential height difference between 500 and 850 hPa.",
        )
    else:
        registry.skipped("thickness_500_850", "geometric", "Missing z_500 and/or z_850.")

    if {"ELEVATION", "z"}.issubset(out.columns):
        out["elevation_diff"] = _safe_numeric(out, "ELEVATION") - _safe_numeric(out, "z")
        registry.created(
            "elevation_diff",
            "geometric",
            "Station elevation minus ERA5 surface geopotential-related variable.",
        )
    else:
        registry.skipped("elevation_diff", "geometric", "Missing ELEVATION and/or z.")

    return out


def create_flux_features(df: pd.DataFrame, registry: FeatureRegistry) -> pd.DataFrame:
    """Create moisture transport features."""
    out = df.copy()

    if {"q_850", "u_850", "v_850"}.issubset(out.columns):
        out["moisture_flux_850"] = (
            _safe_numeric(out, "q_850")
            * np.sqrt(_safe_numeric(out, "u_850") ** 2 + _safe_numeric(out, "v_850") ** 2)
        )
        registry.created(
            "moisture_flux_850",
            "flux",
            "850 hPa specific humidity multiplied by horizontal wind magnitude.",
        )
    else:
        registry.skipped(
            "moisture_flux_850",
            "flux",
            "Missing q_850 and/or u_850/v_850.",
        )

    return out


def create_lag_features(
    df: pd.DataFrame,
    group_cols: List[str],
    registry: FeatureRegistry,
    features: Optional[List[str]] = None,
    lags: Optional[List[int]] = None,
) -> Tuple[pd.DataFrame, List[str]]:
    """Create past-only lag features within each station/group."""
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
                "Past-only by construction.",
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
    """Create rolling statistics within each station/group."""
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
        else "Includes current day; valid for same-day classification."
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
        registry.created(new_col, "temporal_dynamics", f"{window}-day rolling mean of {col}.", leakage_note)

    if "moisture_flux_850" in out.columns:
        new_col = f"moisture_flux_{window}d_sum"
        if group_cols:
            out[new_col] = out.groupby(group_cols)["moisture_flux_850"].transform(rolling_sum)
        else:
            out[new_col] = rolling_sum(out["moisture_flux_850"])

        created_cols.append(new_col)
        registry.created(new_col, "temporal_dynamics", f"{window}-day rolling sum of moisture_flux_850.", leakage_note)
    else:
        registry.skipped(f"moisture_flux_{window}d_sum", "temporal_dynamics", "Missing moisture_flux_850.")

    return out, created_cols


# =============================================================================
# TARGET
# =============================================================================

def create_target(
    df: pd.DataFrame,
    group_cols: List[str],
    registry: FeatureRegistry,
    task_mode: str = TASK_MODE,
    threshold_mm: float = PRECIPITATION_THRESHOLD_MM,
) -> pd.DataFrame:
    """Create `rain_target` while preserving PRCP/PRCP_mm."""
    out = sort_panel(df, group_cols)

    if "PRCP_mm" not in out.columns:
        raise ValueError("PRCP_mm is required to create rain_target.")

    if task_mode == "same_day_classification":
        out["target_prcp_mm"] = out["PRCP_mm"]
        out["target_time"] = out["time"]
        registry.created("rain_target", "target", f"Same-day binary target: PRCP_mm > {threshold_mm} mm.")

    elif task_mode == "next_day_forecast":
        if group_cols:
            out["target_prcp_mm"] = out.groupby(group_cols)["PRCP_mm"].shift(-1)
            out["target_time"] = out.groupby(group_cols)["time"].shift(-1)
        else:
            out["target_prcp_mm"] = out["PRCP_mm"].shift(-1)
            out["target_time"] = out["time"].shift(-1)

        registry.created("rain_target", "target", f"Next-day binary target: PRCP_mm(t+1) > {threshold_mm} mm.")

    else:
        raise ValueError(f"Unsupported task_mode: {task_mode}")

    out["rain_target"] = np.where(
        out["target_prcp_mm"].notna(),
        (out["target_prcp_mm"] > threshold_mm).astype(int),
        np.nan,
    )

    before = len(out)
    out = out.dropna(subset=["rain_target"]).copy()
    after = len(out)

    out["rain_target"] = out["rain_target"].astype(int)

    if before != after:
        warnings.warn(f"Dropped {before - after:,} rows with missing target.")

    return out


# =============================================================================
# TEMPORAL NAN HANDLING
# =============================================================================

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

    report = pd.DataFrame(rows)
    if not report.empty:
        report = report.sort_values("missing_rate", ascending=False)

    report.to_csv(REPORT_DIR / "temporal_nan_report.csv", index=False)
    return report


def handle_temporal_nans(
    df: pd.DataFrame,
    group_cols: List[str],
    temporal_cols: List[str],
    policy: str = TEMPORAL_NAN_POLICY,
    add_missing_indicators: bool = ADD_MISSING_INDICATORS,
) -> pd.DataFrame:
    """
    Handle NaNs from lag/diff features.

    Default:
        drop_temporal_na_rows and add_missing_indicators=False
    This avoids creating many 0/1 *_is_missing columns.
    """
    out = df.copy()
    temporal_cols = [col for col in temporal_cols if col in out.columns]

    if add_missing_indicators:
        for col in temporal_cols:
            if out[col].isna().any():
                out[f"{col}_is_missing"] = out[col].isna().astype(int)

    if policy == "keep_nan":
        return out

    if policy == "drop_temporal_na_rows":
        before = len(out)
        out = out.dropna(subset=temporal_cols).reset_index(drop=True)
        after = len(out)
        print(f"Dropped {before - after:,} rows with temporal NaNs from lag/diff features.")
        return out

    if policy == "group_median_fill":
        for col in temporal_cols:
            if not out[col].isna().any():
                continue

            if group_cols:
                out[col] = out.groupby(group_cols)[col].transform(lambda x: x.fillna(x.median()))

            global_median = out[col].median()
            out[col] = out[col].fillna(global_median)

        return out

    raise ValueError(f"Unsupported temporal NaN policy: {policy}")


# =============================================================================
# FINAL DATASET CREATION
# =============================================================================

def create_final_feature_dataset(
    df: pd.DataFrame,
    registry: FeatureRegistry,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create the single final feature file.

    The returned dataframe is directly used by Step 7:
    - keeps time/STATION/rain_target
    - keeps numeric engineered predictors
    - removes leakage/audit columns
    - removes constant predictors
    - removes binary predictor columns by default
    """
    out = df.copy()
    dropped_rows: List[Dict[str, str]] = []

    def mark_drop(col: str, reason: str) -> None:
        dropped_rows.append({"dropped_column": col, "reason": reason})
        registry.dropped(col, "final_feature_selection", reason)

    drop_cols = []

    for col in out.columns:
        if col in ALWAYS_KEEP_COLS:
            continue

        if col in LEAKAGE_AND_AUDIT_COLS:
            drop_cols.append(col)
            mark_drop(col, "Leakage/audit/raw target amount or redundant calendar column.")
            continue

        if not KEEP_LOCATION_FEATURES and col in LOCATION_COLS:
            drop_cols.append(col)
            mark_drop(col, "Location feature removed because KEEP_LOCATION_FEATURES=False.")
            continue

        if out[col].isna().all():
            drop_cols.append(col)
            mark_drop(col, "All values are missing.")
            continue

        if pd.api.types.is_object_dtype(out[col]) or pd.api.types.is_categorical_dtype(out[col]):
            drop_cols.append(col)
            mark_drop(col, "Non-numeric column not used as model feature.")
            continue

        if DROP_CONSTANT_FEATURES and col not in ALWAYS_KEEP_COLS:
            if pd.api.types.is_numeric_dtype(out[col]) and out[col].nunique(dropna=True) <= 1:
                drop_cols.append(col)
                mark_drop(col, "Constant or near-empty feature; no predictive information.")
                continue

        if DROP_BINARY_PREDICTOR_FEATURES and col not in ALWAYS_KEEP_COLS:
            if pd.api.types.is_numeric_dtype(out[col]) and is_binary_series(out[col]):
                drop_cols.append(col)
                mark_drop(col, "Binary predictor removed to keep only rain_target as 0/1 by default.")
                continue

    drop_cols = sorted(set(c for c in drop_cols if c in out.columns))
    out = out.drop(columns=drop_cols, errors="ignore")

    # Keep final order: time, STATION, rain_target, then numeric features.
    ordered_cols = []
    for col in ["time", "STATION", "rain_target"]:
        if col in out.columns:
            ordered_cols.append(col)

    feature_cols = [c for c in out.columns if c not in ordered_cols]
    numeric_feature_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(out[c])]
    out = out[ordered_cols + numeric_feature_cols].copy()

    sort_cols = [c for c in ["STATION", "time"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)

    dropped_df = pd.DataFrame(dropped_rows).drop_duplicates()
    dropped_df.to_csv(REPORT_DIR / "dropped_columns_final_feature_file.csv", index=False)

    return out, dropped_df


# =============================================================================
# VALIDATION AND REPORTING
# =============================================================================

def validate_feature_engineered_data(
    full_df: pd.DataFrame,
    final_df: pd.DataFrame,
    group_cols: List[str],
    task_mode: str,
) -> None:
    """Final sanity checks."""
    if "rain_target" not in final_df.columns:
        raise AssertionError("rain_target was not created in final feature dataset.")

    target_values = set(final_df["rain_target"].dropna().unique().tolist())
    if not target_values.issubset({0, 1}):
        raise AssertionError(f"rain_target must be binary 0/1. Found: {target_values}")

    duplicate_keys = [*group_cols, "time"] if group_cols else ["time"]
    duplicate_count = int(final_df.duplicated(subset=duplicate_keys).sum())
    if duplicate_count > 0:
        raise AssertionError(f"Final feature data has {duplicate_count:,} duplicated rows by {duplicate_keys}.")

    if "PRCP" in full_df.columns:
        unique_prcp = full_df["PRCP"].dropna().unique()
        if len(unique_prcp) <= 2 and set(unique_prcp).issubset({0, 1}):
            raise AssertionError(
                "PRCP appears to be binary. This is wrong for this design. "
                "PRCP must remain continuous before final leakage columns are dropped."
            )

    if task_mode == "same_day_classification" and "PRCP_mm" in full_df.columns:
        mismatch = int(
            ((full_df["PRCP_mm"] > PRECIPITATION_THRESHOLD_MM).astype(int) != full_df["rain_target"]).sum()
        )
        if mismatch > 0:
            raise AssertionError(f"rain_target mismatch with PRCP_mm threshold in {mismatch:,} rows.")

    # In final output, only rain_target should be binary by default.
    binary_cols = []
    for col in final_df.columns:
        if col in {"time", "STATION"}:
            continue
        if is_binary_series(final_df[col]):
            binary_cols.append(col)

    pd.DataFrame({"binary_column": binary_cols}).to_csv(
        REPORT_DIR / "binary_columns_report.csv",
        index=False,
    )

    if DROP_BINARY_PREDICTOR_FEATURES:
        extra_binary = [c for c in binary_cols if c != "rain_target"]
        if extra_binary:
            raise AssertionError(
                "Final feature dataset still has binary predictor columns: "
                + ", ".join(extra_binary)
            )

    missing_report = (
        final_df.isna()
        .mean()
        .rename("missing_rate")
        .reset_index()
        .rename(columns={"index": "column"})
        .sort_values("missing_rate", ascending=False)
    )
    missing_report.to_csv(REPORT_DIR / "feature_missing_rate.csv", index=False)

    if missing_report["missing_rate"].max() > 0:
        warnings.warn("Final feature dataset still contains missing values. Check feature_missing_rate.csv.")


def save_reports_and_metadata(
    full_df: pd.DataFrame,
    final_df: pd.DataFrame,
    registry: FeatureRegistry,
    group_cols: List[str],
    temporal_cols: List[str],
    task_mode: str,
    input_path: Path,
    feature_output_path: Path,
) -> None:
    ensure_directories()

    registry.to_frame().to_csv(REPORT_DIR / "feature_registry.csv", index=False)

    target_dist = (
        final_df["rain_target"]
        .value_counts(dropna=False)
        .rename_axis("rain_target")
        .reset_index(name="count")
    )
    target_dist["rate"] = target_dist["count"] / len(final_df)
    target_dist.to_csv(REPORT_DIR / "target_distribution.csv", index=False)

    schema = pd.DataFrame({
        "column": final_df.columns,
        "dtype": [str(final_df[c].dtype) for c in final_df.columns],
        "missing_rate": [float(final_df[c].isna().mean()) for c in final_df.columns],
        "n_unique": [int(final_df[c].nunique(dropna=True)) for c in final_df.columns],
    })
    schema.to_csv(REPORT_DIR / "feature_schema.csv", index=False)

    used_features = [
        c for c in final_df.columns
        if c not in {"time", "STATION", "rain_target"}
    ]
    pd.DataFrame({"feature": used_features}).to_csv(REPORT_DIR / "used_feature_columns.csv", index=False)

    binary_cols = []
    for col in final_df.columns:
        if col in {"time", "STATION"}:
            continue
        if is_binary_series(final_df[col]):
            binary_cols.append(col)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(input_path),
        "feature_output_path": str(feature_output_path),
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
        "add_missing_indicators": ADD_MISSING_INDICATORS,
        "create_monsoon_binary_feature": CREATE_MONSOON_BINARY_FEATURE,
        "drop_constant_features": DROP_CONSTANT_FEATURES,
        "drop_binary_predictor_features": DROP_BINARY_PREDICTOR_FEATURES,
        "keep_location_features": KEEP_LOCATION_FEATURES,
        "binary_columns_in_final_output": binary_cols,
        "target_policy": (
            "PRCP and PRCP_mm are kept only internally for target creation and lag features. "
            "They are dropped from final feature_engineered_data.csv to avoid leakage. "
            "Only rain_target remains as the binary label."
        ),
        "output_policy": (
            "Only one feature file is produced: feature_engineered_data.csv. "
            "This file is already model-ready for Step 7."
        ),
        "row_policy": (
            "Rows with temporal lag/diff NaNs are dropped by default. "
            "For 5 stations and 2-day lags, this usually removes about 10 rows."
        ),
        "n_rows_internal_full": int(len(full_df)),
        "n_columns_internal_full": int(full_df.shape[1]),
        "n_rows_final": int(len(final_df)),
        "n_columns_final": int(final_df.shape[1]),
        "n_used_features": int(len(used_features)),
        "reports": {
            "feature_registry": str(REPORT_DIR / "feature_registry.csv"),
            "target_distribution": str(REPORT_DIR / "target_distribution.csv"),
            "feature_schema": str(REPORT_DIR / "feature_schema.csv"),
            "used_feature_columns": str(REPORT_DIR / "used_feature_columns.csv"),
            "feature_missing_rate": str(REPORT_DIR / "feature_missing_rate.csv"),
            "temporal_nan_report": str(REPORT_DIR / "temporal_nan_report.csv"),
            "binary_columns_report": str(REPORT_DIR / "binary_columns_report.csv"),
            "dropped_columns": str(REPORT_DIR / "dropped_columns_final_feature_file.csv"),
        },
    }

    _write_json(metadata, REPORT_DIR / "feature_engineering_metadata.json")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def feature_engineering(
    input_path: Path | str = INPUT_PATH,
    feature_output_path: Path | str = FEATURE_OUTPUT_PATH,
    task_mode: str = TASK_MODE,
    temporal_nan_policy: str = TEMPORAL_NAN_POLICY,
) -> pd.DataFrame:
    """Run full Step 6 feature engineering and produce one final feature file."""
    validate_config(task_mode, temporal_nan_policy)
    ensure_directories()

    input_path = Path(input_path)
    feature_output_path = Path(feature_output_path)

    print("\n=== STEP 6: SINGLE-OUTPUT LEAKAGE-AWARE FEATURE ENGINEERING ===")
    print(f"Task mode: {task_mode}")
    print(f"Input: {input_path}")
    print(f"Output: {feature_output_path}")
    print("Output policy: feature_engineered_data.csv is already model-ready.")

    df = load_silver_data(input_path)
    group_cols = infer_groupby_columns(df)
    registry = FeatureRegistry()

    df = create_temporal_features(df, registry)
    df = create_thermodynamic_features(df, registry)
    df, wind_temporal_cols = create_wind_features(df, group_cols, registry)
    df = create_geometric_features(df, registry)
    df = create_flux_features(df, registry)

    temporal_cols: List[str] = []
    temporal_cols.extend(wind_temporal_cols)

    df, lag_cols = create_lag_features(df, group_cols, registry)
    temporal_cols.extend(lag_cols)

    df, rolling_cols = create_rolling_statistics(df, group_cols, registry, task_mode=task_mode)
    temporal_cols.extend(rolling_cols)

    temporal_cols = sorted(set(c for c in temporal_cols if c in df.columns))

    df = create_target(df, group_cols, registry, task_mode=task_mode)

    report_temporal_nans(df, temporal_cols)
    df = handle_temporal_nans(df, group_cols, temporal_cols, policy=temporal_nan_policy)

    final_df, _ = create_final_feature_dataset(df, registry)

    validate_feature_engineered_data(
        full_df=df,
        final_df=final_df,
        group_cols=group_cols,
        task_mode=task_mode,
    )

    feature_output_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(feature_output_path, index=False)

    save_reports_and_metadata(
        full_df=df,
        final_df=final_df,
        registry=registry,
        group_cols=group_cols,
        temporal_cols=temporal_cols,
        task_mode=task_mode,
        input_path=input_path,
        feature_output_path=feature_output_path,
    )

    print(f"\n[SUCCESS] Final feature dataset saved to: {feature_output_path}")
    print(f"          Shape: {final_df.shape}")
    print(f"[SUCCESS] Reports saved to: {REPORT_DIR}")

    return final_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Step 6 single-output feature engineering.")
    parser.add_argument("--input", type=str, default=str(INPUT_PATH), help="Input silver CSV path.")
    parser.add_argument("--output", type=str, default=str(FEATURE_OUTPUT_PATH), help="Final feature CSV path.")
    parser.add_argument("--task-mode", type=str, default=TASK_MODE, choices=sorted(VALID_TASK_MODES))
    parser.add_argument(
        "--temporal-nan-policy",
        type=str,
        default=TEMPORAL_NAN_POLICY,
        choices=sorted(VALID_TEMPORAL_NAN_POLICIES),
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    feature_engineering(
        input_path=args.input,
        feature_output_path=args.output,
        task_mode=args.task_mode,
        temporal_nan_policy=args.temporal_nan_policy,
    )
