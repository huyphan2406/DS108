"""
Step 5 - Silver Layer Integration for DS108 Rainfall Project.

This script builds a clean Silver dataset by integrating:
1. GSOD station observations
2. ERA5 single-level data
3. ERA5 pressure-level data
4. ENSO / MEI climate index

Design goals:
- Produce a clean modeling/feature-engineering input:
    data/clean/silver_data.csv
- Keep audit/source information in reports, not in the final Silver CSV.
- Avoid leaking technical flags into downstream modeling.
- Preserve methodological reports for data quality, missingness, imputation,
  overlap comparison, and merge coverage.

Recommended project structure:
DS108/
├── data/
│   ├── raw/
│   │   ├── gsod/bronze_data.csv
│   │   ├── single/...
│   │   ├── pressure/...
│   │   └── enso/meiv2.data
│   └── clean/
│       ├── ERA5_single_level.parquet
│       ├── ERA5_pressure_final.parquet
│       ├── enso_clean.csv
│       └── silver_data.csv
├── reports/
│   └── data_quality/silver/
└── src/
    └── 05_to_silver.py
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


# =============================================================================
# CONFIGURATION
# =============================================================================

DATA_START_DATE = "2015-01-01"
DATA_END_DATE = "2024-12-31"
GRID_RESOLUTION = 0.25

STATIONARY_COLS = ["DATE", "LATITUDE", "LONGITUDE", "ELEVATION"]
WEATHER_COLS = ["TEMP", "PRCP", "WDSP", "DEWP", "STP", "SLP", "VISIB"]
ALL_COLS = ["STATION"] + STATIONARY_COLS + WEATHER_COLS

# GSOD missing/error codes.
ROGUE_MAPPING = {
    "PRCP": 99.99,
    "VISIB": 999.9,
    "WDSP": 999.9,
    "TEMP": 9999.9,
    "DEWP": 9999.9,
    "STP": 9999.9,
    "SLP": 9999.9,
}

# GSOD units -> metric units.
# Formula: output = (input + offset) * factor
UNIT_CONVERSIONS = {
    "TEMP": {"factor": 5 / 9, "offset": -32, "unit": "°C from °F"},
    "DEWP": {"factor": 5 / 9, "offset": -32, "unit": "°C from °F"},
    "PRCP": {"factor": 25.4, "offset": 0, "unit": "mm from inches"},
    "WDSP": {"factor": 0.514444, "offset": 0, "unit": "m/s from knots"},
    "VISIB": {"factor": 1.60934, "offset": 0, "unit": "km from miles"},
}

# Physical / sanity ranges after conversion.
PRESSURE_MIN, PRESSURE_MAX = 800, 1100
TEMP_MIN_C, TEMP_MAX_C = -20, 55
PRCP_MIN_MM = 0
PRCP_EXTREME_WARNING_MM = 500

# ERA5 single-level columns used to fill GSOD-like weather variables.
ERA5_FILL_MAPPING = {
    "TEMP": "t2m",
    "PRCP": "tp",
    "DEWP": "d2m",
    "STP": "sp",
    "SLP": "msl",
}

FILL_WDSP_FROM_ERA5_UV = True

# VISIB has no direct ERA5 equivalent in this project, so we interpolate
# only within the same station/grid group.
GROUPWISE_INTERPOLATION_COLS = ["VISIB"]
GROUPBY_GRID_COLS = ["STATION", "latitude", "longitude"]

# Keep useful ERA5 auxiliary variables in final Silver.
# sst is often all missing for inland station points and will be dropped if all missing.
AUXILIARY_ERA5_COLS_TO_KEEP = ["u10", "v10", "z", "lsm"]

# Duplicate ERA5 source columns to drop after their values are used for filling.
DUPLICATE_ERA5_SOURCE_COLS = ["t2m", "d2m", "tp", "sp", "msl"]

# Final audit/source/technical columns to remove from Silver final.
FINAL_AUDIT_COLS_TO_DROP = [
    "has_gsod_record",
    "TEMP_source",
    "PRCP_source",
    "DEWP_source",
    "STP_source",
    "SLP_source",
    "WDSP_source",
    "VISIB_source",
    "source_period",
    "MEI_WINDOW",
    "YEAR",
    "MONTH",
]

PRESSURE_FILE_CANDIDATES = [
    "ERA5_pressure_final.parquet",
    "ERA5_Pressure_Final.parquet",
]

ENSO_FILE_NAME = "enso_clean.csv"
SILVER_OUTPUT_NAME = "silver_data.csv"


# =============================================================================
# PATH HELPERS
# =============================================================================

def find_project_root(start: Optional[Path] = None) -> Path:
    """
    Find project root by walking upward until a `data` directory is found.
    Works whether this script is placed in project root or in src/.
    """
    if start is None:
        start = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

    candidates = [start, *start.parents]
    for path in candidates:
        if (path / "data").exists():
            return path

    return start


BASE_DIR = find_project_root()

RAW_DIR = BASE_DIR / "data" / "raw"
CLEAN_DIR = BASE_DIR / "data" / "clean"
REPORT_ROOT_DIR = BASE_DIR / "reports" / "data_quality"

BRONZE_GSOD_PATH = RAW_DIR / "gsod" / "bronze_data.csv"
ERA5_SINGLE_PATH = CLEAN_DIR / "ERA5_single_level.parquet"
ENSO_PATH = CLEAN_DIR / ENSO_FILE_NAME
SILVER_OUTPUT_PATH = CLEAN_DIR / SILVER_OUTPUT_NAME

REPORT_DIR = REPORT_ROOT_DIR / "silver"
PLOT_DIR = REPORT_DIR / "plots"


def ensure_directories() -> None:
    """Create output/report folders."""
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)


def resolve_pressure_path() -> Path:
    """Find pressure-level final parquet using supported filenames."""
    for name in PRESSURE_FILE_CANDIDATES:
        path = CLEAN_DIR / name
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not find pressure-level parquet. Expected one of: "
        + ", ".join(str(CLEAN_DIR / name) for name in PRESSURE_FILE_CANDIDATES)
    )


# =============================================================================
# BASIC UTILITIES
# =============================================================================

def _safe_to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def _write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _missing_rate_by_column(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.isna()
        .mean()
        .rename("missing_rate")
        .reset_index()
        .rename(columns={"index": "column"})
        .sort_values("missing_rate", ascending=False)
    )


def _numeric_summary(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce")
        rows.append({
            "column": col,
            "count": int(s.notna().sum()),
            "missing_count": int(s.isna().sum()),
            "missing_rate": float(s.isna().mean()),
            "mean": float(s.mean()) if s.notna().any() else np.nan,
            "median": float(s.median()) if s.notna().any() else np.nan,
            "std": float(s.std()) if s.notna().any() else np.nan,
            "min": float(s.min()) if s.notna().any() else np.nan,
            "p05": float(s.quantile(0.05)) if s.notna().any() else np.nan,
            "p95": float(s.quantile(0.95)) if s.notna().any() else np.nan,
            "max": float(s.max()) if s.notna().any() else np.nan,
        })

    return pd.DataFrame(rows)


def _aggregate_duplicate_era5_rows(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Ensure ERA5 dataframe has unique time/latitude/longitude keys.
    Duplicate keys are collapsed with mean for numeric variables.
    """
    keys = ["time", "latitude", "longitude"]
    duplicate_count = int(df.duplicated(subset=keys).sum())

    if duplicate_count == 0:
        return df

    warnings.warn(
        f"{name} has {duplicate_count:,} duplicate rows by {keys}. "
        "Collapsing numeric variables by mean."
    )

    numeric_cols = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col not in ["latitude", "longitude"]
    ]

    agg_map = {col: "mean" for col in numeric_cols}
    for col in df.columns:
        if col not in keys and col not in agg_map:
            agg_map[col] = "first"

    return df.groupby(keys, as_index=False).agg(agg_map)


# =============================================================================
# STEP 5.1 - GSOD CLEANING
# =============================================================================

def _ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all expected GSOD columns exist and keep them in a stable order."""
    for col in ALL_COLS:
        if col not in df.columns:
            df[col] = np.nan

    return df[ALL_COLS].copy()


def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate records per station-date."""
    df = df.copy()
    df["DATE"] = _safe_to_datetime(df["DATE"])

    before = len(df)
    df = df.drop_duplicates(subset=["STATION", "DATE"]).reset_index(drop=True)
    after = len(df)

    if before != after:
        print(f"Removed {before - after:,} duplicated GSOD station-date rows.")

    return df


def _remove_rogue_values(df: pd.DataFrame) -> pd.DataFrame:
    """Replace GSOD error codes with NaN."""
    out = df.copy()

    for col, error_val in ROGUE_MAPPING.items():
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out.loc[np.isclose(out[col], error_val, atol=0.001), col] = np.nan

    return out


def _validate_pressure(df: pd.DataFrame) -> pd.DataFrame:
    """Replace physically invalid pressure values with NaN."""
    out = df.copy()

    if "STP" in out.columns:
        out.loc[~out["STP"].between(PRESSURE_MIN, PRESSURE_MAX), "STP"] = np.nan

    if "SLP" in out.columns:
        out.loc[~out["SLP"].between(PRESSURE_MIN, PRESSURE_MAX), "SLP"] = np.nan

    return out


def _convert_to_metric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert GSOD variables to metric units."""
    out = df.copy()

    for col, conversion in UNIT_CONVERSIONS.items():
        if col in out.columns:
            out[col] = (out[col] + conversion["offset"]) * conversion["factor"]

    return out


def _basic_physical_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Apply minimal physical sanity checks after unit conversion."""
    out = df.copy()

    if "PRCP" in out.columns:
        out.loc[out["PRCP"] < PRCP_MIN_MM, "PRCP"] = np.nan

    if "TEMP" in out.columns:
        out.loc[~out["TEMP"].between(TEMP_MIN_C, TEMP_MAX_C), "TEMP"] = np.nan

    if "DEWP" in out.columns:
        out.loc[~out["DEWP"].between(TEMP_MIN_C, TEMP_MAX_C), "DEWP"] = np.nan

    if {"TEMP", "DEWP"}.issubset(out.columns):
        out.loc[out["DEWP"] > out["TEMP"] + 2.0, "DEWP"] = np.nan

    return out


def load_and_clean_gsod(path: Path = BRONZE_GSOD_PATH) -> pd.DataFrame:
    """Load and standardize GSOD station observations."""
    if not path.exists():
        raise FileNotFoundError(f"GSOD bronze file not found: {path}")

    print("\n=== STEP 5.1: Loading and standardizing GSOD observations ===")

    df = pd.read_csv(path)
    df = _ensure_required_columns(df)
    df["STATION"] = df["STATION"].astype(str)

    df = _remove_duplicates(df)
    df = _remove_rogue_values(df)
    df = _validate_pressure(df)
    df = _convert_to_metric(df)
    df = _basic_physical_cleaning(df)

    # Keep this only for audit/reporting. It is removed from final Silver.
    df["has_gsod_record"] = True

    return df


# Backward-compatible alias from the reference script.
def gsod(path: str | Path | None = None) -> pd.DataFrame:
    return load_and_clean_gsod(Path(path) if path is not None else BRONZE_GSOD_PATH)


# =============================================================================
# STEP 5.2 - MISSINGNESS DIAGNOSIS
# =============================================================================

def prepare_station_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Rename DATE to time and quantize station coordinates to ERA5 0.25° grid."""
    out = df.copy()
    out = out.rename(columns={"DATE": "time"})
    out["time"] = _safe_to_datetime(out["time"])
    out["latitude"] = ((out["LATITUDE"] / GRID_RESOLUTION).round() * GRID_RESOLUTION).round(2)
    out["longitude"] = ((out["LONGITUDE"] / GRID_RESOLUTION).round() * GRID_RESOLUTION).round(2)
    return out


def diagnose_missingness_before_imputation(df: pd.DataFrame) -> None:
    """Save missingness diagnostics before any imputation."""
    ensure_directories()
    print("\n=== STEP 5.2: Diagnosing missingness before imputation ===")

    working = prepare_station_grid(df)
    weather_cols = [col for col in WEATHER_COLS if col in working.columns]

    _missing_rate_by_column(working).to_csv(
        REPORT_DIR / "missing_rate_before_imputation_by_column.csv",
        index=False,
    )

    station_rows = []
    for station, group in working.groupby("STATION", dropna=False):
        row = {"STATION": str(station), "n_rows": int(len(group))}
        for col in weather_cols:
            row[f"{col}_missing_rate"] = float(group[col].isna().mean())
        station_rows.append(row)

    pd.DataFrame(station_rows).to_csv(
        REPORT_DIR / "missing_rate_before_imputation_by_station.csv",
        index=False,
    )

    working["YEAR"] = working["time"].dt.year
    working["MONTH"] = working["time"].dt.month

    month_rows = []
    for (year, month), group in working.groupby(["YEAR", "MONTH"], dropna=False):
        if pd.isna(year) or pd.isna(month):
            continue

        row = {"YEAR": int(year), "MONTH": int(month), "n_rows": int(len(group))}
        for col in weather_cols:
            row[f"{col}_missing_rate"] = float(group[col].isna().mean())
        month_rows.append(row)

    pd.DataFrame(month_rows).to_csv(
        REPORT_DIR / "missing_rate_before_imputation_by_month.csv",
        index=False,
    )

    _numeric_summary(working, weather_cols).to_csv(
        REPORT_DIR / "weather_numeric_summary_before_imputation.csv",
        index=False,
    )


# =============================================================================
# STEP 5.3 - COMPLETE PANEL AND SINGLE-LEVEL MERGE
# =============================================================================

def build_complete_station_date_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Create one row per station per day for the full study period."""
    ensure_directories()
    print("\n=== STEP 5.3: Building complete station-date panel ===")

    prepared = prepare_station_grid(df)
    full_dates = pd.date_range(DATA_START_DATE, DATA_END_DATE, freq="D")

    station_meta = (
        prepared[["STATION", "LATITUDE", "LONGITUDE", "ELEVATION", "latitude", "longitude"]]
        .drop_duplicates(subset=["STATION"])
        .reset_index(drop=True)
    )

    if station_meta.empty:
        raise ValueError("No station metadata found after GSOD cleaning.")

    panels = []
    for _, row in station_meta.iterrows():
        panel = pd.DataFrame({"time": full_dates})
        for col in station_meta.columns:
            panel[col] = row[col]
        panels.append(panel)

    panel_df = pd.concat(panels, ignore_index=True)

    obs_cols = ["STATION", "time", "has_gsod_record"] + [
        col for col in WEATHER_COLS if col in prepared.columns
    ]

    observed = prepared[obs_cols].copy()

    complete = pd.merge(
        panel_df,
        observed,
        on=["STATION", "time"],
        how="left",
    )

    complete["has_gsod_record"] = complete["has_gsod_record"].fillna(False).astype(bool)

    duplicate_count = int(complete.duplicated(subset=["STATION", "time"]).sum())
    if duplicate_count > 0:
        raise AssertionError(f"Complete station-date panel has {duplicate_count} duplicate rows.")

    panel_report = {
        "start_date": DATA_START_DATE,
        "end_date": DATA_END_DATE,
        "n_stations": int(complete["STATION"].nunique()),
        "n_rows": int(len(complete)),
        "expected_rows": int(len(full_dates) * complete["STATION"].nunique()),
        "gsod_record_coverage_rate": float(complete["has_gsod_record"].mean()),
        "missing_station_day_count": int((~complete["has_gsod_record"]).sum()),
    }

    _write_json(panel_report, REPORT_DIR / "station_date_panel_report.json")

    return complete


def load_era5_single_level(path: Path = ERA5_SINGLE_PATH) -> pd.DataFrame:
    """Load ERA5 single-level daily dataset."""
    if not path.exists():
        raise FileNotFoundError(f"ERA5 single-level file not found: {path}")

    era5 = pd.read_parquet(path)
    if "time" not in era5.columns:
        raise ValueError("ERA5 single-level data must contain a `time` column.")

    era5["time"] = _safe_to_datetime(era5["time"])
    era5 = era5.dropna(subset=["time", "latitude", "longitude"]).copy()
    era5 = _aggregate_duplicate_era5_rows(era5, "ERA5 single-level")

    return era5


def merge_era5_single_level(panel: pd.DataFrame, era5: pd.DataFrame) -> pd.DataFrame:
    """Merge ERA5 single-level variables into the complete station-date panel."""
    print("\n=== STEP 5.4: Merging ERA5 single-level data ===")

    merged = pd.merge(
        panel,
        era5,
        on=["time", "latitude", "longitude"],
        how="left",
        suffixes=("", "_era5"),
    )

    single_cols = [c for c in era5.columns if c not in ["time", "latitude", "longitude"]]
    _missing_rate_by_column(merged[single_cols]).to_csv(
        REPORT_DIR / "single_level_missing_after_merge.csv",
        index=False,
    )

    return merged


# =============================================================================
# STEP 5.4 - OVERLAP COMPARISON
# =============================================================================

def compare_gsod_with_era5_overlap(df_with_era5: pd.DataFrame) -> pd.DataFrame:
    """Compare observed GSOD values with ERA5 values on overlapping days."""
    print("\n=== STEP 5.5: Comparing GSOD observations with ERA5 overlap ===")

    working = df_with_era5.copy()
    rows = []

    if {"u10", "v10"}.issubset(working.columns):
        working["era5_wind_speed_10m"] = np.sqrt(working["u10"] ** 2 + working["v10"] ** 2)

    mapping = dict(ERA5_FILL_MAPPING)
    if "era5_wind_speed_10m" in working.columns:
        mapping["WDSP"] = "era5_wind_speed_10m"

    for gsod_col, era5_col in mapping.items():
        if gsod_col not in working.columns or era5_col not in working.columns:
            continue

        pair = working[[gsod_col, era5_col]].dropna()

        if pair.empty:
            rows.append({
                "variable": gsod_col,
                "era5_column": era5_col,
                "n_overlap": 0,
            })
            continue

        diff = pair[era5_col] - pair[gsod_col]

        rows.append({
            "variable": gsod_col,
            "era5_column": era5_col,
            "n_overlap": int(len(pair)),
            "gsod_mean": float(pair[gsod_col].mean()),
            "era5_mean": float(pair[era5_col].mean()),
            "gsod_median": float(pair[gsod_col].median()),
            "era5_median": float(pair[era5_col].median()),
            "gsod_std": float(pair[gsod_col].std()),
            "era5_std": float(pair[era5_col].std()),
            "mean_bias_era5_minus_gsod": float(diff.mean()),
            "mae": float(np.abs(diff).mean()),
            "rmse": float(np.sqrt((diff ** 2).mean())),
            "pearson_corr": float(pair.corr(method="pearson").iloc[0, 1]) if len(pair) > 2 else np.nan,
            "spearman_corr": float(pair.corr(method="spearman").iloc[0, 1]) if len(pair) > 2 else np.nan,
        })

    report = pd.DataFrame(rows)
    report.to_csv(REPORT_DIR / "gsod_vs_era5_overlap_comparison.csv", index=False)

    return report


# =============================================================================
# STEP 5.5 - IMPUTATION
# =============================================================================

def fill_from_era5_and_interpolate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing GSOD-like variables using ERA5 and group-wise interpolation.

    Source flags are kept temporarily for audit reports, then dropped from final
    Silver output.
    """
    print("\n=== STEP 5.6: Filling missing values with ERA5 and group-wise interpolation ===")

    out = df.sort_values(GROUPBY_GRID_COLS + ["time"]).reset_index(drop=True).copy()

    if FILL_WDSP_FROM_ERA5_UV and {"u10", "v10"}.issubset(out.columns):
        out["era5_wind_speed_10m"] = np.sqrt(out["u10"] ** 2 + out["v10"] ** 2)

    fill_mapping = dict(ERA5_FILL_MAPPING)
    if "era5_wind_speed_10m" in out.columns:
        fill_mapping["WDSP"] = "era5_wind_speed_10m"

    for target_col, source_col in fill_mapping.items():
        if target_col not in out.columns:
            out[target_col] = np.nan

        if source_col not in out.columns:
            warnings.warn(f"ERA5 source column {source_col} not found. Cannot fill {target_col}.")
            out[f"{target_col}_source"] = np.where(out[target_col].notna(), "GSOD", "MISSING")
            continue

        observed_mask = out[target_col].notna()
        era5_mask = out[source_col].notna()

        out[f"{target_col}_source"] = np.select(
            [observed_mask, (~observed_mask) & era5_mask],
            ["GSOD", "ERA5"],
            default="MISSING",
        )

        out[target_col] = out[target_col].fillna(out[source_col])

    for col in GROUPWISE_INTERPOLATION_COLS:
        if col not in out.columns:
            continue

        original_notna = out[col].notna()

        out[col] = out.groupby(GROUPBY_GRID_COLS)[col].transform(
            lambda x: x.interpolate(method="linear", limit_direction="both")
        )

        out[f"{col}_source"] = np.select(
            [original_notna, (~original_notna) & out[col].notna()],
            ["GSOD", "GROUPWISE_LINEAR_INTERPOLATION"],
            default="MISSING",
        )

    return out


def compare_distribution_before_after_fill(
    before_panel: pd.DataFrame,
    after_fill: pd.DataFrame,
) -> pd.DataFrame:
    """Compare distributions before and after imputation."""
    print("\n=== STEP 5.7: Comparing distributions before and after imputation ===")

    rows = []
    variables = [col for col in WEATHER_COLS if col in after_fill.columns]

    for col in variables:
        before_values = (
            pd.to_numeric(before_panel[col], errors="coerce")
            if col in before_panel.columns
            else pd.Series(dtype=float)
        )
        after_values = pd.to_numeric(after_fill[col], errors="coerce")

        source_col = f"{col}_source"
        if source_col in after_fill.columns:
            filled_values = pd.to_numeric(
                after_fill.loc[after_fill[source_col] != "GSOD", col],
                errors="coerce",
            )
            era5_values = pd.to_numeric(
                after_fill.loc[after_fill[source_col] == "ERA5", col],
                errors="coerce",
            )
        else:
            filled_values = pd.Series(dtype=float)
            era5_values = pd.Series(dtype=float)

        rows.append({
            "variable": col,
            "before_observed_count": int(before_values.notna().sum()),
            "after_count": int(after_values.notna().sum()),
            "filled_non_gsod_count": int(filled_values.notna().sum()),
            "era5_filled_count": int(era5_values.notna().sum()),
            "before_mean": float(before_values.mean()) if before_values.notna().any() else np.nan,
            "after_mean": float(after_values.mean()) if after_values.notna().any() else np.nan,
            "filled_mean": float(filled_values.mean()) if filled_values.notna().any() else np.nan,
            "before_median": float(before_values.median()) if before_values.notna().any() else np.nan,
            "after_median": float(after_values.median()) if after_values.notna().any() else np.nan,
            "filled_median": float(filled_values.median()) if filled_values.notna().any() else np.nan,
            "before_std": float(before_values.std()) if before_values.notna().any() else np.nan,
            "after_std": float(after_values.std()) if after_values.notna().any() else np.nan,
            "filled_std": float(filled_values.std()) if filled_values.notna().any() else np.nan,
            "before_p95": float(before_values.quantile(0.95)) if before_values.notna().any() else np.nan,
            "after_p95": float(after_values.quantile(0.95)) if after_values.notna().any() else np.nan,
            "filled_p95": float(filled_values.quantile(0.95)) if filled_values.notna().any() else np.nan,
            "before_max": float(before_values.max()) if before_values.notna().any() else np.nan,
            "after_max": float(after_values.max()) if after_values.notna().any() else np.nan,
            "filled_max": float(filled_values.max()) if filled_values.notna().any() else np.nan,
        })

    report = pd.DataFrame(rows)
    report.to_csv(REPORT_DIR / "distribution_before_after_imputation.csv", index=False)

    return report


def save_source_coverage_report(df_with_source_flags: pd.DataFrame) -> None:
    """
    Save how many values came from GSOD, ERA5, interpolation, or remain missing.

    This must be called before final source flags are dropped.
    """
    rows = []
    total = len(df_with_source_flags)

    for col in WEATHER_COLS:
        source_col = f"{col}_source"
        if source_col not in df_with_source_flags.columns:
            continue

        counts = df_with_source_flags[source_col].value_counts(dropna=False)

        for source, count in counts.items():
            rows.append({
                "variable": col,
                "source": str(source),
                "count": int(count),
                "rate": float(count / total) if total else np.nan,
            })

    pd.DataFrame(rows).to_csv(REPORT_DIR / "value_source_coverage.csv", index=False)


# =============================================================================
# STEP 5.6 - ENSO AND PRESSURE MERGE
# =============================================================================

def merge_enso_data(df: pd.DataFrame, enso_path: Path = ENSO_PATH) -> pd.DataFrame:
    """Merge ENSO by YEAR/MONTH using left join to avoid silently dropping rows."""
    if not enso_path.exists():
        raise FileNotFoundError(f"ENSO clean file not found: {enso_path}")

    print("\n=== STEP 5.8: Merging ENSO climate index ===")

    out = df.copy()
    out["YEAR"] = pd.to_datetime(out["time"], errors="coerce").dt.year
    out["MONTH"] = pd.to_datetime(out["time"], errors="coerce").dt.month

    enso = pd.read_csv(enso_path)

    required = {"YEAR", "MONTH"}
    if not required.issubset(enso.columns):
        raise ValueError(f"ENSO file must contain columns {required}.")

    # month_index is a technical helper; MEI_WINDOW is audit label and later dropped.
    enso_cols = [col for col in enso.columns if col not in ["month_index"]]
    enso = enso[enso_cols].copy()

    out = pd.merge(out, enso, how="left", on=["YEAR", "MONTH"])

    enso_value_cols = [col for col in out.columns if col.startswith("ENSO")]
    rows = []
    for col in enso_value_cols:
        rows.append({
            "column": col,
            "missing_after_merge_count": int(out[col].isna().sum()),
            "missing_after_merge_rate": float(out[col].isna().mean()),
        })

    pd.DataFrame(rows).to_csv(REPORT_DIR / "enso_missing_after_merge.csv", index=False)

    return out


def merge_pressure_data(df: pd.DataFrame) -> pd.DataFrame:
    """Merge ERA5 pressure-level variables by time/grid location."""
    print("\n=== STEP 5.9: Merging ERA5 pressure-level data ===")

    pressure_path = resolve_pressure_path()
    pressure = pd.read_parquet(pressure_path)

    if "time" not in pressure.columns:
        raise ValueError("Pressure-level data must contain a `time` column.")

    pressure["time"] = _safe_to_datetime(pressure["time"])
    pressure = pressure.dropna(subset=["time", "latitude", "longitude"]).copy()

    # Clip to the study period for consistency, especially if raw batch includes 2025.
    pressure = pressure[
        (pressure["time"] >= DATA_START_DATE) &
        (pressure["time"] <= DATA_END_DATE)
    ].copy()

    pressure = _aggregate_duplicate_era5_rows(pressure, "ERA5 pressure-level")

    out = pd.merge(
        df,
        pressure,
        how="left",
        on=["latitude", "longitude", "time"],
    )

    pressure_cols = [c for c in pressure.columns if c not in ["time", "latitude", "longitude"]]
    if pressure_cols:
        _missing_rate_by_column(out[pressure_cols]).to_csv(
            REPORT_DIR / "pressure_missing_after_merge.csv",
            index=False,
        )

    return out


# Backward-compatible alias from the reference script.
def merge_file(df: pd.DataFrame) -> pd.DataFrame:
    merged = merge_enso_data(df)
    merged = merge_pressure_data(merged)
    return finalize_silver_columns(merged)


# =============================================================================
# STEP 5.7 - FINAL CLEANUP, VALIDATION, METADATA
# =============================================================================

def create_drop_rationale_table() -> pd.DataFrame:
    """Create a transparent table explaining removed columns."""
    rows = []

    for col in DUPLICATE_ERA5_SOURCE_COLS:
        rows.append({
            "column": col,
            "action": "drop",
            "reason": (
                "Duplicate ERA5 source column after values are used to fill "
                "standardized GSOD-like weather columns."
            ),
        })

    rows.extend([
        {
            "column": "era5_wind_speed_10m",
            "action": "drop",
            "reason": "Temporary helper used to fill WDSP from u10/v10.",
        },
        {
            "column": "latitude",
            "action": "drop",
            "reason": "Temporary ERA5 grid latitude; station LATITUDE is retained.",
        },
        {
            "column": "longitude",
            "action": "drop",
            "reason": "Temporary ERA5 grid longitude; station LONGITUDE is retained.",
        },
        {
            "column": "month_index",
            "action": "drop",
            "reason": "Technical ENSO helper column.",
        },
        {
            "column": "*_source and has_gsod_record",
            "action": "drop from final Silver, report separately",
            "reason": (
                "Audit/source flags are useful for data-quality reports, but they "
                "are not meteorological variables and should not be used as model features."
            ),
        },
        {
            "column": "source_period",
            "action": "drop",
            "reason": "Raw ERA5 batch label, not a physical predictor.",
        },
        {
            "column": "MEI_WINDOW",
            "action": "drop",
            "reason": "ENSO bimonthly label; numeric ENSO variables are retained.",
        },
        {
            "column": "YEAR, MONTH",
            "action": "drop",
            "reason": "Redundant date components derivable from time.",
        },
        {
            "column": "all-missing columns, e.g. sst",
            "action": "drop",
            "reason": "No useful information for downstream feature engineering.",
        },
    ])

    table = pd.DataFrame(rows)
    table.to_csv(REPORT_DIR / "dropped_columns_rationale.csv", index=False)
    return table


def finalize_silver_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate/temp/audit/source columns and keep a clean Silver dataset.

    Final Silver should contain only:
    - time/station metadata
    - standardized GSOD-like weather variables
    - useful ERA5 single-level variables
    - ENSO numeric variables
    - ERA5 pressure-level variables
    """
    print("\n=== STEP 5.10: Finalizing clean Silver columns ===")

    create_drop_rationale_table()

    out = df.copy()

    cols_to_drop = (
        list(DUPLICATE_ERA5_SOURCE_COLS)
        + [
            "era5_wind_speed_10m",
            "latitude",
            "longitude",
            "month_index",
        ]
        + FINAL_AUDIT_COLS_TO_DROP
    )

    out = out.drop(columns=cols_to_drop, errors="ignore")

    # Drop columns that are completely missing, e.g. sst.
    all_missing_cols = [col for col in out.columns if out[col].isna().all()]
    if all_missing_cols:
        print(f"Dropping all-missing columns: {all_missing_cols}")
        out = out.drop(columns=all_missing_cols)

    if "STATION" in out.columns:
        out["STATION"] = out["STATION"].astype(str)

    if "time" in out.columns:
        out["time"] = _safe_to_datetime(out["time"])

    sort_cols = [c for c in ["STATION", "time"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)

    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].round(4)

    return out


def validate_silver_dataset(df: pd.DataFrame, strict: bool = True) -> None:
    """Run final structural and physical sanity checks."""
    print("\n=== STEP 5.11: Validating final silver dataset ===")

    if {"STATION", "time"}.issubset(df.columns):
        dup = int(df.duplicated(subset=["STATION", "time"]).sum())
        if dup > 0:
            msg = f"Final Silver dataset has {dup:,} duplicated STATION-time rows."
            if strict:
                raise AssertionError(msg)
            warnings.warn(msg)

    expected_days = len(pd.date_range(DATA_START_DATE, DATA_END_DATE, freq="D"))
    if {"STATION", "time"}.issubset(df.columns):
        n_stations = df["STATION"].nunique()
        expected_rows = expected_days * n_stations
        if len(df) != expected_rows:
            msg = (
                f"Final Silver should have {expected_rows:,} rows "
                f"({n_stations} stations × {expected_days} days), got {len(df):,}."
            )
            if strict:
                raise AssertionError(msg)
            warnings.warn(msg)

    if "PRCP" in df.columns:
        prcp = pd.to_numeric(df["PRCP"], errors="coerce")
        neg = int((prcp < 0).sum())
        extreme = int((prcp > PRCP_EXTREME_WARNING_MM).sum())

        if neg > 0:
            msg = f"Final Silver has {neg:,} negative precipitation values."
            if strict:
                raise AssertionError(msg)
            warnings.warn(msg)

        if extreme > 0:
            warnings.warn(
                f"Final Silver has {extreme:,} PRCP values > "
                f"{PRCP_EXTREME_WARNING_MM} mm. Inspect but not automatically removed."
            )

    if {"TEMP", "DEWP"}.issubset(df.columns):
        temp = pd.to_numeric(df["TEMP"], errors="coerce")
        dewp = pd.to_numeric(df["DEWP"], errors="coerce")
        violations = int((dewp > temp + 2.0).sum())
        if violations > 0:
            warnings.warn(
                f"Final Silver has {violations:,} rows where DEWP > TEMP + 2°C."
            )

    if {"z_500", "z_850"}.issubset(df.columns):
        z500 = pd.to_numeric(df["z_500"], errors="coerce")
        z850 = pd.to_numeric(df["z_850"], errors="coerce")
        bad_height = int((z500 <= z850).sum())
        if bad_height > 0:
            warnings.warn(f"Final Silver has {bad_height:,} rows where z_500 <= z_850.")

    final_missing = _missing_rate_by_column(df)
    final_missing.to_csv(REPORT_DIR / "missing_rate_final_silver.csv", index=False)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    _numeric_summary(df, numeric_cols).to_csv(
        REPORT_DIR / "numeric_summary_final_silver.csv",
        index=False,
    )


def save_silver_metadata(df: pd.DataFrame) -> None:
    """Save metadata explaining integration design and output structure."""
    pressure_path = None
    try:
        pressure_path = str(resolve_pressure_path())
    except FileNotFoundError:
        pressure_path = None

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(BASE_DIR),
        "input_files": {
            "bronze_gsod": str(BRONZE_GSOD_PATH),
            "era5_single_level": str(ERA5_SINGLE_PATH),
            "enso": str(ENSO_PATH),
            "pressure_level": pressure_path,
        },
        "output_file": str(SILVER_OUTPUT_PATH),
        "study_period": {
            "start": DATA_START_DATE,
            "end": DATA_END_DATE,
        },
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "n_stations": int(df["STATION"].nunique()) if "STATION" in df.columns else None,
        "columns": list(df.columns),
        "imputation_policy": {
            "GSOD_like_variables_filled_from_ERA5": ERA5_FILL_MAPPING,
            "WDSP_filled_from_ERA5_uv_wind_speed": FILL_WDSP_FROM_ERA5_UV,
            "VISIB_interpolation": (
                "Linear interpolation within each STATION-latitude-longitude group. "
                "No interpolation is performed across different stations."
            ),
            "audit_flags_policy": (
                "Source flags are saved in reports/data_quality/silver/"
                "value_source_coverage.csv, but removed from final silver_data.csv."
            ),
        },
        "final_column_policy": (
            "Final Silver keeps meteorological variables and station metadata only. "
            "Audit/source flags, raw batch labels, redundant YEAR/MONTH, technical "
            "merge columns, and all-missing columns are removed."
        ),
        "leakage_note": (
            "This Silver dataset is designed as a historical benchmark dataset. "
            "ERA5 reanalysis is acceptable for historical reconstruction and "
            "data rescue. If the task is real-time future forecasting, same-day "
            "ERA5 reanalysis variables should be replaced by forecast products "
            "or lagged/available-at-prediction-time features."
        ),
        "reports": {
            "station_date_panel": str(REPORT_DIR / "station_date_panel_report.json"),
            "missingness_before_imputation": str(REPORT_DIR / "missing_rate_before_imputation_by_column.csv"),
            "gsod_vs_era5_overlap": str(REPORT_DIR / "gsod_vs_era5_overlap_comparison.csv"),
            "distribution_before_after_imputation": str(REPORT_DIR / "distribution_before_after_imputation.csv"),
            "source_coverage": str(REPORT_DIR / "value_source_coverage.csv"),
            "enso_missing_after_merge": str(REPORT_DIR / "enso_missing_after_merge.csv"),
            "pressure_missing_after_merge": str(REPORT_DIR / "pressure_missing_after_merge.csv"),
            "final_missing_rate": str(REPORT_DIR / "missing_rate_final_silver.csv"),
            "drop_rationale": str(REPORT_DIR / "dropped_columns_rationale.csv"),
        },
    }

    _write_json(metadata, REPORT_DIR / "silver_metadata.json")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def build_silver_dataset(
    gsod_path: Path = BRONZE_GSOD_PATH,
    output_path: Path = SILVER_OUTPUT_PATH,
    strict_validation: bool = True,
) -> pd.DataFrame:
    """
    End-to-end Silver layer pipeline.

    Outputs:
    - data/clean/silver_data.csv
    - reports/data_quality/silver/*.csv
    - reports/data_quality/silver/*.json
    """
    ensure_directories()

    gsod_clean = load_and_clean_gsod(gsod_path)

    diagnose_missingness_before_imputation(gsod_clean)

    panel = build_complete_station_date_panel(gsod_clean)

    era5_single = load_era5_single_level(ERA5_SINGLE_PATH)
    panel_with_single = merge_era5_single_level(panel, era5_single)

    compare_gsod_with_era5_overlap(panel_with_single)

    filled = fill_from_era5_and_interpolate(panel_with_single)

    compare_distribution_before_after_fill(panel, filled)

    # Save source/audit coverage BEFORE dropping *_source flags from final Silver.
    save_source_coverage_report(filled)

    merged = merge_enso_data(filled)
    merged = merge_pressure_data(merged)

    final = finalize_silver_columns(merged)

    validate_silver_dataset(final, strict=strict_validation)
    save_silver_metadata(final)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(output_path, index=False)

    print(f"\n[SUCCESS] Silver dataset saved to: {output_path}")
    print(f"          Shape: {final.shape}")
    print(f"          Reports saved to: {REPORT_DIR}")

    return final


# Backward-compatible wrapper from the reference script.
def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wrapper for older call style. For the full pipeline and reports, use
    build_silver_dataset().
    """
    diagnose_missingness_before_imputation(df)
    panel = build_complete_station_date_panel(df)
    era5_single = load_era5_single_level(ERA5_SINGLE_PATH)
    panel_with_single = merge_era5_single_level(panel, era5_single)
    compare_gsod_with_era5_overlap(panel_with_single)
    filled = fill_from_era5_and_interpolate(panel_with_single)
    compare_distribution_before_after_fill(panel, filled)
    save_source_coverage_report(filled)
    return filled


def main() -> None:
    build_silver_dataset(
        gsod_path=BRONZE_GSOD_PATH,
        output_path=SILVER_OUTPUT_PATH,
        strict_validation=True,
    )


if __name__ == "__main__":
    main()
