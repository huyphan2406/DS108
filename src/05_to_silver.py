"""
Silver layer data integration and imputation module.

This module upgrades the original `_05_to_silver.py` pipeline with stronger
methodological rigor for a data-preprocessing course project.

Main improvements:
1. Fix VISIB interpolation so it is performed within each station/grid group,
   not across the entire dataframe.
2. Diagnose missingness BEFORE imputation:
   - missing rate by column
   - missing rate by station
   - missing rate by month
   - simple MCAR/MAR-oriented missingness signals
3. Fill missing GSOD values using ERA5 reanalysis while preserving source flags.
4. Compare GSOD observations and ERA5-filled values to check whether imputation
   distorts distributions:
   - mean / median / std before and after fill
   - filled-only distribution
   - GSOD vs ERA5 correlation on overlapping days
   - diagnostic plots
5. Add data leakage notes and metadata:
   ERA5 reanalysis is acceptable for constructing a historical benchmark dataset,
   but it must not be described as an operational future-forecast input unless
   replaced by real-time forecast products.
6. Keep a removal-rationale table for dropped columns.
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

DATA_START_DATE = "2015-01-01"
DATA_END_DATE = "2024-12-31"
GRID_RESOLUTION = 0.25

STATIONARY_COLS = ["DATE", "LATITUDE", "LONGITUDE", "ELEVATION"]
WEATHER_COLS = ["TEMP", "PRCP", "WDSP", "DEWP", "STP", "SLP", "VISIB"]
ALL_COLS = ["STATION"] + STATIONARY_COLS + WEATHER_COLS

# GSOD missing/error codes
ROGUE_MAPPING = {
    "PRCP": 99.99,
    "VISIB": 999.9,
    "WDSP": 999.9,
    "TEMP": 9999.9,
    "DEWP": 9999.9,
    "STP": 9999.9,
    "SLP": 9999.9,
}

# Unit conversion factors from GSOD units to metric units.
# Formula: output = (input + offset) * factor
UNIT_CONVERSIONS = {
    "TEMP": {"factor": 5 / 9, "offset": -32, "unit": "°C from °F"},
    "DEWP": {"factor": 5 / 9, "offset": -32, "unit": "°C from °F"},
    "PRCP": {"factor": 25.4, "offset": 0, "unit": "mm from inches"},
    "WDSP": {"factor": 0.514444, "offset": 0, "unit": "m/s from knots"},
    "VISIB": {"factor": 1.60934, "offset": 0, "unit": "km from miles"},
}

# Valid physical / sanity ranges.
PRESSURE_MIN, PRESSURE_MAX = 800, 1100
TEMP_MIN_C, TEMP_MAX_C = -20, 55
PRCP_MIN_MM = 0
PRCP_EXTREME_WARNING_MM = 500

# ERA5 single-level columns used to fill GSOD-like fields.
ERA5_FILL_MAPPING = {
    "TEMP": "t2m",
    "PRCP": "tp",
    "DEWP": "d2m",
    "STP": "sp",
    "SLP": "msl",
}

# ERA5 u/v wind can be used to fill missing GSOD WDSP.
FILL_WDSP_FROM_ERA5_UV = True

# Columns to interpolate because ERA5 has no direct equivalent in this pipeline.
GROUPWISE_INTERPOLATION_COLS = ["VISIB"]

# Columns used to define groups for temporal imputation.
GROUPBY_GRID_COLS = ["STATION", "latitude", "longitude"]

# Keep auxiliary ERA5 variables by default because they may be useful for EDA or
# feature engineering. Set to False if you want a smaller silver dataset.
KEEP_AUXILIARY_ERA5 = True
AUXILIARY_ERA5_COLS = ["u10", "v10", "sst", "z", "lsm"]

# Temporary / duplicate columns that can be removed after filling.
# These are not original GSOD variables; they are duplicate ERA5 source columns
# after their values have been used to fill standardized columns.
DUPLICATE_ERA5_SOURCE_COLS = ["t2m", "d2m", "tp", "sp", "msl"]

# Pressure-level final filenames can differ across your older/newer scripts.
PRESSURE_FILE_CANDIDATES = [
    "ERA5_pressure_final.parquet",
    "ERA5_Pressure_Final.parquet",
]

# ENSO output from step 4.
ENSO_FILE_NAME = "enso_clean.csv"

# Final output.
SILVER_OUTPUT_NAME = "silver_data_ver2.csv"


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
RAW_DIR = BASE_DIR / "data" / "raw"
CLEAN_DIR = BASE_DIR / "data" / "clean"
REPORT_DIR = BASE_DIR / "reports" / "data_quality" / "silver"

BRONZE_GSOD_PATH = RAW_DIR / "bronze_data.csv"
ERA5_SINGLE_PATH = CLEAN_DIR / "ERA5_single_level.parquet"
ENSO_PATH = CLEAN_DIR / ENSO_FILE_NAME
SILVER_OUTPUT_PATH = CLEAN_DIR / SILVER_OUTPUT_NAME


def ensure_directories() -> None:
    """Create output/report directories before writing files."""
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "plots").mkdir(parents=True, exist_ok=True)


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


# ============================================================================
# BASIC UTILITIES
# ============================================================================

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


# ============================================================================
# STEP 1: GSOD CLEANING
# ============================================================================

def _ensure_required_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure all expected GSOD columns exist."""
    for col in ALL_COLS:
        if col not in df.columns:
            df[col] = np.nan
    return df[ALL_COLS].copy()


def _remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate records per station-date."""
    df["DATE"] = _safe_to_datetime(df["DATE"])
    before = len(df)
    df = df.drop_duplicates(subset=["STATION", "DATE"]).reset_index(drop=True)
    after = len(df)

    if before != after:
        print(f"Removed {before - after:,} duplicated GSOD station-date rows.")

    return df


def _remove_rogue_values(df: pd.DataFrame) -> pd.DataFrame:
    """Replace GSOD error codes with NaN."""
    for col, error_val in ROGUE_MAPPING.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df.loc[np.isclose(df[col], error_val, atol=0.001), col] = np.nan
    return df


def _validate_pressure(df: pd.DataFrame) -> pd.DataFrame:
    """Replace physically invalid pressure values with NaN."""
    if "STP" in df.columns:
        df.loc[~df["STP"].between(PRESSURE_MIN, PRESSURE_MAX), "STP"] = np.nan
    if "SLP" in df.columns:
        df.loc[~df["SLP"].between(PRESSURE_MIN, PRESSURE_MAX), "SLP"] = np.nan
    return df


def _convert_to_metric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert GSOD variables to metric units."""
    for col, conversion in UNIT_CONVERSIONS.items():
        if col in df.columns:
            df[col] = (df[col] + conversion["offset"]) * conversion["factor"]
    return df


def _basic_physical_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply minimal physical sanity checks after unit conversion.
    These do not impute values; invalid values are marked as missing.
    """
    if "PRCP" in df.columns:
        df.loc[df["PRCP"] < PRCP_MIN_MM, "PRCP"] = np.nan
    if "TEMP" in df.columns:
        df.loc[~df["TEMP"].between(TEMP_MIN_C, TEMP_MAX_C), "TEMP"] = np.nan
    if "DEWP" in df.columns:
        df.loc[~df["DEWP"].between(TEMP_MIN_C, TEMP_MAX_C), "DEWP"] = np.nan
    if {"TEMP", "DEWP"}.issubset(df.columns):
        # Dew point should not be materially higher than air temperature.
        df.loc[df["DEWP"] > df["TEMP"] + 2.0, "DEWP"] = np.nan
    return df


def load_and_clean_gsod(path: Path = BRONZE_GSOD_PATH) -> pd.DataFrame:
    """
    Load and clean GSOD station observations.

    This stage standardizes raw GSOD records but does not impute missing values.
    """
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

    # Keep a flag for observed GSOD rows before building a complete panel.
    df["has_gsod_record"] = True

    return df


# Backward-compatible function name from the old script.
def gsod(path: str | Path | None = None) -> pd.DataFrame:
    return load_and_clean_gsod(Path(path) if path is not None else BRONZE_GSOD_PATH)


# ============================================================================
# STEP 2: MISSINGNESS DIAGNOSIS BEFORE IMPUTATION
# ============================================================================

def prepare_station_grid(df: pd.DataFrame) -> pd.DataFrame:
    """Rename DATE to time and quantize station coordinates to ERA5 grid."""
    out = df.copy()
    out = out.rename(columns={"DATE": "time"})
    out["time"] = _safe_to_datetime(out["time"])
    out["latitude"] = ((out["LATITUDE"] / GRID_RESOLUTION).round() * GRID_RESOLUTION).round(2)
    out["longitude"] = ((out["LONGITUDE"] / GRID_RESOLUTION).round() * GRID_RESOLUTION).round(2)
    return out


def diagnose_missingness_before_imputation(df: pd.DataFrame) -> None:
    """
    Save missingness diagnostics before any imputation is performed.

    This does not prove MCAR/MAR/MNAR completely, but it provides evidence:
    - missingness by variable
    - missingness by station
    - missingness by month
    - correlation between missing indicators and time/month/location signals
    """
    ensure_directories()
    print("\n=== STEP 5.2: Diagnosing missingness before imputation ===")

    working = prepare_station_grid(df)
    weather_cols = [col for col in WEATHER_COLS if col in working.columns]

    _missing_rate_by_column(working).to_csv(
        REPORT_DIR / "missing_rate_before_imputation_by_column.csv",
        index=False,
    )

    station_rows = []
    for station, group in working.groupby("STATION"):
        row = {"STATION": station, "n_rows": int(len(group))}
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
    for (year, month), group in working.groupby(["YEAR", "MONTH"]):
        row = {"YEAR": int(year), "MONTH": int(month), "n_rows": int(len(group))}
        for col in weather_cols:
            row[f"{col}_missing_rate"] = float(group[col].isna().mean())
        month_rows.append(row)

    pd.DataFrame(month_rows).to_csv(
        REPORT_DIR / "missing_rate_before_imputation_by_month.csv",
        index=False,
    )

    # Simple MAR-oriented signals: whether missingness correlates with month,
    # grid coordinates, or time index. This is diagnostic, not a formal proof.
    signal_df = working[["time", "MONTH", "latitude", "longitude"]].copy()
    signal_df["time_ordinal"] = working["time"].map(lambda x: x.toordinal() if pd.notna(x) else np.nan)

    rows = []
    for col in weather_cols:
        indicator = working[col].isna().astype(float)
        for signal in ["time_ordinal", "MONTH", "latitude", "longitude"]:
            if signal_df[signal].notna().sum() > 2:
                corr = pd.concat([indicator, signal_df[signal]], axis=1).corr(method="spearman").iloc[0, 1]
            else:
                corr = np.nan
            rows.append({
                "missing_variable": col,
                "signal": signal,
                "spearman_corr_with_missing_indicator": float(corr) if pd.notna(corr) else np.nan,
            })

    pd.DataFrame(rows).to_csv(
        REPORT_DIR / "missingness_indicator_signal_correlation.csv",
        index=False,
    )

    _plot_missingness_diagnostics(working, weather_cols)


def _plot_missingness_diagnostics(df: pd.DataFrame, weather_cols: List[str]) -> None:
    """Save missingness plots for report/EDA."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        warnings.warn(f"Matplotlib not available. Skipping missingness plots. Error: {exc}")
        return

    plot_dir = REPORT_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: missing rate by variable
    missing = df[weather_cols].isna().mean().sort_values(ascending=True)
    plt.figure(figsize=(8, 4.8))
    plt.barh(missing.index, missing.values)
    plt.xlabel("Missing rate")
    plt.title("Missing Rate by Variable Before Imputation")
    plt.tight_layout()
    plt.savefig(plot_dir / "missing_rate_by_variable_before_imputation.png", dpi=200)
    plt.close()

    # Plot 2: monthly missing rate of key variables
    tmp = df.copy()
    tmp["month_period"] = tmp["time"].dt.to_period("M").astype(str)

    key_cols = [col for col in ["PRCP", "TEMP", "DEWP", "VISIB", "STP", "SLP"] if col in weather_cols]
    if key_cols:
        monthly = tmp.groupby("month_period")[key_cols].apply(lambda x: x.isna().mean())
        plt.figure(figsize=(10, 5))
        for col in key_cols:
            plt.plot(monthly.index, monthly[col], label=col)
        plt.xticks(rotation=90, fontsize=6)
        plt.ylabel("Missing rate")
        plt.title("Monthly Missingness Before Imputation")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / "monthly_missingness_before_imputation.png", dpi=200)
        plt.close()


# ============================================================================
# STEP 3: BUILD COMPLETE PANEL AND MERGE ERA5
# ============================================================================

def build_complete_station_date_panel(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create one row per station per day for the full study period.

    This is better than only appending missing ERA5 rows because it makes the
    target panel explicit and auditable.
    """
    print("\n=== STEP 5.3: Building complete station-date panel ===")

    prepared = prepare_station_grid(df)
    full_dates = pd.date_range(DATA_START_DATE, DATA_END_DATE, freq="D")

    station_meta = (
        prepared[["STATION", "LATITUDE", "LONGITUDE", "ELEVATION", "latitude", "longitude"]]
        .drop_duplicates(subset=["STATION"])
        .reset_index(drop=True)
    )

    panels = []
    for _, row in station_meta.iterrows():
        panel = pd.DataFrame({"time": full_dates})
        for col in station_meta.columns:
            panel[col] = row[col]
        panels.append(panel)

    panel_df = pd.concat(panels, ignore_index=True)

    obs_cols = ["STATION", "time", "has_gsod_record"] + [col for col in WEATHER_COLS if col in prepared.columns]
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
        raise AssertionError(f"Complete station-date panel has {duplicate_count} duplicated rows.")

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
    era5["time"] = _safe_to_datetime(era5["time"])
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

    return merged


# ============================================================================
# STEP 4: DISTRIBUTION AND OVERLAP DIAGNOSTICS
# ============================================================================

def compare_gsod_with_era5_overlap(df_with_era5: pd.DataFrame) -> pd.DataFrame:
    """
    Compare GSOD observed values and ERA5 values on overlapping station-days.

    This provides evidence that ERA5 imputation is not arbitrary.
    """
    print("\n=== STEP 5.5: Comparing GSOD observations with ERA5 overlap ===")

    rows = []

    # Add ERA5 wind speed for WDSP comparison if available.
    if {"u10", "v10"}.issubset(df_with_era5.columns):
        df_with_era5 = df_with_era5.copy()
        df_with_era5["era5_wind_speed_10m"] = np.sqrt(df_with_era5["u10"] ** 2 + df_with_era5["v10"] ** 2)

    mapping = dict(ERA5_FILL_MAPPING)
    if "era5_wind_speed_10m" in df_with_era5.columns:
        mapping["WDSP"] = "era5_wind_speed_10m"

    for gsod_col, era5_col in mapping.items():
        if gsod_col not in df_with_era5.columns or era5_col not in df_with_era5.columns:
            continue

        pair = df_with_era5[[gsod_col, era5_col]].dropna()
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

    _plot_gsod_vs_era5_overlap(df_with_era5, mapping)

    return report


def _plot_gsod_vs_era5_overlap(df: pd.DataFrame, mapping: Dict[str, str]) -> None:
    """Save overlap distribution plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        warnings.warn(f"Matplotlib not available. Skipping overlap plots. Error: {exc}")
        return

    plot_dir = REPORT_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for gsod_col, era5_col in mapping.items():
        if gsod_col not in df.columns or era5_col not in df.columns:
            continue

        pair = df[[gsod_col, era5_col]].dropna()
        if pair.empty:
            continue

        # Use log transform for precipitation due to heavy skew.
        if gsod_col == "PRCP":
            x1 = np.log1p(pair[gsod_col])
            x2 = np.log1p(pair[era5_col])
            xlabel = "log(1 + precipitation mm)"
        else:
            x1 = pair[gsod_col]
            x2 = pair[era5_col]
            xlabel = gsod_col

        plt.figure(figsize=(7, 4.5))
        plt.hist(x1, bins=50, alpha=0.55, label=f"GSOD {gsod_col}")
        plt.hist(x2, bins=50, alpha=0.55, label=f"ERA5 {era5_col}")
        plt.xlabel(xlabel)
        plt.ylabel("Frequency")
        plt.title(f"GSOD vs ERA5 Distribution: {gsod_col}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"gsod_vs_era5_distribution_{gsod_col}.png", dpi=200)
        plt.close()


# ============================================================================
# STEP 5: IMPUTATION
# ============================================================================

def fill_from_era5_and_interpolate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing GSOD variables using ERA5 and group-wise interpolation.

    Important:
    - Numeric variables with ERA5 equivalents are filled from ERA5.
    - VISIB is interpolated WITHIN station/grid groups only.
    - Source flags are stored for auditability.
    """
    print("\n=== STEP 5.6: Filling missing values with ERA5 and group-wise interpolation ===")

    out = df.sort_values(GROUPBY_GRID_COLS + ["time"]).reset_index(drop=True).copy()

    # ERA5 wind speed for filling GSOD WDSP.
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

    # Group-wise interpolation for visibility.
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
    """
    Compare distributions before and after imputation.

    The comparison has three views:
    - GSOD observed distribution before filling
    - Final distribution after filling
    - Filled-only distribution for rows/values supplied by ERA5/interpolation
    """
    print("\n=== STEP 5.7: Comparing distributions before and after imputation ===")

    rows = []
    variables = [col for col in WEATHER_COLS if col in after_fill.columns]

    for col in variables:
        before_values = pd.to_numeric(before_panel[col], errors="coerce") if col in before_panel.columns else pd.Series(dtype=float)
        after_values = pd.to_numeric(after_fill[col], errors="coerce")

        source_col = f"{col}_source"
        if source_col in after_fill.columns:
            filled_values = pd.to_numeric(after_fill.loc[after_fill[source_col] != "GSOD", col], errors="coerce")
            era5_values = pd.to_numeric(after_fill.loc[after_fill[source_col] == "ERA5", col], errors="coerce")
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

    _plot_before_after_imputation(before_panel, after_fill, variables)

    return report


def _plot_before_after_imputation(
    before_panel: pd.DataFrame,
    after_fill: pd.DataFrame,
    variables: List[str],
) -> None:
    """Save distribution and time-series diagnostic plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        warnings.warn(f"Matplotlib not available. Skipping before/after plots. Error: {exc}")
        return

    plot_dir = REPORT_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for col in variables:
        before = pd.to_numeric(before_panel[col], errors="coerce").dropna() if col in before_panel.columns else pd.Series(dtype=float)
        after = pd.to_numeric(after_fill[col], errors="coerce").dropna()

        if before.empty or after.empty:
            continue

        if col == "PRCP":
            before_plot = np.log1p(before)
            after_plot = np.log1p(after)
            xlabel = "log(1 + PRCP mm)"
        else:
            before_plot = before
            after_plot = after
            xlabel = col

        plt.figure(figsize=(7, 4.5))
        plt.hist(before_plot, bins=50, alpha=0.55, label="GSOD observed before fill")
        plt.hist(after_plot, bins=50, alpha=0.55, label="After fill")
        plt.xlabel(xlabel)
        plt.ylabel("Frequency")
        plt.title(f"Before vs After Imputation: {col}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_dir / f"before_after_imputation_distribution_{col}.png", dpi=200)
        plt.close()

    # Time-series before/after for the first station, monthly mean.
    if "STATION" in after_fill.columns and "time" in after_fill.columns:
        first_station = str(after_fill["STATION"].dropna().iloc[0]) if after_fill["STATION"].notna().any() else None
        if first_station:
            for col in [c for c in ["PRCP", "TEMP", "VISIB"] if c in after_fill.columns]:
                before_s = before_panel[before_panel["STATION"].astype(str) == first_station].copy()
                after_s = after_fill[after_fill["STATION"].astype(str) == first_station].copy()
                before_s["month"] = pd.to_datetime(before_s["time"]).dt.to_period("M").dt.to_timestamp()
                after_s["month"] = pd.to_datetime(after_s["time"]).dt.to_period("M").dt.to_timestamp()

                b = before_s.groupby("month")[col].mean()
                a = after_s.groupby("month")[col].mean()

                plt.figure(figsize=(10, 4))
                plt.plot(b.index, b.values, label="Before fill")
                plt.plot(a.index, a.values, label="After fill")
                plt.xlabel("Time")
                plt.ylabel(col)
                plt.title(f"Monthly Mean Before/After Fill - {col} - Station {first_station}")
                plt.legend()
                plt.tight_layout()
                plt.savefig(plot_dir / f"time_series_before_after_{col}_station_{first_station}.png", dpi=200)
                plt.close()


# ============================================================================
# STEP 6: DROP POLICY AND FINAL CLEANUP
# ============================================================================

def create_drop_rationale_table(keep_auxiliary_era5: bool = KEEP_AUXILIARY_ERA5) -> pd.DataFrame:
    """
    Create a transparent table explaining which columns are removed and why.
    """
    rows = []

    for col in DUPLICATE_ERA5_SOURCE_COLS:
        rows.append({
            "column": col,
            "action": "drop",
            "reason": (
                "Duplicate ERA5 source column after being used to fill standardized "
                "GSOD-like columns TEMP/PRCP/DEWP/STP/SLP. Source flag columns "
                "preserve whether the value came from GSOD or ERA5."
            ),
        })

    rows.append({
        "column": "era5_wind_speed_10m",
        "action": "drop",
        "reason": (
            "Temporary derived helper used to fill WDSP from ERA5 u10/v10. "
            "Original u10 and v10 can be retained as auxiliary wind components."
        ),
    })

    rows.extend([
        {"column": "latitude", "action": "drop", "reason": "Temporary ERA5 grid latitude. Original station LATITUDE is retained."},
        {"column": "longitude", "action": "drop", "reason": "Temporary ERA5 grid longitude. Original station LONGITUDE is retained."},
        {"column": "month_index", "action": "drop if present", "reason": "Technical ENSO merge helper, not needed in final daily silver table."},
    ])

    if not keep_auxiliary_era5:
        for col in AUXILIARY_ERA5_COLS:
            rows.append({
                "column": col,
                "action": "drop",
                "reason": "Auxiliary ERA5 variable removed because KEEP_AUXILIARY_ERA5=False.",
            })
    else:
        for col in AUXILIARY_ERA5_COLS:
            rows.append({
                "column": col,
                "action": "keep if present",
                "reason": (
                    "Auxiliary ERA5 variable retained because it may support EDA "
                    "or downstream feature engineering."
                ),
            })

    table = pd.DataFrame(rows)
    table.to_csv(REPORT_DIR / "dropped_columns_rationale.csv", index=False)
    return table


def finalize_silver_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate/temp columns with rationale and round numeric values.
    """
    create_drop_rationale_table(KEEP_AUXILIARY_ERA5)

    cols_to_drop = list(DUPLICATE_ERA5_SOURCE_COLS) + [
        "era5_wind_speed_10m",
        "latitude",
        "longitude",
        "month_index",
    ]

    if not KEEP_AUXILIARY_ERA5:
        cols_to_drop.extend(AUXILIARY_ERA5_COLS)

    out = df.drop(columns=cols_to_drop, errors="ignore").copy()

    # Keep datetime as ISO string for CSV stability.
    out["time"] = _safe_to_datetime(out["time"])

    # Sort for reproducibility.
    sort_cols = [c for c in ["STATION", "time"] if c in out.columns]
    out = out.sort_values(sort_cols).reset_index(drop=True)

    # Round numeric values for stable CSV size/readability.
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].round(4)

    return out


# ============================================================================
# STEP 7: MERGE ENSO AND PRESSURE
# ============================================================================

def merge_enso_data(df: pd.DataFrame, enso_path: Path = ENSO_PATH) -> pd.DataFrame:
    """Merge ENSO by YEAR/MONTH using left join to avoid silently dropping rows."""
    if not enso_path.exists():
        raise FileNotFoundError(f"ENSO file not found: {enso_path}")

    print("\n=== STEP 5.8: Merging ENSO climate index ===")

    out = df.copy()
    out["YEAR"] = pd.to_datetime(out["time"]).dt.year
    out["MONTH"] = pd.to_datetime(out["time"]).dt.month

    enso = pd.read_csv(enso_path)
    enso_cols = [c for c in enso.columns if c not in ["month_index"]]
    enso = enso[enso_cols]

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
    pressure["time"] = _safe_to_datetime(pressure["time"])
    pressure = _aggregate_duplicate_era5_rows(pressure, "ERA5 pressure-level")

    out = pd.merge(
        df,
        pressure,
        how="left",
        on=["latitude", "longitude", "time"],
    )

    pressure_cols = [c for c in pressure.columns if c not in ["time", "latitude", "longitude"]]
    pressure_missing = _missing_rate_by_column(out[pressure_cols]) if pressure_cols else pd.DataFrame()
    pressure_missing.to_csv(REPORT_DIR / "pressure_missing_after_merge.csv", index=False)

    return out


# Backward-compatible function name from the old script.
def merge_file(df: pd.DataFrame) -> pd.DataFrame:
    df = merge_enso_data(df)
    df = merge_pressure_data(df)
    return finalize_silver_columns(df)


# ============================================================================
# VALIDATION AND METADATA
# ============================================================================

def validate_silver_dataset(df: pd.DataFrame, strict: bool = True) -> None:
    """Run final structural and physical sanity checks."""
    print("\n=== STEP 5.10: Validating final silver dataset ===")

    if {"STATION", "time"}.issubset(df.columns):
        dup = int(df.duplicated(subset=["STATION", "time"]).sum())
        if dup > 0:
            msg = f"Final silver dataset has {dup:,} duplicate STATION-time rows."
            if strict:
                raise AssertionError(msg)
            warnings.warn(msg)

    if "PRCP" in df.columns:
        neg = int((pd.to_numeric(df["PRCP"], errors="coerce") < 0).sum())
        extreme = int((pd.to_numeric(df["PRCP"], errors="coerce") > PRCP_EXTREME_WARNING_MM).sum())

        if neg > 0:
            msg = f"Final silver dataset has {neg:,} negative precipitation values."
            if strict:
                raise AssertionError(msg)
            warnings.warn(msg)

        if extreme > 0:
            warnings.warn(
                f"Final silver dataset has {extreme:,} PRCP values > "
                f"{PRCP_EXTREME_WARNING_MM} mm. Inspect in EDA; not removed automatically."
            )

    if {"TEMP", "DEWP"}.issubset(df.columns):
        violations = int((pd.to_numeric(df["DEWP"], errors="coerce") > pd.to_numeric(df["TEMP"], errors="coerce") + 2.0).sum())
        if violations > 0:
            warnings.warn(
                f"Final silver dataset has {violations:,} rows where DEWP > TEMP + 2°C. "
                "These should be inspected."
            )

    final_missing = _missing_rate_by_column(df)
    final_missing.to_csv(REPORT_DIR / "missing_rate_final_silver.csv", index=False)

    _numeric_summary(df, [col for col in WEATHER_COLS if col in df.columns]).to_csv(
        REPORT_DIR / "weather_numeric_summary_final_silver.csv",
        index=False,
    )


def save_source_coverage_report(df: pd.DataFrame) -> None:
    """Save how many values came from GSOD, ERA5, interpolation, or remain missing."""
    rows = []
    for col in WEATHER_COLS:
        source_col = f"{col}_source"
        if source_col not in df.columns:
            continue

        counts = df[source_col].value_counts(dropna=False)
        total = len(df)
        for source, count in counts.items():
            rows.append({
                "variable": col,
                "source": str(source),
                "count": int(count),
                "rate": float(count / total) if total else np.nan,
            })

    pd.DataFrame(rows).to_csv(REPORT_DIR / "value_source_coverage.csv", index=False)


def save_silver_metadata(df: pd.DataFrame) -> None:
    """Save metadata explaining the imputation design and leakage constraints."""
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(BASE_DIR),
        "input_files": {
            "bronze_gsod": str(BRONZE_GSOD_PATH),
            "era5_single_level": str(ERA5_SINGLE_PATH),
            "enso": str(ENSO_PATH),
            "pressure_level": str(resolve_pressure_path()) if any((CLEAN_DIR / n).exists() for n in PRESSURE_FILE_CANDIDATES) else None,
        },
        "output_file": str(SILVER_OUTPUT_PATH),
        "study_period": {
            "start": DATA_START_DATE,
            "end": DATA_END_DATE,
        },
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "n_stations": int(df["STATION"].nunique()) if "STATION" in df.columns else None,
        "imputation_policy": {
            "GSOD_like_variables_filled_from_ERA5": ERA5_FILL_MAPPING,
            "WDSP_filled_from_ERA5_uv_wind_speed": FILL_WDSP_FROM_ERA5_UV,
            "VISIB_interpolation": (
                "Linear interpolation within each STATION-latitude-longitude group. "
                "No interpolation is performed across different stations."
            ),
            "source_flags": "Each imputed variable has *_source column where applicable.",
        },
        "leakage_note": (
            "This Silver dataset is designed as a historical benchmark dataset. "
            "ERA5 reanalysis uses post-processed information and is acceptable for "
            "data rescue / historical reconstruction. If the project is framed as "
            "real-time future forecasting, ERA5 same-day reanalysis values must be "
            "replaced by forecast products or lagged/available-at-prediction-time "
            "features to avoid data leakage."
        ),
        "drop_policy": str(REPORT_DIR / "dropped_columns_rationale.csv"),
        "reports": {
            "missingness_before_imputation": str(REPORT_DIR / "missing_rate_before_imputation_by_column.csv"),
            "gsod_vs_era5_overlap": str(REPORT_DIR / "gsod_vs_era5_overlap_comparison.csv"),
            "distribution_before_after_imputation": str(REPORT_DIR / "distribution_before_after_imputation.csv"),
            "source_coverage": str(REPORT_DIR / "value_source_coverage.csv"),
            "final_missing_rate": str(REPORT_DIR / "missing_rate_final_silver.csv"),
        },
    }

    _write_json(metadata, REPORT_DIR / "silver_metadata.json")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def build_silver_dataset(
    gsod_path: Path = BRONZE_GSOD_PATH,
    output_path: Path = SILVER_OUTPUT_PATH,
    strict_validation: bool = True,
) -> pd.DataFrame:
    """
    End-to-end Silver layer pipeline.

    Output:
    - data/clean/silver_data_ver2.csv
    - reports/data_quality/silver/*.csv
    - reports/data_quality/silver/*.json
    - reports/data_quality/silver/plots/*.png
    """
    ensure_directories()

    # 1. Load and standardize GSOD.
    gsod_clean = load_and_clean_gsod(gsod_path)

    # 2. Diagnose missingness BEFORE imputation.
    diagnose_missingness_before_imputation(gsod_clean)

    # 3. Build complete station-date panel.
    panel = build_complete_station_date_panel(gsod_clean)

    # 4. Merge ERA5 single-level.
    era5 = load_era5_single_level(ERA5_SINGLE_PATH)
    panel_with_era5 = merge_era5_single_level(panel, era5)

    # 5. Compare observed GSOD and ERA5 overlap before filling.
    compare_gsod_with_era5_overlap(panel_with_era5)

    # 6. Fill missing values.
    filled = fill_from_era5_and_interpolate(panel_with_era5)

    # 7. Compare before vs after imputation.
    compare_distribution_before_after_fill(panel, filled)

    # 8. Merge ENSO and pressure-level.
    merged = merge_enso_data(filled)
    merged = merge_pressure_data(merged)

    # 9. Finalize.
    final = finalize_silver_columns(merged)

    # 10. Validate and save reports.
    validate_silver_dataset(final, strict=strict_validation)
    save_source_coverage_report(final)
    save_silver_metadata(final)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(output_path, index=False)

    print(f"\n[SUCCESS] Silver dataset saved to: {output_path}")
    print(f"          Shape: {final.shape}")
    print(f"          Reports saved to: {REPORT_DIR}")

    return final


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Backward-compatible wrapper for the old pipeline style.

    Note: This function assumes that `df` is already the cleaned GSOD dataframe.
    For full reports and final output, prefer `build_silver_dataset()`.
    """
    diagnose_missingness_before_imputation(df)
    panel = build_complete_station_date_panel(df)
    era5 = load_era5_single_level(ERA5_SINGLE_PATH)
    panel_with_era5 = merge_era5_single_level(panel, era5)
    compare_gsod_with_era5_overlap(panel_with_era5)
    filled = fill_from_era5_and_interpolate(panel_with_era5)
    compare_distribution_before_after_fill(panel, filled)
    return filled


def main() -> None:
    build_silver_dataset(
        gsod_path=BRONZE_GSOD_PATH,
        output_path=SILVER_OUTPUT_PATH,
        strict_validation=True,
    )


if __name__ == "__main__":
    main()
