"""
ERA5 single-level data processing module with precipitation quality control.

This module processes ERA5 single-level weather variables from GRIB format and
adds explicit quality checks for total precipitation (tp), because tp is the
most error-prone variable when GRIB files contain both `step` and `valid_time`.

Key improvements over the initial version:
1. Create output directories before saving Parquet/CSV/PNG files.
2. Process total precipitation (`tp`) in a dedicated pipeline.
3. Collapse duplicate `valid_time` records before daily summation to avoid
   double counting caused by multiple forecast steps for the same valid time.
4. Check whether daily precipitation contains negative values or suspiciously
   extreme values.
5. Save data quality reports and metadata.
6. Optionally compare ERA5 daily `tp` with GSOD station `PRCP` after converting
   GSOD precipitation from inches to millimeters.
"""

from __future__ import annotations

import gc
import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr


# ============================================================================
# CONFIGURATION
# ============================================================================

VIETNAM_LAT_SLICE = (24, 8)      # ERA5 latitude is usually descending
VIETNAM_LON_SLICE = (102, 110)   # West -> East
RESAMPLE_FREQUENCY = "1D"
GRID_RESOLUTION = 0.25

# Raw folders are expected to be:
# data/raw/single/Single_level_1520/data.grib
# data/raw/single/Single_level_2125/data.grib
SINGLE_LEVEL_PERIODS = ("1520", "2125")

# Output filenames
OUTPUT_SINGLE_FINAL_NAME = "ERA5_single_level.parquet"

# Columns to drop after converting GRIB to dataframe.
# We keep `valid_time` only in the precipitation-specific extraction function.
COLS_TO_DROP = ["number", "step", "surface", "valid_time"]

# Temperature columns to convert from Kelvin to Celsius
TEMP_COLS_K_TO_C = ["d2m", "t2m", "sst"]

# Pressure columns to convert from Pa to hPa
PRESSURE_COLS_PA_TO_HPA = ["msl", "sp"]

# Daily aggregation rules for non-precipitation variables.
# `tp` is intentionally excluded here and handled separately.
BASE_RESAMPLE_AGG_COLS = {
    "t2m": "mean",     # 2m temperature
    "d2m": "mean",     # 2m dew point
    "sst": "mean",     # sea surface temperature
    "u10": "mean",     # 10m u-wind
    "v10": "mean",     # 10m v-wind
    "sp": "mean",      # surface pressure
    "msl": "mean",     # mean sea-level pressure
    "z": "mean",       # geopotential height after conversion
    "lsm": "first",    # land-sea mask, static
}

# How to collapse duplicate precipitation rows with the same valid_time/lat/lon.
# Rationale: duplicate rows can occur if several forecast reference times point
# to the same valid_time. These rows describe the same target hour, so they must
# NOT be summed. "max" is conservative for accumulated precipitation and avoids
# undercounting if duplicates differ slightly.
TP_DUPLICATE_AGG = "max"

# Physical and sanity thresholds for daily precipitation in Vietnam.
# Negative precipitation is physically impossible. Extreme threshold is used as
# a warning/reporting threshold, not as automatic deletion.
NEGATIVE_TP_TOL_MM = 1e-6
EXTREME_DAILY_TP_MM = 500.0

# GSOD missing/error code and unit conversion
GSOD_PRCP_ERROR_CODE = 99.99     # inches
GSOD_INCH_TO_MM = 25.4

# Quality report paths
QUALITY_REPORT_SUBDIR = Path("reports") / "data_quality" / "single_level"


# ============================================================================
# PATH HELPERS
# ============================================================================

def find_project_root(start: Optional[Path] = None) -> Path:
    """
    Find project root by walking upward until a `data` directory is found.
    This makes the script usable whether it is placed in project root or src/.
    """
    if start is None:
        start = Path(__file__).resolve().parent

    candidates = [start, *start.parents]
    for path in candidates:
        if (path / "data").exists():
            return path

    # Fallback: parent of this file
    return Path(__file__).resolve().parent


BASE_DIR = find_project_root()
BASE_SINGLE_RAW = BASE_DIR / "data" / "raw" / "single"
BASE_SINGLE_CLEAN = BASE_DIR / "data" / "clean" / "single_level"
OUTPUT_SINGLE_FINAL = BASE_DIR / "data" / "clean" / OUTPUT_SINGLE_FINAL_NAME
BRONZE_GSOD_PATH = BASE_DIR / "data" / "raw" / "bronze_data.csv"
REPORT_DIR = BASE_DIR / QUALITY_REPORT_SUBDIR


def ensure_directories() -> None:
    """Create output directories before saving files."""
    BASE_SINGLE_CLEAN.mkdir(parents=True, exist_ok=True)
    OUTPUT_SINGLE_FINAL.parent.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# BASIC CONVERSIONS
# ============================================================================

def _downsample_to_float32(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast float64 columns to float32 to reduce memory usage."""
    float_cols = df.select_dtypes(include=["float64"]).columns
    df[float_cols] = df[float_cols].astype("float32")
    return df


def _convert_temperature_kelvin_to_celsius(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """Convert temperature columns from Kelvin to Celsius."""
    if cols is None:
        cols = TEMP_COLS_K_TO_C

    for col in cols:
        if col in df.columns:
            df[col] = df[col] - 273.15
    return df


def _convert_pressure_pa_to_hpa(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None
) -> pd.DataFrame:
    """Convert pressure columns from Pascal to hPa."""
    if cols is None:
        cols = PRESSURE_COLS_PA_TO_HPA

    for col in cols:
        if col in df.columns:
            df[col] = df[col] / 100.0
    return df


def _convert_geopotential_to_meters(df: pd.DataFrame) -> pd.DataFrame:
    """Convert geopotential from J/kg to geopotential height in meters."""
    if "z" in df.columns:
        df["z"] = df["z"] / 9.81
    return df


def _convert_precipitation_m_to_mm(df: pd.DataFrame) -> pd.DataFrame:
    """Convert ERA5 total precipitation from meters to millimeters."""
    if "tp" in df.columns:
        df["tp"] = df["tp"] * 1000.0
    return df


def _available_agg_cols(df: pd.DataFrame, agg_cols: Dict[str, str]) -> Dict[str, str]:
    """Keep aggregation rules only for columns available in dataframe."""
    return {col: agg for col, agg in agg_cols.items() if col in df.columns}


# ============================================================================
# NON-PRECIPITATION SINGLE-LEVEL PROCESSING
# ============================================================================

def _load_base_single_level_variables(grib_path: Path) -> pd.DataFrame:
    """
    Load non-precipitation ERA5 single-level variables.

    The total precipitation variable `tp` is intentionally removed here and
    processed with a dedicated function, because its GRIB structure can include
    step/valid_time fields that need special handling.
    """
    print(f"\n--- Reading base single-level variables from: {grib_path}")
    ds = xr.open_dataset(grib_path, engine="cfgrib")
    ds_filter = ds.sel(
        latitude=slice(*VIETNAM_LAT_SLICE),
        longitude=slice(*VIETNAM_LON_SLICE),
    )

    df = ds_filter.to_dataframe().reset_index()

    # Remove tp if it appears in the general hypercube.
    # Precipitation is handled separately below.
    df = df.drop(columns=["tp"], errors="ignore")
    df = df.drop(columns=COLS_TO_DROP, errors="ignore")

    if "time" not in df.columns:
        raise ValueError(f"No `time` column found in base variables of {grib_path}")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    # If duplicates exist after dropping GRIB identifiers, collapse them safely.
    # For regular state variables, mean is appropriate; lsm uses first later.
    key_cols = ["time", "latitude", "longitude"]
    duplicate_count = int(df.duplicated(subset=key_cols).sum())
    if duplicate_count > 0:
        warnings.warn(
            f"Base variables have {duplicate_count:,} duplicate rows by "
            f"{key_cols}. Collapsing numeric variables by mean."
        )
        numeric_cols = [
            col for col in df.select_dtypes(include=[np.number]).columns
            if col not in ["latitude", "longitude"]
        ]
        agg_map = {col: "mean" for col in numeric_cols}
        if "lsm" in df.columns:
            agg_map["lsm"] = "first"
        df = df.groupby(key_cols, as_index=False).agg(agg_map)

    df = _convert_temperature_kelvin_to_celsius(df)
    df = _convert_pressure_pa_to_hpa(df)
    df = _convert_geopotential_to_meters(df)
    df = _downsample_to_float32(df)

    return df


def _resample_base_variables_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Resample non-precipitation single-level variables to daily frequency."""
    agg_cols = _available_agg_cols(df, BASE_RESAMPLE_AGG_COLS)
    if not agg_cols:
        raise ValueError("No base variables available for daily aggregation.")

    daily = (
        df.set_index("time")
        .groupby(["latitude", "longitude"])
        .resample(RESAMPLE_FREQUENCY)
        .agg(agg_cols)
        .reset_index()
    )

    return daily


# ============================================================================
# PRECIPITATION-SPECIFIC PROCESSING
# ============================================================================

def _extract_total_precipitation_hourly(grib_path: Path) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Extract ERA5 total precipitation using valid_time and remove duplicate
    valid_time rows before daily summation.

    Why this is necessary:
    - ERA5 GRIB files may contain both `time` and `step`.
    - `valid_time = time + step` represents the actual target timestamp.
    - If multiple rows point to the same `valid_time`, summing them directly
      would double count precipitation for the same hour.
    """
    print(f"--- Extracting total precipitation `tp` from: {grib_path}")

    ds_tp = xr.open_dataset(
        grib_path,
        engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"shortName": "tp"}},
    )
    ds_tp = ds_tp.sel(
        latitude=slice(*VIETNAM_LAT_SLICE),
        longitude=slice(*VIETNAM_LON_SLICE),
    )

    tp = ds_tp.to_dataframe().reset_index()

    if "tp" not in tp.columns:
        raise ValueError(f"`tp` variable not found in {grib_path}")

    # Preserve forecast reference time for diagnostics if available.
    if "valid_time" in tp.columns:
        if "time" in tp.columns:
            tp = tp.rename(columns={"time": "source_time", "valid_time": "time"})
        else:
            tp = tp.rename(columns={"valid_time": "time"})
    elif "time" in tp.columns:
        # Some files may already use `time` as the valid timestamp.
        pass
    else:
        raise ValueError(f"Neither `valid_time` nor `time` found in {grib_path}")

    tp["time"] = pd.to_datetime(tp["time"], errors="coerce")
    tp = _convert_precipitation_m_to_mm(tp)

    key_cols = ["time", "latitude", "longitude"]
    raw_rows = int(len(tp))
    duplicate_rows = int(tp.duplicated(subset=key_cols).sum())

    if len(tp) == 0:
        raise ValueError(f"No precipitation rows extracted from {grib_path}")

    duplicate_group_sizes = tp.groupby(key_cols).size()
    duplicate_group_count = int((duplicate_group_sizes > 1).sum())
    max_duplicates_per_key = int(duplicate_group_sizes.max())

    # Report negative raw hourly precipitation before clipping.
    raw_negative_count = int((tp["tp"] < -NEGATIVE_TP_TOL_MM).sum())
    near_zero_negative_count = int(((tp["tp"] < 0) & (tp["tp"] >= -NEGATIVE_TP_TOL_MM)).sum())

    if near_zero_negative_count > 0:
        # Tiny negatives can be numerical noise; clip to zero but report it.
        tp.loc[(tp["tp"] < 0) & (tp["tp"] >= -NEGATIVE_TP_TOL_MM), "tp"] = 0.0

    # Collapse duplicate valid_time records BEFORE daily sum.
    # This is the key fix that prevents double counting.
    tp_unique = (
        tp.groupby(key_cols, as_index=False)["tp"]
        .agg(TP_DUPLICATE_AGG)
        .sort_values(key_cols)
        .reset_index(drop=True)
    )

    metadata = {
        "grib_path": str(grib_path),
        "raw_tp_rows": raw_rows,
        "unique_tp_rows_by_valid_time_lat_lon": int(len(tp_unique)),
        "duplicate_tp_rows_by_valid_time_lat_lon": duplicate_rows,
        "duplicate_tp_group_count": duplicate_group_count,
        "max_duplicates_per_valid_time_lat_lon": max_duplicates_per_key,
        "tp_duplicate_aggregation": TP_DUPLICATE_AGG,
        "raw_negative_tp_count_below_tolerance": raw_negative_count,
        "near_zero_negative_tp_count_clipped_to_zero": near_zero_negative_count,
    }

    return tp_unique, metadata


def _resample_tp_to_daily(tp_unique: pd.DataFrame) -> pd.DataFrame:
    """
    Resample unique hourly/valid-time precipitation records to daily totals.

    Daily precipitation is an accumulated quantity, therefore summation is used
    after duplicate valid_time records have already been collapsed.
    """
    daily_tp = (
        tp_unique.set_index("time")
        .groupby(["latitude", "longitude"])
        .resample(RESAMPLE_FREQUENCY)
        .agg({"tp": "sum"})
        .reset_index()
    )

    return daily_tp


# ============================================================================
# VALIDATION & QUALITY REPORTS
# ============================================================================

def _duplicate_count(df: pd.DataFrame, keys: List[str]) -> int:
    """Count duplicate rows by a list of key columns."""
    missing_keys = [key for key in keys if key not in df.columns]
    if missing_keys:
        return -1
    return int(df.duplicated(subset=keys).sum())


def _missing_rate_by_column(df: pd.DataFrame) -> pd.DataFrame:
    """Return missing rate report for each column."""
    return (
        df.isna()
        .mean()
        .rename("missing_rate")
        .reset_index()
        .rename(columns={"index": "column"})
        .sort_values("missing_rate", ascending=False)
    )


def _summarize_single_level_quality(
    df: pd.DataFrame,
    dataset_name: str,
    tp_metadata: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Create a compact quality summary for one processed dataset."""
    keys = ["time", "latitude", "longitude"]

    summary: Dict[str, object] = {
        "dataset": dataset_name,
        "n_rows": int(len(df)),
        "n_columns": int(df.shape[1]),
        "n_days": int(pd.to_datetime(df["time"]).dt.date.nunique()) if "time" in df.columns else None,
        "start_date": str(pd.to_datetime(df["time"]).min().date()) if "time" in df.columns and len(df) else None,
        "end_date": str(pd.to_datetime(df["time"]).max().date()) if "time" in df.columns and len(df) else None,
        "n_grid_cells": int(df[["latitude", "longitude"]].drop_duplicates().shape[0])
        if {"latitude", "longitude"}.issubset(df.columns) else None,
        "duplicate_count_by_time_lat_lon": _duplicate_count(df, keys),
        "max_missing_rate": float(df.isna().mean().max()) if len(df) else None,
        "mean_missing_rate": float(df.isna().mean().mean()) if len(df) else None,
    }

    if "tp" in df.columns:
        tp = pd.to_numeric(df["tp"], errors="coerce")
        summary.update({
            "tp_missing_rate": float(tp.isna().mean()),
            "tp_negative_count": int((tp < -NEGATIVE_TP_TOL_MM).sum()),
            "tp_zero_or_near_zero_count": int((tp <= NEGATIVE_TP_TOL_MM).sum()),
            "tp_extreme_count_gt_threshold": int((tp > EXTREME_DAILY_TP_MM).sum()),
            "tp_extreme_threshold_mm": EXTREME_DAILY_TP_MM,
            "tp_min_mm": float(tp.min()) if tp.notna().any() else None,
            "tp_mean_mm": float(tp.mean()) if tp.notna().any() else None,
            "tp_median_mm": float(tp.median()) if tp.notna().any() else None,
            "tp_p95_mm": float(tp.quantile(0.95)) if tp.notna().any() else None,
            "tp_p99_mm": float(tp.quantile(0.99)) if tp.notna().any() else None,
            "tp_max_mm": float(tp.max()) if tp.notna().any() else None,
        })

    if tp_metadata:
        summary.update(tp_metadata)

    return summary


def validate_single_level_data(
    df: pd.DataFrame,
    dataset_name: str,
    strict_assertions: bool = True,
) -> None:
    """
    Validate important physical and structural assumptions.

    Strict assertions should be enabled for final runs. During exploratory runs,
    set strict_assertions=False to generate reports without interrupting.
    """
    keys = ["time", "latitude", "longitude"]
    duplicate_count = _duplicate_count(df, keys)

    if duplicate_count > 0:
        message = (
            f"[{dataset_name}] Found {duplicate_count:,} duplicate rows by "
            f"{keys}. This can cause leakage or duplicated samples."
        )
        if strict_assertions:
            raise AssertionError(message)
        warnings.warn(message)

    if "tp" in df.columns:
        tp = pd.to_numeric(df["tp"], errors="coerce")
        negative_count = int((tp < -NEGATIVE_TP_TOL_MM).sum())
        extreme_count = int((tp > EXTREME_DAILY_TP_MM).sum())

        if negative_count > 0:
            message = (
                f"[{dataset_name}] Found {negative_count:,} daily precipitation "
                "values below zero. Precipitation must be non-negative."
            )
            if strict_assertions:
                raise AssertionError(message)
            warnings.warn(message)

        if extreme_count > 0:
            warnings.warn(
                f"[{dataset_name}] Found {extreme_count:,} daily precipitation "
                f"values > {EXTREME_DAILY_TP_MM} mm. These are not deleted "
                "automatically, but must be inspected in EDA/report."
            )


def _save_quality_reports(
    quality_summaries: List[Dict[str, object]],
    final_df: pd.DataFrame,
    tp_period_metadata: List[Dict[str, object]],
) -> None:
    """Save quality summaries, missing-rate report, and metadata."""
    ensure_directories()

    summary_df = pd.DataFrame(quality_summaries)
    summary_path = REPORT_DIR / "single_level_quality_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    missing_path = REPORT_DIR / "single_level_missing_rate_by_column.csv"
    _missing_rate_by_column(final_df).to_csv(missing_path, index=False)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(BASE_DIR),
        "raw_input_dir": str(BASE_SINGLE_RAW),
        "clean_output_dir": str(BASE_SINGLE_CLEAN),
        "final_output_file": str(OUTPUT_SINGLE_FINAL),
        "periods": list(SINGLE_LEVEL_PERIODS),
        "vietnam_lat_slice": VIETNAM_LAT_SLICE,
        "vietnam_lon_slice": VIETNAM_LON_SLICE,
        "resample_frequency": RESAMPLE_FREQUENCY,
        "tp_duplicate_handling": {
            "key": ["valid_time/time", "latitude", "longitude"],
            "aggregation": TP_DUPLICATE_AGG,
            "reason": (
                "Duplicate valid_time rows are collapsed before daily summation "
                "to prevent double counting precipitation from multiple GRIB steps."
            ),
        },
        "daily_precipitation_extreme_threshold_mm": EXTREME_DAILY_TP_MM,
        "tp_period_metadata": tp_period_metadata,
    }

    metadata_path = REPORT_DIR / "single_level_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\n[SAVED] Quality summary: {summary_path}")
    print(f"[SAVED] Missing-rate report: {missing_path}")
    print(f"[SAVED] Metadata: {metadata_path}")


# ============================================================================
# GSOD vs ERA5 PRECIPITATION COMPARISON
# ============================================================================

def _quantize_coordinates(
    lat: pd.Series,
    lon: pd.Series,
    resolution: float = GRID_RESOLUTION,
) -> Tuple[pd.Series, pd.Series]:
    """Map station coordinates to the nearest ERA5 grid cell."""
    q_lat = ((lat / resolution).round() * resolution).round(2)
    q_lon = ((lon / resolution).round() * resolution).round(2)
    return q_lat, q_lon


def _load_clean_gsod_precipitation(gsod_path: Path = BRONZE_GSOD_PATH) -> pd.DataFrame:
    """
    Load GSOD precipitation from bronze_data.csv and apply only the necessary
    cleaning for comparison with ERA5:
    - Convert DATE to time
    - Convert PRCP to numeric
    - Replace GSOD error code 99.99 with NaN
    - Convert inches to millimeters
    - Quantize coordinates to ERA5 grid
    """
    if not gsod_path.exists():
        raise FileNotFoundError(f"GSOD bronze file not found: {gsod_path}")

    cols_needed = ["STATION", "DATE", "LATITUDE", "LONGITUDE", "PRCP"]
    gsod = pd.read_csv(gsod_path, usecols=lambda c: c in cols_needed)

    required_missing = [col for col in cols_needed if col not in gsod.columns]
    if required_missing:
        raise ValueError(f"Missing required GSOD columns: {required_missing}")

    gsod["time"] = pd.to_datetime(gsod["DATE"], errors="coerce")
    gsod["PRCP"] = pd.to_numeric(gsod["PRCP"], errors="coerce")

    # Replace GSOD precipitation error code before unit conversion.
    gsod.loc[np.isclose(gsod["PRCP"], GSOD_PRCP_ERROR_CODE, atol=0.001), "PRCP"] = np.nan
    gsod["PRCP_mm"] = gsod["PRCP"] * GSOD_INCH_TO_MM

    gsod["latitude"], gsod["longitude"] = _quantize_coordinates(
        gsod["LATITUDE"],
        gsod["LONGITUDE"],
    )

    gsod = gsod.dropna(subset=["time", "latitude", "longitude"])
    return gsod[["STATION", "time", "latitude", "longitude", "PRCP_mm"]]


def compare_era5_tp_with_gsod_prcp(
    era5_daily: pd.DataFrame,
    gsod_path: Path = BRONZE_GSOD_PATH,
    make_plots: bool = True,
) -> Optional[pd.DataFrame]:
    """
    Compare ERA5 daily tp with GSOD station PRCP after coordinate quantization.

    The goal is not to force both sources to be identical. GSOD is station-based,
    while ERA5 is gridded reanalysis. The comparison is used to detect severe
    distributional mismatch or suspicious precipitation scaling errors.
    """
    if not gsod_path.exists():
        warnings.warn(
            f"GSOD file not found at {gsod_path}. Skipping ERA5-vs-GSOD comparison."
        )
        return None

    print("\n--- Comparing ERA5 `tp` with GSOD `PRCP` distribution...")

    gsod = _load_clean_gsod_precipitation(gsod_path)
    era5 = era5_daily[["time", "latitude", "longitude", "tp"]].copy()
    era5["time"] = pd.to_datetime(era5["time"], errors="coerce")

    merged = pd.merge(
        gsod,
        era5,
        on=["time", "latitude", "longitude"],
        how="inner",
    )

    if merged.empty:
        warnings.warn(
            "No overlapping GSOD-ERA5 records found after coordinate quantization. "
            "Check station coordinates and ERA5 grid resolution."
        )
        return None

    merged = merged.dropna(subset=["PRCP_mm", "tp"]).copy()

    if merged.empty:
        warnings.warn("No non-missing GSOD-ERA5 precipitation pairs available.")
        return None

    diff = merged["tp"] - merged["PRCP_mm"]

    summary = {
        "n_pairs": int(len(merged)),
        "n_stations": int(merged["STATION"].nunique()),
        "start_date": str(merged["time"].min().date()),
        "end_date": str(merged["time"].max().date()),
        "gsod_prcp_mean_mm": float(merged["PRCP_mm"].mean()),
        "era5_tp_mean_mm": float(merged["tp"].mean()),
        "gsod_prcp_median_mm": float(merged["PRCP_mm"].median()),
        "era5_tp_median_mm": float(merged["tp"].median()),
        "gsod_prcp_p95_mm": float(merged["PRCP_mm"].quantile(0.95)),
        "era5_tp_p95_mm": float(merged["tp"].quantile(0.95)),
        "gsod_prcp_p99_mm": float(merged["PRCP_mm"].quantile(0.99)),
        "era5_tp_p99_mm": float(merged["tp"].quantile(0.99)),
        "gsod_prcp_max_mm": float(merged["PRCP_mm"].max()),
        "era5_tp_max_mm": float(merged["tp"].max()),
        "bias_era5_minus_gsod_mm": float(diff.mean()),
        "mae_mm": float(np.mean(np.abs(diff))),
        "rmse_mm": float(np.sqrt(np.mean(diff ** 2))),
        "pearson_corr": float(merged[["PRCP_mm", "tp"]].corr(method="pearson").iloc[0, 1])
        if len(merged) > 2 else np.nan,
        "spearman_corr": float(merged[["PRCP_mm", "tp"]].corr(method="spearman").iloc[0, 1])
        if len(merged) > 2 else np.nan,
        "gsod_rain_rate_gt_0_1mm": float((merged["PRCP_mm"] > 0.1).mean()),
        "era5_rain_rate_gt_0_1mm": float((merged["tp"] > 0.1).mean()),
    }

    summary_df = pd.DataFrame([summary])
    summary_path = REPORT_DIR / "era5_tp_vs_gsod_prcp_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    station_summary = (
        merged.groupby("STATION")
        .agg(
            n_pairs=("tp", "size"),
            gsod_prcp_mean_mm=("PRCP_mm", "mean"),
            era5_tp_mean_mm=("tp", "mean"),
            gsod_prcp_p95_mm=("PRCP_mm", lambda x: x.quantile(0.95)),
            era5_tp_p95_mm=("tp", lambda x: x.quantile(0.95)),
            gsod_prcp_max_mm=("PRCP_mm", "max"),
            era5_tp_max_mm=("tp", "max"),
        )
        .reset_index()
    )
    station_summary_path = REPORT_DIR / "era5_tp_vs_gsod_prcp_by_station.csv"
    station_summary.to_csv(station_summary_path, index=False)

    sample_path = REPORT_DIR / "era5_tp_vs_gsod_prcp_pairs_sample.csv"
    merged.head(5000).to_csv(sample_path, index=False)

    print(f"[SAVED] ERA5-vs-GSOD summary: {summary_path}")
    print(f"[SAVED] ERA5-vs-GSOD station summary: {station_summary_path}")
    print(f"[SAVED] ERA5-vs-GSOD sample pairs: {sample_path}")

    if make_plots:
        _plot_era5_vs_gsod_precipitation(merged)

    return summary_df


def _plot_era5_vs_gsod_precipitation(merged: pd.DataFrame) -> None:
    """Save simple diagnostic plots comparing ERA5 tp and GSOD PRCP."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        warnings.warn(f"Could not import matplotlib. Skipping plots. Error: {exc}")
        return

    # Plot 1: log1p distribution comparison
    plt.figure(figsize=(8, 5))
    plt.hist(np.log1p(merged["PRCP_mm"]), bins=60, alpha=0.55, label="GSOD PRCP")
    plt.hist(np.log1p(merged["tp"]), bins=60, alpha=0.55, label="ERA5 tp")
    plt.xlabel("log(1 + daily precipitation mm)")
    plt.ylabel("Frequency")
    plt.title("Distribution Comparison: ERA5 tp vs GSOD PRCP")
    plt.legend()
    plt.tight_layout()
    dist_path = REPORT_DIR / "era5_tp_vs_gsod_prcp_distribution.png"
    plt.savefig(dist_path, dpi=200)
    plt.close()

    # Plot 2: log1p scatter
    sample = merged.sample(min(len(merged), 10000), random_state=108)
    plt.figure(figsize=(6, 6))
    plt.scatter(np.log1p(sample["PRCP_mm"]), np.log1p(sample["tp"]), alpha=0.35, s=8)
    plt.xlabel("GSOD log(1 + PRCP mm)")
    plt.ylabel("ERA5 log(1 + tp mm)")
    plt.title("Pointwise Comparison: ERA5 tp vs GSOD PRCP")
    plt.tight_layout()
    scatter_path = REPORT_DIR / "era5_tp_vs_gsod_prcp_scatter.png"
    plt.savefig(scatter_path, dpi=200)
    plt.close()

    print(f"[SAVED] Distribution plot: {dist_path}")
    print(f"[SAVED] Scatter plot: {scatter_path}")


# ============================================================================
# MAIN PERIOD PROCESSING
# ============================================================================

def process_single_level_period(
    period: str,
    strict_assertions: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Process one ERA5 single-level GRIB period.

    Steps:
    1. Load non-precipitation variables and resample them to daily frequency.
    2. Extract `tp` with `valid_time`, collapse duplicate valid_time rows, then
       sum to daily totals.
    3. Merge daily base variables and daily precipitation.
    4. Validate structure and precipitation sanity.
    5. Save intermediate Parquet file.
    """
    ensure_directories()

    grib_path = BASE_SINGLE_RAW / f"Single_level_{period}" / "data.grib"
    if not grib_path.exists():
        raise FileNotFoundError(f"GRIB file not found: {grib_path}")

    print(f"\n=== Processing ERA5 single-level period: {period} ===")

    base_hourly = _load_base_single_level_variables(grib_path)
    base_daily = _resample_base_variables_to_daily(base_hourly)

    tp_unique, tp_metadata = _extract_total_precipitation_hourly(grib_path)
    tp_daily = _resample_tp_to_daily(tp_unique)

    df_daily = pd.merge(
        base_daily,
        tp_daily,
        on=["time", "latitude", "longitude"],
        how="outer",
    ).sort_values(["latitude", "longitude", "time"]).reset_index(drop=True)

    validate_single_level_data(
        df_daily,
        dataset_name=f"single_level_{period}",
        strict_assertions=strict_assertions,
    )

    output_path = BASE_SINGLE_CLEAN / f"single_{period}.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_daily.to_parquet(output_path, index=False)

    print(f"[SAVED] Intermediate single-level file: {output_path}")
    print(f"        Shape: {df_daily.shape}")

    del base_hourly, base_daily, tp_unique, tp_daily
    gc.collect()

    return df_daily, tp_metadata


def process_single_levels(
    strict_assertions: bool = True,
    compare_with_gsod: bool = True,
    make_plots: bool = True,
) -> pd.DataFrame:
    """
    Main pipeline for processing ERA5 single-level data.

    Final outputs:
    - data/clean/single_level/single_1520.parquet
    - data/clean/single_level/single_2125.parquet
    - data/clean/ERA5_single_level.parquet
    - reports/data_quality/single_level/*.csv
    - reports/data_quality/single_level/*.json
    - optional plots comparing ERA5 tp with GSOD PRCP
    """
    ensure_directories()

    print("\n=== START ERA5 SINGLE-LEVEL PROCESSING ===")
    print(f"Project root: {BASE_DIR}")
    print(f"Raw input directory: {BASE_SINGLE_RAW}")
    print(f"Clean output directory: {BASE_SINGLE_CLEAN}")

    all_periods: List[pd.DataFrame] = []
    quality_summaries: List[Dict[str, object]] = []
    tp_metadata_all: List[Dict[str, object]] = []

    for period in SINGLE_LEVEL_PERIODS:
        df_period, tp_metadata = process_single_level_period(
            period,
            strict_assertions=strict_assertions,
        )

        quality_summaries.append(
            _summarize_single_level_quality(
                df_period,
                dataset_name=f"single_level_{period}",
                tp_metadata=tp_metadata,
            )
        )
        tp_metadata_all.append(tp_metadata)
        all_periods.append(df_period)

    print("\n--- Concatenating all single-level periods...")
    final_df = (
        pd.concat(all_periods, ignore_index=True)
        .sort_values(["latitude", "longitude", "time"])
        .reset_index(drop=True)
    )

    # If overlapping periods exist, this catches/blocks duplicate daily grid rows.
    validate_single_level_data(
        final_df,
        dataset_name="ERA5_single_level_final",
        strict_assertions=strict_assertions,
    )

    # Add final summary after validation.
    quality_summaries.append(
        _summarize_single_level_quality(
            final_df,
            dataset_name="ERA5_single_level_final",
            tp_metadata=None,
        )
    )

    OUTPUT_SINGLE_FINAL.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(OUTPUT_SINGLE_FINAL, index=False)
    print(f"\n[SUCCESS] Final ERA5 single-level file saved to: {OUTPUT_SINGLE_FINAL}")
    print(f"          Final shape: {final_df.shape}")

    _save_quality_reports(
        quality_summaries=quality_summaries,
        final_df=final_df,
        tp_period_metadata=tp_metadata_all,
    )

    if compare_with_gsod:
        compare_era5_tp_with_gsod_prcp(
            final_df,
            gsod_path=BRONZE_GSOD_PATH,
            make_plots=make_plots,
        )

    del all_periods
    gc.collect()

    return final_df


if __name__ == "__main__":
    # Keep strict_assertions=True for final reproducible runs.
    # If you are still debugging raw GRIB files, temporarily set it to False.
    process_single_levels(
        strict_assertions=True,
        compare_with_gsod=True,
        make_plots=True,
    )
