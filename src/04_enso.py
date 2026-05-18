"""
ENSO / MEI v2 processing module with leakage-aware lag features.

This module processes raw MEI v2 ENSO data and produces a clean monthly table
for merging with daily weather data.

Key improvements over the initial version:
1. Explain and encode the MEI two-month overlapping seasons:
   DJ, JF, FM, ..., ND are bimonthly windows. We assign each window to the
   ending month of the pair: DJ -> January (1), JF -> February (2), ..., ND ->
   December (12).
2. Synchronize final ENSO output with the main dataset period: 2015-2024.
3. Use an extended loading window to compute lag features safely.
4. Create ENSO_lag_1 and ENSO_lag_2 for forecast-safe modeling.
5. Save metadata and quality reports.
6. Optionally check missing ENSO values after merging with a daily dataset.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

# Main dataset period. This should match the GSOD/ERA5 study period.
MAIN_YEAR_START = 2015
MAIN_YEAR_END = 2024

# Extended year range used internally to compute lag features.
# Example: ENSO_lag_1 for Jan 2015 needs the Dec 2014 MEI window if available.
LAG_PERIODS = [1, 2]
MAX_LAG = max(LAG_PERIODS)

# MEI v2 uses two-month overlapping seasons:
# DJ = Dec-Jan, JF = Jan-Feb, FM = Feb-Mar, ..., ND = Nov-Dec.
#
# We assign each bimonthly value to the ENDING month of the pair:
#   DJ -> January  (month 1)
#   JF -> February (month 2)
#   ...
#   ND -> December (month 12)
#
# Why this mapping?
# - It creates one MEI value per calendar month.
# - It keeps chronological order simple for merging with daily weather data.
# - It avoids treating DJ as "December" of the same row's YEAR, which would
#   shift the value backward and make merge logic more confusing.
MONTH_MAPPING = {
    "DJ": 1,
    "JF": 2,
    "FM": 3,
    "MA": 4,
    "AM": 5,
    "MJ": 6,
    "JJ": 7,
    "JA": 8,
    "AS": 9,
    "SO": 10,
    "ON": 11,
    "ND": 12,
}

EXPECTED_MONTH_COLUMNS = list(MONTH_MAPPING.keys())

# Output filenames
ENSO_OUTPUT_NAME = "enso_clean.csv"
ENSO_METADATA_NAME = "enso_metadata.json"
ENSO_QUALITY_SUMMARY_NAME = "enso_quality_summary.csv"
ENSO_MONTHLY_COVERAGE_NAME = "enso_monthly_coverage.csv"
ENSO_MERGE_COVERAGE_NAME = "enso_merge_coverage.csv"

# Optional default path for checking merge coverage.
# If this file does not exist, the merge coverage check is skipped.
DEFAULT_DAILY_DATASET_FOR_MERGE_CHECK = "silver_data_ver2.csv"


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

    return Path(__file__).resolve().parent


BASE_DIR = find_project_root()
INPUT_FILE = BASE_DIR / "data" / "raw" / "meiv2.data"
CLEAN_DIR = BASE_DIR / "data" / "clean"
OUTPUT_FILE = CLEAN_DIR / ENSO_OUTPUT_NAME
REPORT_DIR = BASE_DIR / "reports" / "data_quality" / "enso"


def ensure_directories() -> None:
    """Create required output/report directories."""
    CLEAN_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# CORE PROCESSING
# ============================================================================

def load_raw_enso_data(input_path: Path = INPUT_FILE) -> pd.DataFrame:
    """
    Load raw MEI v2 data from a whitespace-separated file.

    Expected format:
        YEAR  DJ  JF  FM  MA  AM  MJ  JJ  JA  AS  SO  ON  ND
        1979 ...
        1980 ...
        ...
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Raw ENSO file not found: {input_path}")

    df = pd.read_csv(input_path, sep=r"\s+", engine="python")
    df.columns = [str(col).strip().upper() for col in df.columns]

    if "YEAR" not in df.columns:
        raise ValueError("Raw ENSO file must contain a YEAR column.")

    missing_month_cols = [col for col in EXPECTED_MONTH_COLUMNS if col not in df.columns]
    if missing_month_cols:
        raise ValueError(
            f"Raw ENSO file is missing MEI month-window columns: {missing_month_cols}"
        )

    df["YEAR"] = pd.to_numeric(df["YEAR"], errors="coerce").astype("Int64")
    return df


def filter_extended_year_range(
    df: pd.DataFrame,
    main_year_start: int = MAIN_YEAR_START,
    main_year_end: int = MAIN_YEAR_END,
    max_lag: int = MAX_LAG,
) -> pd.DataFrame:
    """
    Filter a slightly extended year range to compute lag features.

    The final output is still clipped to 2015-2024, but internally we keep at
    least one earlier year if available so Jan/Feb 2015 lags can be calculated.
    """
    extended_start = main_year_start - int(np.ceil(max_lag / 12)) - 1
    extended_end = main_year_end

    out = df[(df["YEAR"] >= extended_start) & (df["YEAR"] <= extended_end)].copy()
    return out.reset_index(drop=True)


def reshape_mei_wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert MEI v2 from wide bimonthly format to tidy monthly format.

    Input:
        YEAR | DJ | JF | FM | ... | ND

    Output:
        YEAR | MONTH | MEI_WINDOW | ENSO

    MONTH is assigned using ending-month convention:
        DJ -> 1, JF -> 2, ..., ND -> 12.
    """
    keep_cols = ["YEAR"] + EXPECTED_MONTH_COLUMNS
    df = df[keep_cols].copy()

    long_df = df.melt(
        id_vars="YEAR",
        value_vars=EXPECTED_MONTH_COLUMNS,
        var_name="MEI_WINDOW",
        value_name="ENSO",
    )

    long_df["MONTH"] = long_df["MEI_WINDOW"].map(MONTH_MAPPING).astype(int)
    long_df["ENSO"] = pd.to_numeric(long_df["ENSO"], errors="coerce")

    long_df = (
        long_df.sort_values(["YEAR", "MONTH"])
        .reset_index(drop=True)
        [["YEAR", "MONTH", "MEI_WINDOW", "ENSO"]]
    )

    return long_df


def add_month_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a continuous monthly index for chronological sorting and lag creation.

    month_index = YEAR * 12 + MONTH
    """
    out = df.copy()
    out["month_index"] = out["YEAR"].astype(int) * 12 + out["MONTH"].astype(int)
    out = out.sort_values("month_index").reset_index(drop=True)
    return out


def add_enso_lag_features(
    df: pd.DataFrame,
    lags: Iterable[int] = LAG_PERIODS,
) -> pd.DataFrame:
    """
    Add ENSO lag features.

    ENSO_lag_1 means the ENSO value from the previous MEI monthly slot.
    ENSO_lag_2 means the ENSO value from two MEI monthly slots earlier.

    For forecast-oriented modeling, prefer lagged ENSO variables because the
    current month's official MEI value may not be known at prediction time.
    """
    out = df.sort_values("month_index").reset_index(drop=True).copy()

    for lag in lags:
        out[f"ENSO_lag_{lag}"] = out["ENSO"].shift(lag)

    return out


def clip_to_main_period(
    df: pd.DataFrame,
    year_start: int = MAIN_YEAR_START,
    year_end: int = MAIN_YEAR_END,
) -> pd.DataFrame:
    """Return final ENSO table synchronized with the main dataset period."""
    out = df[(df["YEAR"] >= year_start) & (df["YEAR"] <= year_end)].copy()
    return out.reset_index(drop=True)


def process_enso_data(
    input_path: Path = INPUT_FILE,
    output_path: Path = OUTPUT_FILE,
    main_year_start: int = MAIN_YEAR_START,
    main_year_end: int = MAIN_YEAR_END,
    lags: Iterable[int] = LAG_PERIODS,
    include_current_enso: bool = True,
    run_merge_check: bool = True,
    daily_dataset_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Main ENSO processing pipeline.

    Parameters
    ----------
    input_path:
        Raw MEI v2 file path.
    output_path:
        Clean ENSO CSV output path.
    main_year_start, main_year_end:
        Final output period. Default is 2015-2024 to match the main dataset.
    lags:
        ENSO lag periods to generate.
    include_current_enso:
        If True, keep current `ENSO` column. For strict future forecasting,
        use lag columns instead of current ENSO unless current-month MEI is
        known at prediction time.
    run_merge_check:
        If True, check missing ENSO after merging with a daily dataset if the
        dataset exists.
    daily_dataset_path:
        Optional daily dataset for merge coverage check.
    """
    ensure_directories()

    print("\n=== PROCESSING ENSO / MEI v2 DATA ===")
    print(f"Input file: {input_path}")
    print(f"Main synchronized period: {main_year_start}-{main_year_end}")
    print("MEI mapping convention: DJ->1, JF->2, ..., ND->12 using ending month.")

    raw = load_raw_enso_data(input_path)
    extended = filter_extended_year_range(
        raw,
        main_year_start=main_year_start,
        main_year_end=main_year_end,
        max_lag=max(lags) if lags else 0,
    )

    long_df = reshape_mei_wide_to_long(extended)
    long_df = add_month_index(long_df)
    long_df = add_enso_lag_features(long_df, lags=lags)
    final = clip_to_main_period(
        long_df,
        year_start=main_year_start,
        year_end=main_year_end,
    )

    # Keep a clean order of columns.
    lag_cols = [f"ENSO_lag_{lag}" for lag in lags]
    output_cols = ["YEAR", "MONTH", "MEI_WINDOW"]
    if include_current_enso:
        output_cols.append("ENSO")
    output_cols += lag_cols
    output_cols += ["month_index"]

    final = final[output_cols].copy()

    validate_enso_table(
        final,
        main_year_start=main_year_start,
        main_year_end=main_year_end,
        lags=lags,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_csv(output_path, index=False)

    print(f"[SAVED] Clean ENSO data: {output_path}")
    print(f"        Shape: {final.shape}")

    save_enso_reports(
        final=final,
        raw=raw,
        output_path=output_path,
        input_path=input_path,
        main_year_start=main_year_start,
        main_year_end=main_year_end,
        lags=list(lags),
        include_current_enso=include_current_enso,
    )

    if run_merge_check:
        if daily_dataset_path is None:
            daily_dataset_path = CLEAN_DIR / DEFAULT_DAILY_DATASET_FOR_MERGE_CHECK
        check_enso_merge_coverage(final, daily_dataset_path=daily_dataset_path)

    return final


# ============================================================================
# VALIDATION AND REPORTING
# ============================================================================

def validate_enso_table(
    df: pd.DataFrame,
    main_year_start: int = MAIN_YEAR_START,
    main_year_end: int = MAIN_YEAR_END,
    lags: Iterable[int] = LAG_PERIODS,
) -> None:
    """
    Validate structural quality of final ENSO table.
    """
    required_cols = {"YEAR", "MONTH", "MEI_WINDOW", "month_index"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise AssertionError(f"ENSO output missing required columns: {missing_cols}")

    duplicate_count = int(df.duplicated(subset=["YEAR", "MONTH"]).sum())
    if duplicate_count > 0:
        raise AssertionError(
            f"ENSO output contains {duplicate_count} duplicate YEAR-MONTH rows."
        )

    expected_months = (main_year_end - main_year_start + 1) * 12
    actual_months = int(len(df))

    if actual_months != expected_months:
        raise AssertionError(
            f"ENSO output should contain {expected_months} monthly rows, "
            f"but got {actual_months}."
        )

    if not df["MONTH"].between(1, 12).all():
        raise AssertionError("MONTH must be in the range 1..12.")

    # Check chronological continuity.
    sorted_idx = df.sort_values("month_index")["month_index"].to_numpy()
    if len(sorted_idx) > 1 and not np.all(np.diff(sorted_idx) == 1):
        raise AssertionError(
            "ENSO month_index is not continuous. Some monthly rows may be missing."
        )

    # Missing current ENSO should not happen for a complete MEI source.
    if "ENSO" in df.columns:
        enso_missing = int(df["ENSO"].isna().sum())
        if enso_missing > 0:
            warnings.warn(f"ENSO has {enso_missing} missing monthly values.")

    # Lag features may have a small number of missing values at the beginning if
    # the raw file does not include enough previous years. This is reported, not
    # automatically fixed.
    for lag in lags:
        col = f"ENSO_lag_{lag}"
        if col in df.columns:
            missing = int(df[col].isna().sum())
            if missing > 0:
                warnings.warn(
                    f"{col} has {missing} missing values. This usually occurs at "
                    "the beginning of the study period if previous-year MEI is "
                    "not available. Report this explicitly or drop the first "
                    "affected months in forecasting experiments."
                )


def save_enso_reports(
    final: pd.DataFrame,
    raw: pd.DataFrame,
    output_path: Path,
    input_path: Path,
    main_year_start: int,
    main_year_end: int,
    lags: List[int],
    include_current_enso: bool,
) -> None:
    """
    Save ENSO metadata and quality summaries.
    """
    ensure_directories()

    enso_cols = [col for col in final.columns if col.startswith("ENSO")]
    quality_rows = []

    for col in enso_cols:
        s = pd.to_numeric(final[col], errors="coerce")
        quality_rows.append({
            "column": col,
            "missing_count": int(s.isna().sum()),
            "missing_rate": float(s.isna().mean()),
            "min": float(s.min()) if s.notna().any() else np.nan,
            "mean": float(s.mean()) if s.notna().any() else np.nan,
            "median": float(s.median()) if s.notna().any() else np.nan,
            "max": float(s.max()) if s.notna().any() else np.nan,
            "std": float(s.std()) if s.notna().any() else np.nan,
        })

    quality_df = pd.DataFrame(quality_rows)
    quality_path = REPORT_DIR / ENSO_QUALITY_SUMMARY_NAME
    quality_df.to_csv(quality_path, index=False)

    coverage = final[["YEAR", "MONTH", "MEI_WINDOW", *enso_cols]].copy()
    coverage["has_current_enso"] = (
        coverage["ENSO"].notna() if "ENSO" in coverage.columns else np.nan
    )

    for lag in lags:
        col = f"ENSO_lag_{lag}"
        if col in coverage.columns:
            coverage[f"has_{col}"] = coverage[col].notna()

    coverage_path = REPORT_DIR / ENSO_MONTHLY_COVERAGE_NAME
    coverage.to_csv(coverage_path, index=False)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_file": str(input_path),
        "output_file": str(output_path),
        "raw_year_min": int(raw["YEAR"].min()) if raw["YEAR"].notna().any() else None,
        "raw_year_max": int(raw["YEAR"].max()) if raw["YEAR"].notna().any() else None,
        "main_year_start": main_year_start,
        "main_year_end": main_year_end,
        "final_monthly_rows": int(len(final)),
        "expected_monthly_rows": int((main_year_end - main_year_start + 1) * 12),
        "mei_mapping_convention": (
            "Bimonthly MEI windows are assigned to the ending month: "
            "DJ->January, JF->February, ..., ND->December."
        ),
        "leakage_note": (
            "For historical analysis, current ENSO can be used after publication. "
            "For strict future forecasting, prefer ENSO_lag_1 or ENSO_lag_2 "
            "unless current-month MEI is known at prediction time."
        ),
        "lag_features": [f"ENSO_lag_{lag}" for lag in lags],
        "include_current_enso": include_current_enso,
    }

    metadata_path = REPORT_DIR / ENSO_METADATA_NAME
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[SAVED] ENSO quality summary: {quality_path}")
    print(f"[SAVED] ENSO monthly coverage: {coverage_path}")
    print(f"[SAVED] ENSO metadata: {metadata_path}")


# ============================================================================
# MERGE COVERAGE CHECK
# ============================================================================

def check_enso_merge_coverage(
    enso: pd.DataFrame,
    daily_dataset_path: Path,
) -> Optional[pd.DataFrame]:
    """
    Check missing ENSO after merging with a daily dataset.

    This function is useful after the Silver dataset exists. It performs a left
    merge from daily data to ENSO and reports how many daily rows fail to match
    an ENSO month.
    """
    if not daily_dataset_path.exists():
        warnings.warn(
            f"Daily dataset not found: {daily_dataset_path}. "
            "Skipping ENSO merge coverage check. Run this check after the "
            "Silver dataset is created."
        )
        return None

    print(f"\n--- Checking ENSO merge coverage with: {daily_dataset_path}")

    if daily_dataset_path.suffix.lower() == ".parquet":
        daily = pd.read_parquet(daily_dataset_path)
    else:
        daily = pd.read_csv(daily_dataset_path)

    if "time" not in daily.columns:
        raise ValueError("Daily dataset must contain a `time` column for ENSO merge check.")

    daily = daily.copy()
    daily["time"] = pd.to_datetime(daily["time"], errors="coerce")
    daily["YEAR"] = daily["time"].dt.year
    daily["MONTH"] = daily["time"].dt.month

    enso_cols = [col for col in enso.columns if col.startswith("ENSO")]
    merge_cols = ["YEAR", "MONTH", *enso_cols]

    merged = pd.merge(
        daily[["YEAR", "MONTH"]],
        enso[merge_cols],
        how="left",
        on=["YEAR", "MONTH"],
    )

    rows = []
    for col in enso_cols:
        rows.append({
            "daily_dataset_path": str(daily_dataset_path),
            "enso_column": col,
            "daily_rows": int(len(merged)),
            "missing_after_merge_count": int(merged[col].isna().sum()),
            "missing_after_merge_rate": float(merged[col].isna().mean()),
            "covered_rows": int(merged[col].notna().sum()),
        })

    report = pd.DataFrame(rows)
    report_path = REPORT_DIR / ENSO_MERGE_COVERAGE_NAME
    report.to_csv(report_path, index=False)

    print(f"[SAVED] ENSO merge coverage report: {report_path}")

    return report


if __name__ == "__main__":
    process_enso_data(
        input_path=INPUT_FILE,
        output_path=OUTPUT_FILE,
        main_year_start=MAIN_YEAR_START,
        main_year_end=MAIN_YEAR_END,
        lags=LAG_PERIODS,
        include_current_enso=True,
        run_merge_check=True,
        daily_dataset_path=CLEAN_DIR / DEFAULT_DAILY_DATASET_FOR_MERGE_CHECK,
    )
