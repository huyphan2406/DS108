"""
DS108 - Final model script: full scenario comparison with 5 metrics
====================================================================

Purpose
-------
Create ONE final comparison table for daily rainfall amount prediction.
The script compares two complete prediction scenarios on all test days.

Scenario 1: LightGBM_1stage_Tweedie
    - One-stage baseline.
    - Trained on all training days.
    - Target: PRCP in millimeters.
    - Final prediction: direct PRCP prediction.

Scenario 2: LightGBM_2stage_expected
    - Full two-stage hurdle model.
    - Stage 1: LightGBM classifier trained on all training days.
      Output: P(rain | X).
    - Stage 2: LightGBM regressor trained ONLY on rainy training days.
      Target: PRCP_log1p = log(1 + PRCP).
      Output is converted back to millimeters with expm1.
    - Final prediction: P(rain | X) * E(PRCP | rain, X).

Final output
------------
outputs/model_final_single_table/final_model_comparison.csv

Run
---
python src/_08_model.py
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError as exc:  # pragma: no cover
    raise ImportError("LightGBM is required. Install it with: pip install lightgbm") from exc

warnings.filterwarnings("ignore")


# =============================================================================
# Project paths
# =============================================================================
SCRIPT_PATH = Path(__file__).resolve()
BASE_DIR = SCRIPT_PATH.parent.parent if SCRIPT_PATH.parent.name.lower() == "src" else SCRIPT_PATH.parent

DEFAULT_INPUT_CSV = BASE_DIR / "data" / "feature_engineering" / "feature_engineered_data.csv"
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs" / "model_final_single_table"


# =============================================================================
# Columns and split rules
# =============================================================================
TIME_COL = "time"
STATION_COL = "STATION"
TARGET_MM = "PRCP"
TARGET_CLASS = "PRCP_label"
TARGET_LOG = "PRCP_log1p"
RAIN_THRESHOLD_MM = 0.1

# These columns must never be used as input features because they are metadata,
# identifiers, timestamps, current-day targets, or columns derived directly from
# the current-day target.
DROP_COLS = [
    STATION_COL,
    TIME_COL,
    "date",
    "Target",
    TARGET_MM,
    TARGET_CLASS,
    TARGET_LOG,
]

TRAIN_END = pd.Timestamp("2023-01-01")  # train: 2015-2022
VALID_END = pd.Timestamp("2024-01-01")  # valid: 2023, test: 2024

FINAL_TABLE_COLUMNS = [
    "model",
    "mae_mm",
    "rmse_mm",
    "wape_percent",
    "r2",
    "bias_mean_pred_minus_true_mm",
]


# =============================================================================
# Basic utilities
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare full 1-stage Tweedie and full 2-stage hurdle rainfall models."
    )
    parser.add_argument("--input-csv", type=str, default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--tweedie-power", type=float, default=1.3)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def resolve_path(path_like: Union[str, Path]) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (BASE_DIR / path).resolve()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clean_old_outputs(output_dir: Path) -> None:
    """Remove known old artifacts from previous versions to avoid confusion."""
    old_files = [
        "final_model_comparison.csv",
        "rainy_day_regression_comparison.csv",
        "stage1_classifier_support_metrics.csv",
        "end_to_end_comparison.csv",
        "test_predictions_all_days.csv",
        "test_rainy_day_predictions.csv",
        "used_full_features.csv",
        "used_features.csv",
        "feature_importance_1stage_tweedie.csv",
        "feature_importance_stage1_classifier.csv",
        "feature_importance_stage2_regression.csv",
        "feature_importance_stage2_regressor.csv",
        "model_config.json",
    ]
    for filename in old_files:
        file_path = output_dir / filename
        if file_path.exists() and file_path.is_file():
            file_path.unlink()


def round_numeric(df: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].round(digits)
    return out


# =============================================================================
# Data preparation
# =============================================================================
def load_and_prepare_data(input_csv: Path) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    df = pd.read_csv(input_csv)

    required_cols = {TIME_COL, TARGET_MM}
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    if df[TIME_COL].isna().any():
        raise ValueError(f"Column '{TIME_COL}' contains invalid datetime values.")

    df[TARGET_MM] = pd.to_numeric(df[TARGET_MM], errors="coerce")
    if df[TARGET_MM].isna().any():
        raise ValueError(f"Column '{TARGET_MM}' contains missing or non-numeric values.")
    if not np.isfinite(df[TARGET_MM].values).all():
        raise ValueError(f"Column '{TARGET_MM}' contains inf/-inf values.")
    if (df[TARGET_MM] < 0).any():
        n_bad = int((df[TARGET_MM] < 0).sum())
        raise ValueError(f"Column '{TARGET_MM}' contains {n_bad} negative values.")

    # Recreate target columns from PRCP to guarantee consistency.
    # This also handles the case where PRCP_label or PRCP_log1p is missing.
    df[TARGET_CLASS] = (df[TARGET_MM] > RAIN_THRESHOLD_MM).astype(int)
    df[TARGET_LOG] = np.log1p(df[TARGET_MM])

    return df.sort_values(TIME_COL).reset_index(drop=True)


def time_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df[df[TIME_COL] < TRAIN_END].copy()
    valid_df = df[(df[TIME_COL] >= TRAIN_END) & (df[TIME_COL] < VALID_END)].copy()
    test_df = df[df[TIME_COL] >= VALID_END].copy()

    if train_df.empty or valid_df.empty or test_df.empty:
        raise ValueError("Time split produced an empty train/valid/test subset.")

    return train_df, valid_df, test_df


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    feature_cols = [
        col
        for col in df.columns
        if col not in DROP_COLS and pd.api.types.is_numeric_dtype(df[col])
    ]

    if not feature_cols:
        raise ValueError("No numeric feature columns found after dropping leakage columns.")

    leakage = sorted(set(feature_cols) & set(DROP_COLS))
    if leakage:
        raise AssertionError(f"Leakage columns remain in features: {leakage}")

    return feature_cols


def make_x(df: pd.DataFrame, feature_cols: Iterable[str]) -> pd.DataFrame:
    return df.loc[:, list(feature_cols)].replace([np.inf, -np.inf], np.nan)


def clip_nonnegative(values: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(values, dtype=float), 0.0)


# =============================================================================
# Model builders
# =============================================================================
def build_one_stage_tweedie(args: argparse.Namespace) -> Pipeline:
    """One-stage baseline: one model predicts PRCP directly on all training days."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                LGBMRegressor(
                    objective="tweedie",
                    tweedie_variance_power=args.tweedie_power,
                    n_estimators=args.n_estimators,
                    learning_rate=args.learning_rate,
                    num_leaves=args.num_leaves,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    random_state=args.random_state,
                    n_jobs=args.n_jobs,
                    verbose=-1,
                ),
            ),
        ]
    )


def build_stage1_classifier(args: argparse.Namespace) -> Pipeline:
    """Stage 1 of the hurdle model: predict P(rain | X) on all training days."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                LGBMClassifier(
                    objective="binary",
                    n_estimators=args.n_estimators,
                    learning_rate=args.learning_rate,
                    num_leaves=args.num_leaves,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    class_weight="balanced",
                    random_state=args.random_state,
                    n_jobs=args.n_jobs,
                    verbose=-1,
                ),
            ),
        ]
    )


def build_stage2_regressor(args: argparse.Namespace) -> Pipeline:
    """Stage 2 of the hurdle model: predict log1p(PRCP) only for rainy days."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                LGBMRegressor(
                    objective="regression",
                    n_estimators=args.n_estimators,
                    learning_rate=args.learning_rate,
                    num_leaves=args.num_leaves,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    random_state=args.random_state,
                    n_jobs=args.n_jobs,
                    verbose=-1,
                ),
            ),
        ]
    )


# =============================================================================
# Prediction helpers
# =============================================================================
def predict_one_stage_mm(model: Pipeline, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    pred = model.predict(make_x(df, feature_cols))
    return clip_nonnegative(pred)


def predict_stage1_prob(model: Pipeline, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    prob = model.predict_proba(make_x(df, feature_cols))[:, 1]
    return np.asarray(prob, dtype=float)


def predict_stage2_amount_mm(model: Pipeline, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    pred_log = model.predict(make_x(df, feature_cols))
    pred_mm = np.expm1(pred_log)
    return clip_nonnegative(pred_mm)


# =============================================================================
# Final metrics: only MAE, RMSE, WAPE, R2, Bias
# =============================================================================
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = float(np.sum(np.abs(y_true)))
    if denominator <= 0:
        return np.nan
    return float(np.sum(np.abs(y_true - y_pred)) / denominator * 100.0)


def final_metrics_row(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    model: str,
) -> Dict[str, Union[str, int, float]]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    row = {
        "model": model,
        "mae_mm": float(mean_absolute_error(y_true, y_pred)),
        "rmse_mm": rmse(y_true, y_pred),
        "wape_percent": wape(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
        "bias_mean_pred_minus_true_mm": float(np.mean(y_pred - y_true)),
    }
    return row


# =============================================================================
# Main pipeline
# =============================================================================
def main() -> None:
    args = parse_args()

    input_csv = resolve_path(args.input_csv)
    output_dir = resolve_path(args.output_dir)
    ensure_dir(output_dir)
    clean_old_outputs(output_dir)

    print(f"Input : {input_csv}")
    print(f"Output: {output_dir}")

    df = load_and_prepare_data(input_csv)
    train_df, valid_df, test_df = time_split(df)
    feature_cols = select_feature_columns(df)

    print("\nTime split")
    print(f"Train: {train_df[TIME_COL].min().date()} -> {train_df[TIME_COL].max().date()} | n={len(train_df):,}")
    print(f"Valid: {valid_df[TIME_COL].min().date()} -> {valid_df[TIME_COL].max().date()} | n={len(valid_df):,}")
    print(f"Test : {test_df[TIME_COL].min().date()} -> {test_df[TIME_COL].max().date()} | n={len(test_df):,}")
    print(f"\nUsing FULL features only: {len(feature_cols)} numeric features")

    # -------------------------------------------------------------------------
    # Scenario 1: LightGBM 1-stage Tweedie regression
    # -------------------------------------------------------------------------
    print("\n[1/2] Training Scenario 1: LightGBM 1-stage Tweedie regression...")
    one_stage = build_one_stage_tweedie(args)
    one_stage.fit(
        make_x(train_df, feature_cols),
        train_df[TARGET_MM].values.astype(float),
    )

    # -------------------------------------------------------------------------
    # Scenario 2: LightGBM full 2-stage hurdle model
    # -------------------------------------------------------------------------
    print("[2/2] Training Scenario 2: LightGBM full 2-stage hurdle model...")

    # Stage 1 is trained on all training days.
    stage1 = build_stage1_classifier(args)
    stage1.fit(
        make_x(train_df, feature_cols),
        train_df[TARGET_CLASS].values.astype(int),
    )

    # Stage 2 is trained ONLY on rainy training days.
    rainy_train_mask = train_df["PRCP_label"].astype(int).values == 1
    if rainy_train_mask.sum() == 0:
        raise ValueError("No rainy days in training data. Stage 2 cannot be trained.")

    stage2 = build_stage2_regressor(args)
    stage2.fit(
        make_x(train_df.loc[rainy_train_mask], feature_cols),
        train_df.loc[rainy_train_mask, TARGET_LOG].values.astype(float),
    )

    # -------------------------------------------------------------------------
    # Final predictions on all test days
    # -------------------------------------------------------------------------
    y_test = test_df[TARGET_MM].values.astype(float)

    # Scenario 1 final prediction.
    pred_1stage_test = predict_one_stage_mm(one_stage, test_df, feature_cols)

    # Scenario 2 full hurdle final prediction.
    pred_stage1_prob_test = predict_stage1_prob(stage1, test_df, feature_cols)
    pred_stage2_amount_test = predict_stage2_amount_mm(stage2, test_df, feature_cols)

    # Required full two-stage formula:
    # E(PRCP | X) = P(rain | X) * E(PRCP | rain, X)
    pred_2stage_expected_test = pred_stage1_prob_test * pred_stage2_amount_test

    # -------------------------------------------------------------------------
    # ONE FINAL COMPARISON TABLE: two complete scenarios on all test days
    # -------------------------------------------------------------------------
    final_df = pd.DataFrame(
        [
            final_metrics_row(
                y_test,
                pred_1stage_test,
                model="LightGBM_1stage_Tweedie",
            ),
            final_metrics_row(
                y_test,
                pred_2stage_expected_test,
                model="LightGBM_2stage_expected",
            ),
        ],
        columns=FINAL_TABLE_COLUMNS,
    )

    final_path = output_dir / "final_model_comparison.csv"
    final_df.to_csv(final_path, index=False)

    # JSON config is not a comparison table; it is saved only for reproducibility.
    config = {
        "main_output": str(final_path),
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "final_comparison": "LightGBM_1stage_Tweedie vs LightGBM_2stage_expected on all_test_days",
        "final_table_columns": FINAL_TABLE_COLUMNS,
        "metrics": ["MAE", "RMSE", "WAPE", "R2", "Bias"],
        "target": TARGET_MM,
        "rain_label_definition": f"{TARGET_MM} > {RAIN_THRESHOLD_MM} mm",
        "two_stage_formula": "pred_2stage_expected_test = pred_stage1_prob_test * pred_stage2_amount_test",
        "stage2_training_rule": "Stage 2 is trained only on train rows where PRCP_label == 1.",
        "split": {
            "train": [str(train_df[TIME_COL].min().date()), str(train_df[TIME_COL].max().date())],
            "valid": [str(valid_df[TIME_COL].min().date()), str(valid_df[TIME_COL].max().date())],
            "test": [str(test_df[TIME_COL].min().date()), str(test_df[TIME_COL].max().date())],
        },
        "n_rows": {
            "train": int(len(train_df)),
            "valid": int(len(valid_df)),
            "test": int(len(test_df)),
            "stage2_train_rainy_days": int(rainy_train_mask.sum()),
        },
        "n_features": int(len(feature_cols)),
        "drop_cols_from_features": DROP_COLS,
        "lightgbm_params": {
            "n_estimators": args.n_estimators,
            "learning_rate": args.learning_rate,
            "num_leaves": args.num_leaves,
            "tweedie_power": args.tweedie_power,
            "random_state": args.random_state,
            "n_jobs": args.n_jobs,
        },
    }
    with open(output_dir / "model_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 180)

    print("\n" + "=" * 100)
    print("FINAL TABLE: full scenario comparison on ALL TEST DAYS")
    print("Metrics: MAE, RMSE, WAPE, R2, Bias")
    print("=" * 100)
    print(round_numeric(final_df, 4))

    print("\nDone.")
    print(f"Final comparison table: {final_path}")
    print(f"Config               : {output_dir / 'model_config.json'}")


if __name__ == "__main__":
    main()
