"""
DS108 - Final model validation script
=====================================

Purpose
-------
Validate the rainfall benchmark with two complete LightGBM scenarios.

Important logic
---------------
- PRCP is treated as the rainfall target from the feature-engineered dataset.
  It is not filled, interpolated, or replaced inside this script.
- PRCP_label must follow the fixed rule: 1 if PRCP >= 1.0 mm, else 0.
  Missing, non-binary, or inconsistent labels are treated as data errors.
- Scenario-specific target columns may already exist from _07_feature_engineering.py,
  but this script validates and recreates them to guarantee consistency.
- The validation set is used for early stopping / model selection only.
  It is NOT used to choose a rainfall threshold because the threshold is fixed.
- No CSV/JSON output is written. The final tables are printed to console.

Scenario 1: LightGBM_1stage_Tweedie
    target_1stage_prcp_mm = PRCP

Scenario 2: LightGBM_2stage_expected
    target_stage1_rain_label = PRCP_label
    target_stage2_prcp_log1p = log1p(PRCP)
    final prediction = P(rain | X) * E(PRCP | rain, X)
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Iterable, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation
except ImportError as exc:  # pragma: no cover
    raise ImportError("LightGBM is required. Install it with: pip install lightgbm") from exc

warnings.filterwarnings("ignore")


# =============================================================================
# Configuration
# =============================================================================
SCRIPT_PATH = Path(__file__).resolve()
BASE_DIR = SCRIPT_PATH.parent.parent if SCRIPT_PATH.parent.name.lower() == "src" else SCRIPT_PATH.parent

DEFAULT_INPUT_CSV = BASE_DIR / "data" / "features" / "feature_engineered_data.csv"

RAIN_LABEL_MM = 1.0
TRAIN_END = pd.Timestamp("2023-01-01")
VALID_END = pd.Timestamp("2024-01-01")

FINAL_TABLE_COLUMNS = [
    "model",
    "mae_mm",
    "rmse_mm",
    "wape_percent",
    "r2",
    "bias_mean_pred_minus_true_mm",
]

# Columns that must never be used as input features.
# Use explicit column names so leakage control is easy to audit.
DROP_FEATURE_COLUMNS = [
    "STATION",
    "time",
    "date",
    "PRCP",
    "PRCP_label",
    "PRCP_log1p",
    "target_1stage_prcp_mm",
    "target_stage1_rain_label",
    "target_stage2_prcp_log1p",
]


# Selected features used after feature selection.
# Edit this list only when the feature-selection result changes.
SELECTED_FEATURES = [
    "TEMP",
    "DEWP",
    "SLP",
    "WDSP",
    "VISIB",
    "u_850",
    "v_850",
    "q_850",
    "t_850",
    "z_500",
    "z_850",
    "dew_point_depression",
    "moisture_flux_850",
    "PRCP_lag_1",
    "PRCP_lag_2",
    "PRCP_past_3d_mean",
    "PRCP_past_3d_sum",
    "day_sin",
    "day_cos",
    "month_sin",
    "month_cos",
]


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate rainfall benchmark using selected features with 1-stage Tweedie and 2-stage expected LightGBM."
    )
    parser.add_argument("--input-csv", type=str, default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--random-state", type=int, default=42)

    # A high estimator cap is intentional because early stopping chooses the
    # effective number of boosting rounds on the validation set.
    parser.add_argument("--n-estimators", type=int, default=2000)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)

    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--tweedie-power", type=float, default=1.3)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def resolve_path(path_like: Union[str, Path]) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (BASE_DIR / path).resolve()


# =============================================================================
# Data preparation
# =============================================================================
def load_and_prepare_data(input_csv: Path) -> pd.DataFrame:
    if not input_csv.exists():
        raise FileNotFoundError(f"Input file not found: {input_csv}")

    df = pd.read_csv(input_csv)

    required_cols = {"time", "PRCP", "PRCP_label"}
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    if df["time"].isna().any():
        raise ValueError("Column 'time' contains invalid datetime values.")

    # Check xem PRCP có chứa gtri lỗi các thứ ko chứ chưa xử lý làm lại phần này nha
    df["PRCP"] = pd.to_numeric(df["PRCP"], errors="coerce")
    if df["PRCP"].isna().any():
        raise ValueError("Column 'PRCP' contains missing or non-numeric values.")
    if not np.isfinite(df["PRCP"].values).all():
        raise ValueError("Column 'PRCP' contains inf/-inf values.")
    if (df["PRCP"] < 0).any():
        n_bad = int((df["PRCP"] < 0).sum())
        raise ValueError(f"Column 'PRCP' contains {n_bad} negative values.")

    df["PRCP_label"] = pd.to_numeric(df["PRCP_label"], errors="coerce")
    if df["PRCP_label"].isna().any():
        raise ValueError("Column 'PRCP_label' contains missing or non-numeric values.")

    label_values = set(df["PRCP_label"].unique())
    if not label_values.issubset({0, 1}):
        raise ValueError("Column 'PRCP_label' must contain only binary values {0, 1}.")

    df["PRCP_label"] = df["PRCP_label"].astype(int)

    expected_label = (df["PRCP"] >= RAIN_LABEL_MM).astype(int)
    if not np.array_equal(df["PRCP_label"].values, expected_label.values):
        raise ValueError(
            f"PRCP_label must follow the fixed rule PRCP >= {RAIN_LABEL_MM} mm. "
            "Rerun _07_feature_engineering.py before running this script."
        )

    # Scenario-specific targets are recreated here to guarantee consistency.
    df["target_1stage_prcp_mm"] = df["PRCP"].astype(float)
    df["target_stage1_rain_label"] = df["PRCP_label"].astype(int)
    df["target_stage2_prcp_log1p"] = np.log1p(df["PRCP"].astype(float))

    return df.sort_values("time").reset_index(drop=True)


def time_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df[df["time"] < TRAIN_END].copy()
    valid_df = df[(df["time"] >= TRAIN_END) & (df["time"] < VALID_END)].copy()
    test_df = df[df["time"] >= VALID_END].copy()

    if train_df.empty or valid_df.empty or test_df.empty:
        raise ValueError("Time split produced an empty train/valid/test subset.")

    return train_df, valid_df, test_df


def select_feature_columns(df: pd.DataFrame) -> List[str]:
    all_numeric_features = [
        col for col in df.columns
        if col not in DROP_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(df[col])
    ]

    selected_existing = [
        col for col in SELECTED_FEATURES
        if col in all_numeric_features
    ]

    missing_features = sorted(set(SELECTED_FEATURES) - set(df.columns))
    if missing_features:
        print(f"[WARNING] Selected features not found and skipped: {missing_features}")

    non_numeric_features = sorted(
        col for col in SELECTED_FEATURES
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col])
    )
    if non_numeric_features:
        print(f"[WARNING] Selected features are non-numeric and skipped: {non_numeric_features}")

    if not selected_existing:
        raise ValueError("No selected numeric feature columns found after leakage columns are removed.")

    leakage_cols = sorted(set(selected_existing) & set(DROP_FEATURE_COLUMNS))
    if leakage_cols:
        raise AssertionError(f"Leakage columns remain in selected features: {leakage_cols}")

    return selected_existing


def make_x(df: pd.DataFrame, feature_cols: Iterable[str]) -> pd.DataFrame:
    # LightGBM handles NaN natively. Inf values are converted to NaN for safety.
    return df.loc[:, list(feature_cols)].replace([np.inf, -np.inf], np.nan)


def clip_nonnegative(values: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(values, dtype=float), 0.0)


# =============================================================================
# Model builders
# =============================================================================
def build_one_stage_tweedie(args: argparse.Namespace) -> LGBMRegressor:
    return LGBMRegressor(
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
    )


def build_rain_classifier(args: argparse.Namespace) -> LGBMClassifier:
    return LGBMClassifier(
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
    )


def build_rain_amount_regressor(args: argparse.Namespace) -> LGBMRegressor:
    return LGBMRegressor(
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
    )


def early_stop_callbacks(args: argparse.Namespace) -> list:
    return [
        early_stopping(stopping_rounds=args.early_stopping_rounds, verbose=False),
        log_evaluation(period=0),
    ]


# =============================================================================
# Scenario training and prediction
# =============================================================================
def train_one_stage_tweedie(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: List[str],
    args: argparse.Namespace,
) -> LGBMRegressor:
    model = build_one_stage_tweedie(args)
    model.fit(
        make_x(train_df, feature_cols),
        train_df["target_1stage_prcp_mm"].values.astype(float),
        eval_set=[
            (
                make_x(valid_df, feature_cols),
                valid_df["target_1stage_prcp_mm"].values.astype(float),
            )
        ],
        eval_metric="l2",
        callbacks=early_stop_callbacks(args),
    )
    return model


def predict_one_stage(
    model: LGBMRegressor,
    df: pd.DataFrame,
    feature_cols: List[str],
) -> np.ndarray:
    return clip_nonnegative(model.predict(make_x(df, feature_cols)))


def train_two_stage_expected(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: List[str],
    args: argparse.Namespace,
) -> Tuple[LGBMClassifier, LGBMRegressor, int, int]:
    # Stage 1: learn the fixed rain/no-rain label.
    stage1 = build_rain_classifier(args)
    stage1.fit(
        make_x(train_df, feature_cols),
        train_df["target_stage1_rain_label"].values.astype(int),
        eval_set=[
            (
                make_x(valid_df, feature_cols),
                valid_df["target_stage1_rain_label"].values.astype(int),
            )
        ],
        eval_metric="binary_logloss",
        callbacks=early_stop_callbacks(args),
    )

    # Stage 2: learn positive rainfall amount on rainy days only.
    rainy_train = train_df[train_df["target_stage1_rain_label"].astype(int) == 1]
    rainy_valid = valid_df[valid_df["target_stage1_rain_label"].astype(int) == 1]

    if rainy_train.empty:
        raise ValueError("No rainy training rows found. Stage 2 cannot be trained.")

    stage2 = build_rain_amount_regressor(args)

    if rainy_valid.empty:
        # Fallback: train without early stopping if validation has no rainy rows.
        stage2.fit(
            make_x(rainy_train, feature_cols),
            rainy_train["target_stage2_prcp_log1p"].values.astype(float),
        )
    else:
        stage2.fit(
            make_x(rainy_train, feature_cols),
            rainy_train["target_stage2_prcp_log1p"].values.astype(float),
            eval_set=[
                (
                    make_x(rainy_valid, feature_cols),
                    rainy_valid["target_stage2_prcp_log1p"].values.astype(float),
                )
            ],
            eval_metric="l2",
            callbacks=early_stop_callbacks(args),
        )

    return stage1, stage2, len(rainy_train), len(rainy_valid)


def predict_two_stage_expected(
    stage1: LGBMClassifier,
    stage2: LGBMRegressor,
    df: pd.DataFrame,
    feature_cols: List[str],
) -> np.ndarray:
    x = make_x(df, feature_cols)
    rain_prob = stage1.predict_proba(x)[:, 1]
    rain_amount = clip_nonnegative(np.expm1(stage2.predict(x)))
    return rain_prob * rain_amount


# =============================================================================
# Evaluation
# =============================================================================
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = float(np.sum(np.abs(y_true)))
    if denominator <= 0:
        return np.nan
    return float(np.sum(np.abs(y_true - y_pred)) / denominator * 100.0)


def metrics_row(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return {
        "model": model_name,
        "mae_mm": float(mean_absolute_error(y_true, y_pred)), # sai số trung bình theo mm
        "rmse_mm": rmse(y_true, y_pred), # phạt mạnh lỗi lớn
        "wape_percent": wape(y_true, y_pred), # sai số tương đối theo tổng lượng mưa
        "r2": float(r2_score(y_true, y_pred)), # mức giải thích phương sai
        "bias_mean_pred_minus_true_mm": float(np.mean(y_pred - y_true)), # xu hướng dự báo thừa/thiếu
    }


def comparison_table(
    df: pd.DataFrame,
    pred_1stage: np.ndarray,
    pred_2stage: np.ndarray,
) -> pd.DataFrame:
    y_true = df["PRCP"].values.astype(float)

    return pd.DataFrame(
        [
            metrics_row(y_true, pred_1stage, "LightGBM_1stage_Tweedie"),
            metrics_row(y_true, pred_2stage, "LightGBM_2stage_expected"),
        ],
        columns=FINAL_TABLE_COLUMNS,
    )


def round_numeric(df: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].round(digits)
    return out


def print_table(title: str, table: pd.DataFrame) -> None:
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)
    print(round_numeric(table, 4))


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    args = parse_args()
    input_csv = resolve_path(args.input_csv)

    print(f"Input: {input_csv}")
    print(f"Rain label rule: PRCP >= {RAIN_LABEL_MM} mm")
    print(f"Validation role: early stopping / model selection only; no threshold tuning.")

    df = load_and_prepare_data(input_csv)
    train_df, valid_df, test_df = time_split(df)
    feature_cols = select_feature_columns(df)

    print("\nTime split")
    print(f"Train: {train_df['time'].min().date()} -> {train_df['time'].max().date()} | n={len(train_df):,}")
    print(f"Valid: {valid_df['time'].min().date()} -> {valid_df['time'].max().date()} | n={len(valid_df):,}")
    print(f"Test : {test_df['time'].min().date()} -> {test_df['time'].max().date()} | n={len(test_df):,}")
    print(f"Selected features: {len(feature_cols)} numeric columns")

    print("\n[1/2] Training LightGBM_1stage_Tweedie with validation early stopping...")
    one_stage = train_one_stage_tweedie(train_df, valid_df, feature_cols, args)

    print("[2/2] Training LightGBM_2stage_expected with validation early stopping...")
    stage1, stage2, n_rainy_train, n_rainy_valid = train_two_stage_expected(
        train_df=train_df,
        valid_df=valid_df,
        feature_cols=feature_cols,
        args=args,
    )

    valid_pred_1stage = predict_one_stage(one_stage, valid_df, feature_cols)
    valid_pred_2stage = predict_two_stage_expected(stage1, stage2, valid_df, feature_cols)

    test_pred_1stage = predict_one_stage(one_stage, test_df, feature_cols)
    test_pred_2stage = predict_two_stage_expected(stage1, stage2, test_df, feature_cols)

    valid_table = comparison_table(valid_df, valid_pred_1stage, valid_pred_2stage)
    test_table = comparison_table(test_df, test_pred_1stage, test_pred_2stage)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 160)

    print(f"\nStage-2 rainy rows | train={n_rainy_train:,}, valid={n_rainy_valid:,}")
    print(f"Best iterations | 1-stage={one_stage.best_iteration_}, stage1={stage1.best_iteration_}, stage2={getattr(stage2, 'best_iteration_', None)}")

    print_table("VALIDATION TABLE: used for early stopping / model selection only", valid_table)
    print_table("FINAL TEST TABLE: report this table as the final model validation result", test_table)


if __name__ == "__main__":
    main()
