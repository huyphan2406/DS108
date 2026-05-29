"""
DS108 - LightGBM 2x2 model validation
======================================

Run once to compare 4 cases:
1) Full features     + LightGBM Tweedie
2) Full features     + LightGBM Two-stage
3) Selected features + LightGBM Tweedie
4) Selected features + LightGBM Two-stage
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
except ImportError as exc:
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

TARGET_COLUMNS = [
    "PRCP",
    "PRCP_label",
    "target_1stage_prcp_mm",
    "target_stage1_rain_label",
    "target_stage2_prcp_log1p",
]

DROP_FEATURE_COLUMNS = [
    "STATION",
    "time",
    "PRCP",
    "PRCP_label",
    "target_1stage_prcp_mm",
    "target_stage1_rain_label",
    "target_stage2_prcp_log1p",
]

SELECTED_FEATURES = [
    "dew_point_depression",
    "VISIB_lag_1",
    "VISIB",
    "TEMP",
    "moist_convection_850_3d_mean",
    "DEWP_lag_1",
    "VISIB_3d_mean",
    "TEMP_lag_2",
    "v10",
    "u10",
    "q_500",
    "moisture_flux_850",
    "dv_850_500",
    "DEWP",
    "dew_point_depression_lag_2",
    "dew_point_depression_lag_1",
    "TEMP_lag_1",
    "q_850",
    "PRCP_lag_1",
    "u_850",
    "day_sin",
    "day_cos",
    "thickness_500_850",
    "ageostrophic_signal",
    "dv_dt_850",
    "lapse_rate_850_500",
    "moist_convection_850",
    "z_500",
    "t_850",
    "wind_shear_mag",
    "t_500",
    "moisture_flux_850_lag_1",
    "w_850",
    "DEWP_3d_mean",
    "w_500",
    "v_850_lag_1",
    "STP",
    "du_dt_850",
    "du_850_500",
    "SLP_lag_2",
    "VISIB_lag_2",
    "moisture_flux_850_lag_2",
    "LATITUDE",
    "u_500",
    "u_850_lag_1",
]

RESULT_COLUMNS = [
    "feature_set",
    "n_features",
    "model",
    "mae_mm",
    "rmse_mm",
    "wape_percent",
    "r2",
    "bias_mean_pred_minus_true_mm",
]


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare full features vs selected features with LightGBM Tweedie and Two-stage."
    )
    parser.add_argument("--input-csv", type=str, default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--random-state", type=int, default=42)
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

    required_cols = {"time", *TARGET_COLUMNS}
    missing_cols = sorted(required_cols - set(df.columns))
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    if df["time"].isna().any():
        raise ValueError("Column 'time' contains invalid datetime values.")

    for col in TARGET_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        if df[col].isna().any():
            raise ValueError(f"Column '{col}' contains missing or non-numeric values.")
        if not np.isfinite(df[col].to_numpy(dtype=float)).all():
            raise ValueError(f"Column '{col}' contains inf/-inf values.")

    if (df["PRCP"] < 0).any():
        raise ValueError("Column 'PRCP' contains negative values.")

    df["PRCP_label"] = df["PRCP_label"].astype(int)
    df["target_stage1_rain_label"] = df["target_stage1_rain_label"].astype(int)

    label_values = set(df["PRCP_label"].unique())
    if not label_values.issubset({0, 1}):
        raise ValueError("Column 'PRCP_label' must contain only binary values {0, 1}.")

    expected_label = (df["PRCP"] >= RAIN_LABEL_MM).astype(int)
    if not np.array_equal(df["PRCP_label"].values, expected_label.values):
        raise ValueError(
            f"PRCP_label must follow PRCP >= {RAIN_LABEL_MM} mm. "
            "Rerun _07_feature_engineering.py before model validation."
        )

    if not np.allclose(df["target_1stage_prcp_mm"], df["PRCP"], rtol=0, atol=1e-8):
        raise ValueError("target_1stage_prcp_mm must be identical to PRCP.")

    if not np.array_equal(df["target_stage1_rain_label"].values, df["PRCP_label"].values):
        raise ValueError("target_stage1_rain_label must be identical to PRCP_label.")

    expected_log_target = np.log1p(df["PRCP"].astype(float))
    if not np.allclose(df["target_stage2_prcp_log1p"], expected_log_target, rtol=0, atol=1e-6):
        raise ValueError("target_stage2_prcp_log1p must be log1p(PRCP).")

    return df.sort_values("time").reset_index(drop=True)


def time_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = df[df["time"] < TRAIN_END].copy()
    valid_df = df[(df["time"] >= TRAIN_END) & (df["time"] < VALID_END)].copy()
    test_df = df[df["time"] >= VALID_END].copy()

    if train_df.empty or valid_df.empty or test_df.empty:
        raise ValueError("Time split produced an empty train/valid/test subset.")

    return train_df, valid_df, test_df


def get_full_feature_cols(df: pd.DataFrame) -> List[str]:
    feature_cols = [
        col for col in df.columns
        if col not in DROP_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(df[col])
    ]

    if not feature_cols:
        raise ValueError("No numeric feature columns found after leakage columns are removed.")

    return feature_cols


def get_selected_feature_cols(df: pd.DataFrame, full_feature_cols: List[str]) -> List[str]:
    full_feature_set = set(full_feature_cols)
    selected = [col for col in SELECTED_FEATURES if col in full_feature_set]

    missing = sorted(set(SELECTED_FEATURES) - set(df.columns))
    non_numeric = sorted(
        col for col in SELECTED_FEATURES
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col])
    )
    removed_by_leakage = sorted(
        col for col in SELECTED_FEATURES
        if col in df.columns and col not in full_feature_set and col not in non_numeric
    )

    if missing:
        print(f"[WARNING] Selected features not found and skipped: {missing}")
    if non_numeric:
        print(f"[WARNING] Selected features are non-numeric and skipped: {non_numeric}")
    if removed_by_leakage:
        print(f"[WARNING] Selected features removed by leakage rule: {removed_by_leakage}")

    if not selected:
        raise ValueError("No selected feature remains after filtering.")

    return selected


def build_feature_sets(df: pd.DataFrame) -> dict[str, List[str]]:
    full_features = get_full_feature_cols(df)
    selected_features = get_selected_feature_cols(df, full_features)

    return {
        "Full features": full_features,
        "Selected features": selected_features,
    }


def get_X(df: pd.DataFrame, feature_cols: Iterable[str]) -> pd.DataFrame:
    """Prepare model input. LightGBM handles NaN, but not inf/-inf reliably."""
    return df.loc[:, list(feature_cols)].replace([np.inf, -np.inf], np.nan)


def nonnegative(values: np.ndarray) -> np.ndarray:
    """Rainfall cannot be negative, so final predictions are clipped at 0."""
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
# Training and prediction
# =============================================================================
def train_one_stage_tweedie(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: List[str],
    args: argparse.Namespace,
) -> LGBMRegressor:
    model = build_one_stage_tweedie(args)
    model.fit(
        get_X(train_df, feature_cols),
        train_df["target_1stage_prcp_mm"].values.astype(float),
        eval_set=[
            (
                get_X(valid_df, feature_cols),
                valid_df["target_1stage_prcp_mm"].values.astype(float),
            )
        ],
        eval_metric="l2",
        callbacks=early_stop_callbacks(args),
    )
    return model


def predict_one_stage(model: LGBMRegressor, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    return nonnegative(model.predict(get_X(df, feature_cols)))


def train_two_stage_expected(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: List[str],
    args: argparse.Namespace,
) -> Tuple[LGBMClassifier, LGBMRegressor, int, int]:
    stage1 = build_rain_classifier(args)
    stage1.fit(
        get_X(train_df, feature_cols),
        train_df["target_stage1_rain_label"].values.astype(int),
        eval_set=[
            (
                get_X(valid_df, feature_cols),
                valid_df["target_stage1_rain_label"].values.astype(int),
            )
        ],
        eval_metric="binary_logloss",
        callbacks=early_stop_callbacks(args),
    )

    rainy_train = train_df[train_df["target_stage1_rain_label"].astype(int) == 1]
    rainy_valid = valid_df[valid_df["target_stage1_rain_label"].astype(int) == 1]

    if rainy_train.empty:
        raise ValueError("No rainy training rows found. Stage 2 cannot be trained.")

    stage2 = build_rain_amount_regressor(args)

    if rainy_valid.empty:
        stage2.fit(
            get_X(rainy_train, feature_cols),
            rainy_train["target_stage2_prcp_log1p"].values.astype(float),
        )
    else:
        stage2.fit(
            get_X(rainy_train, feature_cols),
            rainy_train["target_stage2_prcp_log1p"].values.astype(float),
            eval_set=[
                (
                    get_X(rainy_valid, feature_cols),
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
    x = get_X(df, feature_cols)
    rain_prob = stage1.predict_proba(x)[:, 1]
    rain_amount = nonnegative(np.expm1(stage2.predict(x)))
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


def metrics_row(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    feature_set_name: str,
    n_features: int,
    model_name: str,
) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return {
        "feature_set": feature_set_name,
        "n_features": n_features,
        "model": model_name,
        "mae_mm": float(mean_absolute_error(y_true, y_pred)),
        "rmse_mm": rmse(y_true, y_pred),
        "wape_percent": wape(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
        "bias_mean_pred_minus_true_mm": float(np.mean(y_pred - y_true)),
    }


def build_result_table(
    df: pd.DataFrame,
    pred_1stage: np.ndarray,
    pred_2stage: np.ndarray,
    feature_set_name: str,
    n_features: int,
) -> pd.DataFrame:
    y_true = df["PRCP"].values.astype(float)

    return pd.DataFrame(
        [
            metrics_row(y_true, pred_1stage, feature_set_name, n_features, "LightGBM_1stage_Tweedie"),
            metrics_row(y_true, pred_2stage, feature_set_name, n_features, "LightGBM_2stage_expected"),
        ],
        columns=RESULT_COLUMNS,
    )


def round_numeric(df: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].round(digits)
    return out


def print_table(title: str, table: pd.DataFrame) -> None:
    print("\n" + "=" * 110)
    print(title)
    print("=" * 110)
    print(round_numeric(table, 4).to_string(index=False))


def run_feature_set(
    feature_set_name: str,
    feature_cols: List[str],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Train both LightGBM scenarios for one feature set and return only test metrics."""
    n_features = len(feature_cols)

    print(f"\nRunning {feature_set_name} | n_features={n_features}")

    one_stage = train_one_stage_tweedie(train_df, valid_df, feature_cols, args)

    stage1, stage2, _, _ = train_two_stage_expected(
        train_df=train_df,
        valid_df=valid_df,
        feature_cols=feature_cols,
        args=args,
    )

    test_pred_1stage = predict_one_stage(one_stage, test_df, feature_cols)
    test_pred_2stage = predict_two_stage_expected(stage1, stage2, test_df, feature_cols)

    return build_result_table(
        df=test_df,
        pred_1stage=test_pred_1stage,
        pred_2stage=test_pred_2stage,
        feature_set_name=feature_set_name,
        n_features=n_features,
    )


# =============================================================================
# Main
# =============================================================================
def main() -> None:
    args = parse_args()
    input_csv = resolve_path(args.input_csv)

    print(f"Input: {input_csv}")
    print(f"Rain label rule: PRCP >= {RAIN_LABEL_MM} mm")
    print("Validation role: early stopping only. Test is used for final report only.")

    df = load_and_prepare_data(input_csv)
    train_df, valid_df, test_df = time_split(df)
    feature_sets = build_feature_sets(df)

    print("\nTime split")
    print(f"Train: {train_df['time'].min().date()} -> {train_df['time'].max().date()} | n={len(train_df):,}")
    print(f"Valid: {valid_df['time'].min().date()} -> {valid_df['time'].max().date()} | n={len(valid_df):,}")
    print(f"Test : {test_df['time'].min().date()} -> {test_df['time'].max().date()} | n={len(test_df):,}")

    test_tables = [
        run_feature_set(
            feature_set_name=feature_set_name,
            feature_cols=feature_cols,
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            args=args,
        )
        for feature_set_name, feature_cols in feature_sets.items()
    ]

    test_result = pd.concat(test_tables, ignore_index=True)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 220)

    print_table("FINAL TEST 2x2 TABLE", test_result)


if __name__ == "__main__":
    main()

