"""
DS108 - Rainfall model validation with 5 regression metrics only
=================================================================

Report tables:
1) Validation results
2) Final test results
3) Full features vs Selected features comparison

Models / scenarios:
0) Persistence baseline: PRCP_hat(t) = PRCP_lag_1
1) Scenario 1 - One-stage regression: LightGBM Tweedie predicts PRCP(t)
2) Scenario 2 - Two-stage expected rainfall: P(rain) * E[PRCP | rain]

Target definition:
- target_1stage_prcp_mm = PRCP(t)
- target_stage1_rain_label = 1 if PRCP(t) >= 0.1 mm else 0
- target_stage2_prcp_log1p = log1p(PRCP(t))

Main evaluation metrics:
- MAE
- RMSE
- WAPE
- R2
- Bias
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
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs" / "model_evaluation"

RAIN_LABEL_MM = 0.1
PERSISTENCE_COL = "PRCP_lag_1"
TRAIN_END = pd.Timestamp("2023-01-01")
VALID_END = pd.Timestamp("2024-01-01")

TARGET_COLUMNS = [
    "PRCP",
    "target_1stage_prcp_mm",
    "target_stage1_rain_label",
    "target_stage2_prcp_log1p",
]

DROP_FEATURE_COLUMNS = [
    "STATION",
    "time",
    "PRCP",
    "target_1stage_prcp_mm",
    "target_stage1_rain_label",
    "target_stage2_prcp_log1p",
    "PRCP_impute",
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
    "LONGITUDE",
    "u_500",
    "u_850_lag_1",
]

RESULT_COLUMNS = [
    "feature_set",
    "scenario",
    "model",
    "n_features",
    "mae_mm",
    "rmse_mm",
    "wape_percent",
    "r2",
    "bias_mean_pred_minus_true_mm",
]

COMPARE_COLUMNS = [
    "scenario",
    "model",
    "full_n_features",
    "selected_n_features",
    "full_mae_mm",
    "selected_mae_mm",
    "full_rmse_mm",
    "selected_rmse_mm",
    "full_wape_percent",
    "selected_wape_percent",
    "full_r2",
    "selected_r2",
    "full_bias_mm",
    "selected_bias_mm",
]

FEATURE_SET_LABELS = {
    "Persistence baseline": "Baseline",
    "Full features": "Full features",
    "Selected features": "Selected features",
}

SCENARIO_LABELS = {
    "Baseline regression": "Baseline: Persistence",
    "Scenario 1 - Regression": "Scenario 1: Regression",
    "Scenario 2 - Two-stage": "Scenario 2: Two-stage",
}

MODEL_LABELS = {
    "Persistence_PRCP_lag_1": "PRCP(t-1)",
    "LightGBM_Tweedie_regression": "LightGBM Tweedie",
    "LightGBM_probability_x_amount": "LightGBM P(rain) x Amount",
}

REPORT_COLUMNS = {
    "feature_set": "Feature set",
    "scenario": "Scenario",
    "model": "Model",
    "n_features": "N features",
    "mae_mm": "MAE (mm)",
    "rmse_mm": "RMSE (mm)",
    "wape_percent": "WAPE (%)",
    "r2": "R2",
    "bias_mean_pred_minus_true_mm": "Bias (mm)",
}

COMPARE_REPORT_COLUMNS = {
    "scenario": "Scenario",
    "model": "Model",
    "full_n_features": "Full N",
    "selected_n_features": "Selected N",
    "full_mae_mm": "Full MAE",
    "selected_mae_mm": "Selected MAE",
    "full_rmse_mm": "Full RMSE",
    "selected_rmse_mm": "Selected RMSE",
    "full_wape_percent": "Full WAPE",
    "selected_wape_percent": "Selected WAPE",
    "full_r2": "Full R2",
    "selected_r2": "Selected R2",
    "full_bias_mm": "Full Bias",
    "selected_bias_mm": "Selected Bias",
}


# =============================================================================
# CLI
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Persistence baseline, one-stage Tweedie, and two-stage LightGBM using 5 rainfall metrics."
    )
    parser.add_argument("--input-csv", type=str, default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
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

    n_before = len(df)
    df = df.dropna(subset=TARGET_COLUMNS).copy()
    n_dropped = n_before - len(df)
    if n_dropped > 0:
        print(f"Dropped rows with missing targets: {n_dropped:,}")

    for col in TARGET_COLUMNS:
        if not np.isfinite(df[col].to_numpy(dtype=float)).all():
            raise ValueError(f"Column '{col}' contains inf/-inf values.")

    if (df["PRCP"] < 0).any():
        raise ValueError("Column 'PRCP' contains negative values.")

    df["target_stage1_rain_label"] = df["target_stage1_rain_label"].astype(int)

    expected_label = (df["PRCP"] >= RAIN_LABEL_MM).astype(int)
    if not np.array_equal(df["target_stage1_rain_label"].values, expected_label.values):
        raise ValueError(f"target_stage1_rain_label must follow PRCP >= {RAIN_LABEL_MM} mm.")

    if not np.allclose(df["target_1stage_prcp_mm"], df["PRCP"], rtol=0, atol=1e-8):
        raise ValueError("target_1stage_prcp_mm must be identical to PRCP for day-t prediction.")

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
        col
        for col in df.columns
        if col not in DROP_FEATURE_COLUMNS and pd.api.types.is_numeric_dtype(df[col])
    ]
    if not feature_cols:
        raise ValueError("No numeric feature columns found after leakage columns are removed.")
    return feature_cols


def get_selected_feature_cols(df: pd.DataFrame, full_feature_cols: List[str]) -> List[str]:
    full_feature_set = set(full_feature_cols)
    selected = [col for col in SELECTED_FEATURES if col in full_feature_set]

    missing = sorted(set(SELECTED_FEATURES) - set(df.columns))
    if missing:
        print(f"[WARNING] Selected features not found and skipped: {missing}")

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
    return df.loc[:, list(feature_cols)].replace([np.inf, -np.inf], np.nan)


def nonnegative(values: np.ndarray) -> np.ndarray:
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
        train_df["PRCP"].values.astype(float),
        eval_set=[(get_X(valid_df, feature_cols), valid_df["PRCP"].values.astype(float))],
        eval_metric="tweedie",
        callbacks=early_stop_callbacks(args),
    )
    return model


def train_two_stage_expected(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    feature_cols: List[str],
    args: argparse.Namespace,
) -> Tuple[LGBMClassifier, LGBMRegressor]:
    stage1 = build_rain_classifier(args)
    stage1.fit(
        get_X(train_df, feature_cols),
        train_df["target_stage1_rain_label"].values.astype(int),
        eval_set=[(get_X(valid_df, feature_cols), valid_df["target_stage1_rain_label"].values.astype(int))],
        eval_metric="binary_logloss",
        callbacks=early_stop_callbacks(args),
    )

    rainy_train = train_df[train_df["target_stage1_rain_label"].astype(int) == 1]
    rainy_valid = valid_df[valid_df["target_stage1_rain_label"].astype(int) == 1]
    if rainy_train.empty:
        raise ValueError("No rainy training rows found. Stage 2 cannot be trained.")

    stage2 = build_rain_amount_regressor(args)
    if rainy_valid.empty:
        stage2.fit(get_X(rainy_train, feature_cols), rainy_train["target_stage2_prcp_log1p"].values.astype(float))
    else:
        stage2.fit(
            get_X(rainy_train, feature_cols),
            rainy_train["target_stage2_prcp_log1p"].values.astype(float),
            eval_set=[(get_X(rainy_valid, feature_cols), rainy_valid["target_stage2_prcp_log1p"].values.astype(float))],
            eval_metric="l2",
            callbacks=early_stop_callbacks(args),
        )

    return stage1, stage2


def predict_one_stage(model: LGBMRegressor, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
    return nonnegative(model.predict(get_X(df, feature_cols)))


def predict_two_stage_expected(
    stage1: LGBMClassifier,
    stage2: LGBMRegressor,
    df: pd.DataFrame,
    feature_cols: List[str],
) -> np.ndarray:
    x = get_X(df, feature_cols)
    rain_prob = stage1.predict_proba(x)[:, 1]
    conditional_amount = nonnegative(np.expm1(stage2.predict(x)))
    return rain_prob * conditional_amount


def predict_persistence(df: pd.DataFrame) -> np.ndarray:
    if PERSISTENCE_COL not in df.columns:
        raise ValueError(f"Missing persistence column: {PERSISTENCE_COL}")
    return nonnegative(df[PERSISTENCE_COL].values.astype(float))


# =============================================================================
# Evaluation and report tables
# =============================================================================
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = float(np.sum(np.abs(y_true)))
    if denominator <= 0:
        return np.nan
    return float(np.sum(np.abs(y_true - y_pred)) / denominator * 100.0)


def metrics_row(
    df: pd.DataFrame,
    y_pred: np.ndarray,
    feature_set: str,
    scenario: str,
    model: str,
    n_features: int,
) -> dict:
    eval_df = df.copy()
    eval_df["_pred"] = y_pred
    eval_df = eval_df.dropna(subset=["PRCP", "_pred"])

    y_true = eval_df["PRCP"].values.astype(float)
    y_pred_clean = eval_df["_pred"].values.astype(float)

    return {
        "feature_set": feature_set,
        "scenario": scenario,
        "model": model,
        "n_features": n_features,
        "mae_mm": float(mean_absolute_error(y_true, y_pred_clean)),
        "rmse_mm": rmse(y_true, y_pred_clean),
        "wape_percent": wape(y_true, y_pred_clean),
        "r2": float(r2_score(y_true, y_pred_clean)),
        "bias_mean_pred_minus_true_mm": float(np.mean(y_pred_clean - y_true)),
    }


def evaluate_persistence(df: pd.DataFrame) -> pd.DataFrame:
    eval_df = df.dropna(subset=["PRCP", PERSISTENCE_COL]).copy()
    pred = predict_persistence(eval_df)
    row = metrics_row(
        df=eval_df,
        y_pred=pred,
        feature_set="Persistence baseline",
        scenario="Baseline regression",
        model="Persistence_PRCP_lag_1",
        n_features=1,
    )
    return pd.DataFrame([row], columns=RESULT_COLUMNS)


def run_feature_set(
    feature_set_name: str,
    feature_cols: List[str],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    n_features = len(feature_cols)
    print(f"\nRunning {feature_set_name} | n_features={n_features}")

    # Scenario 1: one-stage regression model
    one_stage = train_one_stage_tweedie(train_df, valid_df, feature_cols, args)
    valid_pred_s1 = predict_one_stage(one_stage, valid_df, feature_cols)
    test_pred_s1 = predict_one_stage(one_stage, test_df, feature_cols)

    # Scenario 2: two-stage expected rainfall = P(rain) * E(PRCP | rain)
    stage1, stage2 = train_two_stage_expected(train_df, valid_df, feature_cols, args)
    valid_pred_s2 = predict_two_stage_expected(stage1, stage2, valid_df, feature_cols)
    test_pred_s2 = predict_two_stage_expected(stage1, stage2, test_df, feature_cols)

    valid_rows = [
        metrics_row(valid_df, valid_pred_s1, feature_set_name, "Scenario 1 - Regression", "LightGBM_Tweedie_regression", n_features),
        metrics_row(valid_df, valid_pred_s2, feature_set_name, "Scenario 2 - Two-stage", "LightGBM_probability_x_amount", n_features),
    ]
    test_rows = [
        metrics_row(test_df, test_pred_s1, feature_set_name, "Scenario 1 - Regression", "LightGBM_Tweedie_regression", n_features),
        metrics_row(test_df, test_pred_s2, feature_set_name, "Scenario 2 - Two-stage", "LightGBM_probability_x_amount", n_features),
    ]

    return pd.DataFrame(valid_rows, columns=RESULT_COLUMNS), pd.DataFrame(test_rows, columns=RESULT_COLUMNS)


def build_full_vs_selected_table(test_result: pd.DataFrame) -> pd.DataFrame:
    rows = []
    model_rows = test_result[test_result["feature_set"].isin(["Full features", "Selected features"])]

    for (scenario, model), group in model_rows.groupby(["scenario", "model"]):
        full = group[group["feature_set"] == "Full features"]
        selected = group[group["feature_set"] == "Selected features"]
        if full.empty or selected.empty:
            continue

        full_row = full.iloc[0]
        selected_row = selected.iloc[0]
        rows.append(
            {
                "scenario": scenario,
                "model": model,
                "full_n_features": int(full_row["n_features"]),
                "selected_n_features": int(selected_row["n_features"]),
                "full_mae_mm": full_row["mae_mm"],
                "selected_mae_mm": selected_row["mae_mm"],
                "full_rmse_mm": full_row["rmse_mm"],
                "selected_rmse_mm": selected_row["rmse_mm"],
                "full_wape_percent": full_row["wape_percent"],
                "selected_wape_percent": selected_row["wape_percent"],
                "full_r2": full_row["r2"],
                "selected_r2": selected_row["r2"],
                "full_bias_mm": full_row["bias_mean_pred_minus_true_mm"],
                "selected_bias_mm": selected_row["bias_mean_pred_minus_true_mm"],
            }
        )

    return pd.DataFrame(rows, columns=COMPARE_COLUMNS)


def round_numeric(df: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    out = df.copy()
    numeric_cols = out.select_dtypes(include=[np.number]).columns
    out[numeric_cols] = out[numeric_cols].round(digits)
    return out


def format_result_table(table: pd.DataFrame) -> pd.DataFrame:
    """Make Validation/Test tables short and report-friendly."""
    out = round_numeric(table, 4)
    out["feature_set"] = out["feature_set"].replace(FEATURE_SET_LABELS)
    out["scenario"] = out["scenario"].replace(SCENARIO_LABELS)
    out["model"] = out["model"].replace(MODEL_LABELS)
    return out.rename(columns=REPORT_COLUMNS)


def format_comparison_table(table: pd.DataFrame) -> pd.DataFrame:
    """Make Full vs Selected comparison table short and report-friendly."""
    out = round_numeric(table, 4)
    out["scenario"] = out["scenario"].replace(SCENARIO_LABELS)
    out["model"] = out["model"].replace(MODEL_LABELS)
    return out.rename(columns=COMPARE_REPORT_COLUMNS)


def print_table(title: str, table: pd.DataFrame) -> None:
    print("\n" + "=" * 120)
    print(title)
    print("=" * 120)
    print(table.to_string(index=False))

def save_report_tables(
    output_dir: Path,
    validation_report: pd.DataFrame,
    test_report: pd.DataFrame,
    comparison_report: pd.DataFrame,
    validation_result: pd.DataFrame,
    test_result: pd.DataFrame,
    comparison_result: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    validation_report.to_csv(output_dir / "table_1_validation_results_2023.csv", index=False, encoding="utf-8-sig")
    test_report.to_csv(output_dir / "table_2_final_test_results_2024.csv", index=False, encoding="utf-8-sig")
    comparison_report.to_csv(output_dir / "table_3_full_vs_selected_test.csv", index=False, encoding="utf-8-sig")

    validation_result.to_csv(output_dir / "raw_validation_results_2023.csv", index=False, encoding="utf-8-sig")
    test_result.to_csv(output_dir / "raw_test_results_2024.csv", index=False, encoding="utf-8-sig")
    comparison_result.to_csv(output_dir / "raw_full_vs_selected_test.csv", index=False, encoding="utf-8-sig")

    with open(output_dir / "model_evaluation_summary.txt", "w", encoding="utf-8") as f:
        f.write("DS108 rainfall model evaluation\\n")
        f.write("=" * 80 + "\\n\\n")
        f.write(f"Rain label rule: PRCP >= {RAIN_LABEL_MM} mm\\n")
        f.write("Main evaluation metrics: MAE, RMSE, WAPE, R2, Bias\\n\\n")

        f.write("TABLE 1. Validation results on 2023 set\\n")
        f.write(validation_report.to_string(index=False))
        f.write("\\n\\n")

        f.write("TABLE 2. Final test results on 2024 set\\n")
        f.write(test_report.to_string(index=False))
        f.write("\\n\\n")

        f.write("TABLE 3. Full features vs Selected features on test set\\n")
        f.write(comparison_report.to_string(index=False))
        f.write("\\n")

    print(f"\\nSaved model evaluation outputs to: {output_dir}")

# =============================================================================
# Main
# =============================================================================
def main() -> None:
    args = parse_args()
    input_csv = resolve_path(args.input_csv)
    output_dir = resolve_path(args.output_dir)

    print(f"Input: {input_csv}")
    print(f"Rain label rule: PRCP >= {RAIN_LABEL_MM} mm")
    print("Main evaluation metrics: MAE, RMSE, WAPE, R2, Bias")

    df = load_and_prepare_data(input_csv)
    train_df, valid_df, test_df = time_split(df)
    feature_sets = build_feature_sets(df)

    print("\nTime split")
    print(f"Train: {train_df['time'].min().date()} -> {train_df['time'].max().date()} | n={len(train_df):,}")
    print(f"Valid: {valid_df['time'].min().date()} -> {valid_df['time'].max().date()} | n={len(valid_df):,}")
    print(f"Test : {test_df['time'].min().date()} -> {test_df['time'].max().date()} | n={len(test_df):,}")

    validation_tables = [evaluate_persistence(valid_df)]
    test_tables = [evaluate_persistence(test_df)]

    for feature_set_name, feature_cols in feature_sets.items():
        valid_table, test_table = run_feature_set(
            feature_set_name=feature_set_name,
            feature_cols=feature_cols,
            train_df=train_df,
            valid_df=valid_df,
            test_df=test_df,
            args=args,
        )
        validation_tables.append(valid_table)
        test_tables.append(test_table)

    validation_result = pd.concat(validation_tables, ignore_index=True)
    test_result = pd.concat(test_tables, ignore_index=True)
    comparison_result = build_full_vs_selected_table(test_result)

    validation_report = format_result_table(validation_result)
    test_report = format_result_table(test_result)
    comparison_report = format_comparison_table(comparison_result)

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 300)

    print_table("TABLE 1. Validation results on 2023 set", validation_report)
    print_table("TABLE 2. Final test results on 2024 set", test_report)
    print_table("TABLE 3. Full features vs Selected features on test set", comparison_report)

    save_report_tables(
        output_dir=output_dir,
        validation_report=validation_report,
        test_report=test_report,
        comparison_report=comparison_report,
        validation_result=validation_result,
        test_result=test_result,
        comparison_result=comparison_result,
    )

if __name__ == "__main__":
    main()
