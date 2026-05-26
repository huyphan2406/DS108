"""
DS108 - Feature selection comparison for final rainfall models
================================================================

Purpose
-------
This script checks whether feature selection improves the final model comparison.
It compares the same two complete rainfall prediction scenarios using:

1) Full numeric features
2) Selected features chosen without using the test set

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

Feature selection policy
------------------------
- Feature ranking is learned only from the training split.
- The number of selected features is chosen only on the validation split.
- The test split is used only once for the final comparison.

Final output
------------
outputs/model_feature_selection_comparison/final_feature_selection_comparison.csv

Run
---
python src/_08_model_feature_selection.py
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
DEFAULT_OUTPUT_DIR = BASE_DIR / "outputs" / "model_feature_selection_comparison"


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

FINAL_METRIC_COLUMNS = [
    "mae_mm",
    "rmse_mm",
    "wape_percent",
    "r2",
    "bias_mean_pred_minus_true_mm",
]

FINAL_TABLE_COLUMNS = [
    "feature_set",
    "n_features",
    "model",
    "prediction_formula",
    "train_scope",
    "eval_scope",
    *FINAL_METRIC_COLUMNS,
]


# =============================================================================
# Basic utilities
# =============================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare full vs selected features for 1-stage Tweedie and full 2-stage hurdle rainfall models."
    )
    parser.add_argument("--input-csv", type=str, default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--tweedie-power", type=float, default=1.3)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument(
        "--candidate-k",
        type=str,
        default="10,15,20,25,30,40,50,60,70",
        help="Comma-separated candidate numbers of selected features. The value is clipped to the available feature count.",
    )
    return parser.parse_args()


def resolve_path(path_like: Union[str, Path]) -> Path:
    path = Path(path_like)
    return path if path.is_absolute() else (BASE_DIR / path).resolve()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clean_old_outputs(output_dir: Path) -> None:
    old_files = [
        "final_feature_selection_comparison.csv",
        "validation_feature_selection_candidates.csv",
        "selected_features.csv",
        "feature_ranking_train_only.csv",
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
    feature_set: str,
    n_features: int,
    model: str,
    prediction_formula: str,
    train_scope: str,
    eval_scope: str,
) -> Dict[str, Union[str, int, float]]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    return {
        "feature_set": feature_set,
        "n_features": int(n_features),
        "model": model,
        "prediction_formula": prediction_formula,
        "train_scope": train_scope,
        "eval_scope": eval_scope,
        "mae_mm": float(mean_absolute_error(y_true, y_pred)),
        "rmse_mm": rmse(y_true, y_pred),
        "wape_percent": wape(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
        "bias_mean_pred_minus_true_mm": float(np.mean(y_pred - y_true)),
    }


# =============================================================================
# Model training/evaluation for a given feature set
# =============================================================================
def train_and_evaluate_scenarios(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_cols: List[str],
    args: argparse.Namespace,
    *,
    feature_set_name: str,
    eval_scope: str,
) -> pd.DataFrame:
    """Train both complete scenarios and evaluate them on one evaluation set."""
    y_eval = eval_df[TARGET_MM].values.astype(float)

    # Scenario 1: full one-stage baseline.
    one_stage = build_one_stage_tweedie(args)
    one_stage.fit(
        make_x(train_df, feature_cols),
        train_df[TARGET_MM].values.astype(float),
    )
    pred_1stage = predict_one_stage_mm(one_stage, eval_df, feature_cols)

    # Scenario 2: full two-stage hurdle model.
    stage1 = build_stage1_classifier(args)
    stage1.fit(
        make_x(train_df, feature_cols),
        train_df[TARGET_CLASS].values.astype(int),
    )

    # Stage 2 must be trained only on rainy training days.
    rainy_train_mask = train_df[TARGET_CLASS].astype(int).values == 1
    if rainy_train_mask.sum() == 0:
        raise ValueError("No rainy days in training data. Stage 2 cannot be trained.")

    stage2 = build_stage2_regressor(args)
    stage2.fit(
        make_x(train_df.loc[rainy_train_mask], feature_cols),
        train_df.loc[rainy_train_mask, TARGET_LOG].values.astype(float),
    )

    pred_stage1_prob = predict_stage1_prob(stage1, eval_df, feature_cols)
    pred_stage2_amount = predict_stage2_amount_mm(stage2, eval_df, feature_cols)

    # Full two-stage prediction:
    # E(PRCP | X) = P(rain | X) * E(PRCP | rain, X)
    pred_2stage_expected = pred_stage1_prob * pred_stage2_amount

    return pd.DataFrame(
        [
            final_metrics_row(
                y_eval,
                pred_1stage,
                feature_set=feature_set_name,
                n_features=len(feature_cols),
                model="LightGBM_1stage_Tweedie",
                prediction_formula="direct_PRCP_prediction",
                train_scope="all_train_days",
                eval_scope=eval_scope,
            ),
            final_metrics_row(
                y_eval,
                pred_2stage_expected,
                feature_set=feature_set_name,
                n_features=len(feature_cols),
                model="LightGBM_2stage_expected",
                prediction_formula="P(rain|X) * E(PRCP|rain,X)",
                train_scope="stage1_all_train_days__stage2_rainy_train_days_only",
                eval_scope=eval_scope,
            ),
        ],
        columns=FINAL_TABLE_COLUMNS,
    )


# =============================================================================
# Feature selection
# =============================================================================
def normalized_importance(model: Pipeline, feature_cols: List[str]) -> np.ndarray:
    booster = model.named_steps["model"]
    importance = getattr(booster, "feature_importances_", None)
    if importance is None:
        return np.zeros(len(feature_cols), dtype=float)
    importance = np.asarray(importance, dtype=float)
    total = importance.sum()
    if total <= 0:
        return np.zeros(len(feature_cols), dtype=float)
    return importance / total


def build_feature_ranking_train_only(
    train_df: pd.DataFrame,
    full_features: List[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    """Rank features using only training data, never validation or test data."""
    # Fit selector models on train only.
    one_stage = build_one_stage_tweedie(args)
    one_stage.fit(
        make_x(train_df, full_features),
        train_df[TARGET_MM].values.astype(float),
    )

    stage1 = build_stage1_classifier(args)
    stage1.fit(
        make_x(train_df, full_features),
        train_df[TARGET_CLASS].values.astype(int),
    )

    rainy_train_mask = train_df[TARGET_CLASS].astype(int).values == 1
    if rainy_train_mask.sum() == 0:
        raise ValueError("No rainy days in training data. Stage 2 cannot be trained.")

    stage2 = build_stage2_regressor(args)
    stage2.fit(
        make_x(train_df.loc[rainy_train_mask], full_features),
        train_df.loc[rainy_train_mask, TARGET_LOG].values.astype(float),
    )

    imp_1stage = normalized_importance(one_stage, full_features)
    imp_stage1 = normalized_importance(stage1, full_features)
    imp_stage2 = normalized_importance(stage2, full_features)

    ranking_df = pd.DataFrame(
        {
            "feature": full_features,
            "importance_1stage_tweedie_norm": imp_1stage,
            "importance_stage1_classifier_norm": imp_stage1,
            "importance_stage2_regression_norm": imp_stage2,
        }
    )

    # Average importance from all components, so one common feature set is used
    # for both final scenarios.
    ranking_df["aggregate_importance"] = ranking_df[
        [
            "importance_1stage_tweedie_norm",
            "importance_stage1_classifier_norm",
            "importance_stage2_regression_norm",
        ]
    ].mean(axis=1)

    return (
        ranking_df.sort_values("aggregate_importance", ascending=False)
        .reset_index(drop=True)
        .assign(rank=lambda x: np.arange(1, len(x) + 1))
    )


def parse_candidate_k(candidate_k: str, n_features: int) -> List[int]:
    values: List[int] = []
    for item in candidate_k.split(","):
        item = item.strip()
        if not item:
            continue
        try:
            k = int(item)
        except ValueError as exc:
            raise ValueError(f"Invalid candidate-k value: {item}") from exc
        if k <= 0:
            continue
        values.append(min(k, n_features))

    values.append(n_features)  # always include full feature count as a validation reference
    values = sorted(set(values))
    return values


def choose_feature_subset_on_validation(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    ranking_df: pd.DataFrame,
    args: argparse.Namespace,
) -> Tuple[List[str], pd.DataFrame, int]:
    """Choose the number of top-ranked features using validation MAE only."""
    n_total = len(ranking_df)
    candidate_ks = parse_candidate_k(args.candidate_k, n_total)

    rows: List[Dict[str, Union[int, float]]] = []

    for k in candidate_ks:
        top_features = ranking_df.head(k)["feature"].tolist()
        valid_result = train_and_evaluate_scenarios(
            train_df,
            valid_df,
            top_features,
            args,
            feature_set_name=f"top_{k}_features",
            eval_scope="validation_2023",
        )

        one_stage_mae = float(
            valid_result.loc[valid_result["model"] == "LightGBM_1stage_Tweedie", "mae_mm"].iloc[0]
        )
        two_stage_mae = float(
            valid_result.loc[valid_result["model"] == "LightGBM_2stage_expected", "mae_mm"].iloc[0]
        )
        one_stage_rmse = float(
            valid_result.loc[valid_result["model"] == "LightGBM_1stage_Tweedie", "rmse_mm"].iloc[0]
        )
        two_stage_rmse = float(
            valid_result.loc[valid_result["model"] == "LightGBM_2stage_expected", "rmse_mm"].iloc[0]
        )

        rows.append(
            {
                "k_features": int(k),
                "validation_mean_mae_mm": (one_stage_mae + two_stage_mae) / 2.0,
                "validation_1stage_mae_mm": one_stage_mae,
                "validation_2stage_mae_mm": two_stage_mae,
                "validation_mean_rmse_mm": (one_stage_rmse + two_stage_rmse) / 2.0,
                "validation_1stage_rmse_mm": one_stage_rmse,
                "validation_2stage_rmse_mm": two_stage_rmse,
            }
        )

    candidate_df = pd.DataFrame(rows).sort_values(
        ["validation_mean_mae_mm", "validation_mean_rmse_mm", "k_features"],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    best_k = int(candidate_df.loc[0, "k_features"])
    selected_features = ranking_df.head(best_k)["feature"].tolist()
    return selected_features, candidate_df, best_k


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
    full_features = select_feature_columns(df)

    print("\nTime split")
    print(f"Train: {train_df[TIME_COL].min().date()} -> {train_df[TIME_COL].max().date()} | n={len(train_df):,}")
    print(f"Valid: {valid_df[TIME_COL].min().date()} -> {valid_df[TIME_COL].max().date()} | n={len(valid_df):,}")
    print(f"Test : {test_df[TIME_COL].min().date()} -> {test_df[TIME_COL].max().date()} | n={len(test_df):,}")
    print(f"\nFull numeric features: {len(full_features)}")

    print("\n[1/4] Ranking features using TRAIN split only...")
    ranking_df = build_feature_ranking_train_only(train_df, full_features, args)
    ranking_path = output_dir / "feature_ranking_train_only.csv"
    ranking_df.to_csv(ranking_path, index=False)

    print("[2/4] Choosing top-k features using VALIDATION split only...")
    selected_features, candidate_df, best_k = choose_feature_subset_on_validation(
        train_df,
        valid_df,
        ranking_df,
        args,
    )
    candidate_path = output_dir / "validation_feature_selection_candidates.csv"
    candidate_df.to_csv(candidate_path, index=False)

    selected_path = output_dir / "selected_features.csv"
    pd.DataFrame({"rank": np.arange(1, len(selected_features) + 1), "feature": selected_features}).to_csv(
        selected_path,
        index=False,
    )

    print(f"Selected features: top {best_k}/{len(full_features)}")

    print("[3/4] Evaluating FULL features on TEST split...")
    full_test_result = train_and_evaluate_scenarios(
        train_df,
        test_df,
        full_features,
        args,
        feature_set_name="full_features",
        eval_scope="test_2024_all_days",
    )

    print("[4/4] Evaluating SELECTED features on TEST split...")
    selected_test_result = train_and_evaluate_scenarios(
        train_df,
        test_df,
        selected_features,
        args,
        feature_set_name=f"selected_top_{best_k}_features",
        eval_scope="test_2024_all_days",
    )

    final_df = pd.concat([full_test_result, selected_test_result], ignore_index=True)

    final_path = output_dir / "final_feature_selection_comparison.csv"
    final_df.to_csv(final_path, index=False)

    config = {
        "main_output": str(final_path),
        "input_csv": str(input_csv),
        "output_dir": str(output_dir),
        "purpose": "Check whether feature selection improves the final full-scenario rainfall prediction comparison.",
        "selection_policy": {
            "feature_ranking": "LightGBM importance aggregation from 1-stage Tweedie, Stage 1 classifier, and Stage 2 regressor trained on TRAIN only.",
            "k_selection": "Choose top-k by lowest mean validation MAE across the two complete scenarios on VALIDATION only.",
            "test_usage": "TEST is used only for the final comparison table.",
        },
        "final_comparison": "full_features vs selected_features for LightGBM_1stage_Tweedie and LightGBM_2stage_expected",
        "metrics": ["MAE", "RMSE", "WAPE", "R2", "Bias"],
        "target": TARGET_MM,
        "rain_label_definition": f"{TARGET_MM} > {RAIN_THRESHOLD_MM} mm",
        "two_stage_formula": "pred_2stage_expected = pred_stage1_prob * pred_stage2_amount",
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
            "stage2_train_rainy_days": int((train_df[TARGET_CLASS].astype(int).values == 1).sum()),
        },
        "n_features": {
            "full": int(len(full_features)),
            "selected": int(len(selected_features)),
        },
        "candidate_k": parse_candidate_k(args.candidate_k, len(full_features)),
        "best_k": int(best_k),
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
    pd.set_option("display.width", 220)

    print("\n" + "=" * 120)
    print("FINAL TABLE: Does selected features improve the full model comparison?")
    print("Metrics: MAE, RMSE, WAPE, R2, Bias")
    print("=" * 120)
    print(round_numeric(final_df, 4))

    print("\nDone.")
    print(f"Final comparison table: {final_path}")
    print(f"Selected features      : {selected_path}")
    print(f"Validation candidates : {candidate_path}")
    print(f"Feature ranking       : {ranking_path}")
    print(f"Config                : {output_dir / 'model_config.json'}")


if __name__ == "__main__":
    main()
