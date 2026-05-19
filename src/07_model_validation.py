"""
Step 7 - Probabilistic Model-based Dataset Validation for DS108 Rainfall Project.

Important framing
-----------------
This step is NOT the main contribution of the DS108 preprocessing project.
The model is used as an empirical validation tool to check whether the
constructed dataset and engineered features contain useful meteorological signal.

Problem definition
------------------
Probabilistic Binary Rainfall Classification:

    Input  : meteorological features of day t
    Output : P(rain = 1 | features of day t)
    Label  : rain_target = 1 if PRCP_mm > 0.1 mm else 0
    Decision:
        y_pred = 1 if y_prob >= threshold
        y_pred = 0 otherwise

The model first estimates probability, then a validation-optimized threshold
converts the probability into Rain / No-rain.

Input
-----
    data/feature_engineering/feature_engineered_data.csv

Expected required columns:
    time, STATION, rain_target, feature columns...

Output
------
    reports/model_validation/
    models/rainfall_validation/

Main reports:
    - model_metrics_all.csv
    - model_metrics_test_ranked.csv
    - split_summary.csv
    - used_feature_columns.csv
    - predictions/*_predictions.csv
    - threshold_curves/*_threshold_curve.csv
    - calibration/*_calibration_bins.csv
    - plots/*.png
    - model_validation_metadata.json

Metric design
-------------
Because the project predicts rain probability and then classifies rain/no-rain:

1. Probability / ranking quality:
   - PR-AUC  : especially informative for the positive rain class.
   - ROC-AUC : overall discrimination ability.
   - Brier Score : probability calibration/accuracy.
   - ECE : expected calibration error.

2. Threshold-based classification quality:
   - Precision Rain
   - Recall Rain
   - F1 Rain
   - F2 Rain
   - Balanced Accuracy
   - Specificity No-rain

Accuracy is reported but should not be the main metric because the no-rain class
can dominate and majority baselines may look deceptively good.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

try:
    import lightgbm as lgb
except Exception:  # pragma: no cover
    lgb = None


# =============================================================================
# CONFIGURATION
# =============================================================================

RANDOM_SEED = 108
TARGET_COL = "rain_target"
POSITIVE_LABEL = 1

DEFAULT_FEATURE_FILE = "feature_engineered_data.csv"

# Chronological split:
# Train      : 2015-2022
# Validation : 2023
# Test       : 2024
TRAIN_END = "2023-01-01"
VALID_START = "2023-01-01"
VALID_END = "2024-01-01"
TEST_START = "2024-01-01"
TEST_END = "2025-01-01"

# Main threshold selection metric.
# Recommended default:
#   f1_rain = balanced trade-off between missing rain and false rain alarms.
# Alternative:
#   f2_rain = more recall-oriented if missing rain is considered more costly.
THRESHOLD_SELECTION_METRIC = "f1_rain"
VALID_THRESHOLD_METRICS = {
    "f1_rain",
    "f2_rain",
    "balanced_accuracy",
    "youden_j",
}

THRESHOLD_GRID = np.round(np.linspace(0.05, 0.95, 181), 4)

# Whether to keep static station metadata as predictors.
# True is acceptable for same-station temporal validation.
# False is stricter if you want to reduce station memorization concerns.
KEEP_LOCATION_FEATURES = True

LOCATION_COLS = {"LATITUDE", "LONGITUDE", "ELEVATION"}

# Conservative leakage/audit columns to drop if they appear.
ALWAYS_DROP_COLS = {
    "time",
    "target_time",
    "DATE",
    "STATION",
    TARGET_COL,
    "PRCP",
    "PRCP_mm",
    "target_prcp_mm",
    "has_gsod_record",
    "YEAR",
    "MONTH",
    "month",
}

DROP_SUFFIXES = ("_source",)

# Logistic Regression baseline
LOGISTIC_PARAMS = {
    "max_iter": 3000,
    "class_weight": "balanced",
    "solver": "lbfgs",
    "random_state": RANDOM_SEED,
}

# Random Forest baseline
RF_PARAMS = {
    "n_estimators": 400,
    "max_depth": None,
    "min_samples_leaf": 2,
    "class_weight": "balanced_subsample",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

# LightGBM validation model
LIGHTGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.02,
    "num_leaves": 48,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "seed": RANDOM_SEED,
    "verbosity": -1,
}

NUM_BOOST_ROUNDS = 5000
EARLY_STOPPING_ROUNDS = 150
LOG_EVALUATION_PERIOD = 100

CALIBRATION_BINS = 10


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
FEATURE_DIR = DATA_DIR / "feature_engineering"

MODEL_ROOT_DIR = BASE_DIR / "models"
REPORT_ROOT_DIR = BASE_DIR / "reports"

DEFAULT_INPUT_CSV = FEATURE_DIR / DEFAULT_FEATURE_FILE

MODEL_DIR = MODEL_ROOT_DIR / "rainfall_validation"
REPORT_DIR = REPORT_ROOT_DIR / "model_validation"
PLOT_DIR = REPORT_DIR / "plots"
PRED_DIR = REPORT_DIR / "predictions"
THRESHOLD_DIR = REPORT_DIR / "threshold_curves"
CALIBRATION_DIR = REPORT_DIR / "calibration"
IMPORTANCE_DIR = REPORT_DIR / "feature_importance"


def ensure_directories() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)
    THRESHOLD_DIR.mkdir(parents=True, exist_ok=True)
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    IMPORTANCE_DIR.mkdir(parents=True, exist_ok=True)


def _write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


# =============================================================================
# DATA LOADING AND SPLITTING
# =============================================================================

def load_feature_engineered_data(input_csv: Path | str = DEFAULT_INPUT_CSV) -> pd.DataFrame:
    """
    Load model-ready feature-engineered data.

    This function intentionally requires rain_target and does not recreate the
    target from PRCP. Target creation belongs to Step 6.
    """
    input_csv = Path(input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(
            f"Input feature file not found: {input_csv}. "
            "Run Step 6 feature engineering first."
        )

    df = pd.read_csv(input_csv)

    if "time" not in df.columns:
        raise ValueError("Input data must contain a `time` column for chronological splitting.")

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Input data must contain `{TARGET_COL}` created by Step 6. "
            "Do not recreate target from PRCP in Step 7."
        )

    if "STATION" in df.columns:
        df["STATION"] = df["STATION"].astype(str)

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time", TARGET_COL]).copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    invalid_target = set(df[TARGET_COL].unique()) - {0, 1}
    if invalid_target:
        raise ValueError(f"`{TARGET_COL}` must be binary 0/1. Found: {invalid_target}")

    df = df.sort_values("time").reset_index(drop=True)
    return df


def split_train_valid_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data chronologically to avoid future-to-past leakage."""
    train_df = df[df["time"] < TRAIN_END].copy()
    valid_df = df[(df["time"] >= VALID_START) & (df["time"] < VALID_END)].copy()
    test_df = df[(df["time"] >= TEST_START) & (df["time"] < TEST_END)].copy()

    if train_df.empty or valid_df.empty or test_df.empty:
        raise ValueError(
            "Train/valid/test split produced an empty set. "
            "Check date range and time column."
        )

    return train_df, valid_df, test_df


def save_split_summary(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    """Save split sizes and target rates."""
    rows = []
    for name, split in [("train", train_df), ("validation", valid_df), ("test", test_df)]:
        rows.append({
            "split": name,
            "start_date": split["time"].min().strftime("%Y-%m-%d"),
            "end_date": split["time"].max().strftime("%Y-%m-%d"),
            "n_rows": int(len(split)),
            "n_stations": int(split["STATION"].nunique()) if "STATION" in split.columns else None,
            "rain_count": int(split[TARGET_COL].sum()),
            "no_rain_count": int((split[TARGET_COL] == 0).sum()),
            "rain_rate": float(split[TARGET_COL].mean()),
        })

    pd.DataFrame(rows).to_csv(REPORT_DIR / "split_summary.csv", index=False)


# =============================================================================
# FEATURE SELECTION FOR MODEL VALIDATION
# =============================================================================

def build_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Choose model feature columns conservatively:
    - remove target/leakage/time/string/source columns
    - keep numeric predictors only
    """
    drop_cols = set(ALWAYS_DROP_COLS)

    if not KEEP_LOCATION_FEATURES:
        drop_cols.update(LOCATION_COLS)

    candidate_cols: List[str] = []

    for col in df.columns:
        if col in drop_cols:
            continue

        if any(col.endswith(suffix) for suffix in DROP_SUFFIXES):
            continue

        if df[col].isna().all():
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            candidate_cols.append(col)

    if not candidate_cols:
        raise ValueError("No numeric feature columns available after leakage filtering.")

    return candidate_cols


def make_xy(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Create aligned X/y matrices."""
    X_train = train_df[feature_cols].copy()
    y_train = train_df[TARGET_COL].copy()

    X_valid = valid_df[feature_cols].copy()
    y_valid = valid_df[TARGET_COL].copy()

    X_test = test_df[feature_cols].copy()
    y_test = test_df[TARGET_COL].copy()

    return X_train, y_train, X_valid, y_valid, X_test, y_test


def save_feature_columns(feature_cols: List[str]) -> None:
    pd.DataFrame({"feature": feature_cols}).to_csv(
        REPORT_DIR / "used_feature_columns.csv",
        index=False,
    )


# =============================================================================
# METRICS
# =============================================================================

def _safe_auc(y_true: pd.Series | np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(roc_auc_score(y_true, y_prob))


def _safe_pr_auc(y_true: pd.Series | np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(average_precision_score(y_true, y_prob))


def _safe_log_loss(y_true: pd.Series | np.ndarray, y_prob: np.ndarray) -> float:
    eps = 1e-7
    clipped = np.clip(y_prob, eps, 1.0 - eps)
    return float(log_loss(y_true, clipped, labels=[0, 1]))


def expected_calibration_error(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = CALIBRATION_BINS,
) -> Tuple[float, pd.DataFrame]:
    """
    Compute equal-width-bin Expected Calibration Error (ECE).

    ECE = weighted mean absolute difference between predicted probability and
    observed rain frequency in each probability bin.
    """
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob_arr, bins[1:-1], right=False)

    rows = []
    ece = 0.0
    n = len(y_true_arr)

    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        count = int(mask.sum())

        left = float(bins[bin_id])
        right = float(bins[bin_id + 1])

        if count == 0:
            rows.append({
                "bin": bin_id + 1,
                "prob_left": left,
                "prob_right": right,
                "count": 0,
                "mean_predicted_probability": np.nan,
                "observed_rain_rate": np.nan,
                "absolute_gap": np.nan,
            })
            continue

        mean_prob = float(y_prob_arr[mask].mean())
        observed_rate = float(y_true_arr[mask].mean())
        gap = abs(mean_prob - observed_rate)
        ece += (count / n) * gap

        rows.append({
            "bin": bin_id + 1,
            "prob_left": left,
            "prob_right": right,
            "count": count,
            "mean_predicted_probability": mean_prob,
            "observed_rain_rate": observed_rate,
            "absolute_gap": gap,
        })

    return float(ece), pd.DataFrame(rows)


def classification_metrics_at_threshold(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Calculate threshold-based metrics."""
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    specificity_no_rain = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    false_alarm_rate = fp / (fp + tn) if (fp + tn) > 0 else np.nan
    miss_rate = fn / (fn + tp) if (fn + tp) > 0 else np.nan

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_rain": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_rain": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_rain": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2_rain": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "specificity_no_rain": float(specificity_no_rain),
        "false_alarm_rate": float(false_alarm_rate),
        "miss_rate": float(miss_rate),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def probability_metrics(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """Calculate probability/ranking metrics."""
    ece, _ = expected_calibration_error(y_true, y_prob, n_bins=CALIBRATION_BINS)
    return {
        "roc_auc": _safe_auc(y_true, y_prob),
        "pr_auc": _safe_pr_auc(y_true, y_prob),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "log_loss": _safe_log_loss(y_true, y_prob),
        "ece": float(ece),
    }


def evaluate_predictions(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    metrics = classification_metrics_at_threshold(y_true, y_prob, threshold)
    metrics.update(probability_metrics(y_true, y_prob))
    return metrics


def optimize_threshold(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    metric_name: str = THRESHOLD_SELECTION_METRIC,
) -> Tuple[float, pd.DataFrame]:
    """
    Select the classification threshold on validation data only.

    This preserves the test set as a final unseen evaluation period.
    """
    if metric_name not in VALID_THRESHOLD_METRICS:
        raise ValueError(f"Unknown threshold metric: {metric_name}")

    rows = []
    for threshold in THRESHOLD_GRID:
        metrics = classification_metrics_at_threshold(y_true, y_prob, float(threshold))
        metrics["youden_j"] = metrics["recall_rain"] + metrics["specificity_no_rain"] - 1
        rows.append(metrics)

    curve = pd.DataFrame(rows)

    # Tie-breaking:
    # 1. best selected metric
    # 2. higher recall
    # 3. lower threshold distance from 0.5
    curve["_threshold_distance_from_05"] = (curve["threshold"] - 0.5).abs()
    curve = curve.sort_values(
        by=[metric_name, "recall_rain", "_threshold_distance_from_05"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    best_threshold = float(curve.loc[0, "threshold"])

    curve = curve.drop(columns=["_threshold_distance_from_05"])
    curve.to_csv(THRESHOLD_DIR / f"{model_name}_threshold_curve.csv", index=False)

    return best_threshold, curve


# =============================================================================
# PLOTTING AND REPORTING
# =============================================================================

def plot_threshold_curve(curve: pd.DataFrame, model_name: str) -> None:
    """Plot validation threshold curve."""
    plt.figure(figsize=(9, 5))
    for col in ["precision_rain", "recall_rain", "f1_rain", "f2_rain"]:
        if col in curve.columns:
            plt.plot(curve["threshold"], curve[col], label=col)

    plt.xlabel("Decision threshold")
    plt.ylabel("Score")
    plt.title(f"Validation threshold curve - {model_name}")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{model_name}_threshold_curve.png", dpi=160)
    plt.close()


def plot_confusion_matrix_from_values(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    model_name: str,
    split_name: str,
) -> None:
    """Plot confusion matrix."""
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(f"{model_name} - {split_name} confusion matrix")
    plt.colorbar()

    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["No rain", "Rain"])
    plt.yticks(tick_marks, ["No rain", "Rain"])

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{model_name}_{split_name}_confusion_matrix.png", dpi=160)
    plt.close()


def save_calibration_report(
    y_true: pd.Series | np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    split_name: str,
) -> None:
    """Save calibration bins and plot reliability diagram."""
    ece, bins = expected_calibration_error(y_true, y_prob, n_bins=CALIBRATION_BINS)
    bins["ece"] = ece
    bins.to_csv(CALIBRATION_DIR / f"{model_name}_{split_name}_calibration_bins.csv", index=False)

    non_empty = bins[bins["count"] > 0].copy()

    plt.figure(figsize=(5.5, 5.5))
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.plot(
        non_empty["mean_predicted_probability"],
        non_empty["observed_rain_rate"],
        marker="o",
        label=f"{model_name} ({split_name})",
    )
    plt.xlabel("Mean predicted rain probability")
    plt.ylabel("Observed rain frequency")
    plt.title(f"Reliability diagram - {model_name} ({split_name})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{model_name}_{split_name}_calibration.png", dpi=160)
    plt.close()


def save_predictions(
    df_split: pd.DataFrame,
    y_true: pd.Series,
    y_prob: np.ndarray,
    threshold: float,
    model_name: str,
    split_name: str,
) -> None:
    """Save probability and classification output."""
    y_pred = (y_prob >= threshold).astype(int)

    out = pd.DataFrame({
        "time": df_split["time"].values,
        "STATION": df_split["STATION"].values if "STATION" in df_split.columns else np.nan,
        "y_true": y_true.values,
        "y_prob": y_prob,
        "rain_probability_percent": y_prob * 100.0,
        "threshold": threshold,
        "y_pred": y_pred,
        "predicted_label": np.where(y_pred == 1, "Rain", "No rain"),
    })

    out.to_csv(PRED_DIR / f"{model_name}_{split_name}_predictions.csv", index=False)


def save_feature_importance(
    model_name: str,
    model: Any,
    feature_cols: List[str],
) -> None:
    """Save and plot model-specific feature importance when available."""
    rows = []

    if model_name == "LightGBM":
        importance_gain = model.feature_importance(importance_type="gain")
        importance_split = model.feature_importance(importance_type="split")
        rows = [
            {
                "feature": feature,
                "importance_gain": float(gain),
                "importance_split": float(split),
            }
            for feature, gain, split in zip(feature_cols, importance_gain, importance_split)
        ]
        importance_col = "importance_gain"

    elif model_name == "RandomForest":
        rf = model.named_steps.get("classifier") if isinstance(model, Pipeline) else model
        if hasattr(rf, "feature_importances_"):
            rows = [
                {
                    "feature": feature,
                    "importance": float(importance),
                }
                for feature, importance in zip(feature_cols, rf.feature_importances_)
            ]
            importance_col = "importance"
        else:
            return

    elif model_name == "LogisticRegression":
        clf = model.named_steps.get("classifier") if isinstance(model, Pipeline) else model
        if hasattr(clf, "coef_"):
            coef = clf.coef_.ravel()
            rows = [
                {
                    "feature": feature,
                    "coefficient": float(value),
                    "abs_coefficient": float(abs(value)),
                }
                for feature, value in zip(feature_cols, coef)
            ]
            importance_col = "abs_coefficient"
        else:
            return

    else:
        return

    importance_df = pd.DataFrame(rows)
    if importance_df.empty:
        return

    importance_df = importance_df.sort_values(importance_col, ascending=False)
    importance_df.to_csv(IMPORTANCE_DIR / f"{model_name}_feature_importance.csv", index=False)

    top = importance_df.head(20).iloc[::-1]
    plt.figure(figsize=(8, 7))
    plt.barh(top["feature"], top[importance_col])
    plt.xlabel(importance_col)
    plt.title(f"Top 20 feature importance - {model_name}")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / f"{model_name}_feature_importance_top20.png", dpi=160)
    plt.close()


# =============================================================================
# MODEL TRAINING
# =============================================================================

def get_positive_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Get P(y=1) robustly for sklearn-like estimators."""
    proba = model.predict_proba(X)

    classes = getattr(model, "classes_", None)
    if classes is None and isinstance(model, Pipeline):
        last_step = model.steps[-1][1]
        classes = getattr(last_step, "classes_", None)

    if classes is None:
        # assume second column is positive class
        return proba[:, 1]

    classes = list(classes)
    if POSITIVE_LABEL in classes:
        pos_idx = classes.index(POSITIVE_LABEL)
        return proba[:, pos_idx]

    # Model was trained with only one class; this should rarely happen.
    return np.zeros(len(X), dtype=float)


def train_majority_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> DummyClassifier:
    """Always predicts the majority class; useful sanity baseline."""
    model = DummyClassifier(strategy="most_frequent")
    model.fit(X_train, y_train)
    return model


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Train interpretable linear baseline with scaling."""
    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(**LOGISTIC_PARAMS)),
    ])
    model.fit(X_train, y_train)
    return model


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Train non-linear tree baseline."""
    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(**RF_PARAMS)),
    ])
    model.fit(X_train, y_train)
    return model


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> Optional[Any]:
    """Train LightGBM if available."""
    if lgb is None:
        warnings.warn("lightgbm is not installed. Skipping LightGBM.")
        return None

    train_set = lgb.Dataset(X_train, label=y_train, feature_name=list(X_train.columns))
    valid_set = lgb.Dataset(X_valid, label=y_valid, feature_name=list(X_valid.columns), reference=train_set)

    callbacks = [
        lgb.early_stopping(EARLY_STOPPING_ROUNDS),
        lgb.log_evaluation(LOG_EVALUATION_PERIOD),
    ]

    model = lgb.train(
        LIGHTGBM_PARAMS,
        train_set,
        num_boost_round=NUM_BOOST_ROUNDS,
        valid_sets=[train_set, valid_set],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )

    return model


def predict_lightgbm(model: Any, X: pd.DataFrame) -> np.ndarray:
    """Predict P(rain=1) from LightGBM booster."""
    return model.predict(X, num_iteration=model.best_iteration)


def save_model_artifact(model_name: str, model: Any) -> None:
    """Save model artifact if joblib is available."""
    if joblib is None:
        return

    if model_name == "LightGBM":
        model.save_model(str(MODEL_DIR / f"{model_name}.txt"))
    else:
        joblib.dump(model, MODEL_DIR / f"{model_name}.joblib")


# =============================================================================
# MAIN EVALUATION LOOP
# =============================================================================

def evaluate_model_on_splits(
    model_name: str,
    model: Any,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_cols: List[str],
) -> List[Dict[str, Any]]:
    """Predict, optimize threshold on validation, evaluate train/valid/test."""
    if model_name == "LightGBM":
        y_train_prob = predict_lightgbm(model, X_train)
        y_valid_prob = predict_lightgbm(model, X_valid)
        y_test_prob = predict_lightgbm(model, X_test)
    else:
        y_train_prob = get_positive_proba(model, X_train)
        y_valid_prob = get_positive_proba(model, X_valid)
        y_test_prob = get_positive_proba(model, X_test)

    threshold, curve = optimize_threshold(
        y_true=y_valid,
        y_prob=y_valid_prob,
        model_name=model_name,
        metric_name=THRESHOLD_SELECTION_METRIC,
    )

    plot_threshold_curve(curve, model_name)

    results = []

    for split_name, split_df, y_true, y_prob in [
        ("train", train_df, y_train, y_train_prob),
        ("validation", valid_df, y_valid, y_valid_prob),
        ("test", test_df, y_test, y_test_prob),
    ]:
        metrics = evaluate_predictions(y_true, y_prob, threshold)
        metrics.update({
            "model": model_name,
            "split": split_name,
            "threshold_selection_metric": THRESHOLD_SELECTION_METRIC,
            "n_rows": int(len(y_true)),
            "rain_rate": float(np.mean(y_true)),
        })
        results.append(metrics)

        save_predictions(
            df_split=split_df,
            y_true=y_true,
            y_prob=y_prob,
            threshold=threshold,
            model_name=model_name,
            split_name=split_name,
        )

        save_calibration_report(
            y_true=y_true,
            y_prob=y_prob,
            model_name=model_name,
            split_name=split_name,
        )

        if split_name in {"validation", "test"}:
            plot_confusion_matrix_from_values(
                y_true=y_true,
                y_prob=y_prob,
                threshold=threshold,
                model_name=model_name,
                split_name=split_name,
            )

    save_feature_importance(model_name, model, feature_cols)
    save_model_artifact(model_name, model)

    return results


def save_metric_explanation() -> None:
    """Save metric rationale for the report."""
    explanation = [
        {
            "metric": "PR-AUC",
            "role": "Primary probability/ranking metric for the rain class",
            "reason": (
                "Rain is the positive class and may be less frequent than no-rain. "
                "PR-AUC focuses on precision-recall trade-off for rain events and "
                "is more informative than accuracy under class imbalance."
            ),
        },
        {
            "metric": "ROC-AUC",
            "role": "Secondary probability/ranking metric",
            "reason": (
                "Measures how well the model ranks rainy days above non-rainy days "
                "across all thresholds. Useful for discrimination but can look "
                "optimistic under imbalance, so it should not be the only metric."
            ),
        },
        {
            "metric": "Brier Score",
            "role": "Primary probability quality / calibration metric",
            "reason": (
                "The project predicts rain probability, so the numeric probability "
                "must be meaningful, not only the final 0/1 label. Lower Brier Score "
                "means predicted probabilities are closer to actual outcomes."
            ),
        },
        {
            "metric": "ECE",
            "role": "Calibration diagnostic",
            "reason": (
                "Shows whether predicted probabilities match observed rain frequency "
                "in probability bins, e.g. days predicted around 80% rain should rain "
                "approximately 80% of the time."
            ),
        },
        {
            "metric": "F1 Rain",
            "role": "Default threshold selection metric",
            "reason": (
                "Balances Precision Rain and Recall Rain. It is suitable when we want "
                "a fair trade-off between false rain alarms and missed rain events."
            ),
        },
        {
            "metric": "F2 Rain",
            "role": "Recall-oriented diagnostic",
            "reason": (
                "Gives more weight to Recall Rain. Useful if missing rainy days is "
                "considered more costly than false alarms."
            ),
        },
        {
            "metric": "Accuracy",
            "role": "Secondary descriptive metric",
            "reason": (
                "Easy to understand but not sufficient because a majority no-rain "
                "baseline can obtain non-trivial accuracy while failing to detect rain."
            ),
        },
    ]

    pd.DataFrame(explanation).to_csv(REPORT_DIR / "metric_rationale.csv", index=False)


def save_metadata(
    input_csv: Path,
    feature_cols: List[str],
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    trained_models: List[str],
) -> None:
    """Save reproducibility metadata."""
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "task": "Probabilistic Binary Rainfall Classification",
        "task_definition": (
            "Use meteorological features of day t to estimate P(rain=1) for day t, "
            "then classify Rain/No-rain using a threshold optimized on validation data."
        ),
        "project_framing": (
            "The model is used to validate dataset usefulness, not as the main "
            "contribution of the DS108 preprocessing project."
        ),
        "input_csv": str(input_csv),
        "target_col": TARGET_COL,
        "positive_label": POSITIVE_LABEL,
        "train_period": {
            "start": train_df["time"].min().strftime("%Y-%m-%d"),
            "end": train_df["time"].max().strftime("%Y-%m-%d"),
        },
        "validation_period": {
            "start": valid_df["time"].min().strftime("%Y-%m-%d"),
            "end": valid_df["time"].max().strftime("%Y-%m-%d"),
        },
        "test_period": {
            "start": test_df["time"].min().strftime("%Y-%m-%d"),
            "end": test_df["time"].max().strftime("%Y-%m-%d"),
        },
        "threshold_selection": {
            "metric": THRESHOLD_SELECTION_METRIC,
            "selection_data": "validation split only",
            "grid_min": float(THRESHOLD_GRID.min()),
            "grid_max": float(THRESHOLD_GRID.max()),
            "grid_size": int(len(THRESHOLD_GRID)),
        },
        "metric_priority": [
            "PR-AUC",
            "Brier Score",
            "ROC-AUC",
            "Recall Rain",
            "F1 Rain",
            "F2 Rain",
            "Accuracy as secondary metric",
        ],
        "n_features": int(len(feature_cols)),
        "feature_cols": feature_cols,
        "trained_models": trained_models,
        "leakage_controls": [
            "chronological train/validation/test split",
            "threshold selected only on validation data",
            "test set used only for final evaluation",
            "rain_target created in Step 6; PRCP/PRCP_mm not used as direct features",
            "time and STATION dropped from model predictors",
            "only numeric model features used",
        ],
    }

    _write_json(metadata, REPORT_DIR / "model_validation_metadata.json")


def model_validation(input_csv: Path | str = DEFAULT_INPUT_CSV) -> pd.DataFrame:
    """Run full Step 7 model-based validation."""
    ensure_directories()

    input_csv = Path(input_csv)

    print("\n=== STEP 7: PROBABILISTIC MODEL-BASED DATASET VALIDATION ===")
    print("Model is used to verify dataset usefulness, not as the main project contribution.")
    print("Task: estimate P(rain = 1) for day t, then classify Rain/No-rain by threshold.")
    print(f"Input CSV: {input_csv}")

    df = load_feature_engineered_data(input_csv)
    train_df, valid_df, test_df = split_train_valid_test(df)

    save_split_summary(train_df, valid_df, test_df)

    feature_cols = build_feature_columns(df)
    save_feature_columns(feature_cols)

    X_train, y_train, X_valid, y_valid, X_test, y_test = make_xy(
        train_df,
        valid_df,
        test_df,
        feature_cols,
    )

    all_results: List[Dict[str, Any]] = []
    trained_model_names: List[str] = []

    # Majority baseline
    print("\n--- Training Majority Baseline ---")
    majority = train_majority_baseline(X_train, y_train)
    trained_model_names.append("MajorityBaseline")
    all_results.extend(
        evaluate_model_on_splits(
            "MajorityBaseline",
            majority,
            train_df,
            valid_df,
            test_df,
            X_train,
            y_train,
            X_valid,
            y_valid,
            X_test,
            y_test,
            feature_cols,
        )
    )

    # Logistic Regression
    print("\n--- Training Logistic Regression Baseline ---")
    logistic = train_logistic_regression(X_train, y_train)
    trained_model_names.append("LogisticRegression")
    all_results.extend(
        evaluate_model_on_splits(
            "LogisticRegression",
            logistic,
            train_df,
            valid_df,
            test_df,
            X_train,
            y_train,
            X_valid,
            y_valid,
            X_test,
            y_test,
            feature_cols,
        )
    )

    # Random Forest
    print("\n--- Training Random Forest Baseline ---")
    rf = train_random_forest(X_train, y_train)
    trained_model_names.append("RandomForest")
    all_results.extend(
        evaluate_model_on_splits(
            "RandomForest",
            rf,
            train_df,
            valid_df,
            test_df,
            X_train,
            y_train,
            X_valid,
            y_valid,
            X_test,
            y_test,
            feature_cols,
        )
    )

    # LightGBM
    print("\n--- Training LightGBM Validation Model ---")
    lgb_model = train_lightgbm(X_train, y_train, X_valid, y_valid)
    if lgb_model is not None:
        trained_model_names.append("LightGBM")
        all_results.extend(
            evaluate_model_on_splits(
                "LightGBM",
                lgb_model,
                train_df,
                valid_df,
                test_df,
                X_train,
                y_train,
                X_valid,
                y_valid,
                X_test,
                y_test,
                feature_cols,
            )
        )

    metrics_df = pd.DataFrame(all_results)

    # Friendly column order.
    front_cols = [
        "model",
        "split",
        "threshold_selection_metric",
        "threshold",
        "accuracy",
        "balanced_accuracy",
        "precision_rain",
        "recall_rain",
        "f1_rain",
        "f2_rain",
        "specificity_no_rain",
        "false_alarm_rate",
        "miss_rate",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "log_loss",
        "ece",
        "n_rows",
        "rain_rate",
        "tp",
        "fp",
        "tn",
        "fn",
    ]
    ordered_cols = [c for c in front_cols if c in metrics_df.columns]
    remaining_cols = [c for c in metrics_df.columns if c not in ordered_cols]
    metrics_df = metrics_df[ordered_cols + remaining_cols]

    metrics_df.to_csv(REPORT_DIR / "model_metrics_all.csv", index=False)

    test_ranked = (
        metrics_df[metrics_df["split"] == "test"]
        .sort_values(
            by=["pr_auc", "brier_score", "f1_rain"],
            ascending=[False, True, False],
        )
        .reset_index(drop=True)
    )
    test_ranked.to_csv(REPORT_DIR / "model_metrics_test_ranked.csv", index=False)

    save_metric_explanation()
    save_metadata(
        input_csv=input_csv,
        feature_cols=feature_cols,
        train_df=train_df,
        valid_df=valid_df,
        test_df=test_df,
        trained_models=trained_model_names,
    )

    print("\n[SUCCESS] Model validation complete.")
    print(f"Reports saved to: {REPORT_DIR}")
    print(f"Models saved to:  {MODEL_DIR}")

    display_cols = [
        "model",
        "threshold",
        "accuracy",
        "precision_rain",
        "recall_rain",
        "f1_rain",
        "f2_rain",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "ece",
    ]
    display_cols = [c for c in display_cols if c in test_ranked.columns]

    print("\nTest metrics ranked:")
    print(test_ranked[display_cols].to_string(index=False))

    return metrics_df


def main() -> None:
    model_validation(DEFAULT_INPUT_CSV)


if __name__ == "__main__":
    main()
