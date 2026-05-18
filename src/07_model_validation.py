"""
Model validation module for rainfall dataset quality checking.

This file replaces `_07_model.py`.

Important framing for DS108:
- The model is NOT the main contribution of this preprocessing project.
- The model is used as an empirical validation tool to check whether the
  constructed dataset and engineered features contain useful predictive signal.

Main improvements over the original version:
1. Default input is the feature-engineered dataset:
   data/feature_engineering/feature_engineered_data.csv
2. The target is `rain_target`, created in Step 6. This script does NOT
   recreate/binarize PRCP again.
3. Metrics include Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, and
   Brier Score for the rain class.
4. Threshold is optimized on the validation set instead of fixed at 0.5.
5. Baselines are included:
   - Always predict majority class
   - Logistic Regression
   - Random Forest
   - LightGBM
6. All metrics, thresholds, predictions, feature importance, plots, and models
   are saved to disk for reproducibility.
"""

from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
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


# ============================================================================
# CONFIGURATION
# ============================================================================

RANDOM_SEED = 108
TARGET_COL = "rain_target"
POSITIVE_LABEL = 1

# Default input is the feature engineering output, not the Silver layer.
# If you prefer the stricter numeric-only file from Step 6, change this to:
# data/feature_engineering/model_ready_data.csv
DEFAULT_FEATURE_FILE = "feature_engineered_data.csv"

# Time-based split to avoid future-to-past leakage.
TRAIN_END = "2023-01-01"
VALID_START = "2023-01-01"
VALID_END = "2024-01-01"
TEST_START = "2024-01-01"

# Threshold optimization
THRESHOLD_METRIC = "f1"  # one of {"f1", "recall", "precision", "balanced_f1"}
THRESHOLD_GRID = np.round(np.linspace(0.05, 0.95, 181), 4)

# Whether to keep static station coordinates as features.
# For a pure temporal generalization check across the same stations, this can be True.
# If the instructor worries about station memorization, set this to False.
KEEP_LOCATION_FEATURES = True

# Columns that should never be used as model features.
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
}

LOCATION_COLS = {"LATITUDE", "LONGITUDE", "ELEVATION"}

# Drop columns with these suffixes because they are audit/source indicators,
# not meteorological numeric features.
DROP_SUFFIXES = ("_source",)

# LightGBM parameters
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

# Random Forest baseline
RF_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_leaf": 2,
    "class_weight": "balanced_subsample",
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

# Logistic Regression baseline
LOGISTIC_PARAMS = {
    "max_iter": 2000,
    "class_weight": "balanced",
    "solver": "lbfgs",
    "random_state": RANDOM_SEED,
}


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
DEFAULT_INPUT_CSV = BASE_DIR / "data" / "feature_engineering" / DEFAULT_FEATURE_FILE

MODEL_DIR = BASE_DIR / "models" / "rainfall_validation"
REPORT_DIR = BASE_DIR / "reports" / "model_validation"
PLOT_DIR = REPORT_DIR / "plots"
PRED_DIR = REPORT_DIR / "predictions"


def ensure_directories() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)


def _write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


# ============================================================================
# DATA LOADING AND SPLITTING
# ============================================================================

def load_feature_engineered_data(input_csv: Path | str = DEFAULT_INPUT_CSV) -> pd.DataFrame:
    """
    Load feature-engineered data.

    This function intentionally requires `rain_target`. It does not create the
    target from PRCP again, because target creation belongs to Step 6.
    """
    input_csv = Path(input_csv)
    if not input_csv.exists():
        raise FileNotFoundError(
            f"Input feature file not found: {input_csv}. "
            "Run Step 6 feature engineering first."
        )

    df = pd.read_csv(input_csv)

    if "time" not in df.columns:
        raise ValueError("Input data must contain a `time` column for time-based splitting.")

    if TARGET_COL not in df.columns:
        raise ValueError(
            f"Input data must contain `{TARGET_COL}` created by Step 6. "
            "Do not recreate target from PRCP in Step 7."
        )

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time", TARGET_COL]).copy()
    df[TARGET_COL] = df[TARGET_COL].astype(int)

    invalid_target = set(df[TARGET_COL].unique()) - {0, 1}
    if invalid_target:
        raise ValueError(f"`{TARGET_COL}` must be binary 0/1. Found: {invalid_target}")

    df = df.sort_values("time").reset_index(drop=True)

    return df


def split_train_valid_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data chronologically to reduce leakage."""
    train_df = df[df["time"] < TRAIN_END].copy()
    valid_df = df[(df["time"] >= VALID_START) & (df["time"] < VALID_END)].copy()
    test_df = df[df["time"] >= TEST_START].copy()

    if train_df.empty or valid_df.empty or test_df.empty:
        raise ValueError(
            "Train/valid/test split produced an empty set. "
            "Check date range and time column."
        )

    return train_df, valid_df, test_df


# ============================================================================
# FEATURE SELECTION FOR MODEL VALIDATION
# ============================================================================

def build_feature_columns(df: pd.DataFrame) -> List[str]:
    """
    Choose model feature columns.

    The selection is intentionally conservative:
    - target/leakage columns are removed
    - time/string/source columns are removed
    - only numeric columns are kept
    """
    drop_cols = set(ALWAYS_DROP_COLS)

    if not KEEP_LOCATION_FEATURES:
        drop_cols.update(LOCATION_COLS)

    candidate_cols = []
    for col in df.columns:
        if col in drop_cols:
            continue

        if any(col.endswith(suffix) for suffix in DROP_SUFFIXES):
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


# ============================================================================
# METRICS AND THRESHOLD OPTIMIZATION
# ============================================================================

def safe_roc_auc(y_true: pd.Series, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_prob))


def safe_pr_auc(y_true: pd.Series, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


def calculate_metrics(
    y_true: pd.Series,
    y_prob: np.ndarray,
    threshold: float,
    model_name: str,
    split_name: str,
) -> Dict[str, object]:
    """
    Calculate binary classification metrics for the rain class.
    """
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)

    labels = [0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tn, fp, fn, tp = cm.ravel()

    metrics = {
        "model": model_name,
        "split": split_name,
        "threshold": float(threshold),
        "n_samples": int(len(y_true)),
        "positive_rate_true": float(np.mean(y_true)),
        "positive_rate_pred": float(np.mean(y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_rain": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_rain": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_rain": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": safe_roc_auc(y_true, y_prob),
        "pr_auc": safe_pr_auc(y_true, y_prob),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    return metrics


def threshold_score(y_true: pd.Series, y_prob: np.ndarray, threshold: float, metric: str) -> float:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    if metric == "f1":
        return float(f1)
    if metric == "recall":
        return float(recall)
    if metric == "precision":
        return float(precision)
    if metric == "balanced_f1":
        # Penalize thresholds that get recall but collapse precision, or vice versa.
        return float((f1 + min(precision, recall)) / 2)

    raise ValueError(f"Unsupported threshold metric: {metric}")


def optimize_threshold(
    y_valid: pd.Series,
    y_valid_prob: np.ndarray,
    model_name: str,
    metric: str = THRESHOLD_METRIC,
) -> Tuple[float, pd.DataFrame]:
    """
    Optimize decision threshold on validation set only.

    Test set is not used for threshold selection.
    """
    rows = []

    for threshold in THRESHOLD_GRID:
        rows.append({
            "model": model_name,
            "threshold": float(threshold),
            "metric": metric,
            "score": threshold_score(y_valid, y_valid_prob, threshold, metric),
        })

    report = pd.DataFrame(rows)
    best_row = report.sort_values(["score", "threshold"], ascending=[False, True]).iloc[0]
    best_threshold = float(best_row["threshold"])

    return best_threshold, report


# ============================================================================
# BASELINE AND MODEL TRAINING
# ============================================================================

def train_majority_baseline(y_train: pd.Series) -> DummyClassifier:
    """
    Always predict the majority class observed in training data.
    """
    model = DummyClassifier(strategy="most_frequent")
    # DummyClassifier requires X, but strategy ignores it.
    dummy_x = np.zeros((len(y_train), 1))
    model.fit(dummy_x, y_train)
    return model


def predict_majority_baseline(model: DummyClassifier, n_rows: int) -> np.ndarray:
    """
    Return probability of rain from DummyClassifier.
    """
    dummy_x = np.zeros((n_rows, 1))
    proba = model.predict_proba(dummy_x)

    if proba.shape[1] == 1:
        # If training had only one class, handle gracefully.
        cls = int(model.classes_[0])
        return np.ones(n_rows) if cls == 1 else np.zeros(n_rows)

    class_to_index = {int(cls): idx for idx, cls in enumerate(model.classes_)}
    return proba[:, class_to_index.get(1, 0)]


def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """
    Logistic Regression baseline with train-only imputation/scaling inside a pipeline.
    """
    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(**LOGISTIC_PARAMS)),
    ])
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """
    Random Forest baseline with train-only median imputation.
    """
    model = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(**RF_PARAMS)),
    ])
    model.fit(X_train, y_train)
    return model


def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
) -> Optional[object]:
    """
    Train LightGBM model. Returns None if lightgbm is unavailable.
    """
    if lgb is None:
        warnings.warn("lightgbm is not installed. Skipping LightGBM model.")
        return None

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = float(neg / pos) if pos > 0 else 1.0

    params = dict(LIGHTGBM_PARAMS)
    params["scale_pos_weight"] = scale_pos_weight

    dtrain = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain, free_raw_data=False)

    model = lgb.train(
        params,
        dtrain,
        num_boost_round=NUM_BOOST_ROUNDS,
        valid_sets=[dtrain, dvalid],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=LOG_EVALUATION_PERIOD),
        ],
    )

    return model


def predict_probability(model: object, X: pd.DataFrame, model_name: str) -> np.ndarray:
    """
    Predict rain probability for supported model types.
    """
    if model_name == "LightGBM":
        return np.asarray(model.predict(X, num_iteration=getattr(model, "best_iteration", None)))

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.shape[1] == 1:
            # Single-class training edge case.
            if hasattr(model, "classes_"):
                cls = int(model.classes_[0])
            else:
                cls = int(model.named_steps["clf"].classes_[0])
            return np.ones(len(X)) if cls == 1 else np.zeros(len(X))

        # Pipeline exposes classes_ via final estimator.
        if hasattr(model, "classes_"):
            classes = model.classes_
        elif hasattr(model, "named_steps") and "clf" in model.named_steps:
            classes = model.named_steps["clf"].classes_
        else:
            classes = np.array([0, 1])

        class_to_index = {int(cls): idx for idx, cls in enumerate(classes)}
        return proba[:, class_to_index.get(1, 1)]

    raise TypeError(f"Unsupported model type for probability prediction: {model_name}")


# ============================================================================
# SAVING MODELS, REPORTS, AND PLOTS
# ============================================================================

def save_model(model: object, model_name: str) -> Optional[Path]:
    """
    Save model artifact.
    """
    ensure_directories()

    safe_name = model_name.lower().replace(" ", "_")

    if model_name == "LightGBM" and lgb is not None:
        path = MODEL_DIR / f"{safe_name}.txt"
        model.save_model(str(path))
        return path

    if joblib is not None:
        path = MODEL_DIR / f"{safe_name}.joblib"
        joblib.dump(model, path)
        return path

    warnings.warn("joblib is unavailable. Model artifact was not saved.")
    return None


def save_predictions(
    df_split: pd.DataFrame,
    y_true: pd.Series,
    y_prob: np.ndarray,
    threshold: float,
    model_name: str,
    split_name: str,
) -> Path:
    """
    Save predictions for reproducibility and error analysis.
    """
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)

    out = pd.DataFrame({
        "time": df_split["time"].values,
        "y_true": y_true.values,
        "y_prob": y_prob,
        "threshold": threshold,
        "y_pred": y_pred,
    })

    if "STATION" in df_split.columns:
        out.insert(1, "STATION", df_split["STATION"].values)

    path = PRED_DIR / f"{model_name.lower().replace(' ', '_')}_{split_name}_predictions.csv"
    out.to_csv(path, index=False)

    return path


def save_classification_report(
    y_true: pd.Series,
    y_prob: np.ndarray,
    threshold: float,
    model_name: str,
    split_name: str,
) -> Path:
    """
    Save sklearn classification report as text.
    """
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    report = classification_report(y_true, y_pred, digits=4, zero_division=0)

    path = REPORT_DIR / f"{model_name.lower().replace(' ', '_')}_{split_name}_classification_report.txt"
    path.write_text(report, encoding="utf-8")

    return path


def plot_confusion_matrix(
    y_true: pd.Series,
    y_prob: np.ndarray,
    threshold: float,
    model_name: str,
    split_name: str,
) -> Path:
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm)
    ax.set_title(f"Confusion Matrix - {model_name} ({split_name})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No rain", "Rain"])
    ax.set_yticklabels(["No rain", "Rain"])

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    path = PLOT_DIR / f"confusion_matrix_{model_name.lower().replace(' ', '_')}_{split_name}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)

    return path


def plot_threshold_curve(threshold_report: pd.DataFrame, model_name: str) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(threshold_report["threshold"], threshold_report["score"])
    ax.set_xlabel("Threshold")
    ax.set_ylabel(str(threshold_report["metric"].iloc[0]))
    ax.set_title(f"Validation Threshold Optimization - {model_name}")
    fig.tight_layout()

    path = PLOT_DIR / f"threshold_curve_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)

    return path


def save_feature_importance(
    model: object,
    model_name: str,
    feature_cols: List[str],
    top_n: int = 30,
) -> Optional[Path]:
    """
    Save feature importance for tree-based models when available.
    """
    ensure_directories()

    importance = None
    importance_type = ""

    if model_name == "LightGBM" and lgb is not None:
        importance = model.feature_importance(importance_type="gain")
        importance_type = "gain"
    elif model_name == "RandomForest":
        clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
        if hasattr(clf, "feature_importances_"):
            importance = clf.feature_importances_
            importance_type = "gini_importance"

    if importance is None:
        return None

    imp_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importance,
        "importance_type": importance_type,
    }).sort_values("importance", ascending=False)

    csv_path = REPORT_DIR / f"feature_importance_{model_name.lower().replace(' ', '_')}.csv"
    imp_df.to_csv(csv_path, index=False)

    top = imp_df.head(top_n).sort_values("importance", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.22)))
    ax.barh(top["feature"], top["importance"])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importance - {model_name}")
    fig.tight_layout()

    plot_path = PLOT_DIR / f"feature_importance_{model_name.lower().replace(' ', '_')}.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    return csv_path


# ============================================================================
# MODEL RUNNER
# ============================================================================

def evaluate_model_workflow(
    model_name: str,
    model: object,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    optimize: bool = True,
) -> Tuple[List[Dict[str, object]], Optional[pd.DataFrame]]:
    """
    Evaluate a trained model on validation and test sets.
    """
    valid_prob = predict_probability(model, X_valid, model_name)
    test_prob = predict_probability(model, X_test, model_name)

    if optimize:
        best_threshold, threshold_report = optimize_threshold(
            y_valid,
            valid_prob,
            model_name=model_name,
            metric=THRESHOLD_METRIC,
        )
        threshold_report.to_csv(
            REPORT_DIR / f"threshold_report_{model_name.lower().replace(' ', '_')}.csv",
            index=False,
        )
        plot_threshold_curve(threshold_report, model_name)
    else:
        # Majority baseline should remain "always majority", not threshold-tuned.
        majority_class = int(np.round(np.mean(valid_prob))) if len(valid_prob) else 0
        best_threshold = 0.5 if majority_class == 1 else 1.0
        threshold_report = None

    metrics = [
        calculate_metrics(y_valid, valid_prob, best_threshold, model_name, "valid"),
        calculate_metrics(y_test, test_prob, best_threshold, model_name, "test"),
    ]

    save_predictions(valid_df, y_valid, valid_prob, best_threshold, model_name, "valid")
    save_predictions(test_df, y_test, test_prob, best_threshold, model_name, "test")

    save_classification_report(y_test, test_prob, best_threshold, model_name, "test")
    plot_confusion_matrix(y_test, test_prob, best_threshold, model_name, "test")

    return metrics, threshold_report


def train_and_validate_models(
    input_csv: Path | str = DEFAULT_INPUT_CSV,
    run_logistic: bool = True,
    run_random_forest: bool = True,
    run_lightgbm: bool = True,
) -> pd.DataFrame:
    """
    Main model validation pipeline.

    Returns
    -------
    metrics_df:
        Validation and test metrics for all models.
    """
    ensure_directories()

    print("\n=== STEP 7: MODEL-BASED DATASET VALIDATION ===")
    print("Model is used to verify dataset usefulness, not as the main project contribution.")
    print(f"Input CSV: {input_csv}")

    df = load_feature_engineered_data(input_csv)
    train_df, valid_df, test_df = split_train_valid_test(df)

    feature_cols = build_feature_columns(df)

    X_train, y_train, X_valid, y_valid, X_test, y_test = make_xy(
        train_df,
        valid_df,
        test_df,
        feature_cols,
    )

    split_summary = pd.DataFrame([
        {
            "split": "train",
            "n_rows": len(train_df),
            "start_date": str(train_df["time"].min().date()),
            "end_date": str(train_df["time"].max().date()),
            "positive_rate": float(y_train.mean()),
        },
        {
            "split": "valid",
            "n_rows": len(valid_df),
            "start_date": str(valid_df["time"].min().date()),
            "end_date": str(valid_df["time"].max().date()),
            "positive_rate": float(y_valid.mean()),
        },
        {
            "split": "test",
            "n_rows": len(test_df),
            "start_date": str(test_df["time"].min().date()),
            "end_date": str(test_df["time"].max().date()),
            "positive_rate": float(y_test.mean()),
        },
    ])
    split_summary.to_csv(REPORT_DIR / "split_summary.csv", index=False)

    pd.DataFrame({
        "feature": feature_cols,
        "dtype": [str(df[c].dtype) for c in feature_cols],
        "missing_rate": [float(df[c].isna().mean()) for c in feature_cols],
    }).to_csv(REPORT_DIR / "used_feature_columns.csv", index=False)

    all_metrics: List[Dict[str, object]] = []
    model_artifacts: Dict[str, Optional[str]] = {}

    # ---------------------------------------------------------------------
    # Majority baseline
    # ---------------------------------------------------------------------
    print("\n--- Training Majority Baseline ---")
    majority = train_majority_baseline(y_train)
    metrics, _ = evaluate_model_workflow(
        model_name="MajorityBaseline",
        model=majority,
        X_valid=pd.DataFrame(np.zeros((len(valid_df), 1))),
        y_valid=y_valid,
        X_test=pd.DataFrame(np.zeros((len(test_df), 1))),
        y_test=y_test,
        valid_df=valid_df,
        test_df=test_df,
        optimize=False,
    )
    all_metrics.extend(metrics)
    artifact = save_model(majority, "MajorityBaseline")
    model_artifacts["MajorityBaseline"] = str(artifact) if artifact else None

    # ---------------------------------------------------------------------
    # Logistic Regression baseline
    # ---------------------------------------------------------------------
    if run_logistic:
        print("\n--- Training Logistic Regression Baseline ---")
        logistic = train_logistic_regression(X_train, y_train)
        metrics, _ = evaluate_model_workflow(
            model_name="LogisticRegression",
            model=logistic,
            X_valid=X_valid,
            y_valid=y_valid,
            X_test=X_test,
            y_test=y_test,
            valid_df=valid_df,
            test_df=test_df,
            optimize=True,
        )
        all_metrics.extend(metrics)
        artifact = save_model(logistic, "LogisticRegression")
        model_artifacts["LogisticRegression"] = str(artifact) if artifact else None

    # ---------------------------------------------------------------------
    # Random Forest baseline
    # ---------------------------------------------------------------------
    if run_random_forest:
        print("\n--- Training Random Forest Baseline ---")
        rf = train_random_forest(X_train, y_train)
        metrics, _ = evaluate_model_workflow(
            model_name="RandomForest",
            model=rf,
            X_valid=X_valid,
            y_valid=y_valid,
            X_test=X_test,
            y_test=y_test,
            valid_df=valid_df,
            test_df=test_df,
            optimize=True,
        )
        all_metrics.extend(metrics)
        artifact = save_model(rf, "RandomForest")
        model_artifacts["RandomForest"] = str(artifact) if artifact else None
        save_feature_importance(rf, "RandomForest", feature_cols)

    # ---------------------------------------------------------------------
    # LightGBM main validation model
    # ---------------------------------------------------------------------
    if run_lightgbm:
        print("\n--- Training LightGBM Validation Model ---")
        lgb_model = train_lightgbm(X_train, y_train, X_valid, y_valid)
        if lgb_model is not None:
            metrics, _ = evaluate_model_workflow(
                model_name="LightGBM",
                model=lgb_model,
                X_valid=X_valid,
                y_valid=y_valid,
                X_test=X_test,
                y_test=y_test,
                valid_df=valid_df,
                test_df=test_df,
                optimize=True,
            )
            all_metrics.extend(metrics)
            artifact = save_model(lgb_model, "LightGBM")
            model_artifacts["LightGBM"] = str(artifact) if artifact else None
            save_feature_importance(lgb_model, "LightGBM", feature_cols)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df = metrics_df.sort_values(["split", "f1_rain", "pr_auc"], ascending=[True, False, False])
    metrics_df.to_csv(REPORT_DIR / "model_metrics_all.csv", index=False)

    test_metrics = metrics_df[metrics_df["split"] == "test"].copy()
    test_metrics = test_metrics.sort_values(["f1_rain", "pr_auc"], ascending=[False, False])
    test_metrics.to_csv(REPORT_DIR / "model_metrics_test_ranked.csv", index=False)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(input_csv),
        "target_column": TARGET_COL,
        "project_framing": (
            "Models are used as empirical validation that the preprocessed and "
            "feature-engineered dataset contains useful signal. They are not the "
            "main contribution of the DS108 preprocessing project."
        ),
        "split_policy": {
            "train": f"time < {TRAIN_END}",
            "valid": f"{VALID_START} <= time < {VALID_END}",
            "test": f"time >= {TEST_START}",
            "reason": "Chronological split reduces future-to-past leakage.",
        },
        "threshold_policy": {
            "optimized_on": "validation set only",
            "metric": THRESHOLD_METRIC,
            "grid_min": float(np.min(THRESHOLD_GRID)),
            "grid_max": float(np.max(THRESHOLD_GRID)),
            "grid_size": int(len(THRESHOLD_GRID)),
        },
        "feature_policy": {
            "n_features": int(len(feature_cols)),
            "keep_location_features": KEEP_LOCATION_FEATURES,
            "leakage_columns_dropped": sorted(ALWAYS_DROP_COLS),
            "drop_suffixes": DROP_SUFFIXES,
        },
        "model_artifacts": model_artifacts,
        "reports": {
            "split_summary": str(REPORT_DIR / "split_summary.csv"),
            "used_feature_columns": str(REPORT_DIR / "used_feature_columns.csv"),
            "all_metrics": str(REPORT_DIR / "model_metrics_all.csv"),
            "test_metrics_ranked": str(REPORT_DIR / "model_metrics_test_ranked.csv"),
            "predictions_dir": str(PRED_DIR),
            "plots_dir": str(PLOT_DIR),
        },
    }

    _write_json(metadata, REPORT_DIR / "model_validation_metadata.json")

    print("\n[SUCCESS] Model validation complete.")
    print(f"Reports saved to: {REPORT_DIR}")
    print(f"Models saved to:  {MODEL_DIR}")
    print("\nTest metrics ranked:")
    print(test_metrics[[
        "model", "threshold", "accuracy", "precision_rain",
        "recall_rain", "f1_rain", "roc_auc", "pr_auc", "brier_score"
    ]].to_string(index=False))

    return metrics_df


# Backward-compatible alias for old script name.
def train_rainfall_model(input_csv: str | Path | None = None) -> dict:
    if input_csv is None:
        input_csv = DEFAULT_INPUT_CSV

    metrics_df = train_and_validate_models(input_csv=input_csv)
    # Return best test row as dict for compatibility.
    test_df = metrics_df[metrics_df["split"] == "test"].sort_values(
        ["f1_rain", "pr_auc"],
        ascending=[False, False],
    )
    return test_df.iloc[0].to_dict() if not test_df.empty else {}


if __name__ == "__main__":
    train_and_validate_models(input_csv=DEFAULT_INPUT_CSV)
