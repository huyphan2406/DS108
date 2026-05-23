"""
Two-stage rainfall model training and evaluation module.

Stage 1:
- Classification model predicts rainfall occurrence:
  P(PRCP_label = 1)

Stage 2:
- Regression model predicts rainfall amount on rainy days:
  E(PRCP | PRCP_label = 1)

Final rainfall prediction:
- Hard decision:
  if predicted rain -> predicted amount
  else -> 0 mm

- Expected rainfall:
  P(rain) * predicted amount if rain
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import lightgbm as lgb
from pathlib import Path
from typing import Tuple
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

RANDOM_SEED = 108

TRAIN_END = "2023-01-01"
VALID_START = "2023-01-01"
VALID_END = "2024-01-01"
TEST_START = "2024-01-01"

TARGET_CLS = "PRCP_label"
TARGET_REG = "PRCP"
TARGET_REG_LOG = "PRCP_log1p"

COLS_TO_DROP = [
    "STATION",
    "time",
    "PRCP",
    "PRCP_label",
    "PRCP_log1p",
]

CLASSIFIER_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.02,
    "num_leaves": 48,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "is_unbalance": True,
    "seed": RANDOM_SEED,
    "verbosity": -1,
}

REGRESSOR_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
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

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_CSV = BASE_DIR / "data" / "feature_engineering" / "feature_engineered_data.csv"
OUTPUT_DIR = BASE_DIR / "models" / "rainfall_validation"


# ============================================================================
# 1. DATA LOADING & PREPROCESSING
# ============================================================================

def _load_and_prepare_data(csv_path: str | Path) -> pd.DataFrame:
    """
    Load feature-engineered data.

    Expected targets:
    - PRCP_label: classification target
    - PRCP: rainfall amount in mm
    - PRCP_log1p: log1p(PRCP), regression target
    """
    df = pd.read_csv(csv_path)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.dropna(subset=["time"]).reset_index(drop=True)

    required_targets = [TARGET_CLS, TARGET_REG, TARGET_REG_LOG]
    missing = [col for col in required_targets if col not in df.columns]

    if missing:
        raise KeyError(f"Thiếu target columns trong feature file: {missing}")

    df[TARGET_CLS] = pd.to_numeric(df[TARGET_CLS], errors="coerce").astype(int)
    df[TARGET_REG] = pd.to_numeric(df[TARGET_REG], errors="coerce").fillna(0).clip(lower=0)
    df[TARGET_REG_LOG] = pd.to_numeric(df[TARGET_REG_LOG], errors="coerce").fillna(
        np.log1p(df[TARGET_REG])
    )

    if "STATION" in df.columns:
        df["STATION"] = df["STATION"].astype(str)

    return df


def _split_train_valid_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into train, validation, and test sets by time."""
    train_df = df[df["time"] < TRAIN_END].copy()
    valid_df = df[(df["time"] >= VALID_START) & (df["time"] < VALID_END)].copy()
    test_df = df[df["time"] >= TEST_START].copy()

    return train_df, valid_df, test_df


def _get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return numeric feature columns after dropping target and metadata."""
    drop_cols = [col for col in COLS_TO_DROP if col in df.columns]

    X = df.drop(columns=drop_cols, errors="ignore")
    X = X.select_dtypes(include=[np.number])

    return X.columns.tolist()


def _prepare_xy(df: pd.DataFrame, feature_cols: list[str], target_col: str):
    """Prepare X and y."""
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    return X, y


# ============================================================================
# 2. STAGE 1 - CLASSIFICATION
# ============================================================================

def train_classifier(train_df: pd.DataFrame, valid_df: pd.DataFrame, feature_cols: list[str]) -> lgb.Booster:
    """Train Stage 1 LightGBM binary classifier."""
    print("\n=== STAGE 1: TRAIN RAIN / NO-RAIN CLASSIFIER ===")

    X_train, y_train = _prepare_xy(train_df, feature_cols, TARGET_CLS)
    X_valid, y_valid = _prepare_xy(valid_df, feature_cols, TARGET_CLS)

    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)

    model = lgb.train(
        CLASSIFIER_PARAMS,
        dtrain,
        num_boost_round=NUM_BOOST_ROUNDS,
        valid_sets=[dtrain, dvalid],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=LOG_EVALUATION_PERIOD),
        ],
    )

    print(f"✅ Stage 1 complete. Best iteration: {model.best_iteration}")
    return model


def find_best_threshold(y_true: pd.Series, y_prob: np.ndarray) -> tuple[float, pd.DataFrame]:
    """Choose best threshold on validation set using F1 score."""
    rows = []

    for threshold in np.arange(0.05, 0.951, 0.005):
        y_pred = (y_prob >= threshold).astype(int)

        rows.append({
            "threshold": threshold,
            "precision_rain": precision_score(y_true, y_pred, zero_division=0),
            "recall_rain": recall_score(y_true, y_pred, zero_division=0),
            "f1_rain": f1_score(y_true, y_pred, zero_division=0),
            "f2_rain": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        })

    threshold_df = pd.DataFrame(rows)
    best_row = threshold_df.sort_values("f1_rain", ascending=False).iloc[0]

    return float(best_row["threshold"]), threshold_df


def evaluate_classifier(y_true: pd.Series, y_prob: np.ndarray, threshold: float) -> dict:
    """Evaluate Stage 1 classification."""
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_rain": precision_score(y_true, y_pred, zero_division=0),
        "recall_rain": recall_score(y_true, y_pred, zero_division=0),
        "f1_rain": f1_score(y_true, y_pred, zero_division=0),
        "f2_rain": fbeta_score(y_true, y_pred, beta=2, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "brier_score": brier_score_loss(y_true, y_prob),
    }


# ============================================================================
# 3. STAGE 2 - REGRESSION ON RAINY DAYS
# ============================================================================

def train_regressor(train_df: pd.DataFrame, valid_df: pd.DataFrame, feature_cols: list[str]) -> lgb.Booster:
    """
    Train Stage 2 rainfall amount regressor.

    Training data:
    - only rainy days in train set where PRCP_label = 1.
    """
    print("\n=== STAGE 2: TRAIN RAINFALL AMOUNT REGRESSOR ===")

    train_rain = train_df[train_df[TARGET_CLS] == 1].copy()
    valid_rain = valid_df[valid_df[TARGET_CLS] == 1].copy()

    X_train, y_train = _prepare_xy(train_rain, feature_cols, TARGET_REG_LOG)
    X_valid, y_valid = _prepare_xy(valid_rain, feature_cols, TARGET_REG_LOG)

    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)

    model = lgb.train(
        REGRESSOR_PARAMS,
        dtrain,
        num_boost_round=NUM_BOOST_ROUNDS,
        valid_sets=[dtrain, dvalid],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS),
            lgb.log_evaluation(period=LOG_EVALUATION_PERIOD),
        ],
    )

    print(f"✅ Stage 2 complete. Best iteration: {model.best_iteration}")
    return model


def evaluate_regressor_on_rainy_days(
    model: lgb.Booster,
    df: pd.DataFrame,
    feature_cols: list[str],
) -> dict:
    """Evaluate Stage 2 only on true rainy days."""
    rainy_df = df[df[TARGET_CLS] == 1].copy()

    X = rainy_df[feature_cols]
    y_true = rainy_df[TARGET_REG].values

    pred_log = model.predict(X, num_iteration=model.best_iteration)
    y_pred = np.expm1(pred_log).clip(min=0)

    rmse = mean_squared_error(y_true, y_pred) ** 0.5

    return {
        "mae_rainy_days": mean_absolute_error(y_true, y_pred),
        "rmse_rainy_days": rmse,
        "r2_rainy_days": r2_score(y_true, y_pred),
        "n_rainy_days": len(rainy_df),
    }


# ============================================================================
# 4. TWO-STAGE EVALUATION
# ============================================================================

def evaluate_two_stage(
    classifier: lgb.Booster,
    regressor: lgb.Booster,
    df: pd.DataFrame,
    feature_cols: list[str],
    threshold: float,
) -> tuple[dict, pd.DataFrame]:
    """
    Evaluate end-to-end two-stage rainfall prediction.

    Final prediction:
    - hard_pred_mm = regressed amount if classifier predicts rain, else 0
    - expected_pred_mm = P(rain) * regressed amount
    """
    X = df[feature_cols]

    y_true_cls = df[TARGET_CLS].values
    y_true_mm = df[TARGET_REG].values

    rain_prob = classifier.predict(X, num_iteration=classifier.best_iteration)
    rain_pred = (rain_prob >= threshold).astype(int)

    amount_log = regressor.predict(X, num_iteration=regressor.best_iteration)
    amount_if_rain = np.expm1(amount_log).clip(min=0)

    hard_pred_mm = np.where(rain_pred == 1, amount_if_rain, 0)
    expected_pred_mm = rain_prob * amount_if_rain

    cls_metrics = evaluate_classifier(y_true_cls, rain_prob, threshold)

    hard_rmse = mean_squared_error(y_true_mm, hard_pred_mm) ** 0.5
    expected_rmse = mean_squared_error(y_true_mm, expected_pred_mm) ** 0.5

    metrics = {
        **cls_metrics,
        "mae_hard_mm": mean_absolute_error(y_true_mm, hard_pred_mm),
        "rmse_hard_mm": hard_rmse,
        "mae_expected_mm": mean_absolute_error(y_true_mm, expected_pred_mm),
        "rmse_expected_mm": expected_rmse,
    }

    pred_df = df[["time", "STATION"]].copy() if "STATION" in df.columns else df[["time"]].copy()
    pred_df["y_true_label"] = y_true_cls
    pred_df["y_true_prcp_mm"] = y_true_mm
    pred_df["rain_probability"] = rain_prob
    pred_df["rain_probability_percent"] = rain_prob * 100
    pred_df["y_pred_label"] = rain_pred
    pred_df["predicted_rainfall_if_rain_mm"] = amount_if_rain
    pred_df["predicted_rainfall_hard_mm"] = hard_pred_mm
    pred_df["expected_rainfall_mm"] = expected_pred_mm

    return metrics, pred_df


# ============================================================================
# 5. VISUALIZATION
# ============================================================================

def plot_confusion_matrix(y_true, y_pred) -> None:
    """Plot confusion matrix."""
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap="Blues")
    plt.title("Confusion Matrix - Stage 1 Rain Classification")
    plt.show()


def plot_feature_importance(model: lgb.Booster, max_features: int = 15, title: str = "Feature Importance") -> None:
    """Plot top feature importances."""
    lgb.plot_importance(model, max_num_features=max_features, importance_type="gain")
    plt.title(title)
    plt.show()


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def train_rainfall_model(input_csv: str | Path = None) -> dict:
    """
    Main two-stage rainfall modeling pipeline.

    Steps:
    1. Load feature-engineered data
    2. Split into train/valid/test by time
    3. Train Stage 1 classifier on all train data
    4. Select threshold on validation
    5. Train Stage 2 regressor on rainy train days only
    6. Evaluate on validation and test
    7. Save metrics and predictions
    """
    if input_csv is None:
        input_csv = INPUT_CSV

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("📥 Đang tải dữ liệu...")
    df = _load_and_prepare_data(input_csv)

    print("📊 Đang chia dữ liệu theo thời gian...")
    train_df, valid_df, test_df = _split_train_valid_test(df)

    print(f"   - Train: {len(train_df):,} mẫu")
    print(f"   - Valid: {len(valid_df):,} mẫu")
    print(f"   - Test:  {len(test_df):,} mẫu")

    feature_cols = _get_feature_columns(df)

    pd.DataFrame({"feature": feature_cols}).to_csv(
        OUTPUT_DIR / "used_feature_columns.csv",
        index=False,
    )

    print(f"   - Số feature dùng để train: {len(feature_cols)}")

    classifier = train_classifier(train_df, valid_df, feature_cols)

    valid_X = valid_df[feature_cols]
    valid_prob = classifier.predict(valid_X, num_iteration=classifier.best_iteration)
    best_threshold, threshold_df = find_best_threshold(valid_df[TARGET_CLS], valid_prob)

    threshold_df.to_csv(OUTPUT_DIR / "threshold_search_validation.csv", index=False)

    print(f"\n✅ Best threshold on validation: {best_threshold:.3f}")

    regressor = train_regressor(train_df, valid_df, feature_cols)

    print("\n=== VALIDATION EVALUATION ===")
    valid_metrics, valid_pred = evaluate_two_stage(
        classifier,
        regressor,
        valid_df,
        feature_cols,
        best_threshold,
    )

    print("\n=== TEST EVALUATION ===")
    test_metrics, test_pred = evaluate_two_stage(
        classifier,
        regressor,
        test_df,
        feature_cols,
        best_threshold,
    )

    rainy_reg_metrics = evaluate_regressor_on_rainy_days(
        regressor,
        test_df,
        feature_cols,
    )

    test_metrics.update(rainy_reg_metrics)

    metrics_df = pd.DataFrame([
        {"split": "validation", **valid_metrics},
        {"split": "test", **test_metrics},
    ])

    metrics_df.to_csv(OUTPUT_DIR / "two_stage_metrics.csv", index=False)
    valid_pred.to_csv(OUTPUT_DIR / "validation_predictions.csv", index=False)
    test_pred.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

    print("\n--- TEST METRICS ---")
    print(metrics_df[metrics_df["split"] == "test"].T)

    print("\n--- CLASSIFICATION REPORT TEST ---")
    print(classification_report(test_pred["y_true_label"], test_pred["y_pred_label"]))

    plot_confusion_matrix(test_pred["y_true_label"], test_pred["y_pred_label"])
    plot_feature_importance(classifier, title="Stage 1 Classifier Feature Importance")
    plot_feature_importance(regressor, title="Stage 2 Regressor Feature Importance")

    print(f"\n✅ Saved outputs to: {OUTPUT_DIR}")

    return test_metrics


if __name__ == "__main__":
    train_rainfall_model()