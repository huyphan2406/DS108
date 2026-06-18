"""
DS108 Streamlit Dashboard

Dashboard for the DS108 multi-source meteorological data pipeline.

Purpose:
- Inspect the final feature-engineered dataset.
- Show core EDA charts.
- Run model benchmarks directly in the web app when saved benchmark outputs are missing.

This app is for academic demonstration and reproducibility, not for operational weather forecasting.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

try:
    import plotly.express as px
except Exception:
    px = None

try:
    import lightgbm as lgb
    from lightgbm import LGBMClassifier, LGBMRegressor
except Exception:
    lgb = None
    LGBMClassifier = None
    LGBMRegressor = None


# =============================================================================
# App config
# =============================================================================

st.set_page_config(
    page_title="DS108 Rainfall Dataset Dashboard",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

RANDOM_STATE = 42
RAIN_THRESHOLD_MM = 0.1

APP_PATH = Path(__file__).resolve()
PROJECT_ROOT = APP_PATH.parents[1] if len(APP_PATH.parents) >= 2 else Path.cwd()

DEFAULT_DATA_CANDIDATES = [
    PROJECT_ROOT / "data" / "features" / "feature_engineered_data.csv",
    PROJECT_ROOT / "data" / "features" / "feature_engineered_data.parquet",
    PROJECT_ROOT / "feature_engineered_data.csv",
]

DEFAULT_CODEBOOK_CANDIDATES = [
    PROJECT_ROOT / "docs" / "Codebook.csv",
    PROJECT_ROOT / "Codebook.csv",
]

AIRFLOW_URL = "http://localhost:8081"
STREAMLIT_URL = "http://localhost:8502"

LEAKAGE_AND_NON_FEATURE_COLUMNS = {
    "PRCP",
    "PRCP_log1p",
    "PRCP_impute",
    "target_1stage_prcp_mm",
    "target_stage1_rain_label",
    "target_stage2_prcp_log1p",
    "STATION",
    "NAME",
    "DATE",
    "date",
    "time",
    "datetime",
    "year",
    "month",
    "day",
    "day_of_year",
}

PREFERRED_SELECTED_FEATURES = [
    "LATITUDE",
    "LONGITUDE",
    "ELEVATION",
    "latitude",
    "longitude",
    "elevation",
    "TEMP",
    "DEWP",
    "SLP",
    "STP",
    "WDSP",
    "VISIB",
    "t2m",
    "d2m",
    "sp",
    "msl",
    "u10",
    "v10",
    "z",
    "lsm",
    "u_850",
    "v_850",
    "q_850",
    "t_850",
    "z_850",
    "w_850",
    "u_500",
    "v_500",
    "q_500",
    "t_500",
    "z_500",
    "w_500",
    "dew_point_depression",
    "wind_speed_10m",
    "wind_speed_850",
    "wind_speed_500",
    "moisture_flux_850",
    "wind_shear_mag",
    "thickness_500_850",
    "lapse_rate_850_500",
    "ageostrophic_signal",
    "day_sin",
    "day_cos",
    "month_sin",
    "month_cos",
    "PRCP_lag_1",
    "PRCP_lag_2",
    "PRCP_past_3d_mean",
    "PRCP_past_3d_sum",
    "TEMP_lag_1",
    "SLP_lag_1",
    "VISIB_lag_1",
    "u_850_lag_1",
    "v_850_lag_1",
]


# =============================================================================
# Utility functions
# =============================================================================

def first_existing_path(paths: Iterable[Path]) -> Optional[Path]:
    for path in paths:
        if path.exists():
            return path
    return None


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_time_column(df: pd.DataFrame) -> Optional[str]:
    for col in ["time", "date", "DATE", "datetime"]:
        if col in df.columns:
            return col
    return None


def find_station_column(df: pd.DataFrame) -> Optional[str]:
    for col in ["STATION", "station", "station_id", "NAME"]:
        if col in df.columns:
            return col
    return None


def prepare_datetime_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    time_col = find_time_column(df)

    if time_col is not None:
        df["__date"] = pd.to_datetime(df[time_col], errors="coerce")
    else:
        df["__date"] = pd.NaT

    if df["__date"].notna().any():
        df["__year"] = df["__date"].dt.year
        df["__month"] = df["__date"].dt.month
        df["__dayofyear"] = df["__date"].dt.dayofyear
    else:
        if "year" in df.columns:
            df["__year"] = pd.to_numeric(df["year"], errors="coerce")
        else:
            df["__year"] = np.nan

        if "month" in df.columns:
            df["__month"] = pd.to_numeric(df["month"], errors="coerce")
        else:
            df["__month"] = np.nan

        df["__dayofyear"] = np.nan

    return df


def ensure_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "PRCP" not in df.columns:
        return df

    prcp = pd.to_numeric(df["PRCP"], errors="coerce")

    if "target_1stage_prcp_mm" not in df.columns:
        df["target_1stage_prcp_mm"] = prcp

    if "target_stage1_rain_label" not in df.columns:
        df["target_stage1_rain_label"] = np.where(
            prcp.notna(),
            (prcp >= RAIN_THRESHOLD_MM).astype(int),
            np.nan,
        )

    if "target_stage2_prcp_log1p" not in df.columns:
        df["target_stage2_prcp_log1p"] = np.where(
            prcp.notna(),
            np.log1p(np.clip(prcp, 0, None)),
            np.nan,
        )

    return df


@st.cache_data(show_spinner=False)
def load_table_from_path(path_str: str) -> pd.DataFrame:
    path = Path(path_str)

    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    df = normalize_column_names(df)
    df = prepare_datetime_columns(df)
    df = ensure_targets(df)
    return df


def load_uploaded_csv(uploaded_file) -> pd.DataFrame:
    df = pd.read_csv(uploaded_file)
    df = normalize_column_names(df)
    df = prepare_datetime_columns(df)
    df = ensure_targets(df)
    return df


@st.cache_data(show_spinner=False)
def load_codebook_from_path(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)


def safe_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def get_data_summary(df: pd.DataFrame) -> Dict[str, object]:
    station_col = find_station_column(df)

    if "__date" in df.columns and df["__date"].notna().any():
        date_min = df["__date"].min().strftime("%Y-%m-%d")
        date_max = df["__date"].max().strftime("%Y-%m-%d")
    else:
        date_min = "N/A"
        date_max = "N/A"

    if "__year" in df.columns and df["__year"].notna().any():
        years = sorted([int(y) for y in df["__year"].dropna().unique()])
        year_range = f"{min(years)}–{max(years)}"
    else:
        year_range = "N/A"

    n_stations = int(df[station_col].nunique(dropna=True)) if station_col else "N/A"

    if "PRCP" in df.columns:
        valid_target = int(pd.to_numeric(df["PRCP"], errors="coerce").notna().sum())
    else:
        valid_target = 0

    return {
        "rows": int(len(df)),
        "columns": int(df.shape[1]),
        "stations": n_stations,
        "date_min": str(date_min),
        "date_max": str(date_max),
        "year_range": str(year_range),
        "valid_target": int(valid_target),
    }


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.clip(y_pred, 0, None)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {
            "MAE": np.nan,
            "RMSE": np.nan,
            "WAPE (%)": np.nan,
            "R2": np.nan,
            "Bias": np.nan,
        }

    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    denom = float(np.sum(np.abs(y_true)))
    wape = float(np.sum(np.abs(y_true - y_pred)) / denom * 100) if denom > 0 else np.nan

    if len(y_true) >= 2 and np.var(y_true) > 0:
        r2 = float(1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    else:
        r2 = np.nan

    bias = float(np.mean(y_pred - y_true))

    return {
        "MAE": mae,
        "RMSE": rmse,
        "WAPE (%)": wape,
        "R2": r2,
        "Bias": bias,
    }


def format_metric(value: object, digits: int = 4) -> str:
    try:
        if value is None:
            return "N/A"
        value_float = float(value)
        if math.isnan(value_float):
            return "N/A"
        return f"{value_float:,.{digits}f}"
    except Exception:
        return str(value)


# =============================================================================
# Feature and split functions
# =============================================================================

def get_feature_columns(df: pd.DataFrame) -> List[str]:
    numeric_cols = df.select_dtypes(include=[np.number, "bool"]).columns.tolist()

    excluded = set(LEAKAGE_AND_NON_FEATURE_COLUMNS)
    excluded.update([c for c in df.columns if c.startswith("target_")])
    excluded.update([c for c in df.columns if c.startswith("__")])

    candidate_features = [c for c in numeric_cols if c not in excluded]

    valid_features = []
    for col in candidate_features:
        s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if s.notna().sum() == 0:
            continue
        if s.nunique(dropna=True) <= 1:
            continue
        valid_features.append(col)

    return valid_features


def get_selected_feature_columns(df: pd.DataFrame, full_features: List[str]) -> List[str]:
    selected = [c for c in PREFERRED_SELECTED_FEATURES if c in full_features]

    if len(selected) < 10:
        return full_features

    if len(selected) > 46:
        selected = selected[:46]

    return selected


def split_model_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if "PRCP" not in df.columns:
        raise ValueError("Không tìm thấy cột PRCP trong dữ liệu.")

    model_df = df.copy()
    model_df["PRCP"] = pd.to_numeric(model_df["PRCP"], errors="coerce")
    model_df = model_df[model_df["PRCP"].notna()].copy()
    model_df = model_df[model_df["__year"].notna()].copy()
    model_df["__year"] = model_df["__year"].astype(int)

    train_df = model_df[(model_df["__year"] >= 2015) & (model_df["__year"] <= 2022)].copy()
    valid_df = model_df[model_df["__year"] == 2023].copy()
    test_df = model_df[model_df["__year"] == 2024].copy()

    if train_df.empty or valid_df.empty or test_df.empty:
        raise ValueError(
            "Không đủ dữ liệu cho split 2015–2022 train, 2023 validation, 2024 test."
        )

    return train_df, valid_df, test_df


def impute_feature_matrix(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_train = train_df[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X_valid = valid_df[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    X_test = test_df[features].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)

    medians = X_train.median(numeric_only=True).fillna(0)

    X_train = X_train.fillna(medians).fillna(0)
    X_valid = X_valid.fillna(medians).fillna(0)
    X_test = X_test.fillna(medians).fillna(0)

    return X_train, X_valid, X_test


# =============================================================================
# Model benchmark functions
# =============================================================================

def persistence_prediction(df_all: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    if "PRCP_lag_1" in test_df.columns:
        pred = pd.to_numeric(test_df["PRCP_lag_1"], errors="coerce")
        return np.clip(pred.fillna(0).to_numpy(dtype=float), 0, None)

    station_col = find_station_column(df_all)

    if station_col is None or "__date" not in df_all.columns:
        return np.zeros(len(test_df), dtype=float)

    temp = df_all.copy()
    temp["PRCP"] = pd.to_numeric(temp["PRCP"], errors="coerce")
    temp = temp.sort_values([station_col, "__date"])
    temp["__persistence_pred"] = temp.groupby(station_col)["PRCP"].shift(1)

    pred = temp.loc[test_df.index, "__persistence_pred"]
    return np.clip(pred.fillna(0).to_numpy(dtype=float), 0, None)


def fit_lgbm_regressor(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
    objective: str,
) -> LGBMRegressor:
    if LGBMRegressor is None or lgb is None:
        raise ImportError("LightGBM chưa được cài đặt. Kiểm tra requirements.txt.")

    params = dict(
        objective=objective,
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )

    if objective == "tweedie":
        params["tweedie_variance_power"] = 1.3

    model = LGBMRegressor(**params)

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="rmse",
        callbacks=[
            lgb.early_stopping(stopping_rounds=80, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    return model


def fit_lgbm_classifier(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame,
    y_valid: np.ndarray,
) -> LGBMClassifier:
    if LGBMClassifier is None or lgb is None:
        raise ImportError("LightGBM chưa được cài đặt. Kiểm tra requirements.txt.")

    model = LGBMClassifier(
        objective="binary",
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=0.1,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric="binary_logloss",
        callbacks=[
            lgb.early_stopping(stopping_rounds=80, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    return model


def run_tweedie_benchmark(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
) -> np.ndarray:
    X_train, X_valid, X_test = impute_feature_matrix(train_df, valid_df, test_df, features)

    y_train = np.clip(pd.to_numeric(train_df["PRCP"], errors="coerce").to_numpy(dtype=float), 0, None)
    y_valid = np.clip(pd.to_numeric(valid_df["PRCP"], errors="coerce").to_numpy(dtype=float), 0, None)

    model = fit_lgbm_regressor(X_train, y_train, X_valid, y_valid, objective="tweedie")
    pred = model.predict(X_test)

    return np.clip(pred, 0, None)


def run_two_stage_benchmark(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
) -> np.ndarray:
    X_train, X_valid, X_test = impute_feature_matrix(train_df, valid_df, test_df, features)

    y_train_rain = (pd.to_numeric(train_df["PRCP"], errors="coerce") >= RAIN_THRESHOLD_MM).astype(int).to_numpy()
    y_valid_rain = (pd.to_numeric(valid_df["PRCP"], errors="coerce") >= RAIN_THRESHOLD_MM).astype(int).to_numpy()

    clf = fit_lgbm_classifier(X_train, y_train_rain, X_valid, y_valid_rain)
    rain_prob = clf.predict_proba(X_test)[:, 1]

    train_rain_mask = y_train_rain == 1
    valid_rain_mask = y_valid_rain == 1

    if train_rain_mask.sum() < 30 or valid_rain_mask.sum() < 10:
        mean_rain = float(
            np.nanmean(pd.to_numeric(train_df.loc[train_rain_mask, "PRCP"], errors="coerce"))
        )
        return np.clip(rain_prob * mean_rain, 0, None)

    y_train_amount = np.log1p(
        np.clip(
            pd.to_numeric(train_df.loc[train_rain_mask, "PRCP"], errors="coerce").to_numpy(dtype=float),
            0,
            None,
        )
    )

    y_valid_amount = np.log1p(
        np.clip(
            pd.to_numeric(valid_df.loc[valid_rain_mask, "PRCP"], errors="coerce").to_numpy(dtype=float),
            0,
            None,
        )
    )

    reg = fit_lgbm_regressor(
        X_train.loc[train_rain_mask],
        y_train_amount,
        X_valid.loc[valid_rain_mask],
        y_valid_amount,
        objective="regression",
    )

    amount_pred = np.expm1(reg.predict(X_test))
    amount_pred = np.clip(amount_pred, 0, None)

    final_pred = rain_prob * amount_pred
    return np.clip(final_pred, 0, None)


def run_live_benchmark(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    train_df, valid_df, test_df = split_model_data(df)

    full_features = get_feature_columns(df)
    selected_features = get_selected_feature_columns(df, full_features)

    if len(full_features) == 0:
        raise ValueError("Không tìm thấy feature numeric hợp lệ để huấn luyện.")

    y_test = np.clip(pd.to_numeric(test_df["PRCP"], errors="coerce").to_numpy(dtype=float), 0, None)

    rows = []

    persistence_pred = persistence_prediction(df, test_df)
    rows.append({
        "Setting": "Baseline",
        "Model": "Persistence",
        **compute_metrics(y_test, persistence_pred),
    })

    tweedie_full_pred = run_tweedie_benchmark(train_df, valid_df, test_df, full_features)
    rows.append({
        "Setting": f"Full {len(full_features)} features",
        "Model": "LightGBM Tweedie",
        **compute_metrics(y_test, tweedie_full_pred),
    })

    two_stage_full_pred = run_two_stage_benchmark(train_df, valid_df, test_df, full_features)
    rows.append({
        "Setting": f"Full {len(full_features)} features",
        "Model": "LightGBM Two-stage",
        **compute_metrics(y_test, two_stage_full_pred),
    })

    if selected_features != full_features:
        tweedie_selected_pred = run_tweedie_benchmark(train_df, valid_df, test_df, selected_features)
        rows.append({
            "Setting": f"Selected {len(selected_features)} features",
            "Model": "LightGBM Tweedie",
            **compute_metrics(y_test, tweedie_selected_pred),
        })

        two_stage_selected_pred = run_two_stage_benchmark(train_df, valid_df, test_df, selected_features)
        rows.append({
            "Setting": f"Selected {len(selected_features)} features",
            "Model": "LightGBM Two-stage",
            **compute_metrics(y_test, two_stage_selected_pred),
        })

    result_df = pd.DataFrame(rows)

    meta = {
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "test_rows": int(len(test_df)),
        "full_features": full_features,
        "selected_features": selected_features,
        "target_valid_rows": int(df["PRCP"].notna().sum()) if "PRCP" in df.columns else 0,
    }

    return result_df, meta


# =============================================================================
# Plot functions
# =============================================================================

def plot_prcp_distribution(df: pd.DataFrame) -> None:
    if px is None:
        st.warning("Plotly chưa được cài đặt.")
        return

    prcp = safe_numeric(df, "PRCP").dropna()
    prcp = prcp[prcp >= 0]

    if prcp.empty:
        st.warning("Không có dữ liệu PRCP hợp lệ.")
        return

    use_log = st.checkbox("Dùng log1p(PRCP) để dễ quan sát", value=True)
    plot_df = pd.DataFrame({"value": np.log1p(prcp) if use_log else prcp})
    x_label = "log1p(PRCP)" if use_log else "PRCP (mm)"

    fig = px.histogram(
        plot_df,
        x="value",
        nbins=60,
        title="Phân phối lượng mưa hằng ngày",
        labels={"value": x_label},
    )
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)


def plot_rain_balance(df: pd.DataFrame) -> None:
    if px is None:
        st.warning("Plotly chưa được cài đặt.")
        return

    prcp = safe_numeric(df, "PRCP").dropna()

    if prcp.empty:
        st.warning("Không có dữ liệu PRCP hợp lệ.")
        return

    label = np.where(prcp >= RAIN_THRESHOLD_MM, "Rainy day", "Non-rainy day")
    counts = pd.Series(label).value_counts().reset_index()
    counts.columns = ["label", "count"]

    fig = px.bar(
        counts,
        x="label",
        y="count",
        text="count",
        title=f"Tỉ lệ ngày mưa/không mưa theo ngưỡng {RAIN_THRESHOLD_MM} mm",
    )
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)


def plot_monthly_rain_probability(df: pd.DataFrame) -> None:
    if px is None:
        st.warning("Plotly chưa được cài đặt.")
        return

    if "__month" not in df.columns or "PRCP" not in df.columns:
        st.warning("Thiếu cột tháng hoặc PRCP.")
        return

    temp = df[["__month", "PRCP"]].copy()
    temp["PRCP"] = pd.to_numeric(temp["PRCP"], errors="coerce")
    temp = temp[temp["PRCP"].notna() & temp["__month"].notna()].copy()

    if temp.empty:
        st.warning("Không đủ dữ liệu để vẽ mùa vụ.")
        return

    temp["rain_label"] = (temp["PRCP"] >= RAIN_THRESHOLD_MM).astype(int)
    monthly = temp.groupby("__month", as_index=False)["rain_label"].mean()
    monthly["rain_probability"] = monthly["rain_label"] * 100

    fig = px.line(
        monthly,
        x="__month",
        y="rain_probability",
        markers=True,
        title="Tỉ lệ ngày mưa theo tháng",
        labels={"__month": "Tháng", "rain_probability": "Tỉ lệ ngày mưa (%)"},
    )
    fig.update_xaxes(dtick=1)
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)


def plot_station_distribution(df: pd.DataFrame) -> None:
    if px is None:
        st.warning("Plotly chưa được cài đặt.")
        return

    station_col = find_station_column(df)

    if station_col is None or "PRCP" not in df.columns:
        st.warning("Thiếu cột trạm hoặc PRCP.")
        return

    temp = df[[station_col, "PRCP"]].copy()
    temp["PRCP"] = pd.to_numeric(temp["PRCP"], errors="coerce")
    temp = temp[temp["PRCP"].notna()].copy()

    if temp.empty:
        st.warning("Không đủ dữ liệu để vẽ theo trạm.")
        return

    upper = float(temp["PRCP"].quantile(0.99))
    temp["PRCP_plot"] = temp["PRCP"].clip(upper=upper)

    fig = px.box(
        temp,
        x=station_col,
        y="PRCP_plot",
        points=False,
        title="Phân phối lượng mưa theo trạm",
        labels={"PRCP_plot": f"PRCP clipped at p99 ({upper:.2f} mm)"},
    )
    fig.update_layout(height=480)
    st.plotly_chart(fig, use_container_width=True)


def plot_correlation(df: pd.DataFrame) -> None:
    if px is None:
        st.warning("Plotly chưa được cài đặt.")
        return

    candidates = [
        "PRCP",
        "TEMP",
        "DEWP",
        "SLP",
        "STP",
        "WDSP",
        "VISIB",
        "t2m",
        "d2m",
        "msl",
        "sp",
        "u10",
        "v10",
        "u_850",
        "v_850",
        "q_850",
        "t_850",
        "z_500",
        "z_850",
        "moisture_flux_850",
        "dew_point_depression",
        "PRCP_lag_1",
        "PRCP_past_3d_sum",
    ]

    cols = [c for c in candidates if c in df.columns]

    if len(cols) < 3:
        cols = df.select_dtypes(include=[np.number]).columns.tolist()[:20]

    if len(cols) < 3:
        st.warning("Không đủ biến numeric để vẽ correlation heatmap.")
        return

    corr = df[cols].apply(pd.to_numeric, errors="coerce").corr()

    fig = px.imshow(
        corr,
        text_auto=False,
        aspect="auto",
        title="Correlation heatmap",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
    )
    fig.update_layout(height=650)
    st.plotly_chart(fig, use_container_width=True)


def show_leakage_check(df: pd.DataFrame) -> None:
    target_cols = [c for c in df.columns if c.startswith("target_")]
    risky_cols = [c for c in LEAKAGE_AND_NON_FEATURE_COLUMNS if c in df.columns]
    feature_cols = get_feature_columns(df)

    c1, c2, c3 = st.columns(3)
    c1.metric("Target columns", str(len(target_cols)))
    c2.metric("Excluded risky/non-feature columns", str(len(risky_cols)))
    c3.metric("Valid numeric features", str(len(feature_cols)))

    with st.expander("Các cột bị loại khỏi feature input"):
        st.write(sorted(set(risky_cols + target_cols)))

    with st.expander("Feature columns dùng được"):
        st.write(feature_cols)


def find_codebook_variable_column(codebook_df: pd.DataFrame) -> str:
    preferred = [
        "column_name",
        "Column",
        "column",
        "variable",
        "Variable",
        "name",
        "Name",
        "field",
        "Field",
    ]

    for col in preferred:
        if col in codebook_df.columns:
            return col

    return codebook_df.columns[0]


# =============================================================================
# UI
# =============================================================================

st.title("🌧️ DS108 Rainfall Dataset Dashboard")
st.caption(
    "Dashboard minh họa dữ liệu, EDA và model benchmark cho pipeline dự báo lượng mưa hằng ngày. "
    "Ứng dụng này không phải hệ thống dự báo vận hành production."
)

with st.sidebar:
    st.header("⚙️ Cấu hình")

    data_path = first_existing_path(DEFAULT_DATA_CANDIDATES)
    codebook_path = first_existing_path(DEFAULT_CODEBOOK_CANDIDATES)

    st.write("**Project root:**")
    st.code(str(PROJECT_ROOT), language="text")

    uploaded_data = None

    if data_path:
        st.success("Đã tìm thấy feature dataset")
        try:
            st.code(str(data_path.relative_to(PROJECT_ROOT)), language="text")
        except ValueError:
            st.code(str(data_path), language="text")
    else:
        st.error("Chưa tìm thấy feature dataset")
        uploaded_data = st.file_uploader(
            "Upload feature_engineered_data.csv để xem dashboard",
            type=["csv"],
        )

    if codebook_path:
        st.success("Đã tìm thấy Codebook")
        try:
            st.code(str(codebook_path.relative_to(PROJECT_ROOT)), language="text")
        except ValueError:
            st.code(str(codebook_path), language="text")
    else:
        st.warning("Chưa tìm thấy docs/Codebook.csv")

    st.divider()
    st.write("**Ports chuẩn:**")
    st.write(f"Airflow: {AIRFLOW_URL}")
    st.write(f"Streamlit: {STREAMLIT_URL}")

    if st.button("Clear Streamlit cache"):
        st.cache_data.clear()
        st.rerun()


df: Optional[pd.DataFrame] = None

try:
    if data_path:
        df = load_table_from_path(str(data_path))
    elif uploaded_data is not None:
        df = load_uploaded_csv(uploaded_data)
except Exception as exc:
    st.error(f"Không thể đọc dữ liệu: {exc}")
    df = None


if df is None:
    st.warning(
        "Chưa có `data/features/feature_engineered_data.csv`. "
        "Hãy chạy Airflow DAG hoặc upload file CSV ở sidebar."
    )
    st.info(
        f"Mở Airflow tại {AIRFLOW_URL}, bật DAG `ds108_rainfall_pipeline`, "
        "sau đó trigger pipeline để tạo dữ liệu."
    )
    st.stop()


summary = get_data_summary(df)

tab_overview, tab_eda, tab_benchmark, tab_codebook, tab_system = st.tabs(
    ["📌 Overview", "📊 EDA", "🧪 Model Benchmark", "📘 Codebook", "🛠️ System Check"]
)


with tab_overview:
    st.subheader("Tổng quan bộ dữ liệu")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{summary['rows']:,}")
    c2.metric("Columns", f"{summary['columns']:,}")
    c3.metric("Stations", str(summary["stations"]))
    c4.metric("Valid PRCP", f"{summary['valid_target']:,}")

    c5, c6, c7 = st.columns(3)
    c5.metric("Year range", str(summary["year_range"]))
    c6.metric("Start date", str(summary["date_min"]))
    c7.metric("End date", str(summary["date_max"]))

    st.subheader("Preview")
    st.dataframe(df.head(50), use_container_width=True)

    st.subheader("Target definition")
    st.markdown(
        f"""
        - `PRCP`: lượng mưa hằng ngày từ GSOD, đơn vị mm.
        - `target_stage1_rain_label`: 1 nếu `PRCP >= {RAIN_THRESHOLD_MM} mm`, ngược lại là 0.
        - `target_1stage_prcp_mm`: bằng `PRCP`.
        - `target_stage2_prcp_log1p`: `log(1 + PRCP)`.
        - `PRCP_impute`: proxy/internal cho lag/rolling, không thay thế `PRCP` và không làm target.
        """
    )

    show_leakage_check(df)


with tab_eda:
    st.subheader("Exploratory Data Analysis")

    eda_col1, eda_col2 = st.columns(2)

    with eda_col1:
        plot_prcp_distribution(df)

    with eda_col2:
        plot_rain_balance(df)

    eda_col3, eda_col4 = st.columns(2)

    with eda_col3:
        plot_monthly_rain_probability(df)

    with eda_col4:
        plot_station_distribution(df)

    st.divider()
    plot_correlation(df)


with tab_benchmark:
    st.subheader("Model Benchmark chạy trực tiếp trong app")

    st.markdown(
        """
        Khi không có file output kết quả mô hình, app sẽ huấn luyện và đánh giá trực tiếp từ
        `data/features/feature_engineered_data.csv`.

        Thiết lập:
        - Train: 2015–2022
        - Validation: 2023
        - Test: 2024
        - Metrics: MAE, RMSE, WAPE, R2, Bias
        """
    )

    if lgb is None:
        st.error("LightGBM chưa được cài đặt. Hãy kiểm tra `requirements.txt`.")
    else:
        with st.expander("Thông tin feature set"):
            full_features = get_feature_columns(df)
            selected_features = get_selected_feature_columns(df, full_features)

            st.write(f"Full features: {len(full_features)}")
            st.write(f"Selected features: {len(selected_features)}")
            st.write("Selected feature columns:")
            st.write(selected_features)

        run_model = st.button("🚀 Run live benchmark", type="primary")

        if run_model:
            with st.spinner("Đang huấn luyện benchmark models. Quá trình này có thể mất vài phút..."):
                try:
                    result_df, meta = run_live_benchmark(df)

                    st.success("Benchmark hoàn tất.")

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Train rows", f"{meta['train_rows']:,}")
                    m2.metric("Validation rows", f"{meta['valid_rows']:,}")
                    m3.metric("Test rows", f"{meta['test_rows']:,}")

                    display_df = result_df.copy()
                    for col in ["MAE", "RMSE", "WAPE (%)", "R2", "Bias"]:
                        display_df[col] = display_df[col].map(lambda x: format_metric(x, 4))

                    st.dataframe(display_df, use_container_width=True)

                    csv = result_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "Download benchmark result CSV",
                        data=csv,
                        file_name="benchmark_results_live.csv",
                        mime="text/csv",
                    )

                    best_mae = result_df.loc[result_df["MAE"].idxmin()]
                    best_wape = result_df.loc[result_df["WAPE (%)"].idxmin()]
                    best_rmse = result_df.loc[result_df["RMSE"].idxmin()]
                    best_r2 = result_df.loc[result_df["R2"].idxmax()]

                    st.markdown(
                        f"""
                        **Diễn giải nhanh:**
                        - MAE thấp nhất: `{best_mae['Setting']} - {best_mae['Model']}`.
                        - WAPE thấp nhất: `{best_wape['Setting']} - {best_wape['Model']}`.
                        - RMSE thấp nhất: `{best_rmse['Setting']} - {best_rmse['Model']}`.
                        - R2 cao nhất: `{best_r2['Setting']} - {best_r2['Model']}`.

                        Bias gần 0 không đồng nghĩa mô hình tốt nhất nếu MAE, RMSE và WAPE vẫn cao.
                        """
                    )

                except Exception as exc:
                    st.error(f"Benchmark thất bại: {exc}")
                    st.exception(exc)
        else:
            st.info("Bấm nút để chạy benchmark trực tiếp từ dữ liệu feature hiện có.")


with tab_codebook:
    st.subheader("Codebook")

    if codebook_path:
        try:
            codebook_df = load_codebook_from_path(str(codebook_path))
            st.dataframe(codebook_df, use_container_width=True)

            missing_in_codebook = []
            if not codebook_df.empty:
                variable_col = find_codebook_variable_column(codebook_df)
                codebook_vars = set(codebook_df[variable_col].astype(str).str.strip())
                missing_in_codebook = [
                    c for c in df.columns
                    if c not in codebook_vars and not c.startswith("__")
                ]

            if missing_in_codebook:
                st.warning("Một số cột trong dataset chưa thấy trong Codebook:")
                st.write(missing_in_codebook)
            else:
                st.success("Codebook khớp với các cột chính trong dataset.")

        except Exception as exc:
            st.error(f"Không thể đọc Codebook: {exc}")
    else:
        st.warning("Không tìm thấy `docs/Codebook.csv`.")


with tab_system:
    st.subheader("System Check")

    checks = [
        ("Feature dataset exists", data_path is not None),
        ("Codebook exists", codebook_path is not None),
        ("PRCP column exists", "PRCP" in df.columns),
        ("Target stage 1 exists", "target_stage1_rain_label" in df.columns),
        ("Target one-stage exists", "target_1stage_prcp_mm" in df.columns),
        ("Target stage 2 exists", "target_stage2_prcp_log1p" in df.columns),
        ("LightGBM import OK", lgb is not None),
        ("Plotly import OK", px is not None),
    ]

    check_df = pd.DataFrame(checks, columns=["Check", "Status"])
    check_df["Status"] = check_df["Status"].map(lambda x: "OK" if x else "Missing")

    st.dataframe(check_df, use_container_width=True)

    st.markdown(
        f"""
        **Airflow URL:** `{AIRFLOW_URL}`  
        **Streamlit URL:** `{STREAMLIT_URL}`
        """
    )
