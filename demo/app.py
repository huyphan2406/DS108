import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# =========================================================
# CONFIG
# =========================================================
st.set_page_config(
    page_title="DS108 Rainfall Forecasting Demo",
    page_icon="🌧️",
    layout="wide",
)

DATA_PATH = Path("../data/features/feature_engineered_data.csv")
ARTIFACT_DIR = Path("artifacts")

STATION_NAME = {
    "48820099999": "Hà Đông",
    "48845099999": "Vinh",
    "48855099999": "Quảng Ngãi",
    "48877099999": "Nha Trang",
    "48900099999": "Cà Mau",
}

MONTH_LABELS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]


# =========================================================
# LOAD DATA
# =========================================================
@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    df["STATION"] = df["STATION"].astype(str)
    df["station_name"] = df["STATION"].map(STATION_NAME).fillna(df["STATION"])
    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df["PRCP"] = pd.to_numeric(df["PRCP"], errors="coerce")

    df = df.dropna(subset=["time", "PRCP", "PRCP_label"]).copy()
    df = df[df["PRCP"] >= 0].copy()

    df["year"] = df["time"].dt.year
    df["month"] = df["time"].dt.month
    df["month_name"] = pd.Categorical(
        df["month"].map(lambda m: MONTH_LABELS[m - 1]),
        categories=MONTH_LABELS,
        ordered=True,
    )

    df["rain_status"] = np.where(df["PRCP_label"].eq(1), "Có mưa", "Không mưa")

    if "PRCP_log1p" not in df.columns:
        df["PRCP_log1p"] = np.log1p(df["PRCP"])

    return df


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    non_features = {
        "STATION", "station_name", "time", "date", "DATE",
        "year", "month", "month_name", "rain_status",
        "PRCP", "PRCP_label", "PRCP_log1p",
        "target_1stage_prcp_mm",
        "target_stage1_rain_label",
        "target_stage2_prcp_log1p",
        "is_labeled", "PRCP_is_missing",
    }

    return [
        c for c in df.select_dtypes(include="number").columns
        if c not in non_features and df[c].nunique(dropna=True) > 1
    ]


def filter_data(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Bộ lọc dữ liệu")

    station_options = sorted(df["station_name"].unique())
    selected_stations = st.sidebar.multiselect(
        "Chọn trạm quan trắc",
        station_options,
        default=station_options,
    )

    year_min, year_max = int(df["year"].min()), int(df["year"].max())
    year_range = st.sidebar.slider(
        "Khoảng năm",
        min_value=year_min,
        max_value=year_max,
        value=(year_min, year_max),
    )

    filtered = df[
        df["station_name"].isin(selected_stations)
        & df["year"].between(year_range[0], year_range[1])
    ].copy()

    return filtered


# =========================================================
# MAIN
# =========================================================
st.title("🌧️ DS108 Rainfall Dataset Demo")
st.caption(
    "Dashboard tương tác cho bộ dữ liệu dự báo lượng mưa tại Việt Nam "
    "dựa trên dữ liệu GSOD và ERA5."
)

if not DATA_PATH.exists():
    st.error(f"Không tìm thấy file dữ liệu: {DATA_PATH}")
    st.stop()

df = load_data(DATA_PATH)
feature_cols = get_feature_cols(df)
filtered_df = filter_data(df)

if filtered_df.empty:
    st.warning("Không có dữ liệu sau khi lọc.")
    st.stop()
