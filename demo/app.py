from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import streamlit as st


# =============================================================================
# Page configuration
# =============================================================================
st.set_page_config(
    page_title="DS108 Rainfall Forecasting Dashboard",
    page_icon="🌧️",
    layout="wide",
)


# =============================================================================
# Paths
# =============================================================================
def get_project_root() -> Path:
    """
    Resolve project root for both local and Docker execution.

    Docker dashboard service uses:
        DS108_PROJECT_ROOT=/app

    Airflow service uses:
        DS108_PROJECT_ROOT=/opt/airflow
    """
    env_root = os.getenv("DS108_PROJECT_ROOT")
    if env_root:
        return Path(env_root).resolve()

    # demo/app.py -> project root is parent.parent
    return Path(__file__).resolve().parents[1]


PROJECT_ROOT = get_project_root()

FEATURE_DATA_PATH = PROJECT_ROOT / "data" / "features" / "feature_engineered_data.csv"
MODEL_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "model_evaluation"

MODEL_RESULT_CANDIDATES = [
    MODEL_OUTPUT_DIR / "table_1_validation_results_2023.csv",
    MODEL_OUTPUT_DIR / "table_2_final_test_results_2024.csv",
    MODEL_OUTPUT_DIR / "table_3_full_vs_selected_test.csv",
    MODEL_OUTPUT_DIR / "raw_validation_results_2023.csv",
    MODEL_OUTPUT_DIR / "raw_test_results_2024.csv",
    MODEL_OUTPUT_DIR / "raw_full_vs_selected_test.csv",
    MODEL_OUTPUT_DIR / "model_evaluation_summary.txt",
    PROJECT_ROOT / "outputs" / "model_results.csv",
    PROJECT_ROOT / "outputs" / "model_final_single_table" / "metrics.csv",
    PROJECT_ROOT / "reports" / "model_results.csv",
]


# =============================================================================
# Helper functions
# =============================================================================
@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def normalize_time_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    time_candidates = ["time", "date", "DATE", "Time", "datetime", "valid_time"]
    time_col = next((col for col in time_candidates if col in df.columns), None)

    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        if time_col != "time":
            df = df.rename(columns={time_col: "time"})

    return df


def get_station_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ["STATION", "station", "station_id", "station_code"]
    return next((col for col in candidates if col in df.columns), None)


def get_lat_lon_columns(df: pd.DataFrame) -> tuple[Optional[str], Optional[str]]:
    lat_candidates = ["LATITUDE", "latitude", "lat"]
    lon_candidates = ["LONGITUDE", "longitude", "lon"]

    lat_col = next((col for col in lat_candidates if col in df.columns), None)
    lon_col = next((col for col in lon_candidates if col in df.columns), None)
    return lat_col, lon_col


def get_numeric_columns(df: pd.DataFrame) -> list[str]:
    return df.select_dtypes(include="number").columns.tolist()


def safe_metric_value(value, suffix: str = "") -> str:
    if value is None or pd.isna(value):
        return "N/A"
    if isinstance(value, float):
        return f"{value:,.2f}{suffix}"
    return f"{value:,}{suffix}"


def find_existing_files(paths: list[Path]) -> list[Path]:
    return [path for path in paths if path.exists()]


def render_missing_file_message() -> None:
    st.error("Không tìm thấy file dữ liệu cuối cùng.")
    st.info(
        "Hãy chạy Airflow DAG trước để sinh file "
        "`data/features/feature_engineered_data.csv`."
    )

    with st.expander("Đường dẫn đang kiểm tra"):
        st.code(str(FEATURE_DATA_PATH))

    with st.expander("Cách chạy nhanh bằng Docker"):
        st.code(
            "docker compose down --volumes --remove-orphans\n"
            "docker compose up --build\n\n"
            "# Sau đó mở Airflow và trigger DAG:\n"
            "# http://localhost:8081\n"
            "# DAG: ds108_rainfall_pipeline",
            language="bash",
        )


def apply_sidebar_filters(df: pd.DataFrame) -> pd.DataFrame:
    filtered_df = df.copy()

    st.sidebar.header("Bộ lọc dữ liệu")

    if "time" in filtered_df.columns and filtered_df["time"].notna().any():
        min_date = filtered_df["time"].min().date()
        max_date = filtered_df["time"].max().date()

        date_range = st.sidebar.date_input(
            "Khoảng thời gian",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
            filtered_df = filtered_df[
                (filtered_df["time"].dt.date >= start_date)
                & (filtered_df["time"].dt.date <= end_date)
            ]

    station_col = get_station_column(filtered_df)
    if station_col:
        stations = sorted(filtered_df[station_col].dropna().astype(str).unique().tolist())

        default_stations = stations[:5] if len(stations) > 5 else stations
        selected_stations = st.sidebar.multiselect(
            "Trạm khí tượng",
            options=stations,
            default=default_stations,
        )

        if selected_stations:
            filtered_df = filtered_df[
                filtered_df[station_col].astype(str).isin(selected_stations)
            ]

    st.sidebar.divider()

    max_rows = st.sidebar.slider(
        "Số dòng hiển thị trong bảng",
        min_value=100,
        max_value=5000,
        value=1000,
        step=100,
    )
    st.session_state["max_display_rows"] = max_rows

    return filtered_df


def render_overview(df: pd.DataFrame) -> None:
    st.subheader("Tổng quan dữ liệu")

    station_col = get_station_column(df)

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("Số dòng", f"{df.shape[0]:,}")
    col2.metric("Số cột", f"{df.shape[1]:,}")
    col3.metric("Số trạm", f"{df[station_col].nunique():,}" if station_col else "N/A")

    if "time" in df.columns and df["time"].notna().any():
        date_min = df["time"].min().date()
        date_max = df["time"].max().date()
        col4.metric("Từ ngày", str(date_min))
        col5.metric("Đến ngày", str(date_max))
    else:
        col4.metric("Từ ngày", "N/A")
        col5.metric("Đến ngày", "N/A")

    if "PRCP" in df.columns:
        rain_ratio = (df["PRCP"] >= 0.1).mean() * 100
        avg_prcp = df["PRCP"].mean()

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Tỷ lệ ngày mưa", f"{rain_ratio:.2f}%")
        col_b.metric("PRCP trung bình", f"{avg_prcp:.4f} mm")
        col_c.metric("PRCP lớn nhất", f"{df['PRCP'].max():.4f} mm")


def render_data_tab(df: pd.DataFrame) -> None:
    st.subheader("Dữ liệu sau khi lọc")

    max_rows = st.session_state.get("max_display_rows", 1000)
    st.dataframe(df.head(max_rows), use_container_width=True)

    st.subheader("Thông tin cột")
    column_info = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(dtype) for dtype in df.dtypes],
            "missing_count": df.isna().sum().values,
            "missing_rate_percent": (df.isna().mean().values * 100).round(4),
            "unique_count": [df[col].nunique(dropna=True) for col in df.columns],
        }
    )
    st.dataframe(column_info, use_container_width=True)

    st.subheader("Thống kê mô tả")
    st.dataframe(df.describe(include="all"), use_container_width=True)


def render_eda_tab(df: pd.DataFrame) -> None:
    st.subheader("Phân tích khám phá dữ liệu")

    numeric_cols = get_numeric_columns(df)

    if not numeric_cols:
        st.info("Không có cột số để vẽ EDA.")
        return

    selected_col = st.selectbox("Chọn biến để xem phân phối", numeric_cols)

    fig_hist = px.histogram(
        df,
        x=selected_col,
        nbins=50,
        title=f"Phân phối của {selected_col}",
    )
    st.plotly_chart(fig_hist, use_container_width=True)

    if "PRCP" in df.columns:
        temp = df.copy()
        temp["rain_label"] = temp["PRCP"].apply(lambda x: "Rain" if x >= 0.1 else "No Rain")

        rain_count = (
            temp["rain_label"]
            .value_counts()
            .rename_axis("label")
            .reset_index(name="count")
        )

        fig_rain = px.bar(
            rain_count,
            x="label",
            y="count",
            title="Số ngày mưa và không mưa theo ngưỡng PRCP >= 0.1 mm",
        )
        st.plotly_chart(fig_rain, use_container_width=True)

    if "time" in df.columns and "PRCP" in df.columns:
        rain_by_time = (
            df.dropna(subset=["time"])
            .groupby("time", as_index=False)["PRCP"]
            .mean()
            .sort_values("time")
        )

        fig_line = px.line(
            rain_by_time,
            x="time",
            y="PRCP",
            title="Lượng mưa trung bình theo thời gian",
        )
        st.plotly_chart(fig_line, use_container_width=True)

        temp = df.dropna(subset=["time"]).copy()
        temp["month"] = temp["time"].dt.month

        monthly_rain = temp.groupby("month", as_index=False)["PRCP"].mean()

        fig_month = px.bar(
            monthly_rain,
            x="month",
            y="PRCP",
            title="Lượng mưa trung bình theo tháng",
        )
        st.plotly_chart(fig_month, use_container_width=True)

    st.subheader("Tương quan giữa các biến số")

    default_cols = numeric_cols[: min(10, len(numeric_cols))]
    corr_cols = st.multiselect(
        "Chọn biến để tính tương quan",
        numeric_cols,
        default=default_cols,
    )

    if len(corr_cols) >= 2:
        corr = df[corr_cols].corr(numeric_only=True)

        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            title="Correlation Heatmap",
            aspect="auto",
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    else:
        st.info("Chọn ít nhất 2 biến để vẽ heatmap tương quan.")


def render_quality_tab(df: pd.DataFrame) -> None:
    st.subheader("Kiểm tra chất lượng dữ liệu")

    missing_rate = df.isna().mean().sort_values(ascending=False) * 100
    missing_df = missing_rate.reset_index()
    missing_df.columns = ["column", "missing_rate_percent"]

    fig_missing = px.bar(
        missing_df,
        x="column",
        y="missing_rate_percent",
        title="Tỷ lệ missing theo cột",
    )
    st.plotly_chart(fig_missing, use_container_width=True)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Dòng trùng lặp", int(df.duplicated().sum()))
    col2.metric("Tổng giá trị thiếu", int(df.isna().sum().sum()))

    if "PRCP" in df.columns:
        col3.metric("PRCP âm", int((df["PRCP"] < 0).sum()))
        col4.metric("PRCP thiếu", int(df["PRCP"].isna().sum()))
    else:
        col3.metric("PRCP âm", "N/A")
        col4.metric("PRCP thiếu", "N/A")

    st.subheader("Top cột thiếu nhiều nhất")
    st.dataframe(missing_df.head(30), use_container_width=True)


def render_map_tab(df: pd.DataFrame) -> None:
    st.subheader("Bản đồ trạm khí tượng")

    lat_col, lon_col = get_lat_lon_columns(df)
    station_col = get_station_column(df)

    if not lat_col or not lon_col:
        st.info("Không có đủ cột LATITUDE và LONGITUDE để vẽ bản đồ.")
        return

    keep_cols = [lat_col, lon_col]
    if station_col:
        keep_cols.append(station_col)

    map_df = df[keep_cols].dropna().drop_duplicates().copy()
    map_df = map_df.rename(columns={lat_col: "lat", lon_col: "lon"})

    if map_df.empty:
        st.info("Không có dữ liệu tọa độ hợp lệ để hiển thị.")
        return

    st.map(map_df[["lat", "lon"]])

    st.subheader("Danh sách tọa độ trạm")
    st.dataframe(map_df, use_container_width=True)


def render_model_tab() -> None:
    st.subheader("Kết quả mô hình")

    existing_files = find_existing_files(MODEL_RESULT_CANDIDATES)

    if not existing_files:
        st.info("Chưa tìm thấy file kết quả mô hình.")
        st.write("Các vị trí đang kiểm tra:")

        for path in MODEL_RESULT_CANDIDATES:
            st.code(str(path))

        st.info("Hãy chạy `_08_model.py` hoặc Airflow DAG để sinh kết quả vào `outputs/model_evaluation/`.")
        return

    selected_file = st.selectbox(
        "Chọn file kết quả",
        options=existing_files,
        format_func=lambda p: str(p.relative_to(PROJECT_ROOT)) if p.is_relative_to(PROJECT_ROOT) else str(p),
    )

    st.write(f"Đang đọc: `{selected_file}`")

    if selected_file.suffix.lower() == ".txt":
        st.text(selected_file.read_text(encoding="utf-8"))
    else:
        result_df = pd.read_csv(selected_file)
        st.dataframe(result_df, use_container_width=True)

        numeric_cols = get_numeric_columns(result_df)
        if numeric_cols:
            metric_col = st.selectbox("Chọn metric để vẽ", numeric_cols)

            x_col_candidates = [
                col for col in result_df.columns
                if col.lower() in {"model", "scenario", "feature set", "feature_set"}
            ]
            x_col = x_col_candidates[0] if x_col_candidates else result_df.columns[0]

            fig = px.bar(
                result_df,
                x=x_col,
                y=metric_col,
                color=x_col if result_df[x_col].nunique() <= 20 else None,
                title=f"{metric_col} theo {x_col}",
            )
            st.plotly_chart(fig, use_container_width=True)


def render_about_tab() -> None:
    st.subheader("Thông tin project")

    st.markdown(
        """
**DS108 - Rainfall Forecasting Data Pipeline**

Dashboard này dùng để kiểm tra nhanh dữ liệu cuối cùng, EDA, chất lượng dữ liệu,
bản đồ trạm khí tượng và kết quả mô hình.

Luồng chính của project:

```text
Raw Data
  ↓
Data Cleaning
  ↓
Data Integration
  ↓
Feature Engineering
  ↓
Model Evaluation
  ↓
Dashboard
```

Các thành phần chính:

- `Airflow`: tự động hóa pipeline.
- `Docker`: đóng gói môi trường chạy.
- `Streamlit`: hiển thị dashboard.
- `LightGBM`: mô hình dự báo lượng mưa.
"""
    )

    st.subheader("Đường dẫn hệ thống")
    st.code(f"PROJECT_ROOT = {PROJECT_ROOT}")
    st.code(f"FEATURE_DATA_PATH = {FEATURE_DATA_PATH}")
    st.code(f"MODEL_OUTPUT_DIR = {MODEL_OUTPUT_DIR}")


# =============================================================================
# Main app
# =============================================================================
def main() -> None:
    st.title("🌧️ DS108 - Rainfall Forecasting Dashboard")

    st.markdown(
        """
Dashboard hỗ trợ xem dữ liệu sau pipeline, phân tích EDA, kiểm tra chất lượng dữ liệu,
bản đồ trạm khí tượng và kết quả mô hình dự báo lượng mưa theo ngày tại Việt Nam.
"""
    )

    if not FEATURE_DATA_PATH.exists():
        render_missing_file_message()
        return

    df = load_csv(str(FEATURE_DATA_PATH))
    df = normalize_time_column(df)

    filtered_df = apply_sidebar_filters(df)

    render_overview(filtered_df)

    tab_data, tab_eda, tab_quality, tab_map, tab_model, tab_about = st.tabs(
        [
            "Dữ liệu",
            "EDA",
            "Chất lượng dữ liệu",
            "Bản đồ trạm",
            "Kết quả mô hình",
            "Thông tin",
        ]
    )

    with tab_data:
        render_data_tab(filtered_df)

    with tab_eda:
        render_eda_tab(filtered_df)

    with tab_quality:
        render_quality_tab(filtered_df)

    with tab_map:
        render_map_tab(filtered_df)

    with tab_model:
        render_model_tab()

    with tab_about:
        render_about_tab()


if __name__ == "__main__":
    main()
