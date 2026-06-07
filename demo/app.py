from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="DS108 Rainfall Forecasting Dashboard",
    layout="wide",
)

st.title("DS108 - Rainfall Forecasting Dashboard")

st.markdown(
    """
Dashboard tương tác đọc kết quả từ pipeline để hỗ trợ xem dữ liệu, EDA,
chất lượng dữ liệu và kết quả mô hình.
"""
)

DATA_PATH = Path("data/features/feature_engineered_data.csv")


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    time_candidates = ["time", "date", "DATE", "Time", "datetime"]
    time_col = next((col for col in time_candidates if col in df.columns), None)

    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        if time_col != "time":
            df = df.rename(columns={time_col: "time"})

    return df


if not DATA_PATH.exists():
    st.error("Không tìm thấy file data/features/feature_engineered_data.csv")
    st.info("Hãy chạy Airflow DAG trước để sinh dữ liệu đầu ra.")
    st.stop()


df = load_data(DATA_PATH)
filtered_df = df.copy()

st.sidebar.header("Bộ lọc")

if "time" in filtered_df.columns:
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


station_col = "STATION" if "STATION" in filtered_df.columns else None

if station_col:
    stations = sorted(filtered_df[station_col].dropna().unique().tolist())
    selected_stations = st.sidebar.multiselect(
        "Trạm khí tượng",
        options=stations,
        default=stations[:5] if len(stations) > 5 else stations,
    )

    if selected_stations:
        filtered_df = filtered_df[filtered_df[station_col].isin(selected_stations)]


st.subheader("Tổng quan dữ liệu")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Số dòng", f"{filtered_df.shape[0]:,}")
col2.metric("Số cột", f"{filtered_df.shape[1]:,}")
col3.metric("Số trạm", filtered_df[station_col].nunique() if station_col else "N/A")

if "PRCP" in filtered_df.columns:
    rain_ratio = (filtered_df["PRCP"] >= 0.1).mean() * 100
    col4.metric("Tỷ lệ ngày mưa", f"{rain_ratio:.2f}%")
else:
    col4.metric("Tỷ lệ ngày mưa", "N/A")


tab_data, tab_eda, tab_quality, tab_map, tab_model = st.tabs(
    ["Dữ liệu", "EDA", "Chất lượng dữ liệu", "Bản đồ trạm", "Kết quả mô hình"]
)


with tab_data:
    st.subheader("Dữ liệu sau khi lọc")
    st.dataframe(filtered_df.head(1000), use_container_width=True)

    st.subheader("Thống kê mô tả")
    st.dataframe(filtered_df.describe(include="all"), use_container_width=True)


with tab_eda:
    st.subheader("Phân tích khám phá dữ liệu")

    numeric_cols = filtered_df.select_dtypes(include="number").columns.tolist()

    if numeric_cols:
        selected_col = st.selectbox("Chọn biến", numeric_cols)

        fig_hist = px.histogram(
            filtered_df,
            x=selected_col,
            nbins=50,
            title=f"Phân phối của {selected_col}",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    if "PRCP" in filtered_df.columns:
        temp = filtered_df.copy()
        temp["rain_label"] = temp["PRCP"].apply(
            lambda x: "Rain" if x >= 0.1 else "No Rain"
        )

        rain_count = temp["rain_label"].value_counts().reset_index()
        rain_count.columns = ["label", "count"]

        fig_rain = px.bar(
            rain_count,
            x="label",
            y="count",
            title="Số ngày mưa và không mưa",
        )
        st.plotly_chart(fig_rain, use_container_width=True)

    if "time" in filtered_df.columns and "PRCP" in filtered_df.columns:
        rain_by_time = (
            filtered_df.groupby("time", as_index=False)["PRCP"]
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

        temp = filtered_df.copy()
        temp["month"] = temp["time"].dt.month

        monthly_rain = temp.groupby("month", as_index=False)["PRCP"].mean()

        fig_month = px.bar(
            monthly_rain,
            x="month",
            y="PRCP",
            title="Lượng mưa trung bình theo tháng",
        )
        st.plotly_chart(fig_month, use_container_width=True)

    if len(numeric_cols) >= 2:
        default_cols = numeric_cols[: min(10, len(numeric_cols))]
        corr_cols = st.multiselect(
            "Chọn biến để tính tương quan",
            numeric_cols,
            default=default_cols,
        )

        if len(corr_cols) >= 2:
            corr = filtered_df[corr_cols].corr()

            fig_corr = px.imshow(
                corr,
                text_auto=".2f",
                title="Correlation Heatmap",
            )
            st.plotly_chart(fig_corr, use_container_width=True)


with tab_quality:
    st.subheader("Kiểm tra chất lượng dữ liệu")

    missing_rate = filtered_df.isna().mean().sort_values(ascending=False) * 100
    missing_df = missing_rate.reset_index()
    missing_df.columns = ["column", "missing_rate_percent"]

    fig_missing = px.bar(
        missing_df,
        x="column",
        y="missing_rate_percent",
        title="Tỷ lệ missing theo cột",
    )
    st.plotly_chart(fig_missing, use_container_width=True)

    col_a, col_b, col_c = st.columns(3)

    col_a.metric("Dòng trùng lặp", int(filtered_df.duplicated().sum()))

    if "PRCP" in filtered_df.columns:
        col_b.metric("PRCP âm", int((filtered_df["PRCP"] < 0).sum()))
        col_c.metric("PRCP thiếu", int(filtered_df["PRCP"].isna().sum()))
    else:
        col_b.metric("PRCP âm", "N/A")
        col_c.metric("PRCP thiếu", "N/A")


with tab_map:
    st.subheader("Bản đồ trạm")

    lat_candidates = ["LATITUDE", "latitude", "lat"]
    lon_candidates = ["LONGITUDE", "longitude", "lon"]

    lat_col = next((col for col in lat_candidates if col in filtered_df.columns), None)
    lon_col = next((col for col in lon_candidates if col in filtered_df.columns), None)

    if lat_col and lon_col:
        map_df = (
            filtered_df[[lat_col, lon_col]]
            .dropna()
            .drop_duplicates()
            .rename(columns={lat_col: "lat", lon_col: "lon"})
        )

        st.map(map_df)
    else:
        st.info("Không có đủ cột LATITUDE và LONGITUDE.")


with tab_model:
    st.subheader("Kết quả mô hình")

    result_files = [
        Path("outputs/model_results.csv"),
        Path("outputs/model_final_single_table/metrics.csv"),
        Path("reports/model_results.csv"),
    ]

    result_file = next((path for path in result_files if path.exists()), None)

    if result_file:
        result_df = pd.read_csv(result_file)
        st.write(f"Đang đọc: `{result_file}`")
        st.dataframe(result_df, use_container_width=True)
    else:
        st.info("Chưa tìm thấy file kết quả mô hình trong outputs/ hoặc reports/.")