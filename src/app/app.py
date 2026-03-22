import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from pathlib import Path
import sys

# =====================================================================
# 1. CẤU HÌNH HỆ THỐNG
# =====================================================================
st.set_page_config(
    page_title="Weather AI - Trạm sân bay Phú Bài (Huế)",
    page_icon="⛈️",
    layout="wide"
)

# Cấu hình thông tin tác giả
AUTHOR_NAME = "Phan Gia Quốc Huy"
UNIVERSITY = "Trường ĐH Công nghệ Thông tin - ĐHQG TP.HCM (UIT)"

BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

MODEL_PATH = BASE_DIR / "models" / "weather_model_xgb.pkl"
GOLD_PATH = BASE_DIR / "data" / "gold" / "master_data_gold.csv"


# =====================================================================
# 2. TẢI TÀI NGUYÊN
# =====================================================================
@st.cache_resource
def load_assets():
    try:
        if not MODEL_PATH.exists() or not GOLD_PATH.exists():
            return None, None, f"Thiếu file tại: {MODEL_PATH} hoặc {GOLD_PATH}"
        m = joblib.load(MODEL_PATH)
        df = pd.read_csv(GOLD_PATH)
        # Đọc đúng định dạng ngày Việt Nam
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
        return m, df, None
    except Exception as e:
        return None, None, str(e)


model, data_source, error_msg = load_assets()

if error_msg:
    st.error(f"❌ Lỗi hệ thống: {error_msg}")
    st.stop()

# =====================================================================
# 3. GIAO DIỆN CHÍNH (DASHBOARD)
# =====================================================================
st.title("⛈️ Dự báo mưa (dựa trên mô hình học máy XGBoost)")
latest_dt = data_source['date'].max().strftime('%d/%m/%Y')
st.markdown(f"**Vị trí:** Trạm sân bay Phú Bài, Huế | **Dữ liệu cập nhật đến:** {latest_dt}")
st.write("---")

# Sidebar nhập liệu
st.sidebar.header("📍 Nhập thông số hôm nay")
t = st.sidebar.slider("🌡️ Nhiệt độ (°C)", 10.0, 45.0, 27.0, 0.5)
h = st.sidebar.slider("💧 Độ ẩm (%)", 10, 100, 85)
p_chg = st.sidebar.slider("📉 Biến động áp suất (hPa)", -5.0, 5.0, -1.0, 0.1)
clouds = st.sidebar.slider("☁️ Độ che phủ mây", 0.0, 1.0, 0.6, 0.05)
wind = st.sidebar.slider("💨 Tốc độ gió (m/s)", 0.0, 50.0, 5.0, 0.5)
vis = st.sidebar.slider("👁️ Tầm nhìn xa (km)", 1.0, 30.0, 10.0, 1.0)
btn_predict = st.sidebar.button("🚀 DỰ BÁO NGAY", use_container_width=True)

col_res, col_chart = st.columns([1, 2.5])

with col_res:
    st.subheader("🔮 Kết quả AI")
    if btn_predict:
        input_df = data_source.tail(1).copy()
        input_df['temp_c'] = t
        input_df['relative_humidity_2m'] = h
        input_df['pressure_change'] = p_chg
        input_df['cloud_fraction'] = clouds
        input_df['wind_speed_ms'] = wind
        input_df['visibility_km'] = vis

        X = input_df.drop(columns=['date', 'year', 'target_precip_tomorrow', 'target_is_rain_tomorrow'])
        rain = np.clip(model.predict(X)[0], 0, None)

        st.metric(label="Lượng mưa dự kiến", value=f"{rain:.2f} mm")
        if rain < 15:
            st.info("Mưa nhỏ/vừa 🌦️")
        else:
            st.error("Mưa rất to! ⛈️")
        rain_val = float(rain[0]) if isinstance(rain, (np.ndarray, list)) else float(rain)
        st.progress(min(rain_val / 100.0, 1.0))
    else:
        st.info("👈 Hãy chỉnh thông số và nhấn nút.")

with col_chart:
    st.subheader("📈 Xu hướng lượng mưa trong 30 ngày qua")
    df_recent = data_source.tail(30)

    # CHỈNH SỬA TẠI ĐÂY: Thay px.bar bằng px.line
    fig = px.line(df_recent,
                  x='date',
                  y='precipitation_mm',
                  title=None,
                  markers=True,  # Thêm các điểm chấm trên đường kẻ
                  labels={'precipitation_mm': 'Lượng mưa (mm)', 'date': 'Ngày'})

    # Tùy chỉnh màu sắc và đường nét cho đẹp
    fig.update_traces(line_color='#007bff', line_width=3)
    fig.update_layout(height=400, margin=dict(l=0, r=0, t=20, b=0))

    st.plotly_chart(fig, use_container_width=True)
# =====================================================================
# 4. FOOTER (TÁC GIẢ & TRƯỜNG)
# =====================================================================
st.write("---")
st.markdown(f"""
    <div style="text-align: center; color: gray; padding: 20px;">
        <p>© 2026 | Thực hiện bởi: <b>{AUTHOR_NAME}</b></p>
        <p>{UNIVERSITY}</p>
    </div>
    """, unsafe_allow_html=True)