import pandas as pd
import numpy as np
from pathlib import Path

# =====================================================================
# 1️⃣ THIẾT LẬP ĐƯỜNG DẪN TỔNG
# =====================================================================
BASE_DIR = Path.cwd().parent.parent

# Thư mục chứa data Silver (Đầu vào)
SILVER_DIR = BASE_DIR / "data" / "silver"
SILVER_INPUT_FILE = SILVER_DIR / "master_data_silver.csv"

# Thư mục lưu data Gold (Đầu ra)
GOLD_DIR = BASE_DIR / "data" / "gold"
GOLD_DIR.mkdir(parents=True, exist_ok=True)
GOLD_FINAL_OUTPUT = GOLD_DIR / "master_data_gold.csv"

# =====================================================================
# 2️⃣ ĐỌC DỮ LIỆU LỚP SILVER
# =====================================================================
print("🚀 BẮT ĐẦU CHUYỂN ĐỔI TỪ SILVER SANG GOLD...")

if not SILVER_INPUT_FILE.exists():
    raise FileNotFoundError(f"❌ Không tìm thấy file Silver tại: {SILVER_INPUT_FILE}")

print(f"📂 Đang nạp dữ liệu từ: {SILVER_INPUT_FILE.name}")
df_master = pd.read_csv(SILVER_INPUT_FILE)

# Đảm bảo cột date là kiểu datetime để phòng hờ, tuy nhiên ta sẽ dùng nó làm mốc
df_master['date'] = pd.to_datetime(df_master['date'], errors='coerce')

# BƯỚC QUAN TRỌNG: Sắp xếp lại dữ liệu theo Trạm và Thời gian
print("🗂️ Đang sắp xếp dữ liệu theo Tọa độ và Thời gian...")
df_master = df_master.sort_values(by=['latitude', 'longitude', 'date']).reset_index(drop=True)

print(f"✅ Đã nạp và sắp xếp xong! (Tổng: {len(df_master)} dòng, {len(df_master.columns)} cột)\n")

# =====================================================================
# 3️⃣ FEATURE ENGINEERING (TRÍCH XUẤT ĐẶC TRƯNG)
# =====================================================================
print("🧠 BẮT ĐẦU TẠO CÁC ĐẶC TRƯNG TOÁN HỌC (FEATURE ENGINEERING)...")

# Khởi tạo Groupby TRƯỚC để dùng chung (Tránh lấy râu ông nọ cắm cằm bà kia)
grouped = df_master.groupby(['latitude', 'longitude'])

# --- 3.1. XÁC ĐỊNH BIẾN MỤC TIÊU (TARGET) ---
print("  🎯 Đang tạo nhãn Mục tiêu (Target) cho ngày mai...")
target_shift = grouped['precipitation_mm'].shift(-1)
# Hồi quy: Dự báo chính xác số mm
df_master['target_precip_tomorrow'] = target_shift
# Phân loại: Có mưa hay không? (> 0.1 mm coi như có mưa)
df_master['target_is_rain_tomorrow'] = (target_shift > 0.1).astype(float)

# --- 3.2. MÃ HÓA CHU KỲ (CYCLICAL ENCODING) ---
print("  🔄 Đang bẻ cong thời gian bằng Lượng giác (Sin/Cos)...")
df_master['day_sin'] = np.sin(2 * np.pi * (df_master['day_of_year'] - 1) / 365.25)
df_master['day_cos'] = np.cos(2 * np.pi * (df_master['day_of_year'] - 1) / 365.25)
df_master['month_sin'] = np.sin(2 * np.pi * (df_master['month'] - 1) / 12)
df_master['month_cos'] = np.cos(2 * np.pi * (df_master['month'] - 1) / 12)
cols_to_round = ['day_sin', 'day_cos', 'month_sin', 'month_cos']
df_master[cols_to_round] = df_master[cols_to_round].round(10)
df_master[cols_to_round] = df_master[cols_to_round].replace(-0.0, 0.0)

# --- 3.3. TẠO BIẾN TRỄ (LAG FEATURES - TRÍ NHỚ QUÁ KHỨ) ---
print("  ⏪ Đang tạo biến Lag (Dữ liệu của 1-2 ngày trước)...")
lag_cols = ['temp_c', 'precipitation_mm', 'relative_humidity_2m', 'sea_level_pressure_hpa', 'cloud_fraction']
for col in lag_cols:
    df_master[f'{col}_lag_1'] = grouped[col].shift(1)
    df_master[f'{col}_lag_2'] = grouped[col].shift(2)

# --- 3.4. CỬA SỔ TRƯỢT (ROLLING WINDOWS - XU HƯỚNG TÍCH LŨY) ---
print("  🌊 Đang tạo biến Cửa sổ trượt (Tổng/Trung bình 3-7 ngày)...")
df_master['precip_roll_sum_3d'] = grouped['precipitation_mm'].transform(lambda x: x.rolling(3).sum())
df_master['precip_roll_sum_7d'] = grouped['precipitation_mm'].transform(lambda x: x.rolling(7).sum())
df_master['temp_roll_mean_3d'] = grouped['temp_c'].transform(lambda x: x.rolling(3).mean())

# --- 3.5. ĐẶC TRƯNG KHÍ TƯỢNG (DOMAIN FEATURES) ---
print("  🌪️ Đang tạo các chỉ số Khí tượng học chuyên sâu...")
# Biên độ nhiệt trong ngày
df_master['temp_range'] = df_master['max_temp_c'] - df_master['min_temp_c']
# Biến động áp suất (Áp suất hụt sâu là điềm báo mưa bão)
df_master['pressure_change'] = df_master['sea_level_pressure_hpa'] - df_master['sea_level_pressure_hpa_lag_1']

# =====================================================================
# 4️⃣ DỌN DẸP HẬU KỲ VÀ XUẤT FILE
# =====================================================================
print("\n🧹 BẮT ĐẦU DỌN DẸP LẦN CUỐI...")
print(f"  📊 Số dòng trước khi dọn dẹp: {len(df_master)}")

# Việc tạo Lag/Rolling/Target sẽ sinh ra NaN ở đầu và cuối mỗi trạm. Bắt buộc phải drop.
df_master = df_master.dropna().reset_index(drop=True)

# Sau khi xóa NaN, ép lại kiểu nguyên (int) cho target phân loại cho đẹp mắt
df_master['target_is_rain_tomorrow'] = df_master['target_is_rain_tomorrow'].astype(int)

# Đưa cột date về lại chuỗi DD-MM-YYYY để xuất file nhìn cho thân thiện
df_master['date'] = df_master['date'].dt.strftime('%d-%m-%Y')

print(f"  ✅ Số dòng sau khi dọn dẹp: {len(df_master)}")

# Xuất file
print("💾 Đang ghi dữ liệu ra file Gold...")
df_master.to_csv(GOLD_FINAL_OUTPUT, index=False)

print("\n" + "=" * 60)
print(f"✨ HOÀN TẤT TẠO LỚP GOLD! Dữ liệu đã sẵn sàng cho Machine Learning.")
print(f"📍 Vị trí file cuối cùng: {GOLD_FINAL_OUTPUT}")
print(f"📊 Kích thước ma trận tính toán: {len(df_master)} dòng x {len(df_master.columns)} cột")
print("=" * 60)