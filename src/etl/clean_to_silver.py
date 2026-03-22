import pandas as pd
import numpy as np
from pathlib import Path
import yaml

# =====================================================================
# 1️⃣ THIẾT LẬP ĐƯỜNG DẪN TỔNG
# =====================================================================
BASE_DIR = Path.cwd().parent.parent
YAML_PATH = BASE_DIR / "configs" / "airports.yaml"

# Thư mục chứa data Bronze
GSOD_DIR = BASE_DIR / "data" / "bronze" / "batch" / "gsod"
NASA_DIR = BASE_DIR / "data" / "bronze" / "batch" / "nasapower"

# Thư mục lưu data Silver
SILVER_DIR = BASE_DIR / "data" / "silver"
SILVER_DIR.mkdir(parents=True, exist_ok=True)

# File cuối cùng
SILVER_FINAL_OUTPUT = SILVER_DIR / "master_data_silver.csv"

# =====================================================================
# 2️⃣ ĐỌC VÀ GỘP DỮ LIỆU TỪ LỚP BRONZE VÀO BỘ NHỚ (RAM)
# =====================================================================
print("🚀 BẮT ĐẦU ĐỌC DỮ LIỆU GSOD (BATCH)...")
with open(YAML_PATH, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)
airports = data["airports"]

# --- Đọc GSOD ---
gsod_df_list = []
for folder_name, meta in airports.items():
    airport_path = GSOD_DIR / folder_name
    if not airport_path.exists():
        print(f"  ⚠️ Không thấy thư mục: {airport_path.name}")
        continue

    csv_files = sorted(airport_path.glob("*.csv"))
    print(f"  Sân bay: {folder_name} — tìm thấy {len(csv_files)} file CSV")

    for file in csv_files:
        gsod_df_list.append(pd.read_csv(file))

if gsod_df_list:
    # Gán thẳng vào df_gsod, không xuất ra file
    df_gsod = pd.concat(gsod_df_list, ignore_index=True)
    print(f"✅ Xong việc nạp GSOD vào RAM! (Tổng: {len(df_gsod)} dòng)\n")
else:
    df_gsod = pd.DataFrame()

# --- Đọc NASA ---
print("🚀 BẮT ĐẦU ĐỌC DỮ LIỆU NASA POWER (REALTIME)...")
nasa_df_list = []
for code, meta in airports.items():
    file_path = NASA_DIR / f"{code}_nasa_daily.csv"
    if not file_path.exists():
        print(f"  ⚠️ Không thấy file: {file_path.name}")
        continue

    print(f"  Sân bay: {code} — nạp file {file_path.name}")
    nasa_df_list.append(pd.read_csv(file_path))

if nasa_df_list:
    # Gán thẳng vào df_nasa, không xuất ra file
    df_nasa = pd.concat(nasa_df_list, ignore_index=True)
    print(f"✅ Xong việc nạp NASA vào RAM! (Tổng: {len(df_nasa)} dòng)\n")
else:
    df_nasa = pd.DataFrame()

print("🎉 HOÀN TẤT NẠP CẢ 2 NGUỒN VÀO BỘ NHỚ!")

# =====================================================================
# 3️⃣ GỘP CHUNG GSOD VÀ NASA (MERGE TRỰC TIẾP TỪ RAM)
# =====================================================================
print("\n⏳ Đang tiến hành chuẩn hóa để gộp...")

# Đưa ngày về chuẩn
df_gsod['DATE'] = pd.to_datetime(df_gsod['DATE'], errors='coerce')
df_nasa['date'] = pd.to_datetime(df_nasa['date'], errors='coerce')

# Làm tròn tọa độ để JOIN chính xác
df_gsod['join_lat'] = df_gsod['LATITUDE'].round(2)
df_gsod['join_lon'] = df_gsod['LONGITUDE'].round(2)

df_nasa['join_lat'] = df_nasa['latitude'].round(2)
df_nasa['join_lon'] = df_nasa['longitude'].round(2)

# Cắt gọn bảng NASA
nasa_cols_to_keep = [
    'date', 'join_lat', 'join_lon',
    'cloudFraction', 'windDirection10m', 'shortwaveDownwardIrradiance',
    'longwaveDownwardIrradiance', 'earthSkinTemperature', 'surfaceSoilWetness',
    'rootZoneSoilWetness', 'relativeHumidity2m', 'specificHumidity2m'
]
df_nasa_subset = df_nasa[nasa_cols_to_keep]

print("🚀 Đang gộp dữ liệu GSOD và NASA dựa trên Vĩ độ & Kinh độ...")
df_master = pd.merge(
    df_gsod,
    df_nasa_subset,
    left_on=['DATE', 'join_lat', 'join_lon'],
    right_on=['date', 'join_lat', 'join_lon'],
    how='left'
)

# Dọn dẹp khóa trung gian
df_master.drop(columns=['date', 'join_lat', 'join_lon'], inplace=True)
df_master.sort_values(by=['STATION', 'DATE'], inplace=True)
print(f"🎉 HOÀN TẤT MERGE! Tổng số cột hiện tại: {len(df_master.columns)} cột")

# =====================================================================
# 4️⃣ LÀM SẠCH VÀ CHUẨN HÓA DỮ LIỆU
# =====================================================================
print("\n🧹 BẮT ĐẦU LÀM SẠCH VÀ CHUẨN HÓA...")

# --- 4.1 Xóa cột không cần thiết ---
attribute_cols = [col for col in df_master.columns if "_ATTRIBUTES" in col]
cols_to_drop = attribute_cols + ['STATION', 'SNDP', 'GUST', 'NAME', 'ELEVATION']
df_master.drop(columns=cols_to_drop, inplace=True, errors='ignore')
print(f"🧹 Đã xóa thành công {len(cols_to_drop)} cột rác.")

# --- 4.2 Phẫu thuật FRSHTT ---
if 'FRSHTT' in df_master.columns:
    df_master['FRSHTT'] = df_master['FRSHTT'].astype(str).str.replace('.0', '', regex=False).str.zfill(6)
    df_master['is_Fog'] = df_master['FRSHTT'].str[0].astype(int)
    df_master['is_Thunder'] = df_master['FRSHTT'].str[4].astype(int)
    df_master.drop(columns=['FRSHTT'], inplace=True, errors='ignore')
    print("✅ Đã chốt hạ 2 biến bổ trợ: is_Fog và is_Thunder.")

# =====================================================================
# 5️⃣ XỬ LÝ OUTLIER, MISSING DATA VÀ ĐỒNG NHẤT ĐƠN VỊ
# =====================================================================

# --- 5.1 Quét con số ma (999.9, 9999.9) ---
trash_map = {
    9999.9: ['DEWP', 'SLP', 'WDSP', 'MXSPD'],
    999.9: ['STP', 'VISIB', 'TEMP', 'MAX', 'MIN', 'WDSP', 'MXSPD'],
    99.99: ['PRCP']  # Bổ sung cloudFraction để diệt số ma của NASA
}

print("🧹 Đang quét sạch các con số 'ma' (999.9, 9999.9, 99.99)...")
# Xử lý outlier: Ép TẤT CẢ mã lỗi về NaN để thanh lọc triệt để
for trash_val, cols in trash_map.items():
    for col in cols:
        if col in df_master.columns:
            df_master[col] = df_master[col].replace(trash_val, np.nan)

# STP tự lọc những giá trị trên 1000 thành trừ đi 1000 để tiết kiệt bộ nhớ
# STP - Mean station pressure for the day in millibars to tenths. Missing = 9999.9
if 'STP' in df_master.columns:
    # Khôi phục số 1000 bị ẩn đi (Ví dụ 22.3 hPa -> 1022.3 hPa)
    df_master.loc[df_master['STP'] < 500, 'STP'] += 1000

# --- 5.2 Đồng nhất hệ Metric ---
print("📏 Đang đồng nhất đơn vị sang hệ Metric...")
# Nhiệt độ (F -> C)
for col in ['TEMP', 'DEWP', 'MAX', 'MIN']:
    if col in df_master.columns:
        df_master[col] = (df_master[col] - 32) * 5 / 9

# Lượng mưa (Inches -> mm)
if 'PRCP' in df_master.columns:
    df_master['PRCP'] = df_master['PRCP'] * 25.4

# Tốc độ gió (Knots -> m/s)
for col in ['WDSP', 'MXSPD']:
    if col in df_master.columns:
        df_master[col] = df_master[col] * 0.51444

# Tầm nhìn xa (Miles -> km)
if 'VISIB' in df_master.columns:
    df_master['VISIB'] = df_master['VISIB'] * 1.60934

# --- 5.3 Tách thời gian chuẩn Data Science ---
print("📅 Đang tách thời gian thành year, month, day_of_year...")
df_master['DATE'] = pd.to_datetime(df_master['DATE'])

# Lấy Năm, Tháng và Ngày thứ mấy trong năm (1-365)
df_master['year'] = df_master['DATE'].dt.year
df_master['month'] = df_master['DATE'].dt.month
df_master['day_of_year'] = df_master['DATE'].dt.dayofyear

# =====================================================================
# 5.4 ĐỔI TÊN VÀ SẮP XẾP CỘT THEO NHÓM (RENAME & REORDER)
# =====================================================================
print("🏷️ Đang đổi tên và sắp xếp lại các cột theo nhóm logic...")

# 1. Từ điển đổi tên chuẩn snake_case
rename_dict = {
    'DATE': 'date', 'LATITUDE': 'latitude', 'LONGITUDE': 'longitude',
    'PRCP': 'precipitation_mm',
    'TEMP': 'temp_c', 'MAX': 'max_temp_c', 'MIN': 'min_temp_c', 'DEWP': 'dew_point_c',
    'SLP': 'sea_level_pressure_hpa', 'STP': 'station_pressure_hpa',
    'VISIB': 'visibility_km', 'WDSP': 'wind_speed_ms', 'MXSPD': 'max_wind_speed_ms',
    'cloudFraction': 'cloud_fraction', 'earthSkinTemperature': 'earth_skin_temp_c',
    'relativeHumidity2m': 'relative_humidity_2m', 'specificHumidity2m': 'specific_humidity_2m',
    'windDirection10m': 'wind_direction_10m', 'shortwaveDownwardIrradiance': 'shortwave_radiation',
    'longwaveDownwardIrradiance': 'longwave_radiation', 'surfaceSoilWetness': 'surface_soil_wetness',
    'rootZoneSoilWetness': 'root_zone_soil_wetness', 'is_Fog': 'is_fog', 'is_Thunder': 'is_thunder'
}

df_master.rename(columns=rename_dict, inplace=True)

# 2. XÓA BỎ DỮ LIỆU THIẾU MỤC TIÊU (TARGET PURGE)
# Đây là bước quyết định để AI không bị ảo giác
print("🗑️ Đang loại bỏ các dòng bị hỏng cảm biến lượng mưa...")
df_master.dropna(subset=['precipitation_mm'], inplace=True)

# 3. Danh sách thứ tự cột mới (Nhóm lại cho khoa học)
ordered_columns = [
    'date', 'year', 'month', 'day_of_year', 'latitude', 'longitude',
    'precipitation_mm',
    'temp_c', 'max_temp_c', 'min_temp_c', 'earth_skin_temp_c',
    'dew_point_c', 'relative_humidity_2m', 'specific_humidity_2m',
    'sea_level_pressure_hpa', 'station_pressure_hpa',
    'wind_speed_ms', 'max_wind_speed_ms', 'wind_direction_10m',
    'cloud_fraction', 'visibility_km',
    'shortwave_radiation', 'longwave_radiation',
    'surface_soil_wetness', 'root_zone_soil_wetness',
    'is_fog', 'is_thunder'
]

final_columns = [col for col in ordered_columns if col in df_master.columns]
df_master = df_master[final_columns]
print("✅ Đã hoàn tất việc cấu trúc lại bảng dữ liệu!")

# =====================================================================
# 5.5 LÀM TRÒN SỐ (ROUNDING) TRÁNH SAI SỐ ẢO
# =====================================================================
print("✂️ Đang làm tròn các con số thập phân (bảo vệ tọa độ)...")

float_cols = df_master.select_dtypes(include=['float64', 'float32']).columns.tolist()
cols_to_exclude = ['latitude', 'longitude']
cols_to_round = [col for col in float_cols if col not in cols_to_exclude]

df_master[cols_to_round] = df_master[cols_to_round].round(2)
print("✅ Đã làm tròn xong! Dữ liệu gọn gàng và không mất đi tọa độ gốc.")

# =====================================================================
# 6️⃣ XUẤT FILE CUỐI CÙNG
# =====================================================================
df_master.to_csv(SILVER_FINAL_OUTPUT, index=False)

print("\n" + "=" * 60)
print(f"✨ HOÀN TẤT TOÀN BỘ PIPELINE! Dữ liệu đã sạch và đạt chuẩn Gold.")
print(f"📍 Vị trí file cuối cùng: {SILVER_FINAL_OUTPUT}")
print(f"📊 Số lượng cột hiện tại: {len(df_master.columns)}")
print("=" * 60)