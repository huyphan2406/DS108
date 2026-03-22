import requests
import pandas as pd
import time
import yaml
from pathlib import Path

# ==========================================
# 1. THIẾT LẬP ĐƯỜNG DẪN
# ==========================================
current_script_dir = Path(__file__).resolve().parent
base_dir = current_script_dir.parent.parent
output_dir = base_dir / "data" / "bronze" / "batch" / "nasapower"
output_dir.mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. HÀM ĐỌC CẤU HÌNH YAML
# ==========================================
def load_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

airports_config = load_yaml(base_dir / "configs" / "airports.yaml")
pipeline_config = load_yaml(base_dir / "configs" / "pipeline.yaml")

airports = airports_config['airports']
base_url = pipeline_config['source']['base_url']
req_config = pipeline_config['request']
dates = pipeline_config['date_range']

# Lấy bộ từ điển mapping từ YAML (Key: Mã NASA, Value: Tên Cột Mới)
param_mapping = pipeline_config['parameters']

# Nối các Key thành chuỗi để gọi API NASA
nasa_params_str = ",".join(param_mapping.keys())

# Thứ tự cột đầu ra mong muốn
final_columns = [
    "date", "location", "latitude", "longitude",
    "precipitationCorrected", "temperature2m", "temperature2mMax",
    "temperature2mMin", "dewPoint2m", "earthSkinTemperature",
    "relativeHumidity2m", "specificHumidity2m", "surfacePressure",
    "windSpeed10m", "windDirection10m", "shortwaveDownwardIrradiance",
    "longwaveDownwardIrradiance", "surfaceSoilWetness",
    "rootZoneSoilWetness", "cloudFraction"
]

# ==========================================
# 3. TIẾN HÀNH CRAWL DỮ LIỆU
# ==========================================
print("\n🚀 BẮT ĐẦU CRAWL DỮ LIỆU NASA POWER (FULL BIẾN)...")

url = f"{base_url}/daily/point"

for code, info in airports.items():
    station_name = info['station_name']
    print(f"⏳ Đang tải trạm: {station_name} ({code})...")

    params = {
        "parameters": nasa_params_str,
        "community": req_config['community'],
        "longitude": info['lon'],
        "latitude": info['lat'],
        "start": dates['start_date'],
        "end": dates['end_date'],
        "format": req_config['format'],
        "time-standard": req_config['time_standard']
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if "properties" in data and "parameter" in data["properties"]:
            daily_data = data["properties"]["parameter"]

            # Khởi tạo DataFrame
            df = pd.DataFrame(daily_data)

            # Cột index của Pandas đang là ngày tháng -> Biến nó thành cột 'date'
            df.reset_index(names="date", inplace=True)
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')

            # Đổi tên các cột mã NASA thành tên dễ đọc dựa vào file YAML
            df.rename(columns=param_mapping, inplace=True)

            # Thêm các cột định danh và không gian
            df["location"] = station_name
            df["latitude"] = info['lat']
            df["longitude"] = info['lon']

            # Sắp xếp lại thứ tự cột cho đúng chuẩn yêu cầu
            # Dùng list comprehension để lọc những cột có tồn tại tránh lỗi
            ordered_cols = [c for c in final_columns if c in df.columns]
            df = df[ordered_cols]

            # Lưu file
            file_name = f"{code}_nasa_daily.csv"
            output_path = output_dir / file_name
            df.to_csv(output_path, index=False)

            print(f"   ✅ Đã lưu -> {output_path.name} ({len(df)} dòng, {len(df.columns)} cột)")
        else:
            print(f"   ❌ Lỗi: Không tìm thấy dữ liệu cho trạm {station_name}")

    except requests.exceptions.RequestException as e:
        print(f"   ❌ LỖI KẾT NỐI ({station_name}): {e}")
    except Exception as e:
        print(f"   ❌ LỖI HỆ THỐNG ({station_name}): {e}")

    # Nghỉ 3 giây chống block IP
    time.sleep(3)

print(f"🎉 HOÀN TẤT!.")