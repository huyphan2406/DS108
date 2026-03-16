import json
import pandas as pd

# 1. Load file JSON
with open('2025_daily.json', 'r') as f:
    data = json.load(f)

# 2. Lấy metadata và dữ liệu thời tiết
metadata = data['metadata']
parameters = data['nasa_power_response']['properties']['parameter']

# 3. Chuyển dữ liệu sang dataframe
# Lấy tất cả ngày từ PRECTOTCORR
dates = list(parameters['PRECTOTCORR'].keys())

# Tạo dict cho dataframe
df_dict = {
    'DATE': dates
}

# Thêm từng parameter vào dataframe
for param, values in parameters.items():
    df_dict[param] = [values[date] for date in dates]

# 4. Thêm thông tin location
df_dict['LOCATION'] = metadata['location']
df_dict['LATITUDE'] = metadata['latitude']
df_dict['LONGITUDE'] = metadata['longitude']

# 5. Tạo dataframe
df = pd.DataFrame(df_dict)

# 6. Lưu ra CSV
df.to_csv('raw_data.csv', index=False)

print("Đã chuyển JSON sang CSV xong!")