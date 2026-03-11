import json
import csv
from pathlib import Path

# Đường dẫn file JSON
json_file = Path(r"/data/bronze/001_Ca_Mau/2015_hourly.json")
csv_file = json_file.with_suffix(".csv")  # sẽ tạo file cùng tên với .csv

# Đọc JSON
with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

parameters = data.get("properties", {}).get("parameter", {})
if not parameters:
    raise ValueError("Không tìm thấy dữ liệu tham số trong JSON")

param_names = list(parameters.keys())
timestamps = list(next(iter(parameters.values())).keys())

# Ghi CSV
with open(csv_file, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["time"] + param_names)
    for t in timestamps:
        row = [t] + [parameters[p][t] for p in param_names]
        writer.writerow(row)

print(f"Đã convert JSON -> CSV: {csv_file}")