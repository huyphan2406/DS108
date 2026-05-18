# DS108 Final Project — Xây dựng bộ dữ liệu dự báo mưa tại Việt Nam

## 1. Mục tiêu dự án

Dự án tập trung vào **xây dựng và tiền xử lý bộ dữ liệu khí tượng đa nguồn** phục vụ bài toán phân tích/phân loại mưa tại Việt Nam.

Trọng tâm chính của đồ án là:

- Thu thập dữ liệu khí tượng từ nhiều nguồn.
- Làm sạch và chuẩn hóa dữ liệu.
- Bù khuyết dữ liệu thiếu có kiểm soát.
- Phân tích EDA và chất lượng dữ liệu.
- Tạo đặc trưng khí tượng.
- Kiểm chứng dữ liệu bằng mô hình học máy.

> Lưu ý: Mô hình học máy chỉ dùng để **kiểm chứng tính hữu ích của bộ dữ liệu**, không phải đóng góp chính của đồ án.

---

## 2. Nguồn dữ liệu

| Nguồn | Vai trò |
|---|---|
| NOAA GSOD | Dữ liệu quan trắc khí tượng tại trạm |
| ERA5 Single-Level | Dữ liệu tái phân tích bề mặt |
| ERA5 Pressure-Level | Dữ liệu khí quyển tầng 500 hPa và 850 hPa |
| MEI v2 / ENSO | Chỉ số khí hậu quy mô lớn |

---

## 3. Cấu trúc thư mục

```text
project/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/                      # Dữ liệu gốc, không chỉnh tay
│   ├── clean/                    # Dữ liệu đã làm sạch/tích hợp
│   ├── feature_engineering/      # Dữ liệu sau tạo đặc trưng
│   └── processed/                # Dataset cuối cùng nếu cần
│
├── src/
│   ├── 01_crawler_improved.py
│   ├── 02_pressure_improved.py
│   ├── 03_single_level_improved.py
│   ├── 04_enso_improved.py
│   ├── 05_to_silver_improved.py
│   ├── 06_feature_engineering_improved.py
│   └── 07_model_validation_improved.py
│
├── notebooks/
│   └── 08_eda_quality_report.ipynb
│
├── reports/
│   ├── data_quality/
│   ├── eda_quality_report/
│   └── model_validation/
│
├── docs/
│   ├── data_dictionary.csv
│   └── datasheet_for_dataset.md
│
└── models/
    └── rainfall_validation/
```

---

## 4. Các bước pipeline

### Bước 1 — Crawl dữ liệu GSOD

File:

```text
src/01_crawler_improved.py
```

Nhiệm vụ:

- Tải dữ liệu GSOD giai đoạn 2015–2024.
- Lưu dữ liệu thô ở tầng Bronze.
- Lưu metadata: thời điểm crawl, URL nguồn, số dòng từng trạm, số năm thiếu.
- Có retry/backoff khi lỗi mạng.

Output:

```text
data/raw/bronze_data.csv
data/raw/bronze_metadata.json
data/raw/bronze_station_summary.csv
```

---

### Bước 2 — Xử lý ERA5 Pressure-Level

File:

```text
src/02_pressure_improved.py
```

Nhiệm vụ:

- Đọc dữ liệu ERA5 tầng 500 hPa và 850 hPa.
- Chuyển đổi đơn vị.
- Tổng hợp về dữ liệu ngày.
- Kiểm tra:
  - `z_500 > z_850`
  - `t_850 > t_500`
  - Không duplicate theo `time, latitude, longitude`

Output:

```text
data/clean/ERA5_pressure_final.parquet
reports/data_quality/pressure/
```

---

### Bước 3 — Xử lý ERA5 Single-Level

File:

```text
src/03_single_level_improved.py
```

Nhiệm vụ:

- Xử lý các biến bề mặt như nhiệt độ, điểm sương, áp suất, gió, lượng mưa.
- Xử lý riêng biến `tp` để tránh double counting.
- Kiểm tra mưa âm và cực trị bất thường.
- So sánh ERA5 `tp` với GSOD `PRCP`.

Output:

```text
data/clean/ERA5_single_level.parquet
reports/data_quality/single_level/
```

---

### Bước 4 — Xử lý ENSO / MEI v2

File:

```text
src/04_enso_improved.py
```

Nhiệm vụ:

- Chuyển MEI từ dạng rộng sang dạng tháng.
- Mapping:
  - `DJ → tháng 1`
  - `JF → tháng 2`
  - ...
  - `ND → tháng 12`
- Tạo thêm:
  - `ENSO_lag_1`
  - `ENSO_lag_2`
- Đồng bộ giai đoạn 2015–2024.

Output:

```text
data/clean/enso_clean.csv
reports/data_quality/enso/
```

---

### Bước 5 — Tích hợp và bù khuyết dữ liệu Silver

File:

```text
src/05_to_silver_improved.py
```

Nhiệm vụ:

- Làm sạch GSOD.
- Chuẩn hóa đơn vị.
- Chẩn đoán missingness trước imputation.
- Tạo panel đầy đủ theo trạm-ngày.
- Bù khuyết bằng ERA5.
- Nội suy `VISIB` theo từng trạm, không nội suy chéo trạm.
- So sánh trước/sau imputation.
- Lưu cờ nguồn dữ liệu như `PRCP_source`, `TEMP_source`.

Output:

```text
data/clean/silver_data_ver2.csv
reports/data_quality/silver/
```

---

### Bước 6 — Feature Engineering

File:

```text
src/06_feature_engineering_improved.py
```

Nhiệm vụ:

- Tạo target duy nhất: `rain_target`.
- Giữ lượng mưa gốc dưới dạng `PRCP_mm`.
- Tạo đặc trưng thời gian, mùa vụ, độ ẩm, gió, flux, rolling, lag.
- Có chế độ:
  - `same_day_classification`
  - `next_day_forecast`
- Xử lý NaN sau lag/rolling.

Output:

```text
data/feature_engineering/feature_engineered_data.csv
data/feature_engineering/model_ready_data.csv
reports/data_quality/feature_engineering/
```

---

### Bước 7 — Kiểm chứng bằng mô hình

File:

```text
src/07_model_validation_improved.py
```

Nhiệm vụ:

- Đọc dữ liệu feature engineering.
- Dùng `rain_target` làm nhãn.
- So sánh baseline và mô hình:
  - Majority Baseline
  - Logistic Regression
  - Random Forest
  - LightGBM
- Tối ưu threshold trên validation set.
- Báo cáo Precision, Recall, F1, ROC-AUC, PR-AUC, Brier Score.

Output:

```text
models/rainfall_validation/
reports/model_validation/
```

---

### Bước 8 — EDA Quality Report

File:

```text
notebooks/08_eda_quality_report.ipynb
```

Nội dung:

- Missingness heatmap.
- Missing rate theo trạm.
- Phân phối các biến chính.
- Mùa vụ mưa theo tháng.
- Mưa theo trạm.
- Correlation heatmap.
- Class imbalance.
- So sánh GSOD vs ERA5.
- Trước/sau imputation.
- Feature importance.
- Feature selection.

Output:

```text
reports/eda_quality_report/
```

---

## 5. Cách chạy project

Cài môi trường:

```bash
python -m venv .venv
```

Kích hoạt môi trường:

```bash
# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate
```

Cài thư viện:

```bash
pip install -r requirements.txt
```

Chạy pipeline:

```bash
python src/01_crawler_improved.py
python src/02_pressure_improved.py
python src/03_single_level_improved.py
python src/04_enso_improved.py
python src/05_to_silver_improved.py
python src/06_feature_engineering_improved.py
python src/07_model_validation_improved.py
```

Chạy notebook EDA:

```bash
jupyter notebook notebooks/08_eda_quality_report.ipynb
```

---

## 6. Quy tắc tái lập

- Không chỉnh tay dữ liệu trong `data/raw/`.
- Không xử lý dữ liệu bằng Excel.
- Mọi bước tiền xử lý phải chạy bằng code.
- Mọi bước imputation phải có report.
- Target chỉ tạo ở bước 6.
- Model ở bước 7 dùng `rain_target`, không tạo target lại từ `PRCP`.
- Split train/validation/test phải theo thời gian.
- Các report và hình phải lưu trong `reports/`.

---

## 7. Lưu ý data leakage

Không dùng các cột sau làm input model:

```text
PRCP
PRCP_mm
target_prcp_mm
target_time
rain_target
Target
```

Nếu bài toán là **phân loại mưa cùng ngày**, có thể dùng biến khí tượng cùng ngày.

Nếu bài toán là **dự báo ngày mai**, rolling phải dùng dữ liệu quá khứ:

```python
x.shift(1).rolling(window=3, min_periods=1).mean()
```

ERA5 là dữ liệu tái phân tích hậu nghiệm, nên chỉ phù hợp để xây dựng dataset lịch sử. Nếu nói là dự báo thời gian thực, cần giải thích rõ hoặc thay bằng dữ liệu forecast.

---

## 8. Thành phẩm cần nộp

```text
data/clean/silver_data_ver2.csv
data/feature_engineering/feature_engineered_data.csv
data/feature_engineering/model_ready_data.csv
docs/data_dictionary.csv
docs/datasheet_for_dataset.md
reports/data_quality/
reports/eda_quality_report/
reports/model_validation/
README.md
requirements.txt
```

---

## 9. Hạn chế

- Chỉ dùng 5 trạm nên chưa đại diện tuyệt đối toàn bộ Việt Nam.
- ERA5 là dữ liệu dạng lưới, không hoàn toàn giống quan trắc trạm.
- ENSO là chỉ số tháng, ảnh hưởng đến mưa có thể gián tiếp.
- Mô hình chỉ dùng để kiểm chứng dataset, không phải hệ thống dự báo vận hành.

---

## 10. Kết luận

Dự án xây dựng một pipeline dữ liệu khí tượng đa nguồn theo hướng tái lập, có kiểm soát chất lượng và hạn chế data leakage. Thành phẩm chính là bộ dữ liệu đã được làm sạch, tích hợp, bù khuyết, tạo đặc trưng và phân tích EDA đầy đủ để phục vụ bài toán phân loại mưa tại Việt Nam.
