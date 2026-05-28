# DS108 Final Project - Bộ dữ liệu benchmark dự báo mưa hằng ngày tại Việt Nam

## 1. Mục tiêu dự án

Dự án xây dựng một bộ dữ liệu khí tượng đa nguồn có tính tái lập để phục vụ phân tích và kiểm định tín hiệu cho bài toán mưa hằng ngày tại Việt Nam. Đóng góp chính của đồ án là quy trình xây dựng dataset, không phải tối ưu mô hình Machine Learning.

Theo rubric môn DS108, dự án thuộc **Frame 1: Data Integration & Tabular Architecture**. Pipeline tích hợp dữ liệu bảng có cấu trúc từ nhiều nguồn khí tượng thực tế, khác nhau về cơ chế thu thập, độ phân giải không gian, độ phân giải thời gian, đơn vị vật lý và định dạng file:

- NOAA GSOD: dữ liệu quan trắc trạm.
- ERA5 single-level: dữ liệu tái phân tích bề mặt dạng lưới.
- ERA5 pressure-level: dữ liệu tái phân tích khí quyển ở tầng 500 hPa và 850 hPa.

Giá trị thực tiễn của project là tạo một benchmark lịch sử minh bạch để nghiên cứu hiện tượng mưa và lượng mưa theo ngày tại một số trạm khí tượng Việt Nam. Dataset phù hợp cho phân tích học thuật và kiểm định chất lượng dữ liệu. Không nên mô tả project này như một hệ thống dự báo vận hành thời gian thực vì ERA5 là dữ liệu tái phân tích hậu nghiệm.

## 2. Phạm vi dataset

- Giai đoạn dữ liệu: 2015-2024.
- Phạm vi không gian: 5 trạm khí tượng đại diện tại Việt Nam.
- Dataset cuối: `data/feature_engineering/feature_engineered_data.csv`.
- Đơn vị dòng dữ liệu: một trạm trong một ngày sau làm sạch và tạo đặc trưng.
- Kích thước artifact hiện tại: 18,255 dòng và 75 cột.

Các target chính:

| Cột | Định nghĩa |
|---|---|
| `PRCP` | Lượng mưa hằng ngày theo mm. |
| `PRCP_label` | Nhãn nhị phân: 1 nếu `PRCP > 0.1 mm`, ngược lại 0. |
| `PRCP_log1p` | Target hồi quy cho lượng mưa ngày mưa: `log(1 + PRCP)`. |

Lưu ý quan trọng về nguồn target: `PRCP` là biến target chính. Trong pipeline silver, nếu GSOD thiếu `PRCP` và ERA5 `tp` có sẵn sau khi ghép không gian-thời gian, giá trị `PRCP` có thể được bù từ ERA5 `tp`. Chi tiết này cần được công bố trong report và datasheet vì nó ảnh hưởng đến cách hiểu cụm từ "lượng mưa thật".

## 3. Cấu trúc repository

```text
project/
|-- README.md
|-- requirements.txt
|-- data/
|   |-- raw/
|   |   |-- gsod/
|   |   |-- single/
|   |   `-- pressure/
|   |-- clean/
|   |   |-- silver_data.csv
|   |   |-- ERA5_single_level.parquet
|   |   |-- ERA5_pressure_final.parquet
|   |   |-- single_level/
|   |   `-- pressure/
|   `-- feature_engineering/
|       `-- feature_engineered_data.csv
|-- docs/
|   |-- data_dictionary.csv
|   `-- datasheet_for_dataset.md
|-- notebooks/
|   `-- eda_and_features_selection.ipynb
|-- outputs/
|   `-- model_final_single_table/
|       `-- final_model_comparison.csv
`-- src/
    |-- _01_crawler.py
    |-- _02_crawls_single.py
    |-- _03_crawls_pressure.py
    |-- _04_single_level.py
    |-- _05_presure.py
    |-- _06_to_silver.py
    |-- _07_feature_engineering.py
    `-- _08_model.py
```

## 4. Cách chạy pipeline

Chạy các script từ thư mục gốc project.

```bash
python src/_01_crawler.py
python src/_02_crawls_single_level.py
python src/_03_crawls_pressure_level.py
python src/_04_single_level.py
python src/_05_presure_level.py
python src/_06_to_silver.py
python src/_07_feature_engineering.py
python src/_08_model.py
```

Các bước pipeline:

| Bước | Script | Vai trò |
|---|---|---|
| 1 | `src/_01_crawler.py` | Tải dữ liệu NOAA GSOD theo trạm-ngày và lưu metadata bronze. |
| 2 | `src/_02_crawls_single.py` | Tải file ERA5 single-level dạng GRIB theo từng năm. |
| 3 | `src/_03_crawls_pressure.py` | Tải file ERA5 pressure-level dạng GRIB theo từng năm. |
| 4 | `src/_04_single_level.py` | Chuyển ERA5 single-level từ GRIB sang parquet sạch ở mức ngày. |
| 5 | `src/_05_presure.py` | Chuyển ERA5 pressure-level từ GRIB sang parquet sạch ở mức ngày. |
| 6 | `src/_06_to_silver.py` | Làm sạch GSOD, chuẩn hóa trạm-ngày, bù thiếu bằng ERA5 và ghép biến pressure-level. |
| 7 | `src/_07_feature_engineering.py` | Tạo feature thời gian, nhiệt động lực, động lực gió, lag, rolling và target. |
| 8 | `src/_08_model.py` | Kiểm định tín hiệu dataset bằng hai kịch bản LightGBM cuối. |

## 5. Quy tắc tái lập

- Không chỉnh tay dữ liệu trong `data/raw/`.
- Không làm sạch hoặc bù thiếu dữ liệu bằng Excel.
- Mọi biến đổi dữ liệu phải được thực hiện bằng code trong `src/`.
- Tách rõ dữ liệu thô, dữ liệu sạch, dữ liệu feature engineering, tài liệu và output.
- Đánh giá model theo split thời gian.
- Không fit imputation, scaling hoặc thống kê phụ thuộc dữ liệu trên toàn bộ dataset trước khi chia train/test.

## 6. Kiểm soát data leakage

Script model cuối loại các cột target và metadata khỏi input model:

- `PRCP`
- `PRCP_label`
- `PRCP_log1p`
- `STATION`
- `time`
- các cột target cũ hoặc giống target như `Target`, `target_prcp_mm`, `rain_target`

Các feature lịch sử mưa được phép dùng chỉ gồm các biến nhìn về quá khứ như `PRCP_lag_*` và `PRCP_past_*`, được tạo bằng logic shift theo từng trạm. Nếu muốn khẳng định bài toán là dự báo vận hành ngày mai, cần kiểm tra lại toàn bộ feature ERA5 cùng ngày và rolling cùng ngày. Trong project hiện tại, dataset nên được mô tả là benchmark lịch sử.

## 7. Bảng so sánh model cuối

Model chỉ được dùng để kiểm định dataset có tín hiệu dự báo hay không.

Bảng cuối nằm tại:

```text
outputs/model_final_single_table/final_model_comparison.csv
```

Bảng chỉ gồm hai dòng benchmark:

- `LightGBM_1stage_Tweedie`: train trên toàn bộ ngày train và dự đoán trực tiếp `PRCP`.
- `LightGBM_2stage_expected`: Stage 1 dự đoán xác suất mưa trên toàn bộ ngày train; Stage 2 dự đoán `PRCP_log1p` chỉ trên ngày mưa trong train; dự đoán cuối là kỳ vọng lượng mưa theo mm:

```text
P(rain) * expm1(predicted PRCP_log1p | rainy)
```

Các metric chính:

- `mae_mm`
- `rmse_mm`
- `wape_percent`
- `r2`
- `bias_mean_pred_minus_true_mm`

Các metric classification không đưa vào bảng chính để báo cáo tập trung vào kiểm định lượng mưa.

## 8. EDA

Notebook EDA chính:

```text
notebooks/eda_and_features_selection.ipynb
```

Notebook bao gồm:

- tỷ lệ ngày mưa và không mưa;
- mùa vụ mưa theo tháng;
- khác biệt mưa theo trạm;
- phân phối lượng mưa trên ngày mưa và tính zero-inflated/right-skewed;
- phân phối feature theo nhóm mưa/không mưa;
- tương quan đa biến và phân tích dư thừa feature;
- kiểm tra duplicate, missingness, PRCP âm và tính nhất quán target.

Khi nộp bài, nên export các hình và bảng EDA quan trọng sang `reports/eda_quality_report/`.

## 9. Tài liệu đi kèm

Các tài liệu hiện có:

- `docs/data_dictionary.csv`
- `docs/datasheet_for_dataset.md`

Hai tài liệu này nên được trích dẫn trong phụ lục của technical report IEEE/ACM.

## 10. Hạn chế và đạo đức

- Dataset chỉ dùng 5 trạm nên không đại diện đầy đủ cho mọi vi khí hậu tại Việt Nam.
- ERA5 là dữ liệu tái phân tích dạng lưới, không phải quan trắc trực tiếp tại trạm.
- Ghép tọa độ trạm với grid ERA5 có thể gây sai lệch đại diện không gian.
- Nếu GSOD thiếu mưa và `PRCP` được bù từ ERA5 `tp`, target trở thành một phần dữ liệu tái phân tích; cần công bố rõ.
- Mất cân bằng giữa ngày mưa và không mưa là đặc tính tự nhiên, không phải lỗi dữ liệu.
- Benchmark phù hợp cho kiểm định học thuật hồi cứu, không phù hợp để triển khai dự báo vận hành nếu chưa thay ERA5 bằng nguồn dữ liệu forecast-time.
