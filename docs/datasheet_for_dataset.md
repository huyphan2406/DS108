# Datasheet For Dataset

Dataset: Bộ dữ liệu benchmark mưa hằng ngày tại Việt Nam  
Môn học: DS108 - Pre-processing & Constructing Dataset  
Khung bài toán: Data Integration & Tabular Architecture  
Artifact cuối: `data/feature_engineering/feature_engineered_data.csv`

## 1. Động cơ xây dựng

Dataset này được xây dựng để phục vụ nghiên cứu tái lập về xác suất mưa và lượng mưa hằng ngày tại một số trạm khí tượng Việt Nam. Trọng tâm của đồ án không phải tối ưu mô hình Machine Learning, mà là tạo một benchmark dạng bảng có tài liệu khoa học rõ ràng từ nhiều nguồn khí tượng khác nhau.

Dataset nhằm thể hiện:

- tích hợp dữ liệu đa nguồn;
- xây dựng panel trạm-ngày;
- chuẩn hóa đơn vị đo;
- xử lý dữ liệu thiếu;
- tạo đặc trưng khí tượng;
- kiểm định benchmark có kiểm soát data leakage.

## 2. Thành phần dataset

Dataset cuối hiện có:

- 18,255 dòng trạm-ngày;
- 75 cột;
- 5 trạm khí tượng;
- dữ liệu ngày từ 2015-01-03 đến 2024-12-31.

Mỗi dòng biểu diễn một trạm trong một ngày sau các bước làm sạch và feature engineering.

Các cột target chính:

- `PRCP`: lượng mưa hằng ngày theo mm;
- `PRCP_label`: 1 nếu `PRCP > 0.1 mm`, ngược lại 0;
- `PRCP_log1p`: `log(1 + PRCP)`.

Dataset còn bao gồm:

- metadata trạm;
- biến thời tiết bề mặt từ GSOD;
- biến ERA5 single-level;
- biến ERA5 pressure-level ở tầng 500 hPa và 850 hPa;
- feature thời gian dạng chu kỳ;
- feature nhiệt động lực, gió, flux, lag và rolling.

Mô tả chi tiết từng cột nằm trong `docs/data_dictionary.csv`.

## 3. Quy trình thu thập

Dataset được xây dựng bằng code từ các nguồn sau:

| Nguồn | Vai trò |
|---|---|
| NOAA GSOD | Quan trắc trạm hằng ngày và nền target mưa. |
| ERA5 single-level | Biến tái phân tích bề mặt dạng lưới và hỗ trợ bù thiếu. |
| ERA5 pressure-level | Biến khí quyển tầng 500 hPa và 850 hPa. |

Script thu thập dữ liệu thô:

- `src/_01_crawler.py`
- `src/_02_crawls_single.py`
- `src/_03_crawls_pressure.py`

Pipeline không dựa trên thao tác làm sạch thủ công bằng Excel.

## 4. Tiền xử lý và làm sạch

Các bước làm sạch và tích hợp chính nằm trong:

- `src/_04_single_level.py`
- `src/_05_presure.py`
- `src/_06_to_silver.py`
- `src/_07_feature_engineering.py`

Các thao tác chính:

- parse và chuẩn hóa cột thời gian;
- loại duplicate theo `STATION` + `DATE` trong GSOD;
- chuyển đơn vị GSOD sang hệ metric;
- loại mã thiếu bất thường của GSOD;
- kiểm tra khoảng hợp lý của áp suất;
- ghép tọa độ trạm với grid ERA5;
- bù một số biến trạm bị thiếu bằng ERA5;
- ghép biến pressure-level;
- tạo feature lag và rolling theo từng trạm;
- tạo target cho mô hình hai giai đoạn.

Dataset cuối đã được kiểm tra:

- không có duplicate theo `STATION` + `time`;
- không có `PRCP` âm;
- không có `inf` hoặc `-inf`;
- `PRCP_label` và `PRCP_log1p` nhất quán với định nghĩa.

Missingness còn lại:

- `WDSP` còn 598 giá trị thiếu trong dataset cuối.
- Khi train model, biến này phải được impute bên trong pipeline fit trên train set, không impute trên toàn bộ dataset trước khi split.

## 5. Nguồn gốc target

`PRCP` là target lượng mưa chính. Trong pipeline silver, nếu GSOD thiếu `PRCP` và ERA5 `tp` có sẵn sau khi ghép theo grid và thời gian, `PRCP` có thể được bù từ ERA5 `tp`.

Thiết kế này giúp panel trạm-ngày đầy đủ hơn, nhưng làm thay đổi nguồn gốc target ở các dòng được bù. Vì vậy:

- report phải nói rõ `PRCP` không hoàn toàn là quan trắc trạm ở mọi dòng;
- phiên bản sau nên thêm cờ nguồn như `PRCP_source`;
- kết luận nên gọi đây là benchmark lịch sử, không phải bộ đo quan trắc tuyệt đối.

## 6. Mục đích sử dụng phù hợp

Phù hợp cho:

- phân tích chất lượng dữ liệu học thuật;
- EDA về mùa vụ mưa và khác biệt theo trạm;
- benchmark mô hình hồi cứu;
- giảng dạy data engineering tái lập và validation tránh leakage.

Không phù hợp nếu chưa bổ sung thêm xử lý:

- dự báo mưa vận hành thời gian thực;
- kết luận khí hậu cho toàn bộ Việt Nam;
- quyết định thủy văn hoặc an toàn ở môi trường rủi ro cao;
- khẳng định mọi giá trị `PRCP` đều là quan trắc trực tiếp tại trạm.

## 7. Chính sách chống data leakage

Validation model cần tuân thủ:

- split theo thời gian;
- không dùng `PRCP`, `PRCP_label`, `PRCP_log1p` làm input feature;
- không dùng metadata như `STATION` hoặc `time` nếu thiết kế mô hình đã loại metadata;
- fit imputation chỉ trên train set;
- chỉ dùng `PRCP_lag_*` và `PRCP_past_*` vì đây là lịch sử mưa nhìn về quá khứ;
- mô tả rõ các feature ERA5 cùng ngày là feature benchmark hồi cứu.

Bảng model cuối cố ý đơn giản và chỉ gồm:

- `LightGBM_1stage_Tweedie`;
- `LightGBM_2stage_expected`.

Output chính:

```text
outputs/model_final_single_table/final_model_comparison.csv
```

## 8. Bias và hạn chế

Các hạn chế đã biết:

- chỉ có 5 trạm;
- lựa chọn trạm chưa đại diện đầy đủ mọi vùng khí hậu Việt Nam;
- grid ERA5 có thể không phản ánh chính xác vi khí hậu tại trạm;
- ghép không gian giữa trạm và grid có thể gây sai lệch đại diện;
- ERA5 là dữ liệu tái phân tích hậu nghiệm;
- phân phối mưa có nhiều giá trị 0 và lệch phải mạnh;
- số ngày mưa và không mưa mất cân bằng tự nhiên;
- `PRCP` được bù từ ERA5 ở một số trường hợp làm yếu cách hiểu target là quan trắc tuyệt đối.

## 9. Bảo trì dataset

Các bước nên làm ở phiên bản sau:

- thêm cờ nguồn cho biến được bù, đặc biệt là `PRCP_source`;
- export hình EDA vào `reports/eda_quality_report/`;
- giữ data dictionary đồng bộ với schema CSV cuối;
- không commit thư mục IDE hoặc output model cũ;
- cân nhắc Git LFS hoặc lưu ngoài repo cho file GRIB rất lớn.

## 10. Phân phối và đạo đức

Dataset dùng dữ liệu khí tượng và tái phân tích công khai, không chứa thông tin cá nhân. Rủi ro đạo đức chính nằm ở việc diễn giải quá mức:

- không hàm ý độ tin cậy dự báo vận hành chỉ từ validation hồi cứu;
- không che giấu việc bù thiếu và target có thể đến từ tái phân tích;
- không trình bày benchmark 5 trạm như đại diện toàn quốc.

Khi sử dụng dataset trong báo cáo, slide hoặc thí nghiệm sau này, cần trích dẫn nguồn dữ liệu và nêu rõ các hạn chế trên.
