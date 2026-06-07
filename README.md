# DS108 - Rainfall Forecasting Data System

## 1. Giới thiệu

Đây là đồ án xây dựng hệ thống dữ liệu phục vụ bài toán dự báo lượng mưa theo ngày tại Việt Nam. Project được nâng cấp từ dạng notebook/script rời rạc thành một hệ thống có khả năng tái lập, gồm ba thành phần chính:

- **Airflow**: điều phối và giám sát full data pipeline.
- **Streamlit**: dashboard tương tác để demo, xem EDA, kiểm tra chất lượng dữ liệu và kết quả mô hình.
- **Docker Compose**: đóng gói toàn bộ môi trường chạy để có thể triển khai lại nhất quán trên máy khác.

---

## 2. Kiến trúc hệ thống

Luồng xử lý tổng quát:

```text
Raw Data
   ↓
Full Pipeline trong src/_01_*.py → ... → src/_08_*.py
   ↓
Airflow điều phối và ghi log
   ↓
data/features + outputs + reports
   ↓
Streamlit Dashboard đọc output để demo
   ↓
Docker Compose đóng gói toàn bộ hệ thống
```

Vai trò từng thành phần:

```text
Airflow   = tự động hóa pipeline
Streamlit = giao diện dashboard tương tác
Docker    = đóng gói môi trường chạy
```

---

## 3. Cấu trúc thư mục chính

```text
DS108/
├── data/
│   ├── raw/
│   ├── clean/
│   └── features/
├── src/
│   ├── _01_*.py
│   ├── _02_*.py
│   ├── ...
│   └── _08_*.py
├── dags/
│   └── rainfall_pipeline_dag.py
├── demo/
│   └── app.py
├── scripts/
│   └── run_pipeline.py
├── outputs/
├── reports/
├── notebooks/
├── Dockerfile
├── Dockerfile.airflow
├── docker-compose.yaml
├── requirements.txt
├── .dockerignore
├── .gitignore
└── README.md
```

---

## 4. Full Pipeline

Pipeline chính nằm trong thư mục `src/`. Các file pipeline được đặt theo quy ước:

```text
_NN_ten_buoc.py
```

Ví dụ:

```text
_01_*.py
_02_*.py
...
_08_*.py
```

Hệ thống sẽ tự động phát hiện các file có pattern:

```text
^_\d{2}_.+\.py$
```

Sau đó sắp xếp theo số thứ tự và chạy tuần tự từ `_01` đến `_08`.

Các file helper, backup, old, test, tmp sẽ không được xem là pipeline chính.

---

## 5. Airflow

Airflow được dùng để điều phối full pipeline.

DAG chính:

```text
ds108_rainfall_pipeline
```

Các bước chính trong DAG:

```text
check_project_structure
→ prepare_output_dirs
→ run_01_...
→ run_02_...
→ ...
→ run_08_...
→ validate_feature_dataset
→ summarize_outputs
```

Airflow sẽ:

- Kiểm tra cấu trúc project.
- Chuẩn bị thư mục output.
- Tự động phát hiện và chạy các script pipeline trong `src/`.
- Dừng pipeline nếu một task bị lỗi.
- Kiểm tra output quan trọng sau khi chạy.
- Ghi log cho từng bước.

Output quan trọng được kiểm tra:

```text
data/features/feature_engineered_data.csv
```

---

## 6. Streamlit Dashboard

Dashboard nằm tại:

```text
demo/app.py
```

Dashboard đọc dữ liệu từ output của pipeline:

```text
data/features/feature_engineered_data.csv
outputs/
reports/
```

Dashboard hỗ trợ:

- Xem tổng quan dữ liệu.
- Lọc dữ liệu theo thời gian.
- Lọc theo trạm khí tượng nếu có cột `STATION`.
- Xem bảng dữ liệu sau lọc.
- Xem thống kê mô tả.
- Xem phân phối biến.
- Xem phân phối ngày mưa/không mưa.
- Xem lượng mưa theo thời gian.
- Xem tính mùa vụ theo tháng.
- Xem correlation heatmap.
- Kiểm tra missing values.
- Kiểm tra duplicate.
- Kiểm tra giá trị PRCP bất thường.
- Xem bản đồ trạm nếu có `LATITUDE` và `LONGITUDE`.
- Xem kết quả mô hình nếu có file trong `outputs/` hoặc `reports/`.

Dashboard chỉ đọc output, không xử lý pipeline chính.

---

## 7. Docker Compose

Hệ thống được đóng gói bằng Docker Compose với ba service:

```text
airflow   = chạy Airflow Webserver + Scheduler
dashboard = chạy Streamlit Dashboard
pipeline  = chạy full pipeline thủ công khi cần
```

File cấu hình chính:

```text
docker-compose.yaml
```

---

## 8. Cách chạy hệ thống

Yêu cầu trước khi chạy:

- Cài Docker Desktop.
- Mở Docker Desktop trước khi chạy lệnh.
- Đứng tại thư mục gốc project `DS108/`.

Chạy toàn bộ hệ thống:

```bash
docker compose up --build
```

Sau khi chạy, mở Airflow tại:

```text
http://localhost:8080
```

Tài khoản đăng nhập:

```text
username: admin
password: admin
```

Mở Streamlit Dashboard tại:

```text
http://localhost:8501
```

---

## 9. Chạy full pipeline bằng Airflow

Các bước:

```text
1. Mở http://localhost:8080
2. Đăng nhập bằng admin/admin
3. Tìm DAG: ds108_rainfall_pipeline
4. Bật DAG
5. Bấm Trigger DAG
6. Vào Graph để xem trạng thái từng task
```

Ý nghĩa trạng thái:

```text
Màu xanh = task chạy thành công
Màu đỏ   = task bị lỗi, cần vào Logs để xem nguyên nhân
```

---

## 10. Chạy full pipeline thủ công bằng Docker

Có thể chạy full pipeline không qua giao diện Airflow bằng lệnh:

```bash
docker compose --profile manual run --rm pipeline
```

Lệnh này sẽ chạy:

```bash
python scripts/run_pipeline.py
```

Script `run_pipeline.py` sẽ tự động tìm và chạy tuần tự các file pipeline chính trong `src/`.

---

## 11. Dừng hệ thống

Dừng container:

```bash
docker compose down
```

Nếu cần xóa sạch container/volume để chạy lại từ đầu:

```bash
docker compose down --volumes --remove-orphans
```

---

## 12. Lưu ý về tính tái lập

Hệ thống mặc định chạy full pipeline từ các script gốc trong `src/`, không chỉ chạy demo từ các bước cuối.

Nếu một số bước đầu như download/crawl dữ liệu cần internet hoặc API key, cần chuẩn bị trước file `.env` dựa trên `.env.example` hoặc đảm bảo dữ liệu raw đã có trong `data/raw/`.

Không nên đưa file `.env` thật vào bản nộp hoặc Git repository.

---

## 13. Các file không nên nộp/Git tracking

Các file/thư mục cá nhân hoặc nhạy cảm nên được bỏ qua:

```text
.env
.idea/
__pycache__/
*.pyc
.venv/
venv/
logs/
```

Nên dùng `.env.example` để mô tả biến môi trường cần thiết thay vì nộp `.env` thật.

---

## 14. Minh chứng nên chụp khi nộp báo cáo

Nên chụp các hình sau:

```text
1. Docker Compose đang chạy các service.
2. Airflow UI có DAG ds108_rainfall_pipeline.
3. Airflow Graph hiển thị các task từ _01 đến _08.
4. Một DAG run thành công hoặc log task thành công.
5. Streamlit Dashboard trang tổng quan.
6. Biểu đồ EDA hoặc missing value trên dashboard.
```

---

## 15. Mô tả ngắn để đưa vào báo cáo

Hệ thống được triển khai theo hướng production-lite bằng Docker Compose. Airflow chịu trách nhiệm điều phối full pipeline từ các script `src/_01_*.py` đến `src/_08_*.py`, Streamlit cung cấp dashboard tương tác để xem EDA, kiểm tra chất lượng dữ liệu và kết quả mô hình, còn Docker đóng gói toàn bộ môi trường chạy gồm mã nguồn, thư viện Python, Airflow và Streamlit. Cách triển khai này giúp project có khả năng tái lập, dễ demo và phù hợp hơn với quy trình xây dựng hệ thống dữ liệu trong thực tế.
