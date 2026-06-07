# DS108 - Rainfall Forecasting Data System

## 1. Giới thiệu

Đồ án xây dựng hệ thống dữ liệu phục vụ bài toán dự báo lượng mưa theo ngày tại Việt Nam. Hệ thống gồm pipeline xử lý dữ liệu, mô hình dự báo, Airflow để tự động hóa, Docker để đóng gói môi trường và Streamlit để demo kết quả.

Mục tiêu chính là tạo ra bộ dữ liệu đặc trưng cuối cùng và đánh giá mô hình dự báo mưa theo ngày.

---

## 2. Thành phần chính

| Thành phần | Vai trò |
|---|---|
| `src/` | Chứa các bước thu thập, xử lý dữ liệu, tạo đặc trưng và huấn luyện mô hình |
| `dags/` | Chứa DAG Airflow điều phối pipeline |
| `demo/` | Chứa dashboard Streamlit |
| `data/` | Lưu dữ liệu thô, dữ liệu sạch và dữ liệu đặc trưng |
| `outputs/` | Lưu kết quả mô hình |
| `Dockerfile` | Build dashboard Streamlit |
| `Dockerfile.airflow` | Build môi trường Airflow |
| `docker-compose.yaml` | Khởi chạy toàn bộ hệ thống |

---

## 3. Cấu trúc thư mục

```text
DS108/
├── dags/
│   └── rainfall_pipeline_dag.py
├── demo/
│   └── app.py
├── src/
│   ├── _01_*.py
│   ├── _02_*.py
│   ├── _03_*.py
│   ├── _04_*.py
│   ├── _05_*.py
│   ├── _06_*.py
│   ├── _07_*.py
│   └── _08_model.py
├── data/
│   ├── raw/
│   ├── clean/
│   └── features/
├── outputs/
│   └── model_evaluation/
├── Dockerfile
├── Dockerfile.airflow
├── docker-compose.yaml
├── requirements.txt
├── .env.example
├── .dockerignore
├── .gitignore
└── README.md
```

---

## 4. Nguồn dữ liệu

Project sử dụng các nguồn dữ liệu khí tượng chính:

| Nguồn | Mô tả |
|---|---|
| NOAA GSOD | Dữ liệu quan trắc trạm khí tượng theo ngày |
| ERA5 Single-level | Dữ liệu tái phân tích khí tượng tầng bề mặt |
| ERA5 Pressure-level | Dữ liệu tái phân tích theo tầng khí áp |
| ENSO / MEI | Chỉ số khí hậu hỗ trợ phân tích biến động mùa vụ |

---

## 5. Luồng xử lý

```text
Raw Data
   ↓
Cleaning / Processing
   ↓
Feature Engineering
   ↓
Model Training & Evaluation
   ↓
Dashboard
```

---

## 6. Cấu hình môi trường

Project sử dụng hai file môi trường:

```text
.env.example  # file mẫu, được nộp hoặc commit
.env          # file thật, dùng để chạy local, không commit
```

Tạo file `.env`:

```bash
cp .env.example .env
```

Trên Windows PowerShell:

```powershell
copy .env.example .env
```

Nội dung `.env.example`:

```env
CDSAPI_URL=https://cds.climate.copernicus.eu/api
CDSAPI_KEY=YOUR_UID:YOUR_API_KEY
AIRFLOW_UID=50000
```

Sau đó mở `.env` và thay `CDSAPI_KEY` bằng API key thật của tài khoản Copernicus CDS.

---

## 7. Cách chạy bằng Docker

Yêu cầu:

- Đã cài Docker Desktop.
- Đã mở Docker Desktop.
- Đang đứng tại thư mục gốc project `DS108/`.

Chạy hệ thống:

```bash
docker compose down --volumes --remove-orphans
docker compose up --build
```

Sau khi chạy thành công, mở:

```text
Airflow:   http://localhost:8081
Streamlit: http://localhost:8501
```

Tài khoản Airflow mặc định:

```text
Username: admin
Password: admin
```

Nếu `docker-compose.yaml` map port `8080:8080` thì mở Airflow tại:

```text
http://localhost:8080
```

---

## 8. Chạy pipeline bằng Airflow

Các bước:

1. Mở Airflow UI.
2. Đăng nhập bằng `admin/admin`.
3. Tìm DAG:

```text
ds108_rainfall_pipeline
```

4. Bật DAG nếu đang tắt.
5. Nhấn **Trigger DAG**.
6. Theo dõi trạng thái trong tab **Graph** hoặc **Grid**.

Nếu task lỗi, mở **Logs** của task đó để xem nguyên nhân.

---

## 9. Streamlit Dashboard

Dashboard nằm tại:

```text
demo/app.py
```

Mở dashboard tại:

```text
http://localhost:8501
```

Dashboard hỗ trợ:

- Xem dữ liệu sau xử lý.
- Lọc theo thời gian và trạm khí tượng.
- Xem thống kê mô tả.
- Xem EDA cơ bản.
- Kiểm tra missing values, duplicate và giá trị bất thường.
- Xem bản đồ trạm khí tượng.
- Xem kết quả mô hình trong `outputs/model_evaluation/`.

---
