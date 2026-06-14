# 📊 THU THẬP VÀ TIỀN XỬ LÝ DỮ LIỆU KHÍ TƯỢNG ĐA NGUỒN CHO BÀI TOÁN DỰ BÁO LƯỢNG MƯA HẰNG NGÀY TẠI VIỆT NAM GIAI ĐOẠN 2015-2024

> Một pipeline thu thập và tiền xử lý dữ liệu khí tượng đa nguồn, phục vụ bài toán dự báo lượng mưa hằng ngày tại Việt Nam. Dự án bao gồm các bước thu thập dữ liệu, làm sạch dữ liệu, tích hợp dữ liệu, trích xuất đặc trưng và benchmark mô hình dự báo mưa. Ngoài ra, hệ thống được đóng gói Docker, tự động hóa bằng Airflow và minh họa bằng Streamlit để phục vụ cho tái lập và phát triển sau này.

## GVHD:

1. TS. Nguyễn Gia Tuấn Anh
2. CN. Trần Quốc Khánh

---

## Thành viên:

1. Phạm Đình Quang Huy - MSSV: 24520689
2. Phan Gia Quốc Huy - MSSV: 24520694

---

## 🎯 Giới Thiệu Dự Án

Đây là một đồ án phục vụ cho mục đích học tập của môn **DS108 - Thu thập và Tiền xử lý dữ liệu** của trường ĐH Công nghệ Thông tin, ĐHQG-HCM.

**Mục tiêu chính:**

- Xây dựng pipeline thu thập, xử lý và phân tích dữ liệu khí tượng
- Tích hợp dữ liệu GSOD và ERA5 theo đơn vị station-day
- Tạo bộ dữ liệu đặc trưng phục vụ bài toán dự báo lượng mưa hằng ngày
- Benchmark dữ liệu bằng baseline Persistence và các mô hình LightGBM
- Tự động hóa quy trình xử lý dữ liệu bằng Apache Airflow
- Minh họa dữ liệu và kết quả thực nghiệm bằng Streamlit

**Mục đích cuối cùng:** Tạo bộ dữ liệu đặc trưng (feature engineered data) và đánh giá tín hiệu dữ liệu thông qua các mô hình benchmark dự báo lượng mưa.

---

## 📁 Cấu Trúc Thư Mục

```
DS108/
│
├── 📂 dags/                          # DAG của Apache Airflow
│   └── rainfall_pipeline_dag.py      # Pipeline tự động hóa quy trình dữ liệu
│
├── 📂 data/                          # Dữ liệu của dự án (phân 3 giai đoạn)
│   ├── raw/                          # Dữ liệu thô (giai đoạn 1)
│   │   ├── era5_pressure_level/      # Dữ liệu áp suất từ ERA5 (2015-2024)
│   │   ├── era5_single_level/        # Dữ liệu mặt đơn từ ERA5 (2015-2024)
│   │   └── gsod/                     # Dữ liệu từ GSOD
│   │       ├── bronze_data.csv       # Dữ liệu thô từ GSOD
│   │       ├── bronze_metadata.json  # Thông tin metadata
│   │       └── bronze_station_summary.csv  # Tóm tắt trạm quan trắc
│   │
│   ├── processed/                    # Dữ liệu đã xử lý (giai đoạn 2)
│   │   ├── components/               # Các thành phần xử lý trung gian
│   │   ├── era5_pressure_level/      # Dữ liệu áp suất đã xử lý
│   │   ├── era5_single_level/        # Dữ liệu mặt đơn đã xử lý
│   │   └── silver/                   # Dữ liệu silver (đã sạch)
│   │       └── silver_data.csv       # Dữ liệu chính đã xử lý
│   │
│   └── features/                     # Dữ liệu đặc trưng (giai đoạn 3)
│       └── feature_engineered_data.csv  # Dữ liệu với các features được tạo
│
├── 📂 demo/                          # Ứng dụng Streamlit dashboard
│   └── app.py                        # Ứng dụng demo hiển thị kết quả
│
├── 📂 docs/                          # Tài liệu dự án
│   └── Codebook.csv                  # Bộ từ điển giải thích các biến dữ liệu
│
├── 📂 notebooks/                     # Jupyter Notebooks phân tích
│   ├── eda.ipynb                     # Phân tích dữ liệu khám phá (EDA)
│   ├── feature_selection.ipynb       # Lựa chọn features quan trọng
│   └── eda_figs.pkl                  # Các hình vẽ từ phân tích
│
├── 📂 results/                       # Kết quả mô hình
│   └── model_comparison_results_improved.csv  # Kết quả so sánh các mô hình
│
├── 📂 src/                           # Mã nguồn Python - quy trình xử lý
│   ├── _01_crawler.py                # Bước 1: Thu thập dữ liệu
│   ├── _02_crawls_single_level.py    # Bước 2: Thu thập dữ liệu ERA5 mặt đơn
│   ├── _03_crawls_pressure_level.py  # Bước 3: Thu thập dữ liệu ERA5 áp suất
│   ├── _04_single_level.py           # Bước 4: Xử lý dữ liệu mặt đơn
│   ├── _05_pressure_level.py         # Bước 5: Xử lý dữ liệu áp suất
│   ├── _06_to_silver.py              # Bước 6: Chuyển đổi sang dữ liệu silver
│   ├── _07_feature_engineering.py    # Bước 7: Tạo đặc trưng
│   └── _08_model.py                  # Bước 8: Huấn luyện và đánh giá mô hình
│
├── docker-compose.yaml              # Cấu hình docker-compose (chạy Airflow + Streamlit)
├── Dockerfile                       # Build image cho Streamlit dashboard
├── Dockerfile.airflow               # Build image cho Apache Airflow
├── webserver_config.py              # Cấu hình Airflow webserver
├── .dockerignore                    # Các file loại bỏ khi build Docker
├── requirements.txt                 # Danh sách thư viện Python cần cài
├── README.md                      # Tài liệu hướng dẫn chính
├── LICENSE                        # Giấy phép dự án
├── .env.example                   # Template biến môi trường
├── .gitignore                     # Các file loại bỏ khỏi Git
└── silver.ipynb                   # Notebook xử lý silver
```

---

## 🔍 Chi Tiết Các Thành Phần

### 1️⃣ **Thư Mục `src/` - Quy Trình Xử Lý Dữ Liệu**

Chứa 8 bước xử lý dữ liệu theo tuần tự:

| File                           | Bước | Mô Tả                                                             |
| ------------------------------ | ---- | ----------------------------------------------------------------- |
| `_01_crawler.py`               | 1    | Thu thập dữ liệu từ GSOD thông qua public API                     |
| `_02_crawls_single_level.py`   | 2    | Thu thập các biến ở mặt đơn (Single Level) từ ERA5                |
| `_03_crawls_pressure_level.py` | 3    | Thu thập các biến ở các mức áp suất từ ERA5                       |
| `_04_single_level.py`          | 4    | Xử lý dữ liệu mặt đơn (làm sạch, chuyển đổi định dạng)            |
| `_05_pressure_level.py`        | 5    | Xử lý dữ liệu áp suất                                             |
| `_06_to_silver.py`             | 6    | Gộp và chuyển đổi dữ liệu sang định dạng "silver" (dữ liệu chính) |
| `_07_feature_engineering.py`   | 7    | Tạo các đặc trưng mới từ dữ liệu (feature engineering)            |
| `_08_model.py`                 | 8    | Huấn luyện mô hình học máy và đánh giá                            |

**Kỹ thuật xử lý:** Mỗi bước sử dụng thư viện Pandas, NumPy, xarray để xử lý dữ liệu khoa học.

---

### 2️⃣ **Thư Mục `data/` - Dữ Liệu Dự Án**

Dữ liệu được tổ chức theo **3 giai đoạn** (Data Lakehouse architecture):

#### **a) `data/raw/` - Dữ liệu Thô (Bronze Layer)**

- **ERA5 Pressure Level (2015-2024):** Dữ liệu ở các mức áp suất từ Copernicus ERA5
  - Các biến: Nhiệt độ, độ ẩm, độ cao địa thế, v.v. ở các mức áp suất khác nhau
  - Định dạng: GRIB (.grib) - định dạng dữ liệu khí hậu tiêu chuẩn
  - File index: `.idx` file để đánh chỉ mục GRIB

- **ERA5 Single Level (2015-2024):** Dữ liệu khí tượng bề mặt từ ERA5
  - Các biến: nhiệt độ 2m, điểm sương 2m, áp suất bề mặt, áp suất mực biển, gió 10m, địa thế bề mặt, mặt nạ đất--biển, v.v.
  - Lưu ý: biến tổng lượng mưa ERA5 không được dùng để thay thế hoặc bù cho biến mục tiêu PRCP từ GSOD.
  - Định dạng: GRIB

- **GSOD (Global Summary of the Day):** Dữ liệu từ các trạm quan trắc trên mặt đất
  - `bronze_data.csv` - Dữ liệu thô từ các trạm
  - `bronze_metadata.json` - Thông tin về các trạm quan trắc
  - `bronze_station_summary.csv` - Tóm tắt dữ liệu theo trạm

#### **b) `data/processed/` - Dữ liệu Đã Xử Lý (Silver Layer)**

- **components/:** Dữ liệu thành phần trung gian
- **era5_pressure_level/:** Dữ liệu áp suất sau xử lý
- **era5_single_level/:** Dữ liệu mặt đơn sau xử lý
- **silver/:** Dữ liệu chính đã sạch và hợp nhất
  - `silver_data.csv` - Dữ liệu chính (kết hợp tất cả các nguồn)

#### **c) `data/features/` - Dữ liệu Đặc Trưng (Gold Layer)**

- **feature_engineered_data.csv:** Dữ liệu cuối cùng với các features được tạo
  - Bao gồm các biến gốc + các features được tính toán
  - Dùng cho huấn luyện mô hình dự báo

---

### 3️⃣ **Thư Mục `notebooks/` - Jupyter Notebooks Phân Tích**

| Notebook                  | Mục Đích                                                       |
| ------------------------- | -------------------------------------------------------------- |
| `eda.ipynb`               | **Exploratory Data Analysis (EDA)** - Khám phá và hiểu dữ liệu |
| `feature_selection.ipynb` | **Lựa chọn Features** - Chọn các features quan trọng nhất      |
| `eda_figs.pkl`            | File lưu trữ các hình vẽ từ phân tích                          |

**Mục tiêu:** Phân tích thống kê dữ liệu, tìm hiểu phân phối, mối tương quan, và xác định features quan trọng.

---

### 4️⃣ **Thư Mục `dags/` - Apache Airflow Orchestration**

**rainfall_pipeline_dag.py:**

- Xác định quy trình tự động hóa (Directed Acyclic Graph - DAG)
- Liên kết các bước xử lý từ `src/` thành một pipeline
- Cho phép chạy tự động theo lịch trình hoặc thủ công
- Quản lý phụ thuộc giữa các bước

---

### 5️⃣ **Thư Mục `demo/` - Ứng Dụng Streamlit**

**app.py:**

- Ứng dụng web interactive xây dựng bằng Streamlit
- Minh họa dữ liệu, biểu đồ phân tích và kết quả thực nghiệm
- Cho phép người dùng tương tác và xem kết quả benchmark
- Chạy trong container Docker

---

### 6️⃣ **Thư Mục `results/` - Kết Quả Mô Hình**

**model_comparison_results_improved.csv:**

- Bảng so sánh hiệu suất của baseline Persistence và các mô hình LightGBM
- Chứa các chỉ số chính: MAE, RMSE, WAPE, R² và Bias
- Giúp so sánh các mô hình theo từng tiêu chí đánh giá, không kết luận một mô hình tốt nhất tuyệt đối

---

### 7️⃣ **Thư Mục `docs/` - Tài Liệu**

**Codebook.csv:**

- Bộ từ điển giải thích ý nghĩa của mỗi cột (feature) trong dữ liệu
- Giúp hiểu rõ ý nghĩa của các biến
- Bao gồm: tên biến, đơn vị, định nghĩa, nguồn

---

## 📊 Nguồn Dữ Liệu

### 1. **ERA5 (ECMWF Reanalysis v5)**

- Nguồn: Copernicus Climate Data Store (CDS)
- Loại: Dữ liệu tái phân tích khí hậu toàn cầu
- Phân giải: 0.25° x 0.25° (khoảng 30km)
- Khoảng thời gian: 2015-2024
- Các biến:
  - **Single Level:** nhiệt độ 2m, điểm sương 2m, áp suất bề mặt, áp suất mực biển, gió 10m, địa thế bề mặt, mặt nạ đất--biển, v.v.
  - **Pressure Level:** nhiệt độ, độ ẩm riêng, độ cao địa thế vị, thành phần gió và vận tốc thẳng đứng ở hai mức áp suất 500 hPa và 850 hPa.
- Lưu ý: ERA5 được dùng làm nguồn đặc trưng bổ sung; biến tổng lượng mưa ERA5 không được dùng để bù hoặc thay thế PRCP từ GSOD.

### 2. **GSOD (Global Summary of the Day)**

- Nguồn: NOAA National Centers for Environmental Information
- Loại: Dữ liệu từ các trạm quan trắc trên mặt đất
- Dữ liệu: Nhiệt độ, áp suất, lượng mưa, tốc độ gió từ các trạm
- Bao phủ: Toàn cầu

---

## 🔄 Quy Trình Xử Lý Dữ Liệu

### Sơ đồ Luồng Dữ Liệu:

```
┌─────────────────────────────────────────────┐
│             NGUỒN DỮ LIỆU                   │
│        NOAA GSOD | ERA5 Single | ERA5 Press │
└────────────────────┬────────────────────────┘
                     │
                     ▼
      ┌─────────────────────────────────┐
      │  01. DATA CRAWLING              │
      │  - Tải dữ liệu từ API           │
      │  - Lưu dữ liệu thô (raw)        │
      └──────────────┬──────────────────┘
                     │
         ┌───────────┴────────────┐
         ▼                        ▼
 ┌─────────────────┐   ┌──────────────────┐
 │ 02. SINGLE LEVEL│   │03. PRESSURE LEVEL│
 │ DATA CRAWLING   │   │ DATA CRAWLING    │
 └────────┬────────┘   └──────────┬───────┘
          │                       │
          └───────────────────────┘
                     │
                     ▼
      ┌─────────────────────────────────┐
      │  04 & 05. DATA PROCESSING       │
      │  - Làm sạch dữ liệu             │
      │  - Xử lý giá trị thiếu          │
      │  - Chuyển đổi định dạng         │
      │  - Chuẩn hóa dữ liệu            │
      └──────────────┬──────────────────┘
                     │
                     ▼
      ┌─────────────────────────────────┐
      │  06. MERGE TO SILVER            │
      │  - Gộp tất cả nguồn dữ liệu     │
      │  - Tạo dữ liệu chính (silver)   │
      │  - Lưu silver_data.csv          │
      └──────────────┬──────────────────┘
                     │
                     ▼
      ┌─────────────────────────────────┐
      │  07. FEATURE ENGINEERING        │
      │  - Tạo các đặc trưng mới        │
      │  - Tính toán các chỉ số         │
      │  - Lựa chọn features            │
      └──────────────┬──────────────────┘
                     │
                     ▼
      ┌─────────────────────────────────┐
      │  08. MODEL TRAINING & EVAL      │
      │  - Huấn luyện mô hình           │
      │  - Đánh giá hiệu suất           │
      │  - Lưu kết quả                  │
      └──────────────┬──────────────────┘
                     │
                     ▼
      ┌─────────────────────────────────┐
      │       KẾT QUẢ THỰC NGHIỆM       │
      │  - Benchmark metrics            │
      │  - Model comparison results     │
      │  - Dashboard minh họa           │
      └─────────────────────────────────┘
```

---

## 🛠️ Công Nghệ Sử Dụng

### **Python Libraries (trong `requirements.txt`)**

#### **Dữ Liệu & Xử Lý:**

- `numpy==1.26.4` - Tính toán khoa học
- `pandas==2.2.2` - Xử lý dữ liệu bảng
- `scipy==1.14.1` - Tính toán khoa học nâng cao
- `pyarrow==17.0.0` - Định dạng Parquet (lưu trữ hiệu quả)
- `xarray==2024.7.0` - Xử lý dữ liệu nhiều chiều (khí hậu)
- `netCDF4==1.7.2` - Định dạng NetCDF (khí hậu)
- `cfgrib==0.9.14.1` - Đọc file GRIB (khí hậu)
- `eccodes==2.38.0` - Thư viện xử lý GRIB
- `cdsapi==0.7.3` - API tải dữ liệu từ Copernicus CDS

#### **Học Máy:**

- `scikit-learn==1.5.1` - Mô hình ML cơ bản
- `lightgbm==4.5.0` - Mô hình Gradient Boosting
- `joblib==1.4.2` - Lưu/tải mô hình

#### **Trực Quan Hóa:**

- `matplotlib==3.9.2` - Vẽ đồ thị cơ bản
- `seaborn==0.13.2` - Vẽ đồ thị thống kê
- `plotly==5.24.1` - Vẽ đồ thị interactive

#### **Web & Ứng Dụng:**

- `streamlit==1.38.0` - Xây dựng dashboard web
- `requests==2.32.3` - Gọi API HTTP

#### **Tiện Ích:**

- `python-dotenv==1.0.1` - Quản lý biến môi trường
- `tqdm==4.66.5` - Thanh tiến độ (progress bar)

---

### **Docker & Orchestration**

| Công Nghệ          | Mục Đích                                           |
| ------------------ | -------------------------------------------------- |
| **Docker**         | Đóng gói ứng dụng thành container                  |
| **Docker Compose** | Chạy nhiều container (Airflow, Streamlit) cùng lúc |
| **Apache Airflow** | Tự động hóa và lên lịch quy trình                  |

---

### **Phần Mềm & Dịch Vụ Bên Ngoài**

- **Python 3.11** - Ngôn ngữ lập trình
- **Copernicus CDS API** - Tải dữ liệu khí hậu ERA5
- **NOAA GSOD** - Dữ liệu trạm quan trắc

---

## 📚 Các Tệp Cấu Hình Quan Trọng

### **docker-compose.yaml**

- Cấu hình chạy toàn bộ hệ thống trong Docker
- Khởi chạy: Airflow Webserver, Scheduler, và Streamlit
- Cấu hình môi trường: Database, API keys, PYTHONPATH, v.v.
- Sử dụng: `docker-compose up -d`

### **Dockerfile**

- Build image cho ứng dụng Streamlit dashboard
- Sử dụng Python 3.11 slim
- Cài đặt thư viện và dependencies từ `requirements.txt`

### **Dockerfile.airflow**

- Build image cho Apache Airflow
- Dựa trên `apache/airflow:2.9.3-python3.11`
- Cài đặt dependencies khoa học (eccodes, libgomp1, v.v.)

### **.env.example**

- Template biến môi trường
- Ví dụ: `CDSAPI_URL`, `CDSAPI_KEY` (để truy cập ERA5)
- Sao chép sang `.env` và điền giá trị thực

### **requirements.txt**

- Danh sách tất cả thư viện Python cần cài
- Sử dụng: `pip install -r requirements.txt`

### **webserver_config.py**

- Cấu hình Airflow webserver
- Thiết lập: xác thực, giao diện, v.v.

---

## 🚀 Hướng Dẫn Sử Dụng

### **1. Chuẩn Bị Môi Trường**

```bash
# Clone repository
git clone <repo_url>
cd DS108

# Tạo file .env từ template
cp .env.example .env

# Cấu hình biến môi trường trong .env
# Đặc biệt: CDSAPI_URL và CDSAPI_KEY cho Copernicus CDS
```

### **2. Cài Đặt Dependencies (Local)**

```bash
# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows

# Cài đặt packages
pip install -r requirements.txt
```

### **3. Chạy Dự Án với Docker**

```bash
# Build và chạy toàn bộ hệ thống
docker-compose up -d

# Truy cập Airflow: http://localhost:8081
# Truy cập Streamlit: http://localhost:8501

# Xem log
docker-compose logs -f
```

### **4. Chạy Pipeline Cụ Thể**

```bash
# Chạy script xử lý dữ liệu
python src/_01_crawler.py
python src/_02_crawls_single_level.py
python src/_03_crawls_pressure_level.py
python src/_04_single_level.py
python src/_05_pressure_level.py
python src/_06_to_silver.py
python src/_07_feature_engineering.py
python src/_08_model.py

# Hoặc chạy tất cả qua Airflow DAG
# Truy cập http://localhost:8081 và kích hoạt DAG
```

### **5. Phân Tích Dữ Liệu**

```bash
# Chạy Jupyter
jupyter notebook notebooks/

# Mở:
# - eda.ipynb (Phân tích khám phá)
# - feature_selection.ipynb (Lựa chọn features)
```

### **6. Xem Dashboard Streamlit**

```bash
# Nếu chạy local
streamlit run demo/app.py

# Truy cập: http://localhost:8501
```

## 🎓 Nguồn Tài Liệu Liên Quan

### **Dữ Liệu Khí Hậu:**

- [Copernicus ERA5 Documentation](https://cds.climate.copernicus.eu)
- [GRIB Format Guide](https://www.ecmwf.int/en/computing/software/grib)
- [NetCDF Format](https://www.unidata.ucar.edu/software/netcdf/)

### **Tools & Libraries:**

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Xarray Documentation](https://docs.xarray.dev/)
- [Apache Airflow Guide](https://airflow.apache.org/docs/)
- [Streamlit Documentation](https://docs.streamlit.io/)

### **Machine Learning:**

- [Scikit-learn Guide](https://scikit-learn.org/stable/)
- [LightGBM Tutorial](https://lightgbm.readthedocs.io/)

---

## **Đóng Góp & Liên Hệ**

Đây là project học thuật phục vụ đồ án cuối kỳ môn DS108. Mọi góp ý về dữ liệu, pipeline hoặc tài liệu có thể được gửi thông qua Issue trên kho lưu trữ.

**Cập nhật lần cuối:** 14/06/2026

**Trạng thái:** ✅ Hoàn tất cho mục tiêu đồ án học thuật và sẵn sàng bảo vệ.
