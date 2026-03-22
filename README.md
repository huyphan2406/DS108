# 📊 DS108 - Thu thập dữ liệu khí tượng tại Đồng bằng sông Cửu Long (2015–2025) và tiền xử lý cho bài toán dự báo lượng mưa

A comprehensive machine learning project for collecting meteorological data in the Mekong Delta region for the period 2015–2025 and preprocessing it for the rainfall forecasting problem.

## 🎯 Giới thiệu:

Dự án sử dụng nguồn dữ liệu khí tượng từ NASA POWER với mục đích thu thập, làm sạch và tiền xử lý dữ liệu thời tiết phục vụ cho bài toán dự đoán lượng mưa tại Đồng bằng Sông Cửu Long (Việt Nam) trong giai đoạn 2015-2025.

Mục tiêu chính của dự án là xây dựng một pipeline xử lý dữ liệu hoàn chỉnh trước khi áp dụng các mô hình học máy.

## 📊 Dữ liệu:

- **Nguồn:** NASA POWER
- **Vị trí:**
  <div align="center">

  | STT |    Tỉnh    | Kinh độ (lon) | Vĩ độ (lat) |
  | :-: | :--------: | :-----------: | :---------: |
  | 01  |   Cà Mau   |   104.9826    |   9.0851    |
  | 02  |  Bạc Liêu  |   105.4168    |   9.2916    |
  | 03  | Sóc Trăng  |   105.8827    |   9.5064    |
  | 04  | Hậu Giang  |   105.5985    |   9.7735    |
  | 05  |  Trà Vinh  |   106.2405    |   9.8082    |
  | 06  | Kiên Giang |   105.0969    |   9.8460    |
  | 07  | Vĩnh Long  |   105.9318    |   10.1317   |
  | 08  |  Bến Tre   |   106.4671    |   10.1491   |
  | 09  |  Cần Thơ   |   105.4432    |   10.1765   |
  | 10  | Tiền Giang |   106.1655    |   10.3960   |
  | 11  |  An Giang  |   104.8877    |   10.4355   |
  | 12  | Đồng Tháp  |   105.4738    |   10.6866   |
  | 13  |  Long An   |   105.9748    |   10.7068   |

  </div>

- **Khoảng thời gian:** Ngày 01/01/2015 - Ngày 31/12/2025
- **Số lượng mẫu:** ~52.235 mẫu theo ngày.
- **Đặc trưng dữ liệu:** ~15 biến khí tượng.
- **Nhãn đầu ra:** Lượng mưa mỗi ngày theo đơn vị mm/ngày

## 📂 Cây thư mục

```
DS108/
│
├── artifacts/            # Chứa models đã train
├── configs/              # Chứa tham số lat, lon, các giá trị quan trọng để truyền
├── data/                 # Chứa dữ liệu
│   ├── bronze/           # Dữ liệu raw
│   ├── silver/           # Dữ liệu đã qua xử lý
│   └── gold/
├── docs/                 # Chứa báo cáo như report và slide
│   ├── report/
│   └── slide/
├── notebooks/            # Chứa các file jupyter notebook
├── src/                  # Chứa source code
│   ├── ai/
│   ├── app/
│   ├── etl/
│   └── ml/
├── tests/                # Chứa những file kiểm thử
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Quy trình thực hiện

## Kết quả thực nghiệm
