# 📊 DS108 - Thu thập dữ liệu khí tượng tại Đồng bằng sông Cửu Long (2015–2025) và tiền xử lý cho bài toán dự báo lượng mưa

A comprehensive machine learning project for collecting meteorological data in the Mekong Delta region for the period 2015–2025 and preprocessing it for the rainfall forecasting problem.

## 🎯 Giới thiệu:

Dự án sử dụng nguồn dữ liệu khí tượng từ Open-Meteo với mục đích thu thập, làm sạch và tiền xử lý dữ liệu thời tiết phục vụ cho bài toán dự đoán lượng mưa tại Đồng bằng Sông Cửu Long (Việt Nam) trong giai đoạn 2015-2025.

Mục tiêu chính của dự án là xây dựng một pipeline xử lý dữ liệu hoàn chỉnh trước khi áp dụng các mô hình học máy.

## 📊 Dữ liệu:

- Nguồn: Open-Meteo
- Vị trí:
    <div align="center">

  | STT | Tỉnh       |  Kinh độ  |   Vĩ độ   |
  | :-: | :--------- | :-------: | :-------: |
  | 01  | An Giang   | Giá trị 2 | Giá trị 3 |
  | 02  | Bạc Liêu   | Giá trị 2 | Giá trị 3 |
  | 03  | Bến Tre    | Giá trị 5 | Giá trị 6 |
  | 04  | Cà Mau     | Giá trị 2 | Giá trị 3 |
  | 05  | Cần Thơ    | Giá trị 5 | Giá trị 6 |
  | 06  | Đồng Tháp  | Giá trị 2 | Giá trị 3 |
  | 07  | Hậu Giang  | Giá trị 5 | Giá trị 6 |
  | 08  | Kiên Giang | Giá trị 5 | Giá trị 6 |
  | 09  | Long An    | Giá trị 2 | Giá trị 3 |
  | 10  | Tiền Giang | Giá trị 5 | Giá trị 6 |
  | 11  | Trà Vinh   | Giá trị 5 | Giá trị 6 |
  | 12  | Vĩnh Long  | Giá trị 5 | Giá trị 6 |
  | 13  | Sóc Trăng  | Giá trị 5 | Giá trị 6 |

    </p>

- Khoảng thời gian: Ngày 01/01/2015 - Ngày 31/12/2025
- Samples: 9,252+ daily observations
- Đặc trưng dữ liệu: ~10 biến khí tượng.
- Target: Daily precipitation (PRECTOTCORR) in mm/day
