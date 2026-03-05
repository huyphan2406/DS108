import pandas as pd
import numpy as np

def check_nasa_data(path):

    print("="*60)
    print("📂 Loading file:", path)

    df = pd.read_csv(path, index_col=0, parse_dates=True)

    print("\n🔎 1. Thông tin tổng quát")
    print("-"*40)
    print(df.info())

    print("\n📊 2. Thống kê mô tả")
    print("-"*40)
    print(df.describe())

    print("\n📅 3. Kiểm tra khoảng thời gian")
    print("-"*40)
    print("Start:", df.index.min())
    print("End  :", df.index.max())
    print("Total rows:", len(df))

    print("\n🕒 4. Kiểm tra số giờ kỳ vọng")
    print("-"*40)
    expected_hours = int((df.index.max() - df.index.min()).total_seconds() / 3600) + 1
    print("Expected hours:", expected_hours)
    print("Actual hours  :", len(df))
    print("Missing hours :", expected_hours - len(df))

    print("\n❌ 5. Kiểm tra missing values (NaN)")
    print("-"*40)
    print(df.isna().sum())

    print("\n🚨 6. Kiểm tra giá trị -999 (NASA missing flag)")
    print("-"*40)
    print((df == -999).sum())

    print("\n🔁 7. Kiểm tra timestamp trùng")
    print("-"*40)
    duplicates = df.index.duplicated().sum()
    print("Duplicate timestamps:", duplicates)

    print("\n📈 8. Kiểm tra outlier đơn giản")
    print("-"*40)
    for col in df.columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        print(f"{col}: {outliers} outliers")

    print("="*60)

    return df

df = check_nasa_data("data/nasa_2015.csv")