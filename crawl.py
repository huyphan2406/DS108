import pandas as pd
import os
import requests
from io import StringIO

os.makedirs("data", exist_ok=True)
# latitudes  = [9.0851, 9.2916, 9.5064, 9.7735, 9.8082, 9.8460, 10.1317, 10.1491, 10.1765, 10.3960, 10.4355, 10.6866, 10.7068]
# longitudes  = [104.9826, 105.4168, 105.8827, 105.5985, 106.2405, 105.0969, 105.9318, 106.4671, 105.4432, 106.1655, 104.8877, 105.4738, 105.9748]

parameters = ','.join(["T2M", "RH2M", "PS", "CLOUD_AMT", "WS10M", "WD10M", "PRECTOTCORR"])

for year in range(2015, 2026):
    start = f"{year}0101"
    end = f"{year}1231"

    url = f"https://power.larc.nasa.gov/api/temporal/hourly/point?parameters={parameters}&community=RE&longitude=104.9826&latitude=9.0851&start={start}&end={end}&format=csv&header=true"
    print(f"Downloading data for {year}...")

    try:
        response = requests.get(url)
        text = response.text

        # Tìm dòng bắt đầu bằng YEAR
        lines = text.split("\n")
        start_index = next(i for i, line in enumerate(lines) if line.startswith("YEAR"))

        clean_csv = "\n".join(lines[start_index:])
        df = pd.read_csv(StringIO(clean_csv))

        # Tạo datetime index
        
        df = df.rename(columns={
            "YEAR": "year",
            "MO": "month",
            "DY": "day",
            "HR": "hour"
        })

        df["time"] = pd.to_datetime(df[["year", "month", "day", "hour"]])

        df = df.drop(columns=["year", "month", "day", "hour"])
        df = df.set_index("time")

        filename = f"data/nasa_{year}.csv"
        df.to_csv(filename)

        print(f"Saved -> {filename}")

    except Exception as e:
        print("Error:", e)
