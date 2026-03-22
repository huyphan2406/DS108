import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# Tự động xác định đường dẫn gốc DS108
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = BASE_DIR / "models" / "weather_model_xgb.pkl"


def load_trained_model():
    """Nạp file model .pkl từ thư mục models"""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"⚠️ Không tìm thấy model tại: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def make_ai_prediction(model, input_df):
    """
    Nhận model và DataFrame đầu vào (phải đủ 44 cột).
    Trả về con số lượng mưa dự báo (mm).
    """
    # Thực hiện dự báo
    prediction = model.predict(input_df)

    # Ép ngưỡng: Lượng mưa không bao giờ âm
    result = np.clip(prediction[0], 0, None)
    return float(result)