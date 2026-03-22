import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import warnings
from pathlib import Path

# Thư viện Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')


# =====================================================================
# 1. CẤU HÌNH ĐƯỜNG DẪN (PATH MANAGEMENT)
# =====================================================================
def setup_paths():
    try:
        base_dir = Path(__file__).resolve().parent.parent.parent
    except NameError:
        base_dir = Path.cwd().parent.parent

    paths = {
        "gold_data": base_dir / "data" / "gold" / "master_data_gold.csv",
        "model_dir": base_dir / "models",
        "report_dir": base_dir / "docs" / "report"
    }

    # Tạo thư mục nếu chưa có
    paths["model_dir"].mkdir(parents=True, exist_ok=True)
    paths["report_dir"].mkdir(parents=True, exist_ok=True)

    return paths


# =====================================================================
# 2. XỬ LÝ DỮ LIỆU (DATA ENGINE)
# =====================================================================
def prepare_data(file_path):
    print(f"📂 Đang nạp dữ liệu từ: {file_path.name}...")
    df = pd.read_csv(file_path)

    # Các cột loại bỏ (Target và Metadata)
    drop_cols = ['date', 'year', 'target_precip_tomorrow', 'target_is_rain_tomorrow']
    X = df.drop(columns=drop_cols)
    y = df['target_precip_tomorrow']

    # Time Series Split: 80% đầu để học, 20% cuối để kiểm tra
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    return X_train, X_test, y_train, y_test


# =====================================================================
# 3. HUẤN LUYỆN & ĐÁNH GIÁ (MODEL ENGINE)
# =====================================================================
def train_and_evaluate(X_train, X_test, y_train, y_test):
    # Định nghĩa danh sách các mô hình
    model_pool = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42)
    }

    summary_results = []
    trained_models = {}

    print("\n⚙️ Bắt đầu huấn luyện...")
    print("-" * 75)
    print(f"{'Mô hình':<20} | {'MAE':<10} | {'RMSE':<10} | {'R2 Score':<10}")
    print("-" * 75)

    for name, model in model_pool.items():
        # Train
        model.fit(X_train, y_train)

        # Predict & Clip (Mưa không được âm)
        preds = np.clip(model.predict(X_test), 0, None)

        # Metrics
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        # Lưu kết quả
        trained_models[name] = {"model": model, "preds": preds, "mae": mae, "r2": r2}
        summary_results.append({"Model": name, "MAE": mae, "RMSE": rmse, "R2": r2})

        print(f"{name:<20} | {mae:<10.2f} | {rmse:<10.2f} | {r2:<10.2f}")

    return trained_models, pd.DataFrame(summary_results)


# =====================================================================
# 4. TRỰC QUAN HÓA (VISUALIZATION ENGINE)
# =====================================================================
def plot_dashboard(y_test, results, save_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    colors = ['#4A90E2', '#F5A623', '#7ED321', '#9013FE']

    # Lấy giá trị max để vẽ đường chéo chuẩn
    all_preds = [m['preds'].max() for m in results.values()]
    max_val = max(y_test.max(), max(all_preds))

    for i, (name, metrics) in enumerate(results.items()):
        ax = axes[i]
        ax.scatter(y_test, metrics['preds'], color=colors[i], alpha=0.4, s=25, edgecolor='w')

        # Đường chéo lý tưởng
        ax.plot([0, max_val], [0, max_val], 'r--', lw=2, label="Lý tưởng (y=x)")

        # Trang trí
        ax.set_title(f"{name}\nMAE: {metrics['mae']:.2f}mm | $R^2$: {metrics['r2']:.2f}", fontweight='bold')
        ax.set_xlabel("Thực tế (mm)")
        ax.set_ylabel("Dự đoán (mm)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("Hệ thống Đánh giá Hiệu suất Mô hình Dự báo Lượng mưa", fontsize=18, y=0.98, fontweight='black')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(save_path, dpi=300)
    print(f"\n📸 Đã lưu biểu đồ tại: {save_path}")
    plt.show()


# =====================================================================
# 5. CHƯƠNG TRÌNH CHÍNH (MAIN EXECUTION)
# =====================================================================
if __name__ == "__main__":
    # B1: Setup
    paths = setup_paths()

    # B2: Prepare Data
    X_train, X_test, y_train, y_test = prepare_data(paths["gold_data"])

    # B3: Train & Eval
    results_dict, df_metrics = train_and_evaluate(X_train, X_test, y_train, y_test)

    # B4: Lưu mô hình tốt nhất (Dựa trên MAE thấp nhất)
    best_model_name = df_metrics.sort_values("MAE").iloc[0]["Model"]
    joblib.dump(results_dict[best_model_name]["model"], paths["model_dir"] / "weather_model_xgb.pkl")

    # Lưu bảng metrics ra CSV để làm báo cáo
    df_metrics.to_csv(paths["model_dir"] / "model_performance.csv", index=False)

    print(f"\n🏆 Mô hình tốt nhất: {best_model_name}")
    print(f"💾 Đã lưu artifact tại folder: {paths['model_dir'].name}")

    # B5: Vẽ biểu đồ
    plot_dashboard(y_test, results_dict, paths["report_dir"] / "final_evaluation_dashboard.png")