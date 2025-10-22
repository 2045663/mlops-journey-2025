"""
模型评估：计算 MSE、R² 等指标
"""
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
import json
import os


def evaluate_model():
    print("📊 正在评估模型...")
    model = joblib.load('models/rf_model.pkl')

    x_test = pd.read_csv('data/processed/x_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()

    y_pred = model.predict(x_test)

    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # 保存评估结果
    metrics = {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2}
    os.makedirs("reports", exist_ok=True)
    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"✅ 评估完成 | MSE: {mse:.2f} | MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.4f}")
    return metrics


if __name__ == "__main__":
    evaluate_model()