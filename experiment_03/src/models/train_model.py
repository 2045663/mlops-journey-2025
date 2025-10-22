"""
训练随机森林回归模型
"""
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import joblib
import os

def train_model():
    print("🧠 正在训练模型...")
    x = pd.read_csv("data/processed/x_train.csv")
    y = pd.read_csv("data/processed/y_train.csv").values.ravel()

    model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(x, y)

    # 保存模型
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, 'models/rf_model.pkl')
    print("✅ 模型已保存至 models/rf_model.pkl")

if __name__ == '__main__':
    train_model()