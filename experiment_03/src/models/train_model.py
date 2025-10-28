"""
训练随机森林回归模型
"""
from sklearn.ensemble import RandomForestRegressor
import mlflow
from mlflow.models import infer_signature
import pandas as pd
import joblib
import argparse
import os

# 配置 MLflow
TRACKING_URI = "sqlite:///mlflow_tracking/mlflow.db"
EXPERIMENT_NAME = "housing-price-experiment"
RUN_NAME = "housing_price_rf_test"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

def train_model(n_estimators=100, max_depth=5):
    print("🧠 正在训练模型...")

    params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "min_samples_split": 10,
        "min_samples_leaf": 4,
        "bootstrap": True,
        "oob_score": False,
        "random_state": 42
    }

    x_train = pd.read_csv("data/processed/x_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").values.ravel()

    model = RandomForestRegressor(**params)
    model.fit(x_train, y_train)

    # 保存模型
    os.makedirs("models", exist_ok=True)
    model_filename = f"rf_model_n{n_estimators}_d{max_depth}.pkl" # 按参数命名
    model_path = os.path.join("models", model_filename)
    joblib.dump(model, model_path)
    print(f"✅ 模型已保存至 models/{model_filename}")

    with mlflow.start_run(run_name=RUN_NAME) as run: # 获取 run_id，便于后续关联
        # 1.记录模型训练参数
        mlflow.log_params(params)

        # 2.记录其他关键资产作为 artifacts
        mlflow.log_artifact("models/ocean_encoder.pkl")
        mlflow.log_artifact("models/scaler.pkl")
        mlflow.log_artifact("models/feature_columns.pkl")

        # 调用 infer_signature 函数，生产签名对象
        signature = infer_signature(x_train, model.predict(x_train))
        # 3.记录模型
        artifact_path = f"rf_housing_price_n{n_estimators}_d{max_depth}"  # 当前实验 run 记录列表中Models字段值
        mlflow.sklearn.log_model(
            model, name=artifact_path, signature=signature, input_example=x_train[:1]  # 提供一个输入样例
        )
        print(f"✅ MLflow Run ID: {mlflow.active_run().info.run_id}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=5)
    args = parser.parse_args()

    train_model(args.n_estimators, args.max_depth)