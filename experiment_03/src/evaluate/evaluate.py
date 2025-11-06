"""
模型评估：计算 MSE、R² 等指标，并将指标记录到 MLflow
"""
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import mlflow
import joblib
import json
import os

# 配置 MLflow
TRACKING_URI = "http://localhost:5555"
EXPERIMENT_NAME = "housing-price-experiment"
RUN_NAME = "housing_price_rf_test"

mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)


def find_run_by_params(n_estimators, max_depth):
    """根据参数查找对应的 MLflow Run"""
    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        raise ValueError(f"实验 '{EXPERIMENT_NAME}' 不存在")

    # 构建搜索表达式
    search_filter = (
        f"params.n_estimators = '{n_estimators}' "
        f"and params.max_depth = '{max_depth}' "
        f"and tags.mlflow.runName = '{RUN_NAME}' "
    )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=search_filter,
        max_results=1,
        order_by=["attributes.start_time DESC"]
    )

    if len(runs) == 0:
        raise ValueError(f"未找到 n_estimators={n_estimators}, max_depth={max_depth} 的训练记录")

    return runs[0]


def evaluate_model(n_estimators=100, max_depth=5):
    print(f"📊 正在评估模型: n_estimators={n_estimators}, max_depth={max_depth}")

    # ✅ 加载本地模型（按参数命名）
    model_filename = f"rf_model_n{n_estimators}_d{max_depth}.pkl"
    model_path = os.path.join("models", model_filename)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件 {model_path} 不存在，请训练后再进行评估！")

    model = joblib.load(model_path)

    # 加载测试数据
    x_test = pd.read_csv('data/processed/x_test.csv')
    y_test = pd.read_csv('data/processed/y_test.csv').values.ravel()

    # 预测
    y_pred = model.predict(x_test)

    # 计算指标
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    metrics = {"mse": mse, "mae": mae, "rmse": rmse, "r2": r2}

    # ✅ 在 MLflow 中记录指标（关联到训练 Run）
    try:
        run = find_run_by_params(n_estimators, max_depth)
        with mlflow.start_run(run_id=run.info.run_id):
            mlflow.log_metrics(metrics)
            mlflow.set_tag("evaluation", "test_set")
            mlflow.log_param("eval_dataset", "test_set_v1")
        print(f"✅ 指标已记录到 MLflow Run ID: {run.info.run_id}")
    except Exception as e:
        print(f"⚠️ 无法记录到 MLflow: {e}")

    # 保存本地报告
    os.makedirs("reports", exist_ok=True)
    with open(f"reports/metrics_n{n_estimators}_d{max_depth}.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"✅ 评估完成 | MSE: {mse:.2f} | MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.4f}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=5)
    args = parser.parse_args()

    evaluate_model(args.n_estimators, args.max_depth)