from src import data
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from mlflow.models import infer_signature
from mlflow.entities import ViewType
import numpy as np

# 设置实验名称
# TRACKING_URI = "sqlite:///experiment_02/mlflow_tracking/mlflow.db"
TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "housing-price-experiment"
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

run_name = "housing_price_rf_test"
artifact_path = "rf_housing_price"

# 加载数据
df = data.load_data()
# 数据预处理
x_train, x_test, y_train, y_test = data.preprocess_data(df)

params = {
    "n_estimators":150,
    "max_depth":9,
    "min_samples_split":10,
    "min_samples_leaf":4,
    "bootstrap":True,
    "oob_score":False,
    "random_state":42
}

# 模型训练
rf = RandomForestRegressor(**params)
rf.fit(x_train, y_train)

# 模型预测
y_pred = rf.predict(x_test)

# 模型评估
mae = mean_absolute_error(y_test,y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test,y_pred)

metrics = {"mae":mae, "mse":mse, "rmse":rmse, "r2":r2}

# ===== 第一步：记录本次运行 =====
with mlflow.start_run(run_name=run_name) as run:
    # 记录模型训练参数
    mlflow.log_params(params)
    # 记录验证过程中计算的错误指标
    mlflow.log_metrics(metrics)
    # 调用 infer_signature 函数，生产签名对象
    signature = infer_signature(x_train[:1], y_pred)
    # 记录模型
    mlflow.sklearn.log_model(
        rf, name=artifact_path,signature=signature,input_example=x_train[:1] # 提供一个输入样例
    )
    print(f"✅ MLflow Run ID: {mlflow.active_run().info.run_id}")


# ===== 第二步：搜索最佳模型并注册 =====
import time
time.sleep(10)
# 获取最佳模型: mlflow.search_runs() 来查找表现最好的 run_id ✅
best_model = mlflow.search_runs(
    experiment_names = [EXPERIMENT_NAME],
    run_view_type = ViewType.ACTIVE_ONLY,
    filter_string = "metrics.mse < 3800000000 and metrics.r2 > 0.71",
    order_by=["metrics.mse ASC", "metrics.r2 DESC"]
)
# 检查是否有结果
if len(best_model) == 0:
    raise ValueError("⚠️ 没有符合条件的运行记录.")
# 获取最佳 run 的 ID
best_run_id = best_model.iloc[0]["run_id"]
print(f"Best Run ID: {best_run_id}")
# 构造模型在该 run 中的路径
model_uri = f"runs:/{best_run_id}/{artifact_path}"
# 注册模型：
mlflow.register_model(model_uri = model_uri, name = "HousingPriceModel")
