import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import mlflow.pyfunc
from mlflow import MlflowClient
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from utils.mlflow_artifact_loader import MLflowArtifactLoader

# ======================================
# 🔧 配置 MLflow 远程跟踪地址
# ======================================
# 设置远程 MLflow Tracking Server 地址（必须与运行 mlflow server 的地址一致）
# MLFLOW_TRACKING_URI = "sqlite:///mlflow_tracking/mlflow.db"
# 替换为实际地址，例如：http://your-mlflow-server:5000
MLFLOW_TRACKING_URI = "http://localhost:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# 模型注册表中的模型名称（与注册时一致）
MODEL_NAME = "HousingPriceModel"
client = MlflowClient()

# 全局变量（在 lifespan 中初始化）
model = None
encoder = None
scaler = None
expected_columns = None

# ======================================
# 🌱 使用 lifespan 管理生命周期
# ======================================
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global model, encoder, scaler, expected_columns

    print("🚀 应用启动中：正在加载模型和依赖文件...")

    try:
        # 1. 加载模型（Production 环境）
        # model_uri = f"models:/{model_name}@{model_version_alias}
        # 或 model_uri = f"models:/{model_name}@{model_version_alias}"
        model_uri = f"models:/{MODEL_NAME}@production_v1"
        print(f"尝试从 {model_uri} 加载模型...")
        model = mlflow.pyfunc.load_model(model_uri)
        print("✅ 成功加载模型 (Production)")

        run_id = model.metadata.run_id
        run = client.get_run(run_id)
        artifact_uri = run.info.artifact_uri

        encoder = MLflowArtifactLoader.load_joblib(f"{artifact_uri}/ocean_encoder.pkl")
        scaler = MLflowArtifactLoader.load_joblib(f"{artifact_uri}/scaler.pkl")
        feature_columns = MLflowArtifactLoader.load_joblib(f"{artifact_uri}/feature_columns.pkl")

        if not encoder:
            raise RuntimeError("❌ 未找到 ocean_encoder.pkl")
        if not scaler:
            raise RuntimeError("❌ 未找到 scaler.pkl")
        if not feature_columns:
            raise RuntimeError("❌ 未找到 feature_columns.pkl")

        print("✅ 所有依赖文件加载完成！")

    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)}")
        raise  # 让应用启动失败

    yield  # 应用运行

    print("🛑 应用关闭")

# ======================================
# 🚀 创建 FastAPI 应用，传入 lifespan
# ======================================
app = FastAPI(
    title="House Price Prediction API",
    version="1.0",
    lifespan=lifespan  # ✅ 使用 lifespan 而不是 on_event
)


# ======================================
# 🧱 输入数据验证模型
# ======================================
class HouseFeatures(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: str  # e.g., '<1H OCEAN'

    class Config:
        json_schema_extra = {
            "example": {
                "longitude": -122.23,
                "latitude": 37.88,
                "housing_median_age": 15,
                "total_rooms": 5612,
                "total_bedrooms": 1283,
                "population": 1015,
                "households": 478,
                "median_income": 1.4936,
                "ocean_proximity": "<1H OCEAN"
            }
        }


# ======================================
# 🎯 预测接口
# ======================================
@app.post("/predict")
def predict_price(house: HouseFeatures):
    try:
        features = {
            'longitude': house.longitude,
            'latitude': house.latitude,
            'housing_median_age': house.housing_median_age,
            'total_rooms': house.total_rooms,
            'total_bedrooms': house.total_bedrooms,
            'population': house.population,
            'households': house.households,
            'median_income': house.median_income,
            'rooms_per_household': house.total_rooms / house.households,
            'bedrooms_per_room': house.total_bedrooms / house.total_rooms,
            'population_per_household': house.population / house.households,
            'ocean_proximity': house.ocean_proximity
        }

        df = pd.DataFrame([features])
        numerical_cols = [col for col in df.columns if col != 'ocean_proximity']
        x_numerical = df[numerical_cols]
        x_categorical = df[['ocean_proximity']]

        x_categorical_encoded = encoder.transform(x_categorical)
        encoded_columns = encoder.get_feature_names_out(['ocean_proximity'])
        x_categorical_df = pd.DataFrame(x_categorical_encoded, columns=encoded_columns, index=df.index)

        x_final = pd.concat([x_numerical, x_categorical_df], axis=1)
        x_final = x_final.reindex(columns=expected_columns, fill_value=0)
        x_scaled = scaler.transform(x_final)

        prediction = model.predict(x_final)
        predicted_price = prediction[0] if len(prediction) > 0 else 0

        return {"predicted_price": round(float(predicted_price), 2)}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"预测失败: {str(e)}")


# ======================================
# 🧪 健康检查接口
# ======================================
@app.get("/healthz")
def health_check():
    return {"status": "healthy", "model_loaded": True}






# 执行结果如下：
# (.venv) PS F:\workspace\mlops-journey-2025\experiment_03\src> uvicorn app_fast:app --host 0.0.0.0 --port 9000
# INFO:     Started server process [9444]
# INFO:     Waiting for application startup.
# 🚀 应用启动中：正在加载模型和依赖文件...
# 尝试从 models:/HousingPriceModel@production_v1 加载模型...
# ✅ 成功加载模型 (Production)
# ✅ 所有依赖文件加载完成！
# INFO:     Application startup complete.
# INFO:     Uvicorn running on http://0.0.0.0:9000 (Press CTRL+C to quit)
# INFO:     127.0.0.1:4377 - "GET /healthz HTTP/1.1" 200 OK
# INFO:     127.0.0.1:4388 - "POST /healthz HTTP/1.1" 405 Method Not Allowed
# INFO:     127.0.0.1:4406 - "POST /predict HTTP/1.1" 200 OK

# 请求地址：http://localhost:9000/predict
# 请求方式： post
# 请求数据：
# {
#     "longitude": -122.23,
#     "latitude": 37.88,
#     "housing_median_age": 15,
#     "total_rooms": 5612,
#     "total_bedrooms": 1283,
#     "population": 1015,
#     "households": 478,
#     "median_income": 1.4936,
#     "ocean_proximity": "<1H OCEAN"
# }
# 返回结果：
# {
#     "predicted_price": 353772.84
# }