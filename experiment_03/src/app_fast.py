import pandas as pd
from fastapi import FastAPI, HTTPException, Depends
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
import mlflow
from mlflow import MlflowClient
from contextlib import asynccontextmanager

# 导入工具类
from .utils.mlflow_artifact_loader import MLflowArtifactLoader
from .utils.middleware import middleware_manager
from .utils.security import jwt_manager
from .utils.exceptions import validation_exception_handler, general_exception_handler

# ======================================
# 🔧 MLflow 配置
# ======================================
MLFLOW_TRACKING_URI = "http://localhost:5555"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
MODEL_NAME = "HousingPriceModel"
client = MlflowClient()

# 全局变量
model = None
encoder = None
scaler = None
expected_columns = None

# ======================================
# 🌱 生命周期管理
# ======================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, encoder, scaler, expected_columns
    print("🚀 应用启动中：加载模型...")

    try:
        model_uri = f"models:/{MODEL_NAME}@production_v1"
        model = mlflow.pyfunc.load_model(model_uri)
        print("✅ 模型加载成功")

        run_id = model.metadata.run_id
        run = client.get_run(run_id)
        artifact_uri = run.info.artifact_uri

        encoder = MLflowArtifactLoader.load_joblib(f"{artifact_uri}/ocean_encoder.pkl")
        scaler = MLflowArtifactLoader.load_joblib(f"{artifact_uri}/scaler.pkl")
        feature_columns = MLflowArtifactLoader.load_joblib(f"{artifact_uri}/feature_columns.pkl")

        expected_columns = feature_columns
        print("✅ 依赖文件加载完成")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        raise

    yield
    print("🛑 应用关闭")

# ======================================
# 🚀 创建应用
# ======================================
app = FastAPI(
    title="House Price Prediction API",
    version="1.0",
    lifespan=lifespan
)

# 🔌 注册中间件
middleware_manager.setup_cors(app)

# 🔌 注册异常处理器
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

# ======================================
# 🔐 认证接口：获取 JWT Token
# ======================================
@app.post("/token")
def login_for_access_token():
    # 实际项目中应验证用户名密码
    token = jwt_manager.create_access_token(data={"sub": "service-account"})
    return {"access_token": token, "token_type": "bearer"}

# ======================================
# 🧱 输入模型
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
    ocean_proximity: str

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
# 🎯 预测接口（JWT 保护）
# ======================================
@app.post("/predict")
def predict_price(
    house: HouseFeatures,
    payload: dict = Depends(jwt_manager.verify_token)  # 🔐 JWT 验证
):
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

        prediction = model.predict(x_final)
        predicted_price = prediction[0] if len(prediction) > 0 else 0

        return {"predicted_price": round(float(predicted_price), 2)}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"预测失败: {str(e)}")

# ======================================
# 🧪 健康检查
# ======================================
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# ======================================
# 🎉 根路由
# ======================================
@app.get("/")
def root():
    return {"message": "Welcome to House Price Prediction API"}