import pandas as pd
from fastapi import FastAPI, HTTPException
import joblib
from pydantic import BaseModel

# 加载模型 和 scaler
model = joblib.load("experiment_01/models/rf_model.pkl")
encoder = joblib.load("experiment_01/models/ocean_encoder.pkl")
scaler = joblib.load("experiment_01/models/scaler.pkl")
expected_columns = joblib.load("experiment_01/models/feature_columns.pkl")
app = FastAPI(title="House Price Prediction")

class HouseFeatures(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    # median_house_value: float
    ocean_proximity: str   # 例如 '<1H OCEAN'

@app.post("/predict")
def predict_price(house: HouseFeatures):
    try:
        # 构造原始特征
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

        # 转为 DataFrame
        df = pd.DataFrame([features])
        # 提取数值和类别特征
        numerical_cols = [col for col in df.columns if col != 'ocean_proximity']
        x_numerical = df[numerical_cols]
        x_categorical = df[['ocean_proximity']]

        # 使用 OneHotEncoder 编码（自动处理未知类别）
        x_categorical_encoded = encoder.transform(x_categorical)
        encoded_columns = encoder.get_feature_names_out(['ocean_proximity'])
        x_categorical_df = pd.DataFrame(x_categorical_encoded, columns=encoded_columns, index=df.index)

        # 拼接
        x_final = pd.concat([x_numerical, x_categorical_df], axis=1)

        # 确保列顺序与训练时一致
        x_final = x_final.reindex(columns=expected_columns, fill_value=0)

        # 标准化
        x_scaled = scaler.transform(x_final)

        # 预测
        prediction = model.predict(x_scaled)[0]

        return {"predicted_price": round(prediction, 2)}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"预测失败: {str(e)}")

