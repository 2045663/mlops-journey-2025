# from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,OneHotEncoder
import joblib
import pandas as pd
import os

def load_data():
    """加载数据集"""
    # housing = fetch_california_housing(as_frame=True)
    # df = housing.frame  # 自动包含 target 列
    df = pd.read_csv("experiment_01/data/housing.csv")
    return df

def preprocess_data(df, test_size=0.2, random_state=42):
    """数据预处理：划分训练/测试集，标准化"""
    x = df.drop('median_house_value', axis=1).copy()
    y = df['median_house_value']

    # 特征工程：构造有意义的比率特征
    x['rooms_per_household'] = x['total_rooms'] / x['households']
    x['bedrooms_per_room'] = x['total_bedrooms'] / x['total_rooms']
    x['population_per_household'] = x['population'] / x['households']

    # 处理类别特征：对 ocean_proximity 进行独热编码
    # 分离数值特征和类别特征
    numerical_features = x.select_dtypes(include=['float64', 'int64']).columns.tolist()
    categorical_features = ['ocean_proximity']
    # 从数值特征中移除已构造的比率（避免重复） ocean_proximity 已被排除，我们只处理它
    x_numerical = x[numerical_features].copy()
    x_categorical = x[categorical_features].copy()
    # 初始化 OneHotEncoder
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop=None)
    # 拟合并转换类别特征
    x_categorical_encoded = encoder.fit_transform(x_categorical)
    # 获取 one-hot 编码后的列名
    encoded_columns = encoder.get_feature_names_out(categorical_features)
    # 数据格式如下：
    # 0,ocean_proximity_<1H OCEAN
    # 1,ocean_proximity_INLAND
    # 2,ocean_proximity_ISLAND
    # 3,ocean_proximity_NEAR BAY
    # 4,ocean_proximity_NEAR OCEAN

    # 转换为 DataFrame，便于拼接
    x_categorical_df = pd.DataFrame(x_categorical_encoded, columns=encoded_columns, index=x.index)
    # 数据格式如下：
    #    ocean_proximity_<1H OCEAN` ocean_proximity_INLAND``ocean_proximity_ISLAND  ocean_proximity_NEAR BAY  ocean_proximity_NEAR OCEAN
    # 0             0                       0                           1                   0                       0
    # 1             0                       0                           0                   1                       0
    # 2             0                       0                           1                   0                       0
    # 3             1                       0                           0                   0                       0

    # 拼接数值特征和编码后的类别特征
    x_encoded = pd.concat([x_numerical, x_categorical_df], axis=1)
    # 数据格式如下：
    #    longitude   latitude  ... ocean_proximity_<1H OCEAN` ocean_proximity_INLAND``ocean_proximity_ISLAND  ocean_proximity_NEAR BAY  ocean_proximity_NEAR OCEAN
    # 0   -122.23    37.88     ...           0                           1                   0                       0                        0
    # 1   -122.23    37.88     ...           0                           0                   1                      0                        0
    # 2   -122.23    37.88     ...           0                           1                   0                       0                        0

    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(
        x_encoded, y,test_size=test_size, random_state=random_state
    )

    # 特征标准化
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    # 保存训练时的特征列顺序
    os.makedirs("experiment_01/models", exist_ok=True)
    joblib.dump(encoder, "experiment_01/models/ocean_encoder.pkl")
    joblib.dump(scaler, "experiment_01/models/scaler.pkl")
    joblib.dump(x_train.columns.tolist(), "experiment_01/models/feature_columns.pkl")

    return x_train_scaled, x_test_scaled, y_train, y_test

if __name__ == '__main__':
    df = load_data()
    print(df.head())
    print(f"数据形状：{df.shape}")
    print(f"特征: {list(df.columns)}")







