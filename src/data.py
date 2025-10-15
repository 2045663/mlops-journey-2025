# from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

def load_data():
    """加载数据集"""
    # housing = fetch_california_housing(as_frame=True)
    # df = housing.frame  # 自动包含 target 列
    df = pd.read_csv("data/housing.csv")
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
    x = pd.get_dummies(x, columns=['ocean_proximity'], prefix='ocean', dtype=int)

    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(
        x, y,test_size=test_size, random_state=random_state
    )

    # 特征标准化
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    return x_train_scaled, x_test_scaled, y_train, y_test, scaler

if __name__ == '__main__':
    df = load_data()
    print(df.head())
    print(f"数据形状：{df.shape}")
    print(f"特征: {list(df.columns)}")







