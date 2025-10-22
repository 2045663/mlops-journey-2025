"""
下载加州房价数据集并保存为 CSV
"""
from sklearn.datasets import fetch_california_housing
import pandas as pd
import os

def fetch_housing_data():
    print("📥 正在下载加州房价数据...")
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)

    # 创建目录
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv('data/raw/housing.csv',index=False)
    print("✅ 数据已保存至 data/raw/housing.csv")

if __name__ == '__main__':
    fetch_housing_data()