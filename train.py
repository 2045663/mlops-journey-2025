import joblib
import os
from src import data, model, evaluate

def main():
    print("🚀 开始训练房价预测模型...")

    # 1、加载数据
    df = data.load_data()

    # 2、预处理
    x_train,x_test, y_train, y_test, scaler =  data.preprocess_data(df)
    print(f"数据预处理完成，训练集：{x_train.shape}, 测试集：{x_test.shape}")

    # 3、训练模型
    rf_model = model.train_random_forest(x_train, y_train)
    print("随机森林模型训练完成")

    # 4、评估
    evaluate.evaluate_model(rf_model, x_test, y_test)

    # 5、保存模型和标准化器
    os.makedirs("models", exist_ok=True)
    joblib.dump(rf_model, "models/rf_model.pkl")
    joblib.dump(scaler, "models/scaler.pkl")
    print("模型和标准化器已保存到 models/ 目录")

    print("🎉 训练流程完成！")


if __name__ == '__main__':
    main()

# 运行结果如下：
# 🚀 开始训练房价预测模型...
# 数据预处理完成，训练集：(16512, 16), 测试集：(4128, 16)
# 随机森林模型训练完成
# 模型评估结果：
#   RMSE: 49796.9006
#   MAE: 31853.1733
#   R²: 0.8108
# 模型和标准化器已保存到 models/ 目录
# 🎉 训练流程完成！