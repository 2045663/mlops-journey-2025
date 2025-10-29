构建并运行 Docker 容器
1. 构建镜像: docker build -t house-price-api .
2. 运行容器: docker run -d -p 9000:9000 --name house-price-container house-price-api
3. 查看日志: docker logs house-price-container

测试 API
curl -X POST "http://localhost:9000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "longitude": -122.23,
           "latitude": 37.88,
           "housing_median_age": 15,
           "total_rooms": 5612,
           "total_bedrooms": 1283,
           "population": 1015,
           "households": 478,
           "median_income": 1.4936,
           "ocean_proximity": "<1H OCEAN"
         }'

验证 Docker 镜像
1、查看镜像:  docker images | grep house-price
2、查看运行中的容器: docker ps
3、进入容器（可选）: docker exec -it house-price-container /bin/bash


experiment_01 实验：
1. cd experiment_01
2. python train.py 或 python app.py


experiment_02 实验：
1.cd experiment_02
2、创建目录: mkdir -p mlflow_tracking
3、启动 MLflow UI:  
   #mlflow server --host 127.0.0.1 --port 8080 
   mlflow ui --backend-store-uri sqlite:///mlflow_tracking/mlflow.db
4、运训训练脚本: python mlflow_tracking.py
5、查看实验记录: 打开浏览器访问 http://localhost:8080，查看实验记录
6、多次运行，对比不同参数: 修改 n_estimators 和 max_depth，重新运行脚本，观察 UI 中的对比视图


experiment_03 实验：
项目目录结构如下：
mlops-journey-2025/
├── .venv/                    #虚拟环境
├── experiment_03/
│	├── data/
│	│   ├── raw/              # 原始数据
│	│   └── processed/        # 处理后数据
│	├── models/                # 训练好的模型
│	├── reports/               # 评估报告
│	├── src/
│	│   ├── data/
│	│   │   └── make_dataset.py     # 下载/加载数据
│	│   ├── features/
│	│   │   └── build_features.py   # 特征工程
│	│   ├── models/
│	│   │   └── train_model.py      # 模型训练
│	│   └── evaluate/
│	│       └── evaluate.py          # 模型评估
│	│ 
│	└── Makefile                      # 自动化脚本
└─ requirements.txt                    # 依赖文件

1. 进入目录： cd experiment_03
   2. 在 experiment_03 目录下创建 mlflow_tracking 目录：
   mkdir mlflow_tracking
3. 执行命令：
   mlflow ui --backend-store-uri sqlite:///mlflow_tracking/mlflow.db
   或
   mlflow server \
       --backend-store-uri sqlite:///mlflow_tracking/mlflow.db \
       --default-artifact-root file:///path/to/artifacts \  # 或 s3://my-bucket/mlflow
       --host 0.0.0.0 \
       --port 5000
4. 使用 Gitbash 进入命令行 执行 make命令：
    (1)默认参数训练 + 评估：
        make features SKIP_DATA=true
        make model SKIP_DATA=true SKIP_FEATURES=true
        make evaluate SKIP_DATA=true SKIP_FEATURES=true SKIP_MODEL=true
        make all SKIP_DATA=true
    (2)指定参数训练： 
        make model N_ESTIMATORS=120 MAX_DEPTH=6
        make model N_ESTIMATORS=80 MAX_DEPTH=3 SKIP_DATA=true SKIP_FEATURES=true
        make evaluate N_ESTIMATORS=80 MAX_DEPTH=3 SKIP_DATA=true SKIP_FEATURES=true SKIP_MODEL=true
        make all N_ESTIMATORS=60 MAX_DEPTH=4 SKIP_DATA=true
    (3)参数扫描（训练+评估所有组合）：
        make sweep SKIP_DATA=true SKIP_FEATURES=true
    (4)跳过 data 和 features（假设特征已存在）:
        make model SKIP_DATA=true SKIP_FEATURES=true
4. 访问 MLflow 注册模型:
    访问 http://localhost:5000/#/  
    注册模型：HousingPriceModel
    设置模型版本别名：production_v1
5. 运行app:
    python main.py
6. 打开chrome浏览器(其它浏览器可能无法打开 swagger ui)访问 http://localhost:9000/docs
   6.1. 点击右上角的 "Authorize" 按钮 
   6.2. 填写用户名和密码（用于获取 token） ,虽然你在代码中没有使用真实用户名密码，
      但 FastAPI 的 OAuth2 流程需要这些字段来触发 /token 接口。
            在 username: 输入框填入任意字符串（比如 admin）
            在 password: 输入框填入任意字符串（比如 123456）
      ⚠️ 注意：这不会真正验证账号密码，只是触发 /token 接口。你实际的逻辑是在 /token 接口中直接返回 token。
   6.3. 设置客户端凭证位置为 "Authorization header"
   6.4. 点击 "Authorize" 按钮

或者通过postman测试：
获取 Token：curl -X POST "http://localhost:8000/token"
响应：
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.xxxxx",
  "token_type": "bearer"
}

8. 调用预测（带 JWT）
    现在通过swagger ui：
        点击 /predict 接口
        输入 JSON 数据
        点击 "Execute"
    此时请求会自动带上：Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.xxxxx

或者通过postman测试：
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.xxxxx" \
     -d '{
  "longitude": -122.23,
  ...
}'
