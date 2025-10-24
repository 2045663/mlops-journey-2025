构建并运行 Docker 容器
1. 构建镜像: docker build -t house-price-api .
2. 运行容器: docker run -d -p 8000:8000 --name house-price-container house-price-api
3. 查看日志: docker logs house-price-container

测试 API
curl -X POST "http://localhost:8000/predict" \
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


experiment_02 实验：

1、创建目录: mkdir -p experiments
2、启动 MLflow UI:  
   #mlflow server --host 127.0.0.1 --port 8080 
   mlflow ui --backend-store-uri sqlite:///experiment_02/mlflow_tracking/mlflow.db
3、运训训练脚本: python mlflow_tracking.py
4、查看实验记录: 打开浏览器访问 http://localhost:8080，查看实验记录
5、多次运行，对比不同参数: 修改 n_estimators 和 max_depth，重新运行脚本，观察 UI 中的对比视图

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
2. 在 experiment_03 目录下创建 mlflow_tracking 目录： mkdir mlflow_tracking
3. 执行命令：mlflow ui --backend-store-uri sqlite:///mlflow_tracking/mlflow.db
使用 Gitbash 进入命令行 执行 make命令：
(1)默认参数训练 + 评估：make all
(2)指定参数训练： make model N_ESTIMATORS=120 MAX_DEPTH=6
(3)指定参数评估：make evaluate N_ESTIMATORS=120 MAX_DEPTH=6
(4)参数扫描（训练+评估所有组合）：make sweep
(5)跳过 data 步骤（假设数据已存在）：
	make model SKIP_DATA=true
   会检查 data 目标，但内部脚本不执行 make_dataset.py

(6)跳过 data 和 features（假设特征已存在）:
	make model SKIP_DATA=true SKIP_FEATURES=true
   保留依赖链完整性（Make 仍会“进入”features）,但实际不执行任何操作，直接训练模型

(7)结合参数使用 make model N_ESTIMATORS=150 MAX_DEPTH=7 SKIP_DATA=true SKIP_FEATURES=true

默认参数：
make features SKIP_DATA=true
make model SKIP_DATA=true SKIP_FEATURES=true
make evaluate SKIP_DATA=true SKIP_FEATURES=true SKIP_MODEL=true

组合参数：
make model N_ESTIMATORS=80 MAX_DEPTH=3 SKIP_DATA=true SKIP_FEATURES=true
make evaluate N_ESTIMATORS=80 MAX_DEPTH=3 SKIP_DATA=true SKIP_FEATURES=true SKIP_MODEL=true

全流程
make all N_ESTIMATORS=60 MAX_DEPTH=4 SKIP_DATA=true

多参数扫描训练
make sweep SKIP_DATA=true SKIP_FEATURES=true
