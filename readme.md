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


1、创建目录: mkdir -p experiments
2、启动 MLflow UI:  
   #mlflow server --host 127.0.0.1 --port 8080 
   mlflow ui --backend-store-uri sqlite:///experiment_02/mlflow_tracking/mlflow.db
3、运训训练脚本: python mlflow_tracking.py
4、查看实验记录: 打开浏览器访问 http://localhost:8080，查看实验记录
5、多次运行，对比不同参数: 修改 n_estimators 和 max_depth，重新运行脚本，观察 UI 中的对比视图


使用 Gitbash 进入命令行 执行 make 
