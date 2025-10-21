#Dockerfile
FROM python:3.12

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

#安装系统依赖
RUN apt-get update && apt-install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \

# 安装python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制模型和代码
COPY models/ models/
COPY app.py .

# 暴露端口
EXPOSE 8000

# 启动服务
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
