# config/settings.py
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    # 🌐 CORS 配置
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:9000",
        "http://127.0.0.1:9000"
    ]

    # 🔐 JWT 配置
    JWT_SECRET_KEY: str = "my-super-secret-jwt-key-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # 🛠️ 应用配置
    DEBUG: bool = True
    ENVIRONMENT: str = "development"

    class Config:
        env_file = ".env"  # 支持从 .env 文件加载
        case_sensitive = True

# 实例化全局配置
settings = Settings()