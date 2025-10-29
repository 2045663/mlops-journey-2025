from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ..config.settings import settings

class MiddlewareManager:
    """CORS 中间件管理类"""

    @staticmethod
    def setup_cors(app: FastAPI):
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.ALLOWED_ORIGINS,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
            # 更安全的做法：只允许必要 headers
            # allow_headers=["Authorization", "Content-Type"],
        )
        print(f"✅ CORS 已启用，允许来源: {settings.ALLOWED_ORIGINS}")


# 实例化
middleware_manager = MiddlewareManager()