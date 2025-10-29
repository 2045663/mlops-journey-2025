# ======================================
# 项目统一入口，用于启动 FastAPI 应用
# ======================================

from src.app_fast import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9000)