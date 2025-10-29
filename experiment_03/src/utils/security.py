from datetime import datetime, timedelta
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from ..config.settings import settings

# 🔐 OAuth2 密码流（用于获取 token）
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token")


class JWTManager:
    """JWT 工具类"""

    @staticmethod
    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
        to_encode = data.copy()
        expire = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
        return encoded_jwt

    @staticmethod
    def decode_token(token: str):
        try:
            payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
            return payload
        except JWTError:
            return None

    @staticmethod
    def verify_token(token: str = Depends(oauth2_scheme)):
        """依赖注入用的验证函数"""
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="❌ 无效或过期的令牌",
            headers={"WWW-Authenticate": "Bearer"},
        )
        payload = JWTManager.decode_token(token)
        if payload is None:
            raise credentials_exception
        return payload  # 可用于获取用户信息


# 实例化
jwt_manager = JWTManager()