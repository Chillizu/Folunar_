#!/usr/bin/env python3
"""
认证模块
提供JWT token认证、密码验证等安全功能
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi import HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr

logger = logging.getLogger(__name__)

# 密码上下文
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT安全方案
security = HTTPBearer(auto_error=False)

class TokenData(BaseModel):
    """Token数据模型"""
    username: Optional[str] = None
    exp: Optional[datetime] = None

class UserCredentials(BaseModel):
    """用户凭据模型"""
    username: str
    password: str

class AuthManager:
    """认证管理器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('security', {})
        self.jwt_secret_key = self.config.get('jwt_secret_key', 'your-secret-key')
        self.jwt_algorithm = self.config.get('jwt_algorithm', 'HS256')
        self.jwt_expiration_hours = self.config.get('jwt_expiration_hours', 24)
        self.admin_username = self.config.get('admin_username', 'admin')
        self.admin_password_hash = self._hash_password(
            self.config.get('admin_password', 'admin')
        )

    def _hash_password(self, password: str) -> str:
        """哈希密码"""
        return pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """验证密码"""
        return pwd_context.verify(plain_password, hashed_password)

    def authenticate_user(self, username: str, password: str) -> bool:
        """认证用户"""
        if username == self.admin_username:
            return self.verify_password(password, self.admin_password_hash)
        return False

    def create_access_token(self, data: Dict[str, Any]) -> str:
        """创建访问token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(hours=self.jwt_expiration_hours)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.jwt_secret_key, algorithm=self.jwt_algorithm)
        return encoded_jwt

    def verify_token(self, token: str) -> Optional[TokenData]:
        """验证token"""
        try:
            payload = jwt.decode(token, self.jwt_secret_key, algorithms=[self.jwt_algorithm])
            username: str = payload.get("sub")
            exp = payload.get("exp")
            if exp:
                exp = datetime.fromtimestamp(exp)
            if username is None:
                return None
            return TokenData(username=username, exp=exp)
        except JWTError:
            return None

async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    auth_manager: Optional[AuthManager] = None
) -> Optional[str]:
    """获取当前用户"""
    if not credentials:
        return None

    token_data = auth_manager.verify_token(credentials.credentials) if auth_manager else None
    if token_data is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的认证凭据",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return token_data.username

def require_auth(current_user: Optional[str] = Depends(get_current_user)) -> str:
    """需要认证的依赖"""
    if current_user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="需要认证",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return current_user</content>
</xai:function_call">### 2. 实现输入验证

现在我需要创建输入验证模块。让我创建一个 `src/validation.py` 文件来处理输入验证。 

<xai:function_call name="write_to_file">
<parameter name="path">src/validation.py