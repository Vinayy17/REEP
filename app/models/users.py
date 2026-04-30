import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, String

from app.core.config import IST
from app.db import Base


class UserModel(Base):
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password = Column(String(255), nullable=False)
    role = Column(String(50), default="user")
    created_at = Column(DateTime, default=lambda: datetime.now(IST))


__all__ = ["UserModel"]
