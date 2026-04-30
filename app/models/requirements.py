import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, String, Text

from app.core.config import IST
from app.db import Base


class RequirementModel(Base):
    __tablename__ = "requirements"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    customer_name = Column(String(255), nullable=False)
    customer_phone = Column(String(50), nullable=False)
    requirement_items = Column(Text, nullable=False)
    priority = Column(String(20), default="normal")
    status = Column(String(20), default="pending", index=True)
    created_by = Column(String(36))
    created_by_name = Column(String(100))
    created_at = Column(DateTime, default=lambda: datetime.now(IST), index=True)
    completed_at = Column(DateTime, nullable=True)


__all__ = ["RequirementModel"]
