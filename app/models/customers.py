import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, String, Text
from sqlalchemy.orm import relationship

from app.core.config import IST
from app.db import Base


class CustomerModel(Base):
    __tablename__ = "customers"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    email = Column(String(255), nullable=False)
    phone = Column(String(50), nullable=True, index=True)
    address = Column(Text, nullable=True)
    opening_balance = Column(Float, default=0)
    current_balance = Column(Float, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now(IST))

    invoices = relationship("InvoiceModel", back_populates="customer")


class SupplierModel(Base):
    __tablename__ = "suppliers"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    phone = Column(String(50), nullable=True)
    email = Column(String(255), nullable=True)
    address = Column(Text, nullable=True)
    opening_balance = Column(Float, default=0)
    current_balance = Column(Float, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now(IST))


__all__ = ["CustomerModel", "SupplierModel"]
