import uuid
from datetime import datetime

from sqlalchemy import Column, DateTime, ForeignKey, Integer, String

from app.core.config import IST
from app.db import Base


class InventoryTransaction(Base):
    __tablename__ = "inventory_transactions"

    id = Column(String(36), primary_key=True)
    product_id = Column(String(36), ForeignKey("products.id"))
    type = Column(String(3))
    quantity = Column(Integer)
    source = Column(String(50))
    reason = Column(String(255))
    created_by = Column(String(36))
    created_at = Column(DateTime, default=lambda: datetime.now(IST))
    created_by_name = Column(String(100))

    stock_before = Column(Integer, nullable=False, default=0)
    stock_after = Column(Integer, nullable=False, default=0)
    variant_stock_after = Column(Integer, nullable=True)
    variant_sku = Column(String(100), nullable=True)


__all__ = ["InventoryTransaction"]
