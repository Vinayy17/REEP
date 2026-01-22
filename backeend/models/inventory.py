import uuid
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey
from datetime import datetime, timezone, timedelta
from database.base import Base

IST = timezone(timedelta(hours=5, minutes=30))

class InventoryTransaction(Base):
    __tablename__ = "inventory_transactions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    product_id = Column(String(36), ForeignKey("products.id"))

    type = Column(String(3))  # IN / OUT
    quantity = Column(Integer)

    source = Column(String(50))
    reason = Column(String(255))
    created_by = Column(String(36))

    stock_before = Column(Integer, default=0)
    stock_after = Column(Integer, default=0)

    created_at = Column(DateTime, default=lambda: datetime.now(IST))
