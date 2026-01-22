import uuid
from sqlalchemy import (
    Column, String, Text, Float, Integer, ForeignKey, DateTime
)
from sqlalchemy.orm import relationship
from sqlalchemy import JSON
from datetime import datetime, timezone, timedelta
from database.base import Base

IST = timezone(timedelta(hours=5, minutes=30))

class ProductModel(Base):
    __tablename__ = "products"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    product_code = Column(String(50), nullable=False, unique=True, index=True)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)

    category_id = Column(String(36), ForeignKey("categories.id"), nullable=False)

    # Pricing
    cost_price = Column(Float, nullable=False)
    min_selling_price = Column(Float, nullable=False)
    selling_price = Column(Float, nullable=False)

    qr_code_url = Column(String(255), nullable=True)
    images = Column(JSON, nullable=True)

    # Stock
    stock = Column(Integer, default=0)
    min_stock = Column(Integer, default=5)

    sku = Column(String(100), nullable=False, unique=True)
    image_url = Column(String(500), nullable=True)

    is_service = Column(Integer, default=0)  # ✅ SERVICE FLAG

    created_at = Column(DateTime, default=lambda: datetime.now(IST))

    category = relationship("CategoryModel", back_populates="products")
