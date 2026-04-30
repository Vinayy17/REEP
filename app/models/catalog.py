import uuid
from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship

from app.core.config import IST
from app.db import Base


class CategoryModel(Base):
    __tablename__ = "categories"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(IST))

    products = relationship("ProductModel", back_populates="category")


class ProductModel(Base):
    __tablename__ = "products"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    product_code = Column(String(50), nullable=False, unique=True, index=True)
    sku = Column(String(100), nullable=False, unique=True, index=True)

    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)

    category_id = Column(String(36), ForeignKey("categories.id"), nullable=False)

    cost_price = Column(Float, nullable=False, default=0)
    min_selling_price = Column(Float, nullable=False)
    selling_price = Column(Float, nullable=False)

    stock = Column(Integer, nullable=False, default=0)
    min_stock = Column(Integer, nullable=False, default=5)

    variants = Column(JSON, nullable=False)

    qr_code_url = Column(String(255), nullable=True)
    images = Column(JSON, nullable=True)

    is_service = Column(Integer, default=0)
    is_active = Column(Integer, default=1)

    created_at = Column(DateTime, default=lambda: datetime.now(IST))

    category = relationship("CategoryModel", back_populates="products")


__all__ = ["CategoryModel", "ProductModel"]
