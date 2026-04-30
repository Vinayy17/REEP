from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class Category(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    created_at: str


class CategoryCreate(BaseModel):
    name: str
    description: Optional[str] = None


class VariantSchema(BaseModel):
    v_sku: str
    variant_name: Optional[str] = None
    color: Optional[str] = None
    size: Optional[str] = None
    stock: int = 0
    image_url: Optional[str] = None
    qr_code_url: Optional[str] = None
    image: Optional[str] = None
    v_selling_price: Optional[float] = None


class Product(BaseModel):
    id: str
    product_code: str
    sku: str
    name: str
    description: Optional[str] = None
    category_id: str
    cost_price: float
    selling_price: float
    min_selling_price: float
    stock: int
    min_stock: int
    is_service: int
    variants: List[VariantSchema] = Field(default_factory=list)
    images: Optional[List[str]] = Field(default_factory=list)
    qr_code_url: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


class ProductCreate(BaseModel):
    name: str
    description: Optional[str] = None
    category_id: str
    cost_price: float = 0.0
    selling_price: float
    min_selling_price: float
    min_stock: int = 0
    sku: str
    images: List[str] = Field(default_factory=list)
    is_service: int = 0
    variants: List[VariantSchema] = Field(default_factory=list)
    qr_code_url: Optional[str] = None


__all__ = ["Category", "CategoryCreate", "Product", "ProductCreate", "VariantSchema"]
