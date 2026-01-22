from pydantic import BaseModel
from typing import Optional, List

class Product(BaseModel):
    id: str
    product_code: str
    name: str
    description: Optional[str] = None

    category_id: str
    category_name: Optional[str] = None

    cost_price: Optional[float] = None
    min_selling_price: float
    selling_price: float

    stock: int
    min_stock: int

    is_service: int = 0

    sku: str
    image_url: Optional[str] = None
    images: Optional[List[str]] = []

    qr_code_url: Optional[str] = None
    created_at: str


class ProductCreate(BaseModel):
    name: str
    description: Optional[str] = None
    category_id: str

    cost_price: float
    min_selling_price: float
    selling_price: float

    stock: int
    min_stock: int = 5

    is_service: int = 0

    sku: Optional[str] = None
    image_url: Optional[str] = None
    images: Optional[List[str]] = []  # max 5
