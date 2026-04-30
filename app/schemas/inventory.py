from typing import Optional

from pydantic import BaseModel


class MaterialInwardRequest(BaseModel):
    product_id: str
    quantity: int
    sku: Optional[str] = None


class MaterialOutwardRequest(BaseModel):
    product_id: str
    quantity: int
    reason: str
    sku: Optional[str] = None


class MaterialInwardBySkuRequest(BaseModel):
    sku: str
    quantity: int


class MaterialOutwardBySkuRequest(BaseModel):
    sku: str
    quantity: int
    reason: str


class InventoryTransactionResponse(BaseModel):
    id: str
    product_id: str
    product_name: str
    product_code: str
    type: str
    quantity: int
    variant_stock_after: Optional[int]
    created_by: str
    variant_sku: Optional[str]
    stock_after: int
    created_at: str


__all__ = [
    "InventoryTransactionResponse",
    "MaterialInwardBySkuRequest",
    "MaterialInwardRequest",
    "MaterialOutwardBySkuRequest",
    "MaterialOutwardRequest",
]
