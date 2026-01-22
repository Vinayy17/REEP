from pydantic import BaseModel
from typing import Optional

class MaterialInwardRequest(BaseModel):
    product_id: str
    quantity: int

class MaterialOutwardRequest(BaseModel):
    product_id: str
    quantity: int
    reason: str

class InventoryTransactionResponse(BaseModel):
    id: str
    product_id: str
    product_name: str
    product_code: str
    type: str
    quantity: int
    source: Optional[str]
    reason: Optional[str]
    created_by: Optional[str]
    created_at: str
    remaining_stock: int
