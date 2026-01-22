from pydantic import BaseModel
from typing import List, Optional

class InvoiceItem(BaseModel):
    product_id: str
    product_name: str
    quantity: int
    price: float
    gst_rate: float
    total: float
    sku: Optional[str] = None


class Invoice(BaseModel):
    id: str
    invoice_number: str
    customer_id: str
    customer_name: str
    customer_phone: Optional[str] = None
    customer_address: Optional[str] = None

    items: List[InvoiceItem]

    subtotal: float
    gst_amount: float
    discount: float
    total: float

    payment_status: str
    created_at: str


class InvoiceCreate(BaseModel):
    customer_id: Optional[str] = None
    customer_name: str
    customer_phone: Optional[str] = None
    customer_email: str
    customer_address: Optional[str] = None

    items: List[InvoiceItem]

    gst_amount: float = 0
    discount: float = 0
    payment_status: str = "pending"


class InvoiceStatusUpdate(BaseModel):
    payment_status: str
