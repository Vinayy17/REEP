from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class InvoiceItem(BaseModel):
    product_id: Optional[str] = None
    product_name: str
    quantity: int
    price: float
    gst_rate: float
    sku: Optional[str] = None
    is_service: int = 0
    variant_info: Optional[dict] = None
    image_url: Optional[str] = None


class AdditionalCharge(BaseModel):
    label: Optional[str] = None
    amount: Optional[float] = 0

    @field_validator("amount", mode="before")
    @classmethod
    def allow_empty_amount(cls, value):
        if value in ("", None):
            return 0
        return float(value)


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
    customer_email: Optional[str] = None
    customer_address: Optional[str] = None
    items: List[InvoiceItem]
    gst_enabled: bool = True
    gst_rate: float = 0
    discount: float = 0
    payment_status: str = "pending"
    payment_mode: Optional[str] = "cash"
    additional_charges: List[AdditionalCharge] = Field(default_factory=list)


class SalesChartItem(BaseModel):
    name: str
    total: float
    paid: float
    pending: float
    partial: float


class InvoiceStatusUpdate(BaseModel):
    payment_status: str


class DraftFinalizeRequest(BaseModel):
    payment_status: Optional[str] = None
    paid_amount: Optional[float] = 0


class AddPaymentRequest(BaseModel):
    amount: float
    payment_mode: str
    reference: Optional[str] = None


__all__ = [
    "AdditionalCharge",
    "AddPaymentRequest",
    "DraftFinalizeRequest",
    "Invoice",
    "InvoiceCreate",
    "InvoiceItem",
    "InvoiceStatusUpdate",
    "SalesChartItem",
]
