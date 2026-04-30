import uuid
from datetime import datetime

from sqlalchemy import JSON, Column, DateTime, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import relationship

from app.core.config import IST
from app.db import Base


class InvoicePayment(Base):
    __tablename__ = "invoice_payments"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    invoice_id = Column(String(36), ForeignKey("invoices.id"), index=True)
    amount = Column(Float, nullable=False)
    payment_mode = Column(String(50), nullable=False)
    reference = Column(String(100), nullable=True)
    created_by = Column(String(36), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(IST))


class InvoiceModel(Base):
    __tablename__ = "invoices"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    invoice_number = Column(String(50), nullable=True, unique=True, index=True)
    draft_number = Column(String(50), nullable=True)
    invoice_type = Column(String(20), default="FINAL", index=True)

    customer_id = Column(String(36), ForeignKey("customers.id"), nullable=False)
    customer_name = Column(String(255), nullable=False)
    customer_phone = Column(String(50), nullable=True)
    customer_address = Column(Text, nullable=True)

    items = Column(Text, nullable=False)

    subtotal = Column(Float, nullable=False)
    gst_enabled = Column(Integer, default=1)
    gst_rate = Column(Float, default=0)
    gst_amount = Column(Float, nullable=False, default=0)
    discount = Column(Float, nullable=False, default=0)
    additional_charges = Column(JSON, default=[])
    total = Column(Float, nullable=False)

    paid_amount = Column(Float, default=0)
    balance_amount = Column(Float, default=0)
    payment_mode = Column(String(50), default="cash")
    payment_status = Column(String(50), nullable=False, default="pending", index=True)

    created_by = Column(String(36))
    created_by_name = Column(String(100))
    created_at = Column(DateTime, default=lambda: datetime.now(IST), index=True)

    customer = relationship("CustomerModel", back_populates="invoices")

    __table_args__ = (
        Index("idx_invoice_type_status", "invoice_type", "payment_status"),
        Index("idx_invoice_created", "created_at"),
    )


__all__ = ["InvoiceModel", "InvoicePayment"]
