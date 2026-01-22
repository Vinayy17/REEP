import uuid
from sqlalchemy import Column, String, Text, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime, timezone, timedelta
from database.base import Base

IST = timezone(timedelta(hours=5, minutes=30))

class InvoiceModel(Base):
    __tablename__ = "invoices"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    invoice_number = Column(String(50), unique=True, index=True, nullable=False)

    customer_id = Column(String(36), ForeignKey("customers.id"), nullable=False)
    customer_name = Column(String(255), nullable=False)
    customer_phone = Column(String(50))
    customer_address = Column(Text)

    items = Column(Text, nullable=False)  # JSON string
    subtotal = Column(Float, nullable=False)
    gst_amount = Column(Float, default=0)
    discount = Column(Float, default=0)
    total = Column(Float, nullable=False)

    payment_status = Column(String(50), default="pending")
    created_at = Column(DateTime, default=lambda: datetime.now(IST))

    customer = relationship("CustomerModel", back_populates="invoices")
