import uuid
from datetime import datetime

from sqlalchemy import Column, Date, DateTime, Float, ForeignKey, String, Text

from app.core.config import IST
from app.db import Base


class ExpenseCategoryModel(Base):
    __tablename__ = "expense_categories"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(IST))


class ExpenseModel(Base):
    __tablename__ = "expenses"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    title = Column(String(255), nullable=False)
    amount = Column(Float, nullable=False)
    category_id = Column(String(36), ForeignKey("expense_categories.id"), nullable=True)
    payment_mode = Column(String(50), nullable=False)
    description = Column(Text, nullable=True)
    expense_date = Column(Date, default=lambda: datetime.now(IST).date())
    created_by = Column(String(36), nullable=True)
    created_by_name = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(IST))


class ChequeModel(Base):
    __tablename__ = "cheques"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    cheque_number = Column(String(100), nullable=False, unique=True)
    cheque_date = Column(Date, nullable=False)
    amount = Column(Float, nullable=False)
    bank_name = Column(String(255), nullable=True)
    party_name = Column(String(255), nullable=False)
    party_type = Column(String(50), nullable=False)
    status = Column(String(50), default="pending")
    cleared_date = Column(Date, nullable=True)
    payment_mode = Column(String(50), default="cheque")
    notes = Column(Text, nullable=True)
    created_by = Column(String(36), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(IST))


class LedgerEntry(Base):
    __tablename__ = "ledger_entries"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    entry_type = Column(String(50), nullable=False, index=True)
    reference_id = Column(String(36), nullable=True)
    customer_id = Column(String(36), ForeignKey("customers.id"), nullable=True)
    supplier_id = Column(String(36), ForeignKey("suppliers.id"), nullable=True)
    description = Column(String(500), nullable=False)
    debit = Column(Float, default=0)
    credit = Column(Float, default=0)
    payment_mode = Column(String(50), nullable=False)

    cgst = Column(Float, default=0)
    sgst = Column(Float, default=0)
    igst = Column(Float, default=0)

    entry_date = Column(Date, default=lambda: datetime.now(IST).date(), index=True)
    created_by = Column(String(36), nullable=True)
    created_by_name = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(IST), index=True)


__all__ = ["ChequeModel", "ExpenseCategoryModel", "ExpenseModel", "LedgerEntry"]
