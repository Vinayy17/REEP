from datetime import date
from typing import Optional

from pydantic import BaseModel


class ExpenseCategoryCreate(BaseModel):
    name: str
    description: Optional[str] = None


class ExpenseCategory(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    created_at: str


class ExpenseRequest(BaseModel):
    title: str
    amount: float
    category_id: Optional[str] = None
    payment_mode: str
    description: Optional[str] = None
    expense_date: Optional[date] = None


class PaymentInRequest(BaseModel):
    customer_id: str
    amount: float
    payment_mode: str
    reference: Optional[str] = None
    payment_date: Optional[date] = None
    cheque_number: Optional[str] = None
    cheque_date: Optional[date] = None
    bank_name: Optional[str] = None


class PaymentOutRequest(BaseModel):
    supplier_id: Optional[str] = None
    supplier_name: str
    amount: float
    payment_mode: str
    description: Optional[str] = None
    payment_date: Optional[date] = None
    cheque_number: Optional[str] = None
    cheque_date: Optional[date] = None
    bank_name: Optional[str] = None


class ChequeCreate(BaseModel):
    cheque_number: str
    cheque_date: date
    amount: float
    bank_name: Optional[str] = None
    party_name: str
    party_type: str
    notes: Optional[str] = None


class ChequeStatusUpdate(BaseModel):
    status: str
    cleared_date: Optional[date] = None


__all__ = [
    "ChequeCreate",
    "ChequeStatusUpdate",
    "ExpenseCategory",
    "ExpenseCategoryCreate",
    "ExpenseRequest",
    "PaymentInRequest",
    "PaymentOutRequest",
]
