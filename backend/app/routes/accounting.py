import csv
import math
import uuid
from datetime import date, datetime
from io import StringIO
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import and_, func, or_
from sqlalchemy.orm import Session

from app.core.config import IST
from app.db import get_db
from app.models import (
    ChequeModel,
    CustomerModel,
    ExpenseCategoryModel,
    ExpenseModel,
    InvoiceModel,
    LedgerEntry,
    SupplierModel,
    UserModel,
)
from app.schemas.customers import Customer, CustomerCreate, SupplierCreate
from app.schemas.finance import (
    ChequeStatusUpdate,
    ExpenseCategory,
    ExpenseCategoryCreate,
    ExpenseRequest,
    PaymentInRequest,
    PaymentOutRequest,
)
from app.utils import get_current_user

router = APIRouter()


def _older_ledger_entries_filter(boundary_entry: LedgerEntry):
    return or_(
        LedgerEntry.entry_date < boundary_entry.entry_date,
        and_(
            LedgerEntry.entry_date == boundary_entry.entry_date,
            LedgerEntry.created_at < boundary_entry.created_at,
        ),
        and_(
            LedgerEntry.entry_date == boundary_entry.entry_date,
            LedgerEntry.created_at == boundary_entry.created_at,
            LedgerEntry.id < boundary_entry.id,
        ),
    )


@router.post("/api/customers", response_model=Customer)
def create_customer(
    customer_data: CustomerCreate,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    new_customer = CustomerModel(
        name=customer_data.name,
        email=customer_data.email,
        phone=customer_data.phone,
        address=customer_data.address,
        created_at=datetime.now(IST),
    )
    db.add(new_customer)
    db.commit()
    db.refresh(new_customer)

    return Customer(
        id=new_customer.id,
        name=new_customer.name,
        email=new_customer.email,
        phone=new_customer.phone,
        address=new_customer.address,
        created_at=new_customer.created_at.isoformat(),
    )


@router.get("/api/customers")
def get_customers(
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    customers = db.query(CustomerModel).all()
    return [
        {
            "id": c.id,
            "name": c.name,
            "email": c.email,
            "phone": c.phone,
            "address": c.address,
            "current_balance": c.current_balance,
            "created_at": c.created_at.isoformat(),
        }
        for c in customers
    ]


@router.get("/api/customers/search", response_model=Optional[Customer])
def search_customer_by_phone(
    phone: str,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    customer = db.query(CustomerModel).filter(CustomerModel.phone == phone).first()
    if not customer:
        return None

    return Customer(
        id=customer.id,
        name=customer.name,
        email=customer.email,
        phone=customer.phone,
        address=customer.address,
        created_at=customer.created_at.isoformat(),
    )


@router.get("/api/accounts/summary")
def accounts_summary(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(LedgerEntry)

    if start_date:
        query = query.filter(LedgerEntry.entry_date >= start_date)
    if end_date:
        query = query.filter(LedgerEntry.entry_date <= end_date)

    cash_entries = query.filter(LedgerEntry.payment_mode == "cash").all()
    cash_balance = 0
    for e in cash_entries:
        cash_balance += e.credit - e.debit

    bank_entries = query.filter(LedgerEntry.payment_mode.in_(["bank", "upi", "cheque"])) .all()
    bank_balance = 0
    for e in bank_entries:
        bank_balance += e.credit - e.debit

    customers_outstanding = db.query(func.sum(CustomerModel.current_balance)).scalar() or 0
    suppliers_outstanding = db.query(func.sum(SupplierModel.current_balance)).scalar() or 0
    pending_cheques = db.query(func.sum(ChequeModel.amount)).filter(ChequeModel.status == "pending").scalar() or 0

    return {
        "cash_balance": float(cash_balance),
        "bank_balance": float(bank_balance),
        "total_balance": float(cash_balance + bank_balance),
        "customers_outstanding": float(customers_outstanding),
        "suppliers_outstanding": float(suppliers_outstanding),
        "pending_cheques": float(pending_cheques),
    }


@router.get("/api/suppliers")
def get_suppliers(
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    suppliers = db.query(SupplierModel).all()
    return [
        {
            "id": s.id,
            "name": s.name,
            "phone": s.phone,
            "email": s.email,
            "address": s.address,
            "current_balance": s.current_balance,
            "created_at": s.created_at.isoformat(),
        }
        for s in suppliers
    ]


@router.post("/api/suppliers")
def create_supplier(
    data: SupplierCreate,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    opening_balance = float(data.opening_balance or 0)
    supplier = SupplierModel(
        name=data.name,
        phone=data.phone,
        email=data.email,
        address=data.address,
        opening_balance=opening_balance,
        current_balance=opening_balance,
        created_at=datetime.now(IST),
    )

    db.add(supplier)
    db.commit()
    db.refresh(supplier)

    return {"message": "Supplier created successfully", "id": supplier.id}


@router.get("/api/accounts/cashbook")
def cashbook(
    page: int = 1,
    limit: int = 50,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    role: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    if current_user.role not in ["admin", "store_handler"]:
        raise HTTPException(status_code=403, detail="Access restricted")

    base_query = db.query(LedgerEntry).filter(LedgerEntry.payment_mode == "cash")

    if start_date:
        base_query = base_query.filter(LedgerEntry.entry_date >= start_date)
    if end_date:
        base_query = base_query.filter(LedgerEntry.entry_date <= end_date)

    total = base_query.count()
    rows = (
        base_query.order_by(LedgerEntry.entry_date.desc(), LedgerEntry.created_at.desc(), LedgerEntry.id.desc())
        .offset((page - 1) * limit)
        .limit(limit)
        .all()
    )

    opening_balance = 0
    if rows:
        opening_balance = (
            base_query.filter(_older_ledger_entries_filter(rows[-1]))
            .with_entities(func.coalesce(func.sum(LedgerEntry.credit - LedgerEntry.debit), 0))
            .scalar()
            or 0
        )

    balance = opening_balance
    data = []

    for r in reversed(rows):
        balance += r.credit - r.debit
        data.insert(
            0,
            {
                "id": r.id,
                "date": r.entry_date.isoformat() if r.entry_date else None,
                "description": r.description,
                "debit": r.debit,
                "credit": r.credit,
                "balance": balance,
                "type": r.entry_type,
                "created_at": r.created_at.isoformat(),
            },
        )

    return {
        "data": data,
        "pagination": {"page": page, "limit": limit, "total": total, "total_pages": math.ceil(total / limit)},
    }


@router.get("/api/accounts/bankbook")
def bankbook(
    page: int = 1,
    limit: int = 50,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    base_query = db.query(LedgerEntry).filter(
        or_(LedgerEntry.payment_mode == "bank", LedgerEntry.payment_mode == "upi", LedgerEntry.payment_mode == "cheque")
    )

    if start_date:
        base_query = base_query.filter(LedgerEntry.entry_date >= start_date)
    if end_date:
        base_query = base_query.filter(LedgerEntry.entry_date <= end_date)

    total = base_query.count()
    rows = (
        base_query.order_by(LedgerEntry.entry_date.desc(), LedgerEntry.created_at.desc(), LedgerEntry.id.desc())
        .offset((page - 1) * limit)
        .limit(limit)
        .all()
    )

    opening_balance = 0
    if rows:
        opening_balance = (
            base_query.filter(_older_ledger_entries_filter(rows[-1]))
            .with_entities(func.coalesce(func.sum(LedgerEntry.credit - LedgerEntry.debit), 0))
            .scalar()
            or 0
        )

    balance = opening_balance
    data = []

    for r in reversed(rows):
        balance += r.credit - r.debit
        data.insert(
            0,
            {
                "id": r.id,
                "date": r.entry_date.isoformat() if r.entry_date else None,
                "description": r.description,
                "debit": r.debit,
                "credit": r.credit,
                "balance": balance,
                "type": r.entry_type,
                "mode": r.payment_mode,
                "created_at": r.created_at.isoformat(),
            },
        )

    return {
        "data": data,
        "pagination": {"page": page, "limit": limit, "total": total, "total_pages": math.ceil(total / limit)},
    }


@router.get("/api/accounts/gst-ledger")
def gst_ledger(
    page: int = 1,
    limit: int = 50,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    query = db.query(InvoiceModel).filter(
        InvoiceModel.invoice_type == "FINAL",
        InvoiceModel.payment_status == "paid",
        InvoiceModel.gst_amount > 0,
    )

    if start_date:
        query = query.filter(func.date(InvoiceModel.created_at) >= start_date)
    if end_date:
        query = query.filter(func.date(InvoiceModel.created_at) <= end_date)

    total = query.count()
    invoices = query.order_by(InvoiceModel.created_at.desc()).offset((page - 1) * limit).limit(limit).all()

    return {
        "data": [
            {
                "id": inv.id,
                "invoice": inv.invoice_number,
                "customer": inv.customer_name,
                "taxable_amount": inv.subtotal,
                "gst_amount": inv.gst_amount,
                "cgst": getattr(inv, "cgst_amount", None) or (inv.gst_amount / 2 if inv.gst_amount > 0 else 0),
                "sgst": getattr(inv, "sgst_amount", None) or (inv.gst_amount / 2 if inv.gst_amount > 0 else 0),
                "igst": getattr(inv, "igst_amount", None) or 0,
                "total": inv.total,
                "date": inv.created_at.date().isoformat(),
            }
            for inv in invoices
        ],
        "pagination": {"page": page, "limit": limit, "total": total, "total_pages": math.ceil(total / limit)},
    }


@router.get("/api/accounts/profit-loss")
def profit_loss(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    query_income = db.query(func.sum(InvoiceModel.total)).filter(InvoiceModel.invoice_type == "FINAL")
    query_expense = db.query(func.sum(ExpenseModel.amount))

    if start_date:
        query_income = query_income.filter(func.date(InvoiceModel.created_at) >= start_date)
        query_expense = query_expense.filter(ExpenseModel.expense_date >= start_date)
    if end_date:
        query_income = query_income.filter(func.date(InvoiceModel.created_at) <= end_date)
        query_expense = query_expense.filter(ExpenseModel.expense_date <= end_date)

    income = query_income.scalar() or 0
    expenses = query_expense.scalar() or 0

    return {"income": float(income), "expense": float(expenses), "profit": float(income - expenses)}


@router.post("/api/accounts/payment-in")
def payment_in(
    data: PaymentInRequest,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    customer = db.query(CustomerModel).filter(CustomerModel.id == data.customer_id).first()
    if not customer:
        raise HTTPException(404, "Customer not found")

    payment_date = data.payment_date or datetime.now(IST).date()
    customer.current_balance -= data.amount

    if data.payment_mode == "cheque" and data.cheque_number:
        cheque = ChequeModel(
            cheque_number=data.cheque_number,
            cheque_date=data.cheque_date or payment_date,
            amount=data.amount,
            bank_name=data.bank_name,
            party_name=customer.name,
            party_type="customer",
            status="pending",
            payment_mode="cheque",
            created_by=current_user.id,
            created_at=datetime.now(IST),
        )
        db.add(cheque)

    db.add(
        LedgerEntry(
            id=str(uuid.uuid4()),
            entry_type="payment_in",
            reference_id=data.customer_id,
            customer_id=data.customer_id,
            description=f"Payment received from {customer.name} {data.reference or ''}",
            debit=0,
            credit=data.amount,
            payment_mode=data.payment_mode,
            entry_date=payment_date,
            created_by=current_user.id,
            created_by_name=current_user.name,
            created_at=datetime.now(IST),
        )
    )

    db.commit()

    return {"message": "Payment recorded", "amount": data.amount, "new_balance": customer.current_balance}


@router.post("/api/accounts/payment-out")
def payment_out(
    data: PaymentOutRequest,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    payment_date = data.payment_date or datetime.now(IST).date()

    supplier = None
    if data.supplier_id:
        supplier = db.query(SupplierModel).filter(SupplierModel.id == data.supplier_id).first()
        if supplier:
            supplier.current_balance = float(supplier.current_balance or 0) - float(data.amount)

    if data.payment_mode == "cheque" and data.cheque_number:
        cheque = ChequeModel(
            cheque_number=data.cheque_number,
            cheque_date=data.cheque_date or payment_date,
            amount=data.amount,
            bank_name=data.bank_name,
            party_name=data.supplier_name,
            party_type="supplier",
            status="pending",
            payment_mode="cheque",
            created_by=current_user.id,
            created_at=datetime.now(IST),
        )
        db.add(cheque)

    db.add(
        LedgerEntry(
            id=str(uuid.uuid4()),
            entry_type="payment_out",
            reference_id=data.supplier_id,
            supplier_id=data.supplier_id,
            description=data.description or f"Payment to {data.supplier_name}",
            debit=data.amount,
            credit=0,
            payment_mode=data.payment_mode,
            entry_date=payment_date,
            created_by=current_user.id,
            created_by_name=current_user.name,
            created_at=datetime.now(IST),
        )
    )

    db.commit()

    return {"message": "Payment recorded successfully", "amount": data.amount}


@router.get("/api/expense-categories", response_model=list[ExpenseCategory])
def get_expense_categories(
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    categories = db.query(ExpenseCategoryModel).all()
    return [
        ExpenseCategory(
            id=cat.id,
            name=cat.name,
            description=cat.description,
            created_at=cat.created_at.isoformat(),
        )
        for cat in categories
    ]


@router.post("/api/expense-categories", response_model=ExpenseCategory)
def create_expense_category(
    category_data: ExpenseCategoryCreate,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    new_category = ExpenseCategoryModel(
        name=category_data.name,
        description=category_data.description,
        created_at=datetime.now(IST),
    )
    db.add(new_category)
    db.commit()
    db.refresh(new_category)

    return ExpenseCategory(
        id=new_category.id,
        name=new_category.name,
        description=new_category.description,
        created_at=new_category.created_at.isoformat(),
    )


@router.post("/api/expenses")
def create_expense(
    data: ExpenseRequest,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    expense_date = data.expense_date or datetime.now(IST).date()

    expense = ExpenseModel(
        title=data.title,
        amount=data.amount,
        category_id=data.category_id,
        payment_mode=data.payment_mode,
        description=data.description,
        expense_date=expense_date,
        created_by=current_user.id,
        created_by_name=current_user.name,
        created_at=datetime.now(IST),
    )

    db.add(expense)

    db.add(
        LedgerEntry(
            id=str(uuid.uuid4()),
            entry_type="expense",
            description=data.title,
            debit=data.amount,
            credit=0,
            payment_mode=data.payment_mode,
            entry_date=expense_date,
            created_by=current_user.id,
            created_by_name=current_user.name,
            created_at=datetime.now(IST),
        )
    )

    db.commit()

    return {"message": "Expense added successfully", "amount": data.amount}


@router.get("/api/expenses")
def get_expenses(
    page: int = 1,
    limit: int = 20,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    category_id: Optional[str] = None,
    payment_mode: Optional[str] = None,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(ExpenseModel)

    if start_date:
        query = query.filter(ExpenseModel.expense_date >= start_date)
    if end_date:
        query = query.filter(ExpenseModel.expense_date <= end_date)
    if category_id:
        query = query.filter(ExpenseModel.category_id == category_id)
    if payment_mode:
        query = query.filter(ExpenseModel.payment_mode == payment_mode)

    total = query.count()
    expenses = query.order_by(ExpenseModel.expense_date.desc()).offset((page - 1) * limit).limit(limit).all()

    return {
        "data": [
            {
                "id": exp.id,
                "title": exp.title,
                "amount": exp.amount,
                "category_id": exp.category_id,
                "payment_mode": exp.payment_mode,
                "description": exp.description,
                "expense_date": exp.expense_date.isoformat() if exp.expense_date else None,
                "created_by_name": exp.created_by_name,
                "created_at": exp.created_at.isoformat(),
            }
            for exp in expenses
        ],
        "pagination": {"page": page, "limit": limit, "total": total, "total_pages": math.ceil(total / limit)},
    }


@router.get("/api/cheques")
def get_cheques(
    status: Optional[str] = None,
    page: int = 1,
    limit: int = 20,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(ChequeModel)
    if status:
        query = query.filter(ChequeModel.status == status)
    if start_date:
        query = query.filter(ChequeModel.cheque_date >= start_date)
    if end_date:
        query = query.filter(ChequeModel.cheque_date <= end_date)

    total = query.count()
    cheques = query.order_by(ChequeModel.cheque_date.desc()).offset((page - 1) * limit).limit(limit).all()

    return {
        "data": [
            {
                "id": c.id,
                "cheque_number": c.cheque_number,
                "cheque_date": c.cheque_date.isoformat() if c.cheque_date else None,
                "amount": c.amount,
                "bank_name": c.bank_name,
                "party_name": c.party_name,
                "party_type": c.party_type,
                "status": c.status,
                "cleared_date": c.cleared_date.isoformat() if c.cleared_date else None,
                "notes": c.notes,
                "created_at": c.created_at.isoformat(),
            }
            for c in cheques
        ],
        "pagination": {"page": page, "limit": limit, "total": total, "total_pages": math.ceil(total / limit)},
    }


@router.patch("/api/cheques/{cheque_id}/status")
def update_cheque_status(
    cheque_id: str,
    data: ChequeStatusUpdate,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    cheque = db.query(ChequeModel).filter(ChequeModel.id == cheque_id).first()
    if not cheque:
        raise HTTPException(404, "Cheque not found")

    cheque.status = data.status
    if data.cleared_date:
        cheque.cleared_date = data.cleared_date

    db.commit()
    return {"message": "Cheque status updated"}


@router.get("/api/accounts/ledger")
def general_ledger(
    page: int = 1,
    limit: int = 50,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    entry_type: Optional[str] = None,
    payment_mode: Optional[str] = None,
    customer_id: Optional[str] = None,
    role: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin only - Ledger access restricted")

    base_query = db.query(LedgerEntry)
    if start_date:
        base_query = base_query.filter(LedgerEntry.entry_date >= start_date)
    if end_date:
        base_query = base_query.filter(LedgerEntry.entry_date <= end_date)
    if entry_type:
        base_query = base_query.filter(LedgerEntry.entry_type == entry_type)
    if payment_mode:
        base_query = base_query.filter(LedgerEntry.payment_mode == payment_mode)
    if customer_id:
        base_query = base_query.filter(LedgerEntry.customer_id == customer_id)

    total = base_query.count()
    query = base_query.order_by(LedgerEntry.entry_date.desc(), LedgerEntry.created_at.desc(), LedgerEntry.id.desc())
    entries = query.offset((page - 1) * limit).limit(limit).all()

    opening_balance = 0
    if entries:
        opening_balance = (
            base_query.filter(_older_ledger_entries_filter(entries[-1]))
            .with_entities(func.coalesce(func.sum(LedgerEntry.credit - LedgerEntry.debit), 0))
            .scalar()
            or 0
        )

    balance = opening_balance
    data = []
    for e in reversed(entries):
        balance += e.credit - e.debit
        data.insert(
            0,
            {
                "id": e.id,
                "date": e.entry_date.isoformat() if e.entry_date else None,
                "type": e.entry_type,
                "description": e.description,
                "debit": e.debit,
                "credit": e.credit,
                "balance": balance,
                "mode": e.payment_mode,
                "created_by": e.created_by_name,
                "created_at": e.created_at.isoformat(),
            },
        )

    return {
        "data": data,
        "pagination": {"page": page, "limit": limit, "total": total, "total_pages": math.ceil(total / limit)},
    }


@router.get("/api/accounts/customer-ledger/{customer_id}")
def customer_ledger(
    customer_id: str,
    page: int = 1,
    limit: int = 50,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    customer = db.query(CustomerModel).filter(CustomerModel.id == customer_id).first()
    if not customer:
        raise HTTPException(404, "Customer not found")

    query = db.query(LedgerEntry).filter(LedgerEntry.customer_id == customer_id)
    if start_date:
        query = query.filter(LedgerEntry.entry_date >= start_date)
    if end_date:
        query = query.filter(LedgerEntry.entry_date <= end_date)

    total = query.count()
    entries = query.order_by(LedgerEntry.entry_date.asc()).offset((page - 1) * limit).limit(limit).all()

    balance = customer.opening_balance
    result = []
    for e in entries:
        balance += e.debit - e.credit
        result.append(
            {
                "date": e.entry_date.isoformat() if e.entry_date else None,
                "description": e.description,
                "debit": e.debit,
                "credit": e.credit,
                "balance": balance,
                "type": e.entry_type,
            }
        )

    return {
        "customer": {
            "id": customer.id,
            "name": customer.name,
            "phone": customer.phone,
            "current_balance": customer.current_balance,
        },
        "data": result,
        "pagination": {"page": page, "limit": limit, "total": total, "total_pages": math.ceil(total / limit)},
    }


@router.get("/api/accounts/export/ledger")
def export_ledger_csv(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(LedgerEntry)
    if start_date:
        query = query.filter(LedgerEntry.entry_date >= start_date)
    if end_date:
        query = query.filter(LedgerEntry.entry_date <= end_date)

    entries = query.order_by(LedgerEntry.entry_date.asc()).all()

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["Date", "Type", "Description", "Debit", "Credit", "Payment Mode", "Created By"])

    for e in entries:
        writer.writerow([
            e.entry_date.isoformat() if e.entry_date else "",
            e.entry_type,
            e.description,
            e.debit,
            e.credit,
            e.payment_mode,
            e.created_by_name or "",
        ])

    output.seek(0)
    return StreamingResponse(iter([output.getvalue()]), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=ledger.csv"})


@router.get("/api/accounts/export/expenses")
def export_expenses_csv(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(ExpenseModel)
    if start_date:
        query = query.filter(ExpenseModel.expense_date >= start_date)
    if end_date:
        query = query.filter(ExpenseModel.expense_date <= end_date)

    expenses = query.order_by(ExpenseModel.expense_date.asc()).all()

    output = StringIO()
    writer = csv.writer(output)
    writer.writerow(["Date", "Title", "Amount", "Category", "Payment Mode", "Description", "Created By"])

    for exp in expenses:
        writer.writerow([
            exp.expense_date.isoformat() if exp.expense_date else "",
            exp.title,
            exp.amount,
            exp.category_id or "",
            exp.payment_mode,
            exp.description or "",
            exp.created_by_name or "",
        ])

    output.seek(0)
    return StreamingResponse(iter([output.getvalue()]), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=expenses.csv"})


@router.get("/api/accounts/trial-balance")
def trial_balance(
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    debit = db.query(func.sum(LedgerEntry.debit)).scalar() or 0
    credit = db.query(func.sum(LedgerEntry.credit)).scalar() or 0

    return {"total_debit": debit, "total_credit": credit, "balanced": debit == credit}
