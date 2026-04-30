import ast
from math import ceil
from fastapi import Query
from itertools import product
from fastapi import FastAPI, APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from sqlalchemy import and_, create_engine, Column, String, Float, Integer, Text, ForeignKey, DateTime, case, or_, func, Date, Enum as SQLEnum
from sqlalchemy.orm import sessionmaker, declarative_base, Session, relationship
from sqlalchemy.orm.attributes import flag_modified
import os
import logging
from sqlalchemy import or_, func
from pydantic import Field
from fastapi.responses import StreamingResponse
import io
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from fastapi.responses import StreamingResponse

import json
from pathlib import Path
from copy import deepcopy
from pydantic import BaseModel, EmailStr, field_validator
from typing import List, Optional
import uuid
import random
import string
from datetime import datetime, timezone, timedelta, date
from passlib.context import CryptContext
from jose import JWTError, jwt
import math
import qrcode
from fastapi.staticfiles import StaticFiles
from sqlalchemy import JSON
import cloudinary
import cloudinary.uploader
from fastapi import UploadFile, File
from sqlalchemy.exc import OperationalError
import time
from fastapi.responses import JSONResponse
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
import redis.asyncio as redis
import csv
from io import StringIO

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

IST = timezone(timedelta(hours=5, minutes=30))

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

DATABASE_URL = os.environ.get("DATABASE_URL")

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    pool_recycle=1800,
    pool_timeout=30,
    connect_args={"connect_timeout": 10},
)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)
Base = declarative_base()

cloudinary.config(
    cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME"),
    api_key=os.environ.get("CLOUDINARY_API_KEY"),
    api_secret=os.environ.get("CLOUDINARY_API_SECRET"),
    secure=True
)

def upload_image_to_cloudinary(file, folder="products"):
    result = cloudinary.uploader.upload(
        file,
        folder=folder,
        resource_type="image"
    )
    return result["secure_url"]

# ============= MODELS =============

class UserModel(Base):
    __tablename__ = "users"
    __allow_unmapped__ = True

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password = Column(String(255), nullable=False)
    role = Column(String(50), default="user")
    created_at = Column(DateTime, default=lambda: datetime.now(IST))

class CustomerModel(Base):
    __tablename__ = "customers"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    email = Column(String(255), nullable=False)
    phone = Column(String(50), nullable=True, index=True)
    address = Column(Text, nullable=True)
    opening_balance = Column(Float, default=0)
    current_balance = Column(Float, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now(IST))

    invoices = relationship("InvoiceModel", back_populates="customer")

class SupplierModel(Base):
    __tablename__ = "suppliers"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    phone = Column(String(50), nullable=True)
    email = Column(String(255), nullable=True)
    address = Column(Text, nullable=True)
    opening_balance = Column(Float, default=0)
    current_balance = Column(Float, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now(IST))

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
    party_type = Column(String(50), nullable=False)  # customer/supplier
    status = Column(String(50), default="pending")  # pending/cleared/bounced
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
    
    # Tax breakdown
    cgst = Column(Float, default=0)
    sgst = Column(Float, default=0)
    igst = Column(Float, default=0)
    
    entry_date = Column(Date, default=lambda: datetime.now(IST).date(), index=True)
    created_by = Column(String(36), nullable=True)
    created_by_name = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(IST), index=True)

class InvoiceModel(Base):
    __tablename__ = "invoices"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    invoice_number = Column(String(50), nullable=True, unique=True)
    customer_id = Column(String(36), ForeignKey("customers.id"), nullable=False)
    customer_name = Column(String(255), nullable=False)
    customer_phone = Column(String(50), nullable=True)
    customer_address = Column(Text, nullable=True)
    items = Column(Text, nullable=False)
    subtotal = Column(Float, nullable=False)
    gst_amount = Column(Float, nullable=False, default=0)
    cgst_amount = Column(Float, default=0)
    sgst_amount = Column(Float, default=0)
    igst_amount = Column(Float, default=0)
    created_by = Column(String(36))
    additional_charges = Column(JSON, nullable=False, default=list)
    gst_enabled = Column(Integer, default=1)
    gst_rate = Column(Float, default=0)
    created_by_name = Column(String(100))
    invoice_type = Column(String(10), default="FINAL", index=True)
    draft_number = Column(String(50), nullable=True, unique=True)
    discount = Column(Float, nullable=False, default=0)
    total = Column(Float, nullable=False)
    payment_status = Column(String(50), nullable=False, default="pending", index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(IST), index=True)
    
    customer = relationship("CustomerModel", back_populates="invoices")

# ============= PYDANTIC SCHEMAS =============

class UserRegister(BaseModel):
    email: EmailStr
    password: str
    name: str
    role: str = "store_handler"

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class User(BaseModel):
    id: str
    email: str
    name: str
    role: str
    created_at: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user: User

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

# ============= APP SETUP =============

app = FastAPI()

@app.on_event("startup")
async def startup():
    redis_client = redis.from_url(
        "redis://localhost:6379",
        encoding="utf-8",
        decode_responses=True
    )
    await FastAPILimiter.init(redis_client)

api_router = APIRouter(prefix="/api")

SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# ============= HELPER FUNCTIONS =============

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password[:72], hashed_password)

def get_password_hash(password: str) -> str:
    password = password[:72]
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(IST) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db)
):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        if not credentials:
            raise credentials_exception

        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError as e:
        raise credentials_exception

    user = db.query(UserModel).filter(UserModel.email == email).first()
    if user is None:
        raise credentials_exception
    return user

def generate_invoice_number(db: Session):
    now = datetime.now(IST)
    fy_year = now.year if now.month >= 4 else now.year - 1
    fy_suffix = f"{fy_year % 100:02d}-{(fy_year + 1) % 100:02d}"

    last = (
        db.query(InvoiceModel.invoice_number)
        .filter(
            InvoiceModel.invoice_type == "FINAL",
            InvoiceModel.invoice_number.like(f"INV-{fy_suffix}-%")
        )
        .order_by(InvoiceModel.invoice_number.desc())
        .with_for_update()
        .first()
    )

    next_num = 1
    if last and last[0]:
        next_num = int(last[0].split("-")[-1]) + 1

    return f"INV-{fy_suffix}-{next_num:04d}"

# ============= AUTH ENDPOINTS =============

@api_router.post("/auth/register", response_model=Token)
def register(user_data: UserRegister, db: Session = Depends(get_db)):
    existing_user = db.query(UserModel).filter(UserModel.email == user_data.email).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")

    user_id = str(uuid.uuid4())
    hashed_password = get_password_hash(user_data.password)

    new_user = UserModel(
        id=user_id,
        email=user_data.email,
        name=user_data.name,
        password=hashed_password,
        role=user_data.role,
        created_at=datetime.now(IST)
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    access_token = create_access_token(data={"sub": new_user.email, "role": new_user.role})

    user = User(
        id=new_user.id,
        email=new_user.email,
        name=new_user.name,
        role=new_user.role,
        created_at=new_user.created_at.isoformat()
    )

    return Token(access_token=access_token, token_type="bearer", user=user)

@api_router.post("/auth/login", dependencies=[Depends(RateLimiter(times=10, seconds=60))])
def login(user_data: UserLogin, db: Session = Depends(get_db)):
    user = db.query(UserModel).filter(UserModel.email == user_data.email).first()

    if not user or not verify_password(user_data.password, user.password):
        raise HTTPException(status_code=401, detail="Incorrect email or password")

    access_token = create_access_token(data={"sub": user.email, "role": user.role})

    user_obj = User(
        id=user.id,
        email=user.email,
        name=user.name,
        role=user.role,
        created_at=user.created_at.isoformat()
    )

    return Token(access_token=access_token, token_type="bearer", user=user_obj)

# ============= EXPENSE CATEGORY ENDPOINTS =============

@api_router.get("/expense-categories", response_model=List[ExpenseCategory])
def get_expense_categories(
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    categories = db.query(ExpenseCategoryModel).all()
    return [
        ExpenseCategory(
            id=cat.id,
            name=cat.name,
            description=cat.description,
            created_at=cat.created_at.isoformat()
        )
        for cat in categories
    ]

@api_router.post("/expense-categories", response_model=ExpenseCategory)
def create_expense_category(
    category_data: ExpenseCategoryCreate,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    new_category = ExpenseCategoryModel(
        id=str(uuid.uuid4()),
        name=category_data.name,
        description=category_data.description,
        created_at=datetime.now(IST)
    )
    db.add(new_category)
    db.commit()
    db.refresh(new_category)

    return ExpenseCategory(
        id=new_category.id,
        name=new_category.name,
        description=new_category.description,
        created_at=new_category.created_at.isoformat()
    )

# ============= EXPENSE ENDPOINTS =============

@api_router.post("/expenses")
def create_expense(
    data: ExpenseRequest,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user)
):
    expense_date = data.expense_date or datetime.now(IST).date()
    
    expense = ExpenseModel(
        id=str(uuid.uuid4()),
        title=data.title,
        amount=data.amount,
        category_id=data.category_id,
        payment_mode=data.payment_mode,
        description=data.description,
        expense_date=expense_date,
        created_by=current_user.id,
        created_by_name=current_user.name,
        created_at=datetime.now(IST)
    )

    db.add(expense)

    # Ledger Entry
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
            created_at=datetime.now(IST)
        )
    )

    db.commit()

    return {"message": "Expense added successfully", "amount": data.amount}

@api_router.get("/expenses")
def get_expenses(
    page: int = 1,
    limit: int = 20,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    category_id: Optional[str] = None,
    payment_mode: Optional[str] = None,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
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
    
    expenses = (
        query.order_by(ExpenseModel.expense_date.desc())
        .offset((page - 1) * limit)
        .limit(limit)
        .all()
    )

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
                "created_at": exp.created_at.isoformat()
            }
            for exp in expenses
        ],
        "pagination": {
            "page": page,
            "limit": limit,
            "total": total,
            "total_pages": math.ceil(total / limit)
        }
    }

# ============= CUSTOMER ENDPOINTS =============

@api_router.get("/customers")
def get_customers(
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
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
            "created_at": c.created_at.isoformat()
        }
        for c in customers
    ]

# ============= SUPPLIER ENDPOINTS =============

@api_router.get("/suppliers")
def get_suppliers(
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
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
            "created_at": s.created_at.isoformat()
        }
        for s in suppliers
    ]

@api_router.post("/suppliers")
def create_supplier(
    name: str,
    phone: Optional[str] = None,
    email: Optional[str] = None,
    address: Optional[str] = None,
    opening_balance: float = 0,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    supplier = SupplierModel(
        id=str(uuid.uuid4()),
        name=name,
        phone=phone,
        email=email,
        address=address,
        opening_balance=opening_balance,
        current_balance=opening_balance,
        created_at=datetime.now(IST)
    )
    db.add(supplier)
    db.commit()
    db.refresh(supplier)

    return {"message": "Supplier created", "id": supplier.id}

# ============= PAYMENT ENDPOINTS =============

@api_router.post("/accounts/payment-in")
def payment_in(
    data: PaymentInRequest,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user)
):
    customer = db.query(CustomerModel).filter(CustomerModel.id == data.customer_id).first()
    if not customer:
        raise HTTPException(404, "Customer not found")

    payment_date = data.payment_date or datetime.now(IST).date()

    # Update customer balance
    customer.current_balance -= data.amount
    
    # Handle cheque
    if data.payment_mode == "cheque" and data.cheque_number:
        cheque = ChequeModel(
            id=str(uuid.uuid4()),
            cheque_number=data.cheque_number,
            cheque_date=data.cheque_date or payment_date,
            amount=data.amount,
            bank_name=data.bank_name,
            party_name=customer.name,
            party_type="customer",
            status="pending",
            payment_mode="cheque",
            created_by=current_user.id,
            created_at=datetime.now(IST)
        )
        db.add(cheque)

    # Ledger Entry
    db.add(
        LedgerEntry(
            id=str(uuid.uuid4()),
            entry_type="payment_in",
            customer_id=data.customer_id,
            description=f"Payment received from {customer.name} {data.reference or ''}",
            debit=0,
            credit=data.amount,
            payment_mode=data.payment_mode,
            entry_date=payment_date,
            created_by=current_user.id,
            created_by_name=current_user.name,
            created_at=datetime.now(IST)
        )
    )

    db.commit()

    return {"message": "Payment recorded", "new_balance": customer.current_balance}

@api_router.post("/accounts/payment-out")
def payment_out(
    data: PaymentOutRequest,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user)
):
    payment_date = data.payment_date or datetime.now(IST).date()
    
    supplier = None
    if data.supplier_id:
        supplier = db.query(SupplierModel).filter(SupplierModel.id == data.supplier_id).first()
        if supplier:
            supplier.current_balance -= data.amount

    # Handle cheque
    if data.payment_mode == "cheque" and data.cheque_number:
        cheque = ChequeModel(
            id=str(uuid.uuid4()),
            cheque_number=data.cheque_number,
            cheque_date=data.cheque_date or payment_date,
            amount=data.amount,
            bank_name=data.bank_name,
            party_name=data.supplier_name,
            party_type="supplier",
            status="pending",
            payment_mode="cheque",
            created_by=current_user.id,
            created_at=datetime.now(IST)
        )
        db.add(cheque)

    # Ledger Entry
    db.add(
        LedgerEntry(
            id=str(uuid.uuid4()),
            entry_type="payment_out",
            supplier_id=data.supplier_id,
            description=data.description or f"Payment to {data.supplier_name}",
            debit=data.amount,
            credit=0,
            payment_mode=data.payment_mode,
            entry_date=payment_date,
            created_by=current_user.id,
            created_by_name=current_user.name,
            created_at=datetime.now(IST)
        )
    )

    db.commit()

    return {"message": "Payment recorded successfully", "amount": data.amount}

# ============= CHEQUE ENDPOINTS =============

@api_router.get("/cheques")
def get_cheques(
    status: Optional[str] = None,
    page: int = 1,
    limit: int = 20,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    query = db.query(ChequeModel)
    
    if status:
        query = query.filter(ChequeModel.status == status)

    total = query.count()
    
    cheques = (
        query.order_by(ChequeModel.cheque_date.desc())
        .offset((page - 1) * limit)
        .limit(limit)
        .all()
    )

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
                "created_at": c.created_at.isoformat()
            }
            for c in cheques
        ],
        "pagination": {
            "page": page,
            "limit": limit,
            "total": total,
            "total_pages": math.ceil(total / limit)
        }
    }

@api_router.patch("/cheques/{cheque_id}/status")
def update_cheque_status(
    cheque_id: str,
    data: ChequeStatusUpdate,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    cheque = db.query(ChequeModel).filter(ChequeModel.id == cheque_id).first()
    if not cheque:
        raise HTTPException(404, "Cheque not found")

    cheque.status = data.status
    if data.cleared_date:
        cheque.cleared_date = data.cleared_date

    db.commit()

    return {"message": "Cheque status updated"}

# ============= LEDGER ENDPOINTS =============

@api_router.get("/accounts/ledger")
def general_ledger(
    page: int = 1,
    limit: int = 50,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    entry_type: Optional[str] = None,
    payment_mode: Optional[str] = None,
    customer_id: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user)
):
    query = db.query(LedgerEntry)

    if start_date:
        query = query.filter(LedgerEntry.entry_date >= start_date)
    if end_date:
        query = query.filter(LedgerEntry.entry_date <= end_date)
    if entry_type:
        query = query.filter(LedgerEntry.entry_type == entry_type)
    if payment_mode:
        query = query.filter(LedgerEntry.payment_mode == payment_mode)
    if customer_id:
        query = query.filter(LedgerEntry.customer_id == customer_id)

    total = query.count()
    
    entries = query.order_by(LedgerEntry.entry_date.desc(), LedgerEntry.created_at.desc()).offset((page - 1) * limit).limit(limit).all()

    # Calculate running balance
    balance = 0
    data = []

    for e in reversed(entries):
        balance += e.debit - e.credit
        data.insert(0, {
            "id": e.id,
            "date": e.entry_date.isoformat() if e.entry_date else None,
            "type": e.entry_type,
            "description": e.description,
            "debit": e.debit,
            "credit": e.credit,
            "balance": balance,
            "mode": e.payment_mode,
            "created_by": e.created_by_name,
            "created_at": e.created_at.isoformat()
        })

    return {
        "data": data,
        "pagination": {
            "page": page,
            "limit": limit,
            "total": total,
            "total_pages": math.ceil(total / limit)
        }
    }

@api_router.get("/accounts/cashbook")
def cashbook(
    page: int = 1,
    limit: int = 50,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user)
):
    query = db.query(LedgerEntry).filter(LedgerEntry.payment_mode == "cash")

    if start_date:
        query = query.filter(LedgerEntry.entry_date >= start_date)
    if end_date:
        query = query.filter(LedgerEntry.entry_date <= end_date)

    total = query.count()
    
    rows = query.order_by(LedgerEntry.entry_date.desc()).offset((page - 1) * limit).limit(limit).all()

    balance = 0
    data = []

    for r in reversed(rows):
        balance += r.debit - r.credit
        data.insert(0, {
            "id": r.id,
            "date": r.entry_date.isoformat() if r.entry_date else None,
            "description": r.description,
            "debit": r.debit,
            "credit": r.credit,
            "balance": balance,
            "type": r.entry_type,
            "created_at": r.created_at.isoformat()
        })

    return {
        "data": data,
        "pagination": {
            "page": page,
            "limit": limit,
            "total": total,
            "total_pages": math.ceil(total / limit)
        }
    }

@api_router.get("/accounts/bankbook")
def bankbook(
    page: int = 1,
    limit: int = 50,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user)
):
    query = db.query(LedgerEntry).filter(or_(
        LedgerEntry.payment_mode == "bank",
        LedgerEntry.payment_mode == "upi",
        LedgerEntry.payment_mode == "cheque"
    ))

    if start_date:
        query = query.filter(LedgerEntry.entry_date >= start_date)
    if end_date:
        query = query.filter(LedgerEntry.entry_date <= end_date)

    total = query.count()
    
    rows = query.order_by(LedgerEntry.entry_date.desc()).offset((page - 1) * limit).limit(limit).all()

    balance = 0
    data = []

    for r in reversed(rows):
        balance += r.debit - r.credit
        data.insert(0, {
            "id": r.id,
            "date": r.entry_date.isoformat() if r.entry_date else None,
            "description": r.description,
            "debit": r.debit,
            "credit": r.credit,
            "balance": balance,
            "type": r.entry_type,
            "mode": r.payment_mode,
            "created_at": r.created_at.isoformat()
        })

    return {
        "data": data,
        "pagination": {
            "page": page,
            "limit": limit,
            "total": total,
            "total_pages": math.ceil(total / limit)
        }
    }

@api_router.get("/accounts/gst-ledger")
def gst_ledger(
    page: int = 1,
    limit: int = 50,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user)
):
    query = db.query(InvoiceModel).filter(
        InvoiceModel.invoice_type == "FINAL",
        InvoiceModel.payment_status == "paid",
        InvoiceModel.gst_amount > 0
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
                "cgst": inv.cgst_amount or (inv.gst_amount / 2 if inv.gst_amount > 0 else 0),
                "sgst": inv.sgst_amount or (inv.gst_amount / 2 if inv.gst_amount > 0 else 0),
                "igst": inv.igst_amount or 0,
                "total": inv.total,
                "date": inv.created_at.date().isoformat()
            }
            for inv in invoices
        ],
        "pagination": {
            "page": page,
            "limit": limit,
            "total": total,
            "total_pages": math.ceil(total / limit)
        }
    }

@api_router.get("/accounts/profit-loss")
def profit_loss(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user)
):
    query_income = db.query(func.sum(InvoiceModel.total)).filter(
        InvoiceModel.invoice_type == "FINAL",
        InvoiceModel.payment_status == "paid"
    )
    
    query_expense = db.query(func.sum(ExpenseModel.amount))

    if start_date:
        query_income = query_income.filter(func.date(InvoiceModel.created_at) >= start_date)
        query_expense = query_expense.filter(ExpenseModel.expense_date >= start_date)
    
    if end_date:
        query_income = query_income.filter(func.date(InvoiceModel.created_at) <= end_date)
        query_expense = query_expense.filter(ExpenseModel.expense_date <= end_date)

    income = query_income.scalar() or 0
    expenses = query_expense.scalar() or 0

    return {
        "income": float(income),
        "expense": float(expenses),
        "profit": float(income - expenses)
    }

@api_router.get("/accounts/trial-balance")
def trial_balance(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user)
):
    query = db.query(
        func.sum(LedgerEntry.debit).label("total_debit"),
        func.sum(LedgerEntry.credit).label("total_credit")
    )

    if start_date:
        query = query.filter(LedgerEntry.entry_date >= start_date)
    if end_date:
        query = query.filter(LedgerEntry.entry_date <= end_date)

    result = query.first()
    
    debit = float(result.total_debit or 0)
    credit = float(result.total_credit or 0)

    return {
        "total_debit": debit,
        "total_credit": credit,
        "difference": abs(debit - credit),
        "balanced": abs(debit - credit) < 0.01
    }

@api_router.get("/accounts/customer-ledger/{customer_id}")
def customer_ledger(
    customer_id: str,
    page: int = 1,
    limit: int = 50,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user)
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

        result.append({
            "date": e.entry_date.isoformat() if e.entry_date else None,
            "description": e.description,
            "debit": e.debit,
            "credit": e.credit,
            "balance": balance,
            "type": e.entry_type
        })

    return {
        "customer": {
            "id": customer.id,
            "name": customer.name,
            "phone": customer.phone,
            "current_balance": customer.current_balance
        },
        "data": result,
        "pagination": {
            "page": page,
            "limit": limit,
            "total": total,
            "total_pages": math.ceil(total / limit)
        }
    }

# ============= EXPORT ENDPOINTS =============

@api_router.get("/accounts/export/ledger")
def export_ledger_csv(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
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
            e.created_by_name or ""
        ])

    output.seek(0)
    
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=ledger.csv"}
    )

@api_router.get("/accounts/export/expenses")
def export_expenses_csv(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
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
            exp.created_by_name or ""
        ])

    output.seek(0)
    
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=expenses.csv"}
    )

# ============= UPDATE INVOICE STATUS (CRITICAL FIX) =============

@api_router.patch("/invoices/{invoice_id}/status")
def update_invoice_status(
    invoice_id: str,
    payment_status: str,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    invoice = db.query(InvoiceModel).filter(InvoiceModel.id == invoice_id).first()

    if not invoice:
        raise HTTPException(status_code=404, detail="Invoice not found")
    
    if invoice.invoice_type != "FINAL":
        raise HTTPException(400, "Cannot change status of a draft invoice")

    old_status = invoice.payment_status
    invoice.payment_status = payment_status

    # 🔥 CRITICAL: Only create ledger entry when changing to "paid"
    if old_status != "paid" and payment_status == "paid":
        customer = db.query(CustomerModel).filter(CustomerModel.id == invoice.customer_id).first()
        
        if customer:
            customer.current_balance += invoice.total

        # Create sale ledger entry
        db.add(
            LedgerEntry(
                id=str(uuid.uuid4()),
                entry_type="sale",
                reference_id=invoice.id,
                customer_id=invoice.customer_id,
                description=f"Sale Invoice {invoice.invoice_number}",
                debit=invoice.total,
                credit=0,
                payment_mode="credit",
                cgst=invoice.cgst_amount or 0,
                sgst=invoice.sgst_amount or 0,
                igst=invoice.igst_amount or 0,
                entry_date=datetime.now(IST).date(),
                created_by=current_user.id,
                created_by_name=current_user.name,
                created_at=datetime.now(IST)
            )
        )

        # Create payment entry
        db.add(
            LedgerEntry(
                id=str(uuid.uuid4()),
                entry_type="payment_in",
                reference_id=invoice.id,
                customer_id=invoice.customer_id,
                description=f"Payment received for {invoice.invoice_number}",
                debit=0,
                credit=invoice.total,
                payment_mode="cash",
                entry_date=datetime.now(IST).date(),
                created_by=current_user.id,
                created_by_name=current_user.name,
                created_at=datetime.now(IST)
            )
        )

    db.commit()
    
    return {"message": "Invoice status updated successfully"}

# ============= SUMMARY ENDPOINT =============

@api_router.get("/accounts/summary")
def accounts_summary(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Total Cash
    cash_query = db.query(
        func.sum(LedgerEntry.debit - LedgerEntry.credit)
    ).filter(LedgerEntry.payment_mode == "cash")

    # Total Bank
    bank_query = db.query(
        func.sum(LedgerEntry.debit - LedgerEntry.credit)
    ).filter(or_(
        LedgerEntry.payment_mode == "bank",
        LedgerEntry.payment_mode == "upi"
    ))

    if start_date:
        cash_query = cash_query.filter(LedgerEntry.entry_date >= start_date)
        bank_query = bank_query.filter(LedgerEntry.entry_date >= start_date)
    
    if end_date:
        cash_query = cash_query.filter(LedgerEntry.entry_date <= end_date)
        bank_query = bank_query.filter(LedgerEntry.entry_date <= end_date)

    cash_balance = cash_query.scalar() or 0
    bank_balance = bank_query.scalar() or 0

    # Customer Outstanding
    customers_outstanding = db.query(func.sum(CustomerModel.current_balance)).scalar() or 0

    # Supplier Outstanding
    suppliers_outstanding = db.query(func.sum(SupplierModel.current_balance)).scalar() or 0

    # Pending Cheques
    pending_cheques = db.query(func.sum(ChequeModel.amount)).filter(
        ChequeModel.status == "pending"
    ).scalar() or 0

    return {
        "cash_balance": float(cash_balance),
        "bank_balance": float(bank_balance),
        "total_balance": float(cash_balance + bank_balance),
        "customers_outstanding": float(customers_outstanding),
        "suppliers_outstanding": float(suppliers_outstanding),
        "pending_cheques": float(pending_cheques)
    }

# ============= INCLUDE ROUTER =============

app.include_router(api_router)

ALLOWED_ORIGINS = [
    "http://localhost",
    "http://localhost:3000",
    "capacitor://localhost"
]

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    allow_headers=["Authorization", "Content-Type", "Accept"],
)
