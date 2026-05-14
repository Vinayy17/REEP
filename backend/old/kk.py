import ast
from math import ceil
from fastapi import Query
from itertools import product
from fastapi import FastAPI, APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from sqlalchemy import and_, create_engine, Column, String, Float, Integer, Text, ForeignKey, DateTime, Date, case, or_, func,Index
from sqlalchemy.orm import sessionmaker, declarative_base, Session, relationship
from sqlalchemy.orm.attributes import flag_modified
import os
import logging
from sqlalchemy import or_, func
from pydantic import Field
from fastapi.responses import StreamingResponse
import io
from io. import StringIO
import csv
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


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)




IST = timezone(timedelta(hours=5, minutes=30))

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

DATABASE_URL = os.environ.get(
    "DATABASE_URL")
   
# DATABASE_URL = os.environ.get(
#     "DATABASE_URL",
#     "mysql+pymysql://chinaligths_user:StrongPassword123%21@localhost:3306/chinaligths?charset=utf8mb4"
# )

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,          # max connections per worker
    max_overflow=10,      # temporary burst
    pool_recycle=1800,    # avoid MySQL "server has gone away"
    pool_timeout=30,     # wait max 30s for a connection
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

class UserModel(Base):
    __tablename__ = "users"
    __allow_unmapped__ = True   # ✅ REQUIRED

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, index=True, nullable=False)
    password = Column(String(255), nullable=False)
    role = Column(String(50), default="user")
    created_at = Column(DateTime, default=lambda: datetime.now(IST))

class RequirementModel(Base):
    __tablename__ = "requirements"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    customer_name = Column(String(255), nullable=False)
    customer_phone = Column(String(50), nullable=False)

    requirement_items = Column(Text, nullable=False)  # JSON string
    priority = Column(String(20), default="normal")  # low | normal | high | urgent

    status = Column(String(20), default="pending", index=True)

    created_by = Column(String(36))
    created_by_name = Column(String(100))

    created_at = Column(DateTime, default=lambda: datetime.now(IST), index=True)

    completed_at = Column(DateTime, nullable=True)

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

class CategoryModel(Base):
    __tablename__ = "categories"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(IST))

    products = relationship("ProductModel", back_populates="category")

class ProductModel(Base):
    __tablename__ = "products"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    product_code = Column(String(50), nullable=False, unique=True, index=True)

    # 👇 PARENT SKU (PARENT CODE)
    sku = Column(String(100), nullable=False, unique=True, index=True)

    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)

    category_id = Column(String(36), ForeignKey("categories.id"), nullable=False)

    # 💰 Pricing (COMMON)
    cost_price = Column(Float, nullable=False, default=0)
    min_selling_price = Column(Float, nullable=False)
    selling_price = Column(Float, nullable=False)

    # 📦 TOTAL STOCK (AUTO)
    stock = Column(Integer, nullable=False, default=0)
    min_stock = Column(Integer, nullable=False, default=5)

    # 🧩 VARIANTS (INLINE JSON)
    variants = Column(JSON, nullable=False)

    qr_code_url = Column(String(255), nullable=True)
    images = Column(JSON, nullable=True)

    is_service = Column(Integer, default=0)
    is_active = Column(Integer, default=1)

    created_at = Column(DateTime, default=lambda: datetime.now(IST))


    category = relationship("CategoryModel", back_populates="products")

class InventoryTransaction(Base):
    __tablename__ = "inventory_transactions"

    id = Column(String(36), primary_key=True)  # UUID REQUIRED
    product_id = Column(String(36), ForeignKey("products.id"))
    type = Column(String(3))
    quantity = Column(Integer)
    source = Column(String(50))
    reason = Column(String(255))
    created_by = Column(String(36))
    created_at = Column(DateTime, default=lambda: datetime.now(IST))
    created_by_name = Column(String(100))        # ✅ ADD THIS

    stock_before = Column(Integer, nullable=False, default=0)
    stock_after = Column(Integer, nullable=False, default=0)
    variant_stock_after = Column(Integer, nullable=True)

    variant_sku = Column(String(100), nullable=True)  # Stores v_sku if transaction is variant-specific

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

    # ================= PRIMARY =================
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    # invoice numbers
    invoice_number = Column(String(50), nullable=True, unique=True, index=True)
    draft_number = Column(String(50), nullable=True)

    # draft / final
    invoice_type = Column(String(20), default="FINAL", index=True)

    # ================= CUSTOMER =================
    customer_id = Column(String(36), ForeignKey("customers.id"), nullable=False)
    customer_name = Column(String(255), nullable=False)

    customer_phone = Column(String(50), nullable=True)
    customer_address = Column(Text, nullable=True)

    # ================= ITEMS =================
    items = Column(Text, nullable=False)

    # ================= AMOUNTS =================
    subtotal = Column(Float, nullable=False)

    gst_enabled = Column(Integer, default=1)
    gst_rate = Column(Float, default=0)

    gst_amount = Column(Float, nullable=False, default=0)

    discount = Column(Float, nullable=False, default=0)

    # additional charges (delivery, packing etc)
    additional_charges = Column(JSON, default=[])

    total = Column(Float, nullable=False)

    # ================= PAYMENT =================
    paid_amount = Column(Float, default=0)
    balance_amount = Column(Float, default=0)

    payment_mode = Column(String(50), default="cash")

    payment_status = Column(
        String(50),
        nullable=False,
        default="pending",
        index=True
    )

    # 🔥 NEW: Payment History (stores all partial payments)
    payment_history = Column(JSON, default=[])

    # ================= USER INFO =================
    created_by = Column(String(36))
    created_by_name = Column(String(100))

    # ================= TIMESTAMP =================
    created_at = Column(
        DateTime,
        default=lambda: datetime.now(IST),
        index=True
    )

    # ================= RELATIONSHIP =================
    customer = relationship(
        "CustomerModel",
        back_populates="invoices"
    )

    # ================= INDEXES =================
    __table_args__ = (
        Index("idx_invoice_type_status", "invoice_type", "payment_status"),
        Index("idx_invoice_created", "created_at"),
    )


# Create tables

app = FastAPI()
# app.mount("/static", StaticFiles(directory="static"), name="static")
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

pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto"
)

security = HTTPBearer()

# Database dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(
        plain_password[:72],
        hashed_password
    )

def get_password_hash(password: str) -> str:
    password = password[:72]   # truncate STRING, not bytes
    return pwd_context.hash(password)


def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.now(IST) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def generate_product_code(db: Session, prefix: str = "RR") -> str:
    # Get last product code with prefix RR
    last_code = (
        db.query(ProductModel.product_code)
        .filter(ProductModel.product_code.like(f"{prefix}-%"))
        .order_by(ProductModel.product_code.desc())
        .first()
    )

    if not last_code:
        next_number = 1
    else:
        # Extract numeric part: RR-0007 → 7
        last_number = int(last_code[0].split("-")[1])
        next_number = last_number + 1

    return f"{prefix}-{next_number:04d}"


def generate_qr(data: dict):
    qr_text = json.dumps(data, separators=(",", ":"))

    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_Q,
        box_size=8,
        border=2,
    )
    qr.add_data(qr_text)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")

    # 🚀 Upload QR to Cloudinary
    qr_url = upload_qr_to_cloudinary(img, folder="product_qr")

    return qr_url


# Fixed get_current_user to properly handle JWT token errors
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

def safe_images(value):
    if not value:
        return []
    if isinstance(value, list):
        return value
    try:
        return json.loads(value)
    except Exception:
        return []

def calculate_total_stock(variants: list) -> int:
    return sum(int(v.get("stock", 0)) for v in variants)

def has_inventory(db: Session, product_id: str) -> bool:
    return db.query(InventoryTransaction).filter(
        InventoryTransaction.product_id == product_id
    ).first() is not None

def has_invoice(db: Session, product_id: str) -> bool:
    invoices = db.query(InvoiceModel).all()
    for inv in invoices:
        items = parse_invoice_items(inv.items)
        for item in items:
            if item.get("product_id") == product_id:
                return True
    return False

def calculate_additional_total(charges):
    if not charges:
        return 0.0

    total = 0.0
    for c in charges:
        if hasattr(c, "amount"):
            total += float(c.amount or 0)
        elif isinstance(c, dict):
            total += float(c.get("amount", 0))

    return round(total, 2)

@api_router.post( "/upload/product-image",dependencies=[Depends(RateLimiter(times=20, seconds=60))])
def upload_product_image(
    file: UploadFile = File(...),
    current_user: UserModel = Depends(get_current_user)
):
    # 🔐 Only admin
    if current_user.role not in ["admin", "store_handler"]:

        raise HTTPException(status_code=403, detail="Admin only")

    url = upload_image_to_cloudinary(file.file)
    return {"url": url}

# Pydantic Models
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

class Category(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    created_at: str

class CategoryCreate(BaseModel):
    name: str
    description: Optional[str] = None

class VariantSchema(BaseModel):
    v_sku: str
    variant_name: Optional[str] = None
    color: Optional[str] = None
    size: Optional[str] = None
    stock: int = 0
    image_url: Optional[str] = None
    qr_code_url: Optional[str] = None
    image: Optional[str] = None # For variant image
    v_selling_price: Optional[float] = None # For variant specific selling price


class Product(BaseModel):
    id: str
    product_code: str
    sku: str

    name: str
    description: Optional[str] = None

    category_id: str
    category_name: Optional[str] = None

    cost_price: Optional[float] = None
    min_selling_price: float
    selling_price: float

    stock: int
    min_stock: int
    is_service: int = 0

    variants: List[VariantSchema]

    images: Optional[List[str]] = Field(default_factory=list)
    qr_code_url: Optional[str] = None
    created_at: datetime   # ✅ TYPE, NOT Column

    class Config:
        from_attributes = True   

class ProductCreate(BaseModel):
    name: str
    description: Optional[str] = None # Default to None
    category_id: str

    cost_price: float = 0.0 # Default to 0.0
    selling_price: float
    min_selling_price: float

    min_stock: int = 0 # Default to 0
    sku: str
    images: List[str] = Field(default_factory=list) # Default to empty list
    is_service: int = 0
    variants: List[VariantSchema] = Field(default_factory=list)
    qr_code_url: Optional[str] = None # Matches the field being sent/required

class Customer(BaseModel):
    id: str
    name: str
    email: str
    phone: Optional[str] = None
    address: Optional[str] = None
    created_at: str

class CustomerCreate(BaseModel):
    name: str
    email: str
    phone: Optional[str] = None
    address: Optional[str] = None

class InvoiceItem(BaseModel):
    product_id: Optional[str] = None
    product_name: str
    quantity: int
    price: float # This will be the price per unit for the item
    gst_rate: float
    sku: Optional[str] = None # Variant SKU or product SKU
    is_service: int = 0        # ✅ ADD THIS

    variant_info: Optional[dict] = None # Added for variant details
    image_url: Optional[str] = None # Added for item image

class AdditionalCharge(BaseModel):
    label: Optional[str] = None
    amount: Optional[float] = 0

    @field_validator("amount", mode="before")
    def allow_empty_amount(cls, v):
        if v in ("", None):
            return 0
        return float(v)

class ExpenseRequest(BaseModel):
    title: str
    amount: float
    payment_mode: str
    description: Optional[str] = None

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

    # gst_amount: float = 0
    discount: float = 0
    payment_status: str = "pending"
    payment_mode: Optional[str] = "cash"

    # additional_amount: float = 0  # ✅ ADD for manual charges
    # additional_label: Optional[str] = None  # ✅ ADD for charge label
    additional_charges: List[AdditionalCharge] = Field(default_factory=list)



class DashboardStats(BaseModel):
    total_sales: float
    total_orders: int
    total_customers: int
    low_stock_items: int
    recent_invoices: List[Invoice]

class SalesChartItem(BaseModel):
    name: str
    total: float
    paid: float
    pending: float
    partial: float

class MaterialInwardRequest(BaseModel):
    product_id: str
    quantity: int
    # Optional: for variant-level inventory (v_sku or parent sku)
    sku: Optional[str] = None


class MaterialOutwardRequest(BaseModel):
    product_id: str
    quantity: int
    reason: str
    # Optional: for variant-level inventory (v_sku or parent sku)
    sku: Optional[str] = None


class MaterialInwardBySkuRequest(BaseModel):
    sku: str
    quantity: int


class MaterialOutwardBySkuRequest(BaseModel):
    sku: str
    quantity: int
    reason: str

class RequirementItem(BaseModel):
    text: str
    image_url: Optional[str] = None

class RequirementCreate(BaseModel):
    customer_name: str
    customer_phone: str
    requirement_items: List[RequirementItem]
    priority: str = "normal"

class RequirementResponse(BaseModel):
    id: str
    customer_name: str
    customer_phone: str
    requirement_items: List[RequirementItem]
    priority: str
    status: str
    created_by: str
    created_at: str
          # ✅ ADD

# ---------------- STATUS UPDATE SCHEMA ----------------
class InvoiceStatusUpdate(BaseModel):
    payment_status: str

# 🔥 NEW: Add Payment Schema
class AddPaymentRequest(BaseModel):
    amount: float
    payment_mode: str = "cash"
    notes: Optional[str] = None

class InventoryTransactionResponse(BaseModel):
    id: str
    product_id: str
    product_name: str
    product_code: str
    type: str
    quantity: int
    variant_stock_after: Optional[int]  # ✅ ADD
    created_by: str        # 👈 add this

    variant_sku: Optional[str]
    stock_after: int          # ✅ use this, not remaining_stock
    created_at: str


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


class SupplierCreate(BaseModel):
    name: str
    phone: Optional[str] = None
    email: Optional[str] = None
    address: Optional[str] = None
    opening_balance: Optional[float] = 0


def resolve_product_and_variant_by_sku(db: Session, code: str):
    code = code.strip()

    # ================= 1️⃣ VARIANT FIRST =================
    products = (
        db.query(ProductModel)
        .filter(
            ProductModel.is_active == 1,
            ProductModel.variants.isnot(None)
        )
        .with_for_update()
        .all()
    )

    for p in products:
        for v in (p.variants or []):
            if v.get("v_sku", "").lower() == code.lower():
                return p, v["v_sku"]

    # ================= 2️⃣ PRODUCT CODE / SKU =================
    product = (
        db.query(ProductModel)
        .filter(
            ProductModel.is_active == 1,
            or_(
                func.lower(ProductModel.sku) == code.lower(),
                func.lower(ProductModel.product_code) == code.lower()
            )
        )
        .with_for_update()
        .first()
    )

    if product:
        return product, None

    raise HTTPException(404, "Product / Variant not found")

def with_retry(fn, retries=3):
    for attempt in range(retries):
        try:
            return fn()
        except OperationalError as e:
            if "Deadlock found" in str(e):
                time.sleep(0.2)
                continue
            raise
    raise HTTPException(500, "Inventory busy, please retry")

def validate_password_length(password: str):
    if len(password) > 72:
        raise HTTPException(
            status_code=400,
            detail="Password too long (max 72 characters)"
        )

# API Routes
# ================= REGISTER =================
@api_router.post("/auth/register", response_model=Token)
def register(
    user_data: UserRegister,
    db: Session = Depends(get_db)
):
    existing_user = (
        db.query(UserModel)
        .filter(UserModel.email == user_data.email)
        .first()
    )
    if existing_user:
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )

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

    access_token = create_access_token(
        data={
            "sub": new_user.email,
            "role": new_user.role
        }
    )

    user = User(

        id=new_user.id,
        email=new_user.email,
        name=new_user.name,
        role=new_user.role,
        created_at=new_user.created_at.isoformat()
    )

    return Token(
        access_token=access_token,
        token_type="bearer",
        user=user
    )

@api_router.post(
    "/auth/login",
    dependencies=[Depends(RateLimiter(times=10, seconds=60))]
)
def login(
    user_data: UserLogin,
    db: Session = Depends(get_db)
):
    # ✅ ADD THIS
    validate_password_length(user_data.password)

    user = (
        db.query(UserModel)
        .filter(UserModel.email == user_data.email)
        .first()
    )

    if not user or not verify_password(
        user_data.password,
        user.password
    ):
        raise HTTPException(
            status_code=401,
            detail="Incorrect email or password"
        )

    access_token = create_access_token(
        data={
            "sub": user.email,
            "role": user.role
        }
    )

    user_obj = User(
        id=user.id,
        email=user.email,
        name=user.name,
        role=user.role,
        created_at=user.created_at.isoformat()
    )

    return Token(
        access_token=access_token,
        token_type="bearer",
        user=user_obj
    )

@api_router.get("/categories/autocomplete")
def categories_autocomplete(
    q: Optional[str] = Query(None, min_length=2),
    limit: int = Query(20, ge=5, le=50),
    offset: int = Query(0, ge=0),
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Lazy-loaded autocomplete:
    - min 2 chars
    - capped results
    - offset supported
    """

    query = db.query(CategoryModel)

    # 🔒 LAZY LOAD RULE
    if q:
        q = q.strip().lower()
        query = query.filter(
            func.lower(CategoryModel.name).like(f"{q}%")  # 🔥 prefix search (FAST)
        )
    else:
        # 👇 first open → show top categories only
        query = query.order_by(CategoryModel.created_at.desc())

    rows = (
        query
        .order_by(CategoryModel.name.asc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return [
        {
            "label": c.name,
            "value": c.id,
        }
        for c in rows
    ]


@api_router.get("/categories", response_model=List[Category])
def get_categories(
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    categories = db.query(CategoryModel).all()
    return [
        Category(
            id=cat.id,
            name=cat.name,
            description=cat.description,
            created_at=cat.created_at.isoformat()
        )
        for cat in categories
    ]

@api_router.post("/categories", response_model=Category)
def create_category(
    category_data: CategoryCreate,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    new_category = CategoryModel(
        id=str(uuid.uuid4()),
        name=category_data.name,
        description=category_data.description,
        created_at=datetime.now(IST)
    )
    db.add(new_category)
    db.commit()
    db.refresh(new_category)
    
    return Category(
        id=new_category.id,
        name=new_category.name,
        description=new_category.description,
        created_at=new_category.created_at.isoformat()
    )

@api_router.delete("/categories/{category_id}")
def delete_category(
    category_id: str,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    category = db.query(CategoryModel).filter(CategoryModel.id == category_id).first()
    if not category:
        raise HTTPException(status_code=404, detail="Category not found")
    
    db.delete(category)
    db.commit()
    return {"message": "Category deleted successfully"}

def generate_draft_number(db: Session):
    year = datetime.now(IST).year

    last = (
        db.query(InvoiceModel.draft_number)
        .filter(
            InvoiceModel.invoice_type == "DRAFT",
            InvoiceModel.draft_number.like(f"DRF-{year}-%")
        )
        .order_by(InvoiceModel.draft_number.desc())
        .first()
    )

    next_num = 1
    if last and last[0]:
        next_num = int(last[0].split("-")[-1]) + 1

    return f"DRF-{year}-{next_num:04d}"

@api_router.post("/invoices/draft")
def create_invoice_draft(
    invoice_data: InvoiceCreate,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):

    # ---------------- PAYMENT STATUS ----------------
    allowed_status = ["pending", "paid", "partial", "cancelled"]

    status = invoice_data.payment_status or "pending"

    if status not in allowed_status:
        status = "pending"

    # ---------------- CUSTOMER ----------------
    customer = None

    if invoice_data.customer_id:
        customer = db.query(CustomerModel).filter(
            CustomerModel.id == invoice_data.customer_id
        ).first()

    elif invoice_data.customer_phone:
        customer = db.query(CustomerModel).filter(
            CustomerModel.phone == invoice_data.customer_phone
        ).first()

    if not customer:
        customer = CustomerModel(
            id=str(uuid.uuid4()),
            name=invoice_data.customer_name or "Walk-in",
            email=invoice_data.customer_email or f"{invoice_data.customer_phone}@draft.com",
            phone=invoice_data.customer_phone,
            address=invoice_data.customer_address,
            created_at=datetime.now(IST),
        )
        db.add(customer)
        db.flush()

    # ---------------- ITEMS ----------------
    items = []
    subtotal = 0.0

    for item in invoice_data.items:

        # skip invalid items
        if not item.product_name or not item.quantity:
            continue

        quantity = int(item.quantity or 0)
        price = float(item.price or 0)

        if quantity <= 0:
            continue

        line_total = round(price * quantity, 2)
        subtotal += line_total

        image_url = item.image_url

        # AUTO IMAGE FETCH
        if not image_url and item.product_id:
            product = db.query(ProductModel).filter(
                ProductModel.id == item.product_id
            ).first()

            if product:

                # variant image priority
                if product.variants and item.sku:
                    for v in product.variants:
                        if v.get("v_sku") == item.sku:
                            image_url = v.get("image_url")
                            break

                # fallback product image
                if not image_url and product.images:
                    image_url = product.images[0]

        items.append({
            "product_id": item.product_id,
            "sku": item.sku,
            "product_name": item.product_name,
            "quantity": quantity,
            "price": price,
            "gst_rate": float(item.gst_rate or 0),
            "total": line_total,
            "is_service": item.is_service or 0,
            "variant_info": item.variant_info,
            "image_url": image_url,
        })

    # prevent empty draft
    if len(items) == 0:
        raise HTTPException(status_code=400, detail="No valid items in invoice")

    # ---------------- ADDITIONAL CHARGES ----------------
    cleaned_additional_charges = []

    for c in invoice_data.additional_charges or []:

        label = (c.label or "").strip()
        amount = float(c.amount or 0)

        if label or amount > 0:
            cleaned_additional_charges.append({
                "label": label,
                "amount": amount
            })

    additional_total = sum(c["amount"] for c in cleaned_additional_charges)

    # ---------------- TOTAL CALCULATION ----------------
    taxable = subtotal + additional_total

    gst_amount = (
        round((taxable * float(invoice_data.gst_rate or 0)) / 100, 2)
        if invoice_data.gst_enabled else 0
    )

    total = round(
        taxable + gst_amount - float(invoice_data.discount or 0),
        2
    )

    # ---------------- SAVE DRAFT ----------------
    draft = InvoiceModel(
        id=str(uuid.uuid4()),
        invoice_type="DRAFT",
        draft_number=generate_draft_number(db),

        customer_id=customer.id,
        customer_name=customer.name,
        customer_phone=customer.phone,
        customer_address=customer.address,

        items=json.dumps(items),
        subtotal=subtotal,

        additional_charges=cleaned_additional_charges,

        gst_enabled=1 if invoice_data.gst_enabled else 0,
        gst_rate=float(invoice_data.gst_rate or 0),
        gst_amount=gst_amount,

        discount=float(invoice_data.discount or 0),
        total=total,

        # 🔥 DEFAULT STATUS
        payment_status=status,
        
        # 🔥 INITIALIZE PAYMENT HISTORY
        payment_history=[],

        created_by=current_user.id,
        created_by_name=current_user.name,
        created_at=datetime.now(IST),
    )

    db.add(draft)
    db.commit()
    db.refresh(draft)

    return {
        "id": draft.id,
        "draft_number": draft.draft_number,
        "payment_status": draft.payment_status,
        "total": draft.total
    }

@api_router.put("/invoices/draft/{draft_id}")
def update_invoice_draft(
    draft_id: str,
    invoice_data: InvoiceCreate,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # 🔒 LOCK DRAFT
    draft = (
        db.query(InvoiceModel)
        .filter(
            InvoiceModel.id == draft_id,
            InvoiceModel.invoice_type == "DRAFT"
        )
        .with_for_update()
        .first()
    )

    if not draft:
        raise HTTPException(status_code=404, detail="Draft not found")

    # ================= CUSTOMER =================
    customer = None

    if invoice_data.customer_id:
        customer = db.query(CustomerModel).filter(
            CustomerModel.id == invoice_data.customer_id
        ).first()

    elif invoice_data.customer_phone:
        customer = db.query(CustomerModel).filter(
            CustomerModel.phone == invoice_data.customer_phone
        ).first()

    if not customer:
        customer = CustomerModel(
            id=str(uuid.uuid4()),
            name=invoice_data.customer_name or "Walk-in",
            email=invoice_data.customer_email
            or f"{invoice_data.customer_phone}@draft.com",
            phone=invoice_data.customer_phone,
            address=invoice_data.customer_address,
            created_at=datetime.now(IST),
        )
        db.add(customer)
        db.flush()

    # ================= ITEMS =================
    items = []
    subtotal = 0.0

    for item in invoice_data.items:
        if not item.product_name or not item.quantity:
            continue

        quantity = int(item.quantity or 0)
        price = float(item.price or 0)

        if quantity <= 0:
            continue

        line_total = round(price * quantity, 2)
        subtotal += line_total

        image_url = item.image_url

        # AUTO IMAGE FETCH
        if not image_url and item.product_id:
            product = db.query(ProductModel).filter(
                ProductModel.id == item.product_id
            ).first()

            if product:
                if product.variants and item.sku:
                    for v in product.variants:
                        if v.get("v_sku") == item.sku:
                            image_url = v.get("image_url")
                            break

                if not image_url and product.images:
                    image_url = product.images[0]

        items.append({
            "product_id": item.product_id,
            "sku": item.sku,
            "product_name": item.product_name,
            "quantity": quantity,
            "price": price,
            "gst_rate": float(item.gst_rate or 0),
            "total": line_total,
            "is_service": item.is_service or 0,
            "variant_info": item.variant_info,
            "image_url": image_url,
        })

    if len(items) == 0:
        raise HTTPException(status_code=400, detail="No valid items in invoice")

    # ================= ADDITIONAL CHARGES =================
    cleaned_additional_charges = []

    for c in invoice_data.additional_charges or []:
        label = (c.label or "").strip()
        amount = float(c.amount or 0)

        if label or amount > 0:
            cleaned_additional_charges.append({
                "label": label,
                "amount": amount
            })

    additional_total = sum(c["amount"] for c in cleaned_additional_charges)

    # ================= TOTAL CALCULATION =================
    taxable = subtotal + additional_total

    gst_amount = (
        round((taxable * float(invoice_data.gst_rate or 0)) / 100, 2)
        if invoice_data.gst_enabled else 0
    )

    total = round(
        taxable + gst_amount - float(invoice_data.discount or 0),
        2
    )

    # ================= UPDATE DRAFT =================
    draft.customer_id = customer.id
    draft.customer_name = customer.name
    draft.customer_phone = customer.phone
    draft.customer_address = customer.address

    draft.items = json.dumps(items)
    draft.subtotal = subtotal

    draft.additional_charges = cleaned_additional_charges

    draft.gst_enabled = 1 if invoice_data.gst_enabled else 0
    draft.gst_rate = float(invoice_data.gst_rate or 0)
    draft.gst_amount = gst_amount

    draft.discount = float(invoice_data.discount or 0)
    draft.total = total

    # Keep existing payment_history intact during draft updates
    if not draft.payment_history:
        draft.payment_history = []

    flag_modified(draft, "payment_history")
    flag_modified(draft, "additional_charges")

    db.commit()
    db.refresh(draft)

    return {
        "id": draft.id,
        "draft_number": draft.draft_number,
        "total": draft.total
    }


def parse_invoice_items(items_data):
    """
    Parse invoice items from string or list format
    """
    if isinstance(items_data, str):
        try:
            return json.loads(items_data)
        except Exception:
            return []
    elif isinstance(items_data, list):
        return items_data
    return []


def generate_invoice_number(db: Session):
    """
    Generate sequential invoice number: INV-2024-0001
    """
    year = datetime.now(IST).year

    last = (
        db.query(InvoiceModel.invoice_number)
        .filter(
            InvoiceModel.invoice_type == "FINAL",
            InvoiceModel.invoice_number.like(f"INV-{year}-%")
        )
        .order_by(InvoiceModel.invoice_number.desc())
        .first()
    )

    next_num = 1
    if last and last[0]:
        try:
            next_num = int(last[0].split("-")[-1]) + 1
        except Exception:
            next_num = 1

    return f"INV-{year}-{next_num:04d}"


@api_router.post("/invoices/draft/{draft_id}/finalize")
def finalize_draft(
    draft_id: str,
    payment_status: str,
    paid_amount: Optional[float] = None,
    payment_mode: str = Query("cash"),
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    🔥 UPDATED: Finalize draft with proper payment history tracking
    """
    # ---------------- FETCH DRAFT ----------------
    draft = (
        db.query(InvoiceModel)
        .filter(
            InvoiceModel.id == draft_id,
            InvoiceModel.invoice_type == "DRAFT"
        )
        .with_for_update()
        .first()
    )

    if not draft:
        raise HTTPException(status_code=404, detail="Draft not found")

    # ---------------- VALIDATE PAYMENT STATUS ----------------
    allowed_statuses = ["pending", "paid", "partial", "cancelled"]

    if payment_status not in allowed_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid payment status. Allowed: {allowed_statuses}"
        )

    # ---------------- CALCULATE AMOUNTS ----------------
    total = float(draft.total)
    paid = 0.0
    balance = total

    # Initialize payment history list
    payment_history = []

    # ---------------- PAYMENT LOGIC ----------------
    if payment_status == "paid":
        # Full payment
        paid = total
        balance = 0
        
        # Record full payment in history
        payment_history.append({
            "id": str(uuid.uuid4()),
            "amount": total,
            "payment_mode": payment_mode,
            "payment_date": datetime.now(IST).isoformat(),
            "recorded_by": current_user.name,
            "notes": "Full payment on invoice creation"
        })

    elif payment_status == "partial":
        # Partial payment
        if not paid_amount or paid_amount <= 0:
            raise HTTPException(
                status_code=400,
                detail="Paid amount is required for partial payment"
            )

        if paid_amount >= total:
            # If paid amount equals or exceeds total, mark as paid
            paid = total
            balance = 0
            payment_status = "paid"
        else:
            paid = float(paid_amount)
            balance = total - paid

        # Record partial payment in history
        payment_history.append({
            "id": str(uuid.uuid4()),
            "amount": paid,
            "payment_mode": payment_mode,
            "payment_date": datetime.now(IST).isoformat(),
            "recorded_by": current_user.name,
            "notes": "Initial partial payment"
        })

    elif payment_status == "pending":
        # No payment yet
        paid = 0
        balance = total
        # No payment history entry for pending status

    # ---------------- UPDATE DRAFT TO FINAL INVOICE ----------------
    draft.invoice_type = "FINAL"
    draft.invoice_number = generate_invoice_number(db)

    draft.payment_status = payment_status
    draft.payment_mode = payment_mode
    draft.paid_amount = paid
    draft.balance_amount = balance

    # Set payment history
    draft.payment_history = payment_history
    flag_modified(draft, "payment_history")

    # ---------------- INVENTORY REDUCTION ----------------
    items = parse_invoice_items(draft.items)

    for item in items:
        product_id = item.get("product_id")
        sku = item.get("sku")
        quantity = int(item.get("quantity", 0))
        is_service = item.get("is_service", 0)

        # Skip services
        if is_service == 1:
            continue

        if not product_id or quantity <= 0:
            continue

        # LOCK PRODUCT
        product = (
            db.query(ProductModel)
            .filter(ProductModel.id == product_id)
            .with_for_update()
            .first()
        )

        if not product:
            continue

        # Variant SKU?
        is_variant = False
        variant_index = None

        if sku and product.variants:
            for idx, v in enumerate(product.variants):
                if v.get("v_sku") == sku:
                    is_variant = True
                    variant_index = idx
                    break

        if is_variant and variant_index is not None:
            # VARIANT STOCK
            variant = product.variants[variant_index]
            v_stock = int(variant.get("stock", 0))

            if v_stock < quantity:
                raise HTTPException(
                    status_code=400,
                    detail=f"Insufficient stock for variant {sku}"
                )

            new_v_stock = v_stock - quantity
            product.variants[variant_index]["stock"] = new_v_stock

            # Update parent stock
            product.stock = calculate_total_stock(product.variants)

            flag_modified(product, "variants")

            # LOG TRANSACTION
            db.add(
                InventoryTransaction(
                    id=str(uuid.uuid4()),
                    product_id=product.id,
                    type="OUT",
                    quantity=quantity,
                    source="invoice",
                    reason=f"Invoice {draft.invoice_number}",
                    variant_sku=sku,
                    stock_before=v_stock,
                    stock_after=product.stock,
                    variant_stock_after=new_v_stock,
                    created_by=current_user.id,
                    created_by_name=current_user.name,
                    created_at=datetime.now(IST),
                )
            )

        else:
            # PARENT STOCK
            if product.stock < quantity:
                raise HTTPException(
                    status_code=400,
                    detail=f"Insufficient stock for {product.name}"
                )

            stock_before = product.stock
            product.stock -= quantity

            # LOG TRANSACTION
            db.add(
                InventoryTransaction(
                    id=str(uuid.uuid4()),
                    product_id=product.id,
                    type="OUT",
                    quantity=quantity,
                    source="invoice",
                    reason=f"Invoice {draft.invoice_number}",
                    variant_sku=None,
                    stock_before=stock_before,
                    stock_after=product.stock,
                    variant_stock_after=None,
                    created_by=current_user.id,
                    created_by_name=current_user.name,
                    created_at=datetime.now(IST),
                )
            )

    # ---------------- LEDGER ENTRY ----------------
    if payment_status in ["paid", "partial"]:
        db.add(
            LedgerEntry(
                id=str(uuid.uuid4()),
                entry_type="invoice",
                reference_id=draft.id,
                customer_id=draft.customer_id,
                description=f"Invoice {draft.invoice_number} - {draft.customer_name}",
                debit=0,
                credit=paid,
                payment_mode=payment_mode,
                entry_date=datetime.now(IST).date(),
                created_by=current_user.id,
                created_by_name=current_user.name,
                created_at=datetime.now(IST)
            )
        )

    # ---------------- COMMIT ----------------
    db.commit()
    db.refresh(draft)

    return {
        "message": "Invoice finalized successfully",
        "invoice_number": draft.invoice_number,
        "total": draft.total,
        "paid_amount": draft.paid_amount,
        "balance_amount": draft.balance_amount,
        "payment_status": draft.payment_status
    }

# 🔥 NEW ENDPOINT: Add Payment to Existing Invoice
@api_router.post("/invoices/{invoice_id}/add-payment")
def add_payment_to_invoice(
    invoice_id: str,
    payment_data: AddPaymentRequest,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Add a payment to an existing invoice
    Supports multiple partial payments
    Automatically marks invoice as 'paid' when balance reaches 0
    """
    
    # ---------------- FETCH INVOICE ----------------
    invoice = (
        db.query(InvoiceModel)
        .filter(
            InvoiceModel.id == invoice_id,
            InvoiceModel.invoice_type == "FINAL"
        )
        .with_for_update()
        .first()
    )

    if not invoice:
        raise HTTPException(status_code=404, detail="Invoice not found")

    # ---------------- VALIDATE ----------------
    if invoice.payment_status == "paid":
        raise HTTPException(
            status_code=400,
            detail="Invoice is already fully paid"
        )

    if payment_data.amount <= 0:
        raise HTTPException(
            status_code=400,
            detail="Payment amount must be greater than 0"
        )

    current_balance = float(invoice.balance_amount or 0)

    if payment_data.amount > current_balance:
        raise HTTPException(
            status_code=400,
            detail=f"Payment amount (₹{payment_data.amount}) exceeds remaining balance (₹{current_balance})"
        )

    # ---------------- UPDATE PAYMENT ----------------
    new_paid_amount = float(invoice.paid_amount or 0) + payment_data.amount
    new_balance = float(invoice.total) - new_paid_amount

    # ---------------- ADD TO PAYMENT HISTORY ----------------
    payment_history = invoice.payment_history or []
    
    payment_history.append({
        "id": str(uuid.uuid4()),
        "amount": payment_data.amount,
        "payment_mode": payment_data.payment_mode,
        "payment_date": datetime.now(IST).isoformat(),
        "recorded_by": current_user.name,
        "notes": payment_data.notes or ""
    })

    # ---------------- UPDATE INVOICE ----------------
    invoice.paid_amount = new_paid_amount
    invoice.balance_amount = max(0, new_balance)  # Ensure never negative
    invoice.payment_history = payment_history

    # Auto-update status
    if invoice.balance_amount <= 0.01:  # Account for floating point precision
        invoice.payment_status = "paid"
        invoice.balance_amount = 0
    else:
        invoice.payment_status = "partial"

    flag_modified(invoice, "payment_history")

    # ---------------- LEDGER ENTRY ----------------
    db.add(
        LedgerEntry(
            id=str(uuid.uuid4()),
            entry_type="payment_in",
            reference_id=invoice.id,
            customer_id=invoice.customer_id,
            description=f"Payment for Invoice {invoice.invoice_number}",
            debit=0,
            credit=payment_data.amount,
            payment_mode=payment_data.payment_mode,
            entry_date=datetime.now(IST).date(),
            created_by=current_user.id,
            created_by_name=current_user.name,
            created_at=datetime.now(IST)
        )
    )

    # ---------------- COMMIT ----------------
    db.commit()
    db.refresh(invoice)

    return {
        "message": "Payment recorded successfully",
        "invoice_number": invoice.invoice_number,
        "payment_amount": payment_data.amount,
        "total_paid": invoice.paid_amount,
        "remaining_balance": invoice.balance_amount,
        "payment_status": invoice.payment_status
    }


# 🔥 NEW ENDPOINT: Get Payment History
@api_router.get("/invoices/{invoice_id}/payments")
def get_invoice_payments(
    invoice_id: str,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Retrieve complete payment history for an invoice
    """
    
    invoice = (
        db.query(InvoiceModel)
        .filter(
            InvoiceModel.id == invoice_id,
            InvoiceModel.invoice_type == "FINAL"
        )
        .first()
    )

    if not invoice:
        raise HTTPException(status_code=404, detail="Invoice not found")

    payment_history = invoice.payment_history or []

    # Calculate running balance for each payment
    total = float(invoice.total)
    running_balance = total
    
    enriched_history = []
    for payment in payment_history:
        running_balance -= float(payment.get("amount", 0))
        
        enriched_history.append({
            **payment,
            "balance_after": max(0, running_balance)
        })

    return {
        "invoice_number": invoice.invoice_number,
        "total_amount": invoice.total,
        "paid_amount": invoice.paid_amount,
        "balance_amount": invoice.balance_amount,
        "payment_status": invoice.payment_status,
        "payments": enriched_history
    }


@api_router.get("/invoices/drafts")
def get_all_drafts(
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    drafts = (
        db.query(InvoiceModel)
        .filter(InvoiceModel.invoice_type == "DRAFT")
        .order_by(InvoiceModel.created_at.desc())
        .all()
    )

    result = []

    for d in drafts:
        items = parse_invoice_items(d.items)

        result.append({
            "id": d.id,
            "draft_number": d.draft_number,
            "customer_id": d.customer_id,
            "customer_name": d.customer_name,
            "customer_phone": d.customer_phone,
            "customer_address": d.customer_address,
            "items": items,
            "subtotal": d.subtotal,
            "gst_enabled": bool(d.gst_enabled),
            "gst_rate": d.gst_rate,
            "gst_amount": d.gst_amount,
            "discount": d.discount,
            "additional_charges": d.additional_charges or [],
            "total": d.total,
            "payment_status": d.payment_status,
            "created_at": d.created_at.isoformat()
        })

    return result


@api_router.get("/invoices/draft/{draft_id}")
def get_single_draft(
    draft_id: str,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    draft = (
        db.query(InvoiceModel)
        .filter(
            InvoiceModel.id == draft_id,
            InvoiceModel.invoice_type == "DRAFT"
        )
        .first()
    )

    if not draft:
        raise HTTPException(status_code=404, detail="Draft not found")

    items = parse_invoice_items(draft.items)

    return {
        "id": draft.id,
        "draft_number": draft.draft_number,
        "customer_id": draft.customer_id,
        "customer_name": draft.customer_name,
        "customer_phone": draft.customer_phone,
        "customer_address": draft.customer_address,
        "items": items,
        "subtotal": draft.subtotal,
        "gst_enabled": bool(draft.gst_enabled),
        "gst_rate": draft.gst_rate,
        "gst_amount": draft.gst_amount,
        "discount": draft.discount,
        "additional_charges": draft.additional_charges or [],
        "total": draft.total,
        "payment_status": draft.payment_status,
        "created_at": draft.created_at.isoformat()
    }


@api_router.delete("/invoices/draft/{draft_id}")
def delete_draft(
    draft_id: str,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    draft = (
        db.query(InvoiceModel)
        .filter(
            InvoiceModel.id == draft_id,
            InvoiceModel.invoice_type == "DRAFT"
        )
        .first()
    )

    if not draft:
        raise HTTPException(status_code=404, detail="Draft not found")

    db.delete(draft)
    db.commit()

    return {"message": "Draft deleted successfully"}


def calculate_invoice_totals(invoice):
    """
    Calculate subtotal, GST, and total for an invoice
    Handles additional charges properly
    """
    items = parse_invoice_items(invoice.items)

    # Items subtotal
    items_subtotal = sum(float(item.get("total", 0)) for item in items)

    # Additional charges total
    additional_charges = invoice.additional_charges or []
    additional_total = sum(
        float(c.get("amount", 0))
        for c in additional_charges
        if isinstance(c, dict)
    )

    # Taxable amount (before GST and discount)
    subtotal = items_subtotal + additional_total

    # GST calculation
    gst_amount = float(invoice.gst_amount or 0)

    # Final total
    total = subtotal + gst_amount - float(invoice.discount or 0)

    return {
        "subtotal": round(subtotal, 2),
        "gst_amount": round(gst_amount, 2),
        "total": round(total, 2)
    }


@api_router.get("/invoices")
def get_all_invoices(
    page: int = 1,
    limit: int = 20,
    status: Optional[str] = None,
    range: Optional[str] = None,
    month: Optional[str] = None,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    🔥 UPDATED: Returns invoices with payment_history
    """
    query = db.query(InvoiceModel).filter(
        InvoiceModel.invoice_type == "FINAL"
    )

    # Status filter
    if status:
        query = query.filter(InvoiceModel.payment_status == status)

    # Date range filter
    now = datetime.now(IST)

    if range == "last10":
        start_date = now - timedelta(days=10)
        query = query.filter(InvoiceModel.created_at >= start_date)

    elif range == "last30":
        start_date = now - timedelta(days=30)
        query = query.filter(InvoiceModel.created_at >= start_date)

    # Month filter (format: "2024-05")
    if month:
        try:
            year, month_num = month.split("-")
            start = datetime(int(year), int(month_num), 1, tzinfo=IST)

            if int(month_num) == 12:
                end = datetime(int(year) + 1, 1, 1, tzinfo=IST)
            else:
                end = datetime(int(year), int(month_num) + 1, 1, tzinfo=IST)

            query = query.filter(
                InvoiceModel.created_at >= start,
                InvoiceModel.created_at < end
            )
        except Exception:
            pass

    # Pagination
    total = query.count()
    invoices = (
        query.order_by(InvoiceModel.created_at.desc())
        .offset((page - 1) * limit)
        .limit(limit)
        .all()
    )

    # Format response
    result = []

    for inv in invoices:
        items = parse_invoice_items(inv.items)

        result.append({
            "id": inv.id,
            "invoice_number": inv.invoice_number,
            "customer_id": inv.customer_id,
            "customer_name": inv.customer_name,
            "customer_phone": inv.customer_phone,
            "customer_address": inv.customer_address,
            "items": items,
            "subtotal": inv.subtotal,
            "gst_enabled": bool(inv.gst_enabled),
            "gst_rate": inv.gst_rate,
            "gst_amount": inv.gst_amount,
            "discount": inv.discount,
            "additional_charges": inv.additional_charges or [],
            "total": inv.total,
            "paid_amount": inv.paid_amount,
            "balance_amount": inv.balance_amount,
            "payment_status": inv.payment_status,
            "payment_mode": inv.payment_mode,
            "payment_history": inv.payment_history or [],  # 🔥 ADDED
            "created_by": inv.created_by_name,
            "created_at": inv.created_at.isoformat()
        })

    return {
        "data": result,
        "pagination": {
            "page": page,
            "limit": limit,
            "total": total,
            "total_pages": math.ceil(total / limit)
        }
    }


@api_router.get("/invoices/{invoice_id}")
def get_single_invoice(
    invoice_id: str,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    🔥 UPDATED: Returns single invoice with payment_history
    """
    invoice = (
        db.query(InvoiceModel)
        .filter(
            InvoiceModel.id == invoice_id,
            InvoiceModel.invoice_type == "FINAL"
        )
        .first()
    )

    if not invoice:
        raise HTTPException(status_code=404, detail="Invoice not found")

    items = parse_invoice_items(invoice.items)

    return {
        "id": invoice.id,
        "invoice_number": invoice.invoice_number,
        "customer_id": invoice.customer_id,
        "customer_name": invoice.customer_name,
        "customer_phone": invoice.customer_phone,
        "customer_address": invoice.customer_address,
        "items": items,
        "subtotal": invoice.subtotal,
        "gst_enabled": bool(invoice.gst_enabled),
        "gst_rate": invoice.gst_rate,
        "gst_amount": invoice.gst_amount,
        "discount": invoice.discount,
        "additional_charges": invoice.additional_charges or [],
        "total": invoice.total,
        "paid_amount": invoice.paid_amount,
        "balance_amount": invoice.balance_amount,
        "payment_status": invoice.payment_status,
        "payment_mode": invoice.payment_mode,
        "payment_history": invoice.payment_history or [],  # 🔥 ADDED
        "created_by": invoice.created_by_name,
        "created_at": invoice.created_at.isoformat()
    }


@api_router.patch("/invoices/{invoice_id}/status")
def update_invoice_status(
    invoice_id: str,
    status_update: InvoiceStatusUpdate,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update invoice payment status
    """
    invoice = (
        db.query(InvoiceModel)
        .filter(
            InvoiceModel.id == invoice_id,
            InvoiceModel.invoice_type == "FINAL"
        )
        .first()
    )

    if not invoice:
        raise HTTPException(status_code=404, detail="Invoice not found")

    allowed_statuses = ["pending", "paid", "partial", "cancelled"]

    if status_update.payment_status not in allowed_statuses:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid status. Allowed: {allowed_statuses}"
        )

    invoice.payment_status = status_update.payment_status

    db.commit()
    db.refresh(invoice)

    return {
        "message": "Invoice status updated",
        "invoice_number": invoice.invoice_number,
        "new_status": invoice.payment_status
    }


@api_router.get("/products", response_model=List[Product])
def get_products(
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    products = (
        db.query(ProductModel)
        .filter(ProductModel.is_active == 1)
        .all()
    )

    return [
        Product(
            id=p.id,
            product_code=p.product_code,
            sku=p.sku,
            name=p.name,
            description=p.description,
            category_id=p.category_id,
            category_name=p.category.name if p.category else None,
            cost_price=p.cost_price,
            min_selling_price=p.min_selling_price,
            selling_price=p.selling_price,
            stock=p.stock,
            min_stock=p.min_stock,
            is_service=p.is_service,
            variants=p.variants or [],
            images=p.images or [],
            qr_code_url=p.qr_code_url,
            created_at=p.created_at
        )
        for p in products
    ]


@api_router.get("/products/search")
def search_products(
    q: str,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    q = q.strip().lower()

    products = (
        db.query(ProductModel)
        .filter(
            ProductModel.is_active == 1,
            or_(
                func.lower(ProductModel.name).like(f"%{q}%"),
                func.lower(ProductModel.sku).like(f"%{q}%"),
                func.lower(ProductModel.product_code).like(f"%{q}%"),
            )
        )
        .limit(20)
        .all()
    )

    return [
        {
            "id": p.id,
            "product_code": p.product_code,
            "sku": p.sku,
            "name": p.name,
            "description": p.description,
            "category_id": p.category_id,
            "category_name": p.category.name if p.category else None,
            "selling_price": p.selling_price,
            "min_selling_price": p.min_selling_price,
            "cost_price": p.cost_price,
            "stock": p.stock,
            "min_stock": p.min_stock,
            "is_service": p.is_service,
            "variants": p.variants or [],
            "images": p.images or [],
            "qr_code_url": p.qr_code_url,
        }
        for p in products
    ]


@api_router.post("/products", response_model=Product)
def create_product(
    product_data: ProductCreate,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role not in ["admin", "store_handler"]:
        raise HTTPException(status_code=403, detail="Admin or Store Handler only")

    # Generate product code
    product_code = generate_product_code(db)

    # Calculate total stock from variants
    total_stock = calculate_total_stock(product_data.variants)

    # QR Code generation for parent product
    parent_qr_data = {
        "type": "product",
        "product_code": product_code,
        "sku": product_data.sku,
        "name": product_data.name,
    }
    parent_qr_url = generate_qr(parent_qr_data)

    # Generate QR for each variant
    enriched_variants = []

    for v in product_data.variants:
        variant_qr_data = {
            "type": "variant",
            "product_code": product_code,
            "parent_sku": product_data.sku,
            "v_sku": v.v_sku,
            "variant_name": v.variant_name,
            "color": v.color,
            "size": v.size,
        }

        variant_qr_url = generate_qr(variant_qr_data)

        enriched_variants.append({
            "v_sku": v.v_sku,
            "variant_name": v.variant_name,
            "color": v.color,
            "size": v.size,
            "stock": v.stock,
            "image_url": v.image or v.image_url,
            "qr_code_url": variant_qr_url,
            "v_selling_price": v.v_selling_price or product_data.selling_price,
        })

    # Create new product
    new_product = ProductModel(
        id=str(uuid.uuid4()),
        product_code=product_code,
        sku=product_data.sku,
        name=product_data.name,
        description=product_data.description,
        category_id=product_data.category_id,
        cost_price=product_data.cost_price,
        min_selling_price=product_data.min_selling_price,
        selling_price=product_data.selling_price,
        stock=total_stock,
        min_stock=product_data.min_stock,
        is_service=product_data.is_service,
        variants=enriched_variants,
        images=product_data.images or [],
        qr_code_url=parent_qr_url,
        is_active=1,
        created_at=datetime.now(IST)
    )

    db.add(new_product)
    db.commit()
    db.refresh(new_product)

    return Product(
        id=new_product.id,
        product_code=new_product.product_code,
        sku=new_product.sku,
        name=new_product.name,
        description=new_product.description,
        category_id=new_product.category_id,
        category_name=new_product.category.name if new_product.category else None,
        cost_price=new_product.cost_price,
        min_selling_price=new_product.min_selling_price,
        selling_price=new_product.selling_price,
        stock=new_product.stock,
        min_stock=new_product.min_stock,
        is_service=new_product.is_service,
        variants=new_product.variants or [],
        images=new_product.images or [],
        qr_code_url=new_product.qr_code_url,
        created_at=new_product.created_at
    )


@api_router.put("/products/{product_id}")
def update_product(
    product_id: str,
    product_data: ProductCreate,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role not in ["admin", "store_handler"]:
        raise HTTPException(status_code=403, detail="Admin or Store Handler only")

    product = (
        db.query(ProductModel)
        .filter(ProductModel.id == product_id)
        .with_for_update()
        .first()
    )

    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    # Prevent SKU update if product has inventory or invoice history
    if product_data.sku != product.sku:
        if has_inventory(db, product_id):
            raise HTTPException(
                status_code=400,
                detail="Cannot change SKU - product has inventory history"
            )

        if has_invoice(db, product_id):
            raise HTTPException(
                status_code=400,
                detail="Cannot change SKU - product has invoice history"
            )

    # Calculate new total stock
    total_stock = calculate_total_stock(product_data.variants)

    # Update product fields
    product.sku = product_data.sku
    product.name = product_data.name
    product.description = product_data.description
    product.category_id = product_data.category_id
    product.cost_price = product_data.cost_price
    product.min_selling_price = product_data.min_selling_price
    product.selling_price = product_data.selling_price
    product.stock = total_stock
    product.min_stock = product_data.min_stock
    product.is_service = product_data.is_service
    product.variants = product_data.variants
    product.images = product_data.images or []

    flag_modified(product, "variants")
    flag_modified(product, "images")

    db.commit()
    db.refresh(product)

    return {
        "message": "Product updated successfully",
        "id": product.id,
        "product_code": product.product_code,
        "sku": product.sku,
        "name": product.name,
        "stock": product.stock
    }


@api_router.delete("/products/{product_id}")
def delete_product(
    product_id: str,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role not in ["admin", "store_handler"]:
        raise HTTPException(status_code=403, detail="Admin or Store Handler only")

    product = db.query(ProductModel).filter(ProductModel.id == product_id).first()

    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    # Soft delete
    product.is_active = 0
    db.commit()

    return {"message": "Product deleted successfully"}


@api_router.post("/products/material-inward")
def material_inward(
    data: MaterialInwardRequest,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Add stock to a product or variant
    """
    product = (
        db.query(ProductModel)
        .filter(ProductModel.id == data.product_id)
        .with_for_update()
        .first()
    )

    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    # Check if SKU is a variant
    is_variant = False
    variant_index = None

    if data.sku and product.variants:
        for idx, v in enumerate(product.variants):
            if v.get("v_sku") == data.sku:
                is_variant = True
                variant_index = idx
                break

    stock_before = product.stock

    if is_variant and variant_index is not None:
        # Update variant stock
        variant = product.variants[variant_index]
        v_stock_before = int(variant.get("stock", 0))
        new_v_stock = v_stock_before + data.quantity

        product.variants[variant_index]["stock"] = new_v_stock

        # Recalculate parent stock
        product.stock = calculate_total_stock(product.variants)

        flag_modified(product, "variants")

        # Log transaction
        db.add(
            InventoryTransaction(
                id=str(uuid.uuid4()),
                product_id=product.id,
                type="IN",
                quantity=data.quantity,
                source="manual",
                reason="Material Inward",
                variant_sku=data.sku,
                stock_before=v_stock_before,
                stock_after=product.stock,
                variant_stock_after=new_v_stock,
                created_by=current_user.id,
                created_by_name=current_user.name,
                created_at=datetime.now(IST),
            )
        )

    else:
        # Update parent stock
        product.stock += data.quantity

        # Log transaction
        db.add(
            InventoryTransaction(
                id=str(uuid.uuid4()),
                product_id=product.id,
                type="IN",
                quantity=data.quantity,
                source="manual",
                reason="Material Inward",
                variant_sku=None,
                stock_before=stock_before,
                stock_after=product.stock,
                variant_stock_after=None,
                created_by=current_user.id,
                created_by_name=current_user.name,
                created_at=datetime.now(IST),
            )
        )

    db.commit()
    db.refresh(product)

    return {
        "message": "Stock added successfully",
        "product_name": product.name,
        "new_stock": product.stock
    }


@api_router.post("/products/material-outward")
def material_outward(
    data: MaterialOutwardRequest,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Remove stock from a product or variant
    """
    product = (
        db.query(ProductModel)
        .filter(ProductModel.id == data.product_id)
        .with_for_update()
        .first()
    )

    if not product:
        raise HTTPException(status_code=404, detail="Product not found")

    # Check if SKU is a variant
    is_variant = False
    variant_index = None

    if data.sku and product.variants:
        for idx, v in enumerate(product.variants):
            if v.get("v_sku") == data.sku:
                is_variant = True
                variant_index = idx
                break

    stock_before = product.stock

    if is_variant and variant_index is not None:
        # Update variant stock
        variant = product.variants[variant_index]
        v_stock = int(variant.get("stock", 0))

        if v_stock < data.quantity:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient variant stock. Available: {v_stock}"
            )

        new_v_stock = v_stock - data.quantity
        product.variants[variant_index]["stock"] = new_v_stock

        # Recalculate parent stock
        product.stock = calculate_total_stock(product.variants)

        flag_modified(product, "variants")

        # Log transaction
        db.add(
            InventoryTransaction(
                id=str(uuid.uuid4()),
                product_id=product.id,
                type="OUT",
                quantity=data.quantity,
                source="manual",
                reason=data.reason,
                variant_sku=data.sku,
                stock_before=v_stock,
                stock_after=product.stock,
                variant_stock_after=new_v_stock,
                created_by=current_user.id,
                created_by_name=current_user.name,
                created_at=datetime.now(IST),
            )
        )

    else:
        # Update parent stock
        if product.stock < data.quantity:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient stock. Available: {product.stock}"
            )

        product.stock -= data.quantity

        # Log transaction
        db.add(
            InventoryTransaction(
                id=str(uuid.uuid4()),
                product_id=product.id,
                type="OUT",
                quantity=data.quantity,
                source="manual",
                reason=data.reason,
                variant_sku=None,
                stock_before=stock_before,
                stock_after=product.stock,
                variant_stock_after=None,
                created_by=current_user.id,
                created_by_name=current_user.name,
                created_at=datetime.now(IST),
            )
        )

    db.commit()
    db.refresh(product)

    return {
        "message": "Stock removed successfully",
        "product_name": product.name,
        "new_stock": product.stock
    }


@api_router.post("/products/material-inward-by-sku")
def material_inward_by_sku(
    data: MaterialInwardBySkuRequest,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Add stock using SKU (product or variant)
    """
    product, variant_sku = resolve_product_and_variant_by_sku(db, data.sku)

    # Delegate to material_inward
    return material_inward(
        MaterialInwardRequest(
            product_id=product.id,
            quantity=data.quantity,
            sku=variant_sku
        ),
        current_user,
        db
    )


@api_router.post("/products/material-outward-by-sku")
def material_outward_by_sku(
    data: MaterialOutwardBySkuRequest,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Remove stock using SKU (product or variant)
    """
    product, variant_sku = resolve_product_and_variant_by_sku(db, data.sku)

    # Delegate to material_outward
    return material_outward(
        MaterialOutwardRequest(
            product_id=product.id,
            quantity=data.quantity,
            reason=data.reason,
            sku=variant_sku
        ),
        current_user,
        db
    )


@api_router.get("/inventory/transactions", response_model=List[InventoryTransactionResponse])
def get_inventory_transactions(
    page: int = 1,
    limit: int = 50,
    product_id: Optional[str] = None,
    type: Optional[str] = None,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    query = db.query(InventoryTransaction, ProductModel).join(
        ProductModel, InventoryTransaction.product_id == ProductModel.id
    )

    if product_id:
        query = query.filter(InventoryTransaction.product_id == product_id)

    if type:
        query = query.filter(InventoryTransaction.type == type)

    total = query.count()

    transactions = (
        query.order_by(InventoryTransaction.created_at.desc())
        .offset((page - 1) * limit)
        .limit(limit)
        .all()
    )

    return [
        InventoryTransactionResponse(
            id=t.id,
            product_id=t.product_id,
            product_name=p.name,
            product_code=p.product_code,
            type=t.type,
            quantity=t.quantity,
            variant_sku=t.variant_sku,
            stock_after=t.stock_after,
            variant_stock_after=t.variant_stock_after,
            created_by=t.created_by_name or "Unknown",
            created_at=t.created_at.isoformat()
        )
        for t, p in transactions
    ]


@api_router.get("/customers/search")
def search_customer(
    phone: str,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    customer = (
        db.query(CustomerModel)
        .filter(CustomerModel.phone == phone)
        .first()
    )

    if not customer:
        return None

    return {
        "id": customer.id,
        "name": customer.name,
        "email": customer.email,
        "phone": customer.phone,
        "address": customer.address
    }


@api_router.post("/customers", response_model=Customer)
def create_customer(
    customer_data: CustomerCreate,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    new_customer = CustomerModel(
        id=str(uuid.uuid4()),
        name=customer_data.name,
        email=customer_data.email,
        phone=customer_data.phone,
        address=customer_data.address,
        created_at=datetime.now(IST)
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
        created_at=new_customer.created_at.isoformat()
    )


@api_router.get("/dashboard/stats")
def get_dashboard_stats(
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # Total sales
    total_sales = (
        db.query(func.sum(InvoiceModel.total))
        .filter(
            InvoiceModel.invoice_type == "FINAL",
            InvoiceModel.payment_status == "paid"
        )
        .scalar()
        or 0
    )

    # Total orders
    total_orders = (
        db.query(func.count(InvoiceModel.id))
        .filter(InvoiceModel.invoice_type == "FINAL")
        .scalar()
        or 0
    )

    # Total customers
    total_customers = db.query(func.count(CustomerModel.id)).scalar() or 0

    # Low stock items
    low_stock_items = (
        db.query(func.count(ProductModel.id))
        .filter(
            ProductModel.is_active == 1,
            ProductModel.stock <= ProductModel.min_stock
        )
        .scalar()
        or 0
    )

    # Recent invoices
    recent = (
        db.query(InvoiceModel)
        .filter(InvoiceModel.invoice_type == "FINAL")
        .order_by(InvoiceModel.created_at.desc())
        .limit(5)
        .all()
    )

    recent_invoices = []

    for inv in recent:
        items = parse_invoice_items(inv.items)

        recent_invoices.append({
            "id": inv.id,
            "invoice_number": inv.invoice_number,
            "customer_name": inv.customer_name,
            "customer_phone": inv.customer_phone,
            "items": items,
            "total": inv.total,
            "payment_status": inv.payment_status,
            "created_at": inv.created_at.isoformat()
        })

    return {
        "total_sales": float(total_sales),
        "total_orders": total_orders,
        "total_customers": total_customers,
        "low_stock_items": low_stock_items,
        "recent_invoices": recent_invoices
    }


@api_router.get("/dashboard/low-stock")
def low_stock_products(
    limit: int = 10,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    products = (
        db.query(ProductModel)
        .filter(
            ProductModel.is_active == 1,
            ProductModel.stock <= ProductModel.min_stock
        )
        .order_by(ProductModel.stock.asc())
        .limit(limit)
        .all()
    )

    return [
        {
            "product_name": p.name,
            "stock": p.stock,
            "min_stock": p.min_stock
        }
        for p in products
    ]


@api_router.get("/dashboard/top-products")
def top_products(
    limit: int = 5,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    results = (
        db.query(
            ProductModel.name,
            func.sum(InventoryTransaction.quantity).label("qty")
        )
        .join(ProductModel, ProductModel.id == InventoryTransaction.product_id)
        .filter(
            InventoryTransaction.type == "OUT",
            ProductModel.is_service == 0
        )
        .group_by(ProductModel.name)
        .order_by(func.sum(InventoryTransaction.quantity).desc())
        .limit(limit)
        .all()
      
    )

    return [
        { "name": r.name, "quantity": int(r.qty or 0) }
        for r in results
    ]

@api_router.get("/dashboard/inventory-movement")
def inventory_movement(
    days: int = 7,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    start = datetime.now(IST) - timedelta(days=days)

    results = (
        db.query(
            func.date(InventoryTransaction.created_at).label("day"),
            func.sum(case((InventoryTransaction.type == "IN", InventoryTransaction.quantity), else_=0)).label("inward"),
            func.sum(case((InventoryTransaction.type == "OUT", InventoryTransaction.quantity), else_=0)).label("outward"),
        )
        .filter(InventoryTransaction.created_at >= start)
        .group_by("day")
        .order_by("day")
        .all()
    )

    return [
        {
            "day": r.day.strftime("%d %b"),
            "inward": int(r.inward or 0),
            "outward": int(r.outward or 0)
        }
        for r in results
    ]

@api_router.get("/dashboard/activity")
def dashboard_activity(
    limit: int = 10,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    invoices = (
    db.query(InvoiceModel)
    .filter(InvoiceModel.invoice_type == "FINAL")
    .order_by(InvoiceModel.created_at.desc())
    .limit(limit)
    .all()
)


    inventory = (
        db.query(InventoryTransaction, ProductModel)
        .join(ProductModel)
        .order_by(InventoryTransaction.created_at.desc())
        .limit(limit)
        .all()
    )

    activity = []

    for inv in invoices:
        activity.append({
            "type": "invoice",
            "text": f"Invoice {inv.invoice_number} – ₹{inv.total}",
            "date": inv.created_at.isoformat()
        })

    for txn, prod in inventory:
        activity.append({
        "type": "inventory",
        "text": f"{txn.type} – {prod.name} ({txn.quantity}) by {txn.created_by_name}",
        "date": txn.created_at.isoformat()
    })


    return sorted(activity, key=lambda x: x["date"], reverse=True)[:limit]

@api_router.get("/dashboard/hourly-sales")
def hourly_sales_today(
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # 🔹 Use local server time (IMPORTANT for MySQL)
    now = datetime.now(IST)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    results = (
        db.query(
            func.hour(InvoiceModel.created_at).label("hour"),
            func.sum(InvoiceModel.total).label("total")
        )
        .filter(
            InvoiceModel.invoice_type == "FINAL",

            InvoiceModel.created_at >= start,
            InvoiceModel.created_at <= now,
            InvoiceModel.payment_status == "paid"
        )
        .group_by("hour")
        .order_by("hour")
        .all()
    )

    data = []
    for hour, total in results:
        data.append({
            "label": f"{hour:02d}:00–{hour+1:02d}:00",
            "total": float(total or 0)
        })

    return data


@api_router.get("/dashboard/sales", response_model=List[SalesChartItem])
def get_sales_data(
    filter: str = "today",
    year: int | None = None,
    month: int | None = None,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    now = datetime.now(IST)

    # ---------- DATE RANGE ----------
    if filter == "today":
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end = now

    elif filter == "yesterday":
        y = now - timedelta(days=1)
        start = y.replace(hour=0, minute=0, second=0, microsecond=0)
        end = y.replace(hour=23, minute=59, second=59)

    elif filter == "last_10_days":
        start = now - timedelta(days=10)
        end = now

    elif filter == "last_30_days":
        start = now - timedelta(days=30)
        end = now

    elif filter == "month":
        start = datetime(year, month, 1, tzinfo=IST)
        if month == 12:
            end = datetime(year + 1, 1, 1, tzinfo=IST)
        else:
            end = datetime(year, month + 1, 1, tzinfo=IST)

    else:
        raise HTTPException(status_code=400, detail="Invalid filter")

    results = (
        db.query(
            func.date(InvoiceModel.created_at).label("day"),
            func.sum(InvoiceModel.total).label("total"),
            func.sum(case((InvoiceModel.payment_status == "paid", InvoiceModel.total), else_=0)).label("paid"),
            func.sum(case((InvoiceModel.payment_status == "pending", InvoiceModel.total), else_=0)).label("pending"),
            func.sum(case((InvoiceModel.payment_status == "partial", InvoiceModel.total), else_=0)).label("partial"),

        )
        .filter(
                InvoiceModel.invoice_type == "FINAL",

            InvoiceModel.created_at >= start,
            InvoiceModel.created_at <= end
        )
        .group_by("day")
        .order_by("day")
        .all()
    )

    return [
        SalesChartItem(
            name=row.day.strftime("%d %b"),
            total=float(row.total or 0),
            paid=float(row.paid or 0),
            pending=float(row.pending or 0),
            partial=float(row.partial or 0),
        )
        for row in results
    ]

def upload_qr_to_cloudinary(pil_image, folder="qr"):
    import io

    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)

    result = cloudinary.uploader.upload(
        buffer,
        folder=folder,
        resource_type="image"
    )
    return result["secure_url"]

@api_router.get("/products/sku/{code}")
def get_product_by_sku(
    code: str,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    code = code.strip()

    product, variant_sku = resolve_product_and_variant_by_sku(db, code)

    # ================= VARIANT LEVEL =================
    if variant_sku:
        for v in product.variants or []:
            if v.get("v_sku") == variant_sku:
                v_price = (
                    v.get("v_selling_price")
                    or v.get("v_price")
                    or product.selling_price
                )

                return {
                    "id": product.id,
                    "product_code": product.product_code,
                    "sku": product.sku,
                    "name": product.name,
                    "description": product.description,
                    "category_id": product.category_id,
                    "category_name": product.category.name if product.category else None,
                    "selling_price": v_price,
                    "min_selling_price": product.min_selling_price,
                    "stock": v.get("stock", 0),
                    "min_stock": product.min_stock,
                    "is_service": product.is_service,
                    "images": product.images or [],
                    "variants": product.variants or [],
                    "variant": {
                        "v_sku": v.get("v_sku"),
                        "variant_name": v.get("variant_name"),
                        "size": v.get("size"),
                        "color": v.get("color"),
                        "stock": v.get("stock", 0),
                        "v_price": v_price
                    }
                }

        raise HTTPException(500, "Variant resolved but missing")

    # ================= PRODUCT LEVEL =================
    return {
        "id": product.id,
        "product_code": product.product_code,
        "sku": product.sku,
        "name": product.name,
        "description": product.description,
        "category_id": product.category_id,
        "category_name": product.category.name if product.category else None,
        "selling_price": product.selling_price,
        "min_selling_price": product.min_selling_price,
        "stock": product.stock,
        "min_stock": product.min_stock,
        "is_service": product.is_service,
        "images": product.images or [],
        "variants": product.variants or [],
        "variant": None
    }



@api_router.get("/invoices/{invoice_id}/pdf")
def download_invoice_pdf(invoice_id: str, db: Session = Depends(get_db)):
    invoice = db.query(InvoiceModel).filter(
        InvoiceModel.id == invoice_id
    ).first()

    if not invoice:
        raise HTTPException(404, "Invoice not found")

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    elements = []

    styles = getSampleStyleSheet()

    elements.append(Paragraph(f"Invoice: {invoice.invoice_number}", styles["Title"]))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(f"Customer: {invoice.customer_name}", styles["Normal"]))
    elements.append(Paragraph(f"Phone: {invoice.customer_phone}", styles["Normal"]))
    elements.append(Spacer(1, 10))

    items = json.loads(invoice.items)

    data = [["Product", "Qty", "Price", "Total"]]

    for item in items:
        data.append([
            item.get("product_name"),
            str(item.get("quantity")),
            str(item.get("price")),
            str(item.get("total")),
        ])

    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(f"Total: ₹{invoice.total}", styles["Heading2"]))

    doc.build(elements)
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename=Invoice-{invoice.invoice_number}.pdf"
        }
    )

####################################### ACCOUNTING LEDGER ENTRY ENDPOINTS #######################################

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

    # Ledger Entry - Expenses are money going OUT
    # Debit = money going out (shown in red)
    db.add(
        LedgerEntry(
            id=str(uuid.uuid4()),
            entry_type="expense",
            description=data.title,
            debit=data.amount,  # Debit = money going out (shown in red)
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

@api_router.get("/accounts/summary")
def accounts_summary(
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

    # ================= CASH =================
    cash_entries = query.filter(
        LedgerEntry.payment_mode == "cash"
    ).all()

    cash_balance = 0

    for e in cash_entries:
        # Credit = money in (increases balance)
        # Debit = money out (decreases balance)
        cash_balance += e.credit - e.debit


    # ================= BANK / UPI =================
    bank_entries = query.filter(
        LedgerEntry.payment_mode.in_(["bank", "upi", "cheque"])
    ).all()

    bank_balance = 0

    for e in bank_entries:
        # Credit = money in (increases balance)
        # Debit = money out (decreases balance)
        bank_balance += e.credit - e.debit


    # ================= CUSTOMER OUTSTANDING =================
    customers_outstanding = (
        db.query(func.sum(CustomerModel.current_balance)).scalar() or 0
    )

    # ================= SUPPLIER OUTSTANDING =================
    suppliers_outstanding = (
        db.query(func.sum(SupplierModel.current_balance)).scalar() or 0
    )

    # ================= PENDING CHEQUES =================
    pending_cheques = (
        db.query(func.sum(ChequeModel.amount))
        .filter(ChequeModel.status == "pending")
        .scalar()
        or 0
    )

    return {
        "cash_balance": float(cash_balance),
        "bank_balance": float(bank_balance),
        "total_balance": float(cash_balance + bank_balance),
        "customers_outstanding": float(customers_outstanding),
        "suppliers_outstanding": float(suppliers_outstanding),
        "pending_cheques": float(pending_cheques),
    }

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
    data: SupplierCreate,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    supplier = SupplierModel(
        id=str(uuid.uuid4()),
        name=data.name,
        phone=data.phone,
        email=data.email,
        address=data.address,
        opening_balance=data.opening_balance,
        current_balance=data.opening_balance,
        created_at=datetime.now(IST)
    )

    db.add(supplier)
    db.commit()
    db.refresh(supplier)

    return {
        "message": "Supplier created successfully",
        "id": supplier.id
    }

@api_router.get("/accounts/cashbook")
def cashbook(
    page: int = 1,
    limit: int = 50,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    role: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user)
):

    if current_user.role not in ["admin", "store_handler"]:
        raise HTTPException(status_code=403, detail="Access restricted")

    base_query = db.query(LedgerEntry).filter(LedgerEntry.payment_mode == "cash")

    if start_date:
        base_query = base_query.filter(LedgerEntry.entry_date >= start_date)
    if end_date:
        base_query = base_query.filter(LedgerEntry.entry_date <= end_date)

    total = base_query.count()

    rows = base_query.order_by(
        LedgerEntry.entry_date.desc()
    ).offset((page - 1) * limit).limit(limit).all()

    opening_balance = db.query(
        func.coalesce(func.sum(LedgerEntry.credit - LedgerEntry.debit), 0)
    ).filter(
        LedgerEntry.payment_mode == "cash",
        LedgerEntry.entry_date < rows[-1].entry_date if rows else None
    ).scalar() or 0

    balance = opening_balance
    data = []

    for r in reversed(rows):
        balance += r.credit - r.debit

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

    base_query = db.query(LedgerEntry).filter(or_(
        LedgerEntry.payment_mode == "bank",
        LedgerEntry.payment_mode == "upi",
        LedgerEntry.payment_mode == "cheque"
    ))

    if start_date:
        base_query = base_query.filter(LedgerEntry.entry_date >= start_date)
    if end_date:
        base_query = base_query.filter(LedgerEntry.entry_date <= end_date)

    total = base_query.count()

    rows = base_query.order_by(
        LedgerEntry.entry_date.desc()
    ).offset((page - 1) * limit).limit(limit).all()

    opening_balance = db.query(
        func.coalesce(func.sum(LedgerEntry.credit - LedgerEntry.debit), 0)
    ).filter(
        or_(
            LedgerEntry.payment_mode == "bank",
            LedgerEntry.payment_mode == "upi",
            LedgerEntry.payment_mode == "cheque"
        ),
        LedgerEntry.entry_date < rows[-1].entry_date if rows else None
    ).scalar() or 0

    balance = opening_balance
    data = []

    for r in reversed(rows):
        balance += r.credit - r.debit

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
    # Count all FINAL invoices (pending and paid both count as income)
    query_income = db.query(func.sum(InvoiceModel.total)).filter(
        InvoiceModel.invoice_type == "FINAL"
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

    # Update customer balance - DECREASE (customer owes less)
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

    # Ledger Entry - Cash/Bank increases (Credit)
    # Payment IN = Money we receive = Credit (increases cash/bank balance)
    db.add(
        LedgerEntry(
            id=str(uuid.uuid4()),
            entry_type="payment_in",
            reference_id=data.customer_id,
            customer_id=data.customer_id,
            description=f"Payment received from {customer.name} {data.reference or ''}",
            debit=0,
            credit=data.amount,  # Credit = money coming in (shown in green)
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
                supplier.current_balance = float(supplier.current_balance or 0) - float(data.amount)

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

    # Ledger Entry - Payment OUT = Money we pay = Debit (decreases cash/bank)
    # Debit = Money going out (shown in red)
    db.add(
        LedgerEntry(
        id=str(uuid.uuid4()),
        entry_type="payment_out",
        reference_id=data.supplier_id,
        supplier_id=data.supplier_id,
        description=data.description or f"Payment to {data.supplier_name}",
        debit=data.amount,   # Debit = money going out (shown in red)
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

@api_router.get("/accounts/ledger")
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
    current_user: UserModel = Depends(get_current_user)
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

    query = base_query.order_by(
        LedgerEntry.entry_date.desc(),
        LedgerEntry.created_at.desc()
    )

    entries = query.offset((page - 1) * limit).limit(limit).all()

    # 🔹 Opening balance before this page
    opening_balance = db.query(
        func.coalesce(func.sum(LedgerEntry.credit - LedgerEntry.debit), 0)
    ).filter(
        LedgerEntry.entry_date < entries[-1].entry_date if entries else None
    ).scalar() or 0

    balance = opening_balance
    data = []

    for e in reversed(entries):
        balance += e.credit - e.debit

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

@api_router.get("/accounts/trial-balance")
def trial_balance(
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user)
):

    debit = db.query(func.sum(LedgerEntry.debit)).scalar() or 0
    credit = db.query(func.sum(LedgerEntry.credit)).scalar() or 0

    return {
        "total_debit": debit,
        "total_credit": credit,
        "balanced": debit == credit
    }

app.include_router(api_router)

ALLOWED_ORIGINS = [
    "http://localhost",
    "http://localhost:3000",
    "https://rridegarage.com",
    "https://www.rridegarage.com",
    "capacitor://localhost"
]

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "Accept"
    ],
)
