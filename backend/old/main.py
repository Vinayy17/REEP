import ast
from math import ceil
from fastapi import Query
from itertools import product
from fastapi import FastAPI, APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from sqlalchemy import and_, create_engine, Column, String, Float, Integer, Text, ForeignKey, DateTime, Date, case, or_, func, Index, text
from sqlalchemy.orm import sessionmaker, declarative_base, Session, relationship
from sqlalchemy.orm.attributes import flag_modified
import os
import logging
from sqlalchemy import or_, func
from pydantic import Field
from fastapi.responses import StreamingResponse
import io
from io import StringIO
import csv
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
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
    __allow_unmapped__ = True   # âœ… REQUIRED

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

class InvoicePayment(Base):
    __tablename__ = "invoice_payments"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    invoice_id = Column(String(36), ForeignKey("invoices.id"), index=True)

    amount = Column(Float, nullable=False)

    payment_mode = Column(String(50), nullable=False)

    reference = Column(String(100), nullable=True)

    created_by = Column(String(36), nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.now(IST))

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

    # ðŸ‘‡ PARENT SKU (PARENT CODE)
    sku = Column(String(100), nullable=False, unique=True, index=True)

    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)

    category_id = Column(String(36), ForeignKey("categories.id"), nullable=False)

    # ðŸ’° Pricing (COMMON)
    cost_price = Column(Float, nullable=False, default=0)
    min_selling_price = Column(Float, nullable=False)
    selling_price = Column(Float, nullable=False)

    # ðŸ“¦ TOTAL STOCK (AUTO)
    stock = Column(Integer, nullable=False, default=0)
    min_stock = Column(Integer, nullable=False, default=5)

    # ðŸ§© VARIANTS (INLINE JSON)
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
    created_by_name = Column(String(100))        # âœ… ADD THIS

    stock_before = Column(Integer, nullable=False, default=0)
    stock_after = Column(Integer, nullable=False, default=0)
    variant_stock_after = Column(Integer, nullable=True)

    variant_sku = Column(String(100), nullable=True)  # Stores v_sku if transaction is variant-specific

class ComboModel(Base):
    __tablename__ = "combos"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)

    # JSON of products inside combo
    # Example: [{product_id, sku, quantity}]
    items = Column(JSON, nullable=False)

    price = Column(Float, nullable=False)  # optional combo price

    is_active = Column(Integer, default=1)

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

    original_total = Column(Float, nullable=False, default=0)
    advance_used = Column(Float, nullable=False, default=0)
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
ACCESS_TOKEN_EXPIRE_MINUTES = 60

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
    return encoded_jwt, expire

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
        # Extract numeric part: RR-0007 â†’ 7
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

    # ðŸš€ Upload QR to Cloudinary
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


def resolve_invoice_item_price(item, product, variant=None) -> float:
    frontend_price = getattr(item, "price", None)
    if frontend_price is not None and float(frontend_price) > 0:
        return float(frontend_price)

    if variant and variant.get("v_selling_price") is not None:
        variant_price = float(variant.get("v_selling_price") or 0)
        if variant_price > 0:
            return variant_price

    return float(product.selling_price or 0)

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
    # ðŸ” Only admin
    if current_user.role not in ["admin", "store_handler"]:

        raise HTTPException(status_code=403, detail="Admin only")

    url = upload_image_to_cloudinary(file.file)
    return {"url": url}

# Duplicate function removed to avoid conflict
# def generate_product_code():
#     date_part = datetime.now(IST).strftime("%Y%m%d")
#     rand = "".join(random.choices(string.ascii_uppercase + string.digits, k=4))
#     return f"PRD-{date_part}-{rand}"

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
    expires_at: str
    expires_in_seconds: int
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
    created_at: datetime   # âœ… TYPE, NOT Column

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
    images: List[str] = Field(default_factory=list)
 # Default to empty list
    is_service: int = 0
    variants: List[VariantSchema] = Field(default_factory=list)
    qr_code_url: Optional[str] = None # Matches the field being sent/required

class Customer(BaseModel):
    id: str
    name: str
    email: str
    phone: Optional[str] = None
    address: Optional[str] = None
    current_balance: float = 0
    created_at: str

class CustomerCreate(BaseModel):
    name: str
    email: str
    phone: Optional[str] = None
    address: Optional[str] = None

class CustomerUpdate(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None

class InvoiceItem(BaseModel):
    product_id: Optional[str] = None
    combo_id: Optional[str] = None   # 👈 ADD THIS

    product_name: str
    quantity: int
    price: float # This will be the price per unit for the item
    gst_rate: float
    sku: Optional[str] = None # Variant SKU or product SKU
    is_service: int = 0        # âœ… ADD THIS

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
    paid_amount: Optional[float] = 0
    use_advance: Optional[bool] = True

    # additional_amount: float = 0  # âœ… ADD for manual charges
    # additional_label: Optional[str] = None  # âœ… ADD for charge label
    additional_charges: List[AdditionalCharge] = Field(default_factory=list)


class ComboItem(BaseModel):
    product_id: str
    sku: Optional[str] = None
    quantity: int

class ComboCreate(BaseModel):
    name: str
    items: List[ComboItem]
    price: float
    
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
          # âœ… ADD

# ---------------- STATUS UPDATE SCHEMA ----------------
class InvoiceStatusUpdate(BaseModel):
    payment_status: str


class InventoryTransactionResponse(BaseModel):
    id: str
    product_id: str
    product_name: str
    product_code: str
    type: str
    quantity: int
    variant_stock_after: Optional[int]  # âœ… ADD
    created_by: str        # ðŸ‘ˆ add this

    variant_sku: Optional[str]
    stock_after: int          # âœ… use this, not remaining_stock
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


class AdvancePaymentRequest(BaseModel):
    customer_id: str
    amount: float
    payment_mode: str
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

    # ================= 1ï¸âƒ£ VARIANT FIRST =================
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

    # ================= 2ï¸âƒ£ PRODUCT CODE / SKU =================
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

    access_token, expires_at = create_access_token(
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
        expires_at=expires_at.isoformat(),
        expires_in_seconds=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
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
    # âœ… ADD THIS
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

    access_token, expires_at = create_access_token(
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
        expires_at=expires_at.isoformat(),
        expires_in_seconds=ACCESS_TOKEN_EXPIRE_MINUTES * 60,
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

    # ðŸ”’ LAZY LOAD RULE
    if q:
        q = q.strip().lower()
        query = query.filter(
            func.lower(CategoryModel.name).like(f"{q}%")  # ðŸ”¥ prefix search (FAST)
        )
    else:
        # ðŸ‘‡ first open â†’ show top categories only
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
    categories = db.query(CategoryModel).order_by(CategoryModel.name.asc()).all()
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
from sqlalchemy.exc import IntegrityError

def generate_draft_number(db: Session):
    year = datetime.now(IST).year

    counter = db.execute(
        text("SELECT year, last_number FROM draft_counters WHERE year = :year FOR UPDATE"),
        {"year": year}
    ).fetchone()

    last_draft = (
        db.query(InvoiceModel.draft_number)
        .filter(
            InvoiceModel.invoice_type == "DRAFT",
            InvoiceModel.draft_number.like(f"DRF-{year}-%")
        )
        .order_by(InvoiceModel.draft_number.desc())
        .with_for_update()
        .first()
    )

    counter_value = int(counter.last_number) if counter and counter.last_number is not None else 0
    invoice_value = 0

    if last_draft and last_draft[0]:
        try:
            invoice_value = int(str(last_draft[0]).split("-")[-1])
        except (TypeError, ValueError):
            invoice_value = 0

    next_num = max(counter_value, invoice_value) + 1

    if not counter:
        db.execute(
            text("INSERT INTO draft_counters (year, last_number) VALUES (:year, :num)"),
            {"year": year, "num": next_num}
        )
    else:
        db.execute(
            text("UPDATE draft_counters SET last_number = :num WHERE year = :year"),
            {"num": next_num, "year": year}
        )

    return f"DRF-{year}-{next_num:04d}"

def normalize_draft_payment_status(status: Optional[str]) -> str:
    allowed_status = {"pending", "paid", "partial", "cancelled"}
    normalized = (status or "pending").strip().lower()
    return normalized if normalized in allowed_status else "pending"

def normalize_draft_payment_mode(mode: Optional[str]) -> str:
    allowed_modes = {"cash", "upi", "bank", "cheque"}
    normalized = (mode or "cash").strip().lower()
    return normalized if normalized in allowed_modes else "cash"


def should_use_advance(value: Optional[bool]) -> bool:
    return value is not False


def calculate_invoice_advance(
    customer: Optional[CustomerModel],
    original_total: float,
    use_advance: bool = True,
    db: Optional[Session] = None
):
    if not customer or not use_advance:
        return 0.0, round(original_total, 2)

    current_balance = (
        calculate_customer_current_balance(db, customer)
        if db else
        float(customer.current_balance or 0)
    )
    if current_balance >= 0:
        return 0.0, round(original_total, 2)

    advance_available = abs(current_balance)
    advance_used = min(advance_available, float(original_total or 0))
    final_total = round(float(original_total or 0) - advance_used, 2)
    return round(advance_used, 2), final_total


def apply_customer_invoice_balance(customer: Optional[CustomerModel], original_total: float, paid_amount: float):
    return


def calculate_customer_current_balance(db: Session, customer: Optional[CustomerModel]) -> float:
    if not customer:
        return 0.0

    pending_total = db.query(
        func.coalesce(
            func.sum(
                case(
                    (InvoiceModel.payment_status.in_(["pending", "partial"]), func.coalesce(InvoiceModel.balance_amount, 0)),
                    else_=0
                )
            ),
            0
        )
    ).filter(
        InvoiceModel.customer_id == customer.id,
        InvoiceModel.invoice_type == "FINAL",
        InvoiceModel.payment_status != "cancelled",
    ).scalar() or 0

    advance_in_total = db.query(
        func.coalesce(func.sum(LedgerEntry.credit), 0)
    ).filter(
        LedgerEntry.customer_id == customer.id,
        LedgerEntry.entry_type == "advance_in"
    ).scalar() or 0

    advance_used_total = db.query(
        func.coalesce(func.sum(LedgerEntry.debit), 0)
    ).filter(
        LedgerEntry.customer_id == customer.id,
        LedgerEntry.entry_type == "advance_used"
    ).scalar() or 0

    opening_balance = float(customer.opening_balance or 0)
    return round(
        opening_balance + float(pending_total or 0) - (float(advance_in_total or 0) - float(advance_used_total or 0)),
        2
    )


def sync_customer_current_balance(db: Session, customer: Optional[CustomerModel]) -> float:
    if not customer:
        return 0.0

    customer.current_balance = calculate_customer_current_balance(db, customer)
    return float(customer.current_balance or 0)


def add_advance_sale_ledger_entry(
    db: Session,
    invoice_id: str,
    invoice_number: str,
    customer_id: Optional[str],
    advance_used: float,
    current_user: UserModel,
):
    advance_amount = round(float(advance_used or 0), 2)

    if advance_amount > 0:
        db.add(LedgerEntry(
            id=str(uuid.uuid4()),
            entry_type="advance_used",
            reference_id=invoice_id,
            customer_id=customer_id,
            description=f"Advance adjusted in invoice {invoice_number}",
            debit=advance_amount,
            credit=0,
            payment_mode="adjustment",
            entry_date=datetime.now(IST).date(),
            created_by=current_user.id,
            created_by_name=current_user.name,
            created_at=datetime.now(IST),
        ))


def add_sale_ledger_entry(
    db: Session,
    invoice_id: str,
    invoice_number: str,
    customer_id: Optional[str],
    original_total: float,
    current_user: UserModel,
):
    sale_amount = round(float(original_total or 0), 2)
    if sale_amount <= 0:
        return

    db.add(LedgerEntry(
        id=str(uuid.uuid4()),
        entry_type="sale",
        reference_id=invoice_id,
        customer_id=customer_id,
        description=f"Sale Invoice {invoice_number}",
        debit=0,
        credit=sale_amount,
        payment_mode="invoice",
        entry_date=datetime.now(IST).date(),
        created_by=current_user.id,
        created_by_name=current_user.name,
        created_at=datetime.now(IST),
    ))

def calculate_draft_payment_summary(
    total: float,
    payment_status: str,
    paid_amount: Optional[float] = 0,
    advance_used: float = 0
):
    if payment_status == "paid":
        return round(total, 2), 0.0

    if payment_status == "partial":
        partial_paid = round(float(paid_amount or 0), 2)

        if partial_paid <= 0:
            return 0.0, round(total, 2)

        if partial_paid > total:
            raise HTTPException(400, "Partial exceeds total")

        return partial_paid, round(total - partial_paid, 2)

    return 0.0, round(total, 2)


def resolve_invoice_payment_status(
    requested_status: Optional[str],
    advance_used: float,
    balance_amount: float,
    paid_amount: float = 0,
):
    status = normalize_draft_payment_status(requested_status)
    if balance_amount <= 0:
        return "paid"
    if status == "paid":
        return "paid"
    if status == "partial":
        return "partial"
    if float(advance_used or 0) > 0 or float(paid_amount or 0) > 0:
        return "partial"
    return "pending"


def build_invoice_merge_key(item: dict) -> str:
    product_id = item.get("product_id") or "manual"
    combo_id = item.get("combo_id") or "single"
    sku = item.get("sku") or "no-sku"
    product_name = (item.get("product_name") or "").strip().lower()
    is_service = int(item.get("is_service") or 0)

    if not item.get("product_id"):
        return f"manual::{product_name}::{sku}::{is_service}::{combo_id}"

    return f"{product_id}::{combo_id}::{sku}::{is_service}"

def prepare_invoice_draft_data(invoice_data: InvoiceCreate, db: Session):
    payment_status = normalize_draft_payment_status(invoice_data.payment_status)
    payment_mode = normalize_draft_payment_mode(invoice_data.payment_mode)

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

    raw_items = []
    subtotal = 0.0

    for item in invoice_data.items:
        if getattr(item, "combo_id", None):
            if getattr(item, "product_id", None):
                product = db.query(ProductModel).filter(
                    ProductModel.id == item.product_id
                ).first()

                if not product or product.is_active == 0:
                    raise HTTPException(404, "Product not found for combo item")

                quantity = int(item.quantity)
                variant = None

                if item.variant_info and item.variant_info.get("v_sku"):
                    variant = next(
                        (v for v in (product.variants or [])
                         if v.get("v_sku") == item.variant_info.get("v_sku")),
                        None
                    )

                price = resolve_invoice_item_price(item, product, variant)
                line_total = round(price * quantity, 2)
                subtotal += line_total

                raw_items.append({
                    "product_id": product.id,
                    "combo_id": item.combo_id,
                    "sku": (
                        variant.get("v_sku")
                        if variant else
                        (item.sku or product.sku)
                    ),
                    "product_name": item.product_name or product.name,
                    "quantity": quantity,
                    "price": price,
                    "gst_rate": 0,
                    "total": line_total,
                    "is_service": product.is_service,
                    "variant_info": variant or item.variant_info,
                    "image_url": (
                        variant.get("image_url")
                        if variant and variant.get("image_url")
                        else (item.image_url or (product.images or [None])[0])
                    ),
                })

                continue

            combo = db.query(ComboModel).filter(
                ComboModel.id == item.combo_id,
                ComboModel.is_active == 1
            ).first()

            if not combo:
                raise HTTPException(404, "Combo not found")

            for c in combo.items:
                product = db.query(ProductModel).filter(
                    ProductModel.id == c["product_id"]
                ).first()

                if not product or product.is_active == 0:
                    continue

                quantity = int(c["quantity"]) * int(item.quantity)
                variant = None

                if item.variant_info and item.variant_info.get("v_sku"):
                    variant = next(
                        (v for v in (product.variants or [])
                         if v.get("v_sku") == item.variant_info.get("v_sku")),
                        None
                    )
                elif c.get("sku") and product.variants:
                    variant = next(
                        (v for v in (product.variants or [])
                         if v.get("v_sku") == c.get("sku")),
                        None
                    )

                if variant and variant.get("v_selling_price") is not None:
                    price = float(variant.get("v_selling_price"))
                else:
                    price = float(product.selling_price or 0)

                line_total = round(price * quantity, 2)
                subtotal += line_total

                raw_items.append({
                    "product_id": product.id,
                    "combo_id": item.combo_id,
                    "sku": variant.get("v_sku") if variant else c.get("sku"),
                    "product_name": product.name,
                    "quantity": quantity,
                    "price": price,
                    "gst_rate": 0,
                    "total": line_total,
                    "is_service": product.is_service,
                    "variant_info": variant,
                    "image_url": (
                        variant.get("image_url")
                        if variant and variant.get("image_url")
                        else (product.images or [None])[0]
                    ),
                })

            continue

        quantity = int(item.quantity)
        price = float(item.price)
        line_total = round(price * quantity, 2)
        subtotal += line_total

        raw_items.append({
            "product_id": item.product_id,
            "sku": item.sku,
            "product_name": item.product_name,
            "quantity": quantity,
            "price": price,
            "gst_rate": float(item.gst_rate or 0),
            "total": line_total,
            "is_service": item.is_service or 0,
            "variant_info": item.variant_info,
            "image_url": item.image_url,
        })

    if not raw_items:
        raise HTTPException(400, "No valid items")

    merged = {}
    for item in raw_items:
        key = f"{item['product_id']}_{item.get('sku')}"
        if key not in merged:
            merged[key] = item
        else:
            merged[key]["quantity"] += item["quantity"]
            merged[key]["total"] = round(
                merged[key]["quantity"] * merged[key]["price"], 2
            )

    items = list(merged.values())
    additional_charges = [
        charge.dict() for charge in (invoice_data.additional_charges or [])
    ]
    additional_total = round(
        sum(float(charge.get("amount") or 0) for charge in additional_charges),
        2
    )
    gst_enabled = bool(invoice_data.gst_enabled)
    gst_rate = float(invoice_data.gst_rate or 0)
    discount = float(invoice_data.discount or 0)
    taxable = subtotal + additional_total
    gst_amount = round((taxable * gst_rate) / 100, 2) if gst_enabled else 0.0
    original_total = round(taxable + gst_amount - discount, 2)
    advance_used, total = calculate_invoice_advance(
        customer,
        original_total,
        should_use_advance(invoice_data.use_advance),
        db
    )
    payment_status = resolve_invoice_payment_status(
        payment_status,
        advance_used,
        total,
        invoice_data.paid_amount or 0
    )
    paid_amount, balance_amount = calculate_draft_payment_summary(
        total,
        payment_status,
        invoice_data.paid_amount,
        advance_used
    )

    return {
        "customer": customer,
        "customer_name": invoice_data.customer_name or customer.name,
        "customer_phone": invoice_data.customer_phone or customer.phone,
        "customer_address": invoice_data.customer_address or customer.address,
        "items": items,
        "subtotal": round(subtotal, 2),
        "additional_charges": additional_charges,
        "gst_enabled": 1 if gst_enabled else 0,
        "gst_rate": gst_rate,
        "gst_amount": gst_amount,
        "discount": discount,
        "original_total": original_total,
        "advance_used": advance_used,
        "total": total,
        "payment_status": payment_status,
        "payment_mode": payment_mode,
        "paid_amount": paid_amount,
        "balance_amount": balance_amount,
    }

@api_router.post("/invoices/draft")
def create_invoice_draft(
    invoice_data: InvoiceCreate,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    draft_data = prepare_invoice_draft_data(invoice_data, db)

    draft = None
    for _ in range(3):
        draft = InvoiceModel(
            id=str(uuid.uuid4()),
            invoice_type="DRAFT",
            draft_number=generate_draft_number(db),

            customer_id=draft_data["customer"].id,
            customer_name=draft_data["customer_name"],
            customer_phone=draft_data["customer_phone"],
            customer_address=draft_data["customer_address"],

            items=json.dumps(draft_data["items"]),
            subtotal=draft_data["subtotal"],
            additional_charges=draft_data["additional_charges"],

            gst_enabled=draft_data["gst_enabled"],
            gst_rate=draft_data["gst_rate"],
            gst_amount=draft_data["gst_amount"],

            discount=draft_data["discount"],
            original_total=draft_data["original_total"],
            advance_used=draft_data["advance_used"],
            total=draft_data["total"],

            paid_amount=draft_data["paid_amount"],
            balance_amount=draft_data["balance_amount"],
            payment_status=draft_data["payment_status"],
            payment_mode=draft_data["payment_mode"],

            created_by=current_user.id,
            created_by_name=current_user.name,
            created_at=datetime.now(IST),
        )

        try:
            db.add(draft)
            db.commit()
            db.refresh(draft)
            break
        except IntegrityError as exc:
            db.rollback()
            if "draft_number" not in str(exc).lower():
                raise
    else:
        raise HTTPException(500, "Failed to generate unique draft number")

    return {
        "id": draft.id,
        "draft_number": draft.draft_number,
        "total": draft.total
    }

    allowed_status = ["pending", "paid", "partial", "cancelled"]
    status = invoice_data.payment_status or "pending"
    if status not in allowed_status:
        status = "pending"

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
            email=invoice_data.customer_email or f"{invoice_data.customer_phone}@draft.com",
            phone=invoice_data.customer_phone,
            address=invoice_data.customer_address,
            created_at=datetime.now(IST),
        )
        db.add(customer)
        db.flush()

    raw_items = []
    subtotal = 0.0

    for item in invoice_data.items:

        # ================= COMBO =================
        if getattr(item, "combo_id", None):

            if getattr(item, "product_id", None):
                product = db.query(ProductModel).filter(
                    ProductModel.id == item.product_id
                ).first()

                if not product or product.is_active == 0:
                    raise HTTPException(404, "Product not found for combo item")

                quantity = int(item.quantity)
                variant = None

                if item.variant_info and item.variant_info.get("v_sku"):
                    variant = next(
                        (v for v in (product.variants or [])
                         if v.get("v_sku") == item.variant_info.get("v_sku")),
                        None
                    )

                price = resolve_invoice_item_price(item, product, variant)
                line_total = round(price * quantity, 2)
                subtotal += line_total

                raw_items.append({
                    "product_id": product.id,
                    "combo_id": item.combo_id,
                    "sku": (
                        variant.get("v_sku")
                        if variant else
                        (item.sku or product.sku)
                    ),
                    "product_name": item.product_name or product.name,
                    "quantity": quantity,
                    "price": price,
                    "gst_rate": 0,
                    "total": line_total,
                    "is_service": product.is_service,
                    "variant_info": variant or item.variant_info,
                    "image_url": (
                        variant.get("image_url")
                        if variant and variant.get("image_url")
                        else (item.image_url or (product.images or [None])[0])
                    ),
                })

                continue

            combo = db.query(ComboModel).filter(
                ComboModel.id == item.combo_id,
                ComboModel.is_active == 1
            ).first()

            if not combo:
                raise HTTPException(404, "Combo not found")

            for c in combo.items:
                product = db.query(ProductModel).filter(
                    ProductModel.id == c["product_id"]
                ).first()

                if not product or product.is_active == 0:
                    continue

                quantity = int(c["quantity"]) * int(item.quantity)

                # ✅ FIXED VARIANT LOGIC
                variant = None

                # 🔥 PRIORITY: frontend selected variant
                if item.variant_info and item.variant_info.get("v_sku"):
                    variant = next(
                        (v for v in (product.variants or [])
                         if v.get("v_sku") == item.variant_info.get("v_sku")),
                        None
                    )

                # fallback combo sku
                elif c.get("sku") and product.variants:
                    variant = next(
                        (v for v in (product.variants or [])
                         if v.get("v_sku") == c.get("sku")),
                        None
                    )

                # ✅ FIXED PRICE
                if variant and variant.get("v_selling_price") is not None:
                    price = float(variant.get("v_selling_price"))
                else:
                    price = float(product.selling_price or 0)

                line_total = round(price * quantity, 2)
                subtotal += line_total

                raw_items.append({
                    "product_id": product.id,
                    "combo_id": item.combo_id,  # ✅ IMPORTANT
                    "sku": variant.get("v_sku") if variant else c.get("sku"),
                    "product_name": product.name,
                    "quantity": quantity,
                    "price": price,
                    "gst_rate": 0,
                    "total": line_total,
                    "is_service": product.is_service,

                    "variant_info": variant,

                    "image_url": (
                        variant.get("image_url")
                        if variant and variant.get("image_url")
                        else (product.images or [None])[0]
                    ),
                })

            continue

        # ================= NORMAL =================
        quantity = int(item.quantity)
        price = float(item.price)

        line_total = round(price * quantity, 2)
        subtotal += line_total

        raw_items.append({
            "product_id": item.product_id,
            "sku": item.sku,
            "product_name": item.product_name,
            "quantity": quantity,
            "price": price,
            "gst_rate": float(item.gst_rate or 0),
            "total": line_total,
            "is_service": item.is_service or 0,
            "variant_info": item.variant_info,
            "image_url": item.image_url,
        })

    if not raw_items:
        raise HTTPException(400, "No valid items")

    # ================= MERGE =================
    merged = {}
    for item in raw_items:
        key = build_invoice_merge_key(item)
        if key not in merged:
            merged[key] = item
        else:
            merged[key]["quantity"] += item["quantity"]
            merged[key]["total"] = round(
                merged[key]["quantity"] * merged[key]["price"], 2
            )

    items = list(merged.values())

    additional_charges = [
        charge.dict() for charge in (invoice_data.additional_charges or [])
    ]

    additional_total = sum(c["amount"] for c in additional_charges)
    taxable = subtotal + additional_total

    gst_amount = (
        round((taxable * float(invoice_data.gst_rate or 0)) / 100, 2)
        if invoice_data.gst_enabled else 0
    )

    total = round(
        taxable + gst_amount - float(invoice_data.discount or 0),
        2
    )

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
        additional_charges=additional_charges,

        gst_enabled=1 if invoice_data.gst_enabled else 0,
        gst_rate=float(invoice_data.gst_rate or 0),
        gst_amount=gst_amount,

        discount=float(invoice_data.discount or 0),
        total=total,

        payment_status=status,

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
        "total": draft.total
    }
@api_router.put("/invoices/draft/{draft_id}")
def update_invoice_draft(
    draft_id: str,
    invoice_data: InvoiceCreate,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    draft = db.query(InvoiceModel).filter(
        InvoiceModel.id == draft_id,
        InvoiceModel.invoice_type == "DRAFT"
    ).first()

    if not draft:
        raise HTTPException(404, "Draft not found")

    draft_data = prepare_invoice_draft_data(invoice_data, db)

    draft.customer_id = draft_data["customer"].id
    draft.customer_name = draft_data["customer_name"]
    draft.customer_phone = draft_data["customer_phone"]
    draft.customer_address = draft_data["customer_address"]
    draft.items = json.dumps(draft_data["items"])
    draft.subtotal = draft_data["subtotal"]
    draft.additional_charges = draft_data["additional_charges"]
    draft.gst_enabled = draft_data["gst_enabled"]
    draft.gst_rate = draft_data["gst_rate"]
    draft.gst_amount = draft_data["gst_amount"]
    draft.discount = draft_data["discount"]
    draft.original_total = draft_data["original_total"]
    draft.advance_used = draft_data["advance_used"]
    draft.total = draft_data["total"]
    draft.payment_status = draft_data["payment_status"]
    draft.payment_mode = draft_data["payment_mode"]
    draft.paid_amount = draft_data["paid_amount"]
    draft.balance_amount = draft_data["balance_amount"]

    db.commit()

    return {
        "message": "Draft updated successfully",
        "id": draft.id,
        "draft_number": draft.draft_number,
        "subtotal": draft.subtotal,
        "gst_amount": draft.gst_amount,
        "total": draft.total,
        "payment_status": draft.payment_status,
        "payment_mode": draft.payment_mode
    }

    raw_items = []
    subtotal = 0.0

    for item in invoice_data.items:

        if getattr(item, "combo_id", None):

            if getattr(item, "product_id", None):
                product = db.query(ProductModel).filter(
                    ProductModel.id == item.product_id
                ).first()

                if not product or product.is_active == 0:
                    raise HTTPException(404, "Product not found for combo item")

                quantity = int(item.quantity)
                variant = None

                if item.variant_info and item.variant_info.get("v_sku"):
                    variant = next(
                        (v for v in (product.variants or [])
                         if v.get("v_sku") == item.variant_info.get("v_sku")),
                        None
                    )

                price = resolve_invoice_item_price(item, product, variant)
                line_total = round(price * quantity, 2)
                subtotal += line_total

                raw_items.append({
                    "product_id": product.id,
                    "combo_id": item.combo_id,
                    "sku": (
                        variant.get("v_sku")
                        if variant else
                        (item.sku or product.sku)
                    ),
                    "product_name": item.product_name or product.name,
                    "quantity": quantity,
                    "price": price,
                    "gst_rate": 0,
                    "total": line_total,
                    "is_service": product.is_service,
                    "variant_info": variant or item.variant_info,
                    "image_url": (
                        variant.get("image_url")
                        if variant and variant.get("image_url")
                        else (item.image_url or (product.images or [None])[0])
                    ),
                })

                continue

            combo = db.query(ComboModel).filter(
                ComboModel.id == item.combo_id,
                ComboModel.is_active == 1
            ).first()

            for c in combo.items:
                product = db.query(ProductModel).filter(
                    ProductModel.id == c["product_id"]
                ).first()

                quantity = int(c["quantity"]) * int(item.quantity)

                # ✅ SAME FIX
                variant = None

                if item.variant_info and item.variant_info.get("v_sku"):
                    variant = next(
                        (v for v in (product.variants or [])
                         if v.get("v_sku") == item.variant_info.get("v_sku")),
                        None
                    )

                elif c.get("sku") and product.variants:
                    variant = next(
                        (v for v in (product.variants or [])
                         if v.get("v_sku") == c.get("sku")),
                        None
                    )

                if variant and variant.get("v_selling_price") is not None:
                    price = float(variant.get("v_selling_price"))
                else:
                    price = float(product.selling_price or 0)

                line_total = round(price * quantity, 2)
                subtotal += line_total

                raw_items.append({
                    "product_id": product.id,
                    "combo_id": item.combo_id,
                    "sku": variant.get("v_sku") if variant else c.get("sku"),
                    "product_name": product.name,
                    "quantity": quantity,
                    "price": price,
                    "gst_rate": 0,
                    "total": line_total,
                    "is_service": product.is_service,
                    "variant_info": variant,
                    "image_url": (product.images or [None])[0],
                })

            continue

        quantity = int(item.quantity)
        price = float(item.price)

        line_total = round(price * quantity, 2)
        subtotal += line_total

        raw_items.append({
            "product_id": item.product_id,
            "sku": item.sku,
            "product_name": item.product_name,
            "quantity": quantity,
            "price": price,
            "gst_rate": float(item.gst_rate or 0),
            "total": line_total,
            "is_service": item.is_service or 0,
            "variant_info": item.variant_info,
            "image_url": item.image_url,
        })

    draft.items = json.dumps(raw_items)
    draft.subtotal = subtotal

    db.commit()

    return {
        "message": "Draft updated successfully",
        "total": subtotal
    }

@api_router.get("/invoices/drafts")
def list_drafts(
    page: int = 1,
    limit: int = 10,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    q = db.query(InvoiceModel).filter(
        InvoiceModel.invoice_type == "DRAFT"
    )

    total = q.count()

    drafts = (
        q.order_by(InvoiceModel.created_at.desc())
        .offset((page - 1) * limit)
        .limit(limit)
        .all()
    )

    return {
        "data": [
            {
                "id": d.id,
                "draft_number": d.draft_number,
                "customer_name": d.customer_name,
                "total": d.total,
                "created_at": d.created_at.isoformat()

            }
            for d in drafts
        ],
        "total": total
    }

@api_router.get("/invoices/draft/{draft_id}")
def get_single_draft(
    draft_id: str,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    draft = (
        db.query(InvoiceModel)
        .filter(
            InvoiceModel.id == draft_id,
            InvoiceModel.invoice_type == "DRAFT",
        )
        .first()
    )

    if not draft:
        raise HTTPException(404, "Draft not found")

    return {
    "id": draft.id,
    "draft_number": draft.draft_number,

    "customer_id": draft.customer_id,
    "customer_name": draft.customer_name,
    "customer_phone": draft.customer_phone,
    "customer_address": draft.customer_address,

    "items": json.loads(draft.items) if draft.items else [],

    "subtotal": draft.subtotal,
    "additional_charges": draft.additional_charges or [],
    "discount": draft.discount,

    "gst_enabled": bool(draft.gst_enabled),
    "gst_rate": draft.gst_rate,
        "gst_amount": draft.gst_amount,

        "original_total": float(draft.original_total or draft.total or 0),
        "advance_used": float(draft.advance_used or 0),
        "total": draft.total,
        "payment_status": draft.payment_status,
        "payment_mode": draft.payment_mode,
    "paid_amount": draft.paid_amount,
    "balance_amount": draft.balance_amount,
}


@api_router.delete("/invoices/draft/{draft_id}")
def delete_draft(
    draft_id: str,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    draft = db.query(InvoiceModel).filter(
        InvoiceModel.id == draft_id,
        InvoiceModel.invoice_type == "DRAFT"
    ).first()

    if not draft:
        raise HTTPException(404, "Draft not found")

    db.delete(draft)
    db.commit()

    return {"message": "Draft deleted"}

class DraftFinalizeRequest(BaseModel):
    payment_status: Optional[str] = None
    paid_amount: Optional[float] = 0
    use_advance: Optional[bool] = True

@api_router.post("/invoices/draft/{draft_id}/finalize")
def finalize_draft(
    draft_id: str,
    data: Optional[DraftFinalizeRequest] = None,
    payment_status: Optional[str] = Query(None),
    payment_mode: Optional[str] = Query(None),
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):

    # ================= PAYMENT STATUS =================
    status_from_body = (
        data.payment_status.strip().lower()
        if data and data.payment_status
        else None
    )

    status_from_query = payment_status.strip().lower() if payment_status else None
    final_payment_status = status_from_body or status_from_query

    # ================= GET DRAFT (FIXED 🔥) =================
    draft = db.query(InvoiceModel).filter(
        InvoiceModel.id == draft_id
    ).with_for_update().first()

    if not draft:
        raise HTTPException(404, "Draft not found")

    if not status_from_body and not status_from_query:
        final_payment_status = normalize_draft_payment_status(draft.payment_status)

    if final_payment_status not in ["pending", "paid", "cancelled", "partial"]:
        raise HTTPException(400, "Invalid payment status")

    final_payment_mode = (
        payment_mode.strip().lower()
        if payment_mode
        else normalize_draft_payment_mode(draft.payment_mode)
    )

    # ✅ HANDLE DOUBLE CALL / ALREADY FINALIZED
    if draft.invoice_type != "DRAFT":
        return {
            "message": "Already finalized",
            "invoice_id": draft.id,
            "invoice_number": draft.invoice_number,
            "total": draft.total,
            "payment_status": draft.payment_status
        }

    # ================= ITEMS =================
    items = parse_invoice_items(draft.items)

    if not items:
        raise HTTPException(400, "Draft has no items")

    subtotal = 0.0

    # ================= INVENTORY =================
    for item in items:

        quantity = int(item.get("quantity", 0))
        price = float(item.get("price", 0))
        subtotal += price * quantity

        if item.get("is_service") == 1 or not item.get("product_id"):
            continue

        product = db.query(ProductModel).filter(
            ProductModel.id == item["product_id"]
        ).with_for_update().first()

        if not product:
            raise HTTPException(404, "Product not found")

        variants = list(product.variants or [])
        sku = item.get("sku")
        variant_stock_after = None

        # 🔥 VARIANT
        if variants:

            if not sku:
                raise HTTPException(400, "Variant SKU required")

            for v in variants:
                if v.get("v_sku") == sku:

                    stock_before = int(v.get("stock", 0))

                    if stock_before < quantity:
                        raise HTTPException(400, f"Insufficient stock for {sku}")

                    v["stock"] = stock_before - quantity
                    variant_stock_after = v["stock"]
                    break
            else:
                raise HTTPException(400, "Variant not found")

            product.variants = variants
            product.stock = calculate_total_stock(variants)
            stock_after = product.stock

            flag_modified(product, "variants")
            flag_modified(product, "stock")

        # 🔥 NORMAL PRODUCT
        else:
            stock_before = int(product.stock or 0)

            if stock_before < quantity:
                raise HTTPException(400, f"Insufficient stock for {product.name}")

            product.stock = stock_before - quantity
            stock_after = product.stock

            flag_modified(product, "stock")

        db.flush()

        db.add(InventoryTransaction(
            id=str(uuid.uuid4()),
            product_id=product.id,
            type="OUT",
            quantity=quantity,
            source="INVOICE",
            stock_before=stock_before,
            stock_after=stock_after,
            variant_sku=sku,
            variant_stock_after=variant_stock_after,
            created_by=current_user.id,
            created_by_name=current_user.name,
            created_at=datetime.now(IST)
        ))

    # ================= TOTAL =================
    additional_total = calculate_additional_total(draft.additional_charges or [])
    taxable = subtotal + additional_total

    draft.subtotal = subtotal

    draft.gst_amount = (
        round((taxable * draft.gst_rate) / 100, 2)
        if draft.gst_enabled else 0
    )

    original_total = round(taxable + draft.gst_amount - draft.discount, 2)

    # ================= PAYMENT =================
    customer = db.query(CustomerModel).filter(
        CustomerModel.id == draft.customer_id
    ).first()

    advance_used, final_total = calculate_invoice_advance(
        customer,
        original_total,
        should_use_advance(data.use_advance if data else True),
        db
    )
    effective_advance_used = round(max(float(advance_used or 0), float(draft.advance_used or 0)), 2)

    draft.original_total = original_total
    draft.advance_used = effective_advance_used
    draft.total = final_total

    paid_amount = 0
    balance_amount = draft.total

    if final_payment_status == "paid":
        paid_amount = draft.total
        balance_amount = 0

    if final_payment_status == "partial":
        partial_amount = (
            data.paid_amount
            if data and data.paid_amount is not None
            else draft.paid_amount
        )

        if partial_amount <= 0:
            partial_amount = 0

        if partial_amount > draft.total:
            raise HTTPException(400, "Partial exceeds total")

        paid_amount = round(float(partial_amount or 0), 2)
        balance_amount = round(draft.total - paid_amount, 2)

    final_payment_status = resolve_invoice_payment_status(
        final_payment_status,
        effective_advance_used,
        balance_amount,
        paid_amount,
    )

    draft.paid_amount = paid_amount
    draft.balance_amount = balance_amount
    draft.payment_mode = final_payment_mode

    # ================= FINALIZE =================
    invoice_number = generate_invoice_number(db)

    draft.invoice_type = "FINAL"
    draft.invoice_number = invoice_number
    draft.draft_number = None
    draft.payment_status = final_payment_status
    draft.created_at = datetime.now(IST)

    add_sale_ledger_entry(
        db,
        draft.id,
        invoice_number,
        draft.customer_id,
        original_total,
        current_user,
    )

    if customer:
        add_advance_sale_ledger_entry(
            db,
            draft.id,
            invoice_number,
            draft.customer_id,
            effective_advance_used,
            current_user,
        )

        if paid_amount > 0:
            db.add(InvoicePayment(
                invoice_id=draft.id,
                amount=paid_amount,
                payment_mode=final_payment_mode,
                created_by=current_user.id
            ))

        sync_customer_current_balance(db, customer)

    db.commit()

    return {
        "invoice_id": draft.id,
        "invoice_number": invoice_number,
        "total": draft.total,
        "original_total": draft.original_total,
        "advance_used": draft.advance_used,
        "paid_amount": paid_amount,
        "balance_amount": balance_amount,
        "payment_status": final_payment_status,
        "message": "Draft finalized successfully"
    }

@api_router.post("/upload/variant-image")
def upload_variant_image(
    file: UploadFile = File(...),
    current_user: UserModel = Depends(get_current_user)
):
    if current_user.role not in ["admin", "store_handler"]:
            raise HTTPException(status_code=403, detail="Not allowed")


    url = upload_image_to_cloudinary(file.file, folder="variant_images")
    return {"url": url}
@api_router.post("/upload/requirement-image")
def upload_requirement_image(
    file: UploadFile = File(...),
    current_user: UserModel = Depends(get_current_user)
):
    # ðŸ” Allow admin & store handler
    if current_user.role not in ["admin", "store_handler"]:
        raise HTTPException(status_code=403, detail="Not allowed")

    url = upload_image_to_cloudinary(
        file.file,
        folder="requirements"
    )

    return {"url": url}

@api_router.delete("/requirements/cleanup")
def cleanup_old_requirements(
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin only")

    cutoff_date = datetime.now(IST) - timedelta(days=90)

    old_reqs = db.query(RequirementModel).filter(
        RequirementModel.status.in_(["completed", "rejected"]),
        RequirementModel.created_at < cutoff_date
    ).all()

    deleted_count = 0

    for req in old_reqs:
        # ================= DELETE ITEM IMAGES =================
        try:
            items = json.loads(req.requirement_items or "[]")
        except Exception:
            items = []

        for item in items:
            image_url = item.get("image_url")
            if image_url:
                try:
                    # Extract public_id safely
                    public_id = image_url.split("/")[-1].split(".")[0]
                    cloudinary.uploader.destroy(
                        f"requirements/{public_id}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Cloudinary delete failed for {image_url}: {e}"
                    )

        # ================= DELETE DB ROW =================
        db.delete(req)
        deleted_count += 1

    db.commit()

    return {
        "message": "Old requirements cleaned successfully",
        "deleted": deleted_count
    }

@api_router.post("/requirements", status_code=201)
def create_requirement(
    data: RequirementCreate,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    requirement = RequirementModel(
    customer_name=data.customer_name,
    customer_phone=data.customer_phone,
    requirement_items=json.dumps([
    {
        "text": item.text,
        "image_url": item.image_url
    }
    for item in data.requirement_items
]),    priority=data.priority,
    status="pending",
    created_by=current_user.id,
    created_by_name=current_user.name,
    created_at=datetime.now(IST),
)


    db.add(requirement)
    db.commit()
    db.refresh(requirement)

    return {"message": "Requirement created successfully"}
@api_router.get("/requirements")
def list_requirements(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=5, le=50),
    status: Optional[str] = Query("all"),
    priority: Optional[str] = Query("all"),
    search: Optional[str] = None,

    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    query = db.query(RequirementModel)

    # ---------- FILTERS ----------
    if status != "all":
        query = query.filter(RequirementModel.status == status)

    if priority != "all":
        query = query.filter(RequirementModel.priority == priority)

    if search:
        s = f"%{search.lower()}%"
        query = query.filter(
            or_(
                func.lower(RequirementModel.customer_name).like(s),
                func.lower(RequirementModel.customer_phone).like(s),
                func.lower(RequirementModel.created_by_name).like(s),
            )
        )

    total = query.count()

    rows = (
        query
        .order_by(RequirementModel.created_at.desc())
        .offset((page - 1) * limit)
        .limit(limit)
        .all()
    )

    data = []
    for r in rows:
        try:
            items = json.loads(r.requirement_items or "[]")
            if not isinstance(items, list):
                items = []
        except Exception:
            items = []

        data.append({
            "id": r.id,
            "customer_name": r.customer_name,
            "customer_phone": r.customer_phone,
            "requirement_items": items,
            "priority": r.priority,
            "status": r.status,
            "created_by": r.created_by_name,
            "created_at": r.created_at.isoformat(),
        })

    return {
        "data": data,
        "page": page,
        "limit": limit,
        "total": total,
        "total_pages": math.ceil(total / limit)
    }


@api_router.post("/requirements/{requirement_id}/complete")
def complete_requirement(
    requirement_id: str,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    req = (
        db.query(RequirementModel)
        .filter(RequirementModel.id == requirement_id)
        .first()
    )
    

    if not req:
        raise HTTPException(404, "Requirement not found")
    if req.status in ["completed", "rejected"]:
        raise HTTPException(400, "Requirement already finalized")
    # âœ… Mark as completed only
    req.status = "completed"
    req.completed_at = datetime.now(IST)

    db.commit()

    return {
        "message": "Requirement marked as completed",
        "requirement_id": req.id,
        "completed_at": req.completed_at.isoformat()
    }

@api_router.post("/requirements/{requirement_id}/reject")
def reject_requirement(
    requirement_id: str,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    req = db.query(RequirementModel).filter(
        RequirementModel.id == requirement_id
    ).first()
    

    if not req:
        raise HTTPException(404, "Requirement not found")
    if req.status in ["completed", "rejected"]:
        raise HTTPException(400, "Requirement already finalized")
    
    req.status = "rejected"
    req.completed_at = datetime.now(IST)   # âœ… ADD THIS

    db.commit()

    return {"message": "Requirement rejected"}

@api_router.get("/products/recent-skus")
def get_recent_skus(
    limit: int = Query(2, ge=1, le=10),
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    rows = (
        db.query(ProductModel.product_code)
        .filter(ProductModel.is_active == 1)
        .order_by(ProductModel.created_at.desc())
        .limit(limit)
        .all()
    )

    return {
        "recent_skus": [r.product_code for r in rows]
    }

@api_router.get("/products")
def get_products(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    search: Optional[str] = Query(None),
    category_id: Optional[str] = Query(None),
    sort: Optional[str] = Query(None),
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    offset = (page - 1) * limit

    query = db.query(ProductModel).filter(ProductModel.is_active == 1)

    # ðŸ” SEARCH FILTER (GLOBAL)
    if search:
        s = f"%{search.lower().strip()}%"
        query = query.filter(
            or_(
                func.lower(ProductModel.name).like(s),
                func.lower(ProductModel.sku).like(s),
                func.lower(ProductModel.product_code).like(s),
            )
        )

    # ðŸ“‚ CATEGORY FILTER
    if category_id and category_id != "all":
        query = query.filter(ProductModel.category_id == category_id)

    sort_map = {
        "name-asc": ProductModel.name.asc(),
        "name-desc": ProductModel.name.desc(),
        "sku-asc": ProductModel.sku.asc(),
        "sku-desc": ProductModel.sku.desc(),
        "selling-low-high": ProductModel.selling_price.asc(),
        "selling-high-low": ProductModel.selling_price.desc(),
        "cost-low-high": ProductModel.cost_price.asc(),
        "cost-high-low": ProductModel.cost_price.desc(),
        "stock-low-high": ProductModel.stock.asc(),
        "stock-high-low": ProductModel.stock.desc(),
        "category-asc": CategoryModel.name.asc(),
        "category-desc": CategoryModel.name.desc(),
    }

    sort_tokens = [token.strip() for token in (sort or "").split(",") if token.strip()]

    if any(token in {"category-asc", "category-desc"} for token in sort_tokens):
        query = query.outerjoin(CategoryModel, ProductModel.category_id == CategoryModel.id)

    total = query.count()
    order_by_clauses = []

    for token in sort_tokens:
        clause = sort_map.get(token)
        if clause is not None:
            order_by_clauses.append(clause)

    order_by_clauses.append(ProductModel.created_at.desc())

    products = (
        query
        .order_by(*order_by_clauses)
        .offset(offset)
        .limit(limit)
        .all()
    )

    result = []
    for p in products:
        item = {
            "id": p.id,
            "product_code": p.product_code,
            "sku": p.sku,
            "name": p.name,
            "description": p.description,
            "category_id": p.category_id,
            "category_name": p.category.name if p.category else None,
            "selling_price": p.selling_price,
            "min_selling_price": p.min_selling_price,
            "stock": p.stock,
            "min_stock": p.min_stock,
            "variants": p.variants or [],
            "images": safe_images(p.images),
            "qr_code_url": p.qr_code_url,
            "is_service": p.is_service,
            "created_at": p.created_at.isoformat(),
        }

        if current_user.role == "admin":
            item["cost_price"] = p.cost_price

        result.append(item)

    return {
        "data": result,
        "page": page,
        "limit": limit,
        "total": total,
        "total_pages": ceil(total / limit),
    }


def build_inventory_report(
    db: Session,
    search: Optional[str] = None,
    category_id: Optional[str] = None,
):
    product_match = and_(
        ProductModel.category_id == CategoryModel.id,
        ProductModel.is_active == 1,
        ProductModel.is_service == 0,
    )

    query = (
        db.query(
            CategoryModel.id.label("category_id"),
            CategoryModel.name.label("category_name"),
            ProductModel.id.label("product_id"),
            ProductModel.name.label("product_name"),
            ProductModel.product_code.label("product_code"),
            ProductModel.sku.label("sku"),
            ProductModel.stock.label("stock"),
            ProductModel.min_stock.label("min_stock"),
            ProductModel.selling_price.label("selling_price"),
        )
        .outerjoin(ProductModel, product_match)
    )

    if category_id and category_id != "all":
        query = query.filter(CategoryModel.id == category_id)

    if search:
        search_term = f"%{search.lower().strip()}%"
        query = query.filter(
            or_(
                func.lower(ProductModel.name).like(search_term),
                func.lower(ProductModel.sku).like(search_term),
                func.lower(ProductModel.product_code).like(search_term),
                func.lower(CategoryModel.name).like(search_term),
            )
        )

    rows = query.order_by(
        CategoryModel.name.asc(),
        ProductModel.name.asc(),
        ProductModel.created_at.asc(),
    ).all()

    grouped = {}
    summary = {
        "total_categories": 0,
        "total_products": 0,
        "total_stock": 0,
        "low_stock_products": 0,
    }

    for row in rows:
        category_key = row.category_id
        if category_key not in grouped:
            grouped[category_key] = {
                "category_id": row.category_id,
                "category_name": row.category_name,
                "product_count": 0,
                "total_stock": 0,
                "low_stock_count": 0,
                "products": [],
            }

        if not row.product_id:
            continue

        stock = int(row.stock or 0)
        min_stock = int(row.min_stock or 0)
        is_low_stock = stock <= min_stock

        grouped[category_key]["products"].append({
            "id": row.product_id,
            "name": row.product_name,
            "product_code": row.product_code,
            "sku": row.sku,
            "stock": stock,
            "min_stock": min_stock,
            "selling_price": float(row.selling_price or 0),
            "is_low_stock": is_low_stock,
        })
        grouped[category_key]["product_count"] += 1
        grouped[category_key]["total_stock"] += stock
        if is_low_stock:
            grouped[category_key]["low_stock_count"] += 1

        summary["total_products"] += 1
        summary["total_stock"] += stock
        if is_low_stock:
            summary["low_stock_products"] += 1

    categories = sorted(
        grouped.values(),
        key=lambda item: (
            -(item["product_count"] or 0),
            -(item["total_stock"] or 0),
            (item["category_name"] or "").lower(),
        ),
    )
    summary["total_categories"] = len(categories)

    return {
        "summary": summary,
        "categories": categories,
    }


@api_router.get("/inventory/report")
def get_inventory_report(
    search: Optional[str] = Query(None),
    category_id: Optional[str] = Query(None),
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return build_inventory_report(
        db=db,
        search=search,
        category_id=category_id,
    )


@api_router.get("/inventory/report/pdf")
def download_inventory_report_pdf(
    search: Optional[str] = Query(None),
    category_id: Optional[str] = Query(None),
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    report = build_inventory_report(
        db=db,
        search=search,
        category_id=category_id,
    )

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(letter),
        leftMargin=18,
        rightMargin=18,
        topMargin=18,
        bottomMargin=18,
    )
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Inventory Report", styles["Title"]))
    elements.append(Spacer(1, 8))
    elements.append(
        Paragraph(
            (
                f"Generated on: {datetime.now(IST).strftime('%d %b %Y, %I:%M %p')} | "
                f"Categories: {report['summary']['total_categories']} | "
                f"Products: {report['summary']['total_products']} | "
                f"Total Stock: {report['summary']['total_stock']}"
            ),
            styles["Normal"],
        )
    )
    elements.append(Spacer(1, 12))

    for category in report["categories"]:
        elements.append(
            Paragraph(
                (
                    f"{category['category_name']} "
                    f"({category['product_count']} products, stock {category['total_stock']})"
                ),
                styles["Heading2"],
            )
        )
        elements.append(Spacer(1, 6))

        table_data = [[
            "Check",
            "Product Name",
            "Product Code",
            "SKU",
            "Stock",
            "Selling Price",
        ]]

        if category["products"]:
            for product in category["products"]:
                table_data.append([
                    "[   ]",
                    Paragraph(product["name"], styles["BodyText"]),
                    product["product_code"],
                    product["sku"],
                    str(product["stock"]),
                    f"{float(product['selling_price']):.2f}",
                ])
        else:
            table_data.append([
                "",
                "No products found",
                "-",
                "-",
                "-",
                "-",
            ])

        table = Table(
            table_data,
            repeatRows=1,
            colWidths=[52, 280, 110, 120, 70, 112],
        )
        table_style = [
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 8),
            ("TOPPADDING", (0, 0), (-1, 0), 8),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 7),
            ("TOPPADDING", (0, 1), (-1, -1), 7),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("ALIGN", (0, 0), (0, -1), "CENTER"),
            ("ALIGN", (4, 1), (5, -1), "RIGHT"),
        ]

        for row_index, product in enumerate(category["products"], start=1):
            if product["is_low_stock"]:
                table_style.append(
                    ("TEXTCOLOR", (4, row_index), (4, row_index), colors.HexColor("#dc2626"))
                )
            else:
                table_style.append(
                    ("TEXTCOLOR", (4, row_index), (4, row_index), colors.HexColor("#16a34a"))
                )

        table.setStyle(TableStyle(table_style))
        elements.append(table)
        elements.append(Spacer(1, 14))

    doc.build(elements)
    buffer.seek(0)

    filename = f"inventory-report-{datetime.now(IST).strftime('%Y%m%d-%H%M%S')}.pdf"
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename={filename}"
        }
    )

@api_router.get("/products/ageing")
def product_ageing_report(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=50),
    bucket: Optional[str] = Query(None, pattern="^(daily|weekly|monthly)$"),

    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # âœ… USE NAIVE DATETIME (IMPORTANT)
    now = datetime.now(IST).replace(tzinfo=None)

    offset = (page - 1) * limit

    base_products = (
        db.query(ProductModel)
        .filter(
            ProductModel.is_active == 1,
            ProductModel.is_service == 0
        )
        .order_by(ProductModel.created_at.asc())
        .all()
    )

    def get_bucket(days: int) -> str:
        if days <= 30:
            return "latest"
        elif days <= 45:
            return "new"
        elif days <= 60:
            return "medium"
        elif days <= 90:
            return "old"
        elif days <= 150:
            return "very_old"
        else:
            return "dead_stock"

    enriched = []

    for p in base_products:
        # ðŸ”¹ FIRST INWARD DATE
        first_inward = (
            db.query(func.min(InventoryTransaction.created_at))
            .filter(
                InventoryTransaction.product_id == p.id,
                InventoryTransaction.type == "IN"
            )
            .scalar()
        )

        # âœ… NORMALIZE TO NAIVE
        base_date = first_inward or p.created_at
        if base_date.tzinfo is not None:
            base_date = base_date.replace(tzinfo=None)

        age_days = (now - base_date).days
        age_bucket = get_bucket(age_days)

        if bucket and age_bucket != bucket:
            continue

        enriched.append({
            "product_code": p.product_code,
            "sku": p.sku,
            "name": p.name,
            "qty": p.stock,
            "age_bucket": age_bucket,
            "first_inward_date": (
                first_inward.isoformat() if first_inward else None
            ),
            "age_days": age_days
        })

    total = len(enriched)
    paginated = enriched[offset : offset + limit]

    return {
        "page": page,
        "limit": limit,
        "total": total,
        "total_pages": math.ceil(total / limit),
        "data": paginated
    }

@api_router.get("/inventory/lookup/{code}")
def inventory_lookup(
    code: str,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    product, variant_sku = resolve_product_and_variant_by_sku(db, code)

    variants = list(product.variants or [])

    # ================= VARIANT LEVEL =================
    if variant_sku:
        for v in variants:
            if v.get("v_sku") == variant_sku:
                return {
                    "level": "VARIANT",
                    "product_id": product.id,
                    "product_name": product.name,
                    "parent_sku": product.sku,
                    "variant": {
                        "v_sku": v.get("v_sku"),
                        "color": v.get("color"),
                        "size": v.get("size"),
                        "stock": v.get("stock", 0),
                    },
                    "total_stock": product.stock
                }

        raise HTTPException(500, "Variant SKU resolved but missing")

    # ================= PRODUCT LEVEL =================
    return {
        "level": "PRODUCT",
        "product_id": product.id,
        "product_name": product.name,
        "parent_sku": product.sku,
        "variants": [
            {
                "v_sku": v.get("v_sku"),
                "color": v.get("color"),
                "size": v.get("size"),
                "stock": v.get("stock", 0),
            }
            for v in variants
        ],
        "total_stock": product.stock
    }

def safe_commit(db: Session, retries: int = 3):
    for attempt in range(retries):
        try:
            db.commit()
            return
        except OperationalError as e:
            db.rollback()

            # MySQL deadlock error code
            if "1213" in str(e):
                if attempt == retries - 1:
                    raise HTTPException(
                        status_code=409,
                        detail="Inventory is busy. Please retry."
                    )
                time.sleep(0.2)  # small backoff
            else:
                raise
            
@api_router.post(
    "/inventory/material-inward/sku",
    dependencies=[Depends(RateLimiter(times=30, seconds=60))]
)

def material_inward_by_sku(
    request: MaterialInwardBySkuRequest,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if request.quantity <= 0:
        raise HTTPException(status_code=400, detail="Quantity must be > 0")

    product, variant_sku = resolve_product_and_variant_by_sku(db, request.sku)
    if product.is_active == 0:
        raise HTTPException(
        status_code=400,
        detail="Product is archived and cannot be used"
    )

    if product.is_service == 1:
        raise HTTPException(status_code=400, detail="Inventory not allowed for services")

    variants = list(product.variants or [])

    # ðŸš¨ ENFORCE VARIANT-ONLY IF VARIANTS EXIST
    if variants and not variant_sku:
        raise HTTPException(
            status_code=400,
            detail="Product has variants. Use variant SKU for inventory."
        )

    variant_stock_after = None

    # ================= VARIANT LEVEL =================
    if variant_sku:
        total_before = calculate_total_stock(variants)

        for v in variants:
            if v.get("v_sku") == variant_sku:
                v["stock"] = int(v.get("stock", 0)) + request.quantity
                variant_stock_after = v["stock"]
                break
        else:
            raise HTTPException(status_code=400, detail="Variant SKU not found")

        product.variants = variants
        flag_modified(product, "variants")

        # âœ… ALWAYS derive product stock from variants
        product.stock = calculate_total_stock(variants)

        stock_before = total_before
        stock_after = product.stock

    # ================= PRODUCT LEVEL (NO VARIANTS) =================
    else:
        stock_before = int(product.stock or 0)
        product.stock = stock_before + request.quantity
        stock_after = product.stock

    db.add(InventoryTransaction(
    id=str(uuid.uuid4()),
    product_id=product.id,
    type="IN",
    quantity=request.quantity,
    source="MATERIAL_INWARD",

    stock_before=stock_before,
    stock_after=stock_after,
    variant_sku=variant_sku,
    variant_stock_after=variant_stock_after,

    created_by=current_user.id,
    created_by_name=current_user.name,   # âœ… ADD THIS
    created_at=datetime.now(IST)
))


    safe_commit(db)

    return {
        "message": "Stock added successfully",
        "variant_sku": variant_sku,
        "stock_after": stock_after,
        "variant_stock_after": variant_stock_after
    }

@api_router.post(
    "/inventory/material-outward/sku",
    dependencies=[Depends(RateLimiter(times=30, seconds=60))]
)
def material_outward_by_sku(
    request: MaterialOutwardBySkuRequest,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if request.quantity <= 0:
        raise HTTPException(status_code=400, detail="Quantity must be > 0")

    # âœ… FIRST resolve product
    product, variant_sku = resolve_product_and_variant_by_sku(db, request.sku)

    # âœ… NOW product exists â†’ safe check
    if product.is_active == 0:
        raise HTTPException(
            status_code=400,
            detail="Product is archived and cannot be used"
        )

    if product.is_service == 1:
        raise HTTPException(
            status_code=400,
            detail="Inventory not allowed for services"
        )

    variants = list(product.variants or [])

    # ðŸš¨ ENFORCE VARIANT-ONLY IF VARIANTS EXIST
    if variants and not variant_sku:
        raise HTTPException(
            status_code=400,
            detail="Product has variants. Use variant SKU for inventory."
        )

    variant_stock_after = None

    # ================= VARIANT LEVEL =================
    if variant_sku:
        total_before = calculate_total_stock(variants)

        for v in variants:
            if v.get("v_sku") == variant_sku:
                current_stock = int(v.get("stock", 0))
                if current_stock < request.quantity:
                    raise HTTPException(
                        status_code=400,
                        detail="Insufficient variant stock"
                    )

                v["stock"] = current_stock - request.quantity
                variant_stock_after = v["stock"]
                break
        else:
            raise HTTPException(status_code=400, detail="Variant SKU not found")

        product.variants = variants
        flag_modified(product, "variants")

        # âœ… ALWAYS derive product stock from variants
        product.stock = calculate_total_stock(variants)

        stock_before = total_before
        stock_after = product.stock

    # ================= PRODUCT LEVEL (NO VARIANTS) =================
    else:
        current_stock = int(product.stock or 0)
        if current_stock < request.quantity:
            raise HTTPException(
                status_code=400,
                detail="Insufficient product stock"
            )

        stock_before = current_stock
        product.stock = current_stock - request.quantity
        stock_after = product.stock

    db.add(
        InventoryTransaction(
            id=str(uuid.uuid4()),
            product_id=product.id,
            type="OUT",
            quantity=request.quantity,
            source="MATERIAL_OUTWARD",
            reason=request.reason,
            stock_before=stock_before,
            stock_after=stock_after,
            variant_sku=variant_sku,

            variant_stock_after=variant_stock_after,
            created_by=current_user.id,
            created_by_name=current_user.name,   # âœ… ADD

            created_at=datetime.now(IST),
        )
    )

    safe_commit(db)

    return {
        "message": "Stock deducted successfully",
        "variant_sku": variant_sku,
        "stock_after": stock_after,
        "variant_stock_after": variant_stock_after,
    }

@api_router.get("/inventory/transactions")
def get_inventory_transactions(
    page: int = 1,
    limit: int = 30,
    type: Optional[str] = None,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    offset = (page - 1) * limit

    q = (
        db.query(InventoryTransaction, ProductModel)
        .join(ProductModel, InventoryTransaction.product_id == ProductModel.id)
    )

    if type in {"IN", "OUT"}:
        q = q.filter(InventoryTransaction.type == type)

    total = q.count()

    rows = (
        q.order_by(InventoryTransaction.created_at.desc())
        .offset(offset)
        .limit(limit)
        .all()
    )

    return {
        "data": [
            {
                "id": txn.id,
                "product_id": txn.product_id,
                "product_name": prod.name,
                "product_code": prod.product_code,
                "type": txn.type,
                "quantity": txn.quantity,
                "variant_sku": txn.variant_sku,
                "variant_stock_after": txn.variant_stock_after,
                "stock_after": txn.stock_after,
                "created_at": txn.created_at.isoformat(),
                "created_by": txn.created_by_name,   # âœ… ADD

            }
            for txn, prod in rows
        ],
        "total": total,
        "page": page,
        "limit": limit
    }

@api_router.post("/products", status_code=201)
def create_product(
    product_data: ProductCreate,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # ðŸ” NON-ADMIN â†’ FORCE COST PRICE = 0
    if current_user.role != "admin":
        product_data.cost_price = 0

    # ðŸ” VALIDATIONS
    if product_data.selling_price < product_data.min_selling_price:
        raise HTTPException(400, "Selling price cannot be below minimum selling price")

    if len(product_data.images) > 5:
        raise HTTPException(400, "Maximum 5 images allowed")

    category = db.query(CategoryModel).filter(
        CategoryModel.id == product_data.category_id
    ).first()
    if not category:
        raise HTTPException(400, "Invalid category")

    parent_sku = product_data.sku or f"SKU-{uuid.uuid4().hex[:8].upper()}"

    # ==========================================================
    # ðŸ”„ CHECK EXISTING SKU (ACTIVE OR ARCHIVED)
    # ==========================================================
    existing = db.query(ProductModel).filter(
        ProductModel.sku == parent_sku
    ).first()

    # ==========================================================
    # ðŸ”„ RESTORE ARCHIVED PRODUCT
    # ==========================================================
    if existing:
        if existing.is_active == 0:
            existing.is_active = 1
            existing.name = product_data.name
            existing.description = product_data.description
            existing.category_id = product_data.category_id
            existing.selling_price = product_data.selling_price
            existing.min_selling_price = product_data.min_selling_price
            existing.images = product_data.images
            existing.is_service = product_data.is_service
            existing.min_stock = product_data.min_stock if not product_data.is_service else 0

            # âœ… COST PRICE â†’ ADMIN ONLY
            if current_user.role == "admin":
                existing.cost_price = product_data.cost_price

            # ðŸ”’ ERP RULE: NO STOCK DURING RESTORE
            if product_data.is_service == 1:
                existing.variants = []
                existing.stock = 0
            else:
                restored_variants = []
                for v in product_data.variants or []:
                    vd = v.dict()
                    vd["stock"] = 0  # ðŸ”’ FORCE ZERO
                    vd["qr_code_url"] = generate_qr({
                        "type": "variant",
                        "product_name": product_data.name,
                        "sku": parent_sku,
                        "v_sku": vd.get("v_sku"),
                        "price": product_data.selling_price,
                        "color": vd.get("color"),
                        "size": vd.get("size"),
                    })
                    restored_variants.append(vd)

                existing.variants = restored_variants
                existing.stock = 0

            db.commit()
            db.refresh(existing)
            return existing

        # âŒ ACTIVE SKU EXISTS
        raise HTTPException(409, "SKU already exists")

    # ==========================================================
    # ðŸ†• CREATE NEW PRODUCT
    # ==========================================================
    product_code = generate_product_code(db)

    if product_data.is_service == 1:
        variants = []
        min_stock = 0
    else:
        variants = []
        for v in product_data.variants or []:
            vd = v.dict()
            vd["stock"] = 0
            variants.append(vd)
        min_stock = product_data.min_stock

    # ðŸ“¦ PRODUCT QR
    qr_code_url = generate_qr({
        "type": "product",
        "name": product_data.name,
        "sku": parent_sku,
        "price": product_data.selling_price,
    })

    # ðŸ“¦ VARIANT QR
    enriched_variants = []
    for v in variants:
        v["qr_code_url"] = generate_qr({
            "type": "variant",
            "product_name": product_data.name,
            "sku": parent_sku,
            "v_sku": v.get("v_sku"),
            "price": product_data.selling_price,
            "color": v.get("color"),
            "size": v.get("size"),
        })
        enriched_variants.append(v)

    product = ProductModel(
        id=str(uuid.uuid4()),
        product_code=product_code,
        sku=parent_sku,
        name=product_data.name,
        description=product_data.description,
        category_id=product_data.category_id,
        cost_price=product_data.cost_price,  # already 0 for non-admin
        selling_price=product_data.selling_price,
        min_selling_price=product_data.min_selling_price,
        stock=0,
        min_stock=min_stock,
        variants=enriched_variants,
        images=product_data.images,
        is_service=product_data.is_service,
        qr_code_url=qr_code_url,
        is_active=1,
        created_at=datetime.now(IST)
    )

    db.add(product)
    db.commit()
    db.refresh(product)

    return product


@api_router.put("/products/{product_id}")
def update_product(
    product_id: str,
    product_data: ProductCreate,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    product = db.query(ProductModel).filter(ProductModel.id == product_id).first()
    if not product:
        raise HTTPException(404, "Product not found")

    if current_user.role not in ["admin", "store_handler"]:
        raise HTTPException(403, "Not allowed")


    if len(product_data.images) > 5:
        raise HTTPException(400, "Maximum 5 images allowed")

    if product_data.selling_price < product_data.min_selling_price:
        raise HTTPException(400, "Selling price cannot be below minimum selling price")
    if current_user.role == "admin":
        product.cost_price = product_data.cost_price

    product.name = product_data.name
    product.description = product_data.description
    product.category_id = product_data.category_id
    product.sku = product_data.sku or product.sku
    product.selling_price = product_data.selling_price
    product.min_selling_price = product_data.min_selling_price
    product.images = product_data.images
    product.is_service = product_data.is_service
    product.min_stock = product_data.min_stock

    # ================= STRICT ERP RULE =================
    # âŒ DO NOT CHANGE STOCK
    # âŒ DO NOT RECALCULATE STOCK

    if product_data.is_service == 1:
        product.variants = []
    else:
        existing_variants = {v["v_sku"]: v for v in (product.variants or [])}
        updated_variants = []

        for v in product_data.variants or []:
            vd = v.dict()
            old = existing_variants.get(vd.get("v_sku"), {})

            vd["stock"] = old.get("stock", 0)   # ðŸ”’ PRESERVE STOCK

            vd["qr_code_url"] = generate_qr({
                "type": "variant",
                "product_name": product.name,
                "sku": product.sku,
                "v_sku": vd.get("v_sku"),
                "price": vd.get("v_selling_price") or product.selling_price,
                "color": vd.get("color"),
                "size": vd.get("size"),
            })

            updated_variants.append(vd)

        product.variants = updated_variants

    db.commit()
    db.refresh(product)
    return product

@api_router.delete("/products/{product_id}")
def delete_or_archive_product(
    product_id: str,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role != "admin":
        raise HTTPException(403, "Admin only")

    product = db.query(ProductModel).filter(
        ProductModel.id == product_id,
        ProductModel.is_active == 1
    ).first()

    if not product:
        raise HTTPException(404, "Product not found")

    if not has_inventory(db, product_id) and not has_invoice(db, product_id):
        # âœ… HARD DELETE
        db.delete(product)
        db.commit()
        return {
            "message": "Product deleted permanently",
            "action": "hard_delete"
        }

    # âŒ USED â†’ ARCHIVE
    product.is_active = 0
    db.commit()
    return {
        "message": "Product archived (used in inventory/invoices)",
        "action": "archived"
    }
################################COMBOS#############################
@api_router.post("/combos")
def create_combo(
    data: ComboCreate,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:
        print("COMBO REQUEST:", data.dict())

        # ================= BASIC VALIDATION =================
        if not data.name or not data.name.strip():
            raise HTTPException(400, "Combo name is required")

        if not data.items or len(data.items) == 0:
            raise HTTPException(400, "Combo must have at least 1 item")

        if data.price is None or data.price <= 0:
            raise HTTPException(400, "Combo price must be greater than 0")

        merged_items = {}

        # ================= PROCESS ITEMS =================
        for item in data.items:
            print("ITEM:", item.dict())

            if item.quantity <= 0:
                raise HTTPException(400, "Quantity must be greater than 0")

            product = db.query(ProductModel).filter(
                ProductModel.id == item.product_id,
                ProductModel.is_active == 1
            ).first()

            if not product:
                raise HTTPException(404, f"Product not found: {item.product_id}")

            variants = list(product.variants or [])
            sku = item.sku

            # ================= VARIANT HANDLING =================
            if len(variants) > 0:

                # 🔥 AUTO PICK FIRST VARIANT IF SKU NOT PROVIDED
                if not sku:
                    first_variant = variants[0]
                    sku = first_variant.get("v_sku")
                    print(f"Auto-selected SKU for {product.name}: {sku}")

                variant = next(
                    (v for v in variants if v.get("v_sku") == sku),
                    None
                )

                if not variant:
                    raise HTTPException(
                        400,
                        f"Invalid SKU for product: {product.name}"
                    )

            else:
                # no variants → ignore sku
                sku = None

            # ================= MERGE DUPLICATES =================
            key = f"{item.product_id}_{sku}"

            if key not in merged_items:
                merged_items[key] = {
                    "product_id": item.product_id,
                    "sku": sku,
                    "quantity": item.quantity
                }
            else:
                merged_items[key]["quantity"] += item.quantity

        # ================= FINAL ITEMS =================
        items_data = list(merged_items.values())

        # ================= CREATE COMBO =================
        combo = ComboModel(
            id=str(uuid.uuid4()),
            name=data.name.strip(),
            items=items_data,
            price=float(data.price),
            is_active=1,
            created_at=datetime.now(IST)
        )

        db.add(combo)
        db.commit()
        db.refresh(combo)

        return {
            "id": combo.id,
            "name": combo.name,
            "items": combo.items,
            "price": combo.price,
            "message": "Combo created successfully"
        }

    except HTTPException as e:
        print("HTTP ERROR:", e.detail)
        raise
    except Exception as e:
        print("UNKNOWN ERROR:", str(e))
        raise HTTPException(500, "Failed to create combo")
@api_router.put("/combos/{combo_id}")
def update_combo(
    combo_id: str,
    data: ComboCreate,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    combo = db.query(ComboModel).filter(
        ComboModel.id == combo_id
    ).first()

    if not combo:
        raise HTTPException(404, "Combo not found")

    # ✅ VALIDATE PRICE
    if data.price is None or float(data.price) < 0:
        raise HTTPException(400, "Invalid combo price")

    items_data = []

    if not data.items or len(data.items) == 0:
        raise HTTPException(400, "Combo must have at least one item")

    for item in data.items:

        # ✅ VALIDATE PRODUCT
        product = db.query(ProductModel).filter(
            ProductModel.id == item.product_id,
            ProductModel.is_active == 1
        ).first()

        if not product:
            raise HTTPException(
                404,
                f"Product not found: {item.product_id}"
            )

        # ✅ VALIDATE QUANTITY
        if not item.quantity or int(item.quantity) <= 0:
            raise HTTPException(
                400,
                f"Invalid quantity for product: {product.name}"
            )

        selected_variant = None

        # ================= VARIANT VALIDATION =================
        if product.variants and len(product.variants) > 0:

            if not item.sku:
                raise HTTPException(
                    400,
                    f"Variant SKU required for product: {product.name}"
                )

            selected_variant = next(
                (v for v in (product.variants or [])
                 if v.get("v_sku") == item.sku),
                None
            )

            if not selected_variant:
                raise HTTPException(
                    400,
                    f"Invalid SKU for product: {product.name}"
                )

        # ================= FINAL ITEM =================
        items_data.append({
            "product_id": item.product_id,
            "sku": item.sku if selected_variant else None,
            "quantity": int(item.quantity)
        })

    # ================= UPDATE =================
    combo.name = data.name.strip()
    combo.items = items_data
    combo.price = float(data.price)

    db.commit()
    db.refresh(combo)

    return {
        "message": "Combo updated successfully",
        "id": combo.id,
        "name": combo.name,
        "total_items": len(items_data),
        "price": combo.price
    }


@api_router.delete("/combos/{combo_id}")
def delete_combo(
    combo_id: str,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    combo = db.query(ComboModel).filter(
        ComboModel.id == combo_id
    ).first()

    if not combo:
        raise HTTPException(404, "Combo not found")

    # ❗ OPTIONAL SAFETY CHECK
    # Prevent deleting if used in invoices
    invoices = db.query(InvoiceModel).all()

    for inv in invoices:
        try:
            items = json.loads(inv.items or "[]")
        except:
            items = []

        for item in items:
            if item.get("combo_id") == combo_id:
                raise HTTPException(
                    400,
                    f"Cannot delete combo. Used in invoice: {inv.invoice_number or inv.id}"
                )

    # ✅ DELETE
    db.delete(combo)
    db.commit()

    return {
        "message": "Combo deleted successfully",
        "id": combo_id
    }

@api_router.get("/combos")
def get_combos(
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user)
):
    combos = db.query(ComboModel).filter(
        ComboModel.is_active == 1
    ).all()

    result = []

    for combo in combos:

        enriched_items = []

        for item in combo.items:

            product = db.query(ProductModel).filter(
                ProductModel.id == item["product_id"]
            ).first()

            if not product:
                continue

            sku = item.get("sku")
            variant = None

            # 🔥 FIND VARIANT (DICT SAFE)
            if sku and product.variants:
                variant = next(
                    (v for v in (product.variants or []) if v.get("v_sku") == sku),
                    None
                )

            # 🔥 PREPARE VARIANTS LIST (IMPORTANT)
            variants_list = []
            if product.variants:
                for v in (product.variants or []):
                    variants_list.append({
                        "v_sku": v.get("v_sku"),
                        "variant_name": v.get("variant_name"),
                        "color": v.get("color"),
                        "size": v.get("size"),
                        "v_selling_price": (
                            float(v.get("v_selling_price"))
                            if v.get("v_selling_price") is not None
                            else float(product.selling_price or 0)
                        ),
                        "selling_price": float(product.selling_price or 0),
                        "image_url": v.get("image_url"),
                        "stock": v.get("stock", 0)
                    })

            item_price = (
                float(variant.get("v_selling_price"))
                if variant and variant.get("v_selling_price") is not None
                else float(product.selling_price or 0)
            )
            enriched_items.append({
                "product_id": product.id,
                "product_name": product.name,

                # 🔥 PRICE LOGIC
                "price": item_price,
                "selling_price": float(product.selling_price or 0),

                "quantity": item.get("quantity", 1),
                "sku": sku,

                "image_url": (
                    variant.get("image_url")
                    if variant and variant.get("image_url")
                    else (product.images or [None])[0]
                ),

                # ✅ REQUIRED FOR FRONTEND
                "has_variants": len(product.variants or []) > 0,
                "variants": variants_list,

                # OPTIONAL SELECTED VARIANT INFO
                "variant_info": {
                    "v_sku": variant.get("v_sku"),
                    "variant_name": variant.get("variant_name"),
                    "color": variant.get("color"),
                    "size": variant.get("size"),
                } if variant else None
            })

        result.append({
            "id": combo.id,
            "name": combo.name,
            "price": combo.price,
            "items": enriched_items
        })

    return result

##############################CUSTOMERS & INVOICES (DASHBOARD)##############################
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
        current_balance=float(new_customer.current_balance or 0),
        created_at=new_customer.created_at.isoformat()
    )

@api_router.get("/customers")
def get_customers(
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    customers = (
        db.query(
            CustomerModel,
            func.count(InvoiceModel.id).label("total_invoices"),
            func.coalesce(func.sum(InvoiceModel.total), 0).label("total_bill")
        )
        .outerjoin(
            InvoiceModel,
            and_(
                InvoiceModel.customer_id == CustomerModel.id,
                InvoiceModel.invoice_type == "FINAL",
                InvoiceModel.payment_status != "cancelled"
            )
        )
        .group_by(CustomerModel.id)
        .order_by(CustomerModel.created_at.desc())
        .all()
    )

    return [
        {
            "id": c.id,
            "name": c.name,
            "email": c.email,
            "phone": c.phone,
            "address": c.address,
            "current_balance": calculate_customer_current_balance(db, c),
            "created_at": c.created_at.isoformat(),
            "total_invoices": int(total_invoices),
            "total_bill": float(total_bill),
        }
        for c, total_invoices, total_bill in customers
    ]


@api_router.get("/customers/{customer_id}/statement")
def get_customer_statement(
    customer_id: str,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    customer = db.query(CustomerModel).filter(CustomerModel.id == customer_id).first()
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")

    ledger_entries = (
        db.query(LedgerEntry)
        .filter(LedgerEntry.customer_id == customer_id)
        .order_by(LedgerEntry.entry_date.asc(), LedgerEntry.created_at.asc())
        .all()
    )

    running_balance = float(customer.opening_balance or 0)
    ledger_data = []
    total_debit = 0.0
    total_credit = 0.0

    for entry in ledger_entries:
        debit = float(entry.debit or 0)
        credit = float(entry.credit or 0)
        total_debit += debit
        total_credit += credit
        running_balance += credit - debit

        ledger_data.append({
            "id": entry.id,
            "date": entry.entry_date.isoformat() if entry.entry_date else None,
            "description": entry.description,
            "debit": debit,
            "credit": credit,
            "balance": round(running_balance, 2),
            "type": entry.entry_type,
            "payment_mode": entry.payment_mode,
            "reference_id": entry.reference_id,
        })

    invoices = (
        db.query(InvoiceModel)
        .filter(
            InvoiceModel.customer_id == customer_id,
            InvoiceModel.invoice_type == "FINAL"
        )
        .order_by(InvoiceModel.created_at.desc())
        .all()
    )

    invoice_data = []
    total_sales = 0.0
    total_paid = 0.0
    total_pending = 0.0
    current_balance = calculate_customer_current_balance(db, customer)

    for invoice in invoices:
        display_total = float(invoice.original_total or invoice.total or 0)
        paid_amount = float(invoice.paid_amount or 0)
        balance_amount = (
            float(invoice.balance_amount or 0)
            if invoice.payment_status in ["pending", "partial"]
            else 0.0
        )
        advance_used = float(invoice.advance_used or 0)
        total_sales += display_total
        total_paid += paid_amount
        total_pending += balance_amount

        try:
            raw_items = json.loads(invoice.items or "[]")
            parsed_items = raw_items if isinstance(raw_items, list) else []
        except Exception:
            parsed_items = []

        item_list = []
        for item in parsed_items:
            variant_info = item.get("variant_info") or {}
            item_list.append({
                "product_name": item.get("product_name") or item.get("name") or "Item",
                "quantity": float(item.get("quantity") or 0),
                "price": float(item.get("price") or 0),
                "total": float(item.get("total") or 0),
                "color": item.get("color") or variant_info.get("color"),
                "size": item.get("size") or variant_info.get("size"),
                "variant_name": item.get("variant_name") or variant_info.get("variant_name"),
            })

        invoice_data.append({
            "id": invoice.id,
            "invoice_number": invoice.invoice_number,
            "created_at": invoice.created_at.isoformat() if invoice.created_at else None,
            "payment_status": invoice.payment_status,
            "payment_mode": invoice.payment_mode,
            "total": float(invoice.total or 0),
            "display_total": display_total,
            "original_total": display_total,
            "advance_used": advance_used,
            "paid_amount": paid_amount,
            "balance_amount": balance_amount,
            "subtotal": float(invoice.subtotal or 0),
            "gst_amount": float(invoice.gst_amount or 0),
            "discount": float(invoice.discount or 0),
            "items": item_list,
        })

    return {
        "customer": {
            "id": customer.id,
            "name": customer.name,
            "email": customer.email,
            "phone": customer.phone,
            "address": customer.address,
            "current_balance": current_balance,
            "opening_balance": float(customer.opening_balance or 0),
            "total_invoices": len([inv for inv in invoices if inv.payment_status != "cancelled"]),
            "total_bill": round(total_sales, 2),
        },
        "summary": {
            "total_sales": round(total_sales, 2),
            "total_paid": round(total_paid, 2),
            "total_pending": round(total_pending, 2),
            "total_advance_available": round(abs(current_balance), 2) if current_balance < 0 else 0,
            "ledger_debit": round(total_debit, 2),
            "ledger_credit": round(total_credit, 2),
        },
        "ledger": list(reversed(ledger_data)),
        "invoices": invoice_data,
    }

@api_router.put("/customers/{customer_id}", response_model=Customer)
def update_customer(
    customer_id: str,
    customer_data: CustomerUpdate,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    customer = db.query(CustomerModel).filter(CustomerModel.id == customer_id).first()
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")

    customer.name = customer_data.name
    customer.email = customer_data.email or ""
    customer.phone = customer_data.phone
    customer.address = customer_data.address

    db.commit()
    db.refresh(customer)

    return Customer(
        id=customer.id,
        name=customer.name,
        email=customer.email,
        phone=customer.phone,
        address=customer.address,
        current_balance=calculate_customer_current_balance(db, customer),
        created_at=customer.created_at.isoformat()
    )

@api_router.delete("/customers/{customer_id}")
def delete_customer(
    customer_id: str,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    customer = db.query(CustomerModel).filter(CustomerModel.id == customer_id).first()
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")

    linked_invoices = db.query(InvoiceModel.id).filter(
        InvoiceModel.customer_id == customer_id,
        InvoiceModel.invoice_type == "FINAL"
    ).first()

    if linked_invoices:
        raise HTTPException(
            status_code=400,
            detail="Customer has invoices and cannot be deleted"
        )

    db.delete(customer)
    db.commit()

    return {"success": True}


def generate_invoice_number(db: Session):
    now = datetime.now(IST)

    # Financial Year (Aprâ€“Mar)
    fy_year = now.year if now.month >= 4 else now.year - 1
    fy_suffix = f"{fy_year % 100:02d}-{(fy_year + 1) % 100:02d}"

    # ðŸ”’ LOCK + GET LAST INVOICE NUMBER
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


def parse_invoice_items(raw_items):
    if not raw_items:
        return []
    try:
        return json.loads(raw_items)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(raw_items)
        except Exception:
            return []

# Fixed get_invoices to include JWT authentication and use json instead of eval

@api_router.get("/invoices")
def get_invoices(
    page: int = 1,
    limit: int = 10,
    status: Optional[str] = None,   # paid | overdue | ending | cancelled
    range: Optional[str] = None,    # today | week | last10 | last30 | lastMonth
    month: Optional[str] = None,    # YYYY-MM
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):

    query = db.query(InvoiceModel).filter(
        InvoiceModel.invoice_type == "FINAL"
    )

    # ================= TIME SETUP =================
    now = datetime.now(IST)

    start_of_today = datetime(
        now.year, now.month, now.day,
        tzinfo=IST
    )

    end_of_today = start_of_today + timedelta(days=1)

    # ================= STATUS FILTER =================
    if status == "paid":
        query = query.filter(InvoiceModel.payment_status == "paid")

    elif status == "pending":
        query = query.filter(InvoiceModel.payment_status == "pending")

    elif status == "cancelled":
        query = query.filter(InvoiceModel.payment_status == "cancelled")

    elif status == "partial":
        query = query.filter(InvoiceModel.payment_status == "partial")

    elif status == "ending":
        query = query.filter(
            InvoiceModel.payment_status != "paid",
            InvoiceModel.created_at.between(
                start_of_today,
                start_of_today + timedelta(days=5)
            )
        )

    # ================= DATE RANGE =================
    if range == "today":
        query = query.filter(
            InvoiceModel.created_at.between(
                start_of_today,
                end_of_today
            )
        )

    elif range == "week":
        start_of_week = start_of_today - timedelta(days=start_of_today.weekday())
        query = query.filter(InvoiceModel.created_at >= start_of_week)

    elif range == "last10":
        query = query.filter(
            InvoiceModel.created_at >= start_of_today - timedelta(days=9)
        )

    elif range == "last30":
        query = query.filter(
            InvoiceModel.created_at >= start_of_today - timedelta(days=29)
        )

    elif range == "lastMonth":
        first_day_this_month = datetime(now.year, now.month, 1, tzinfo=IST)
        last_day_previous_month = first_day_this_month - timedelta(days=1)
        start_of_last_month = datetime(
            last_day_previous_month.year,
            last_day_previous_month.month,
            1,
            tzinfo=IST
        )
        end_of_last_month = first_day_this_month
        query = query.filter(
            InvoiceModel.created_at.between(start_of_last_month, end_of_last_month)
        )

    # ================= MONTH FILTER =================
    if month:

        year, month_num = map(int, month.split("-"))

        start_date = datetime(
            year, month_num, 1,
            tzinfo=IST
        )

        end_date = start_date + timedelta(days=31)

        query = query.filter(
            InvoiceModel.created_at.between(start_date, end_date)
        )

    # ================= PAGINATION =================
    total = query.count()

    invoices = (
        query
        .order_by(InvoiceModel.created_at.desc())
        .offset((page - 1) * limit)
        .limit(limit)
        .all()
    )

    # ================= RESPONSE =================
    return {
        "data": [
            {
                "id": inv.id,
                "invoice_number": inv.invoice_number,

                "customer_id": inv.customer_id,
                "customer_name": inv.customer_name,
                "customer_phone": inv.customer_phone,
                "customer_address": inv.customer_address,

                "items": parse_invoice_items(inv.items),

                "subtotal": inv.subtotal,

                "gst_amount": inv.gst_amount,
                "gst_enabled": bool(inv.gst_enabled),
                "gst_rate": inv.gst_rate,

                "discount": inv.discount,
                "original_total": float(inv.original_total or inv.total or 0),
                "advance_used": float(inv.advance_used or 0),
                "display_total": float(inv.original_total or inv.total or 0),
                "total": inv.total,

                # â­ PAYMENT DETAILS
                "paid_amount": float(inv.paid_amount or 0),
                "balance_amount": float(inv.balance_amount or 0),  # âœ… Fixed: Don't use inv.total as fallback
                "payment_mode": inv.payment_mode,

                "payment_status": inv.payment_status,

                "additional_charges": inv.additional_charges or [],

                "pdf_url": f"/api/invoices/{inv.id}/pdf",

                "created_at": inv.created_at.isoformat(),
                "created_by": inv.created_by_name,
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

@api_router.get("/invoices/export/pdf")
def export_invoices_pdf(
    range: str = Query("today"),
    status: Optional[str] = Query(None),
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    query = db.query(InvoiceModel).filter(InvoiceModel.invoice_type == "FINAL")

    now = datetime.now(IST)
    start_of_today = datetime(now.year, now.month, now.day, tzinfo=IST)
    end_of_today = start_of_today + timedelta(days=1)

    if status and status != "all":
        query = query.filter(InvoiceModel.payment_status == status)

    if range == "today":
        query = query.filter(InvoiceModel.created_at.between(start_of_today, end_of_today))
    elif range == "week":
        start_of_week = start_of_today - timedelta(days=start_of_today.weekday())
        query = query.filter(InvoiceModel.created_at >= start_of_week)
    elif range == "last10":
        query = query.filter(InvoiceModel.created_at >= start_of_today - timedelta(days=9))
    elif range == "lastMonth":
        first_day_this_month = datetime(now.year, now.month, 1, tzinfo=IST)
        last_day_previous_month = first_day_this_month - timedelta(days=1)
        start_of_last_month = datetime(
            last_day_previous_month.year,
            last_day_previous_month.month,
            1,
            tzinfo=IST
        )
        query = query.filter(InvoiceModel.created_at.between(start_of_last_month, first_day_this_month))
    else:
        raise HTTPException(status_code=400, detail="Unsupported export range")

    invoices = query.order_by(InvoiceModel.created_at.desc()).all()

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=landscape(letter),
        leftMargin=24,
        rightMargin=24,
        topMargin=24,
        bottomMargin=24
    )

    styles = getSampleStyleSheet()
    elements = [
        Paragraph("Invoice Export Report", styles["Title"]),
        Spacer(1, 8),
        Paragraph(
            f"Range: {range} | Generated on: {datetime.now(IST).strftime('%d %b %Y, %I:%M %p')}",
            styles["BodyText"]
        ),
        Spacer(1, 12),
    ]

    table_data = [[
        "Invoice #",
        "Date",
        "Customer",
        "Phone",
        "Total",
        "Paid",
        "Balance",
        "Mode",
        "Status",
    ]]

    total_amount = 0.0
    total_paid = 0.0
    total_balance = 0.0

    for invoice in invoices:
        invoice_total = float(invoice.total or 0)
        invoice_paid = float(invoice.paid_amount or 0)
        invoice_balance = float(invoice.balance_amount or 0)

        total_amount += invoice_total
        total_paid += invoice_paid
        total_balance += invoice_balance

        table_data.append([
            invoice.invoice_number or "-",
            invoice.created_at.astimezone(IST).strftime("%d-%b-%Y"),
            invoice.customer_name or "-",
            invoice.customer_phone or "-",
            f"Rs. {invoice_total:.2f}",
            f"Rs. {invoice_paid:.2f}",
            f"Rs. {invoice_balance:.2f}",
            (invoice.payment_mode or "-").upper(),
            (invoice.payment_status or "-").upper(),
        ])

    table_data.append([
        "",
        "",
        "Totals",
        "",
        f"Rs. {total_amount:.2f}",
        f"Rs. {total_paid:.2f}",
        f"Rs. {total_balance:.2f}",
        "",
        str(len(invoices)),
    ])

    table = Table(
        table_data,
        repeatRows=1,
        colWidths=[88, 62, 132, 86, 70, 70, 70, 62, 62]
    )
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.white, colors.HexColor("#f8fafc")]),
        ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#e2e8f0")),
        ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
        ("ALIGN", (4, 1), (6, -1), "RIGHT"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(table)

    doc.build(elements)
    buffer.seek(0)

    filename = f"invoices-{range}-{datetime.now(IST).strftime('%Y%m%d-%H%M%S')}.pdf"
    return StreamingResponse(
        buffer,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )

@api_router.post(
    "/invoices",
    dependencies=[Depends(RateLimiter(times=20, seconds=60))]
)
def create_invoice(
    invoice_data: InvoiceCreate,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    try:

        # ================= VALIDATION =================
        allowed_status = ["pending", "paid", "partial", "cancelled"]

        if invoice_data.payment_status not in allowed_status:
            raise HTTPException(400, "Invalid payment status")

        if not invoice_data.items:
            raise HTTPException(400, "Invoice must contain items")

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
                email=invoice_data.customer_email or "walkin@test.com",
                phone=invoice_data.customer_phone,
                address=invoice_data.customer_address,
                created_at=datetime.now(IST),
            )
            db.add(customer)
            db.flush()

        # ================= ITEMS =================
        raw_items = []
        subtotal = 0.0

        for item in invoice_data.items:

            # 🔥 COMBO SUPPORT
            if getattr(item, "combo_id", None):

                combo = db.query(ComboModel).filter(
                    ComboModel.id == item.combo_id,
                    ComboModel.is_active == 1
                ).first()

                if not combo:
                    raise HTTPException(404, "Combo not found")

                for c in combo.items:
                    product = db.query(ProductModel).filter(
                        ProductModel.id == c["product_id"]
                    ).first()

                    if not product:
                        continue

                    quantity = int(c["quantity"]) * int(item.quantity)

                    variant = None
                    if c.get("sku") and product.variants:
                        variant = next(
                            (v for v in product.variants if v.get("v_sku") == c.get("sku")),
                            None
                        )

                    price = (
                        float(variant.get("v_selling_price"))
                        if variant and variant.get("v_selling_price") is not None
                        else float(product.selling_price)
                    )

                    subtotal += price * quantity

                    raw_items.append({
                        "product_id": product.id,
                        "sku": c.get("sku"),
                        "product_name": product.name,
                        "quantity": quantity,
                        "price": price,
                        "is_service": product.is_service,
                    })

                continue

            # 🔥 NORMAL PRODUCT
            quantity = int(item.quantity)
            price = float(item.price)

            subtotal += price * quantity

            raw_items.append({
                "product_id": item.product_id,
                "sku": item.sku,
                "product_name": item.product_name,
                "quantity": quantity,
                "price": price,
                "is_service": item.is_service or 0,
            })

        # ================= MERGE =================
        merged = {}
        for i in raw_items:
            key = f"{i['product_id']}_{i.get('sku')}"
            if key not in merged:
                merged[key] = i
            else:
                merged[key]["quantity"] += i["quantity"]

        invoice_items = list(merged.values())

        original_total = round(subtotal, 2)
        advance_used, total = calculate_invoice_advance(
            customer,
            original_total,
            should_use_advance(invoice_data.use_advance),
            db
        )

        requested_payment_status = normalize_draft_payment_status(invoice_data.payment_status)
        paid_amount, balance_amount = calculate_draft_payment_summary(
            total,
            requested_payment_status,
            invoice_data.paid_amount,
            advance_used
        )
        resolved_payment_status = resolve_invoice_payment_status(
            requested_payment_status,
            advance_used,
            balance_amount,
            paid_amount,
        )

        invoice = InvoiceModel(
            id=str(uuid.uuid4()),
            invoice_number=generate_invoice_number(db),
            invoice_type="FINAL",

            customer_id=customer.id,
            customer_name=customer.name,

            items=json.dumps(invoice_items),
            subtotal=subtotal,
            original_total=original_total,
            advance_used=advance_used,
            total=total,

            paid_amount=paid_amount,
            balance_amount=balance_amount,
            payment_mode=normalize_draft_payment_mode(invoice_data.payment_mode),
            payment_status=resolved_payment_status,

            created_by=current_user.id,
            created_by_name=current_user.name,
            created_at=datetime.now(IST),
        )

        db.add(invoice)
        db.flush()

        # ================= INVENTORY + TRANSACTION =================
        for item in invoice_items:

            if item.get("is_service") == 1 or not item.get("product_id"):
                continue

            product = db.query(ProductModel).filter(
                ProductModel.id == item["product_id"]
            ).with_for_update().first()

            quantity = int(item["quantity"])
            sku = item.get("sku")

            variants = list(product.variants or [])
            variant_stock_after = None

            if variants:
                for v in variants:
                    if v.get("v_sku") == sku:
                        stock_before = int(v.get("stock", 0))

                        if stock_before < quantity:
                            raise HTTPException(400, f"Insufficient stock for {sku}")

                        v["stock"] = stock_before - quantity
                        variant_stock_after = v["stock"]
                        break

                product.variants = variants
                product.stock = sum(int(v.get("stock", 0)) for v in variants)
                stock_after = product.stock

                flag_modified(product, "variants")
                flag_modified(product, "stock")

            else:
                stock_before = int(product.stock or 0)

                if stock_before < quantity:
                    raise HTTPException(400, f"Insufficient stock for {product.name}")

                product.stock = stock_before - quantity
                stock_after = product.stock

                flag_modified(product, "stock")

            db.flush()

            # 🔥 INVENTORY ENTRY
            db.add(InventoryTransaction(
                id=str(uuid.uuid4()),
                product_id=product.id,
                type="OUT",
                quantity=quantity,
                source="INVOICE",

                stock_before=stock_before,
                stock_after=stock_after,

                variant_sku=sku,
                variant_stock_after=variant_stock_after,

                created_by=current_user.id,
                created_by_name=current_user.name,
                created_at=datetime.now(IST)
            ))

        # ================= LEDGER =================
        payment_mode = invoice.payment_mode

        add_sale_ledger_entry(
            db,
            invoice.id,
            invoice.invoice_number,
            customer.id,
            original_total,
            current_user,
        )

        add_advance_sale_ledger_entry(
            db,
            invoice.id,
            invoice.invoice_number,
            customer.id,
            advance_used,
            current_user,
        )

        if paid_amount > 0:
            db.add(InvoicePayment(
                id=str(uuid.uuid4()),
                invoice_id=invoice.id,
                amount=paid_amount,
                payment_mode=payment_mode,
                created_by=current_user.id,
                created_at=datetime.now(IST),
            ))

        sync_customer_current_balance(db, customer)

        db.commit()

        return {
            "invoice_id": invoice.id,
            "invoice_number": invoice.invoice_number,
            "total": invoice.total,
            "original_total": invoice.original_total,
            "advance_used": invoice.advance_used,
            "paid_amount": invoice.paid_amount,
            "balance_amount": invoice.balance_amount,
        }

    except Exception as e:
        db.rollback()
        raise e
@api_router.api_route("/invoices/{invoice_id}/status", methods=["PUT", "PATCH"])
def update_invoice_status(
    invoice_id: str,
    payment_status: str,
    payment_mode: str = "cash",
    amount: float = 0,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):

    payment_status = payment_status.lower()
    payment_mode = payment_mode.lower()

    invoice = db.query(InvoiceModel).filter(
        InvoiceModel.id == invoice_id
    ).with_for_update().first()

    if not invoice:
        raise HTTPException(404, "Invoice not found")

    customer = db.query(CustomerModel).filter(
        CustomerModel.id == invoice.customer_id
    ).first()

    # 🚫 block only for payment actions
    if payment_status in ["paid", "partial"] and invoice.balance_amount <= 0:
        raise HTTPException(
            status_code=400,
            detail="Invoice already fully paid"
        )

    # ================= CANCEL =================
    if payment_status == "cancelled":

        if invoice.payment_status == "cancelled":
            raise HTTPException(400, "Already cancelled")

        if invoice.payment_status in ["paid", "partial"]:
            raise HTTPException(400, "Paid or partial invoice cannot be cancelled")

        if invoice.payment_status != "pending":
            raise HTTPException(400, "Only pending invoices can be cancelled")

        items = json.loads(invoice.items or "[]")

        for item in items:

            if item.get("is_service") == 1 or not item.get("product_id"):
                continue

            product = db.query(ProductModel).filter(
                ProductModel.id == item["product_id"]
            ).with_for_update().first()

            if not product:
                continue

            quantity = int(item.get("quantity", 0))
            sku = item.get("sku")
            stock_before = int(product.stock or 0)
            stock_after = stock_before
            variant_stock_after = None

            variants = list(product.variants or [])

            if variants and sku:
                matched_variant = False

                for v in variants:
                    if v.get("v_sku") == sku:
                        variant_before = int(v.get("stock", 0))
                        v["stock"] = variant_before + quantity
                        variant_stock_after = v["stock"]
                        matched_variant = True
                        break

                if matched_variant:
                    product.variants = variants
                    product.stock = calculate_total_stock(variants)
                    stock_after = int(product.stock or 0)

                    flag_modified(product, "variants")
                    flag_modified(product, "stock")
                else:
                    product.stock = stock_before + quantity
                    stock_after = int(product.stock or 0)
                    flag_modified(product, "stock")
            else:
                product.stock = stock_before + quantity
                stock_after = int(product.stock or 0)
                flag_modified(product, "stock")

            db.flush()

            db.add(
                InventoryTransaction(
                    id=str(uuid.uuid4()),
                    product_id=product.id,
                    type="IN",
                    quantity=quantity,
                    source="INVOICE_CANCEL",
                    reason=f"Cancel {invoice.invoice_number}",
                    stock_before=stock_before,
                    stock_after=stock_after,
                    variant_sku=sku,
                    variant_stock_after=variant_stock_after,
                    created_by=current_user.id,
                    created_by_name=current_user.name,
                    created_at=datetime.now(IST),
                )
            )

        invoice.balance_amount = 0
        invoice.paid_amount = 0
        invoice.payment_status = "cancelled"

        advance_entries = db.query(LedgerEntry).filter(
            LedgerEntry.reference_id == invoice.id,
            LedgerEntry.entry_type == "advance_used"
        ).all()
        for entry in advance_entries:
            db.delete(entry)

        sale_entries = db.query(LedgerEntry).filter(
            LedgerEntry.reference_id == invoice.id,
            LedgerEntry.entry_type == "sale"
        ).all()
        for entry in sale_entries:
            db.delete(entry)

        sync_customer_current_balance(db, customer)

        db.commit()

        return {
            "message": "Invoice cancelled successfully",
            "inventory": "reversed",
            "invoice_number": invoice.invoice_number
        }

    # ================= PARTIAL =================
    if payment_status == "partial":

        if amount <= 0:
            raise HTTPException(400, "Amount required")

        remaining = invoice.total - invoice.paid_amount

        if amount > remaining:
            raise HTTPException(400, "Too much amount")

        invoice.paid_amount += amount
        invoice.balance_amount = invoice.total - invoice.paid_amount
        invoice.payment_status = "partial" if invoice.balance_amount > 0 else "paid"
        invoice.payment_mode = payment_mode

        db.add(InvoicePayment(
            invoice_id=invoice.id,
            amount=amount,
            payment_mode=payment_mode,
            created_by=current_user.id
        ))

        sync_customer_current_balance(db, customer)

        db.commit()

        return {"message": "Partial payment updated"}

    # ================= PAID =================
    if payment_status == "paid":

        remaining = invoice.total - invoice.paid_amount

        if remaining <= 0:
            raise HTTPException(400, "Already paid")

        invoice.paid_amount = invoice.total
        invoice.balance_amount = 0
        invoice.payment_status = "paid"
        invoice.payment_mode = payment_mode

        db.add(InvoicePayment(
            invoice_id=invoice.id,
            amount=remaining,
            payment_mode=payment_mode,
            created_by=current_user.id
        ))

        sync_customer_current_balance(db, customer)

        db.commit()

        return {"message": "Invoice marked as paid"}

    raise HTTPException(400, "Invalid status")

@api_router.post("/invoices/{invoice_id}/complete-payment")
def complete_invoice_payment(
    invoice_id: str,
    payment_mode: str = "cash",
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user)
):

    invoice = db.query(InvoiceModel).filter(
        InvoiceModel.id == invoice_id
    ).with_for_update().first()

    if not invoice:
        raise HTTPException(404, "Invoice not found")

    if invoice.payment_status == "paid":
        raise HTTPException(400, "Invoice already fully paid")

    balance = invoice.total - invoice.paid_amount

    if balance <= 0:
        raise HTTPException(400, "No balance remaining")

    # UPDATE INVOICE
    invoice.paid_amount += balance
    invoice.balance_amount = 0
    invoice.payment_status = "paid"
    invoice.payment_mode = payment_mode

    # STORE PAYMENT HISTORY
    db.add(
        InvoicePayment(
            invoice_id=invoice.id,
            amount=balance,
            payment_mode=payment_mode,
            created_by=current_user.id
        )
    )

    # UPDATE CUSTOMER BALANCE
    customer = db.query(CustomerModel).filter(
        CustomerModel.id == invoice.customer_id
    ).first()

    sync_customer_current_balance(db, customer)

    db.commit()

    return {
        "message": "Payment completed",
        "invoice_number": invoice.invoice_number,
        "paid_amount": invoice.paid_amount,
        "balance": invoice.balance_amount,
        "status": invoice.payment_status
    }

@api_router.get("/invoices/{invoice_id}/payments")
def invoice_payments(
    invoice_id: str,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user)
):

    payments = db.query(InvoicePayment).filter(
        InvoicePayment.invoice_id == invoice_id
    ).order_by(InvoicePayment.created_at.asc()).all()

    return [
        {
            "id": p.id,
            "amount": p.amount,
            "payment_mode": p.payment_mode,
            "reference": p.reference,
            "created_at": p.created_at.isoformat()
        }
        for p in payments
    ]

# ================= NEW: ADD PAYMENT ENDPOINT =================
class AddPaymentRequest(BaseModel):
    amount: float
    payment_mode: str  # cash | upi | bank | cheque
    reference: Optional[str] = None  # UPI ID, Bank account, Cheque number, etc.

@api_router.post("/invoices/{invoice_id}/add-payment")
def add_payment_to_invoice(
    invoice_id: str,
    payment_data: AddPaymentRequest,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user)
):
    
    # Validate payment amount
    if payment_data.amount <= 0:
        raise HTTPException(400, "Payment amount must be greater than 0")
    
    if payment_data.payment_mode.lower() not in ["cash", "upi", "bank", "cheque"]:
        raise HTTPException(400, "Invalid payment mode")

    # Lock invoice for update
    invoice = (
        db.query(InvoiceModel)
        .filter(InvoiceModel.id == invoice_id)
        .with_for_update()
        .first()
    )

    if not invoice:
        raise HTTPException(404, "Invoice not found")

    if invoice.payment_status == "paid":
        raise HTTPException(400, "Invoice already fully paid")

    # Calculate remaining balance
    remaining_balance = invoice.total - invoice.paid_amount

    if payment_data.amount > remaining_balance:
        raise HTTPException(400, f"Payment amount exceeds balance of {remaining_balance}")

    # Update invoice
    invoice.paid_amount += payment_data.amount
    invoice.balance_amount = invoice.total - invoice.paid_amount

    # Auto-update payment status
    if invoice.balance_amount <= 0:
        invoice.payment_status = "paid"
        invoice.balance_amount = 0
    else:
        invoice.payment_status = "partial"

    # Store payment history
    db.add(
        InvoicePayment(
            id=str(uuid.uuid4()),
            invoice_id=invoice.id,
            amount=payment_data.amount,
            payment_mode=payment_data.payment_mode.lower(),
            reference=payment_data.reference,
            created_by=current_user.id,
            created_at=datetime.now(IST)
        )
    )

    # Update customer balance
    customer = db.query(CustomerModel).filter(
        CustomerModel.id == invoice.customer_id
    ).first()

    sync_customer_current_balance(db, customer)

    db.commit()

    return {
        "message": "Payment recorded successfully",
        "invoice_number": invoice.invoice_number,
        "payment_amount": payment_data.amount,
        "paid_amount": invoice.paid_amount,
        "balance_amount": invoice.balance_amount,
        "payment_status": invoice.payment_status
    }

# ================= DELETE PAYMENT ENDPOINT =================
@api_router.delete("/invoices/payment/{payment_id}")
def delete_payment(
    payment_id: str,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user)
):
    
    payment = db.query(InvoicePayment).filter(
        InvoicePayment.id == payment_id
    ).first()

    if not payment:
        raise HTTPException(404, "Payment not found")

    invoice_id = payment.invoice_id
    payment_amount = payment.amount

    # Get invoice and lock it
    invoice = (
        db.query(InvoiceModel)
        .filter(InvoiceModel.id == invoice_id)
        .with_for_update()
        .first()
    )

    if not invoice:
        raise HTTPException(404, "Invoice not found")

    # Reverse payment
    invoice.paid_amount -= payment_amount
    invoice.balance_amount = invoice.total - invoice.paid_amount

    # Update payment status
    if invoice.paid_amount <= 0:
        invoice.payment_status = "pending"
        invoice.paid_amount = 0
        invoice.balance_amount = invoice.total
    else:
        invoice.payment_status = "partial"

    # Update customer balance
    customer = db.query(CustomerModel).filter(
        CustomerModel.id == invoice.customer_id
    ).first()

    # Delete payment record
    db.delete(payment)

    sync_customer_current_balance(db, customer)

    db.commit()

    return {
        "message": "Payment deleted and reversed",
        "invoice_number": invoice.invoice_number,
        "paid_amount": invoice.paid_amount,
        "balance_amount": invoice.balance_amount,
        "payment_status": invoice.payment_status
    }
@api_router.get(
    "/customers/search",
    response_model=Optional[Customer]
)
def search_customer_by_phone(
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

    return Customer(
        id=customer.id,
        name=customer.name,
        email=customer.email,
        phone=customer.phone,
        address=customer.address,
        current_balance=calculate_customer_current_balance(db, customer),
        created_at=customer.created_at.isoformat()
    )


@api_router.get("/products/list")
def list_products(
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    products = (
    db.query(ProductModel)
    .filter(
        ProductModel.is_service == 0,
        ProductModel.is_active == 1   # âœ… ADD
    )
    .all()
)


    return [
        {
            "id": prod.id,
            "product_code": prod.product_code,
            "name": prod.name,
            "sku": prod.sku,
            "stock": prod.stock,
            "min_stock": prod.min_stock,
            "category_name": prod.category.name if prod.category else "Unknown",
        }
        for prod in products
    ]

@api_router.get(
    "/products/search",
    dependencies=[Depends(RateLimiter(times=100, seconds=60))]
)
def search_products(
    q: str,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    search_term = f"%{q.lower().strip()}%"
    products = (
    db.query(ProductModel)
    .filter(
        ProductModel.is_active == 1,   # âœ… ADD
        or_(
            func.lower(ProductModel.name).like(search_term),
            func.lower(ProductModel.sku).like(search_term),
            func.lower(ProductModel.product_code).like(search_term)
        )
    )
    .limit(20)
    .all()
)

    
    return [
        {
            "id": p.id,
            "name": p.name,
            "sku": p.sku,
            "product_code": p.product_code,
            "selling_price": p.selling_price,
            "min_selling_price": p.min_selling_price,
            "variants": p.variants or [],
            "images": safe_images(p.images),
            "stock": p.stock,
            "is_service": p.is_service
        }
        for p in products
    ]

    
# Re-defining parse_invoice_items to ensure it's available if called before the get_invoices endpoint.
# This is a workaround for potential ordering issues if not explicitly handled.

        
@api_router.get(
    "/dashboard",
    dependencies=[Depends(RateLimiter(times=30, seconds=60))]
)
def get_dashboard_stats(
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
        end = y.replace(hour=23, minute=59, second=59, microsecond=999999)

    elif filter == "last_10_days":
        start = now - timedelta(days=10)
        end = now

    elif filter == "last_30_days":
        start = now - timedelta(days=30)
        end = now

    elif filter == "month":
        if not year or not month:
            raise HTTPException(status_code=400, detail="Year and month required")

        start = datetime(year, month, 1, tzinfo=IST)
        end = (
            datetime(year + 1, 1, 1, tzinfo=IST)
            if month == 12
            else datetime(year, month + 1, 1, tzinfo=IST)
        )

    else:
        raise HTTPException(status_code=400, detail="Invalid filter")

    # ---------- SALES ----------
    invoice_q = db.query(InvoiceModel).filter(
        InvoiceModel.invoice_type == "FINAL",
        InvoiceModel.payment_status == "paid",
        InvoiceModel.created_at >= start,
        InvoiceModel.created_at < end
    

    )

    total_sales = invoice_q.with_entities(
        func.coalesce(func.sum(InvoiceModel.total), 0)
    ).scalar()

    total_orders = invoice_q.count()

    total_customers = invoice_q.with_entities(
        func.count(func.distinct(InvoiceModel.customer_id))
    ).scalar()

    # ---------- LOW STOCK (PARTS ONLY) ----------
    low_stock = db.query(ProductModel).filter(
        ProductModel.is_service == 0,             # âœ… EXCLUDE SERVICES
        ProductModel.stock <= ProductModel.min_stock
    ).count()

    return {
        "total_sales": float(total_sales),
        "total_orders": total_orders,
        "total_customers": total_customers,
        "low_stock_items": low_stock
    }


@api_router.get("/dashboard/today")
def dashboard_today(
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    now = datetime.now(IST)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + timedelta(days=1)

    invoices_today = db.query(InvoiceModel).filter(
    InvoiceModel.invoice_type == "FINAL",
    InvoiceModel.created_at >= start,
    InvoiceModel.created_at < end
).count()

    total_sales_today = db.query(
        func.coalesce(func.sum(InvoiceModel.total), 0)
    ).filter(
        InvoiceModel.invoice_type == "FINAL",
        InvoiceModel.created_at >= start,
        InvoiceModel.created_at < end
    ).scalar()

    cash_sales_today = db.query(
        func.coalesce(func.sum(InvoiceModel.total), 0)
    ).filter(
        InvoiceModel.invoice_type == "FINAL",
        InvoiceModel.created_at >= start,
        InvoiceModel.created_at < end,
        InvoiceModel.payment_mode == "cash"
    ).scalar()

    online_sales_today = db.query(
        func.coalesce(func.sum(InvoiceModel.total), 0)
    ).filter(
        InvoiceModel.invoice_type == "FINAL",
        InvoiceModel.created_at >= start,
        InvoiceModel.created_at < end,
        InvoiceModel.payment_mode.in_(["upi", "bank", "cheque"])
    ).scalar()


    items_sold_today = db.query(
        func.coalesce(func.sum(InventoryTransaction.quantity), 0)
    ).filter(
        InventoryTransaction.type == "OUT",
        InventoryTransaction.created_at >= start,
        InventoryTransaction.created_at < end
    ).scalar()

    new_customers = db.query(CustomerModel).filter(
        CustomerModel.created_at >= start,
        CustomerModel.created_at < end
    ).count()

    inventory_out = db.query(InventoryTransaction).filter(
        InventoryTransaction.type == "OUT",
        InventoryTransaction.created_at >= start,
        InventoryTransaction.created_at < end
    ).count()

    return {
        "invoices_today": invoices_today,
        "total_sales_today": float(total_sales_today or 0),
        "cash_sales_today": float(cash_sales_today or 0),
        "online_sales_today": float(online_sales_today or 0),
        "items_sold_today": int(items_sold_today or 0),
        "inventory_out_today": inventory_out,
        "new_customers_today": new_customers
    }
@api_router.get("/dashboard/low-stock")
def low_stock_products(
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    products = (
        db.query(ProductModel)
        .filter(
            ProductModel.is_service == 0,         # âœ… EXCLUDE SERVICES
            ProductModel.stock <= ProductModel.min_stock
        )
        .order_by(ProductModel.stock.asc())
        .limit(10)
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
            "text": f"Invoice {inv.invoice_number} â€“ â‚¹{inv.total}",
            "date": inv.created_at.isoformat()
        })

    for txn, prod in inventory:
        activity.append({
        "type": "inventory",
        "text": f"{txn.type} â€“ {prod.name} ({txn.quantity}) by {txn.created_by_name}",
        "date": txn.created_at.isoformat()
    })


    return sorted(activity, key=lambda x: x["date"], reverse=True)[:limit]

@api_router.get("/dashboard/hourly-sales")
def hourly_sales_today(
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # ðŸ”¹ Use local server time (IMPORTANT for MySQL)
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
            "label": f"{hour:02d}:00â€“{hour+1:02d}:00",
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
    elements.append(Paragraph(f"Total: â‚¹{float(invoice.original_total or invoice.total or 0)}", styles["Heading2"]))

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

    # Date filters
    if start_date:
        query = query.filter(func.date(InvoiceModel.created_at) >= start_date)

    if end_date:
        query = query.filter(func.date(InvoiceModel.created_at) <= end_date)

    total = query.count()

    invoices = (
        query.order_by(InvoiceModel.created_at.desc())
        .offset((page - 1) * limit)
        .limit(limit)
        .all()
    )

    data = []

    for inv in invoices:

        gst = float(inv.gst_amount or 0)

        # Split GST into CGST + SGST
        cgst = gst / 2
        sgst = gst / 2

        data.append({
            "id": inv.id,
            "invoice": inv.invoice_number,
            "customer": inv.customer_name,
            "taxable_amount": float(inv.subtotal or 0),
            "gst_amount": gst,
            "cgst": cgst,
            "sgst": sgst,
            "igst": 0,  # IGST not used in this system
            "total": float(inv.total or 0),
            "date": inv.created_at.date().isoformat() if inv.created_at else None
        })

    return {
        "data": data,
        "pagination": {
            "page": page,
            "limit": limit,
            "total": total,
            "total_pages": math.ceil(total / limit) if total else 1
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
    advance_payload = AdvancePaymentRequest(
        customer_id=data.customer_id,
        amount=data.amount,
        payment_mode=data.payment_mode,
        payment_date=data.payment_date,
        cheque_number=data.cheque_number,
        cheque_date=data.cheque_date,
        bank_name=data.bank_name,
    )
    return add_advance_payment(advance_payload, db, current_user)


@api_router.post("/payments/advance")
def add_advance_payment(
    data: AdvancePaymentRequest,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user)
):
    if data.amount <= 0:
        raise HTTPException(400, "Advance amount must be greater than 0")

    payment_mode = normalize_draft_payment_mode(data.payment_mode)
    customer = db.query(CustomerModel).filter(CustomerModel.id == data.customer_id).first()
    if not customer:
        raise HTTPException(404, "Customer not found")

    payment_date = data.payment_date or datetime.now(IST).date()
    received_amount = round(float(data.amount), 2)
    pending_invoices = (
        db.query(InvoiceModel)
        .filter(
            InvoiceModel.customer_id == customer.id,
            InvoiceModel.invoice_type == "FINAL",
            InvoiceModel.payment_status.in_(["pending", "partial"]),
            InvoiceModel.balance_amount > 0,
        )
        .order_by(InvoiceModel.created_at.asc())
        .all()
    )

    remaining_to_allocate = received_amount
    cleared_pending = 0.0

    for invoice in pending_invoices:
        if remaining_to_allocate <= 0:
            break

        invoice_balance = round(float(invoice.balance_amount or 0), 2)
        if invoice_balance <= 0:
            continue

        allocation = round(min(invoice_balance, remaining_to_allocate), 2)
        if allocation <= 0:
            continue

        invoice.paid_amount = round(float(invoice.paid_amount or 0) + allocation, 2)
        invoice.balance_amount = round(invoice_balance - allocation, 2)
        invoice.payment_status = "paid" if invoice.balance_amount <= 0 else "partial"
        invoice.payment_mode = payment_mode

        db.add(InvoicePayment(
            id=str(uuid.uuid4()),
            invoice_id=invoice.id,
            amount=allocation,
            payment_mode=payment_mode,
            created_by=current_user.id,
            created_at=datetime.now(IST),
        ))

        cleared_pending = round(cleared_pending + allocation, 2)
        remaining_to_allocate = round(remaining_to_allocate - allocation, 2)

    remaining_advance = round(max(received_amount - cleared_pending, 0), 2)

    if payment_mode == "cheque" and data.cheque_number:
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

    if remaining_advance > 0:
        db.add(
            LedgerEntry(
                id=str(uuid.uuid4()),
                entry_type="advance_in",
                reference_id=data.customer_id,
                customer_id=data.customer_id,
                description=f"Advance received from {customer.name}",
                debit=0,
                credit=remaining_advance,
                payment_mode=payment_mode,
                entry_date=payment_date,
                created_by=current_user.id,
                created_by_name=current_user.name,
                created_at=datetime.now(IST)
            )
        )

    sync_customer_current_balance(db, customer)

    db.commit()

    return {
        "message": "Advance received successfully",
        "customer_id": customer.id,
        "customer_name": customer.name,
        "amount": received_amount,
        "cleared_pending": cleared_pending,
        "advance_added": remaining_advance,
        "current_balance": float(customer.current_balance or 0)
    }

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

    # ðŸ”¹ Opening balance before this page
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
