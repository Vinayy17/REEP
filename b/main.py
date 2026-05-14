import ast
from math import ceil
from xml.sax.saxutils import escape
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
from datetime import datetime, time

from sqlalchemy import or_, func
from pydantic import Field
from fastapi.responses import StreamingResponse
import io
from io import StringIO
import csv
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.enums import TA_LEFT, TA_RIGHT, TA_CENTER
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
from fastapi.responses import JSONResponse
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
import redis.asyncio as redis


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

INVOICE_COMPANY_NAME = "R RIDE"
INVOICE_COMPANY_TAGLINE = "BIKE GARAGE & STUDIO"
INVOICE_COMPANY_ADDRESS = (
    "Mahaveer Building, gala no 09, near payal talkies, mandai road, "
    "Dhamankar naka, Bhiwandi, 421305"
)
INVOICE_COMPANY_CONTACT = "Phone: +91 8793678780 | Email: rridemods@gmail.com"
INVOICE_TERMS = [
    "1. Payment is due within 30 days of invoice date.",
    "2. Late payments may incur additional charges.",
    "3. Goods once sold will not be taken back or exchanged.",
    "4. All disputes subject to local jurisdiction only.",
]



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

class EmployeeModel(Base):
    __tablename__ = "employees"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False, index=True)
    role = Column(String(100), nullable=True)
    salary = Column(Float, nullable=False, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now(IST))

class SalaryPaymentModel(Base):
    __tablename__ = "salary_payments"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    employee_id = Column(String(36), ForeignKey("employees.id"), nullable=False, index=True)
    total_salary = Column(Float, nullable=False, default=0)
    paid_amount = Column(Float, nullable=False, default=0)
    pending_amount = Column(Float, nullable=False, default=0)
    payment_date = Column(Date, default=lambda: datetime.now(IST).date(), index=True)
    payment_mode = Column(String(50), nullable=True)
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(IST))

    employee = relationship("EmployeeModel")

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

class SupplierInvoiceModel(Base):
    __tablename__ = "supplier_invoices"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    supplier_id = Column(String(36), ForeignKey("suppliers.id"), nullable=False, index=True)
    total_amount = Column(Float, nullable=False, default=0)
    paid_amount = Column(Float, nullable=False, default=0)
    pending_amount = Column(Float, nullable=False, default=0)
    invoice_date = Column(Date, default=lambda: datetime.now(IST).date(), index=True)
    status = Column(String(50), default="pending", index=True)
    created_at = Column(DateTime, default=lambda: datetime.now(IST))

    supplier = relationship("SupplierModel")

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
    try:
        Base.metadata.create_all(
            bind=engine,
            tables=[
                EmployeeModel.__table__,
                SalaryPaymentModel.__table__,
                SupplierInvoiceModel.__table__,
            ],
        )

        db = SessionLocal()
        try:
            ensure_expense_category(db, "Power Bill", "Electricity and power utility bills")
            db.commit()
        finally:
            db.close()
    except Exception as exc:
        logger.error("Startup accounting setup failed: %s", exc)

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
    opening_balance: Optional[float] = 0
    current_balance: Optional[float] = 0
    display_balance: Optional[float] = 0
    pending_balance: Optional[float] = 0
    advance_balance: Optional[float] = 0
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
    use_advance: bool = False

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

class EmployeeCreate(BaseModel):
    name: str
    role: Optional[str] = None
    salary: float

class EmployeeUpdate(BaseModel):
    name: str
    role: Optional[str] = None
    salary: float

class SalaryRecordCreate(BaseModel):
    payment_date: Optional[date] = None
    notes: Optional[str] = None

class SalaryPaymentRequest(BaseModel):
    amount: float
    payment_mode: str
    payment_date: Optional[date] = None
    notes: Optional[str] = None

class PowerBillRequest(BaseModel):
    title: Optional[str] = "Power Bill"
    amount: float
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

class CustomerAdvanceRequest(BaseModel):
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

class SupplierInvoiceCreate(BaseModel):
    supplier_id: str
    total_amount: float
    invoice_date: Optional[date] = None

class SupplierInvoiceForSupplierCreate(BaseModel):
    total_amount: float
    invoice_date: Optional[date] = None

class SupplierInvoicePaymentRequest(BaseModel):
    amount: float
    payment_mode: str
    payment_date: Optional[date] = None
    notes: Optional[str] = None


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

def calculate_draft_payment_summary(
    total: float,
    payment_status: str,
    paid_amount: Optional[float] = 0
):
    if payment_status == "paid":
        return round(total, 2), 0.0

    if payment_status == "partial":
        partial_paid = round(float(paid_amount or 0), 2)

        if partial_paid < 0:
            raise HTTPException(400, "Partial payment cannot be negative")

        if partial_paid > total:
            raise HTTPException(400, "Partial exceeds total")

        return partial_paid, round(total - partial_paid, 2)

    return 0.0, round(total, 2)

INITIAL_INVOICE_PAYMENT_REFERENCE = "__initial_invoice_payment__"
CUSTOMER_LEDGER_VISIBLE_TYPES = {
    "advance_in",
    "advance_used",
    "payment_in",
}
GENERAL_LEDGER_EXCLUDED_TYPES = {"sale_due", "supplier_bill"}
GENERAL_LEDGER_TRACKING_TYPES = {"advance_used"}
SUPPLIER_CASH_OUT_ENTRY_TYPES = {"payment_out", "supplier_payment"}
LEDGER_MONEY_IN_ENTRY_TYPES = {"sale_payment", "payment_in", "advance_in"}


def round_currency(value: Optional[float]) -> float:
    return round(float(value or 0), 2)


def get_supplier_cash_out_value(entry: Optional[LedgerEntry]) -> float:
    if not entry:
        return 0.0

    debit_amount = round_currency(getattr(entry, "debit", 0))
    credit_amount = round_currency(getattr(entry, "credit", 0))

    return debit_amount if debit_amount > 0 else credit_amount


def get_normalized_ledger_debit_credit(entry: Optional[LedgerEntry]) -> tuple[float, float]:
    if not entry:
        return 0.0, 0.0

    if entry.entry_type in GENERAL_LEDGER_TRACKING_TYPES:
        return 0.0, 0.0

    if entry.entry_type in SUPPLIER_CASH_OUT_ENTRY_TYPES:
        return get_supplier_cash_out_value(entry), 0.0

    return round_currency(entry.debit), round_currency(entry.credit)


def build_supplier_cash_out_expression():
    return case(
        (LedgerEntry.debit > 0, LedgerEntry.debit),
        else_=LedgerEntry.credit,
    )


def build_ledger_balance_expression():
    supplier_cash_out = build_supplier_cash_out_expression()

    return case(
        (
            LedgerEntry.entry_type.in_(list(GENERAL_LEDGER_TRACKING_TYPES)),
            0,
        ),
        (
            LedgerEntry.entry_type.in_(list(SUPPLIER_CASH_OUT_ENTRY_TYPES)),
            -supplier_cash_out,
        ),
        else_=(LedgerEntry.credit - LedgerEntry.debit),
    )


def money_in_range_filters(
    date_column,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
):
    filters = []
    if start_date:
        filters.append(date_column >= start_date)
    if end_date:
        filters.append(date_column <= end_date)
    return filters


def ensure_expense_category(
    db: Session,
    name: str,
    description: Optional[str] = None,
) -> ExpenseCategoryModel:
    category = (
        db.query(ExpenseCategoryModel)
        .filter(func.lower(ExpenseCategoryModel.name) == name.lower())
        .first()
    )

    if category:
        return category

    category = ExpenseCategoryModel(
        id=str(uuid.uuid4()),
        name=name,
        description=description,
        created_at=datetime.now(IST),
    )
    db.add(category)
    db.flush()
    return category


def create_expense_with_ledger(
    db: Session,
    *,
    title: str,
    amount: float,
    payment_mode: str,
    category_id: Optional[str],
    description: Optional[str],
    expense_date: date,
    current_user: UserModel,
) -> ExpenseModel:
    amount = round_currency(amount)
    if amount <= 0:
        raise HTTPException(400, "Expense amount must be greater than zero")

    expense = ExpenseModel(
        id=str(uuid.uuid4()),
        title=title,
        amount=amount,
        category_id=category_id,
        payment_mode=payment_mode,
        description=description,
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
            reference_id=expense.id,
            description=title,
            debit=amount,
            credit=0,
            payment_mode=payment_mode,
            entry_date=expense_date,
            created_by=current_user.id,
            created_by_name=current_user.name,
            created_at=datetime.now(IST),
        )
    )
    return expense


def get_salary_month_start(target_date: Optional[date] = None) -> date:
    reference_date = target_date or datetime.now(IST).date()
    return reference_date.replace(day=1)


def get_next_salary_month_start(target_date: Optional[date] = None) -> date:
    month_start = get_salary_month_start(target_date)
    if month_start.month == 12:
        return month_start.replace(year=month_start.year + 1, month=1)
    return month_start.replace(month=month_start.month + 1)


def get_salary_month_label(target_date: Optional[date] = None) -> str:
    return get_salary_month_start(target_date).strftime("%b %Y")


def get_employee_month_salary_record(
    db: Session,
    employee_id: str,
    target_date: Optional[date] = None,
):
    month_start = get_salary_month_start(target_date)
    next_month_start = get_next_salary_month_start(target_date)

    return (
        db.query(SalaryPaymentModel)
        .filter(
            SalaryPaymentModel.employee_id == employee_id,
            SalaryPaymentModel.payment_date >= month_start,
            SalaryPaymentModel.payment_date < next_month_start,
        )
        .order_by(SalaryPaymentModel.payment_date.desc(), SalaryPaymentModel.created_at.desc())
        .first()
    )


def serialize_salary_record(record: Optional[SalaryPaymentModel]) -> Optional[dict]:
    if not record:
        return None

    return {
        "id": record.id,
        "employee_id": record.employee_id,
        "total_salary": round_currency(record.total_salary),
        "paid_amount": round_currency(record.paid_amount),
        "pending_amount": round_currency(record.pending_amount),
        "payment_date": record.payment_date.isoformat() if record.payment_date else None,
        "payment_mode": record.payment_mode,
        "notes": record.notes,
        "created_at": record.created_at.isoformat() if record.created_at else None,
    }


def serialize_employee(db: Session, employee: EmployeeModel) -> dict:
    current_month_record = get_employee_month_salary_record(db, employee.id)
    current_month_paid = round_currency(current_month_record.paid_amount) if current_month_record else 0.0
    current_month_pending = (
        round_currency(current_month_record.pending_amount)
        if current_month_record
        else round_currency(employee.salary)
    )
    open_record = current_month_record if current_month_record and round_currency(current_month_record.pending_amount) > 0 else None

    return {
        "id": employee.id,
        "name": employee.name,
        "role": employee.role,
        "salary": round_currency(employee.salary),
        "paid_amount": current_month_paid,
        "pending_amount": current_month_pending,
        "salary_period": get_salary_month_label(),
        "open_salary_record": serialize_salary_record(open_record),
        "created_at": employee.created_at.isoformat() if employee.created_at else None,
    }


def serialize_supplier_invoice(invoice: SupplierInvoiceModel, supplier: Optional[SupplierModel] = None) -> dict:
    resolved_supplier = supplier or invoice.supplier
    total_amount = round_currency(invoice.total_amount)
    paid_amount = round_currency(invoice.paid_amount)
    pending_amount = round_currency(invoice.pending_amount)

    if pending_amount <= 0 and total_amount > 0:
        resolved_status = "paid"
    elif paid_amount > 0:
        resolved_status = "partial"
    else:
        resolved_status = "pending"

    invoice_date = invoice.invoice_date
    if not invoice_date and invoice.created_at:
        invoice_date = (
            invoice.created_at.astimezone(IST).date()
            if getattr(invoice.created_at, "tzinfo", None)
            else invoice.created_at.date()
        )

    invoice_ref = str(invoice.id or "").split("-")[0].upper()
    bill_number = (
        f"SB-{invoice_date.strftime('%y%m')}-{invoice_ref}"
        if invoice_date
        else f"SB-{invoice_ref}"
    )

    return {
        "id": invoice.id,
        "bill_number": bill_number,
        "supplier_id": invoice.supplier_id,
        "supplier_name": resolved_supplier.name if resolved_supplier else None,
        "total_amount": total_amount,
        "paid_amount": paid_amount,
        "pending_amount": pending_amount,
        "invoice_date": invoice_date.isoformat() if invoice_date else None,
        "status": resolved_status,
        "created_at": invoice.created_at.isoformat() if invoice.created_at else None,
    }


def update_supplier_invoice_status(invoice: SupplierInvoiceModel):
    invoice.paid_amount = round_currency(invoice.paid_amount)
    invoice.pending_amount = round_currency(max(float(invoice.total_amount or 0) - float(invoice.paid_amount or 0), 0))

    if invoice.pending_amount <= 0:
        invoice.status = "paid"
    elif invoice.paid_amount > 0:
        invoice.status = "partial"
    else:
        invoice.status = "pending"


def get_supplier_invoice_summary(db: Session, supplier_id: str) -> dict:
    rows = (
        db.query(SupplierInvoiceModel)
        .filter(SupplierInvoiceModel.supplier_id == supplier_id)
        .all()
    )

    return {
        "total_bill": round_currency(sum(float(row.total_amount or 0) for row in rows)),
        "paid_amount": round_currency(sum(float(row.paid_amount or 0) for row in rows)),
        "pending_amount": round_currency(sum(float(row.pending_amount or 0) for row in rows)),
    }


def build_supplier_ledger_snapshot(
    db: Session,
    supplier: SupplierModel,
    *,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
):
    invoice_query = db.query(SupplierInvoiceModel).filter(
        SupplierInvoiceModel.supplier_id == supplier.id
    )
    ledger_query = db.query(LedgerEntry).filter(
        LedgerEntry.supplier_id == supplier.id
    )

    if start_date:
        invoice_query = invoice_query.filter(SupplierInvoiceModel.invoice_date >= start_date)
        ledger_query = ledger_query.filter(LedgerEntry.entry_date >= start_date)

    if end_date:
        invoice_query = invoice_query.filter(SupplierInvoiceModel.invoice_date <= end_date)
        ledger_query = ledger_query.filter(LedgerEntry.entry_date <= end_date)

    invoices = invoice_query.order_by(
        SupplierInvoiceModel.invoice_date.asc(),
        SupplierInvoiceModel.created_at.asc(),
    ).all()

    ledger_entries = ledger_query.order_by(
        LedgerEntry.entry_date.asc(),
        LedgerEntry.created_at.asc(),
    ).all()

    combined_entries = []
    opening_balance = round_currency(supplier.opening_balance)
    opening_date = (
        supplier.created_at.astimezone(IST).date()
        if getattr(supplier.created_at, "tzinfo", None)
        else (supplier.created_at.date() if supplier.created_at else None)
    )

    if opening_balance:
        combined_entries.append(
            {
                "id": f"opening-{supplier.id}",
                "date": opening_date.isoformat() if opening_date else None,
                "description": f"Opening balance for {supplier.name}",
                "debit": opening_balance if opening_balance > 0 else 0.0,
                "credit": abs(opening_balance) if opening_balance < 0 else 0.0,
                "amount": abs(opening_balance),
                "type": "opening_balance",
                "mode": "opening",
                "created_by": None,
                "created_at": supplier.created_at.isoformat() if supplier.created_at else None,
                "_sort_at": supplier.created_at or datetime.now(IST),
                "_affects_balance": True,
            }
        )

    for invoice in invoices:
        serialized_invoice = serialize_supplier_invoice(invoice, supplier)
        combined_entries.append(
            {
                "id": f"bill-{invoice.id}",
                "date": serialized_invoice["invoice_date"],
                "description": f"Supplier bill {serialized_invoice['bill_number']}",
                "debit": serialized_invoice["total_amount"],
                "credit": 0.0,
                "amount": serialized_invoice["total_amount"],
                "type": "supplier_bill",
                "mode": "ledger",
                "created_by": None,
                "created_at": serialized_invoice["created_at"],
                "bill_number": serialized_invoice["bill_number"],
                "invoice_id": invoice.id,
                "status": serialized_invoice["status"],
                "_sort_at": invoice.created_at or datetime.now(IST),
                "_affects_balance": True,
            }
        )

    for entry in ledger_entries:
        if entry.entry_type == "supplier_bill":
            continue

        raw_debit = round_currency(entry.debit)
        raw_credit = round_currency(entry.credit)

        if entry.entry_type in {"supplier_payment", "payment_out"}:
            debit = 0.0
            credit = round_currency(max(raw_debit, raw_credit))
        else:
            debit = raw_debit
            credit = raw_credit

        combined_entries.append(
            {
                "id": entry.id,
                "date": entry.entry_date.isoformat() if entry.entry_date else None,
                "description": entry.description,
                "debit": debit,
                "credit": credit,
                "amount": round_currency(max(abs(debit), abs(credit))),
                "type": entry.entry_type,
                "mode": entry.payment_mode,
                "created_by": entry.created_by_name,
                "created_at": entry.created_at.isoformat() if entry.created_at else None,
                "reference_id": entry.reference_id,
                "_sort_at": entry.created_at or datetime.now(IST),
                "_affects_balance": True,
            }
        )

    combined_entries.sort(
        key=lambda item: (
            item["date"] or "",
            item["_sort_at"],
            item["id"],
        )
    )

    balance = 0.0
    total_debit = 0.0
    total_credit = 0.0
    ledger_rows = []

    for entry in combined_entries:
        total_debit += entry["debit"]
        total_credit += entry["credit"]

        if entry["_affects_balance"]:
            balance = round_currency(balance + entry["debit"] - entry["credit"])

        ledger_rows.append(
            {
                "id": entry["id"],
                "date": entry["date"],
                "description": entry["description"],
                "debit": round_currency(entry["debit"]),
                "credit": round_currency(entry["credit"]),
                "amount": round_currency(entry["amount"]),
                "balance": balance,
                "type": entry["type"],
                "mode": entry["mode"],
                "created_by": entry["created_by"],
                "created_at": entry["created_at"],
                "bill_number": entry.get("bill_number"),
                "invoice_id": entry.get("invoice_id"),
                "status": entry.get("status"),
                "reference_id": entry.get("reference_id"),
            }
        )

    serialized_invoices = sorted(
        [serialize_supplier_invoice(invoice, supplier) for invoice in invoices],
        key=lambda row: (
            row["invoice_date"] or "",
            row["created_at"] or "",
        ),
        reverse=True,
    )

    return {
        "ledger_rows": ledger_rows,
        "invoices": serialized_invoices,
        "total_debit": round_currency(total_debit),
        "total_credit": round_currency(total_credit),
        "computed_balance": round_currency(balance),
    }


def normalize_customer_balance_value(value: Optional[float]) -> float:
    normalized = round_currency(value)
    return 0.0 if abs(normalized) < 0.01 else normalized


def get_customer_pending_invoice_amount(
    db: Session,
    customer_id: Optional[str],
) -> float:
    if not customer_id:
        return 0.0

    amount = (
        db.query(func.coalesce(func.sum(InvoiceModel.balance_amount), 0))
        .filter(
            InvoiceModel.customer_id == customer_id,
            InvoiceModel.invoice_type == "FINAL",
            InvoiceModel.payment_status.in_(["pending", "partial"]),
        )
        .scalar()
    )
    return round_currency(amount)


def get_customer_pending_invoice_map(
    db: Session,
    customer_ids: List[str],
) -> dict:
    if not customer_ids:
        return {}

    rows = (
        db.query(
            InvoiceModel.customer_id,
            func.coalesce(func.sum(InvoiceModel.balance_amount), 0),
        )
        .filter(
            InvoiceModel.customer_id.in_(customer_ids),
            InvoiceModel.invoice_type == "FINAL",
            InvoiceModel.payment_status.in_(["pending", "partial"]),
        )
        .group_by(InvoiceModel.customer_id)
        .all()
    )

    return {
        customer_id: round_currency(amount)
        for customer_id, amount in rows
    }


def get_customer_balance_snapshot(
    db: Session,
    customer: Optional[CustomerModel],
    pending_invoice_amount: Optional[float] = None,
) -> dict:
    if not customer:
        return {
            "current_balance": 0.0,
            "display_balance": 0.0,
            "pending_balance": 0.0,
            "advance_balance": 0.0,
            "stored_current_balance": 0.0,
            "pending_invoice_amount": 0.0,
            "can_add_advance": True,
        }

    stored_balance = normalize_customer_balance_value(customer.current_balance)
    pending_amount = (
        round_currency(pending_invoice_amount)
        if pending_invoice_amount is not None
        else get_customer_pending_invoice_amount(db, customer.id)
    )
    pending_amount = max(pending_amount, 0.0)

    effective_balance = (
        normalize_customer_balance_value(pending_amount)
        if pending_amount > 0
        else stored_balance
    )
    display_balance = normalize_customer_balance_value(
        get_customer_display_balance(effective_balance)
    )
    advance_balance = 0.0 if pending_amount > 0 else round_currency(max(-stored_balance, 0))

    return {
        "current_balance": effective_balance,
        "display_balance": display_balance,
        "pending_balance": round_currency(pending_amount),
        "advance_balance": advance_balance,
        "stored_current_balance": stored_balance,
        "pending_invoice_amount": round_currency(pending_amount),
        "can_add_advance": pending_amount <= 0,
    }


def get_customer_pending_amount(customer: Optional[CustomerModel]) -> float:
    if not customer:
        return 0.0
    balance = normalize_customer_balance_value(customer.current_balance)
    return round_currency(max(balance, 0))


def get_customer_advance_amount(customer: Optional[CustomerModel]) -> float:
    if not customer:
        return 0.0
    balance = normalize_customer_balance_value(customer.current_balance)
    return round_currency(max(-balance, 0))


def get_customer_display_balance(balance: Optional[float]) -> float:
    return round_currency(-normalize_customer_balance_value(balance))


CUSTOMER_BALANCE_ENTRY_TYPES = {
    "advance_in",
    "advance_used",
    "sale_due",
    "sale_payment",
    "payment_in",
}


def get_customer_effective_balance(db: Session, customer: Optional[CustomerModel]) -> float:
    if not customer:
        return 0.0
    return get_customer_balance_snapshot(db, customer)["current_balance"]


def update_customer_balance(
    customer: Optional[CustomerModel],
    delta: Optional[float] = 0,
) -> float:
    if not customer:
        return 0.0

    current_balance = normalize_customer_balance_value(customer.current_balance)
    delta_amount = round_currency(delta)

    customer.current_balance = normalize_customer_balance_value(
        current_balance + delta_amount
    )
    return customer.current_balance


def get_invoice_balance_adjustment(
    *,
    balance_amount: Optional[float] = 0,
    advance_used: Optional[float] = 0,
    cash_paid_amount: Optional[float] = 0,
) -> float:
    return round_currency(
        max(float(balance_amount or 0), 0)
        + max(float(advance_used or 0), 0)
        + max(float(cash_paid_amount or 0), 0)
    )


def delete_invoice_balance_ledger_entries(
    db: Session,
    invoice_id: Optional[str],
    customer_id: Optional[str] = None,
    entry_types: Optional[set] = None,
):
    if not invoice_id:
        return

    query = db.query(LedgerEntry).filter(
        LedgerEntry.reference_id == invoice_id,
    )

    if entry_types:
        query = query.filter(LedgerEntry.entry_type.in_(list(entry_types)))

    if customer_id:
        query = query.filter(LedgerEntry.customer_id == customer_id)

    for entry in query.all():
        db.delete(entry)


def delete_invoice_payment_records(db: Session, invoice_id: Optional[str]):
    if not invoice_id:
        return

    payments = db.query(InvoicePayment).filter(
        InvoicePayment.invoice_id == invoice_id
    ).all()

    for payment in payments:
        db.delete(payment)


def serialize_customer_overview(
    db: Session,
    customer: CustomerModel,
    *,
    total_invoices: Optional[int] = 0,
    total_bill: Optional[float] = 0,
    pending_invoice_amount: Optional[float] = None,
):
    balance_snapshot = get_customer_balance_snapshot(
        db,
        customer,
        pending_invoice_amount=pending_invoice_amount,
    )

    return {
        "id": customer.id,
        "name": customer.name,
        "email": customer.email,
        "phone": customer.phone,
        "address": customer.address,
        "opening_balance": round_currency(customer.opening_balance),
        "current_balance": balance_snapshot["current_balance"],
        "display_balance": balance_snapshot["display_balance"],
        "pending_balance": balance_snapshot["pending_balance"],
        "advance_balance": balance_snapshot["advance_balance"],
        "pending_invoice_amount": balance_snapshot["pending_invoice_amount"],
        "stored_current_balance": balance_snapshot["stored_current_balance"],
        "can_add_advance": balance_snapshot["can_add_advance"],
        "created_at": customer.created_at.isoformat(),
        "total_invoices": int(total_invoices or 0),
        "total_bill": round_currency(total_bill),
    }


def get_invoice_advance_used_amount(db: Session, invoice_id: Optional[str]) -> float:
    if not invoice_id:
        return 0.0

    amount = (
        db.query(func.coalesce(func.sum(LedgerEntry.debit), 0))
        .filter(
            LedgerEntry.reference_id == invoice_id,
            LedgerEntry.entry_type == "advance_used",
        )
        .scalar()
    )
    return round_currency(amount)


def get_invoice_payment_snapshot(
    total: Optional[float],
    paid_amount: Optional[float] = 0,
    advance_used: Optional[float] = 0,
    current_status: Optional[str] = None,
):
    total_amount = round_currency(total)
    cash_paid_amount = round_currency(max(float(paid_amount or 0), 0))
    advance_adjusted = round_currency(max(float(advance_used or 0), 0))
    payable_after_advance = round_currency(max(total_amount - advance_adjusted, 0))
    balance_amount = round_currency(max(payable_after_advance - cash_paid_amount, 0))

    if (current_status or "").lower() == "cancelled":
        return {
            "cash_paid_amount": cash_paid_amount,
            "advance_used": advance_adjusted,
            "payable_after_advance": payable_after_advance,
            "balance_amount": 0.0,
            "payment_status": "cancelled",
        }

    if payable_after_advance <= 0 or balance_amount <= 0:
        payment_status = "paid"
    elif cash_paid_amount > 0 or advance_adjusted > 0:
        payment_status = "partial"
    else:
        payment_status = "pending"

    return {
        "cash_paid_amount": cash_paid_amount,
        "advance_used": advance_adjusted,
        "payable_after_advance": payable_after_advance,
        "balance_amount": balance_amount,
        "payment_status": payment_status,
    }


def serialize_invoice_summary(invoice: InvoiceModel, advance_used: Optional[float] = 0):
    payment_snapshot = get_invoice_payment_snapshot(
        total=invoice.total,
        paid_amount=invoice.paid_amount,
        advance_used=advance_used,
        current_status=invoice.payment_status,
    )

    return {
        "id": invoice.id,
        "invoice_number": invoice.invoice_number,
        "customer_id": invoice.customer_id,
        "customer_name": invoice.customer_name,
        "customer_phone": invoice.customer_phone,
        "customer_address": invoice.customer_address,
        "items": parse_invoice_items(invoice.items),
        "subtotal": invoice.subtotal,
        "gst_amount": invoice.gst_amount,
        "gst_enabled": bool(invoice.gst_enabled),
        "gst_rate": invoice.gst_rate,
        "discount": invoice.discount,
        "total": invoice.total,
        "paid_amount": payment_snapshot["cash_paid_amount"],
        "balance_amount": payment_snapshot["balance_amount"],
        "advance_used": payment_snapshot["advance_used"],
        "payment_mode": invoice.payment_mode,
        "payment_status": payment_snapshot["payment_status"],
        "additional_charges": invoice.additional_charges or [],
        "pdf_url": f"/api/invoices/{invoice.id}/pdf",
        "created_at": invoice.created_at.isoformat(),
        "created_by": invoice.created_by_name,
    }


def sync_invoice_pending_ledger(
    db: Session,
    invoice: InvoiceModel,
    customer_id: Optional[str],
    balance_amount: Optional[float],
    current_user: Optional[UserModel] = None,
):
    if not invoice or not customer_id:
        return

    # 🔴 FIX 1: STOP EVERYTHING IF CANCELLED
    if (invoice.payment_status or "").lower() == "cancelled":
        db.query(LedgerEntry).filter(
            LedgerEntry.reference_id == invoice.id,
            LedgerEntry.customer_id == customer_id,
            LedgerEntry.entry_type == "sale_due",
        ).delete(synchronize_session=False)
        return

    pending_balance = round_currency(max(float(balance_amount or 0), 0))

    pending_entries = (
        db.query(LedgerEntry)
        .filter(
            LedgerEntry.reference_id == invoice.id,
            LedgerEntry.customer_id == customer_id,
            LedgerEntry.entry_type == "sale_due",
        )
        .order_by(LedgerEntry.created_at.desc())
        .all()
    )

    # 🔴 FIX 2: if no pending → delete
    if pending_balance <= 0:
        for pending_entry in pending_entries:
            db.delete(pending_entry)
        return

    pending_entry = pending_entries[0] if pending_entries else None

    description = f"Pending balance for {invoice.invoice_number}"

    entry_date = (
        invoice.created_at.astimezone(IST).date()
        if getattr(invoice.created_at, "tzinfo", None)
        else (invoice.created_at.date() if invoice.created_at else datetime.now(IST).date())
    )

    if pending_entry:
        pending_entry.description = description
        pending_entry.debit = pending_balance
        pending_entry.credit = 0
        pending_entry.payment_mode = "ledger"
        pending_entry.entry_date = entry_date

        for extra_entry in pending_entries[1:]:
            db.delete(extra_entry)
        return

    if not current_user:
        return

    db.add(
        LedgerEntry(
            id=str(uuid.uuid4()),
            entry_type="sale_due",
            reference_id=invoice.id,
            customer_id=customer_id,
            description=description,
            debit=pending_balance,
            credit=0,
            payment_mode="ledger",
            entry_date=entry_date,
            created_by=current_user.id,
            created_by_name=current_user.name,
            created_at=datetime.now(IST),
        )
    )


def ensure_customer_can_add_advance(db: Session, customer: CustomerModel):
    if get_customer_pending_invoice_amount(db, customer.id) > 0:
        raise HTTPException(
            400,
            "Advance cannot be added while pending balance exists for this customer",
        )


def calculate_invoice_settlement(
    total: float,
    payment_status: Optional[str],
    cash_paid_amount: Optional[float] = 0,
    customer: Optional[CustomerModel] = None,
    use_advance: bool = False,
):
    normalized_status = normalize_draft_payment_status(payment_status)
    total_amount = round_currency(total)
    available_advance = get_customer_advance_amount(customer)
    advance_used = (
        round_currency(min(available_advance, total_amount))
        if use_advance and available_advance > 0
        else 0.0
    )

    remaining_after_advance = round_currency(max(total_amount - advance_used, 0))

    if normalized_status == "cancelled":
        return {
            "payment_status": "cancelled",
            "cash_paid_amount": 0.0,
            "balance_amount": 0.0,
            "advance_used": 0.0,
        }

    if remaining_after_advance <= 0:
        return {
            "payment_status": "paid",
            "cash_paid_amount": 0.0,
            "balance_amount": 0.0,
            "advance_used": advance_used,
        }

    if normalized_status == "paid":
        return {
            "payment_status": "paid",
            "cash_paid_amount": remaining_after_advance,
            "balance_amount": 0.0,
            "advance_used": advance_used,
        }

    if normalized_status == "partial":
        partial_paid = round_currency(cash_paid_amount)

        if partial_paid < 0:
            raise HTTPException(400, "Partial payment cannot be negative")

        if partial_paid > remaining_after_advance:
            raise HTTPException(400, "Partial exceeds payable amount")

        balance_amount = round_currency(remaining_after_advance - partial_paid)

        return {
            "payment_status": "paid" if balance_amount <= 0 else "partial",
            "cash_paid_amount": partial_paid,
            "balance_amount": max(balance_amount, 0.0),
            "advance_used": advance_used,
        }

    if advance_used > 0 and remaining_after_advance > 0:
        normalized_status = "partial"

    return {
        "payment_status": normalized_status,
        "cash_paid_amount": 0.0,
        "balance_amount": remaining_after_advance,
        "advance_used": advance_used,
    }


def add_ledger_entry(
    db: Session,
    *,
    entry_type: str,
    description: str,
    payment_mode: str,
    debit: float = 0,
    credit: float = 0,
    reference_id: Optional[str] = None,
    customer_id: Optional[str] = None,
    supplier_id: Optional[str] = None,
    entry_date: Optional[date] = None,
    current_user: Optional[UserModel] = None,
):
    db.add(
        LedgerEntry(
            id=str(uuid.uuid4()),
            entry_type=entry_type,
            reference_id=reference_id,
            customer_id=customer_id,
            supplier_id=supplier_id,
            description=description,
            debit=round_currency(debit),
            credit=round_currency(credit),
            payment_mode=payment_mode,
            entry_date=entry_date or datetime.now(IST).date(),
            created_by=getattr(current_user, "id", None),
            created_by_name=getattr(current_user, "name", None),
            created_at=datetime.now(IST),
        )
    )


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
    total = round(taxable + gst_amount - discount, 2)
    settlement = calculate_invoice_settlement(
        total=total,
        payment_status=payment_status,
        cash_paid_amount=invoice_data.paid_amount,
        customer=customer,
        use_advance=invoice_data.use_advance,
    )

    paid_amount = settlement["cash_paid_amount"]
    balance_amount = settlement["balance_amount"]
    payment_status = settlement["payment_status"]
    advance_used = settlement["advance_used"]

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
        "total": total,
        "payment_status": payment_status,
        "payment_mode": (
            "advance"
            if advance_used > 0 and paid_amount <= 0
            else payment_mode
        ),
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
    use_advance: bool = False

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

    draft.total = round(taxable + draft.gst_amount - draft.discount, 2)

    customer = db.query(CustomerModel).filter(
        CustomerModel.id == draft.customer_id
    ).with_for_update().first()

    # ================= PAYMENT =================
    settlement = calculate_invoice_settlement(
        total=draft.total,
        payment_status=final_payment_status,
        cash_paid_amount=(
            data.paid_amount
            if data and data.paid_amount is not None
            else draft.paid_amount
        ),
        customer=customer,
        use_advance=bool(data.use_advance) if data else False,
    )

    paid_amount = settlement["cash_paid_amount"]
    balance_amount = settlement["balance_amount"]
    advance_used = settlement["advance_used"]
    final_payment_status = settlement["payment_status"]

    draft.paid_amount = paid_amount
    draft.balance_amount = balance_amount
    draft.payment_mode = (
        "advance"
        if advance_used > 0 and paid_amount <= 0
        else final_payment_mode
    )

    # ================= FINALIZE =================
    invoice_number = generate_invoice_number(db)

    draft.invoice_type = "FINAL"
    draft.invoice_number = invoice_number
    draft.draft_number = None
    draft.payment_status = final_payment_status
    draft.created_at = datetime.now(IST)

    if customer:
        invoice_balance_delta = get_invoice_balance_adjustment(
            balance_amount=balance_amount,
            advance_used=advance_used,
        )
        if invoice_balance_delta > 0:
            update_customer_balance(customer, invoice_balance_delta)

        if advance_used > 0:
            add_ledger_entry(
                db,
                entry_type="advance_used",
                reference_id=draft.id,
                customer_id=draft.customer_id,
                description=f"Advance used for invoice {invoice_number}",
                debit=advance_used,
                credit=0,
                payment_mode="adjustment",
                current_user=current_user,
            )

        if balance_amount > 0:
            sync_invoice_pending_ledger(
                db,
                draft,
                draft.customer_id,
                balance_amount,
                current_user,
            )

        if paid_amount > 0:
            db.add(
                InvoicePayment(
                    id=str(uuid.uuid4()),
                    invoice_id=draft.id,
                    amount=paid_amount,
                    payment_mode=final_payment_mode,
                    reference=INITIAL_INVOICE_PAYMENT_REFERENCE,
                    created_by=current_user.id,
                    created_at=datetime.now(IST),
                )
            )

            add_ledger_entry(
                db,
                entry_type="sale_payment",
                reference_id=draft.id,
                customer_id=draft.customer_id,
                description=f"{final_payment_mode.title()} received for invoice {invoice_number}",
                debit=0,
                credit=paid_amount,
                payment_mode=final_payment_mode,
                current_user=current_user,
            )

    db.commit()

    return {
        "invoice_id": draft.id,
        "invoice_number": invoice_number,
        "total": draft.total,
        "paid_amount": paid_amount,
        "balance_amount": balance_amount,
        "advance_used": advance_used,
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
        opening_balance=new_customer.opening_balance,
        current_balance=normalize_customer_balance_value(new_customer.current_balance),
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
            func.coalesce(
                func.sum(
                    case(
                        (InvoiceModel.payment_status != "cancelled", InvoiceModel.total),
                        else_=0,
                    )
                ),
                0,
            ).label("total_bill")
        )
        .outerjoin(
            InvoiceModel,
            and_(
                InvoiceModel.customer_id == CustomerModel.id,
                InvoiceModel.invoice_type == "FINAL"   # âœ… EXCLUDE DRAFTS
            )
        )
        .group_by(CustomerModel.id)
        .order_by(CustomerModel.created_at.desc())
        .all()
    )

    customer_ids = [c.id for c, _, _ in customers]
    pending_map = get_customer_pending_invoice_map(db, customer_ids)

    return [
        serialize_customer_overview(
            db,
            c,
            total_invoices=total_invoices,
            total_bill=total_bill,
            pending_invoice_amount=pending_map.get(c.id, 0),
        )
        for c, total_invoices, total_bill in customers
    ]

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
        opening_balance=customer.opening_balance,
        current_balance=normalize_customer_balance_value(customer.current_balance),
        display_balance=get_customer_display_balance(customer.current_balance),
        pending_balance=get_customer_pending_amount(customer),
        advance_balance=get_customer_advance_amount(customer),
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


@api_router.post("/customers/{customer_id}/advance")
def add_customer_advance(
    customer_id: str,
    data: CustomerAdvanceRequest,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    customer = db.query(CustomerModel).filter(
        CustomerModel.id == customer_id
    ).with_for_update().first()
    if not customer:
        raise HTTPException(404, "Customer not found")

    amount = round_currency(data.amount)

    if amount <= 0:
        raise HTTPException(400, "Advance amount must be greater than 0")

    payment_mode = normalize_draft_payment_mode(data.payment_mode)
    ensure_customer_can_add_advance(db, customer)

    payment_date = data.payment_date or datetime.now(IST).date()
    update_customer_balance(customer, -amount)

    if payment_mode == "cheque" and data.cheque_number:
        db.add(
            ChequeModel(
                id=str(uuid.uuid4()),
                cheque_number=data.cheque_number,
                cheque_date=data.cheque_date or payment_date,
                amount=amount,
                bank_name=data.bank_name,
                party_name=customer.name,
                party_type="customer",
                status="pending",
                payment_mode="cheque",
                created_by=current_user.id,
                created_at=datetime.now(IST),
            )
        )

    add_ledger_entry(
        db,
        entry_type="advance_in",
        reference_id=customer.id,
        customer_id=customer.id,
        description=f"Advance received from {customer.name}",
        debit=0,
        credit=amount,
        payment_mode=payment_mode,
        entry_date=payment_date,
        current_user=current_user,
    )

    db.commit()

    return {
        "message": "Advance added successfully",
        "customer_id": customer.id,
        "current_balance": normalize_customer_balance_value(customer.current_balance),
        "pending_balance": get_customer_pending_amount(customer),
        "advance_balance": get_customer_advance_amount(customer),
    }


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

    invoice_ids = [inv.id for inv in invoices]
    advance_used_map = {}

    if invoice_ids:
        advance_rows = (
            db.query(
                LedgerEntry.reference_id,
                func.coalesce(func.sum(LedgerEntry.debit), 0)
            )
            .filter(
                LedgerEntry.reference_id.in_(invoice_ids),
                LedgerEntry.entry_type == "advance_used"
            )
            .group_by(LedgerEntry.reference_id)
            .all()
        )

        advance_used_map = {
            reference_id: round_currency(amount)
            for reference_id, amount in advance_rows
        }

    invoice_data = [
        serialize_invoice_summary(inv, advance_used_map.get(inv.id, 0))
        for inv in invoices
    ]

    return {
        "data": invoice_data,
        "pagination": {
            "page": page,
            "limit": limit,
            "total": total,
            "total_pages": math.ceil(total / limit) if total else 1
        }
    }

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
                "total": inv.total,

                # â­ PAYMENT DETAILS
                "paid_amount": float(inv.paid_amount or 0),
                "balance_amount": float(inv.balance_amount or 0),  # âœ… Fixed: Don't use inv.total as fallback
                "advance_used": float(advance_used_map.get(inv.id, 0)),
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
            "total_pages": math.ceil(total / limit) if total else 1
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
            ).with_for_update().first()

        elif invoice_data.customer_phone:
            customer = db.query(CustomerModel).filter(
                CustomerModel.phone == invoice_data.customer_phone
            ).with_for_update().first()

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

        total = round(subtotal, 2)
        payment_mode = normalize_draft_payment_mode(invoice_data.payment_mode)
        settlement = calculate_invoice_settlement(
            total=total,
            payment_status=invoice_data.payment_status,
            cash_paid_amount=invoice_data.paid_amount,
            customer=customer,
            use_advance=invoice_data.use_advance,
        )

        invoice = InvoiceModel(
            id=str(uuid.uuid4()),
            invoice_number=generate_invoice_number(db),
            invoice_type="FINAL",

            customer_id=customer.id,
            customer_name=customer.name,
            customer_phone=customer.phone,
            customer_address=customer.address,

            items=json.dumps(invoice_items),
            subtotal=subtotal,
            total=total,

            paid_amount=settlement["cash_paid_amount"],
            balance_amount=settlement["balance_amount"],
            payment_mode=(
                "advance"
                if settlement["advance_used"] > 0 and settlement["cash_paid_amount"] <= 0
                else payment_mode
            ),
            payment_status=settlement["payment_status"],

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
        advance_used = settlement["advance_used"]
        invoice_balance_delta = get_invoice_balance_adjustment(
            balance_amount=settlement["balance_amount"],
            advance_used=advance_used,
        )

        if invoice_balance_delta > 0:
            update_customer_balance(customer, invoice_balance_delta)

        if advance_used > 0:
            add_ledger_entry(
                db,
                entry_type="advance_used",
                reference_id=invoice.id,
                customer_id=customer.id,
                description=f"Advance used for invoice {invoice.invoice_number}",
                debit=advance_used,
                credit=0,
                payment_mode="adjustment",
                current_user=current_user,
            )

        if settlement["balance_amount"] > 0:
            sync_invoice_pending_ledger(
                db,
                invoice,
                customer.id,
                settlement["balance_amount"],
                current_user,
            )

        if settlement["cash_paid_amount"] > 0:
            db.add(
                InvoicePayment(
                    id=str(uuid.uuid4()),
                    invoice_id=invoice.id,
                    amount=settlement["cash_paid_amount"],
                    payment_mode=payment_mode,
                    reference=INITIAL_INVOICE_PAYMENT_REFERENCE,
                    created_by=current_user.id,
                    created_at=datetime.now(IST),
                )
            )

            add_ledger_entry(
                db,
                entry_type="sale_payment",
                reference_id=invoice.id,
                customer_id=customer.id,
                description=f"{payment_mode.title()} received for invoice {invoice.invoice_number}",
                debit=0,
                credit=settlement["cash_paid_amount"],
                payment_mode=payment_mode,
                current_user=current_user,
            )

        db.commit()

        return {
            "invoice_id": invoice.id,
            "invoice_number": invoice.invoice_number,
            "total": invoice.total,
            "paid_amount": settlement["cash_paid_amount"],
            "balance_amount": settlement["balance_amount"],
            "advance_used": advance_used,
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
    reference: Optional[str] = None,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):

    payment_status = payment_status.lower()
    payment_mode = payment_mode.lower()
    amount = round_currency(amount)

    invoice = db.query(InvoiceModel).filter(
        InvoiceModel.id == invoice_id
    ).with_for_update().first()

    if not invoice:
        raise HTTPException(404, "Invoice not found")

    customer = db.query(CustomerModel).filter(
        CustomerModel.id == invoice.customer_id
    ).with_for_update().first()

    advance_used = get_invoice_advance_used_amount(db, invoice.id)

    payment_snapshot = get_invoice_payment_snapshot(
        total=invoice.total,
        paid_amount=invoice.paid_amount,
        advance_used=advance_used,
        current_status=invoice.payment_status,
    )

    # 🚫 block duplicate payment
    if payment_status in ["paid", "partial"] and payment_snapshot["balance_amount"] <= 0:
        raise HTTPException(400, "Invoice already fully paid")

    # =====================================================
    # 🔴 CANCELLED (FULL FIX)
    # =====================================================
    if payment_status == "cancelled":

        if invoice.payment_status == "cancelled":
            raise HTTPException(400, "Already cancelled")

        # 🧾 STOCK REVERSAL (same as your code)
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

            variants = list(product.variants or [])
            variant_stock_after = None

            if variants and sku:
                for v in variants:
                    if v.get("v_sku") == sku:
                        v["stock"] = int(v.get("stock", 0)) + quantity
                        variant_stock_after = v["stock"]
                        break

                product.variants = variants
                product.stock = calculate_total_stock(variants)
                flag_modified(product, "variants")
                flag_modified(product, "stock")
            else:
                product.stock = stock_before + quantity
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
                    stock_after=product.stock,
                    variant_sku=sku,
                    variant_stock_after=variant_stock_after,
                    created_by=current_user.id,
                    created_by_name=current_user.name,
                    created_at=datetime.now(IST),
                )
            )

        pending_amount = payment_snapshot["balance_amount"]
        cash_paid_amount = payment_snapshot["cash_paid_amount"]

        reversal_amount = get_invoice_balance_adjustment(
            balance_amount=pending_amount,
            advance_used=advance_used,
            cash_paid_amount=cash_paid_amount,
        )

        if customer and reversal_amount > 0:
            update_customer_balance(customer, -reversal_amount)

        delete_invoice_balance_ledger_entries(
            db,
            invoice.id,
            invoice.customer_id,
        )
        delete_invoice_payment_records(db, invoice.id)

        invoice.payment_status = "cancelled"
        invoice.balance_amount = 0
        invoice.paid_amount = 0

        db.commit()

        return {
            "message": "Invoice cancelled successfully",
            "invoice_number": invoice.invoice_number
        }

    # =====================================================
    # 🟡 PARTIAL
    # =====================================================
    if payment_status == "partial":

        if amount <= 0:
            raise HTTPException(400, "Amount required")

        remaining = payment_snapshot["balance_amount"]

        if amount > remaining:
            raise HTTPException(400, "Too much amount")

        invoice.paid_amount = round_currency(invoice.paid_amount + amount)

        updated_snapshot = get_invoice_payment_snapshot(
            total=invoice.total,
            paid_amount=invoice.paid_amount,
            advance_used=advance_used,
        )

        invoice.balance_amount = updated_snapshot["balance_amount"]
        invoice.payment_status = updated_snapshot["payment_status"]
        invoice.payment_mode = payment_mode

        # ✅ sync ledger
        sync_invoice_pending_ledger(
            db,
            invoice,
            invoice.customer_id,
            invoice.balance_amount,
            current_user,
        )

        # ✅ update balance
        if customer:
            update_customer_balance(customer, -amount)

        db.add(InvoicePayment(
            invoice_id=invoice.id,
            amount=amount,
            payment_mode=payment_mode,
            reference=reference,
            created_by=current_user.id
        ))

        db.add(
            LedgerEntry(
                id=str(uuid.uuid4()),
                entry_type="sale_payment",
                reference_id=invoice.id,
                customer_id=invoice.customer_id,
                description=f"{payment_mode.title()} received for invoice {invoice.invoice_number}",
                debit=0,
                credit=amount,
                payment_mode=payment_mode,
                entry_date=datetime.now(IST).date(),
                created_by=current_user.id,
                created_by_name=current_user.name,
                created_at=datetime.now(IST),
            )
        )

        db.commit()
        return {"message": "Partial payment updated"}

    # =====================================================
    # 🟢 PAID
    # =====================================================
    if payment_status == "paid":

        remaining = payment_snapshot["balance_amount"]

        if remaining <= 0:
            raise HTTPException(400, "Already paid")

        invoice.paid_amount += remaining
        invoice.balance_amount = 0
        invoice.payment_status = "paid"
        invoice.payment_mode = payment_mode

        sync_invoice_pending_ledger(
            db,
            invoice,
            invoice.customer_id,
            invoice.balance_amount,
            current_user,
        )

        if customer:
            update_customer_balance(customer, -remaining)

        db.add(InvoicePayment(
            invoice_id=invoice.id,
            amount=remaining,
            payment_mode=payment_mode,
            reference=reference,
            created_by=current_user.id
        ))

        db.add(
            LedgerEntry(
                id=str(uuid.uuid4()),
                entry_type="sale_payment",
                reference_id=invoice.id,
                customer_id=invoice.customer_id,
                description=f"{payment_mode.title()} received for invoice {invoice.invoice_number}",
                debit=0,
                credit=remaining,
                payment_mode=payment_mode,
                entry_date=datetime.now(IST).date(),
                created_by=current_user.id,
                created_by_name=current_user.name,
                created_at=datetime.now(IST),
            )
        )

        db.commit()
        return {"message": "Invoice marked as paid"}

    raise HTTPException(400, "Invalid status")

def handle_cancelled_invoice(db: Session, invoice: InvoiceModel, customer: CustomerModel):
    if not customer:
        return

    # 1. Remove pending from balance
    update_customer_balance(
        customer,
        delta = -invoice.balance_amount
    )

    # 2. Restore advance used
    advance_used = get_invoice_advance_used_amount(db, invoice.id)

    if advance_used > 0:
        update_customer_balance(
            customer,
            delta = -advance_used
        )

    # 3. Delete ledger entries
    db.query(LedgerEntry).filter(
        LedgerEntry.reference_id == invoice.id,
        LedgerEntry.customer_id == customer.id
    ).delete(synchronize_session=False)

    # 4. Delete payments (optional)
    db.query(InvoicePayment).filter(
        InvoicePayment.invoice_id == invoice.id
    ).delete(synchronize_session=False)

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

    advance_used = get_invoice_advance_used_amount(db, invoice.id)
    payment_snapshot = get_invoice_payment_snapshot(
        total=invoice.total,
        paid_amount=invoice.paid_amount,
        advance_used=advance_used,
        current_status=invoice.payment_status,
    )

    if payment_snapshot["payment_status"] == "paid":
        raise HTTPException(400, "Invoice already fully paid")

    balance = payment_snapshot["balance_amount"]

    if balance <= 0:
        raise HTTPException(400, "No balance remaining")

    # UPDATE INVOICE
    invoice.paid_amount = round_currency(invoice.paid_amount + balance)
    invoice.balance_amount = 0
    invoice.payment_status = "paid"
    invoice.payment_mode = payment_mode
    sync_invoice_pending_ledger(
        db,
        invoice,
        invoice.customer_id,
        invoice.balance_amount,
        current_user,
    )

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
    ).with_for_update().first()

    if customer:
        update_customer_balance(customer, -balance)

    # LEDGER ENTRY
    db.add(
        LedgerEntry(
            id=str(uuid.uuid4()),
            entry_type="sale_payment",
            reference_id=invoice.id,
            customer_id=invoice.customer_id,
            description=f"{payment_mode.title()} received for invoice {invoice.invoice_number}",
            debit=0,
            credit=balance,
            payment_mode=payment_mode,
            entry_date=datetime.now(IST).date(),
            created_by=current_user.id,
            created_by_name=current_user.name,
            created_at=datetime.now(IST),
        )
    )

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
            "reference": None if p.reference == INITIAL_INVOICE_PAYMENT_REFERENCE else p.reference,
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
    payment_amount = round_currency(payment_data.amount)

    # Validate payment amount
    if payment_amount <= 0:
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

    advance_used = get_invoice_advance_used_amount(db, invoice.id)
    payment_snapshot = get_invoice_payment_snapshot(
        total=invoice.total,
        paid_amount=invoice.paid_amount,
        advance_used=advance_used,
        current_status=invoice.payment_status,
    )

    if payment_snapshot["payment_status"] == "paid":
        raise HTTPException(400, "Invoice already fully paid")

    # Calculate remaining balance
    remaining_balance = payment_snapshot["balance_amount"]

    if payment_amount > remaining_balance:
        raise HTTPException(400, f"Payment amount exceeds balance of {remaining_balance}")

    # Update invoice
    invoice.paid_amount = round_currency(invoice.paid_amount + payment_amount)
    updated_snapshot = get_invoice_payment_snapshot(
        total=invoice.total,
        paid_amount=invoice.paid_amount,
        advance_used=advance_used,
    )
    invoice.balance_amount = updated_snapshot["balance_amount"]
    invoice.payment_mode = payment_data.payment_mode.lower()

    # Auto-update payment status
    if updated_snapshot["balance_amount"] <= 0:
        invoice.payment_status = "paid"
        invoice.balance_amount = 0
    else:
        invoice.payment_status = updated_snapshot["payment_status"]
    sync_invoice_pending_ledger(
        db,
        invoice,
        invoice.customer_id,
        invoice.balance_amount,
        current_user,
    )

    # Store payment history
    db.add(
        InvoicePayment(
            id=str(uuid.uuid4()),
            invoice_id=invoice.id,
            amount=payment_amount,
            payment_mode=payment_data.payment_mode.lower(),
            reference=payment_data.reference,
            created_by=current_user.id,
            created_at=datetime.now(IST)
        )
    )

    # Update customer balance
    customer = db.query(CustomerModel).filter(
        CustomerModel.id == invoice.customer_id
    ).with_for_update().first()

    if customer:
        update_customer_balance(customer, -payment_amount)

    # Add ledger entry
    db.add(
        LedgerEntry(
            id=str(uuid.uuid4()),
            entry_type="sale_payment",
            reference_id=invoice.id,
            customer_id=invoice.customer_id,
            description=f"{payment_data.payment_mode.title()} received for invoice {invoice.invoice_number}",
            debit=0,
            credit=payment_amount,
            payment_mode=payment_data.payment_mode.lower(),
            entry_date=datetime.now(IST).date(),
            created_by=current_user.id,
            created_by_name=current_user.name,
            created_at=datetime.now(IST),
        )
    )

    db.commit()

    return {
        "message": "Payment recorded successfully",
        "invoice_number": invoice.invoice_number,
        "payment_amount": payment_amount,
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
    payment_amount = round_currency(payment.amount)

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
    invoice.paid_amount = round_currency(max(invoice.paid_amount - payment_amount, 0))
    advance_used = get_invoice_advance_used_amount(db, invoice.id)
    updated_snapshot = get_invoice_payment_snapshot(
        total=invoice.total,
        paid_amount=invoice.paid_amount,
        advance_used=advance_used,
    )
    invoice.balance_amount = updated_snapshot["balance_amount"]
    invoice.payment_status = updated_snapshot["payment_status"]
    sync_invoice_pending_ledger(
        db,
        invoice,
        invoice.customer_id,
        invoice.balance_amount,
        current_user,
    )

    # Update customer balance
    customer = db.query(CustomerModel).filter(
        CustomerModel.id == invoice.customer_id
    ).with_for_update().first()

    if customer:
        update_customer_balance(customer, payment_amount)

    # Delete payment record
    db.delete(payment)

    # Delete corresponding ledger entry
    ledger_entry = db.query(LedgerEntry).filter(
        LedgerEntry.reference_id == invoice_id,
        LedgerEntry.credit == payment_amount,
        LedgerEntry.entry_type.in_(["sale", "sale_payment"]),
        LedgerEntry.payment_mode == payment.payment_mode
    ).order_by(LedgerEntry.created_at.desc()).first()

    if ledger_entry:
        db.delete(ledger_entry)

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

    balance_snapshot = get_customer_balance_snapshot(db, customer)

    return Customer(
        id=customer.id,
        name=customer.name,
        email=customer.email,
        phone=customer.phone,
        address=customer.address,
        opening_balance=customer.opening_balance,
        current_balance=balance_snapshot["current_balance"],
        display_balance=balance_snapshot["display_balance"],
        pending_balance=balance_snapshot["pending_balance"],
        advance_balance=balance_snapshot["advance_balance"],
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

    def money(value: Optional[float]) -> str:
        return f"Rs. {round_currency(value):,.2f}"

    def build_product_paragraph(item: dict, style) -> Paragraph:
        name = escape(str(item.get("product_name") or "Item"))
        variant_info = item.get("variant_info") or {}
        details = []

        sku_value = item.get("sku") or variant_info.get("v_sku")
        if sku_value:
            details.append(f"SKU: {sku_value}")
        if variant_info.get("variant_name"):
            details.append(f"Variant: {variant_info.get('variant_name')}")
        if variant_info.get("color"):
            details.append(f"Color: {variant_info.get('color')}")
        if variant_info.get("size"):
            details.append(f"Size: {variant_info.get('size')}")

        markup = f"<b>{name}</b>"
        if details:
            markup += (
                "<br/><font size='8' color='#64748b'>"
                f"{escape(' | '.join(details))}"
                "</font>"
            )
        return Paragraph(markup, style)

    invoice_date = (
        invoice.created_at.astimezone(IST).date()
        if getattr(invoice.created_at, "tzinfo", None)
        else (invoice.created_at.date() if invoice.created_at else datetime.now(IST).date())
    )
    items = parse_invoice_items(invoice.items)
    additional_charges = [
        charge
        for charge in (invoice.additional_charges or [])
        if round_currency((charge or {}).get("amount")) > 0
    ]
    advance_used = get_invoice_advance_used_amount(db, invoice.id)
    payment_snapshot = get_invoice_payment_snapshot(
        total=invoice.total,
        paid_amount=invoice.paid_amount,
        advance_used=advance_used,
        current_status=invoice.payment_status,
    )

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        leftMargin=24,
        rightMargin=24,
        topMargin=24,
        bottomMargin=24,
    )
    elements = []

    styles = getSampleStyleSheet()
    company_name_style = ParagraphStyle(
        "InvoiceCompanyName",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=22,
        leading=24,
        textColor=colors.HexColor("#0f172a"),
        alignment=TA_LEFT,
        spaceAfter=2,
    )
    company_tagline_style = ParagraphStyle(
        "InvoiceCompanyTagline",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=13,
        textColor=colors.HexColor("#475569"),
        alignment=TA_LEFT,
        spaceAfter=6,
    )
    small_style = ParagraphStyle(
        "InvoiceSmall",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#475569"),
    )
    section_label_style = ParagraphStyle(
        "InvoiceSectionLabel",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=11,
        textColor=colors.HexColor("#64748b"),
        alignment=TA_LEFT,
    )
    section_value_style = ParagraphStyle(
        "InvoiceSectionValue",
        parent=styles["Normal"],
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#0f172a"),
        alignment=TA_LEFT,
    )
    right_value_style = ParagraphStyle(
        "InvoiceRightValue",
        parent=section_value_style,
        alignment=TA_RIGHT,
    )
    item_style = ParagraphStyle(
        "InvoiceItem",
        parent=styles["Normal"],
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#0f172a"),
    )
    total_label_style = ParagraphStyle(
        "InvoiceTotalLabel",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#475569"),
        alignment=TA_RIGHT,
    )
    total_value_style = ParagraphStyle(
        "InvoiceTotalValue",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=12,
        textColor=colors.HexColor("#0f172a"),
        alignment=TA_RIGHT,
    )
    total_value_accent_style = ParagraphStyle(
        "InvoiceTotalValueAccent",
        parent=total_value_style,
        fontSize=12,
        textColor=colors.HexColor("#166534"),
    )
    total_value_alert_style = ParagraphStyle(
        "InvoiceTotalValueAlert",
        parent=total_value_style,
        textColor=colors.HexColor("#b91c1c"),
    )
    centered_small_style = ParagraphStyle(
        "InvoiceCenteredSmall",
        parent=small_style,
        alignment=TA_CENTER,
    )

    header_table = Table(
        [[
            [
                Paragraph(escape(INVOICE_COMPANY_NAME), company_name_style),
                Paragraph(escape(INVOICE_COMPANY_TAGLINE), company_tagline_style),
                Paragraph(escape(INVOICE_COMPANY_ADDRESS), small_style),
                Paragraph(escape(INVOICE_COMPANY_CONTACT), small_style),
            ],
            [
                Paragraph("INVOICE", company_tagline_style),
                Paragraph(
                    (
                        f"<b>No:</b> {escape(invoice.invoice_number or invoice.id)}<br/>"
                        f"<b>Date:</b> {escape(invoice_date.strftime('%d %b %Y'))}<br/>"
                        f"<b>Status:</b> {escape(str(payment_snapshot['payment_status']).upper())}<br/>"
                        f"<b>Created By:</b> {escape(invoice.created_by_name or '-')}"
                    ),
                    right_value_style,
                ),
            ],
        ]],
        colWidths=[330, 190],
    )
    header_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    elements.append(header_table)
    elements.append(Spacer(1, 14))

    customer_markup = (
        f"<b>{escape(invoice.customer_name or 'Walk-in Customer')}</b><br/>"
        f"{escape(invoice.customer_phone or '-')}<br/>"
        f"{escape(invoice.customer_address or '-')}"
    )
    invoice_info_markup = (
        f"<b>Payment Mode:</b> {escape((invoice.payment_mode or 'cash').upper())}<br/>"
        f"<b>Cash Paid:</b> {escape(money(payment_snapshot['cash_paid_amount']))}<br/>"
        f"<b>Advance Used:</b> {escape(money(payment_snapshot['advance_used']))}<br/>"
        f"<b>Balance:</b> {escape(money(payment_snapshot['balance_amount']))}"
    )

    info_table = Table(
        [[
            [
                Paragraph("INVOICE TO", section_label_style),
                Spacer(1, 4),
                Paragraph(customer_markup, section_value_style),
            ],
            [
                Paragraph("INVOICE DETAILS", section_label_style),
                Spacer(1, 4),
                Paragraph(invoice_info_markup, section_value_style),
            ],
        ]],
        colWidths=[260, 260],
    )
    info_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.white),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.6, colors.HexColor("#e2e8f0")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))
    elements.append(info_table)
    elements.append(Spacer(1, 14))

    items_table_data = [[
        "#",
        "Product",
        "Qty",
        "Price",
        "GST",
        "Amount",
    ]]

    for index, item in enumerate(items, start=1):
        quantity = int(item.get("quantity") or 0)
        price = round_currency(item.get("price"))
        line_total = round_currency(
            item.get("total")
            if item.get("total") is not None
            else quantity * price
        )
        gst_display = (
            f"{round_currency(item.get('gst_rate') or invoice.gst_rate):.0f}%"
            if invoice.gst_enabled
            else "0%"
        )

        items_table_data.append([
            str(index),
            build_product_paragraph(item, item_style),
            str(quantity),
            money(price),
            gst_display,
            money(line_total),
        ])

    items_table = Table(
        items_table_data,
        colWidths=[32, 238, 48, 70, 52, 80],
        repeatRows=1,
    )
    items_style = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, 0), 9),
        ("FONTSIZE", (0, 1), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.6, colors.HexColor("#cbd5e1")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ("ALIGN", (2, 1), (-1, -1), "RIGHT"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
    ]
    items_table.setStyle(TableStyle(items_style))
    elements.append(items_table)
    elements.append(Spacer(1, 12))

    totals_rows = [
        [
            Paragraph("Subtotal", total_label_style),
            Paragraph(money(invoice.subtotal), total_value_style),
        ]
    ]

    for charge in additional_charges:
        charge_label = escape(str(charge.get("label") or "Additional Charge"))
        totals_rows.append([
            Paragraph(charge_label, total_label_style),
            Paragraph(money(charge.get("amount")), total_value_style),
        ])

    if round_currency(invoice.gst_amount) > 0:
        totals_rows.append([
            Paragraph("GST Amount", total_label_style),
            Paragraph(money(invoice.gst_amount), total_value_style),
        ])

    if round_currency(invoice.discount) > 0:
        totals_rows.append([
            Paragraph("Discount", total_label_style),
            Paragraph(f"- {money(invoice.discount)}", total_value_style),
        ])

    totals_rows.append([
        Paragraph("Total Amount", total_label_style),
        Paragraph(money(invoice.total), total_value_accent_style),
    ])

    if payment_snapshot["advance_used"] > 0:
        totals_rows.append([
            Paragraph("Advance Used", total_label_style),
            Paragraph(money(payment_snapshot["advance_used"]), total_value_style),
        ])

    if payment_snapshot["cash_paid_amount"] > 0:
        totals_rows.append([
            Paragraph("Paid Amount", total_label_style),
            Paragraph(money(payment_snapshot["cash_paid_amount"]), total_value_style),
        ])

    if payment_snapshot["balance_amount"] > 0:
        totals_rows.append([
            Paragraph("Balance Due", total_label_style),
            Paragraph(money(payment_snapshot["balance_amount"]), total_value_alert_style),
        ])

    totals_table = Table(totals_rows, colWidths=[150, 110], hAlign="RIGHT")
    totals_table.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.6, colors.HexColor("#e2e8f0")),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#ffffff")),
        ("BACKGROUND", (0, len(totals_rows) - 1), (-1, len(totals_rows) - 1), colors.HexColor("#fef2f2") if payment_snapshot["balance_amount"] > 0 else colors.HexColor("#ecfdf5")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("ALIGN", (0, 0), (-1, -1), "RIGHT"),
    ]))
    elements.append(totals_table)
    elements.append(Spacer(1, 18))

    terms_markup = "<br/>".join(
        f"{index}. {escape(term)}"
        for index, term in enumerate(INVOICE_TERMS, start=1)
    )
    footer_table = Table(
        [[
            [
                Paragraph("TERMS & CONDITIONS", section_label_style),
                Spacer(1, 4),
                Paragraph(terms_markup, small_style),
            ],
            [
                Spacer(1, 18),
                Paragraph("Customer Signature", centered_small_style),
                Spacer(1, 30),
                Paragraph("Authorized Signatory", centered_small_style),
                Paragraph(escape(INVOICE_COMPANY_NAME), centered_small_style),
            ],
        ]],
        colWidths=[340, 180],
    )
    footer_table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.6, colors.HexColor("#e2e8f0")),
        ("LEFTPADDING", (0, 0), (-1, -1), 12),
        ("RIGHTPADDING", (0, 0), (-1, -1), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 12),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
    ]))
    elements.append(footer_table)

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
    expense = create_expense_with_ledger(
        db,
        title=data.title,
        amount=data.amount,
        category_id=data.category_id,
        payment_mode=data.payment_mode,
        description=data.description,
        expense_date=expense_date,
        current_user=current_user,
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


@api_router.get("/employees")
def get_employees(
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    employees = db.query(EmployeeModel).order_by(EmployeeModel.name.asc()).all()
    return [serialize_employee(db, employee) for employee in employees]


@api_router.post("/employees")
def create_employee(
    data: EmployeeCreate,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if round_currency(data.salary) <= 0:
        raise HTTPException(400, "Salary must be greater than zero")

    employee = EmployeeModel(
        id=str(uuid.uuid4()),
        name=data.name.strip(),
        role=data.role,
        salary=round_currency(data.salary),
        created_at=datetime.now(IST),
    )
    db.add(employee)
    db.commit()
    db.refresh(employee)

    return serialize_employee(db, employee)


@api_router.put("/employees/{employee_id}")
def update_employee(
    employee_id: str,
    data: EmployeeUpdate,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    employee = (
        db.query(EmployeeModel)
        .filter(EmployeeModel.id == employee_id)
        .with_for_update()
        .first()
    )
    if not employee:
        raise HTTPException(404, "Employee not found")

    updated_salary = round_currency(data.salary)
    if updated_salary <= 0:
        raise HTTPException(400, "Salary must be greater than zero")

    employee.name = data.name.strip()
    employee.role = data.role
    employee.salary = updated_salary

    current_month_record = get_employee_month_salary_record(db, employee.id)
    if current_month_record and round_currency(current_month_record.pending_amount) > 0:
        current_month_record.total_salary = updated_salary
        current_month_record.pending_amount = round_currency(
            max(updated_salary - float(current_month_record.paid_amount or 0), 0)
        )

    db.commit()
    db.refresh(employee)
    return serialize_employee(db, employee)


@api_router.delete("/employees/{employee_id}")
def delete_employee(
    employee_id: str,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    employee = db.query(EmployeeModel).filter(EmployeeModel.id == employee_id).first()
    if not employee:
        raise HTTPException(404, "Employee not found")

    salary_history = (
        db.query(SalaryPaymentModel.id)
        .filter(SalaryPaymentModel.employee_id == employee_id)
        .first()
    )
    if salary_history:
        raise HTTPException(400, "Employee has salary history and cannot be deleted")

    db.delete(employee)
    db.commit()
    return {"message": "Employee deleted successfully"}


@api_router.get("/salary-payments")
def get_salary_payments(
    employee_id: Optional[str] = None,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(SalaryPaymentModel)
    if employee_id:
        query = query.filter(SalaryPaymentModel.employee_id == employee_id)

    rows = query.order_by(SalaryPaymentModel.created_at.desc()).all()
    employees = {
        employee.id: employee
        for employee in db.query(EmployeeModel).filter(
            EmployeeModel.id.in_([row.employee_id for row in rows] or [""])
        ).all()
    }

    return [
        {
            **serialize_salary_record(row),
            "employee_name": employees.get(row.employee_id).name if employees.get(row.employee_id) else None,
            "employee_role": employees.get(row.employee_id).role if employees.get(row.employee_id) else None,
            "salary_period": get_salary_month_label(row.payment_date),
        }
        for row in rows
    ]


@api_router.post("/employees/{employee_id}/salary-records")
def create_salary_record(
    employee_id: str,
    data: SalaryRecordCreate,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    employee = db.query(EmployeeModel).filter(EmployeeModel.id == employee_id).first()
    if not employee:
        raise HTTPException(404, "Employee not found")

    payment_date = data.payment_date or datetime.now(IST).date()
    existing_record = get_employee_month_salary_record(db, employee.id, payment_date)
    if existing_record:
        raise HTTPException(
            400,
            f"Salary record already exists for {get_salary_month_label(payment_date)}",
        )

    total_salary = round_currency(employee.salary)
    record = SalaryPaymentModel(
        id=str(uuid.uuid4()),
        employee_id=employee.id,
        total_salary=total_salary,
        paid_amount=0,
        pending_amount=total_salary,
        payment_date=payment_date,
        payment_mode=None,
        notes=data.notes,
        created_at=datetime.now(IST),
    )
    db.add(record)
    db.commit()
    db.refresh(record)

    return serialize_salary_record(record)


def pay_salary_record(
    db: Session,
    *,
    record: SalaryPaymentModel,
    employee: EmployeeModel,
    amount: float,
    payment_mode: str,
    payment_date: date,
    notes: Optional[str],
    current_user: UserModel,
) -> SalaryPaymentModel:
    amount = round_currency(amount)
    if amount <= 0:
        raise HTTPException(400, "Payment amount must be greater than zero")

    if amount > round_currency(record.pending_amount):
        raise HTTPException(400, "Payment amount cannot exceed pending salary")

    record.paid_amount = round_currency(float(record.paid_amount or 0) + amount)
    record.pending_amount = round_currency(max(float(record.total_salary or 0) - float(record.paid_amount or 0), 0))
    record.payment_mode = payment_mode
    record.payment_date = payment_date
    record.notes = notes or record.notes

    db.add(
        LedgerEntry(
            id=str(uuid.uuid4()),
            entry_type="salary_expense",
            reference_id=record.id,
            description=f"Salary paid to {employee.name}",
            debit=amount,
            credit=0,
            payment_mode=payment_mode,
            entry_date=payment_date,
            created_by=current_user.id,
            created_by_name=current_user.name,
            created_at=datetime.now(IST),
        )
    )
    return record


@api_router.post("/salary-payments/{salary_payment_id}/pay")
def pay_existing_salary_record(
    salary_payment_id: str,
    data: SalaryPaymentRequest,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    record = (
        db.query(SalaryPaymentModel)
        .filter(SalaryPaymentModel.id == salary_payment_id)
        .with_for_update()
        .first()
    )
    if not record:
        raise HTTPException(404, "Salary record not found")

    employee = db.query(EmployeeModel).filter(EmployeeModel.id == record.employee_id).first()
    if not employee:
        raise HTTPException(404, "Employee not found")

    record = pay_salary_record(
        db,
        record=record,
        employee=employee,
        amount=data.amount,
        payment_mode=data.payment_mode,
        payment_date=data.payment_date or datetime.now(IST).date(),
        notes=data.notes,
        current_user=current_user,
    )
    db.commit()
    db.refresh(record)
    return serialize_salary_record(record)


@api_router.post("/employees/{employee_id}/salary-payments/pay")
def pay_employee_salary(
    employee_id: str,
    data: SalaryPaymentRequest,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    employee = db.query(EmployeeModel).filter(EmployeeModel.id == employee_id).first()
    if not employee:
        raise HTTPException(404, "Employee not found")

    payment_date = data.payment_date or datetime.now(IST).date()
    month_start = get_salary_month_start(payment_date)
    next_month_start = get_next_salary_month_start(payment_date)

    record = (
        db.query(SalaryPaymentModel)
        .filter(
            SalaryPaymentModel.employee_id == employee.id,
            SalaryPaymentModel.payment_date >= month_start,
            SalaryPaymentModel.payment_date < next_month_start,
        )
        .order_by(SalaryPaymentModel.payment_date.desc(), SalaryPaymentModel.created_at.desc())
        .with_for_update()
        .first()
    )

    if record and round_currency(record.pending_amount) <= 0:
        raise HTTPException(
            400,
            f"Salary already fully paid for {get_salary_month_label(payment_date)}",
        )

    if not record:
        total_salary = round_currency(employee.salary)
        record = SalaryPaymentModel(
            id=str(uuid.uuid4()),
            employee_id=employee.id,
            total_salary=total_salary,
            paid_amount=0,
            pending_amount=total_salary,
            payment_date=payment_date,
            payment_mode=None,
            notes=data.notes,
            created_at=datetime.now(IST),
        )
        db.add(record)
        db.flush()

    record = pay_salary_record(
        db,
        record=record,
        employee=employee,
        amount=data.amount,
        payment_mode=data.payment_mode,
        payment_date=payment_date,
        notes=data.notes,
        current_user=current_user,
    )
    db.commit()
    db.refresh(record)
    return serialize_salary_record(record)


@api_router.post("/power-bills")
def create_power_bill(
    data: PowerBillRequest,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    category = ensure_expense_category(db, "Power Bill", "Electricity and power utility bills")
    expense = create_expense_with_ledger(
        db,
        title=data.title or "Power Bill",
        amount=data.amount,
        category_id=category.id,
        payment_mode=data.payment_mode,
        description=data.description,
        expense_date=data.expense_date or datetime.now(IST).date(),
        current_user=current_user,
    )
    db.commit()
    db.refresh(expense)

    return {"message": "Power bill added successfully", "id": expense.id, "amount": expense.amount}


@api_router.get("/power-bills")
def get_power_bills(
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    category = ensure_expense_category(db, "Power Bill", "Electricity and power utility bills")
    db.commit()

    bills = (
        db.query(ExpenseModel)
        .filter(ExpenseModel.category_id == category.id)
        .order_by(ExpenseModel.expense_date.desc(), ExpenseModel.created_at.desc())
        .all()
    )

    return [
        {
            "id": bill.id,
            "title": bill.title,
            "amount": round_currency(bill.amount),
            "payment_mode": bill.payment_mode,
            "description": bill.description,
            "expense_date": bill.expense_date.isoformat() if bill.expense_date else None,
            "created_by_name": bill.created_by_name,
            "created_at": bill.created_at.isoformat() if bill.created_at else None,
        }
        for bill in bills
    ]


@api_router.get("/customers")
def get_customers(
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    customers = (
        db.query(
            CustomerModel,
            func.count(InvoiceModel.id).label("total_invoices"),
            func.coalesce(
                func.sum(
                    case(
                        (InvoiceModel.payment_status != "cancelled", InvoiceModel.total),
                        else_=0,
                    )
                ),
                0,
            ).label("total_bill")
        )
        .outerjoin(
            InvoiceModel,
            and_(
                InvoiceModel.customer_id == CustomerModel.id,
                InvoiceModel.invoice_type == "FINAL"
            )
        )
        .group_by(CustomerModel.id)
        .order_by(CustomerModel.created_at.desc())
        .all()
    )

    customer_ids = [c.id for c, _, _ in customers]
    pending_map = get_customer_pending_invoice_map(db, customer_ids)

    return [
        serialize_customer_overview(
            db,
            c,
            total_invoices=total_invoices,
            total_bill=total_bill,
            pending_invoice_amount=pending_map.get(c.id, 0),
        )
        for c, total_invoices, total_bill in customers
    ]


@api_router.get("/accounts/summary")
def get_accounts_summary(
    current_user: Optional[UserModel] = Depends(get_current_user),
    db: Session = Depends(get_db),
    start_date: Optional[date] = None,
    end_date: Optional[date] = None
):

    def calculate_stats(start_date=None, end_date=None):
        expense_filter = []
        ledger_filter = []

        # ✅ DATE FILTER FIX (IMPORTANT)
        if start_date:
            expense_filter.append(ExpenseModel.expense_date >= start_date)
            ledger_filter.append(LedgerEntry.entry_date >= start_date)

        if end_date:
            expense_filter.append(ExpenseModel.expense_date <= end_date)
            ledger_filter.append(LedgerEntry.entry_date <= end_date)

        def expense_total(mode: Optional[str] = None):
            query = db.query(func.coalesce(func.sum(ExpenseModel.amount), 0)).filter(*expense_filter)
            if mode:
                query = query.filter(ExpenseModel.payment_mode == mode)
            return round_currency(query.scalar())

        def ledger_credit_total(entry_types: List[str], modes: Optional[List[str]] = None):
            query = db.query(func.coalesce(func.sum(LedgerEntry.credit), 0)).filter(
                LedgerEntry.entry_type.in_(entry_types),
                LedgerEntry.entry_type.notin_(list(GENERAL_LEDGER_EXCLUDED_TYPES)),
                *ledger_filter,
            )
            if modes:
                query = query.filter(LedgerEntry.payment_mode.in_(modes))
            return round_currency(query.scalar())

        def ledger_debit_total(entry_types: List[str], modes: Optional[List[str]] = None):
            supplier_cash_out = build_supplier_cash_out_expression()
            normalized_debit = case(
                (
                    LedgerEntry.entry_type.in_(list(SUPPLIER_CASH_OUT_ENTRY_TYPES)),
                    supplier_cash_out,
                ),
                else_=LedgerEntry.debit,
            )

            query = db.query(func.coalesce(func.sum(normalized_debit), 0)).filter(
                LedgerEntry.entry_type.in_(entry_types),
                LedgerEntry.entry_type.notin_(list(GENERAL_LEDGER_EXCLUDED_TYPES)),
                *ledger_filter,
            )
            if modes:
                query = query.filter(LedgerEntry.payment_mode.in_(modes))
            return round_currency(query.scalar())

        def ledger_balance_total(modes: List[str]):
            balance_expression = build_ledger_balance_expression()
            query = db.query(func.coalesce(func.sum(balance_expression), 0)).filter(
                LedgerEntry.entry_type.notin_(list(GENERAL_LEDGER_EXCLUDED_TYPES)),
                LedgerEntry.payment_mode.in_(modes),
                *ledger_filter,
            )
            return round_currency(query.scalar())

        supplier_types = list(SUPPLIER_CASH_OUT_ENTRY_TYPES)
        money_in_types = list(LEDGER_MONEY_IN_ENTRY_TYPES)

        cash_balance = ledger_balance_total(["cash"])
        bank_balance = ledger_balance_total(["bank", "upi", "cheque"])
        total_received = ledger_credit_total(money_in_types)
        total_expense = expense_total()
        salary_expenses = ledger_debit_total(["salary_expense"])
        total_supplier_payment = ledger_debit_total(supplier_types)

        power_bill_category = (
            db.query(ExpenseCategoryModel)
            .filter(func.lower(ExpenseCategoryModel.name) == "power bill")
            .first()
        )
        if power_bill_category:
            power_bill_query = db.query(func.coalesce(func.sum(ExpenseModel.amount), 0)).filter(*expense_filter)
            power_bill_query = power_bill_query.filter(ExpenseModel.category_id == power_bill_category.id)
            power_bill_expenses = round_currency(power_bill_query.scalar())
        else:
            power_bill_expenses = 0.0

        net_profit = cash_balance + bank_balance - total_expense - total_supplier_payment - salary_expenses

        # ===== ✅ TOTAL SALES (FIXED) =====
        total_sales_query = db.query(func.sum(InvoiceModel.total))\
            .filter(
                InvoiceModel.invoice_type == "FINAL",
                InvoiceModel.payment_status != "cancelled",
            )

        if start_date:
            start_dt = datetime.combine(start_date, time.min)
            total_sales_query = total_sales_query.filter(InvoiceModel.created_at >= start_dt)

        if end_date:
            end_dt = datetime.combine(end_date, time.max)
            total_sales_query = total_sales_query.filter(InvoiceModel.created_at <= end_dt)

        total_sales = total_sales_query.scalar() or 0

        return {
            "total_sales": round(float(total_sales), 2),
            "income": round(float(total_received), 2),
            "expense": round(float(total_expense), 2),
            "total_expenses": round(float(total_expense), 2),
            "salary_expenses": round(float(salary_expenses), 2),
            "power_bill_expenses": round(float(power_bill_expenses), 2),
            "supplier_payment": round(float(total_supplier_payment), 2),
            "supplier_payments": round(float(total_supplier_payment), 2),
            "profit": round(float(net_profit), 2),
            "cash_balance": round(float(cash_balance), 2),
            "bank_balance": round(float(bank_balance), 2),
        }

    # ===== FILTERED =====
    filtered_stats = calculate_stats(start_date, end_date)

    # ===== OVERALL =====
    overall_stats = calculate_stats(None, None)

    return {
        "filtered": filtered_stats,
        "overall": overall_stats
    }

@api_router.get("/suppliers")
def get_suppliers(
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    suppliers = db.query(SupplierModel).order_by(SupplierModel.name.asc()).all()
    result = []

    for s in suppliers:
        summary = get_supplier_invoice_summary(db, s.id)
        result.append({
            "id": s.id,
            "name": s.name,
            "phone": s.phone,
            "email": s.email,
            "address": s.address,
            "current_balance": round_currency(s.current_balance),
            "total_bill": summary["total_bill"],
            "paid_amount": summary["paid_amount"],
            "pending_amount": summary["pending_amount"],
            "created_at": s.created_at.isoformat()
        })

    return result

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


@api_router.get("/supplier-invoices")
def get_supplier_invoices(
    supplier_id: Optional[str] = None,
    status: Optional[str] = None,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(SupplierInvoiceModel)

    if supplier_id:
        query = query.filter(SupplierInvoiceModel.supplier_id == supplier_id)
    if status and status != "all":
        query = query.filter(SupplierInvoiceModel.status == status)

    invoices = query.order_by(SupplierInvoiceModel.invoice_date.desc(), SupplierInvoiceModel.created_at.desc()).all()
    supplier_ids = [invoice.supplier_id for invoice in invoices]
    supplier_map = {
        supplier.id: supplier
        for supplier in db.query(SupplierModel).filter(SupplierModel.id.in_(supplier_ids or [""])).all()
    }

    return [serialize_supplier_invoice(invoice, supplier_map.get(invoice.supplier_id)) for invoice in invoices]


@api_router.post("/supplier-invoices")
def create_supplier_invoice(
    data: SupplierInvoiceCreate,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    supplier = (
        db.query(SupplierModel)
        .filter(SupplierModel.id == data.supplier_id)
        .with_for_update()
        .first()
    )
    if not supplier:
        raise HTTPException(404, "Supplier not found")

    total_amount = round_currency(data.total_amount)
    if total_amount <= 0:
        raise HTTPException(400, "Supplier bill amount must be greater than zero")

    invoice = SupplierInvoiceModel(
        id=str(uuid.uuid4()),
        supplier_id=supplier.id,
        total_amount=total_amount,
        paid_amount=0,
        pending_amount=total_amount,
        invoice_date=data.invoice_date or datetime.now(IST).date(),
        status="pending",
        created_at=datetime.now(IST),
    )
    supplier.current_balance = round_currency(float(supplier.current_balance or 0) + total_amount)

    db.add(invoice)
    add_ledger_entry(
        db,
        entry_type="supplier_bill",
        reference_id=invoice.id,
        supplier_id=supplier.id,
        description=f"Supplier bill {serialize_supplier_invoice(invoice, supplier)['bill_number']} added for {supplier.name}",
        debit=total_amount,
        credit=0,
        payment_mode="ledger",
        entry_date=invoice.invoice_date,
        current_user=current_user,
    )
    db.commit()
    db.refresh(invoice)

    return serialize_supplier_invoice(invoice, supplier)


@api_router.post("/supplier-invoices/{invoice_id}/pay")
def pay_supplier_invoice(
    invoice_id: str,
    data: SupplierInvoicePaymentRequest,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    invoice = (
        db.query(SupplierInvoiceModel)
        .filter(SupplierInvoiceModel.id == invoice_id)
        .with_for_update()
        .first()
    )
    if not invoice:
        raise HTTPException(404, "Supplier bill not found")

    supplier = (
        db.query(SupplierModel)
        .filter(SupplierModel.id == invoice.supplier_id)
        .with_for_update()
        .first()
    )
    if not supplier:
        raise HTTPException(404, "Supplier not found")

    amount = round_currency(data.amount)
    if amount <= 0:
        raise HTTPException(400, "Payment amount must be greater than zero")
    if amount > round_currency(invoice.pending_amount):
        raise HTTPException(400, "Payment amount cannot exceed supplier pending amount")

    invoice.paid_amount = round_currency(float(invoice.paid_amount or 0) + amount)
    update_supplier_invoice_status(invoice)
    supplier.current_balance = round_currency(float(supplier.current_balance or 0) - amount)

    payment_date = data.payment_date or datetime.now(IST).date()

    db.add(
        LedgerEntry(
            id=str(uuid.uuid4()),
            entry_type="supplier_payment",
            reference_id=invoice.id,
            supplier_id=supplier.id,
            description=f"Supplier payment for {serialize_supplier_invoice(invoice, supplier)['bill_number']}",
            debit=amount,
            credit=0,
            payment_mode=data.payment_mode,
            entry_date=payment_date,
            created_by=current_user.id,
            created_by_name=current_user.name,
            created_at=datetime.now(IST),
        )
    )

    db.commit()
    db.refresh(invoice)
    return serialize_supplier_invoice(invoice, supplier)


@api_router.post("/suppliers/{supplier_id}/invoices")
def create_supplier_invoice_for_supplier(
    supplier_id: str,
    data: SupplierInvoiceForSupplierCreate,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return create_supplier_invoice(
        SupplierInvoiceCreate(
            supplier_id=supplier_id,
            total_amount=data.total_amount,
            invoice_date=data.invoice_date,
        ),
        current_user,
        db,
    )


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

    base_query = db.query(LedgerEntry).filter(
        LedgerEntry.payment_mode == "cash",
        LedgerEntry.entry_type.notin_(list(GENERAL_LEDGER_EXCLUDED_TYPES)),
    )

    if start_date:
        base_query = base_query.filter(LedgerEntry.entry_date >= start_date)
    if end_date:
        base_query = base_query.filter(LedgerEntry.entry_date <= end_date)

    total = base_query.count()

    rows = base_query.order_by(
        LedgerEntry.entry_date.desc()
    ).offset((page - 1) * limit).limit(limit).all()

    opening_balance = db.query(
        func.coalesce(func.sum(build_ledger_balance_expression()), 0)
    ).filter(
        LedgerEntry.payment_mode == "cash",
        LedgerEntry.entry_type.notin_(list(GENERAL_LEDGER_EXCLUDED_TYPES)),
        LedgerEntry.entry_date < rows[-1].entry_date if rows else None
    ).scalar() or 0

    balance = opening_balance
    data = []

    for r in reversed(rows):
        display_debit, display_credit = get_normalized_ledger_debit_credit(r)
        balance += display_credit - display_debit

        data.insert(0, {
            "id": r.id,
            "date": r.entry_date.isoformat() if r.entry_date else None,
            "description": r.description,
            "debit": display_debit,
            "credit": display_credit,
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
    base_query = db.query(LedgerEntry).filter(
        or_(
            LedgerEntry.payment_mode == "bank",
            LedgerEntry.payment_mode == "upi",
            LedgerEntry.payment_mode == "cheque"
        ),
        LedgerEntry.entry_type.notin_(list(GENERAL_LEDGER_EXCLUDED_TYPES)),
    )

    if start_date:
        base_query = base_query.filter(LedgerEntry.entry_date >= start_date)
    if end_date:
        base_query = base_query.filter(LedgerEntry.entry_date <= end_date)

    total = base_query.count()

    rows = base_query.order_by(
        LedgerEntry.entry_date.desc()
    ).offset((page - 1) * limit).limit(limit).all()

    opening_balance = db.query(
        func.coalesce(func.sum(build_ledger_balance_expression()), 0)
    ).filter(
        or_(
            LedgerEntry.payment_mode == "bank",
            LedgerEntry.payment_mode == "upi",
            LedgerEntry.payment_mode == "cheque"
        ),
        LedgerEntry.entry_type.notin_(list(GENERAL_LEDGER_EXCLUDED_TYPES)),
        LedgerEntry.entry_date < rows[-1].entry_date if rows else None
    ).scalar() or 0

    balance = opening_balance
    data = []

    for r in reversed(rows):
        display_debit, display_credit = get_normalized_ledger_debit_credit(r)
        balance += display_credit - display_debit

        data.insert(0, {
            "id": r.id,
            "date": r.entry_date.isoformat() if r.entry_date else None,
            "description": r.description,
            "debit": display_debit,
            "credit": display_credit,
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
  
  # ================= INCOME =================
  # Total from FINAL invoices only
  query_income = db.query(func.sum(InvoiceModel.total)).filter(
    InvoiceModel.invoice_type == "FINAL"
  )
  
  # ================= EXPENSES =================
  query_expense = db.query(func.sum(ExpenseModel.amount))
  
  # ================= SUPPLIER PAYMENTS =================
  # Sum of all DEBIT entries (money going out to suppliers)
  query_supplier_payment = db.query(
    func.coalesce(func.sum(LedgerEntry.debit), 0)
  ).filter(
    LedgerEntry.entry_type == "payment_out",
    LedgerEntry.supplier_id != None
  )
  
  # Apply date filters
  if start_date:
    query_income = query_income.filter(func.date(InvoiceModel.created_at) >= start_date)
    query_expense = query_expense.filter(ExpenseModel.expense_date >= start_date)
    query_supplier_payment = query_supplier_payment.filter(LedgerEntry.entry_date >= start_date)
  
  if end_date:
    query_income = query_income.filter(func.date(InvoiceModel.created_at) <= end_date)
    query_expense = query_expense.filter(ExpenseModel.expense_date <= end_date)
    query_supplier_payment = query_supplier_payment.filter(LedgerEntry.entry_date <= end_date)
  
  # ================= GET CASH + BANK BALANCE =================
  cash_balance_query = db.query(
    func.coalesce(func.sum(LedgerEntry.credit - LedgerEntry.debit), 0)
  ).filter(LedgerEntry.payment_mode == "cash")
  
  bank_balance_query = db.query(
    func.coalesce(func.sum(LedgerEntry.credit - LedgerEntry.debit), 0)
  ).filter(or_(
    LedgerEntry.payment_mode == "bank",
    LedgerEntry.payment_mode == "upi",
    LedgerEntry.payment_mode == "cheque"
  ))
  
  if start_date:
    cash_balance_query = cash_balance_query.filter(LedgerEntry.entry_date >= start_date)
    bank_balance_query = bank_balance_query.filter(LedgerEntry.entry_date >= start_date)
  
  if end_date:
    cash_balance_query = cash_balance_query.filter(LedgerEntry.entry_date <= end_date)
    bank_balance_query = bank_balance_query.filter(LedgerEntry.entry_date <= end_date)
  
  # ================= CALCULATE =================
  income = query_income.scalar() or 0
  expenses = query_expense.scalar() or 0
  supplier_payments = query_supplier_payment.scalar() or 0
  cash_balance = cash_balance_query.scalar() or 0
  bank_balance = bank_balance_query.scalar() or 0
  
  # Profit = Cash + Bank - Expenses - Supplier Payments
  available_funds = float(cash_balance) + float(bank_balance)
  total_deductions = float(expenses) + float(supplier_payments)
  profit = available_funds - total_deductions
  
  return {
    "income": float(income),
    "expense": float(expenses),
    "supplier_payment": float(supplier_payments),
    "cash_balance": round(float(cash_balance), 2),
    "bank_balance": round(float(bank_balance), 2),
    "available_funds": round(available_funds, 2),
    "profit": round(profit, 2),
    # Overall data
    "overall": {
      "income": float(income),
      "expense": float(expenses),
      "supplier_payment": float(supplier_payments),
      "profit": round(profit, 2)
    }
  }
@api_router.post("/accounts/payment-in")
def payment_in(
    data: PaymentInRequest,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user)
):
    customer = (
        db.query(CustomerModel)
        .filter(CustomerModel.id == data.customer_id)
        .with_for_update()
        .first()
    )

    if not customer:
        raise HTTPException(404, "Customer not found")

    amount = round_currency(data.amount)

    if amount <= 0:
        raise HTTPException(
            400,
            "Payment amount must be greater than 0"
        )

    payment_date = data.payment_date or datetime.now(IST).date()

    # Decrease pending balance
    update_customer_balance(customer, -amount)

    # Handle cheque payment
    if data.payment_mode == "cheque" and data.cheque_number:
        cheque = ChequeModel(
            id=str(uuid.uuid4()),
            cheque_number=data.cheque_number,
            cheque_date=data.cheque_date or payment_date,
            amount=amount,
            bank_name=data.bank_name,
            party_name=customer.name,
            party_type="customer",
            status="pending",
            payment_mode="cheque",
            created_by=current_user.id,
            created_at=datetime.now(IST)
        )
        db.add(cheque)

    # Ledger Entry (money received)
    db.add(
        LedgerEntry(
            id=str(uuid.uuid4()),
            entry_type="payment_in",
            reference_id=data.customer_id,
            customer_id=data.customer_id,
            description=f"Payment received from {customer.name} {data.reference or ''}",
            debit=0,
            credit=amount,
            payment_mode=data.payment_mode,
            entry_date=payment_date,
            created_by=current_user.id,
            created_by_name=current_user.name,
            created_at=datetime.now(IST)
        )
    )

    db.commit()

    return {
        "message": "Payment recorded",
        "new_balance": normalize_customer_balance_value(
            customer.current_balance
        ),
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
    supplier_id=data.supplier_id,  # ✅ Must have supplier_id
    description=f"Payment to {supplier.name} {data.description or ''}",
    debit=data.amount,  # ✅ DEBIT = Money going out
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

    base_query = db.query(LedgerEntry).filter(
        LedgerEntry.entry_type.notin_(list(GENERAL_LEDGER_EXCLUDED_TYPES))
    )

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
    opening_balance_query = db.query(
        func.coalesce(func.sum(build_ledger_balance_expression()), 0)
    ).filter(
        LedgerEntry.entry_type.notin_(list(GENERAL_LEDGER_EXCLUDED_TYPES))
    )

    if start_date:
        opening_balance_query = opening_balance_query.filter(LedgerEntry.entry_date >= start_date)
    if end_date:
        opening_balance_query = opening_balance_query.filter(LedgerEntry.entry_date <= end_date)
    if entry_type:
        opening_balance_query = opening_balance_query.filter(LedgerEntry.entry_type == entry_type)
    if payment_mode:
        opening_balance_query = opening_balance_query.filter(LedgerEntry.payment_mode == payment_mode)
    if customer_id:
        opening_balance_query = opening_balance_query.filter(LedgerEntry.customer_id == customer_id)

    if entries:
        oldest_entry = entries[-1]
        opening_balance_query = opening_balance_query.filter(
            or_(
                LedgerEntry.entry_date < oldest_entry.entry_date,
                and_(
                    LedgerEntry.entry_date == oldest_entry.entry_date,
                    LedgerEntry.created_at < oldest_entry.created_at,
                ),
            )
        )
        opening_balance = opening_balance_query.scalar() or 0
    else:
        opening_balance = 0

    balance = opening_balance
    data = []

    for e in reversed(entries):
        raw_amount = round_currency(max(abs(float(e.debit or 0)), abs(float(e.credit or 0))))
        display_debit, display_credit = get_normalized_ledger_debit_credit(e)
        balance += display_credit - display_debit

        data.insert(0, {
            "id": e.id,
            "date": e.entry_date.isoformat() if e.entry_date else None,
            "type": e.entry_type,
            "description": e.description,
            "amount": raw_amount,
            "debit": display_debit,
            "credit": display_credit,
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

    cancelled_invoice_ids = {
        invoice_id
        for (invoice_id,) in db.query(InvoiceModel.id).filter(
            InvoiceModel.customer_id == customer_id,
            InvoiceModel.invoice_type == "FINAL",
            InvoiceModel.payment_status == "cancelled",
        ).all()
    }

    ledger_query = db.query(LedgerEntry).filter(
        LedgerEntry.customer_id == customer_id,
        LedgerEntry.entry_type.in_(list(CUSTOMER_LEDGER_VISIBLE_TYPES)),
    )
    payment_query = (
        db.query(InvoicePayment, InvoiceModel.invoice_number)
        .join(InvoiceModel, InvoicePayment.invoice_id == InvoiceModel.id)
        .filter(
            InvoiceModel.customer_id == customer_id,
            InvoiceModel.invoice_type == "FINAL",
            InvoiceModel.payment_status != "cancelled",
        )
    )

    if start_date:
        ledger_query = ledger_query.filter(LedgerEntry.entry_date >= start_date)
        payment_query = payment_query.filter(func.date(InvoicePayment.created_at) >= start_date)
    if end_date:
        ledger_query = ledger_query.filter(LedgerEntry.entry_date <= end_date)
        payment_query = payment_query.filter(func.date(InvoicePayment.created_at) <= end_date)

    ledger_entries = ledger_query.order_by(
        LedgerEntry.entry_date.asc(),
        LedgerEntry.created_at.asc(),
    ).all()

    combined_entries = []

    for entry in ledger_entries:
        if entry.reference_id in cancelled_invoice_ids:
            continue

        debit = round_currency(entry.debit)
        credit = round_currency(entry.credit)
        amount = debit if debit > 0 else credit

        if entry.entry_type == "payment_in":
            combined_entries.append(
                {
                    "id": entry.id,
                    "date": entry.entry_date.isoformat() if entry.entry_date else None,
                    "description": entry.description,
                    "debit": 0.0,
                    "credit": 0.0,
                    "amount": amount,
                    "type": "PAYMENT",
                    "mode": entry.payment_mode,
                    "_sort_at": entry.created_at or datetime.now(IST),
                    "_affects_balance": False,
                }
            )
            continue

        combined_entries.append(
            {
                "id": entry.id,
                "date": entry.entry_date.isoformat() if entry.entry_date else None,
                "description": entry.description,
                "debit": debit,
                "credit": credit,
                "amount": amount,
                "type": entry.entry_type,
                "mode": entry.payment_mode,
                "_sort_at": entry.created_at or datetime.now(IST),
                "_affects_balance": True,
            }
        )

    for payment, invoice_number in payment_query.order_by(InvoicePayment.created_at.asc()).all():
        payment_date = (
            payment.created_at.astimezone(IST).date()
            if getattr(payment.created_at, "tzinfo", None)
            else payment.created_at.date()
        )

        combined_entries.append(
            {
                "id": f"payment-{payment.id}",
                "date": payment_date.isoformat(),
                "description": f"Payment received for invoice {invoice_number}",
                "debit": 0.0,
                "credit": 0.0,
                "amount": round_currency(payment.amount),
                "type": "PAYMENT",
                "mode": payment.payment_mode,
                "_sort_at": payment.created_at,
                "_affects_balance": False,
            }
        )

    combined_entries.sort(
        key=lambda entry: (
            entry["date"] or "",
            entry["_sort_at"],
            entry["id"],
        )
    )

    balance = round_currency(customer.opening_balance)
    total_debit = 0.0
    total_credit = 0.0
    ledger_rows = []

    for entry in combined_entries:
        total_debit += entry["debit"]
        total_credit += entry["credit"]

        if entry["_affects_balance"]:
            balance = round_currency(balance + entry["debit"] - entry["credit"])

        ledger_rows.append(
            {
                "id": entry["id"],
                "date": entry["date"],
                "description": entry["description"],
                "debit": entry["debit"],
                "credit": entry["credit"],
                "amount": entry["amount"],
                "balance": balance,
                "display_balance": get_customer_display_balance(balance),
                "type": entry["type"],
                "mode": entry["mode"],
            }
        )

    total = len(ledger_rows)
    start_index = max((page - 1) * limit, 0)
    end_index = start_index + limit
    result = ledger_rows[start_index:end_index]

    customer_invoices = (
        db.query(InvoiceModel)
        .filter(
            InvoiceModel.customer_id == customer_id,
            InvoiceModel.invoice_type == "FINAL",
        )
        .order_by(InvoiceModel.created_at.desc())
        .all()
    )

    invoice_ids = [invoice.id for invoice in customer_invoices]
    advance_used_map = {}

    if invoice_ids:
        advance_rows = (
            db.query(
                LedgerEntry.reference_id,
                func.coalesce(func.sum(LedgerEntry.debit), 0),
            )
            .filter(
                LedgerEntry.reference_id.in_(invoice_ids),
                LedgerEntry.entry_type == "advance_used",
            )
            .group_by(LedgerEntry.reference_id)
            .all()
        )
        advance_used_map = {
            reference_id: round_currency(amount)
            for reference_id, amount in advance_rows
        }

    serialized_invoices = [
        serialize_invoice_summary(invoice, advance_used_map.get(invoice.id, 0))
        for invoice in customer_invoices
    ]

    item_ledger = []
    for invoice in serialized_invoices:
        for index, item in enumerate(invoice.get("items") or []):
            variant = item.get("variant_info") or {}
            item_ledger.append(
                {
                    "id": f"{invoice['id']}-{index}",
                    "invoice_id": invoice["id"],
                    "invoice_number": invoice["invoice_number"],
                    "created_at": invoice["created_at"],
                    "payment_status": invoice["payment_status"],
                    "product_name": item.get("product_name"),
                    "sku": item.get("v_sku") or item.get("sku") or variant.get("v_sku"),
                    "variant_name": item.get("variant_name") or variant.get("variant_name"),
                    "color": item.get("color") or variant.get("color"),
                    "size": item.get("size") or variant.get("size"),
                    "quantity": int(item.get("quantity") or 0),
                    "price": round_currency(item.get("price")),
                    "total": round_currency(item.get("total")),
                }
            )

    balance_snapshot = get_customer_balance_snapshot(db, customer)

    return {
        "customer": {
            "id": customer.id,
            "name": customer.name,
            "phone": customer.phone,
            "email": customer.email,
            "address": customer.address,
            "opening_balance": round_currency(customer.opening_balance),
            "current_balance": balance_snapshot["current_balance"],
            "display_balance": balance_snapshot["display_balance"],
            "pending_balance": balance_snapshot["pending_balance"],
            "advance_balance": balance_snapshot["advance_balance"],
            "pending_invoice_amount": balance_snapshot["pending_invoice_amount"],
            "stored_current_balance": balance_snapshot["stored_current_balance"],
            "can_add_advance": balance_snapshot["can_add_advance"],
            "total_invoices": len(serialized_invoices),
            "total_bill": round_currency(
                sum(
                    float(invoice.get("total") or 0)
                    for invoice in serialized_invoices
                    if invoice.get("payment_status") != "cancelled"
                )
            ),
        },
        "data": result,
        "invoices": serialized_invoices,
        "item_ledger": item_ledger,
        "total_debit": round_currency(total_debit),
        "total_credit": round_currency(total_credit),
        "pagination": {
            "page": page,
            "limit": limit,
            "total": total,
            "total_pages": math.ceil(total / limit) if total else 1
        }
    }


@api_router.get("/accounts/supplier-ledger")
def supplier_ledger_overview(
    page: int = 1,
    limit: int = 50,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    supplier_id: Optional[str] = None,
    search: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin only - Supplier ledger access restricted")

    supplier_query = db.query(SupplierModel)

    if supplier_id:
        supplier_query = supplier_query.filter(SupplierModel.id == supplier_id)

    if search:
        search_term = f"%{search.strip().lower()}%"
        supplier_query = supplier_query.filter(
            or_(
                func.lower(SupplierModel.name).like(search_term),
                func.lower(SupplierModel.phone).like(search_term),
                func.lower(SupplierModel.email).like(search_term),
            )
        )

    suppliers = supplier_query.order_by(SupplierModel.name.asc()).all()

    rows = []
    total_bill = 0.0
    paid_amount = 0.0
    pending_amount = 0.0
    total_debit = 0.0
    total_credit = 0.0

    for supplier in suppliers:
        supplier_snapshot = build_supplier_ledger_snapshot(
            db,
            supplier,
            start_date=start_date,
            end_date=end_date,
        )
        supplier_summary = get_supplier_invoice_summary(db, supplier.id)

        total_bill += float(supplier_summary["total_bill"] or 0)
        paid_amount += float(supplier_summary["paid_amount"] or 0)
        pending_amount += float(supplier_summary["pending_amount"] or 0)
        total_debit += float(supplier_snapshot["total_debit"] or 0)
        total_credit += float(supplier_snapshot["total_credit"] or 0)

        for row in supplier_snapshot["ledger_rows"]:
            rows.append(
                {
                    **row,
                    "supplier_id": supplier.id,
                    "supplier_name": supplier.name,
                    "supplier_phone": supplier.phone,
                    "supplier_email": supplier.email,
                    "supplier_balance": row["balance"],
                }
            )

    rows.sort(
        key=lambda item: (
            item.get("date") or "",
            item.get("created_at") or "",
            item.get("supplier_name") or "",
            item.get("id") or "",
        ),
        reverse=True,
    )

    total = len(rows)
    start_index = max((page - 1) * limit, 0)
    end_index = start_index + limit

    return {
        "data": rows[start_index:end_index],
        "summary": {
            "supplier_count": len({row["supplier_id"] for row in rows}),
            "total_bill": round_currency(total_bill),
            "paid_amount": round_currency(paid_amount),
            "pending_amount": round_currency(pending_amount),
            "total_debit": round_currency(total_debit),
            "total_credit": round_currency(total_credit),
            "net_balance": round_currency(total_debit - total_credit),
        },
        "pagination": {
            "page": page,
            "limit": limit,
            "total": total,
            "total_pages": math.ceil(total / limit) if total else 1,
        }
    }


@api_router.get("/accounts/supplier-ledger/{supplier_id}")
def supplier_ledger(
    supplier_id: str,
    page: int = 1,
    limit: int = 50,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    supplier = db.query(SupplierModel).filter(SupplierModel.id == supplier_id).first()
    if not supplier:
        raise HTTPException(404, "Supplier not found")

    ledger_snapshot = build_supplier_ledger_snapshot(
        db,
        supplier,
        start_date=start_date,
        end_date=end_date,
    )
    summary = get_supplier_invoice_summary(db, supplier.id)

    total = len(ledger_snapshot["ledger_rows"])
    start_index = max((page - 1) * limit, 0)
    end_index = start_index + limit

    return {
        "supplier": {
            "id": supplier.id,
            "name": supplier.name,
            "phone": supplier.phone,
            "email": supplier.email,
            "address": supplier.address,
            "opening_balance": round_currency(supplier.opening_balance),
            "current_balance": round_currency(supplier.current_balance),
            "computed_balance": ledger_snapshot["computed_balance"],
            "total_bill": summary["total_bill"],
            "paid_amount": summary["paid_amount"],
            "pending_amount": summary["pending_amount"],
            "created_at": supplier.created_at.isoformat() if supplier.created_at else None,
        },
        "data": ledger_snapshot["ledger_rows"][start_index:end_index],
        "invoices": ledger_snapshot["invoices"],
        "total_debit": ledger_snapshot["total_debit"],
        "total_credit": ledger_snapshot["total_credit"],
        "pagination": {
            "page": page,
            "limit": limit,
            "total": total,
            "total_pages": math.ceil(total / limit) if total else 1,
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
    query = db.query(LedgerEntry).filter(
        LedgerEntry.entry_type.notin_(list(GENERAL_LEDGER_EXCLUDED_TYPES))
    )

    if start_date:
        query = query.filter(LedgerEntry.entry_date >= start_date)
    if end_date:
        query = query.filter(LedgerEntry.entry_date <= end_date)

    entries = query.order_by(LedgerEntry.entry_date.asc()).all()

    output = StringIO()
    writer = csv.writer(output)
    
    writer.writerow(["Date", "Type", "Description", "Amount", "Debit", "Credit", "Payment Mode", "Created By"])
    
    for e in entries:
        raw_amount = round_currency(max(abs(float(e.debit or 0)), abs(float(e.credit or 0))))
        display_debit = 0.0 if e.entry_type in GENERAL_LEDGER_TRACKING_TYPES else round_currency(e.debit)
        display_credit = 0.0 if e.entry_type in GENERAL_LEDGER_TRACKING_TYPES else round_currency(e.credit)

        writer.writerow([
            e.entry_date.isoformat() if e.entry_date else "",
            e.entry_type,
            e.description,
            raw_amount,
            display_debit,
            display_credit,
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

  
    
    
