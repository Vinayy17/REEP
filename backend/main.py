import ast
from math import ceil
from fastapi import Query
from itertools import product
from fastapi import FastAPI, APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, String, Float, Integer, Text, ForeignKey, DateTime, case, or_, func
from sqlalchemy.orm import sessionmaker, declarative_base, Session, relationship
from sqlalchemy.orm.attributes import flag_modified
import os
import logging
from sqlalchemy import or_, func

import json
from pathlib import Path
from copy import deepcopy
from pydantic import BaseModel, EmailStr
from typing import List, Optional
import uuid
import random
import string
from datetime import datetime, timezone, timedelta
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

class CategoryModel(Base):
    __tablename__ = "categories"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
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

class CustomerModel(Base):
    __tablename__ = "customers"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String(255), nullable=False)
    email = Column(String(255), nullable=False)
    phone = Column(String(50), nullable=True)
    address = Column(Text, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(IST))
    
    invoices = relationship("InvoiceModel", back_populates="customer")

class InvoiceModel(Base):
    __tablename__ = "invoices"
    
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    invoice_number = Column(String(50), nullable=False, unique=True, index=True)
    customer_id = Column(String(36), ForeignKey("customers.id"), nullable=False)
    customer_name = Column(String(255), nullable=False)
    customer_phone = Column(String(50), nullable=True)
    customer_address = Column(Text, nullable=True)
    items = Column(Text, nullable=False)  # Store as JSON string
    subtotal = Column(Float, nullable=False)
    additional_amount = Column(Float, nullable=False, default=0)  # ✅ For manual charges
    additional_label = Column(String(255), nullable=True)  # ✅ For charge description
    gst_amount = Column(Float, nullable=False, default=0)
    created_by = Column(String(36))          # ✅ ADD THIS

    gst_enabled = Column(Integer, default=1)  # 1 or 0
    gst_rate = Column(Float, default=0)
    created_by_name = Column(String(100))         # ✅ ADD THIS

    discount = Column(Float, nullable=False, default=0)
    total = Column(Float, nullable=False)
    payment_status = Column(String(50), nullable=False, default="pending")
    created_at = Column(DateTime, default=lambda: datetime.now(IST))
    customer = relationship("CustomerModel", back_populates="invoices") # FIX: Should be back_populates="invoices" if relationship is defined in CustomerModel

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

    images: Optional[List[str]] = []
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
    images: List[str] = [] # Default to empty list
    is_service: int = 0
    variants: List[VariantSchema] = [] # Default to empty list
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
    product_id: str
    product_name: str
    quantity: int
    price: float # This will be the price per unit for the item
    gst_rate: float
    total: float
    sku: Optional[str] = None # Variant SKU or product SKU
    is_service: int = 0        # ✅ ADD THIS

    variant_info: Optional[dict] = None # Added for variant details
    image_url: Optional[str] = None # Added for item image


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
    customer_email: str
    customer_address: Optional[str] = None
    items: List[InvoiceItem]

    gst_enabled: bool = True
    gst_rate: float = 0

    gst_amount: float = 0
    discount: float = 0
    payment_status: str = "pending"
    additional_amount: float = 0  # ✅ ADD for manual charges
    additional_label: Optional[str] = None  # ✅ ADD for charge label


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
    overdue: float

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

@api_router.post("/upload/variant-image")
def upload_variant_image(
    file: UploadFile = File(...),
    current_user: UserModel = Depends(get_current_user)
):
    

    url = upload_image_to_cloudinary(file.file, folder="variant_images")
    return {"url": url}
@api_router.post("/upload/requirement-image")
def upload_requirement_image(
    file: UploadFile = File(...),
    current_user: UserModel = Depends(get_current_user)
):
    # 🔐 Allow admin & store handler
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
    # ✅ Mark as completed only
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
    req.completed_at = datetime.now(IST)   # ✅ ADD THIS

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
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    offset = (page - 1) * limit

    query = db.query(ProductModel).filter(ProductModel.is_active == 1)

    # 🔍 SEARCH FILTER (GLOBAL)
    if search:
        s = f"%{search.lower().strip()}%"
        query = query.filter(
            or_(
                func.lower(ProductModel.name).like(s),
                func.lower(ProductModel.sku).like(s),
                func.lower(ProductModel.product_code).like(s),
            )
        )

    # 📂 CATEGORY FILTER
    if category_id and category_id != "all":
        query = query.filter(ProductModel.category_id == category_id)

    total = query.count()

    products = (
        query
        .order_by(ProductModel.created_at.desc())
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

@api_router.get("/products/ageing")
def product_ageing_report(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=50),
    bucket: Optional[str] = Query(None, pattern="^(daily|weekly|monthly)$"),

    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    # ✅ USE NAIVE DATETIME (IMPORTANT)
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
        # 🔹 FIRST INWARD DATE
        first_inward = (
            db.query(func.min(InventoryTransaction.created_at))
            .filter(
                InventoryTransaction.product_id == p.id,
                InventoryTransaction.type == "IN"
            )
            .scalar()
        )

        # ✅ NORMALIZE TO NAIVE
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

    # 🚨 ENFORCE VARIANT-ONLY IF VARIANTS EXIST
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

        # ✅ ALWAYS derive product stock from variants
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
    created_by_name=current_user.name,   # ✅ ADD THIS
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

    # ✅ FIRST resolve product
    product, variant_sku = resolve_product_and_variant_by_sku(db, request.sku)

    # ✅ NOW product exists → safe check
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

    # 🚨 ENFORCE VARIANT-ONLY IF VARIANTS EXIST
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

        # ✅ ALWAYS derive product stock from variants
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
            created_by_name=current_user.name,   # ✅ ADD

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
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    offset = (page - 1) * limit

    q = (
        db.query(InventoryTransaction, ProductModel)
        .join(ProductModel, InventoryTransaction.product_id == ProductModel.id)
    )

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
                "created_by": txn.created_by_name,   # ✅ ADD

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
    # 🔐 NON-ADMIN → FORCE COST PRICE = 0
    if current_user.role != "admin":
        product_data.cost_price = 0

    # 🔍 VALIDATIONS
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
    # 🔄 CHECK EXISTING SKU (ACTIVE OR ARCHIVED)
    # ==========================================================
    existing = db.query(ProductModel).filter(
        ProductModel.sku == parent_sku
    ).first()

    # ==========================================================
    # 🔄 RESTORE ARCHIVED PRODUCT
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

            # ✅ COST PRICE → ADMIN ONLY
            if current_user.role == "admin":
                existing.cost_price = product_data.cost_price

            # 🔒 ERP RULE: NO STOCK DURING RESTORE
            if product_data.is_service == 1:
                existing.variants = []
                existing.stock = 0
            else:
                restored_variants = []
                for v in product_data.variants or []:
                    vd = v.dict()
                    vd["stock"] = 0  # 🔒 FORCE ZERO
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

        # ❌ ACTIVE SKU EXISTS
        raise HTTPException(409, "SKU already exists")

    # ==========================================================
    # 🆕 CREATE NEW PRODUCT
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

    # 📦 PRODUCT QR
    qr_code_url = generate_qr({
        "type": "product",
        "name": product_data.name,
        "sku": parent_sku,
        "price": product_data.selling_price,
    })

    # 📦 VARIANT QR
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
    # ❌ DO NOT CHANGE STOCK
    # ❌ DO NOT RECALCULATE STOCK

    if product_data.is_service == 1:
        product.variants = []
    else:
        existing_variants = {v["v_sku"]: v for v in (product.variants or [])}
        updated_variants = []

        for v in product_data.variants or []:
            vd = v.dict()
            old = existing_variants.get(vd.get("v_sku"), {})

            vd["stock"] = old.get("stock", 0)   # 🔒 PRESERVE STOCK

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
        # ✅ HARD DELETE
        db.delete(product)
        db.commit()
        return {
            "message": "Product deleted permanently",
            "action": "hard_delete"
        }

    # ❌ USED → ARCHIVE
    product.is_active = 0
    db.commit()
    return {
        "message": "Product archived (used in inventory/invoices)",
        "action": "archived"
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
            InvoiceModel.customer_id == CustomerModel.id
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
            "created_at": c.created_at.isoformat(),
            "total_invoices": int(total_invoices),
            "total_bill": float(total_bill),
        }
        for c, total_invoices, total_bill in customers
    ]


def generate_invoice_number(db: Session):
    now = datetime.now(IST) # Changed to IST
    fy_year = now.year if now.month >= 4 else now.year - 1
    fy_suffix = f"{fy_year % 100:02d}-{(fy_year + 1) % 100:02d}"
    
    # Count invoices in current FY
    start_date = datetime(fy_year, 4, 1, tzinfo=IST) # Changed to IST
    count = db.query(InvoiceModel).filter(
        InvoiceModel.created_at >= start_date
    ).count()
    
    return f"INV-{fy_suffix}-{count + 1:04d}"

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
    range: Optional[str] = None,    # last10 | last30
    month: Optional[str] = None,    # YYYY-MM
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    query = db.query(InvoiceModel)

    # ================= TIME SETUP (CRITICAL FIX) =================
    now = datetime.now(IST) # Changed to IST

    start_of_today = datetime(
        now.year, now.month, now.day,
        tzinfo=IST # Changed to IST
    )

    end_of_today = start_of_today + timedelta(days=1)

    # ================= STATUS FILTER =================
    if status == "paid":
        query = query.filter(InvoiceModel.payment_status == "paid")

    elif status == "cancelled":
        query = query.filter(InvoiceModel.payment_status == "cancelled")

    elif status == "overdue":
        query = query.filter(
            InvoiceModel.payment_status != "paid",
            InvoiceModel.created_at < start_of_today # Changed to created_at for overdue check
        )

    elif status == "ending":
        query = query.filter(
            InvoiceModel.payment_status != "paid",
            InvoiceModel.created_at.between( # Changed to created_at for ending check
                start_of_today,
                start_of_today + timedelta(days=5)
            )
        )

    # ================= DATE RANGE FILTER =================
    if range == "last10":
        query = query.filter(
            InvoiceModel.created_at >= start_of_today - timedelta(days=9)
        )

    elif range == "last30":
        query = query.filter(
            InvoiceModel.created_at >= start_of_today - timedelta(days=29)
        )

    # ================= MONTH FILTER =================
    if month:
        year, month_num = map(int, month.split("-"))

        start_date = datetime(
            year, month_num, 1, 0, 0, 0,
            tzinfo=IST # Changed to IST
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
                "additional_amount": inv.additional_amount,  # ✅ INCLUDE ADDITIONAL AMOUNT
                "additional_label": inv.additional_label,  # ✅ INCLUDE LABEL
                "gst_amount": inv.gst_amount,
                "gst_enabled": bool(inv.gst_enabled),
                "gst_rate": inv.gst_rate,

                "discount": inv.discount,
                "total": inv.total,
                "payment_status": inv.payment_status,
                "created_at": inv.created_at.isoformat(),
                "created_by": inv.created_by_name,   # ✅ ADD

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
@api_router.post(
    "/invoices",
    dependencies=[Depends(RateLimiter(times=20, seconds=60))]
)
def create_invoice(
    invoice_data: InvoiceCreate,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
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
            or f"{invoice_data.customer_phone}@example.com",
            phone=invoice_data.customer_phone,
            address=invoice_data.customer_address,
            created_at=datetime.now(IST),
        )
        db.add(customer)
        db.flush()

    # ================= ITEMS =================
    invoice_items = []
    subtotal = 0.0
    
    # ================= ADDITIONAL AMOUNT (MANUAL CHARGE) =================
    additional_amount = float(invoice_data.additional_amount or 0)

    for item in invoice_data.items:

        # =====================================================
        # 🟢 SERVICE / ADDITIONAL CHARGE (NO INVENTORY)
        # =====================================================
        if item.is_service == 1 or not item.product_id or item.product_id == "SERVICE":
            line_total = float(item.price) * int(item.quantity)
            subtotal += line_total

            invoice_items.append({
                "product_id": None,
                "sku": item.sku or "SERVICE",
                "product_name": item.product_name,
                "quantity": int(item.quantity),
                "gst_rate": float(item.gst_rate),
                "price": float(item.price),
                "total": line_total,
                "is_service": 1,
                "variant_info": None,
                "image_url": None,
            })
            continue

        # =====================================================
        # 🔵 NORMAL PRODUCT
        # =====================================================
        product = (
            db.query(ProductModel)
            .filter(ProductModel.id == item.product_id)
            .with_for_update()
            .first()
        )

        if not product:
            raise HTTPException(404, "Product not found")

        if product.is_active == 0:
            raise HTTPException(
                status_code=400,
                detail=f"Product '{product.name}' is archived and cannot be sold",
            )

        price = float(item.price)
        quantity = int(item.quantity)
        line_total = price * quantity
        subtotal += line_total

        # ================= DB SERVICE PRODUCT =================
        if product.is_service == 1:
            invoice_items.append({
                "product_id": product.id,
                "sku": item.sku or product.sku,
                "product_name": product.name,
                "quantity": quantity,
                "gst_rate": float(item.gst_rate),
                "price": price,
                "total": line_total,
                "is_service": 1,
                "variant_info": None,
                "image_url": None,
            })
            continue

        # ================= INVENTORY =================
        variants = list(product.variants or [])
        variant_details = None

        # ---------- PRODUCT WITH VARIANTS ----------
        if variants:
            if not item.sku:
                raise HTTPException(
                    400,
                    f"Product {product.name} has variants. Variant SKU required."
                )

            total_before = calculate_total_stock(variants)
            variant_stock_after = None

            for v in variants:
                if v.get("v_sku") == item.sku:
                    current_stock = int(v.get("stock", 0))
                    if current_stock < quantity:
                        raise HTTPException(
                            400,
                            f"Insufficient stock for variant {item.sku}. Available: {current_stock}",
                        )

                    v["stock"] = current_stock - quantity
                    variant_stock_after = v["stock"]

                    variant_details = {
                        "v_sku": v.get("v_sku"),
                        "variant_name": v.get("variant_name"),
                        "color": v.get("color"),
                        "size": v.get("size"),
                        "v_image_url": v.get("image_url"),
                    }
                    break
            else:
                raise HTTPException(400, f"Variant SKU {item.sku} not found")

            product.variants = variants
            product.stock = calculate_total_stock(variants)
            flag_modified(product, "variants")
            flag_modified(product, "stock")
            db.flush()

            db.add(
                InventoryTransaction(
                    id=str(uuid.uuid4()),
                    product_id=product.id,
                    type="OUT",
                    quantity=quantity,
                    source="INVOICE",
                    reason="Invoice",
                    stock_before=total_before,
                    stock_after=product.stock,
                    variant_sku=item.sku,
                    variant_stock_after=variant_stock_after,
                    created_by=current_user.id,
                    created_by_name=current_user.name,
                    created_at=datetime.now(IST),
                )
            )

        # ---------- PRODUCT WITHOUT VARIANTS ----------
        else:
            stock_before = int(product.stock or 0)
            if stock_before < quantity:
                raise HTTPException(
                    400,
                    f"Insufficient stock for {product.name}. Available: {stock_before}",
                )

            product.stock = stock_before - quantity
            flag_modified(product, "stock")
            db.flush()

            db.add(
                InventoryTransaction(
                    id=str(uuid.uuid4()),
                    product_id=product.id,
                    type="OUT",
                    quantity=quantity,
                    source="INVOICE",
                    reason="Invoice",
                    stock_before=stock_before,
                    stock_after=product.stock,
                    created_by=current_user.id,
                    created_by_name=current_user.name,
                    created_at=datetime.now(IST),
                )
            )

        invoice_items.append({
            "product_id": product.id,
            "sku": item.sku or product.sku,
            "product_name": product.name,
            "variant_info": variant_details,
            "image_url": (
                variant_details.get("v_image_url")
                if variant_details
                else (product.images[0] if product.images else None)
            ),
            "quantity": quantity,
            "gst_rate": float(item.gst_rate),
            "price": price,
            "total": line_total,
            "is_service": 0,
        })

    # ================= TOTALS =================
    # Add additional amount to subtotal before GST calculation
    taxable_subtotal = subtotal + additional_amount
    
    gst_amount = (
        round((taxable_subtotal * invoice_data.gst_rate) / 100, 2)
        if invoice_data.gst_enabled
        else 0
    )

    total = round(taxable_subtotal + gst_amount - invoice_data.discount, 2)
    invoice_number = generate_invoice_number(db)

    invoice = InvoiceModel(
        id=str(uuid.uuid4()),
        invoice_number=invoice_number,
        customer_id=customer.id,
        customer_name=customer.name,
        customer_phone=customer.phone,
        customer_address=customer.address,
        items=json.dumps(invoice_items),
        subtotal=subtotal,
        additional_amount=additional_amount,  # ✅ STORE ADDITIONAL AMOUNT
        additional_label=invoice_data.additional_label,  # ✅ STORE LABEL
        gst_amount=gst_amount,
        gst_rate=invoice_data.gst_rate,
        gst_enabled=1 if invoice_data.gst_enabled else 0,
        discount=invoice_data.discount,
        total=total,
        payment_status=invoice_data.payment_status,
        created_by=current_user.id,
        created_by_name=current_user.name,
        created_at=datetime.now(IST),
    )

    db.add(invoice)
    db.commit()
    db.refresh(invoice)

    return {
        "id": invoice.id,
        "invoice_number": invoice.invoice_number,
        "total": invoice.total,
        "created_at": invoice.created_at,
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
        ProductModel.is_active == 1   # ✅ ADD
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
        ProductModel.is_active == 1,   # ✅ ADD
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
    
    invoice.payment_status = payment_status
    db.commit()
    return {"message": "Invoice status updated successfully"}

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
        ProductModel.is_service == 0,             # ✅ EXCLUDE SERVICES
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

    invoices_today = db.query(InvoiceModel).filter(
        InvoiceModel.created_at >= start
    ).count()

    items_sold_today = db.query(
        func.coalesce(func.sum(InventoryTransaction.quantity), 0)
    ).filter(
        InventoryTransaction.type == "OUT",
        InventoryTransaction.created_at >= start
    ).scalar()

    new_customers = db.query(CustomerModel).filter(
        CustomerModel.created_at >= start
    ).count()

    inventory_out = db.query(InventoryTransaction).filter(
        InventoryTransaction.type == "OUT",
        InventoryTransaction.created_at >= start
    ).count()

    return {
        "invoices_today": invoices_today,
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
            ProductModel.is_service == 0,         # ✅ EXCLUDE SERVICES
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
            func.sum(case((InvoiceModel.payment_status == "overdue", InvoiceModel.total), else_=0)).label("overdue"),
        )
        .filter(
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
            overdue=float(row.overdue or 0),
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

app.include_router(api_router)
ALLOWED_ORIGINS = [
    "https://rridegarage.com",
    "https://www.rridegarage.com",
]
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "Accept"
    ],
)
