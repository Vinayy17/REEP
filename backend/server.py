import ast
from fastapi import FastAPI, APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, String, Float, Integer, Text, ForeignKey, DateTime, case
from sqlalchemy.orm import sessionmaker, declarative_base, Session, relationship
from sqlalchemy.orm.attributes import flag_modified
import os
import logging
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
from sqlalchemy import func
import math
import qrcode
from fastapi.staticfiles import StaticFiles
from sqlalchemy import JSON
import cloudinary
import cloudinary.uploader
from fastapi import UploadFile, File
from sqlalchemy.exc import OperationalError
import time


IST = timezone(timedelta(hours=5, minutes=30))

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Database Configuration
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "mysql+pymysql://root:143%40Vinay@localhost/chinaligths")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
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
    created_at = Column(DateTime, default=datetime.utcnow)

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
    cost_price = Column(Float, nullable=False)
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

    created_at = Column(DateTime, default=datetime.utcnow)  # ✅ HERE


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

    stock_before = Column(Integer, nullable=False, default=0)
    stock_after = Column(Integer, nullable=False, default=0)
    
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
    gst_amount = Column(Float, nullable=False, default=0)
    discount = Column(Float, nullable=False, default=0)
    total = Column(Float, nullable=False)
    payment_status = Column(String(50), nullable=False, default="pending")
    created_at = Column(DateTime, default=lambda: datetime.now(IST))
    customer = relationship("CustomerModel", back_populates="invoices") # FIX: Should be back_populates="invoices" if relationship is defined in CustomerModel

# Create tables
Base.metadata.create_all(bind=engine)

app = FastAPI()
# app.mount("/static", StaticFiles(directory="static"), name="static")

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

def generate_product_code():
    date_part = datetime.now(IST).strftime("%Y%m%d")
    rand = "".join(random.choices(string.ascii_uppercase + string.digits, k=4))
    return f"PRD-{date_part}-{rand}"

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

@api_router.post("/upload/product-image")
def upload_product_image(
    file: UploadFile = File(...),
    current_user: UserModel = Depends(get_current_user)
):
    # 🔐 Only admin
    if current_user.role != "admin":
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
    gst_amount: float = 0
    discount: float = 0
    payment_status: str = "pending"

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
    variant_sku: Optional[str]
    stock_after: int          # ✅ use this, not remaining_stock
    created_at: str

def resolve_product_and_variant_by_sku(db: Session, sku: str):
    sku = sku.strip()

    # 1️⃣ Parent SKU or Product Code
    product = db.query(ProductModel).filter(
        (ProductModel.sku == sku) |
        (ProductModel.product_code == sku)
    ).first()

    if product:
        return product, None   # ✅ PRODUCT LEVEL

    # 2️⃣ Variant SKU
    products = db.query(ProductModel).all()
    for p in products:
        for v in (p.variants or []):
            if v.get("v_sku") == sku:
                return p, sku   # ✅ VARIANT LEVEL

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

# ================= LOGIN =================
@api_router.post("/auth/login", response_model=Token)
def login(
    user_data: UserLogin,
    db: Session = Depends(get_db)
):
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
    if current_user.role != "admin":
        raise HTTPException(403, "Admin only")

    url = upload_image_to_cloudinary(file.file, folder="variant_images")
    return {"url": url}

@api_router.get("/products")
def get_products(
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    products = db.query(ProductModel).all()
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
            "variants": p.variants,
            "images": safe_images(p.images),
            "qr_code_url": p.qr_code_url,
            "is_service": p.is_service,
            "created_at": p.created_at.isoformat()
        }

        if current_user.role == "admin":
            item["cost_price"] = p.cost_price

        result.append(item)

    return result
@api_router.get("/inventory/lookup/{sku}")
def inventory_lookup(
    sku: str,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    product, variant_sku = resolve_product_and_variant_by_sku(db, sku)

    if variant_sku:
        for v in product.variants or []:
            if v.get("v_sku") == variant_sku:
                return {
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

    return {
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
            for v in (product.variants or [])
        ],
        "total_stock": product.stock
    }
@api_router.post("/inventory/material-inward/sku")
def material_inward_by_sku(
    request: MaterialInwardBySkuRequest,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if request.quantity <= 0:
        raise HTTPException(400, "Quantity must be > 0")

    product, variant_sku = resolve_product_and_variant_by_sku(db, request.sku)

    if product.is_service == 1:
        raise HTTPException(400, "Inventory not allowed for services")

    variants = deepcopy(product.variants or [])

    # ================= VARIANT EXISTS =================
    if variant_sku:
        total_before = calculate_total_stock(variants)
        found = False

        for v in variants:
            if v["v_sku"] == variant_sku:
                v["stock"] = (v.get("stock") or 0) + request.quantity
                found = True
                break

        if not found:
            # ⚠️ VARIANT SKU NOT FOUND → PRODUCT LEVEL
            product.stock += request.quantity
            stock_before = product.stock - request.quantity
            stock_after = product.stock
            variant_sku = None
        else:
            product.variants = variants
            product.stock = calculate_total_stock(variants)
            stock_before = total_before
            stock_after = product.stock

    # ================= PRODUCT LEVEL =================
    else:
        stock_before = product.stock
        product.stock += request.quantity
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
        created_by=current_user.id,
        created_at=datetime.now(IST)
    ))

    db.commit()

    return {
        "message": "Stock added successfully",
        "variant_sku": variant_sku,
        "stock_after": stock_after
    }

@api_router.post("/inventory/material-outward/sku")
def material_outward_by_sku(
    request: MaterialOutwardBySkuRequest,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if request.quantity <= 0:
        raise HTTPException(400, "Quantity must be > 0")

    product, variant_sku = resolve_product_and_variant_by_sku(db, request.sku)

    if product.is_service == 1:
        raise HTTPException(400, "Inventory not allowed for services")

    variants = deepcopy(product.variants or [])

    # ================= VARIANT EXISTS =================
    if variant_sku:
        total_before = calculate_total_stock(variants)
        found = False

        for v in variants:
            if v["v_sku"] == variant_sku:
                if v["stock"] < request.quantity:
                    raise HTTPException(400, "Insufficient variant stock")
                v["stock"] -= request.quantity
                found = True
                break

        if not found:
            # ⚠️ VARIANT NOT FOUND → PRODUCT LEVEL
            if product.stock < request.quantity:
                raise HTTPException(400, "Insufficient product stock")

            stock_before = product.stock
            product.stock -= request.quantity
            stock_after = product.stock
            variant_sku = None
        else:
            product.variants = variants
            product.stock = calculate_total_stock(variants)
            stock_before = total_before
            stock_after = product.stock

    # ================= PRODUCT LEVEL =================
    else:
        if product.stock < request.quantity:
            raise HTTPException(400, "Insufficient product stock")

        stock_before = product.stock
        product.stock -= request.quantity
        stock_after = product.stock

    db.add(InventoryTransaction(
        id=str(uuid.uuid4()),
        product_id=product.id,
        type="OUT",
        quantity=request.quantity,
        source="MATERIAL_OUTWARD",
        reason=request.reason,
        stock_before=stock_before,
        stock_after=stock_after,
        variant_sku=variant_sku,
        created_by=current_user.id,
        created_at=datetime.now(IST)
    ))

    db.commit()

    return {
        "message": "Stock deducted successfully",
        "variant_sku": variant_sku,
        "stock_after": stock_after
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
            InventoryTransactionResponse(
    id=txn.id,
    product_id=txn.product_id,
    product_name=prod.name,
    product_code=prod.product_code,
    type=txn.type,
    quantity=txn.quantity,
    variant_sku=txn.variant_sku,
    stock_after=txn.stock_after,   # ✅ matches model
    created_at=txn.created_at.isoformat()
)

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
    if current_user.role != "admin":
        raise HTTPException(403, "Admin only")

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

    if db.query(ProductModel).filter(ProductModel.sku == parent_sku).first():
        raise HTTPException(409, "SKU already exists")

    product_code = generate_product_code()

    # ================= SERVICE VS PRODUCT =================
    if product_data.is_service == 1:
        variants = []
        total_stock = 0
        min_stock = 0
    else:
        # For tangible products, variants are optional:
        # - if present → stock is managed at variant level
        # - if empty   → stock is managed at product level via inventory
        # If variants list exists and has items, use them. Otherwise, default to empty list.
        if product_data.variants and len(product_data.variants) > 0:
            variants = [v.dict() for v in product_data.variants]

            v_skus = [v["v_sku"] for v in variants if v.get("v_sku")]
            if len(v_skus) != len(set(v_skus)):
                raise HTTPException(400, "Duplicate variant SKU")

            total_stock = sum(v.get("stock", 0) for v in variants)
        else:
            variants = []
            # For products without variants, stock can be updated later via inventory
            total_stock = 0

        min_stock = product_data.min_stock

    # Product-level QR
    qr_code_url = generate_qr({
        "type": "product",
        "name": product_data.name,
        "sku": parent_sku,
        "price": product_data.selling_price,
    })

    # Variant-level QR (if any)
    enriched_variants = []
    for v in variants:
        variant_qr = generate_qr({
            "type": "variant",
            "product_name": product_data.name,
            "sku": parent_sku,
            "v_sku": v.get("v_sku"),
            "price": product_data.selling_price, # Using main selling price for QR code
            "color": v.get("color"),
            "size": v.get("size"),
        })
        v["qr_code_url"] = variant_qr
        enriched_variants.append(v)

    variants = enriched_variants

    product = ProductModel(
        id=str(uuid.uuid4()),
        product_code=product_code,
        sku=parent_sku,
        name=product_data.name,
        description=product_data.description,
        category_id=product_data.category_id,
        cost_price=product_data.cost_price,
        selling_price=product_data.selling_price,
        min_selling_price=product_data.min_selling_price,
        stock=total_stock,
        min_stock=min_stock,
        variants=variants,
        images=product_data.images,
        is_service=product_data.is_service,
        qr_code_url=qr_code_url,
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

    if current_user.role != "admin":
        raise HTTPException(403, "Admin only")

    if len(product_data.images) > 5:
        raise HTTPException(400, "Maximum 5 images allowed")

    if product_data.selling_price < product_data.min_selling_price:
        raise HTTPException(400, "Selling price cannot be below minimum selling price")

    product.name = product_data.name
    product.description = product_data.description
    product.category_id = product_data.category_id
    product.sku = product_data.sku or product.sku
    product.selling_price = product_data.selling_price
    product.min_selling_price = product_data.min_selling_price
    product.images = product_data.images
    product.is_service = product_data.is_service
    product.cost_price = product_data.cost_price

    if product_data.is_service == 1:
        product.variants = []
        product.stock = 0
        product.min_stock = 0
    else:
        # Variants optional on update as well
        if product_data.variants:
            variants = [v.dict() for v in product_data.variants]

            # Regenerate variant QR codes on update
            enriched_variants = []
            for v in variants:
                variant_qr = generate_qr({
                    "type": "variant",
                    "product_name": product.name,
                    "sku": product.sku,
                    "v_sku": v.get("v_sku"),
                    "price": v.get("v_selling_price") or product.selling_price, # Use variant price if available
                    "color": v.get("color"),
                    "size": v.get("size"),
                })
                v["qr_code_url"] = variant_qr
                enriched_variants.append(v)

            product.variants = enriched_variants
            product.stock = sum(v.get("stock", 0) for v in enriched_variants)
        else:
            product.variants = []
            product.stock = 0

        product.min_stock = product_data.min_stock

    db.commit()
    db.refresh(product)
    return product

@api_router.delete("/products/{product_id}")
def delete_product(
    product_id: str,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin only")

    product = db.query(ProductModel).filter(ProductModel.id == product_id).first()
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    
    db.delete(product)
    db.commit()
    return {"message": "Product deleted successfully"}

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

@api_router.get("/customers/{customer_id}", response_model=Customer)
def get_customer(
    customer_id: str,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    customer = db.query(CustomerModel).filter(CustomerModel.id == customer_id).first()
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    
    return Customer(
        id=customer.id,
        name=customer.name,
        email=customer.email,
        phone=customer.phone,
        address=customer.address,
        created_at=customer.created_at.isoformat()
    )

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
                "gst_amount": inv.gst_amount,
                "discount": inv.discount,
                "total": inv.total,
                "payment_status": inv.payment_status,
                "created_at": inv.created_at.isoformat(),
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

@api_router.post("/invoices", status_code=201)
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
            name=invoice_data.customer_name,
            email=invoice_data.customer_email,
            phone=invoice_data.customer_phone,
            address=invoice_data.customer_address,
            created_at=datetime.now(IST)
        )
        db.add(customer)
        db.flush()

    # ================= ITEMS =================
    invoice_items = []
    subtotal = 0

    for item in invoice_data.items:
        # 🔍 FETCH PRODUCT BY ID (PRIMARY)
        product = (
            db.query(ProductModel)
            .filter(ProductModel.id == item.product_id)
            .with_for_update() # Lock the row to prevent race conditions
            .first()
        )

        if not product:
            raise HTTPException(
                status_code=404,
                detail="Product not found"
            )

        # 💰 PRICE (TRUSTED FROM DB)
        price = product.selling_price
        quantity = item.quantity
        line_total = price * quantity
        subtotal += line_total

        # ============================
        # ✅ SERVICE → NO STOCK / NO INVENTORY
        # ============================
        if product.is_service == 1:
            invoice_items.append({
                "product_id": product.id,
                "sku": item.sku or product.sku,
                "product_name": product.name,
                "quantity": quantity,
                "gst_rate": 18,
                "price": price,
                "total": line_total
            })
            continue   # 🚀 CRITICAL: SKIP STOCK LOGIC

        # ============================
        # 📦 PART → STOCK + INVENTORY (VARIANT-AWARE)
        # ============================
        variants = product.variants or []
        stock_before = 0
        stock_after = 0
        variant_found = False

        if variants and item.sku:
            # 🔍 Try to match variant by v_sku
            updated_variants = []
            for v in variants:
                if v.get("v_sku") == item.sku:
                    current_variant_stock = int(v.get("stock", 0))
                    stock_before = current_variant_stock
                    
                    if current_variant_stock < quantity:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Insufficient stock for variant {item.sku}. Available: {current_variant_stock}"
                        )
                    
                    stock_after = current_variant_stock - quantity
                    v["stock"] = stock_after # Update individual variant stock
                    variant_found = True
                
                updated_variants.append(v)

            if variant_found:
                product.variants = updated_variants # Re-assign list to trigger SQLAlchemy modification flag
                flag_modified(product, "variants")
        
        if not variant_found:
            stock_before = product.stock
            stock_after = stock_before - quantity
            
            if stock_after < 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"Insufficient stock for {product.name}. Available: {stock_before}"
                )
            
            product.stock = stock_after

        db.add(InventoryTransaction(
            id=str(uuid.uuid4()),
            product_id=product.id,
            type="OUT",
            quantity=quantity,
            source="INVOICE",
            reason="Vehicle Service",
            stock_before=stock_before,
            stock_after=stock_after,
            variant_sku=item.sku if variant_found else None, # Store variant SKU if found
            created_by=current_user.id,
            created_at=datetime.now(IST)
        ))

        invoice_items.append({
            "product_id": product.id,
            "sku": item.sku or product.sku,
            "product_name": product.name,
            "quantity": quantity,
            "gst_rate": 18,
            "price": price,
            "total": line_total
        })

    # ================= INVOICE =================
    total = subtotal + invoice_data.gst_amount - invoice_data.discount
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
        gst_amount=invoice_data.gst_amount,
        discount=invoice_data.discount,
        total=total,
        payment_status=invoice_data.payment_status,
        created_at=datetime.now(IST)
    )

    db.add(invoice)
    db.commit()
    db.refresh(invoice)

    return Invoice(
        id=invoice.id,
        invoice_number=invoice.invoice_number,
        customer_id=invoice.customer_id,
        customer_name=invoice.customer_name,
        customer_phone=invoice.customer_phone,
        customer_address=invoice.customer_address,
        items=invoice_items,
        subtotal=subtotal,
        gst_amount=invoice.gst_amount,
        discount=invoice.discount,
        total=invoice.total,
        payment_status=invoice.payment_status,
        created_at=invoice.created_at.isoformat()
    )

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
        .filter(ProductModel.is_service == 0)  # ✅ EXCLUDE SERVICES
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

        
@api_router.get("/dashboard")
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
            "text": f"{txn.type} – {prod.name} ({txn.quantity})",
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

@api_router.get("/products/sku/{sku}")
def get_product_by_sku(
    sku: str,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    sku = sku.strip()

    all_products = db.query(ProductModel).all()
    
    for product in all_products:
        if product.variants:
            for variant in product.variants:
                if variant.get("v_sku") == sku:
                    # Found matching variant - return product with this specific variant highlighted
                    return {
                        "id": product.id,
                        "product_code": product.product_code,
                        "sku": product.sku,
                        "name": product.name,
                        "selling_price": variant.get("v_selling_price", product.selling_price),
                        "stock": variant.get("stock", 0),
                        "min_stock": product.min_stock,
                        "is_service": product.is_service,
                        "images": product.images,
                        "variant": {
                            "v_id": variant.get("v_id"),
                            "v_sku": variant.get("v_sku"),
                            "variant_name": variant.get("variant_name"),
                            "size": variant.get("size"),
                            "color": variant.get("color"),
                            "image": variant.get("v_image_url"),  # Return variant's image
                            "stock": variant.get("stock", 0),
                        }
                    }

    product = db.query(ProductModel).filter(
        (ProductModel.sku == sku) |
        (ProductModel.product_code == sku)
    ).first()

    if product:
        return {
            "id": product.id,
            "product_code": product.product_code,
            "sku": product.sku,
            "name": product.name,
            "selling_price": product.selling_price,
            "stock": product.stock,
            "min_stock": product.min_stock,
            "is_service": product.is_service,
            "images": product.images,
            "variant": None,  # No specific variant
        }

    raise HTTPException(status_code=404, detail="Product or variant not found")


app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
