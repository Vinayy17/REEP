import math
import time
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi_limiter.depends import RateLimiter
from sqlalchemy import func, or_
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.core.config import IST
from app.db import get_db
from app.models import CategoryModel, InventoryTransaction, ProductModel, UserModel
from app.schemas.catalog import ProductCreate
from app.utils import (
    calculate_total_stock,
    generate_product_code,
    generate_qr,
    get_current_user,
    has_inventory,
    has_invoice,
    safe_commit,
    safe_images,
    upload_image_to_cloudinary,
)

router = APIRouter()


@router.post("/api/upload/product-image", dependencies=[Depends(RateLimiter(times=20, seconds=60))])
def upload_product_image(
    file: UploadFile = File(...),
    current_user: UserModel = Depends(get_current_user),
):
    if current_user.role not in ["admin", "store_handler"]:
        raise HTTPException(status_code=403, detail="Admin only")

    url = upload_image_to_cloudinary(file.file)
    return {"url": url}


@router.post("/api/upload/variant-image")
def upload_variant_image(
    file: UploadFile = File(...),
    current_user: UserModel = Depends(get_current_user),
):
    if current_user.role not in ["admin", "store_handler"]:
        raise HTTPException(status_code=403, detail="Not allowed")

    url = upload_image_to_cloudinary(file.file, folder="variant_images")
    return {"url": url}


def resolve_product_and_variant_by_sku(db: Session, code: str):
    code = code.strip()

    products = (
        db.query(ProductModel)
        .filter(ProductModel.is_active == 1, ProductModel.variants.isnot(None))
        .with_for_update()
        .all()
    )

    for product in products:
        for variant in (product.variants or []):
            if variant.get("v_sku", "").lower() == code.lower():
                return product, variant["v_sku"]

    product = (
        db.query(ProductModel)
        .filter(
            ProductModel.is_active == 1,
            or_(
                func.lower(ProductModel.sku) == code.lower(),
                func.lower(ProductModel.product_code) == code.lower(),
            ),
        )
        .with_for_update()
        .first()
    )

    if product:
        return product, None

    raise HTTPException(404, "Product / Variant not found")


def with_retry(fn, retries=3):
    for _ in range(retries):
        try:
            return fn()
        except OperationalError as exc:
            if "Deadlock found" in str(exc):
                time.sleep(0.2)
                continue
            raise
    raise HTTPException(500, "Inventory busy, please retry")


@router.get("/api/products/recent-skus")
def get_recent_skus(
    limit: int = Query(2, ge=1, le=10),
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(ProductModel.product_code)
        .filter(ProductModel.is_active == 1)
        .order_by(ProductModel.created_at.desc())
        .limit(limit)
        .all()
    )
    return {"recent_skus": [row.product_code for row in rows]}


@router.get("/api/products")
def get_products(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=100),
    search: Optional[str] = Query(None),
    category_id: Optional[str] = Query(None),
    sort: Optional[str] = Query(None),
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    offset = (page - 1) * limit
    query = db.query(ProductModel).filter(ProductModel.is_active == 1)

    if search:
        search_term = f"%{search.lower().strip()}%"
        query = query.filter(
            or_(
                func.lower(ProductModel.name).like(search_term),
                func.lower(ProductModel.sku).like(search_term),
                func.lower(ProductModel.product_code).like(search_term),
            )
        )

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
    }

    sort_tokens = [token.strip() for token in (sort or "").split(",") if token.strip()]
    order_by_clauses = []

    for token in sort_tokens:
        clause = sort_map.get(token)
        if clause is not None:
            order_by_clauses.append(clause)

    order_by_clauses.append(ProductModel.created_at.desc())

    total = query.count()
    products = (
        query.order_by(*order_by_clauses).offset(offset).limit(limit).all()
    )

    result = []
    for product in products:
        item = {
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
            "variants": product.variants or [],
            "images": safe_images(product.images),
            "qr_code_url": product.qr_code_url,
            "is_service": product.is_service,
            "created_at": product.created_at.isoformat(),
        }
        if current_user.role == "admin":
            item["cost_price"] = product.cost_price

        result.append(item)

    return {
        "data": result,
        "page": page,
        "limit": limit,
        "total": total,
        "total_pages": math.ceil(total / limit),
    }


@router.get("/api/products/ageing")
def product_ageing_report(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=1, le=50),
    bucket: Optional[str] = Query(None, pattern="^(daily|weekly|monthly)$"),
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    now = datetime.now(IST).replace(tzinfo=None)
    offset = (page - 1) * limit

    base_products = (
        db.query(ProductModel)
        .filter(ProductModel.is_active == 1, ProductModel.is_service == 0)
        .order_by(ProductModel.created_at.asc())
        .all()
    )

    def get_bucket(days: int) -> str:
        if days <= 30:
            return "latest"
        if days <= 45:
            return "new"
        if days <= 60:
            return "medium"
        if days <= 90:
            return "old"
        if days <= 150:
            return "very_old"
        return "dead_stock"

    enriched = []

    for product in base_products:
        first_inward = (
            db.query(func.min(InventoryTransaction.created_at))
            .filter(
                InventoryTransaction.product_id == product.id,
                InventoryTransaction.type == "IN",
            )
            .scalar()
        )

        base_date = first_inward or product.created_at
        if base_date.tzinfo is not None:
            base_date = base_date.replace(tzinfo=None)

        age_days = (now - base_date).days
        age_bucket = get_bucket(age_days)

        if bucket and age_bucket != bucket:
            continue

        enriched.append(
            {
                "product_code": product.product_code,
                "sku": product.sku,
                "name": product.name,
                "qty": product.stock,
                "age_bucket": age_bucket,
                "first_inward_date": first_inward.isoformat() if first_inward else None,
                "age_days": age_days,
            }
        )

    total = len(enriched)
    paginated = enriched[offset : offset + limit]

    return {
        "page": page,
        "limit": limit,
        "total": total,
        "total_pages": math.ceil(total / limit),
        "data": paginated,
    }


@api_router.post("/products", status_code=201)
def create_product(
    product_data: ProductCreate,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db)
):

    # ================= SECURITY =================
    if current_user.role != "admin":
        product_data.cost_price = 0

    # ================= VALIDATIONS =================
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

    existing = db.query(ProductModel).filter(
        ProductModel.sku == parent_sku
    ).first()

    # ================= RESTORE ARCHIVED =================
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

            if current_user.role == "admin":
                existing.cost_price = product_data.cost_price

            # ===== COMBO SUPPORT =====
            existing.is_combo = product_data.is_combo
            existing.combo_items = (
                [c.dict() for c in product_data.combo_items]
                if product_data.is_combo
                else None
            )

            if product_data.is_service == 1:
                existing.variants = []
                existing.stock = 0

            else:

                restored_variants = []

                for v in product_data.variants or []:
                    vd = v.dict()
                    vd["stock"] = 0

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

        raise HTTPException(409, "SKU already exists")

    # ================= CREATE NEW =================
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

    # ================= PRODUCT QR =================
    qr_code_url = generate_qr({
        "type": "product",
        "name": product_data.name,
        "sku": parent_sku,
        "price": product_data.selling_price,
    })

    # ================= VARIANT QR =================
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

        cost_price=product_data.cost_price,
        selling_price=product_data.selling_price,
        min_selling_price=product_data.min_selling_price,

        stock=0,
        min_stock=min_stock,

        variants=enriched_variants,
        images=product_data.images,

        # ===== COMBO SUPPORT =====
        is_combo=product_data.is_combo,
        combo_items=[c.dict() for c in product_data.combo_items] if product_data.is_combo else None,

        is_service=product_data.is_service,

        qr_code_url=qr_code_url,
        is_active=1,
        created_at=datetime.now(IST)
    )

    db.add(product)
    db.commit()
    db.refresh(product)

    return product

@router.put("/api/products/{product_id}")
def update_product(
    product_id: str,
    product_data: ProductCreate,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
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

    if product_data.is_service == 1:
        product.variants = []
    else:
        existing_variants = {v["v_sku"]: v for v in (product.variants or [])}
        updated_variants = []

        for variant in product_data.variants or []:
            variant_dict = variant.dict()
            old = existing_variants.get(variant_dict.get("v_sku"), {})
            variant_dict["stock"] = old.get("stock", 0)
            variant_dict["qr_code_url"] = generate_qr(
                {
                    "type": "variant",
                    "product_name": product.name,
                    "sku": product.sku,
                    "v_sku": variant_dict.get("v_sku"),
                    "price": variant_dict.get("v_selling_price") or product.selling_price,
                    "color": variant_dict.get("color"),
                    "size": variant_dict.get("size"),
                }
            )
            updated_variants.append(variant_dict)

        product.variants = updated_variants

    db.commit()
    db.refresh(product)
    return product


@router.delete("/api/products/{product_id}")
def delete_or_archive_product(
    product_id: str,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user.role != "admin":
        raise HTTPException(403, "Admin only")

    product = (
        db.query(ProductModel)
        .filter(ProductModel.id == product_id, ProductModel.is_active == 1)
        .first()
    )

    if not product:
        raise HTTPException(404, "Product not found")

    if not has_inventory(db, product_id) and not has_invoice(db, product_id):
        db.delete(product)
        db.commit()
        return {"message": "Product deleted permanently", "action": "hard_delete"}

    product.is_active = 0
    db.commit()
    return {"message": "Product archived (used in inventory/invoices)", "action": "archived"}


@router.get("/api/products/list")
def list_products(
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    products = (
        db.query(ProductModel)
        .filter(ProductModel.is_service == 0, ProductModel.is_active == 1)
        .all()
    )

    return [
        {
            "id": product.id,
            "product_code": product.product_code,
            "name": product.name,
            "sku": product.sku,
            "stock": product.stock,
            "min_stock": product.min_stock,
            "category_name": product.category.name if product.category else "Unknown",
        }
        for product in products
    ]


@router.get("/api/products/search", dependencies=[Depends(RateLimiter(times=100, seconds=60))])
def search_products(
    q: str,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    search_term = f"%{q.lower().strip()}%"
    products = (
        db.query(ProductModel)
        .filter(
            ProductModel.is_active == 1,
            or_(
                func.lower(ProductModel.name).like(search_term),
                func.lower(ProductModel.sku).like(search_term),
                func.lower(ProductModel.product_code).like(search_term),
            ),
        )
        .limit(20)
        .all()
    )

    return [
        {
            "id": product.id,
            "name": product.name,
            "sku": product.sku,
            "product_code": product.product_code,
            "selling_price": product.selling_price,
            "min_selling_price": product.min_selling_price,
            "variants": product.variants or [],
            "images": safe_images(product.images),
            "stock": product.stock,
            "is_service": product.is_service,
        }
        for product in products
    ]


@router.get("/api/products/sku/{code}")
def get_product_by_sku(
    code: str,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    code = code.strip()
    product, variant_sku = resolve_product_and_variant_by_sku(db, code)

    if variant_sku:
        for variant in product.variants or []:
            if variant.get("v_sku") == variant_sku:
                v_price = (
                    variant.get("v_selling_price")
                    or variant.get("v_price")
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
                    "stock": variant.get("stock", 0),
                    "min_stock": product.min_stock,
                    "is_service": product.is_service,
                    "images": product.images or [],
                    "variants": product.variants or [],
                    "variant": {
                        "v_sku": variant.get("v_sku"),
                        "variant_name": variant.get("variant_name"),
                        "size": variant.get("size"),
                        "color": variant.get("color"),
                        "stock": variant.get("stock", 0),
                        "v_price": v_price,
                    },
                }

        raise HTTPException(500, "Variant resolved but missing")

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
        "variant": None,
    }
