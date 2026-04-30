import time
import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from fastapi_limiter.depends import RateLimiter
from sqlalchemy import func, or_
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from app.core.config import IST
from app.db import get_db
from app.models import InventoryTransaction, ProductModel, UserModel
from app.schemas.inventory import MaterialInwardBySkuRequest, MaterialOutwardBySkuRequest
from app.utils import calculate_total_stock, get_current_user, safe_commit

router = APIRouter()


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


@router.get("/api/inventory/lookup/{code}")
def inventory_lookup(
    code: str,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    product, variant_sku = resolve_product_and_variant_by_sku(db, code)

    variants = list(product.variants or [])

    if variant_sku:
        for variant in variants:
            if variant.get("v_sku") == variant_sku:
                return {
                    "level": "VARIANT",
                    "product_id": product.id,
                    "product_name": product.name,
                    "parent_sku": product.sku,
                    "variant": {
                        "v_sku": variant.get("v_sku"),
                        "color": variant.get("color"),
                        "size": variant.get("size"),
                        "stock": variant.get("stock", 0),
                    },
                    "total_stock": product.stock,
                }

        raise HTTPException(500, "Variant SKU resolved but missing")

    return {
        "level": "PRODUCT",
        "product_id": product.id,
        "product_name": product.name,
        "parent_sku": product.sku,
        "variants": [
            {
                "v_sku": variant.get("v_sku"),
                "color": variant.get("color"),
                "size": variant.get("size"),
                "stock": variant.get("stock", 0),
            }
            for variant in variants
        ],
        "total_stock": product.stock,
    }


@router.post("/api/inventory/material-inward/sku", dependencies=[Depends(RateLimiter(times=30, seconds=60))])
def material_inward_by_sku(
    request: MaterialInwardBySkuRequest,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if request.quantity <= 0:
        raise HTTPException(status_code=400, detail="Quantity must be > 0")

    product, variant_sku = resolve_product_and_variant_by_sku(db, request.sku)
    if product.is_active == 0:
        raise HTTPException(status_code=400, detail="Product is archived and cannot be used")

    if product.is_service == 1:
        raise HTTPException(status_code=400, detail="Inventory not allowed for services")

    variants = list(product.variants or [])

    if variants and not variant_sku:
        raise HTTPException(status_code=400, detail="Product has variants. Use variant SKU for inventory.")

    variant_stock_after = None

    if variant_sku:
        total_before = calculate_total_stock(variants)

        for variant in variants:
            if variant.get("v_sku") == variant_sku:
                variant["stock"] = int(variant.get("stock", 0)) + request.quantity
                variant_stock_after = variant["stock"]
                break
        else:
            raise HTTPException(status_code=400, detail="Variant SKU not found")

        product.variants = variants
        flag_modified(product, "variants")

        product.stock = calculate_total_stock(variants)

        stock_before = total_before
        stock_after = product.stock

    else:
        stock_before = int(product.stock or 0)
        product.stock = stock_before + request.quantity
        stock_after = product.stock

    db.add(
        InventoryTransaction(
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
            created_by_name=current_user.name,
            created_at=datetime.now(IST),
        )
    )

    safe_commit(db)

    return {
        "message": "Stock added successfully",
        "variant_sku": variant_sku,
        "stock_after": stock_after,
        "variant_stock_after": variant_stock_after,
    }


@router.post("/api/inventory/material-outward/sku", dependencies=[Depends(RateLimiter(times=30, seconds=60))])
def material_outward_by_sku(
    request: MaterialOutwardBySkuRequest,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if request.quantity <= 0:
        raise HTTPException(status_code=400, detail="Quantity must be > 0")

    product, variant_sku = resolve_product_and_variant_by_sku(db, request.sku)

    if product.is_active == 0:
        raise HTTPException(status_code=400, detail="Product is archived and cannot be used")

    if product.is_service == 1:
        raise HTTPException(status_code=400, detail="Inventory not allowed for services")

    variants = list(product.variants or [])

    if variants and not variant_sku:
        raise HTTPException(status_code=400, detail="Product has variants. Use variant SKU for inventory.")

    variant_stock_after = None

    if variant_sku:
        total_before = calculate_total_stock(variants)

        for variant in variants:
            if variant.get("v_sku") == variant_sku:
                current_stock = int(variant.get("stock", 0))
                if current_stock < request.quantity:
                    raise HTTPException(status_code=400, detail="Insufficient variant stock")

                variant["stock"] = current_stock - request.quantity
                variant_stock_after = variant["stock"]
                break
        else:
            raise HTTPException(status_code=400, detail="Variant SKU not found")

        product.variants = variants
        flag_modified(product, "variants")
        product.stock = calculate_total_stock(variants)

        stock_before = total_before
        stock_after = product.stock

    else:
        current_stock = int(product.stock or 0)
        if current_stock < request.quantity:
            raise HTTPException(status_code=400, detail="Insufficient product stock")

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
            created_by_name=current_user.name,
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


@router.get("/api/inventory/transactions")
def get_inventory_transactions(
    page: int = 1,
    limit: int = 30,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    offset = (page - 1) * limit

    query = db.query(InventoryTransaction, ProductModel).join(
        ProductModel, InventoryTransaction.product_id == ProductModel.id
    )
    total = query.count()

    rows = query.order_by(InventoryTransaction.created_at.desc()).offset(offset).limit(limit).all()

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
                "created_by": txn.created_by_name,
            }
            for txn, prod in rows
        ],
        "total": total,
        "page": page,
        "limit": limit,
    }
