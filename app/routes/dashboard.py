from datetime import datetime, timedelta
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from fastapi_limiter.depends import RateLimiter
from sqlalchemy import case, func
from sqlalchemy.orm import Session

from app.core.config import IST
from app.db import get_db
from app.models import CustomerModel, InventoryTransaction, InvoiceModel, ProductModel, UserModel
from app.schemas.invoices import SalesChartItem
from app.utils import get_current_user

router = APIRouter()


@router.get("/api/dashboard", dependencies=[Depends(RateLimiter(times=30, seconds=60))])
def get_dashboard_stats(
    filter: str = "today",
    year: int | None = None,
    month: int | None = None,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    now = datetime.now(IST)

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
        end = datetime(year + 1, 1, 1, tzinfo=IST) if month == 12 else datetime(year, month + 1, 1, tzinfo=IST)
    else:
        raise HTTPException(status_code=400, detail="Invalid filter")

    invoice_q = db.query(InvoiceModel).filter(
        InvoiceModel.invoice_type == "FINAL",
        InvoiceModel.payment_status == "paid",
        InvoiceModel.created_at >= start,
        InvoiceModel.created_at < end,
    )

    total_sales = invoice_q.with_entities(func.coalesce(func.sum(InvoiceModel.total), 0)).scalar()
    total_orders = invoice_q.count()
    total_customers = invoice_q.with_entities(func.count(func.distinct(InvoiceModel.customer_id))).scalar()

    low_stock = db.query(ProductModel).filter(ProductModel.is_service == 0, ProductModel.stock <= ProductModel.min_stock).count()

    return {
        "total_sales": float(total_sales),
        "total_orders": total_orders,
        "total_customers": total_customers,
        "low_stock_items": low_stock,
    }


@router.get("/api/dashboard/today")
def dashboard_today(
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    now = datetime.now(IST)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    invoices_today = db.query(InvoiceModel).filter(InvoiceModel.invoice_type == "FINAL", InvoiceModel.created_at >= start).count()

    items_sold_today = db.query(func.coalesce(func.sum(InventoryTransaction.quantity), 0)).filter(
        InventoryTransaction.type == "OUT", InventoryTransaction.created_at >= start
    ).scalar()

    new_customers = db.query(CustomerModel).filter(CustomerModel.created_at >= start).count()

    inventory_out = db.query(InventoryTransaction).filter(
        InventoryTransaction.type == "OUT", InventoryTransaction.created_at >= start
    ).count()

    return {
        "invoices_today": invoices_today,
        "items_sold_today": int(items_sold_today or 0),
        "inventory_out_today": inventory_out,
        "new_customers_today": new_customers,
    }


@router.get("/api/dashboard/low-stock")
def low_stock_products(
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    products = (
        db.query(ProductModel)
        .filter(ProductModel.is_service == 0, ProductModel.stock <= ProductModel.min_stock)
        .order_by(ProductModel.stock.asc())
        .limit(10)
        .all()
    )

    return [{"product_name": p.name, "stock": p.stock, "min_stock": p.min_stock} for p in products]


@router.get("/api/dashboard/top-products")
def top_products(
    limit: int = 5,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    results = (
        db.query(ProductModel.name, func.sum(InventoryTransaction.quantity).label("qty"))
        .join(ProductModel, ProductModel.id == InventoryTransaction.product_id)
        .filter(InventoryTransaction.type == "OUT", ProductModel.is_service == 0)
        .group_by(ProductModel.name)
        .order_by(func.sum(InventoryTransaction.quantity).desc())
        .limit(limit)
        .all()
    )

    return [{"name": r.name, "quantity": int(r.qty or 0)} for r in results]


@router.get("/api/dashboard/inventory-movement")
def inventory_movement(
    days: int = 7,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
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

    return [{"day": r.day.strftime("%d %b"), "inward": int(r.inward or 0), "outward": int(r.outward or 0)} for r in results]


@router.get("/api/dashboard/activity")
def dashboard_activity(
    limit: int = 10,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    invoices = db.query(InvoiceModel).filter(InvoiceModel.invoice_type == "FINAL").order_by(InvoiceModel.created_at.desc()).limit(limit).all()

    inventory = db.query(InventoryTransaction, ProductModel).join(ProductModel).order_by(InventoryTransaction.created_at.desc()).limit(limit).all()

    activity = []
    for inv in invoices:
        activity.append({"type": "invoice", "text": f"Invoice {inv.invoice_number} - Rs.{inv.total}", "date": inv.created_at.isoformat()})

    for txn, prod in inventory:
        activity.append({"type": "inventory", "text": f"{txn.type} - {prod.name} ({txn.quantity}) by {txn.created_by_name}", "date": txn.created_at.isoformat()})

    return sorted(activity, key=lambda x: x["date"], reverse=True)[:limit]


@router.get("/api/dashboard/hourly-sales")
def hourly_sales_today(
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    now = datetime.now(IST)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)

    results = (
        db.query(func.hour(InvoiceModel.created_at).label("hour"), func.sum(InvoiceModel.total).label("total"))
        .filter(
            InvoiceModel.invoice_type == "FINAL",
            InvoiceModel.created_at >= start,
            InvoiceModel.created_at <= now,
            InvoiceModel.payment_status == "paid",
        )
        .group_by("hour")
        .order_by("hour")
        .all()
    )

    data = []
    for hour, total in results:
        data.append({"label": f"{hour:02d}:00-{hour+1:02d}:00", "total": float(total or 0)})

    return data


@router.get("/api/dashboard/sales", response_model=List[SalesChartItem])
def get_sales_data(
    filter: str = "today",
    year: int | None = None,
    month: int | None = None,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    now = datetime.now(IST)

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
        end = datetime(year + 1, 1, 1, tzinfo=IST) if month == 12 else datetime(year, month + 1, 1, tzinfo=IST)
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
            InvoiceModel.created_at <= end,
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
