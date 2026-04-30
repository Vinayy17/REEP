import ast
import json
import time

from fastapi import HTTPException
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.models.inventory import InventoryTransaction
from app.models.invoices import InvoiceModel


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
    return (
        db.query(InventoryTransaction)
        .filter(InventoryTransaction.product_id == product_id)
        .first()
        is not None
    )


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
    for charge in charges:
        if hasattr(charge, "amount"):
            total += float(charge.amount or 0)
        elif isinstance(charge, dict):
            total += float(charge.get("amount", 0))

    return round(total, 2)


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


def safe_commit(db: Session, retries: int = 3):
    for attempt in range(retries):
        try:
            db.commit()
            return
        except OperationalError as exc:
            db.rollback()

            if "1213" in str(exc):
                if attempt == retries - 1:
                    raise HTTPException(
                        status_code=409,
                        detail="Inventory is busy. Please retry.",
                    )
                time.sleep(0.2)
            else:
                raise


__all__ = [
    "calculate_additional_total",
    "calculate_total_stock",
    "has_inventory",
    "has_invoice",
    "parse_invoice_items",
    "safe_commit",
    "safe_images",
    "with_retry",
]
