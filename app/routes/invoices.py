
import io
import json
import math
import uuid
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi_limiter.depends import RateLimiter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from sqlalchemy.orm import Session
from sqlalchemy.orm.attributes import flag_modified

from app.core.config import IST
from app.db import get_db
from app.models import CustomerModel, InvoiceModel, InvoicePayment, LedgerEntry, ProductModel, UserModel
from app.schemas.invoices import AddPaymentRequest, DraftFinalizeRequest, InvoiceCreate
from app.utils import calculate_additional_total, calculate_total_stock, get_current_user, parse_invoice_items

router = APIRouter()


def generate_draft_number(db: Session) -> str:
    year = datetime.now(IST).year
    last = (
        db.query(InvoiceModel.draft_number)
        .filter(InvoiceModel.invoice_type == "DRAFT", InvoiceModel.draft_number.like(f"DRF-{year}-%"))
        .order_by(InvoiceModel.draft_number.desc())
        .first()
    )
    next_num = int(last[0].split("-")[-1]) + 1 if last and last[0] else 1
    return f"DRF-{year}-{next_num:04d}"


def generate_invoice_number(db: Session) -> str:
    now = datetime.now(IST)
    fy_year = now.year if now.month >= 4 else now.year - 1
    fy_suffix = f"{fy_year % 100:02d}-{(fy_year + 1) % 100:02d}"
    last = (
        db.query(InvoiceModel.invoice_number)
        .filter(InvoiceModel.invoice_type == "FINAL", InvoiceModel.invoice_number.like(f"INV-{fy_suffix}-%"))
        .order_by(InvoiceModel.invoice_number.desc())
        .with_for_update()
        .first()
    )
    next_num = int(last[0].split("-")[-1]) + 1 if last and last[0] else 1
    return f"INV-{fy_suffix}-{next_num:04d}"


@router.post("/api/invoices/draft")
def create_invoice_draft(
    invoice_data: InvoiceCreate,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    allowed_status = ["pending", "paid", "partial", "cancelled"]
    status = invoice_data.payment_status or "pending"
    if status not in allowed_status:
        status = "pending"

    customer = None
    if invoice_data.customer_id:
        customer = db.query(CustomerModel).filter(CustomerModel.id == invoice_data.customer_id).first()
    elif invoice_data.customer_phone:
        customer = db.query(CustomerModel).filter(CustomerModel.phone == invoice_data.customer_phone).first()

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
        if not image_url and item.product_id:
            product = db.query(ProductModel).filter(ProductModel.id == item.product_id).first()
            if product:
                if product.variants and item.sku:
                    for variant in product.variants:
                        if variant.get("v_sku") == item.sku:
                            image_url = variant.get("image_url")
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

    cleaned_additional_charges = []
    for charge in invoice_data.additional_charges or []:
        label = (charge.label or "").strip()
        amount = float(charge.amount or 0)
        if label or amount > 0:
            cleaned_additional_charges.append({"label": label, "amount": amount})

    taxable = subtotal + sum(c["amount"] for c in cleaned_additional_charges)
    gst_amount = round((taxable * float(invoice_data.gst_rate or 0)) / 100, 2) if invoice_data.gst_enabled else 0
    total = round(taxable + gst_amount - float(invoice_data.discount or 0), 2)

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
        payment_status=status,
        created_by=current_user.id,
        created_by_name=current_user.name,
        created_at=datetime.now(IST),
    )
    db.add(draft)
    db.commit()
    db.refresh(draft)
    return {"id": draft.id, "draft_number": draft.draft_number, "payment_status": draft.payment_status, "total": draft.total}


@router.put("/api/invoices/draft/{draft_id}")
def update_invoice_draft(
    draft_id: str,
    invoice_data: InvoiceCreate,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    draft = db.query(InvoiceModel).filter(InvoiceModel.id == draft_id, InvoiceModel.invoice_type == "DRAFT").with_for_update().first()
    if not draft:
        raise HTTPException(status_code=404, detail="Draft not found")

    customer = None
    if invoice_data.customer_id:
        customer = db.query(CustomerModel).filter(CustomerModel.id == invoice_data.customer_id).first()
    elif invoice_data.customer_phone:
        customer = db.query(CustomerModel).filter(CustomerModel.phone == invoice_data.customer_phone).first()
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
    else:
        customer.name = invoice_data.customer_name
        customer.phone = invoice_data.customer_phone
        customer.address = invoice_data.customer_address
        if invoice_data.customer_email:
            customer.email = invoice_data.customer_email

    draft.customer_id = customer.id
    draft.customer_name = customer.name
    draft.customer_phone = customer.phone
    draft.customer_address = customer.address

    items = []
    subtotal = 0.0
    for item in invoice_data.items or []:
        quantity = int(item.quantity or 0)
        price = float(item.price or 0)
        line_total = round(price * quantity, 2)
        subtotal += line_total
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
            "image_url": item.image_url,
        })

    cleaned_additional_charges = []
    for charge in invoice_data.additional_charges or []:
        label = (charge.label or "").strip()
        amount = float(charge.amount or 0)
        if label or amount > 0:
            cleaned_additional_charges.append({"label": label, "amount": amount})

    taxable = subtotal + sum(c["amount"] for c in cleaned_additional_charges)
    gst_amount = round((taxable * float(invoice_data.gst_rate or 0)) / 100, 2) if invoice_data.gst_enabled else 0
    total = round(taxable + gst_amount - float(invoice_data.discount or 0), 2)

    draft.items = json.dumps(items)
    draft.subtotal = subtotal
    draft.additional_charges = cleaned_additional_charges
    draft.gst_enabled = 1 if invoice_data.gst_enabled else 0
    draft.gst_rate = float(invoice_data.gst_rate or 0)
    draft.gst_amount = gst_amount
    draft.discount = float(invoice_data.discount or 0)
    draft.total = total
    draft.payment_status = "pending"
    db.commit()

    return {
        "message": "Draft updated successfully",
        "draft_id": draft.id,
        "items_count": len(items),
        "additional_charges_count": len(cleaned_additional_charges),
        "total": draft.total,
    }


@router.get("/api/invoices/drafts")
def list_drafts(
    page: int = 1,
    limit: int = 10,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(InvoiceModel).filter(InvoiceModel.invoice_type == "DRAFT")
    total = query.count()
    drafts = query.order_by(InvoiceModel.created_at.desc()).offset((page - 1) * limit).limit(limit).all()
    return {
        "data": [
            {
                "id": draft.id,
                "draft_number": draft.draft_number,
                "customer_name": draft.customer_name,
                "total": draft.total,
                "created_at": draft.created_at.isoformat(),
            }
            for draft in drafts
        ],
        "total": total,
    }


@router.get("/api/invoices/draft/{draft_id}")
def get_single_draft(
    draft_id: str,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    draft = db.query(InvoiceModel).filter(InvoiceModel.id == draft_id, InvoiceModel.invoice_type == "DRAFT").first()
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
    }


@router.delete("/api/invoices/draft/{draft_id}")
def delete_draft(
    draft_id: str,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    draft = db.query(InvoiceModel).filter(InvoiceModel.id == draft_id, InvoiceModel.invoice_type == "DRAFT").first()
    if not draft:
        raise HTTPException(404, "Draft not found")
    db.delete(draft)
    db.commit()
    return {"message": "Draft deleted"}


@router.post("/api/invoices/draft/{draft_id}/finalize")
def finalize_draft(
    draft_id: str,
    data: Optional[DraftFinalizeRequest] = None,
    payment_status: Optional[str] = Query(None),
    payment_mode: Optional[str] = Query(None),
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    status_from_body = data.payment_status.strip().lower() if data and data.payment_status else None
    status_from_query = payment_status.strip().lower() if payment_status else None
    final_payment_status = status_from_body or status_from_query or "pending"
    if final_payment_status not in ["pending", "paid", "cancelled", "partial"]:
        raise HTTPException(400, "Invalid payment status")
    final_payment_mode = payment_mode.strip().lower() if payment_mode else "cash"

    draft = db.query(InvoiceModel).filter(InvoiceModel.id == draft_id, InvoiceModel.invoice_type == "DRAFT").with_for_update().first()
    if not draft:
        raise HTTPException(404, "Draft not found")

    items = parse_invoice_items(draft.items)
    if not items:
        raise HTTPException(400, "Draft has no items")

    subtotal = 0.0
    for item in items:
        quantity = int(item.get("quantity", 0))
        price = float(item.get("price", 0))
        subtotal += price * quantity

        if item.get("is_service") == 1 or not item.get("product_id"):
            continue

        product = db.query(ProductModel).filter(ProductModel.id == item["product_id"]).with_for_update().first()
        if not product:
            raise HTTPException(404, "Product not found")

        variants = list(product.variants or [])
        sku = item.get("sku")
        if variants:
            if not sku:
                raise HTTPException(400, "Variant SKU required")
            for variant in variants:
                if variant.get("v_sku") == sku:
                    current_stock = int(variant.get("stock", 0))
                    if current_stock < quantity:
                        raise HTTPException(400, f"Insufficient stock for variant {sku}")
                    variant["stock"] = current_stock - quantity
                    break
            else:
                raise HTTPException(400, "Variant not found")
            product.variants = variants
            product.stock = calculate_total_stock(variants)
            flag_modified(product, "variants")
            flag_modified(product, "stock")
        else:
            stock_before = int(product.stock or 0)
            if stock_before < quantity:
                raise HTTPException(400, f"Insufficient stock for {product.name}")
            product.stock = stock_before - quantity
            flag_modified(product, "stock")
        db.flush()

    additional_total = calculate_additional_total(draft.additional_charges or [])
    taxable = subtotal + additional_total
    draft.subtotal = subtotal
    draft.gst_amount = round((taxable * draft.gst_rate) / 100, 2) if draft.gst_enabled else 0
    draft.total = round(taxable + draft.gst_amount - draft.discount, 2)

    paid_amount = 0
    balance_amount = draft.total
    if final_payment_status == "paid":
        paid_amount = draft.total
        balance_amount = 0
    if final_payment_status == "partial":
        partial_amount = data.paid_amount if data else 0
        if partial_amount <= 0:
            raise HTTPException(400, "Partial payment amount required")
        if partial_amount > draft.total:
            raise HTTPException(400, "Partial amount exceeds invoice total")
        paid_amount = partial_amount
        balance_amount = draft.total - partial_amount

    draft.paid_amount = paid_amount
    draft.balance_amount = balance_amount
    draft.payment_mode = final_payment_mode
    invoice_number = generate_invoice_number(db)
    draft.invoice_type = "FINAL"
    draft.invoice_number = invoice_number
    draft.draft_number = None
    draft.payment_status = final_payment_status
    draft.created_at = datetime.now(IST)

    customer = db.query(CustomerModel).filter(CustomerModel.id == draft.customer_id).first()
    if customer:
        if balance_amount > 0:
            customer.current_balance += balance_amount
        if paid_amount > 0:
            db.add(InvoicePayment(invoice_id=draft.id, amount=paid_amount, payment_mode=final_payment_mode, created_by=current_user.id))
            db.add(
                LedgerEntry(
                    id=str(uuid.uuid4()),
                    entry_type="sale_payment",
                    reference_id=draft.id,
                    customer_id=draft.customer_id,
                    description=f"Invoice Payment {invoice_number}",
                    debit=0,
                    credit=paid_amount,
                    payment_mode=final_payment_mode,
                    entry_date=datetime.now(IST).date(),
                    created_by=current_user.id,
                    created_by_name=current_user.name,
                    created_at=datetime.now(IST),
                )
            )

    db.commit()
    return {
        "invoice_id": draft.id,
        "invoice_number": invoice_number,
        "total": draft.total,
        "paid_amount": paid_amount,
        "balance_amount": balance_amount,
        "payment_status": final_payment_status,
        "message": "Draft finalized successfully",
    }


@router.get("/api/invoices")
def get_invoices(
    page: int = 1,
    limit: int = 10,
    status: Optional[str] = None,
    range: Optional[str] = None,
    month: Optional[str] = None,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(InvoiceModel).filter(InvoiceModel.invoice_type == "FINAL")
    now = datetime.now(IST)
    start_of_today = datetime(now.year, now.month, now.day, tzinfo=IST)
    end_of_today = start_of_today + timedelta(days=1)

    if status == "paid":
        query = query.filter(InvoiceModel.payment_status == "paid")
    elif status == "pending":
        query = query.filter(InvoiceModel.payment_status == "pending")
    elif status == "cancelled":
        query = query.filter(InvoiceModel.payment_status == "cancelled")
    elif status == "partial":
        query = query.filter(InvoiceModel.payment_status == "partial")
    elif status == "ending":
        query = query.filter(InvoiceModel.payment_status != "paid", InvoiceModel.created_at.between(start_of_today, start_of_today + timedelta(days=5)))

    if range == "today":
        query = query.filter(InvoiceModel.created_at.between(start_of_today, end_of_today))
    elif range == "last10":
        query = query.filter(InvoiceModel.created_at >= start_of_today - timedelta(days=9))
    elif range == "last30":
        query = query.filter(InvoiceModel.created_at >= start_of_today - timedelta(days=29))

    if month:
        year, month_num = map(int, month.split("-"))
        start_date = datetime(year, month_num, 1, tzinfo=IST)
        end_date = start_date + timedelta(days=31)
        query = query.filter(InvoiceModel.created_at.between(start_date, end_date))

    total = query.count()
    invoices = query.order_by(InvoiceModel.created_at.desc()).offset((page - 1) * limit).limit(limit).all()

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
                "paid_amount": float(inv.paid_amount or 0),
                "balance_amount": float(inv.balance_amount or 0),
                "payment_mode": inv.payment_mode,
                "payment_status": inv.payment_status,
                "additional_charges": inv.additional_charges or [],
                "created_at": inv.created_at.isoformat(),
                "created_by": inv.created_by_name,
            }
            for inv in invoices
        ],
        "pagination": {"page": page, "limit": limit, "total": total, "total_pages": math.ceil(total / limit)},
    }


@router.post("/api/invoices", dependencies=[Depends(RateLimiter(times=20, seconds=60))])
def create_invoice(
    invoice_data: InvoiceCreate,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if invoice_data.payment_status not in ["pending", "paid", "partial", "cancelled"]:
        raise HTTPException(status_code=400, detail="Please select a valid payment status")
    if invoice_data.payment_status == "paid" and not invoice_data.payment_mode:
        raise HTTPException(status_code=400, detail="Payment mode required when payment status is paid")
    if not invoice_data.items:
        raise HTTPException(status_code=400, detail="Invoice must contain at least one item")

    customer = None
    if invoice_data.customer_id:
        customer = db.query(CustomerModel).filter(CustomerModel.id == invoice_data.customer_id).first()
    elif invoice_data.customer_phone:
        customer = db.query(CustomerModel).filter(CustomerModel.phone == invoice_data.customer_phone).first()
    if not customer:
        customer = CustomerModel(
            id=str(uuid.uuid4()),
            name=invoice_data.customer_name or "Walk-in",
            email=invoice_data.customer_email or f"{invoice_data.customer_phone}@example.com",
            phone=invoice_data.customer_phone,
            address=invoice_data.customer_address,
            created_at=datetime.now(IST),
        )
        db.add(customer)
        db.flush()

    invoice_items = []
    subtotal = 0.0
    for item in invoice_data.items:
        if not item.product_name or int(item.quantity) <= 0:
            continue
        quantity = int(item.quantity)
        price = float(item.price)
        line_total = round(price * quantity, 2)
        subtotal += line_total
        invoice_items.append({"product_id": item.product_id, "sku": item.sku, "product_name": item.product_name, "quantity": quantity, "price": price, "gst_rate": float(item.gst_rate or 0), "total": line_total})

    if len(invoice_items) == 0:
        raise HTTPException(status_code=400, detail="No valid items in invoice")

    cleaned_additional_charges = []
    for charge in invoice_data.additional_charges or []:
        label = (charge.label or "").strip()
        amount = float(charge.amount or 0)
        if label or amount > 0:
            cleaned_additional_charges.append({"label": label, "amount": amount})

    taxable = subtotal + sum(c["amount"] for c in cleaned_additional_charges)
    gst_amount = round((taxable * float(invoice_data.gst_rate or 0)) / 100, 2) if invoice_data.gst_enabled else 0
    total = round(taxable + gst_amount - float(invoice_data.discount or 0), 2)

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
        additional_charges=cleaned_additional_charges,
        gst_amount=gst_amount,
        gst_rate=float(invoice_data.gst_rate or 0),
        gst_enabled=1 if invoice_data.gst_enabled else 0,
        discount=float(invoice_data.discount or 0),
        total=total,
        payment_status=invoice_data.payment_status,
        created_by=current_user.id,
        created_by_name=current_user.name,
        created_at=datetime.now(IST),
    )
    db.add(invoice)
    db.flush()

    if invoice.payment_status == "paid":
        mode = (invoice_data.payment_mode or "cash").lower()
        db.add(LedgerEntry(id=str(uuid.uuid4()), entry_type="sale", reference_id=invoice.id, customer_id=customer.id, description=f"Sale Invoice {invoice.invoice_number}", debit=invoice.total, credit=0, payment_mode=mode, entry_date=datetime.now(IST).date(), created_by=current_user.id, created_by_name=current_user.name, created_at=datetime.now(IST)))
    else:
        customer.current_balance += invoice.total

    db.commit()
    db.refresh(invoice)
    return {"id": invoice.id, "invoice_number": invoice.invoice_number, "total": invoice.total, "created_at": invoice.created_at}


@router.api_route("/api/invoices/{invoice_id}/status", methods=["PUT", "PATCH"])
def update_invoice_status(
    invoice_id: str,
    payment_status: str,
    payment_mode: str = "cash",
    amount: float = 0,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    payment_status = payment_status.lower()
    payment_mode = payment_mode.lower()
    invoice = db.query(InvoiceModel).filter(InvoiceModel.id == invoice_id).with_for_update().first()
    if not invoice:
        raise HTTPException(404, "Invoice not found")
    customer = db.query(CustomerModel).filter(CustomerModel.id == invoice.customer_id).first()

    if payment_status == "partial":
        if amount <= 0:
            raise HTTPException(400, "Partial payment amount required")
        if invoice.paid_amount + amount > invoice.total:
            raise HTTPException(400, "Amount exceeds remaining balance")
        invoice.paid_amount += amount
        invoice.balance_amount = invoice.total - invoice.paid_amount
        invoice.payment_status = "paid" if invoice.balance_amount == 0 else "partial"
        if customer:
            customer.current_balance -= amount
        db.add(InvoicePayment(invoice_id=invoice.id, amount=amount, payment_mode=payment_mode, created_by=current_user.id))
        db.add(LedgerEntry(id=str(uuid.uuid4()), entry_type="sale_payment", reference_id=invoice.id, customer_id=invoice.customer_id, description=f"Partial payment for {invoice.invoice_number}", debit=0, credit=amount, payment_mode=payment_mode, entry_date=datetime.now(IST).date(), created_by=current_user.id, created_by_name=current_user.name, created_at=datetime.now(IST)))
        db.commit()
        return {"message": "Partial payment recorded", "paid_amount": invoice.paid_amount, "balance_amount": invoice.balance_amount, "payment_status": invoice.payment_status}

    if payment_status == "paid":
        remaining = invoice.total - invoice.paid_amount
        if remaining <= 0:
            raise HTTPException(400, "Invoice already paid")
        invoice.paid_amount = invoice.total
        invoice.balance_amount = 0
        invoice.payment_status = "paid"
        if customer:
            customer.current_balance -= remaining
        db.add(InvoicePayment(invoice_id=invoice.id, amount=remaining, payment_mode=payment_mode, created_by=current_user.id))
        db.add(LedgerEntry(id=str(uuid.uuid4()), entry_type="sale_payment", reference_id=invoice.id, customer_id=invoice.customer_id, description=f"Final payment {invoice.invoice_number}", debit=0, credit=remaining, payment_mode=payment_mode, entry_date=datetime.now(IST).date(), created_by=current_user.id, created_by_name=current_user.name, created_at=datetime.now(IST)))
        db.commit()
        return {"message": "Invoice fully paid", "invoice_number": invoice.invoice_number, "paid_amount": invoice.paid_amount, "balance_amount": 0, "payment_status": "paid"}

    raise HTTPException(400, "Invalid status change")


@router.post("/api/invoices/{invoice_id}/complete-payment")
def complete_invoice_payment(
    invoice_id: str,
    payment_mode: str = "cash",
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    invoice = db.query(InvoiceModel).filter(InvoiceModel.id == invoice_id).with_for_update().first()
    if not invoice:
        raise HTTPException(404, "Invoice not found")
    if invoice.payment_status == "paid":
        raise HTTPException(400, "Invoice already fully paid")

    balance = invoice.total - invoice.paid_amount
    if balance <= 0:
        raise HTTPException(400, "No balance remaining")

    invoice.paid_amount += balance
    invoice.balance_amount = 0
    invoice.payment_status = "paid"
    invoice.payment_mode = payment_mode

    db.add(InvoicePayment(invoice_id=invoice.id, amount=balance, payment_mode=payment_mode, created_by=current_user.id))
    customer = db.query(CustomerModel).filter(CustomerModel.id == invoice.customer_id).first()
    if customer:
        customer.current_balance -= balance

    db.add(LedgerEntry(id=str(uuid.uuid4()), entry_type="sale_payment", reference_id=invoice.id, customer_id=invoice.customer_id, description=f"Final payment for {invoice.invoice_number}", debit=0, credit=balance, payment_mode=payment_mode, entry_date=datetime.now(IST).date(), created_by=current_user.id, created_by_name=current_user.name, created_at=datetime.now(IST)))
    db.commit()

    return {"message": "Payment completed", "invoice_number": invoice.invoice_number, "paid_amount": invoice.paid_amount, "balance": invoice.balance_amount, "status": invoice.payment_status}


@router.get("/api/invoices/{invoice_id}/payments")
def invoice_payments(
    invoice_id: str,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    payments = db.query(InvoicePayment).filter(InvoicePayment.invoice_id == invoice_id).order_by(InvoicePayment.created_at.asc()).all()
    return [{"id": payment.id, "amount": payment.amount, "payment_mode": payment.payment_mode, "reference": payment.reference, "created_at": payment.created_at.isoformat()} for payment in payments]


@router.post("/api/invoices/{invoice_id}/add-payment")
def add_payment_to_invoice(
    invoice_id: str,
    payment_data: AddPaymentRequest,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    if payment_data.amount <= 0:
        raise HTTPException(400, "Payment amount must be greater than 0")
    if payment_data.payment_mode.lower() not in ["cash", "upi", "bank", "cheque"]:
        raise HTTPException(400, "Invalid payment mode")

    invoice = db.query(InvoiceModel).filter(InvoiceModel.id == invoice_id).with_for_update().first()
    if not invoice:
        raise HTTPException(404, "Invoice not found")
    if invoice.payment_status == "paid":
        raise HTTPException(400, "Invoice already fully paid")

    remaining_balance = invoice.total - invoice.paid_amount
    if payment_data.amount > remaining_balance:
        raise HTTPException(400, f"Payment amount exceeds balance of {remaining_balance}")

    invoice.paid_amount += payment_data.amount
    invoice.balance_amount = invoice.total - invoice.paid_amount
    if invoice.balance_amount <= 0:
        invoice.payment_status = "paid"
        invoice.balance_amount = 0
    else:
        invoice.payment_status = "partial"

    db.add(InvoicePayment(id=str(uuid.uuid4()), invoice_id=invoice.id, amount=payment_data.amount, payment_mode=payment_data.payment_mode.lower(), reference=payment_data.reference, created_by=current_user.id, created_at=datetime.now(IST)))

    customer = db.query(CustomerModel).filter(CustomerModel.id == invoice.customer_id).first()
    if customer:
        customer.current_balance -= payment_data.amount

    db.add(LedgerEntry(id=str(uuid.uuid4()), entry_type="sale_payment", reference_id=invoice.id, customer_id=invoice.customer_id, description=f"Payment for {invoice.invoice_number} via {payment_data.payment_mode}", debit=0, credit=payment_data.amount, payment_mode=payment_data.payment_mode.lower(), entry_date=datetime.now(IST).date(), created_by=current_user.id, created_by_name=current_user.name, created_at=datetime.now(IST)))

    db.commit()
    return {"message": "Payment recorded successfully", "invoice_number": invoice.invoice_number, "payment_amount": payment_data.amount, "paid_amount": invoice.paid_amount, "balance_amount": invoice.balance_amount, "payment_status": invoice.payment_status}


@router.delete("/api/invoices/payment/{payment_id}")
def delete_payment(
    payment_id: str,
    db: Session = Depends(get_db),
    current_user: UserModel = Depends(get_current_user),
):
    payment = db.query(InvoicePayment).filter(InvoicePayment.id == payment_id).first()
    if not payment:
        raise HTTPException(404, "Payment not found")

    invoice = db.query(InvoiceModel).filter(InvoiceModel.id == payment.invoice_id).with_for_update().first()
    if not invoice:
        raise HTTPException(404, "Invoice not found")

    invoice.paid_amount -= payment.amount
    invoice.balance_amount = invoice.total - invoice.paid_amount
    if invoice.paid_amount <= 0:
        invoice.payment_status = "pending"
        invoice.paid_amount = 0
        invoice.balance_amount = invoice.total
    else:
        invoice.payment_status = "partial"

    customer = db.query(CustomerModel).filter(CustomerModel.id == invoice.customer_id).first()
    if customer:
        customer.current_balance += payment.amount

    db.delete(payment)
    ledger_entry = db.query(LedgerEntry).filter(LedgerEntry.reference_id == invoice.id, LedgerEntry.credit == payment.amount, LedgerEntry.entry_type == "sale_payment", LedgerEntry.payment_mode == payment.payment_mode).order_by(LedgerEntry.created_at.desc()).first()
    if ledger_entry:
        db.delete(ledger_entry)

    db.commit()
    return {"message": "Payment deleted and reversed", "invoice_number": invoice.invoice_number, "paid_amount": invoice.paid_amount, "balance_amount": invoice.balance_amount, "payment_status": invoice.payment_status}


@router.get("/api/invoices/{invoice_id}/pdf")
def download_invoice_pdf(invoice_id: str, db: Session = Depends(get_db)):
    invoice = db.query(InvoiceModel).filter(InvoiceModel.id == invoice_id).first()
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
    table_data = [["Product", "Qty", "Price", "Total"]]
    for item in items:
        table_data.append([item.get("product_name"), str(item.get("quantity")), str(item.get("price")), str(item.get("total"))])

    table = Table(table_data)
    table.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.grey), ("GRID", (0, 0), (-1, -1), 1, colors.black)]))
    elements.append(table)
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(f"Total: Rs. {invoice.total}", styles["Heading2"]))

    doc.build(elements)
    buffer.seek(0)

    return StreamingResponse(buffer, media_type="application/pdf", headers={"Content-Disposition": f"attachment; filename=Invoice-{invoice.invoice_number}.pdf"})
