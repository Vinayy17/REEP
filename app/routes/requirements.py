import json
import logging
import math
from datetime import datetime, timedelta
from typing import Optional

import cloudinary.uploader
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from app.core.config import IST
from app.db import get_db
from app.models import RequirementModel, UserModel
from app.schemas.requirements import RequirementCreate
from app.utils import get_current_user, upload_image_to_cloudinary

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/api/upload/requirement-image")
def upload_requirement_image(
    file: UploadFile = File(...),
    current_user: UserModel = Depends(get_current_user),
):
    if current_user.role not in ["admin", "store_handler"]:
        raise HTTPException(status_code=403, detail="Not allowed")

    url = upload_image_to_cloudinary(file.file, folder="requirements")
    return {"url": url}


@router.delete("/api/requirements/cleanup")
def cleanup_old_requirements(
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin only")

    cutoff_date = datetime.now(IST) - timedelta(days=90)

    old_reqs = (
        db.query(RequirementModel)
        .filter(
            RequirementModel.status.in_(["completed", "rejected"]),
            RequirementModel.created_at < cutoff_date,
        )
        .all()
    )

    deleted_count = 0

    for req in old_reqs:
        try:
            items = json.loads(req.requirement_items or "[]")
        except Exception:
            items = []

        for item in items:
            image_url = item.get("image_url")
            if image_url:
                try:
                    public_id = image_url.split("/")[-1].split(".")[0]
                    cloudinary.uploader.destroy(f"requirements/{public_id}")
                except Exception as exc:
                    logger.warning("Cloudinary delete failed for %s: %s", image_url, exc)

        db.delete(req)
        deleted_count += 1

    db.commit()

    return {"message": "Old requirements cleaned successfully", "deleted": deleted_count}


@router.post("/api/requirements", status_code=201)
def create_requirement(
    data: RequirementCreate,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    requirement = RequirementModel(
        customer_name=data.customer_name,
        customer_phone=data.customer_phone,
        requirement_items=json.dumps(
            [{"text": item.text, "image_url": item.image_url} for item in data.requirement_items]
        ),
        priority=data.priority,
        status="pending",
        created_by=current_user.id,
        created_by_name=current_user.name,
        created_at=datetime.now(IST),
    )

    db.add(requirement)
    db.commit()
    db.refresh(requirement)

    return {"message": "Requirement created successfully"}


@router.get("/api/requirements")
def list_requirements(
    page: int = Query(1, ge=1),
    limit: int = Query(10, ge=5, le=50),
    status: Optional[str] = Query("all"),
    priority: Optional[str] = Query("all"),
    search: Optional[str] = None,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(RequirementModel)

    if status != "all":
        query = query.filter(RequirementModel.status == status)

    if priority != "all":
        query = query.filter(RequirementModel.priority == priority)

    if search:
        search_term = f"%{search.lower()}%"
        query = query.filter(
            or_(
                func.lower(RequirementModel.customer_name).like(search_term),
                func.lower(RequirementModel.customer_phone).like(search_term),
                func.lower(RequirementModel.created_by_name).like(search_term),
            )
        )

    total = query.count()
    rows = query.order_by(RequirementModel.created_at.desc()).offset((page - 1) * limit).limit(limit).all()

    data = []
    for row in rows:
        try:
            items = json.loads(row.requirement_items or "[]")
            if not isinstance(items, list):
                items = []
        except Exception:
            items = []

        data.append(
            {
                "id": row.id,
                "customer_name": row.customer_name,
                "customer_phone": row.customer_phone,
                "requirement_items": items,
                "priority": row.priority,
                "status": row.status,
                "created_by": row.created_by_name,
                "created_at": row.created_at.isoformat(),
            }
        )

    return {
        "data": data,
        "page": page,
        "limit": limit,
        "total": total,
        "total_pages": math.ceil(total / limit),
    }


@router.post("/api/requirements/{requirement_id}/complete")
def complete_requirement(
    requirement_id: str,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    req = db.query(RequirementModel).filter(RequirementModel.id == requirement_id).first()
    if not req:
        raise HTTPException(404, "Requirement not found")
    if req.status in ["completed", "rejected"]:
        raise HTTPException(400, "Requirement already finalized")

    req.status = "completed"
    req.completed_at = datetime.now(IST)

    db.commit()

    return {
        "message": "Requirement marked as completed",
        "requirement_id": req.id,
        "completed_at": req.completed_at.isoformat(),
    }


@router.post("/api/requirements/{requirement_id}/reject")
def reject_requirement(
    requirement_id: str,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    req = db.query(RequirementModel).filter(RequirementModel.id == requirement_id).first()
    if not req:
        raise HTTPException(404, "Requirement not found")
    if req.status in ["completed", "rejected"]:
        raise HTTPException(400, "Requirement already finalized")

    req.status = "rejected"
    req.completed_at = datetime.now(IST)

    db.commit()

    return {"message": "Requirement rejected"}
