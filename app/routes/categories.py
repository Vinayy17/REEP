from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.config import IST
from app.db import get_db
from app.models import CategoryModel, UserModel
from app.schemas.catalog import Category, CategoryCreate
from app.utils import get_current_user

router = APIRouter()


@router.get("/api/categories/autocomplete")
def categories_autocomplete(
    q: Optional[str] = Query(None, min_length=2),
    limit: int = Query(20, ge=5, le=50),
    offset: int = Query(0, ge=0),
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(CategoryModel)

    if q:
        q = q.strip().lower()
        query = query.filter(func.lower(CategoryModel.name).like(f"{q}%"))
    else:
        query = query.order_by(CategoryModel.created_at.desc())

    rows = query.order_by(CategoryModel.name.asc()).offset(offset).limit(limit).all()
    return [{"label": cat.name, "value": cat.id} for cat in rows]


@router.get("/api/categories", response_model=List[Category])
def get_categories(
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    categories = db.query(CategoryModel).all()
    return [
        Category(
            id=cat.id,
            name=cat.name,
            description=cat.description,
            created_at=cat.created_at.isoformat(),
        )
        for cat in categories
    ]


@router.post("/api/categories", response_model=Category)
def create_category(
    category_data: CategoryCreate,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    new_category = CategoryModel(
        name=category_data.name,
        description=category_data.description,
        created_at=datetime.now(IST),
    )
    db.add(new_category)
    db.commit()
    db.refresh(new_category)

    return Category(
        id=new_category.id,
        name=new_category.name,
        description=new_category.description,
        created_at=new_category.created_at.isoformat(),
    )


@router.delete("/api/categories/{category_id}")
def delete_category(
    category_id: str,
    current_user: UserModel = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    category = db.query(CategoryModel).filter(CategoryModel.id == category_id).first()
    if not category:
        raise HTTPException(status_code=404, detail="Category not found")

    db.delete(category)
    db.commit()
    return {"message": "Category deleted successfully"}
