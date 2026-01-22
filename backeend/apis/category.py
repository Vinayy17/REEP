from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import uuid
from datetime import datetime

from models.category import CategoryModel
from schemas.category import Category, CategoryCreate
from deps import get_db, get_current_user
from core.time import IST

router = APIRouter(prefix="/categories", tags=["Categories"])


@router.get("", response_model=list[Category])
def get_categories(
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return [
        Category(
            id=c.id,
            name=c.name,
            description=c.description,
            created_at=c.created_at.isoformat(),
        )
        for c in db.query(CategoryModel).all()
    ]


@router.post("", response_model=Category)
def create_category(
    data: CategoryCreate,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    category = CategoryModel(
        id=str(uuid.uuid4()),
        name=data.name,
        description=data.description,
        created_at=datetime.now(IST),
    )
    db.add(category)
    db.commit()
    db.refresh(category)

    return Category(
        id=category.id,
        name=category.name,
        description=category.description,
        created_at=category.created_at.isoformat(),
    )


@router.delete("/{category_id}")
def delete_category(
    category_id: str,
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    category = db.query(CategoryModel).filter_by(id=category_id).first()
    if not category:
        raise HTTPException(status_code=404, detail="Category not found")

    db.delete(category)
    db.commit()
    return {"message": "Category deleted"}
