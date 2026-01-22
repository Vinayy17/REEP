from pydantic import BaseModel
from typing import Optional

class Category(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    created_at: str

class CategoryCreate(BaseModel):
    name: str
    description: Optional[str] = None
