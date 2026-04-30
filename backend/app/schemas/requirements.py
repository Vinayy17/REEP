from typing import List, Optional

from pydantic import BaseModel


class RequirementItem(BaseModel):
    text: str
    image_url: Optional[str] = None


class RequirementCreate(BaseModel):
    customer_name: str
    customer_phone: str
    requirement_items: List[RequirementItem]
    priority: str = "normal"


class RequirementResponse(BaseModel):
    id: str
    customer_name: str
    customer_phone: str
    requirement_items: List[RequirementItem]
    priority: str
    status: str
    created_by: str
    created_at: str


__all__ = ["RequirementCreate", "RequirementItem", "RequirementResponse"]
