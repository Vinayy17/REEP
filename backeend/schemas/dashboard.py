from pydantic import BaseModel
from typing import List

class DashboardStats(BaseModel):
    total_sales: float
    total_orders: int
    total_customers: int
    low_stock_items: int

class SalesChartItem(BaseModel):
    name: str
    total: float
    paid: float
    pending: float
    overdue: float
