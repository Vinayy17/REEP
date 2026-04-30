from app.routes.auth import router as auth_router
from app.routes.accounting import router as accounting_router
from app.routes.categories import router as categories_router
from app.routes.dashboard import router as dashboard_router
from app.routes.inventory import router as inventory_router
from app.routes.invoices import router as invoices_router
from app.routes.products import router as products_router
from app.routes.requirements import router as requirements_router

__all__ = [
    "accounting_router",
    "auth_router",
    "categories_router",
    "dashboard_router",
    "inventory_router",
    "invoices_router",
    "products_router",
    "requirements_router",
]
