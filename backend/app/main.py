from fastapi import FastAPI
from fastapi_limiter import FastAPILimiter
import redis.asyncio as redis
from starlette.middleware.cors import CORSMiddleware

from app.core.config import ALLOWED_ORIGINS
from app.routes import (
    accounting_router,
    auth_router,
    categories_router,
    dashboard_router,
    inventory_router,
    invoices_router,
    products_router,
    requirements_router,
)



def create_app() -> FastAPI:
    app = FastAPI(title="Lights Backend API")

    @app.on_event("startup")
    async def startup() -> None:
        redis_client = redis.from_url(
            "redis://localhost:6379",
            encoding="utf-8",
            decode_responses=True,
        )
        await FastAPILimiter.init(redis_client)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    for router in [
        auth_router,
        categories_router,
        products_router,
        inventory_router,
        requirements_router,
        invoices_router,
        dashboard_router,
        accounting_router,
    ]:
        app.include_router(router)

    app.add_middleware(
        CORSMiddleware,
        allow_credentials=True,
        allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGINS else ["*"],
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE"],
        allow_headers=["Authorization", "Content-Type", "Accept"],
    )

    return app


app = create_app()
