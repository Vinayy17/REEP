from fastapi import APIRouter

from app.api_router_source import api_router as source_api_router


def build_router(paths: set[str]) -> APIRouter:
    router = APIRouter()
    for route in source_api_router.routes:
        if getattr(route, "path", None) in paths:
            router.routes.append(route)
    return router
