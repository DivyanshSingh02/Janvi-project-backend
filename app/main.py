from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from app.api.errors import domain_exception_handler, http_exception_handler, validation_exception_handler
from app.api.middleware import request_logging_middleware
from app.api.routes import (
    admin,
    admin_audit,
    admin_cases,
    admin_detections,
    admin_embeddings,
    admin_found_person_reports,
    auth,
    cases,
    detections,
    found_person_reports,
    health,
    photos,
    search,
)
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.core.rate_limit import SlidingWindowRateLimiter
from starlette.exceptions import HTTPException


def create_app() -> FastAPI:
    settings = get_settings()
    settings.validate_production()
    configure_logging(settings.log_level, settings.log_json)

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        docs_url="/docs" if settings.enable_api_docs else None,
        redoc_url=None,
        openapi_url="/openapi.json" if settings.enable_api_docs else None,
    )
    app.state.rate_limiter = SlidingWindowRateLimiter()

    # Trusted hosts protect Host-header routing. CORS is deliberately absent when the allow-list is empty.
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.trusted_hosts)
    if settings.cors_allowed_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_allowed_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type"],
            expose_headers=["X-Request-ID", "Retry-After"],
        )

    app.middleware("http")(request_logging_middleware)
    app.add_exception_handler(HTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, domain_exception_handler)

    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(admin.router)
    app.include_router(admin_audit.router)
    app.include_router(admin_cases.router)
    app.include_router(admin_embeddings.router)
    app.include_router(admin_detections.router)
    app.include_router(admin_found_person_reports.router)
    app.include_router(cases.router)
    app.include_router(detections.router)
    app.include_router(found_person_reports.router)
    app.include_router(photos.router)
    app.include_router(search.router)
    return app


app = create_app()
