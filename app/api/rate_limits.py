import hashlib

from fastapi import Request

from app.core.config import get_settings
from app.core.rate_limit import RateLimit


def _client_ip(request: Request) -> str:
    return request.client.host if request.client is not None else "unknown"


def limit_login(request: Request, email: str) -> None:
    settings = get_settings()
    limiter = request.app.state.rate_limiter
    limiter.check("login_ip", _client_ip(request), RateLimit(settings.login_ip_limit, settings.login_window_seconds))
    email_hash = hashlib.sha256(email.strip().casefold().encode("utf-8")).hexdigest()
    limiter.check("login_email", email_hash, RateLimit(settings.login_email_limit, settings.login_window_seconds))


def limit_registration(request: Request) -> None:
    settings = get_settings()
    request.app.state.rate_limiter.check(
        "registration_ip", _client_ip(request),
        RateLimit(settings.registration_ip_limit, settings.registration_window_seconds),
    )


def limit_admin(request: Request, admin_id, *, embeddings: bool = False) -> None:
    settings = get_settings()
    count = settings.embedding_admin_limit if embeddings else settings.face_search_admin_limit
    scope = "embedding_admin" if embeddings else "face_search_admin"
    request.app.state.rate_limiter.check(
        scope, str(admin_id), RateLimit(count, settings.admin_recognition_window_seconds)
    )


def limit_photo_upload(request: Request, user_id) -> None:
    settings = get_settings()
    request.app.state.rate_limiter.check(
        "photo_upload_user", str(user_id),
        RateLimit(settings.photo_upload_user_limit, settings.photo_upload_window_seconds),
    )
