import logging
import time
import uuid

from fastapi import Request
from starlette.responses import Response

from app.api.errors import error_response, domain_exception_handler
from app.core.config import get_settings

logger = logging.getLogger("app.request")


def _security_headers(response: Response, *, production: bool) -> None:
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
    content_type = response.headers.get("content-type", "").lower()
    if "application/json" in content_type:
        response.headers["Cache-Control"] = "no-store"
    if production:
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"


async def request_logging_middleware(request: Request, call_next):
    """Request id, global streaming body cap, safe request logging and response security headers."""
    settings = get_settings()
    request_id = uuid.uuid4().hex
    request.state.request_id = request_id
    started = time.perf_counter()
    fields = {"request_id": request_id, "method": request.method, "path": request.url.path}

    # Reject declared oversize bodies before reading them.
    declared = request.headers.get("content-length")
    if declared is not None:
        try:
            declared_size = int(declared)
        except ValueError:
            response = error_response(request, 400, "bad_content_length", "Invalid Content-Length")
            _security_headers(response, production=settings.app_env == "production")
            response.headers["X-Request-ID"] = request_id
            return response
        if declared_size < 0 or declared_size > settings.max_request_body_bytes:
            response = error_response(request, 413, "request_too_large", "The request body is too large")
            _security_headers(response, production=settings.app_env == "production")
            return response

    # Read through ASGI streaming (including chunked requests), enforce the same limit,
    # then replay the already bounded body for FastAPI's normal parsers.
    body_buffer = bytearray()
    async for chunk in request.stream():
        body_buffer.extend(chunk)
        if len(body_buffer) > settings.max_request_body_bytes:
            response = error_response(request, 413, "request_too_large", "The request body is too large")
            _security_headers(response, production=settings.app_env == "production")
            return response
    body = bytes(body_buffer)
    request._body = body
    sent = False

    async def replay_body():
        nonlocal sent
        if not sent:
            sent = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    request._receive = replay_body

    try:
        response = await call_next(request)
    except Exception as exc:
        # Domain exceptions are mapped centrally; unexpected errors are logged by the handler with
        # this request id, while only a generic message is returned to the client.
        response = await domain_exception_handler(request, exc)

    # Some outer Starlette middleware (notably TrustedHost/CORS preflight) emits plain-text
    # errors. Normalize those too, so every client-visible error has the same JSON envelope.
    if response.status_code >= 400 and "application/json" not in response.headers.get("content-type", "").lower():
        status_code = response.status_code
        code = {400: "bad_request", 403: "forbidden", 404: "not_found", 413: "request_too_large"}.get(
            status_code, f"http_{status_code}"
        )
        preserved = {
            name: response.headers[name]
            for name in ("WWW-Authenticate", "Retry-After")
            if name in response.headers
        }
        response = error_response(request, status_code, code, "The request was rejected", headers=preserved)

    duration_ms = round((time.perf_counter() - started) * 1000, 1)
    fields["status_code"] = response.status_code
    fields["duration_ms"] = duration_ms
    logger.info("%s %s %s %sms", request.method, request.url.path, response.status_code,
                duration_ms, extra=fields)
    response.headers["X-Request-ID"] = request_id
    _security_headers(response, production=settings.app_env == "production")
    return response
