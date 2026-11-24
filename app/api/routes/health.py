import logging
import uuid

from fastapi import APIRouter, Response

from app.core.config import get_settings
from app.db.session import check_database
from app.recognition import get_recognition_service
from app.schemas.health import HealthResponse
from app.storage import get_storage

logger = logging.getLogger(__name__)
router = APIRouter(tags=["System"])


def check_storage_writable() -> bool:
    """Write and remove a tiny opaque probe object; never logs or returns its key."""
    storage = None
    key = f"health-check/{uuid.uuid4().hex}"
    try:
        storage = get_storage()
        storage.save(key, b"health-check", "application/octet-stream")
        storage.delete(key)
        return True
    except Exception:
        if storage is not None:
            try:
                storage.delete(key)
            except Exception:
                pass
        logger.warning("storage_health_check_failed")
        return False


@router.get(
    "/health",
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse, "description": "A critical dependency is unavailable"}},
)
def health(response: Response) -> HealthResponse:
    db_ok = check_database()
    storage_ok = check_storage_writable()
    try:
        recognition_ready = bool(get_recognition_service().is_configured)
    except Exception:
        recognition_ready = False
    healthy = db_ok and storage_ok
    if not healthy:
        response.status_code = 503
    return HealthResponse(
        status="ok" if healthy else "degraded",
        version=get_settings().app_version,
        database="ok" if db_ok else "unavailable",
        storage="ok" if storage_ok else "unavailable",
        recognition="ready" if recognition_ready else "unconfigured",
    )
