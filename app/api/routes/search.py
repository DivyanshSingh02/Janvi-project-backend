import logging
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError

from app.api.deps import AdminUser, DbSession
from app.api.rate_limits import limit_admin
from app.api.recognition_errors import RECOGNITION_ERRORS
from app.api.routes.photos import _HANDLED as _IMAGE_ERRORS
from app.api.routes.photos import _BodyLimitedRoute
from app.core.config import get_settings
from app.recognition import RecognitionDep
from app.schemas.search import SearchResponse
from app.services import search as search_service

logger = logging.getLogger(__name__)

# Same early Content-Length check as the photo upload (a search sends one image, so the limit is generous).
router = APIRouter(tags=["Search"], route_class=_BodyLimitedRoute)


def _search_limit(limit: Annotated[int | None, Query(ge=1, description="Maximum number of candidates")] = None) -> int:
    """SEARCH_DEFAULT_LIMIT when absent; at most SEARCH_MAX_LIMIT (a 422 in the usual error shape otherwise)."""
    settings = get_settings()
    if limit is None:
        return settings.search_default_limit
    if limit > settings.search_max_limit:
        raise RequestValidationError(
            [
                {
                    "type": "less_than_equal",
                    "loc": ("query", "limit"),
                    "msg": f"Input should be less than or equal to {settings.search_max_limit}",
                    "ctx": {"le": settings.search_max_limit},
                }
            ]
        )
    return limit


SearchLimit = Annotated[int, Depends(_search_limit)]


@router.post("/search", response_model=SearchResponse)
def search_by_face(
    admin: AdminUser,
    db: DbSession,
    request: Request,
    recognition: RecognitionDep,
    file: UploadFile,
    limit: SearchLimit,
    min_similarity: Annotated[
        float | None, Query(ge=-1.0, le=1.0, description="Optional: drop candidates below this cosine similarity")
    ] = None,
):
    """Admin only. Multipart form with ONE image in the field "file" that shows exactly one face. The image is
    validated and normalised exactly like a case photo, then compared with the embeddings of APPROVED cases made
    by the same model. The answer is a ranked list of LEADS, not proof of identity: a human must review them."""
    limit_admin(request, admin.id)
    try:
        outcome = search_service.search_by_face(
            db, admin, recognition, file.file, limit=limit, min_similarity=min_similarity
        )
    except _IMAGE_ERRORS as exc:
        logger.info("face_search_rejected", extra={"admin_id": str(admin.id), "reason": type(exc).__name__})
        raise
    except RECOGNITION_ERRORS as exc:
        logger.info("face_search_rejected", extra={"admin_id": str(admin.id), "reason": type(exc).__name__})
        raise
    # Ids and counts only: never the image, the embedding or anything about the cases.
    logger.info(
        "face_search",
        extra={
            "admin_id": str(admin.id),
            "result_count": len(outcome.candidates),
            "candidates_considered": outcome.candidates_considered,
            "stale_embeddings_skipped": outcome.stale_embeddings_skipped,
        },
    )
    return SearchResponse.from_outcome(outcome)
