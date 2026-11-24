import logging
from io import BytesIO
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Query, Request, Response
from fastapi.exceptions import RequestValidationError

from app.api.deps import AdminUser, DbSession
from app.api.rate_limits import limit_admin
from app.api.recognition_errors import RECOGNITION_ERRORS
from app.api.routes.photos import StorageDep, _HANDLED as _IMAGE_ERRORS, image_response
from app.api.routes.search import SearchLimit
from app.recognition import RecognitionDep
from app.schemas.common import Page
from app.schemas.found_person_report import (
    AdminFoundPersonReportRead,
    FoundPersonReportDeleteRead,
    FoundPersonReportReview,
    FoundPersonReportStatus,
)
from app.schemas.search import SearchResponse
from app.services import found_person_reports as report_service
from app.services import search as search_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin/found-person-reports", tags=["Found person reports"])


@router.get("", response_model=Page[AdminFoundPersonReportRead])
def list_found_person_reports(
    admin: AdminUser,
    db: DbSession,
    review_status: FoundPersonReportStatus | None = Query(default="pending"),
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    rows, total = report_service.list_admin_reports(
        db, review_status=review_status, limit=limit, offset=offset
    )
    return Page[AdminFoundPersonReportRead](
        items=[AdminFoundPersonReportRead.from_report(report, reporter) for report, reporter in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{report_id}/photo", response_class=Response, responses={200: {"content": {"image/jpeg": {}}}})
def download_found_person_report_photo(
    report_id: UUID,
    admin: AdminUser,
    db: DbSession,
    storage: StorageDep,
):
    report = report_service.get_report_for_admin(db, report_id)
    data = report_service.get_report_photo(report, storage)
    return image_response(data, report.content_type)


@router.post("/{report_id}/search", response_model=SearchResponse)
def search_found_person_report(
    report_id: UUID,
    admin: AdminUser,
    db: DbSession,
    storage: StorageDep,
    recognition: RecognitionDep,
    request: Request,
    limit: SearchLimit,
    min_similarity: Annotated[
        float | None, Query(ge=-1.0, le=1.0, description="Optional: drop candidates below this cosine similarity")
    ] = None,
):
    """Admin-only matching against approved cases; results are never returned to the submitting reporter."""
    limit_admin(request, admin.id)
    report = report_service.get_report_for_admin(db, report_id)
    image = report_service.get_report_photo(report, storage)
    try:
        outcome = search_service.search_by_face(
            db, admin, recognition, BytesIO(image), limit=limit, min_similarity=min_similarity
        )
    except _IMAGE_ERRORS + RECOGNITION_ERRORS as exc:
        logger.info(
            "found_person_report_search_rejected",
            extra={"report_id": str(report_id), "admin_id": str(admin.id), "reason": type(exc).__name__},
        )
        raise
    logger.info(
        "found_person_report_searched",
        extra={
            "report_id": str(report_id),
            "admin_id": str(admin.id),
            "result_count": len(outcome.candidates),
            "candidates_considered": outcome.candidates_considered,
        },
    )
    return SearchResponse.from_outcome(outcome)


@router.patch("/{report_id}/review", response_model=AdminFoundPersonReportRead)
def review_found_person_report(
    report_id: UUID,
    payload: FoundPersonReportReview,
    admin: AdminUser,
    db: DbSession,
):
    report = report_service.review_report(
        db, admin, report_id, payload.review_status, payload.note
    )
    reporter = report_service.get_reporter_for_admin(db, report.reporter_id)
    logger.info(
        "found_person_report_reviewed",
        extra={
            "report_id": str(report.id),
            "admin_id": str(admin.id),
            "review_status": report.review_status,
        },
    )
    return AdminFoundPersonReportRead.from_report(report, reporter)


@router.delete("/{report_id}", response_model=FoundPersonReportDeleteRead)
def delete_found_person_report(
    report_id: UUID,
    admin: AdminUser,
    db: DbSession,
    storage: StorageDep,
    confirm: bool = Query(False, description="Must be true to permanently delete the report and its photo"),
):
    if confirm is not True:
        raise RequestValidationError([{
            "type": "value_error",
            "loc": ("query", "confirm"),
            "msg": "Value error, confirm=true is required for permanent deletion",
        }])
    photo_deleted = report_service.delete_report(db, storage, report_id)
    logger.info(
        "found_person_report_deleted",
        extra={"report_id": str(report_id), "admin_id": str(admin.id), "photo_deleted": photo_deleted},
    )
    return FoundPersonReportDeleteRead(report_id=report_id, photo_deleted=photo_deleted)
