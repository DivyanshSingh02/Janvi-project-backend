import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, HTTPException, Query, Response, status
from fastapi.exceptions import RequestValidationError

from app.api.deps import AdminUser, DbSession
from app.api.routes.photos import _HANDLED as _PHOTO_ERRORS
from app.api.routes.photos import StorageDep, image_response
from app.models.case import CaseStatus
from app.recognition import RecognitionDep
from app.schemas.common import Page
from app.schemas.review import (
    AdminCaseDetail,
    AdminCaseSummary,
    ApprovedCaseDetail,
    ApproveRequest,
    RejectRequest,
    StatusChangeRequest,
    CaseHardDeleteRead,
)
from app.services import cases as case_service
from app.services import embeddings as embedding_service
from app.services import photos as photo_service
from app.services import review as review_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["Admin"])

_REVIEW_ERRORS = (
    review_service.CaseHasNoPhotosError,
    review_service.ReasonRequiredError,
    review_service.InvalidReasonError,
    review_service.InvalidTransitionError,
)


def _change_status(
    db, admin, case_id: UUID, target: CaseStatus, reason: str | None, **after_approval_services
) -> tuple[review_service.StatusChange, review_service.CaseDetail]:
    """Shared by approve, reject and PATCH status. Logs ids and statuses only - NEVER the reason text.
    `after_approval_services` (recognition=..., storage=...) are only passed by approve."""
    try:
        change = review_service.change_status(db, admin, case_id, target, reason, **after_approval_services)
    except _REVIEW_ERRORS as exc:
        extra = {"case_id": str(case_id), "admin_id": str(admin.id), "error": type(exc).__name__}
        if isinstance(exc, review_service.InvalidTransitionError):
            extra.update(from_status=exc.from_status.value, to_status=exc.to_status.value)
        logger.info("case_status_change_rejected", extra=extra)
        raise
    logger.info(
        "case_status_changed",
        extra={
            "case_id": str(change.case_id),
            "admin_id": str(admin.id),
            "from_status": change.from_status.value,
            "to_status": change.to_status.value,
        },
    )
    return change, review_service.get_case_detail(db, case_id)


# ---------------------------------------------------------------- reading
@router.get("/cases", response_model=Page[AdminCaseSummary])
def list_cases_for_review(
    admin: AdminUser,
    db: DbSession,
    status_filter: Annotated[CaseStatus, Query(alias="status")] = CaseStatus.PENDING,
    sort: Annotated[review_service.ReviewSort, Query()] = review_service.ReviewSort.OLDEST,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    """The review queue: pending cases by default, oldest first (first in, first out)."""
    rows, total = review_service.list_cases_for_review(
        db, status=status_filter, sort=sort, limit=limit, offset=offset
    )
    return Page[AdminCaseSummary](
        items=[AdminCaseSummary.from_case(case, photo_count) for case, photo_count in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/cases/{case_id}", response_model=AdminCaseDetail)
def get_case_for_review(case_id: UUID, admin: AdminUser, db: DbSession):
    try:
        return AdminCaseDetail.from_detail(review_service.get_case_detail(db, case_id))
    except case_service.CaseNotFoundError:
        raise


@router.get(
    "/cases/{case_id}/photos/{photo_id}",
    response_class=Response,
    responses={200: {"content": {"image/jpeg": {}}}},
)
def download_case_photo(case_id: UUID, photo_id: UUID, admin: AdminUser, db: DbSession, storage: StorageDep):
    """The stored, metadata-free JPEG of any case, for admins only. Same headers as the owner's endpoint."""
    try:
        photo = photo_service.get_photo_for_admin(db, case_id, photo_id)
        data = photo_service.read_photo(storage, photo)
    except _PHOTO_ERRORS as exc:
        raise
    return image_response(data, photo.content_type)


# ---------------------------------------------------------------- changing the status
@router.post("/cases/{case_id}/approve", response_model=ApprovedCaseDetail)
def approve_case(
    case_id: UUID,
    admin: AdminUser,
    db: DbSession,
    recognition: RecognitionDep,
    storage: StorageDep,
    payload: ApproveRequest | None = None,
):
    """pending -> approved. Needs at least one photo. The reason is optional. After the approval is saved, the
    face embeddings are generated as a separate step; with no recognition model the case simply stays approved with
    its photos pending (see "embeddings" in the response). An approval is never undone because of recognition."""
    change, detail = _change_status(
        db,
        admin,
        case_id,
        CaseStatus.APPROVED,
        payload.reason if payload else None,
        recognition=recognition,
        storage=storage,
    )
    summary = change.embeddings or embedding_service.summarise(db, case_id)     # None only if the step itself broke
    return ApprovedCaseDetail.from_approval(detail, summary)


@router.post("/cases/{case_id}/reject", response_model=AdminCaseDetail)
def reject_case(case_id: UUID, payload: RejectRequest, admin: AdminUser, db: DbSession):
    """pending -> rejected. The reason is required; the reporter can read it."""
    return AdminCaseDetail.from_detail(_change_status(db, admin, case_id, CaseStatus.REJECTED, payload.reason)[1])


@router.patch("/cases/{case_id}/status", response_model=AdminCaseDetail)
def change_case_status(case_id: UUID, payload: StatusChangeRequest, admin: AdminUser, db: DbSession):
    """found / closed (and any invalid target, which answers 409). approved and rejected are refused with 422."""
    return AdminCaseDetail.from_detail(_change_status(db, admin, case_id, payload.status, payload.reason)[1])


@router.delete("/cases/{case_id}", response_model=CaseHardDeleteRead)
def hard_delete_case(
    case_id: UUID, admin: AdminUser, db: DbSession, storage: StorageDep,
    confirm: bool = Query(False, description="Must be true to permanently delete this case"),
):
    """Permanently remove a case and dependent data. Files are deleted only after DB commit."""
    if confirm is not True:
        raise RequestValidationError([{
            "type": "value_error", "loc": ("query", "confirm"),
            "msg": "Value error, confirm=true is required for permanent deletion",
        }])
    result = case_service.hard_delete_case(db, storage, actor=admin, case_id=case_id)
    logger.info(
        "case_hard_deleted",
        extra={
            "case_id": str(result.case_id),
            "admin_id": str(admin.id),
            "photos_deleted": result.photos_deleted,
            "embeddings_deleted": result.embeddings_deleted,
            "detections_deleted": result.detections_deleted,
            "status_events_deleted": result.status_events_deleted,
            "failed_file_deletions": len(result.failed_photo_ids),
        },
    )
    return CaseHardDeleteRead(
        case_id=result.case_id,
        photos_deleted=result.photos_deleted,
        embeddings_deleted=result.embeddings_deleted,
        detections_deleted=result.detections_deleted,
        status_events_deleted=result.status_events_deleted,
        files_deleted=result.files_deleted,
        failed_photo_ids=result.failed_photo_ids,
    )
