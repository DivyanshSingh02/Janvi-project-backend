import logging
from datetime import datetime
from typing import Annotated, Literal
from uuid import UUID

from fastapi import APIRouter, Query, status
from fastapi.exceptions import RequestValidationError

from app.api.deps import AdminUser, DbSession
from app.schemas.common import Page
from app.schemas.detection import DetectionCreate, DetectionRead, DetectionReview
from app.services import detections as detection_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["Detections"])


def _validate_time_filters(detected_from: datetime | None, detected_to: datetime | None) -> None:
    for value in (detected_from, detected_to):
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            field = "detected_from" if value is detected_from else "detected_to"
            raise RequestValidationError([{
                "type": "value_error", "loc": ("query", field),
                "msg": "Value error, a timezone is required",
            }])
    if detected_from is not None and detected_to is not None and detected_from > detected_to:
        raise RequestValidationError([{
            "type": "value_error", "loc": ("query", "detected_from"),
            "msg": "Value error, detected_from must not be after detected_to",
        }])


@router.post("/cases/{case_id}/detections", response_model=DetectionRead, status_code=status.HTTP_201_CREATED)
def create_detection(case_id: UUID, payload: DetectionCreate, admin: AdminUser, db: DbSession):
    detection = detection_service.record_detection(db, admin, case_id, payload)
    logger.info(
        "detection_recorded",
        extra={"detection_id": str(detection.id), "case_id": str(case_id), "admin_id": str(admin.id)},
    )
    return detection


@router.get("/cases/{case_id}/detections", response_model=Page[DetectionRead])
def case_detection_history(
    case_id: UUID, admin: AdminUser, db: DbSession,
    camera_id: str | None = None,
    review_status: Literal["unreviewed", "confirmed", "dismissed"] | None = None,
    min_confidence: Annotated[float | None, Query(ge=0, le=1)] = None,
    detected_from: datetime | None = None, detected_to: datetime | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    _validate_time_filters(detected_from, detected_to)
    items, total = detection_service.list_case_detections(
        db, case_id, camera_id=camera_id, review_status=review_status,
        min_confidence=min_confidence, detected_from=detected_from,
        detected_to=detected_to, limit=limit, offset=offset,
    )
    return Page[DetectionRead](items=items, total=total, limit=limit, offset=offset)


@router.get("/detections", response_model=Page[DetectionRead])
def global_detection_history(
    admin: AdminUser, db: DbSession,
    case_id: UUID | None = None, camera_id: str | None = None,
    review_status: Literal["unreviewed", "confirmed", "dismissed"] | None = None,
    min_confidence: Annotated[float | None, Query(ge=0, le=1)] = None,
    detected_from: datetime | None = None, detected_to: datetime | None = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
):
    _validate_time_filters(detected_from, detected_to)
    items, total = detection_service.list_all_detections(
        db, case_id=case_id, camera_id=camera_id, review_status=review_status,
        min_confidence=min_confidence, detected_from=detected_from,
        detected_to=detected_to, limit=limit, offset=offset,
    )
    return Page[DetectionRead](items=items, total=total, limit=limit, offset=offset)


@router.get("/detections/{detection_id}", response_model=DetectionRead)
def detection_detail(detection_id: UUID, admin: AdminUser, db: DbSession):
    return detection_service.get_detection(db, detection_id)


@router.patch("/detections/{detection_id}/review", response_model=DetectionRead)
def review_detection(detection_id: UUID, payload: DetectionReview, admin: AdminUser, db: DbSession):
    detection = detection_service.review_detection(
        db, admin, detection_id, payload.review_status, payload.note
    )
    logger.info(
        "detection_reviewed",
        extra={"detection_id": str(detection.id), "case_id": str(detection.case_id), "admin_id": str(admin.id)},
    )
    return detection
