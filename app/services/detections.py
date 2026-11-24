import uuid
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.case import CaseStatus
from app.models.detection import Detection
from app.models.user import User
from app.schemas.detection import DetectionCreate
from app.services import cases as case_service


class DetectionNotFoundError(Exception):
    """The detection does not exist."""


class CaseNotApprovedError(Exception):
    """Detections can only be recorded against approved cases."""


class DetectionAlreadyReviewedError(Exception):
    """Only an unreviewed detection can receive a review decision."""


def record_detection(db: Session, admin: User, case_id: uuid.UUID, data: DetectionCreate) -> Detection:
    # Reuse the case service's row lock so a simultaneous status transition cannot
    # approve/reject/close the case between the check and the insert.
    case = case_service.get_case_for_update(db, case_id)
    if case.status != CaseStatus.APPROVED.value:
        db.rollback()
        raise CaseNotApprovedError
    detection = Detection(
        case_id=case.id,
        detected_at=data.detected_at,
        source=data.source,
        camera_id=data.camera_id,
        location_text=data.location_text,
        latitude=data.latitude,
        longitude=data.longitude,
        confidence=data.confidence,
        frame_reference=data.frame_reference,
        recorded_by_id=admin.id,
    )
    db.add(detection)
    db.commit()
    db.refresh(detection)
    return detection


def _filters(
    *, camera_id: str | None, review_status: str | None, min_confidence: float | None,
    detected_from: datetime | None, detected_to: datetime | None, case_id: uuid.UUID | None = None,
):
    conditions = []
    if camera_id is not None:
        conditions.append(Detection.camera_id == camera_id)
    if review_status is not None:
        conditions.append(Detection.review_status == review_status)
    if min_confidence is not None:
        conditions.append(Detection.confidence >= min_confidence)
    if detected_from is not None:
        conditions.append(Detection.detected_at >= detected_from)
    if detected_to is not None:
        conditions.append(Detection.detected_at <= detected_to)
    if case_id is not None:
        conditions.append(Detection.case_id == case_id)
    return conditions


def _page(db: Session, conditions: list, *, limit: int, offset: int):
    total = db.scalar(select(func.count()).select_from(Detection).where(*conditions)) or 0
    items = db.scalars(
        select(Detection).where(*conditions)
        .order_by(Detection.detected_at.desc(), Detection.id.desc())
        .limit(limit).offset(offset)
    ).all()
    return list(items), total


def list_case_detections(
    db: Session, case_id: uuid.UUID, *, camera_id: str | None = None,
    review_status: str | None = None, min_confidence: float | None = None,
    detected_from: datetime | None = None, detected_to: datetime | None = None,
    limit: int, offset: int,
):
    case_service.get_case(db, case_id)
    conditions = _filters(
        camera_id=camera_id, review_status=review_status, min_confidence=min_confidence,
        detected_from=detected_from, detected_to=detected_to, case_id=case_id,
    )
    return _page(db, conditions, limit=limit, offset=offset)


def list_all_detections(
    db: Session, *, case_id: uuid.UUID | None = None, camera_id: str | None = None,
    review_status: str | None = None, min_confidence: float | None = None,
    detected_from: datetime | None = None, detected_to: datetime | None = None,
    limit: int, offset: int,
):
    conditions = _filters(
        camera_id=camera_id, review_status=review_status, min_confidence=min_confidence,
        detected_from=detected_from, detected_to=detected_to, case_id=case_id,
    )
    return _page(db, conditions, limit=limit, offset=offset)


def get_detection(db: Session, detection_id: uuid.UUID) -> Detection:
    detection = db.get(Detection, detection_id)
    if detection is None:
        raise DetectionNotFoundError
    return detection


def review_detection(
    db: Session, admin: User, detection_id: uuid.UUID, review_status: str, note: str | None
) -> Detection:
    detection = db.scalar(
        select(Detection).where(Detection.id == detection_id).with_for_update()
        .execution_options(populate_existing=True)
    )
    if detection is None:
        raise DetectionNotFoundError
    if detection.review_status != "unreviewed":
        db.rollback()
        raise DetectionAlreadyReviewedError
    detection.review_status = review_status
    detection.reviewed_by_id = admin.id
    detection.reviewed_at = datetime.now(timezone.utc)
    detection.review_note = note
    db.commit()
    db.refresh(detection)
    return detection


def list_owner_confirmed_detections(
    db: Session, case_id: uuid.UUID, owner: User, *, limit: int, offset: int
):
    # Ownership is checked before querying detection rows: other users receive the same 404.
    case_service.get_case_for_owner(db, case_id, owner)
    conditions = [Detection.case_id == case_id, Detection.review_status == "confirmed"]
    return _page(db, conditions, limit=limit, offset=offset)
