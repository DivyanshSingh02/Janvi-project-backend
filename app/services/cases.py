import logging
import uuid
from dataclasses import dataclass

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.models.case import CaseStatus, MissingCase
from app.models.case_status_event import CaseStatusEvent
from app.models.detection import Detection
from app.models.face_embedding import FaceEmbedding
from app.models.photo import CasePhoto
from app.models.user import User
from app.services import audit
from app.storage import Storage
from app.schemas.case import CaseCreate

logger = logging.getLogger(__name__)


class CaseNotFoundError(Exception):
    """The case does not exist OR belongs to someone else. The two are deliberately
    indistinguishable, so nobody can probe which case ids exist."""


def create_case(db: Session, reporter: User, data: CaseCreate) -> MissingCase:
    """Every new case starts as "pending" and belongs to the logged-in reporter.
    Neither value ever comes from the request body."""
    case = MissingCase(
        reporter_id=reporter.id,
        name=data.name,
        age=data.age,
        gender=data.gender.value,
        contact_phone=data.contact_phone,
        contact_email=data.contact_email,
        city=data.city,
        state=data.state,
        address=data.address,
        last_seen_location=data.last_seen_location,
        last_seen_date=data.last_seen_date,
        status=CaseStatus.PENDING.value,
    )
    db.add(case)
    db.commit()
    return case


def list_cases_for_owner(
    db: Session, owner: User, *, status: CaseStatus | None = None, limit: int, offset: int
) -> tuple[list[MissingCase], int]:
    """The owner's own cases, newest first. Returns (cases on this page, total matching)."""
    conditions = [MissingCase.reporter_id == owner.id]
    if status is not None:
        conditions.append(MissingCase.status == status.value)
    total = db.scalar(select(func.count()).select_from(MissingCase).where(*conditions)) or 0
    items = db.scalars(
        select(MissingCase)
        .where(*conditions)
        .order_by(MissingCase.created_at.desc(), MissingCase.id.desc())
        .limit(limit)
        .offset(offset)
    ).all()
    return list(items), total


def get_case_for_owner(db: Session, case_id: uuid.UUID, owner: User) -> MissingCase:
    """Raises CaseNotFoundError for an unknown id and for someone else's case alike."""
    case = db.scalar(
        select(MissingCase).where(MissingCase.id == case_id, MissingCase.reporter_id == owner.id)
    )
    if case is None:
        raise CaseNotFoundError
    return case


def get_case(db: Session, case_id: uuid.UUID) -> MissingCase:
    """NO ownership check: only for admin code, which has already passed the AdminUser dependency.
    Raises CaseNotFoundError for an unknown id."""
    case = db.get(MissingCase, case_id)
    if case is None:
        raise CaseNotFoundError
    return case


def get_case_for_update(db: Session, case_id: uuid.UUID) -> MissingCase:
    """NO ownership check (admin code and the embeddings service). SELECT ... FOR UPDATE: the case row stays
    locked until the end of the transaction, so nobody can change the same case at the same moment.
    populate_existing makes sure the values are the ones read under the lock, not an older copy held by the
    session. Raises CaseNotFoundError for an unknown id."""
    case = db.scalar(
        select(MissingCase)
        .where(MissingCase.id == case_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    if case is None:
        raise CaseNotFoundError
    return case


@dataclass(frozen=True)
class HardDeleteResult:
    case_id: uuid.UUID
    photos_deleted: int
    embeddings_deleted: int
    detections_deleted: int
    status_events_deleted: int
    files_deleted: int
    failed_photo_ids: list[uuid.UUID]


def hard_delete_case(db: Session, storage: Storage, *, actor: User, case_id: uuid.UUID) -> HardDeleteResult:
    """Permanently delete a case in one DB transaction, then remove its files after commit.

    Database child rows are explicitly removed in FK-safe order. Search logs survive with their
    top_case_id set NULL by the existing FK. Audit target IDs intentionally survive deletion.
    """
    try:
        case = get_case_for_update(db, case_id)
        photo_rows = list(db.scalars(select(CasePhoto).where(CasePhoto.case_id == case.id)))
        photo_keys = [(photo.id, photo.storage_key) for photo in photo_rows]

        detections_deleted = db.execute(delete(Detection).where(Detection.case_id == case.id)).rowcount or 0
        embeddings_deleted = db.execute(delete(FaceEmbedding).where(FaceEmbedding.case_id == case.id)).rowcount or 0
        status_events_deleted = db.execute(
            delete(CaseStatusEvent).where(CaseStatusEvent.case_id == case.id)
        ).rowcount or 0
        photos_deleted = db.execute(delete(CasePhoto).where(CasePhoto.case_id == case.id)).rowcount or 0
        db.execute(delete(MissingCase).where(MissingCase.id == case.id))
        audit.record_event(
            db, actor_id=actor.id, action="case_hard_deleted", target_type="case", target_id=case.id,
            details={
                "cases_deleted": 1,
                "photos_deleted": photos_deleted,
                "embeddings_deleted": embeddings_deleted,
                "detections_deleted": detections_deleted,
                "status_events_deleted": status_events_deleted,
            },
        )
        db.commit()
    except Exception:
        db.rollback()
        raise

    failed_photo_ids: list[uuid.UUID] = []
    files_deleted = 0
    for photo_id, storage_key in photo_keys:
        try:
            storage.delete(storage_key)
            files_deleted += 1
        except Exception:
            failed_photo_ids.append(photo_id)
            logger.error("hard_delete_photo_file_failed", extra={"photo_id": str(photo_id)})

    return HardDeleteResult(
        case_id=case_id,
        photos_deleted=photos_deleted,
        embeddings_deleted=embeddings_deleted,
        detections_deleted=detections_deleted,
        status_events_deleted=status_events_deleted,
        files_deleted=files_deleted,
        failed_photo_ids=failed_photo_ids,
    )
