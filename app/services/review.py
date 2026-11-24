"""Admin review of cases: the status transition rules (pure) and the database operations.

The rules at the top of this file (ALLOWED_TRANSITIONS, the reason rules) do not touch the database or HTTP, so
they are tested on their own. change_status() is the ONLY place that changes a case's status after creation:
it locks the case row, applies the rules, and writes the status, the review fields and the history row in one
transaction. Only after an approval has been committed does it try to generate the face embeddings (Stage 6).
"""
import logging
import uuid
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.case import CaseStatus, MissingCase
from app.models.case_status_event import CaseStatusEvent
from app.models.photo import CasePhoto
from app.models.user import User
from app.services import audit
from app.recognition import RecognitionService
from app.services import cases as case_service
from app.services import embeddings as embedding_service
from app.services import photos as photo_service
from app.storage import Storage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------- the rules (pure)
# pending -> approved | rejected; approved -> found | closed; found -> closed; rejected -> closed; closed is terminal.
ALLOWED_TRANSITIONS = MappingProxyType(
    {
        CaseStatus.PENDING: frozenset({CaseStatus.APPROVED, CaseStatus.REJECTED}),
        CaseStatus.APPROVED: frozenset({CaseStatus.FOUND, CaseStatus.CLOSED}),
        CaseStatus.FOUND: frozenset({CaseStatus.CLOSED}),
        CaseStatus.REJECTED: frozenset({CaseStatus.CLOSED}),
        CaseStatus.CLOSED: frozenset(),
    }
)

# A reason is mandatory when a case is rejected or closed, optional otherwise. A reason that is given has to be
# REASON_MIN_LENGTH..REASON_MAX_LENGTH characters long after trimming, whatever the status.
REASON_REQUIRED_FOR = frozenset({CaseStatus.REJECTED, CaseStatus.CLOSED})
REASON_MIN_LENGTH = 5
REASON_MAX_LENGTH = 1000

# Approving or rejecting is the review decision: it records who decided, when, and why (reviewed_by_id,
# reviewed_at, review_reason). The other transitions only add a history row.
REVIEW_DECISIONS = frozenset({CaseStatus.APPROVED, CaseStatus.REJECTED})


class InvalidTransitionError(Exception):
    def __init__(self, from_status: CaseStatus, to_status: CaseStatus):
        super().__init__("Invalid status transition")
        self.from_status = from_status
        self.to_status = to_status


class ReasonRequiredError(Exception):
    """A reason is mandatory for this target status and none (or only blanks) was given."""


class InvalidReasonError(Exception):
    """The reason is shorter than REASON_MIN_LENGTH or longer than REASON_MAX_LENGTH after trimming."""


class CaseHasNoPhotosError(Exception):
    """A case without a photo cannot be approved."""


class ReviewSort(StrEnum):
    OLDEST = "oldest"      # first in, first out: the default review queue order
    NEWEST = "newest"


def can_transition(from_status: str, to_status: str) -> bool:
    return CaseStatus(to_status) in ALLOWED_TRANSITIONS[CaseStatus(from_status)]


def check_transition(from_status: str, to_status: str) -> None:
    """Raises InvalidTransitionError for every pair that is not in ALLOWED_TRANSITIONS (staying in the
    same status included)."""
    if not can_transition(from_status, to_status):
        raise InvalidTransitionError(CaseStatus(from_status), CaseStatus(to_status))


def normalise_reason(to_status: str, reason: str | None) -> str | None:
    """The trimmed reason, or None when none was given (None or only blanks).
    Raises ReasonRequiredError / InvalidReasonError."""
    text = reason.strip() if reason is not None else ""
    if not text:
        if CaseStatus(to_status) in REASON_REQUIRED_FOR:
            raise ReasonRequiredError
        return None
    if not REASON_MIN_LENGTH <= len(text) <= REASON_MAX_LENGTH:
        raise InvalidReasonError
    return text


# ---------------------------------------------------------------- after an approval (Stage 6)
def after_approval(
    db: Session, case: MissingCase, *, recognition: RecognitionService, storage: Storage
) -> embedding_service.EmbeddingSummary:
    """Called once per approval, AFTER the approval was committed, as a separate step: it tries to generate the
    case's face embeddings. An approval is never rolled back because of the recognition service: if it is
    unavailable (or anything else goes wrong) the case stays approved with its photos "pending" and a warning
    with the case id is logged. Returns the photo counts by embedding status."""
    return embedding_service.after_approval_enrolment(db, case.id, recognition=recognition, storage=storage)


# ---------------------------------------------------------------- changing a status
@dataclass(frozen=True)
class StatusChange:
    case_id: uuid.UUID
    from_status: CaseStatus
    to_status: CaseStatus
    embeddings: embedding_service.EmbeddingSummary | None = None    # only set by an approval (see after_approval)


def _lock_case(db: Session, case_id: uuid.UUID) -> MissingCase:
    """The case row stays locked until the end of the transaction, so two admins (or an admin and the owner
    deleting a photo) cannot change the same case at the same moment. The lock itself lives in the cases service,
    where the embeddings service uses it too."""
    return case_service.get_case_for_update(db, case_id)


def change_status(
    db: Session,
    admin: User,
    case_id: uuid.UUID,
    to_status: CaseStatus,
    reason: str | None,
    *,
    recognition: RecognitionService | None = None,
    storage: Storage | None = None,
) -> StatusChange:
    """The one place a case changes status. Raises CaseNotFoundError, ReasonRequiredError, InvalidReasonError,
    InvalidTransitionError or CaseHasNoPhotosError; after any of them nothing has been changed.
    When an approval is committed and `recognition` and `storage` are given, after_approval() runs as a separate
    step afterwards; its failure never undoes the approval."""
    target = CaseStatus(to_status)
    clean_reason = normalise_reason(target, reason)              # input rules first: no lock for a bad request
    try:
        case = _lock_case(db, case_id)
        current = CaseStatus(case.status)
        check_transition(current, target)
        if target is CaseStatus.APPROVED and _photo_count(db, case.id) == 0:
            raise CaseHasNoPhotosError

        # clock_timestamp() is read AFTER the lock, so the history stays in order even when one admin had to
        # wait for another (now() would be the time the waiting transaction started).
        changed_at = db.scalar(select(func.clock_timestamp()))
        case.status = target.value
        if target in REVIEW_DECISIONS:
            case.review_reason = clean_reason
            case.reviewed_by_id = admin.id
            case.reviewed_at = changed_at
        db.add(
            CaseStatusEvent(
                case_id=case.id,
                from_status=current.value,
                to_status=target.value,
                changed_by_id=admin.id,
                reason=clean_reason,
                created_at=changed_at,
            )
        )
        # The detailed reason and transition are already in Stage 5's status-event row.
        # The audit event adds an immutable action/actor/target marker without copying that text.
        action = (
            "case_review_approved" if target is CaseStatus.APPROVED else
            "case_review_rejected" if target is CaseStatus.REJECTED else
            "case_status_changed"
        )
        audit.record_event(
            db, actor_id=admin.id, action=action, target_type="case", target_id=case.id
        )
        db.commit()
    except Exception:
        db.rollback()                                            # releases the row lock; nothing is saved
        raise
    embeddings = None
    if target is CaseStatus.APPROVED and recognition is not None and storage is not None:
        try:
            embeddings = after_approval(db, case, recognition=recognition, storage=storage)
        except Exception as exc:                                  # the approval is saved: whatever happens here stays here
            logger.warning(
                "embeddings_pending_after_approval", extra={"case_id": str(case_id), "reason": type(exc).__name__}
            )
    return StatusChange(case_id=case_id, from_status=current, to_status=target, embeddings=embeddings)


def _photo_count(db: Session, case_id: uuid.UUID) -> int:
    return db.scalar(select(func.count()).select_from(CasePhoto).where(CasePhoto.case_id == case_id)) or 0


# ---------------------------------------------------------------- reading
def list_cases_for_review(
    db: Session, *, status: CaseStatus, sort: ReviewSort, limit: int, offset: int
) -> tuple[list[tuple[MissingCase, int]], int]:
    """Cases with the given status, as (case, photo_count) pairs. Oldest first by default (a first-in-first-out
    queue). Returns (pairs on this page, total matching)."""
    total = db.scalar(select(func.count()).select_from(MissingCase).where(MissingCase.status == status.value)) or 0
    photo_count = (
        select(func.count(CasePhoto.id))
        .where(CasePhoto.case_id == MissingCase.id)
        .correlate(MissingCase)
        .scalar_subquery()
    )
    if sort is ReviewSort.NEWEST:
        order = (MissingCase.created_at.desc(), MissingCase.id.desc())
    else:
        order = (MissingCase.created_at.asc(), MissingCase.id.asc())
    rows = db.execute(
        select(MissingCase, photo_count.label("photo_count"))
        .where(MissingCase.status == status.value)
        .order_by(*order)
        .limit(limit)
        .offset(offset)
    ).all()
    return [(case, count) for case, count in rows], total


@dataclass(frozen=True)
class CaseDetail:
    case: MissingCase
    reporter: User
    photos: list[CasePhoto]
    history: list[CaseStatusEvent]


def get_case_detail(db: Session, case_id: uuid.UUID) -> CaseDetail:
    """Everything an admin needs to review one case. Raises CaseNotFoundError for an unknown id."""
    case = case_service.get_case(db, case_id)
    reporter = db.execute(select(User).where(User.id == case.reporter_id)).scalar_one()   # guaranteed by the FK
    photos = photo_service.list_photos_for_admin(db, case.id)
    history = list(
        db.scalars(
            select(CaseStatusEvent)
            .where(CaseStatusEvent.case_id == case.id)
            .order_by(CaseStatusEvent.created_at, CaseStatusEvent.id)
        )
    )
    return CaseDetail(case=case, reporter=reporter, photos=photos, history=history)
