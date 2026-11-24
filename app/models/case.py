import uuid
from datetime import date, datetime, timezone
from enum import StrEnum

from sqlalchemy import CheckConstraint, Date, DateTime, ForeignKey, Index, Integer, String, Text, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class CaseStatus(StrEnum):
    """All five statuses exist from the start. The rules for moving between them live in
    app/services/review.py (ALLOWED_TRANSITIONS), added with the admin review in Stage 5."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    FOUND = "found"
    CLOSED = "closed"


class Gender(StrEnum):
    MALE = "male"
    FEMALE = "female"
    OTHER = "other"


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class MissingCase(Base):
    __tablename__ = "missing_cases"
    # With the naming convention in app/db/base.py these become ck_missing_cases_status_valid,
    # ck_missing_cases_gender_valid, ck_missing_cases_age_range, ck_missing_cases_contact_present and
    # ck_missing_cases_rejected_requires_reason. Migrations 0003 and 0005 spell out the same names.
    __table_args__ = (
        CheckConstraint("status IN ('pending', 'approved', 'rejected', 'found', 'closed')", name="status_valid"),
        CheckConstraint("gender IN ('male', 'female', 'other')", name="gender_valid"),
        CheckConstraint("age BETWEEN 0 AND 120", name="age_range"),
        CheckConstraint("contact_phone IS NOT NULL OR contact_email IS NOT NULL", name="contact_present"),
        CheckConstraint("status <> 'rejected' OR review_reason IS NOT NULL", name="rejected_requires_reason"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    reporter_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("users.id", ondelete="RESTRICT"))

    name: Mapped[str] = mapped_column(String(120))
    age: Mapped[int] = mapped_column(Integer)
    gender: Mapped[str] = mapped_column(String(20))
    contact_phone: Mapped[str | None] = mapped_column(String(20))
    contact_email: Mapped[str | None] = mapped_column(String(254))
    city: Mapped[str] = mapped_column(String(80))
    state: Mapped[str] = mapped_column(String(80))
    address: Mapped[str] = mapped_column(String(300))
    last_seen_location: Mapped[str] = mapped_column(String(200))
    last_seen_date: Mapped[date] = mapped_column(Date)

    status: Mapped[str] = mapped_column(
        String(20), default=CaseStatus.PENDING.value, server_default=CaseStatus.PENDING.value
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, server_default=func.now()
    )

    # Set by the admin review (Stage 5). review_reason and reviewed_* are written by approve and reject only;
    # found and closed leave them alone (their reasons live in case_status_events).
    review_reason: Mapped[str | None] = mapped_column(Text)
    reviewed_by_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("users.id", ondelete="RESTRICT"))
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


# "My cases, newest first" is the main query. This index also serves any lookup by reporter_id
# alone (it is the leading column), so there is no separate reporter_id index.
Index("ix_missing_cases_reporter_id_created_at", MissingCase.reporter_id, MissingCase.created_at.desc())
Index("ix_missing_cases_status", MissingCase.status)
