import uuid
from datetime import datetime, timezone

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, String, Text, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class CaseStatusEvent(Base):
    """One row per status change of a case: the case's history. Rows are only ever added by
    app/services/review.py, in the same transaction as the status change itself.
    With the naming convention in app/db/base.py the checks become ck_case_status_events_from_status_valid
    and ck_case_status_events_to_status_valid. Migration 0005 spells out the same names."""

    __tablename__ = "case_status_events"
    __table_args__ = (
        CheckConstraint("from_status IN ('pending', 'approved', 'rejected', 'found', 'closed')", name="from_status_valid"),
        CheckConstraint("to_status IN ('pending', 'approved', 'rejected', 'found', 'closed')", name="to_status_valid"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    case_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("missing_cases.id", ondelete="CASCADE"))
    from_status: Mapped[str] = mapped_column(String(20))
    to_status: Mapped[str] = mapped_column(String(20))
    changed_by_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("users.id", ondelete="RESTRICT"))
    reason: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )


# "The history of a case, oldest first" is the main query; this index also serves lookups by case_id alone.
Index("ix_case_status_events_case_id_created_at", CaseStatusEvent.case_id, CaseStatusEvent.created_at)
