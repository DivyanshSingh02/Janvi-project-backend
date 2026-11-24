import uuid
from datetime import datetime, timezone

from sqlalchemy import CheckConstraint, DateTime, Float, ForeignKey, Integer, String, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class FaceSearchLog(Base):
    """One row per face search: who searched, when, and counts. It NEVER stores the query image, its embedding
    or any personal data. top_case_id / top_similarity describe the best candidate that was returned.
    With the naming convention in app/db/base.py the check becomes ck_face_search_logs_result_count_non_negative.
    Migration 0006 spells out the same name."""

    __tablename__ = "face_search_logs"
    __table_args__ = (CheckConstraint("result_count >= 0", name="result_count_non_negative"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    admin_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("users.id", ondelete="RESTRICT"))
    searched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )
    result_count: Mapped[int] = mapped_column(Integer)
    top_case_id: Mapped[uuid.UUID | None] = mapped_column(Uuid, ForeignKey("missing_cases.id", ondelete="SET NULL"))
    top_similarity: Mapped[float | None] = mapped_column(Float)
    model_name: Mapped[str] = mapped_column(String(100))
    model_version: Mapped[str] = mapped_column(String(50))
