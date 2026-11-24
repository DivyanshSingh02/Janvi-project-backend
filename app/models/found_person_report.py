import uuid
from datetime import datetime, timezone

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, Integer, String, Text, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class FoundPersonReport(Base):
    """A reporter-submitted found-person photo awaiting administrator review."""

    __tablename__ = "found_person_reports"
    __table_args__ = (
        CheckConstraint("content_type = 'image/jpeg'", name="content_type_valid"),
        CheckConstraint("size_bytes > 0", name="size_positive"),
        CheckConstraint("width > 0", name="width_positive"),
        CheckConstraint("height > 0", name="height_positive"),
        CheckConstraint("sha256 ~ '^[0-9a-f]{64}$'", name="sha256_format"),
        CheckConstraint(
            "review_status IN ('pending', 'reviewed', 'contacted', 'dismissed')",
            name="review_status_valid",
        ),
        CheckConstraint(
            "(review_status = 'pending' AND reviewed_by_id IS NULL AND reviewed_at IS NULL) OR "
            "(review_status <> 'pending' AND reviewed_by_id IS NOT NULL AND reviewed_at IS NOT NULL)",
            name="review_fields_match_status",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    reporter_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )
    storage_key: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    content_type: Mapped[str] = mapped_column(String(50), nullable=False, default="image/jpeg")
    size_bytes: Mapped[int] = mapped_column(Integer, nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False)
    height: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    review_status: Mapped[str] = mapped_column(
        String(20), default="pending", server_default="pending", nullable=False
    )
    reviewed_by_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("users.id", ondelete="RESTRICT")
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    review_note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now(), nullable=False
    )


Index("ix_found_person_reports_reporter_id_created_at", FoundPersonReport.reporter_id, FoundPersonReport.created_at.desc())
Index("ix_found_person_reports_review_status", FoundPersonReport.review_status)
