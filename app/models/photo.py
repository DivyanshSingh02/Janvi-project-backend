import uuid
from datetime import datetime, timezone

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, Integer, String, UniqueConstraint, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base

JPEG_CONTENT_TYPE = "image/jpeg"

# Where a photo stands for face recognition (Stage 6). Only photos of approved cases are ever processed.
EMBEDDING_PENDING = "pending"    # not processed yet (or the case is not approved, or no model is configured)
EMBEDDING_DONE = "done"          # an embedding exists
EMBEDDING_FAILED = "failed"      # processed, but no usable single face: embedding_error says why
EMBEDDING_STATUSES = (EMBEDDING_PENDING, EMBEDDING_DONE, EMBEDDING_FAILED)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class CasePhoto(Base):
    """One stored photo of a case. The original file name is deliberately NOT stored anywhere, and
    the storage key / sha256 are internal: the API never returns them."""

    __tablename__ = "case_photos"
    # With the naming convention in app/db/base.py the checks become ck_case_photos_content_type_valid,
    # ck_case_photos_size_positive, ck_case_photos_width_positive, ck_case_photos_height_positive,
    # ck_case_photos_sha256_format and ck_case_photos_embedding_status_valid. The unique constraint carries an explicit full name (the convention would
    # only use its first column). Migrations 0004 and 0006 spell out the same names.
    __table_args__ = (
        CheckConstraint("content_type = 'image/jpeg'", name="content_type_valid"),
        CheckConstraint("size_bytes > 0", name="size_positive"),
        CheckConstraint("width > 0", name="width_positive"),
        CheckConstraint("height > 0", name="height_positive"),
        CheckConstraint("sha256 ~ '^[0-9a-f]{64}$'", name="sha256_format"),
        CheckConstraint("embedding_status IN ('pending', 'done', 'failed')", name="embedding_status_valid"),
        UniqueConstraint("case_id", "sha256", name="uq_case_photos_case_id_sha256"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    case_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("missing_cases.id", ondelete="CASCADE"))
    storage_key: Mapped[str] = mapped_column(String(255), unique=True)
    content_type: Mapped[str] = mapped_column(String(50), default=JPEG_CONTENT_TYPE)
    size_bytes: Mapped[int] = mapped_column(Integer)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64))
    embedding_status: Mapped[str] = mapped_column(
        String(20), default=EMBEDDING_PENDING, server_default=EMBEDDING_PENDING
    )
    embedding_error: Mapped[str | None] = mapped_column(String(50))     # a short code such as 'no_face'
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )


# "The photos of a case, oldest first" is the main query; this index also serves lookups by case_id alone.
Index("ix_case_photos_case_id_created_at", CasePhoto.case_id, CasePhoto.created_at)
