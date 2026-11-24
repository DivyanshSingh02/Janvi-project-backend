import uuid
from datetime import datetime, timezone

from sqlalchemy import CheckConstraint, DateTime, ForeignKey, Index, Integer, LargeBinary, String, UniqueConstraint, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class FaceEmbedding(Base):
    """The face embedding of one case photo, made by one model version. Only photos of APPROVED cases get one.
    The vector is float32 little-endian and L2-normalised (app/recognition/vectors.py); the API never returns it.
    With the naming convention in app/db/base.py the checks become ck_face_embeddings_dimension_positive and
    ck_face_embeddings_vector_length_matches_dimension. Migration 0006 spells out the same names.
    case_id repeats case_photos.case_id so a search can filter by case without a join through the photos."""

    __tablename__ = "face_embeddings"
    __table_args__ = (
        CheckConstraint("dimension > 0", name="dimension_positive"),
        CheckConstraint("octet_length(vector) = dimension * 4", name="vector_length_matches_dimension"),
        UniqueConstraint("photo_id", "model_name", "model_version", name="uq_face_embeddings_photo_model_version"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    photo_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("case_photos.id", ondelete="CASCADE"))
    case_id: Mapped[uuid.UUID] = mapped_column(Uuid, ForeignKey("missing_cases.id", ondelete="CASCADE"))
    model_name: Mapped[str] = mapped_column(String(100))
    model_version: Mapped[str] = mapped_column(String(50))
    dimension: Mapped[int] = mapped_column(Integer)
    vector: Mapped[bytes] = mapped_column(LargeBinary)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now()
    )


# A search loads the embeddings of approved cases; deleting or re-enrolling a case looks them up by case_id.
Index("ix_face_embeddings_case_id", FaceEmbedding.case_id)
