import uuid
from datetime import datetime, timezone

from sqlalchemy import CheckConstraint, DateTime, Float, ForeignKey, Index, String, Text, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Detection(Base):
    """A human- or system-recorded observation associated with an approved case.

    A detection is a lead, never proof of identity. Review fields are populated only
    by an administrator reviewing an unreviewed detection.
    """

    __tablename__ = "detections"
    __table_args__ = (
        CheckConstraint("source IN ('camera', 'video', 'manual')", name="source_valid"),
        CheckConstraint("confidence >= 0 AND confidence <= 1", name="confidence_range"),
        CheckConstraint(
            "(latitude IS NULL AND longitude IS NULL) OR "
            "(latitude IS NOT NULL AND longitude IS NOT NULL)",
            name="coordinates_pair",
        ),
        CheckConstraint("latitude IS NULL OR latitude BETWEEN -90 AND 90", name="latitude_range"),
        CheckConstraint("longitude IS NULL OR longitude BETWEEN -180 AND 180", name="longitude_range"),
        CheckConstraint(
            "camera_id IS NOT NULL OR location_text IS NOT NULL OR "
            "(latitude IS NOT NULL AND longitude IS NOT NULL)",
            name="location_present",
        ),
        CheckConstraint("source <> 'camera' OR camera_id IS NOT NULL", name="camera_source_requires_camera_id"),
        CheckConstraint(
            "review_status IN ('unreviewed', 'confirmed', 'dismissed')",
            name="review_status_valid",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    case_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("missing_cases.id", ondelete="RESTRICT"), nullable=False
    )
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    source: Mapped[str] = mapped_column(String(20), nullable=False)
    camera_id: Mapped[str | None] = mapped_column(String(64))
    location_text: Mapped[str | None] = mapped_column(String(200))
    latitude: Mapped[float | None] = mapped_column(Float)
    longitude: Mapped[float | None] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    frame_reference: Mapped[str | None] = mapped_column(String(255))
    review_status: Mapped[str] = mapped_column(
        String(20), default="unreviewed", server_default="unreviewed", nullable=False
    )
    reviewed_by_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("users.id", ondelete="RESTRICT")
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    review_note: Mapped[str | None] = mapped_column(Text)
    recorded_by_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("users.id", ondelete="RESTRICT"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, server_default=func.now(), nullable=False
    )


Index(
    "ix_detections_case_id_detected_at",
    Detection.case_id,
    Detection.detected_at.desc(),
)
Index("ix_detections_detected_at", Detection.detected_at.desc())
Index("ix_detections_review_status", Detection.review_status)
