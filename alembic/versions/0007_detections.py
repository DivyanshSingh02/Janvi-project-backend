"""detections and detection history

Revision ID: 0007
Revises: 0006
Create Date: 2026-11-03
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0007"
down_revision: Union[str, None] = "0006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "detections",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("case_id", sa.Uuid(), nullable=False),
        sa.Column("detected_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source", sa.String(length=20), nullable=False),
        sa.Column("camera_id", sa.String(length=64), nullable=True),
        sa.Column("location_text", sa.String(length=200), nullable=True),
        sa.Column("latitude", sa.Float(), nullable=True),
        sa.Column("longitude", sa.Float(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("frame_reference", sa.String(length=255), nullable=True),
        sa.Column("review_status", sa.String(length=20), server_default="unreviewed", nullable=False),
        sa.Column("reviewed_by_id", sa.Uuid(), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("review_note", sa.Text(), nullable=True),
        sa.Column("recorded_by_id", sa.Uuid(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("source IN ('camera', 'video', 'manual')", name=op.f("ck_detections_source_valid")),
        sa.CheckConstraint("confidence >= 0 AND confidence <= 1", name=op.f("ck_detections_confidence_range")),
        sa.CheckConstraint(
            "(latitude IS NULL AND longitude IS NULL) OR (latitude IS NOT NULL AND longitude IS NOT NULL)",
            name=op.f("ck_detections_coordinates_pair"),
        ),
        sa.CheckConstraint("latitude IS NULL OR latitude BETWEEN -90 AND 90", name=op.f("ck_detections_latitude_range")),
        sa.CheckConstraint("longitude IS NULL OR longitude BETWEEN -180 AND 180", name=op.f("ck_detections_longitude_range")),
        sa.CheckConstraint(
            "camera_id IS NOT NULL OR location_text IS NOT NULL OR (latitude IS NOT NULL AND longitude IS NOT NULL)",
            name=op.f("ck_detections_location_present"),
        ),
        sa.CheckConstraint(
            "source <> 'camera' OR camera_id IS NOT NULL",
            name=op.f("ck_detections_camera_source_requires_camera_id"),
        ),
        sa.CheckConstraint(
            "review_status IN ('unreviewed', 'confirmed', 'dismissed')",
            name=op.f("ck_detections_review_status_valid"),
        ),
        sa.ForeignKeyConstraint(
            ["case_id"], ["missing_cases.id"], name=op.f("fk_detections_case_id_missing_cases"), ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["reviewed_by_id"], ["users.id"], name=op.f("fk_detections_reviewed_by_id_users"), ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["recorded_by_id"], ["users.id"], name=op.f("fk_detections_recorded_by_id_users"), ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_detections")),
    )
    op.create_index(
        op.f("ix_detections_case_id_detected_at"), "detections",
        ["case_id", sa.text("detected_at DESC")],
    )
    op.create_index(
        op.f("ix_detections_detected_at"), "detections", [sa.text("detected_at DESC")]
    )
    op.create_index(op.f("ix_detections_review_status"), "detections", ["review_status"])


def downgrade() -> None:
    op.drop_index(op.f("ix_detections_review_status"), table_name="detections")
    op.drop_index(op.f("ix_detections_detected_at"), table_name="detections")
    op.drop_index(op.f("ix_detections_case_id_detected_at"), table_name="detections")
    op.drop_table("detections")
