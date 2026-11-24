"""reporter-submitted found-person photos for administrator review

Revision ID: 0009
Revises: 0008
Create Date: 2026-11-24
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0009"
down_revision: Union[str, None] = "0008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "found_person_reports",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("reporter_id", sa.Uuid(), nullable=False),
        sa.Column("storage_key", sa.String(length=255), nullable=False),
        sa.Column("content_type", sa.String(length=50), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("review_status", sa.String(length=20), server_default="pending", nullable=False),
        sa.Column("reviewed_by_id", sa.Uuid(), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("review_note", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("content_type = 'image/jpeg'", name=op.f("ck_found_person_reports_content_type_valid")),
        sa.CheckConstraint("size_bytes > 0", name=op.f("ck_found_person_reports_size_positive")),
        sa.CheckConstraint("width > 0", name=op.f("ck_found_person_reports_width_positive")),
        sa.CheckConstraint("height > 0", name=op.f("ck_found_person_reports_height_positive")),
        sa.CheckConstraint("sha256 ~ '^[0-9a-f]{64}$'", name=op.f("ck_found_person_reports_sha256_format")),
        sa.CheckConstraint(
            "review_status IN ('pending', 'reviewed', 'contacted', 'dismissed')",
            name=op.f("ck_found_person_reports_review_status_valid"),
        ),
        sa.CheckConstraint(
            "(review_status = 'pending' AND reviewed_by_id IS NULL AND reviewed_at IS NULL) OR "
            "(review_status <> 'pending' AND reviewed_by_id IS NOT NULL AND reviewed_at IS NOT NULL)",
            name=op.f("ck_found_person_reports_review_fields_match_status"),
        ),
        sa.ForeignKeyConstraint(
            ["reporter_id"], ["users.id"], name=op.f("fk_found_person_reports_reporter_id_users"), ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["reviewed_by_id"], ["users.id"], name=op.f("fk_found_person_reports_reviewed_by_id_users"), ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_found_person_reports")),
        sa.UniqueConstraint("storage_key", name=op.f("uq_found_person_reports_storage_key")),
    )
    op.create_index(
        op.f("ix_found_person_reports_reporter_id_created_at"),
        "found_person_reports",
        ["reporter_id", sa.text("created_at DESC")],
    )
    op.create_index(
        op.f("ix_found_person_reports_review_status"), "found_person_reports", ["review_status"]
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_found_person_reports_review_status"), table_name="found_person_reports")
    op.drop_index(op.f("ix_found_person_reports_reporter_id_created_at"), table_name="found_person_reports")
    op.drop_table("found_person_reports")
