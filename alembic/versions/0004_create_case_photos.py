"""create case photos

Revision ID: 0004
Revises: 0003
Create Date: 2026-10-31
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Constraint and index names are written out in full (op.f) and match the naming
    # convention in app/db/base.py, so the database and the models always agree.
    op.create_table(
        "case_photos",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("case_id", sa.Uuid(), nullable=False),
        sa.Column("storage_key", sa.String(length=255), nullable=False),
        sa.Column("content_type", sa.String(length=50), nullable=False),
        sa.Column("size_bytes", sa.Integer(), nullable=False),
        sa.Column("width", sa.Integer(), nullable=False),
        sa.Column("height", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("content_type = 'image/jpeg'", name=op.f("ck_case_photos_content_type_valid")),
        sa.CheckConstraint("size_bytes > 0", name=op.f("ck_case_photos_size_positive")),
        sa.CheckConstraint("width > 0", name=op.f("ck_case_photos_width_positive")),
        sa.CheckConstraint("height > 0", name=op.f("ck_case_photos_height_positive")),
        sa.CheckConstraint("sha256 ~ '^[0-9a-f]{64}$'", name=op.f("ck_case_photos_sha256_format")),
        sa.ForeignKeyConstraint(
            ["case_id"], ["missing_cases.id"], name=op.f("fk_case_photos_case_id_missing_cases"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_case_photos")),
        sa.UniqueConstraint("storage_key", name=op.f("uq_case_photos_storage_key")),
        sa.UniqueConstraint("case_id", "sha256", name=op.f("uq_case_photos_case_id_sha256")),
    )
    op.create_index(op.f("ix_case_photos_case_id_created_at"), "case_photos", ["case_id", "created_at"])


def downgrade() -> None:
    # Drops the rows only. Files already written to storage are not touched by a migration.
    op.drop_index(op.f("ix_case_photos_case_id_created_at"), table_name="case_photos")
    op.drop_table("case_photos")
