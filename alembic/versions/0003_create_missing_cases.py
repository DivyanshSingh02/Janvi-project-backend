"""create missing cases

Revision ID: 0003
Revises: 0002
Create Date: 2026-10-31
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Constraint and index names are written out in full (op.f) and match the naming
    # convention in app/db/base.py, so the database and the models always agree.
    op.create_table(
        "missing_cases",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("reporter_id", sa.Uuid(), nullable=False),
        sa.Column("name", sa.String(length=120), nullable=False),
        sa.Column("age", sa.Integer(), nullable=False),
        sa.Column("gender", sa.String(length=20), nullable=False),
        sa.Column("contact_phone", sa.String(length=20), nullable=True),
        sa.Column("contact_email", sa.String(length=254), nullable=True),
        sa.Column("city", sa.String(length=80), nullable=False),
        sa.Column("state", sa.String(length=80), nullable=False),
        sa.Column("address", sa.String(length=300), nullable=False),
        sa.Column("last_seen_location", sa.String(length=200), nullable=False),
        sa.Column("last_seen_date", sa.Date(), nullable=False),
        sa.Column("status", sa.String(length=20), server_default="pending", nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint(
            "status IN ('pending', 'approved', 'rejected', 'found', 'closed')",
            name=op.f("ck_missing_cases_status_valid"),
        ),
        sa.CheckConstraint("gender IN ('male', 'female', 'other')", name=op.f("ck_missing_cases_gender_valid")),
        sa.CheckConstraint("age BETWEEN 0 AND 120", name=op.f("ck_missing_cases_age_range")),
        sa.CheckConstraint(
            "contact_phone IS NOT NULL OR contact_email IS NOT NULL",
            name=op.f("ck_missing_cases_contact_present"),
        ),
        sa.ForeignKeyConstraint(
            ["reporter_id"], ["users.id"], name=op.f("fk_missing_cases_reporter_id_users"), ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_missing_cases")),
    )
    op.create_index(
        op.f("ix_missing_cases_reporter_id_created_at"),
        "missing_cases",
        ["reporter_id", sa.text("created_at DESC")],
    )
    op.create_index(op.f("ix_missing_cases_status"), "missing_cases", ["status"])


def downgrade() -> None:
    op.drop_index(op.f("ix_missing_cases_status"), table_name="missing_cases")
    op.drop_index(op.f("ix_missing_cases_reporter_id_created_at"), table_name="missing_cases")
    op.drop_table("missing_cases")
