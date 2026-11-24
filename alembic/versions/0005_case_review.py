"""case review: review fields on missing_cases and the case_status_events table

Revision ID: 0005
Revises: 0004
Create Date: 2026-11-01
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_STATUSES = "('pending', 'approved', 'rejected', 'found', 'closed')"


def upgrade() -> None:
    # Constraint and index names are written out in full (op.f) and match the naming
    # convention in app/db/base.py, so the database and the models always agree.
    op.add_column("missing_cases", sa.Column("review_reason", sa.Text(), nullable=True))
    op.add_column("missing_cases", sa.Column("reviewed_by_id", sa.Uuid(), nullable=True))
    op.add_column("missing_cases", sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True))
    op.create_foreign_key(
        op.f("fk_missing_cases_reviewed_by_id_users"),
        "missing_cases",
        "users",
        ["reviewed_by_id"],
        ["id"],
        ondelete="RESTRICT",
    )
    # No API could reject a case before this stage, so normally there is nothing to fix. A row that someone set
    # to 'rejected' by hand would make the new CHECK impossible to add, so it gets a neutral placeholder instead.
    op.execute(
        "UPDATE missing_cases SET review_reason = 'Rejected before review reasons were recorded.' "
        "WHERE status = 'rejected' AND review_reason IS NULL"
    )
    op.create_check_constraint(
        op.f("ck_missing_cases_rejected_requires_reason"),
        "missing_cases",
        "status <> 'rejected' OR review_reason IS NOT NULL",
    )

    op.create_table(
        "case_status_events",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("case_id", sa.Uuid(), nullable=False),
        sa.Column("from_status", sa.String(length=20), nullable=False),
        sa.Column("to_status", sa.String(length=20), nullable=False),
        sa.Column("changed_by_id", sa.Uuid(), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint(f"from_status IN {_STATUSES}", name=op.f("ck_case_status_events_from_status_valid")),
        sa.CheckConstraint(f"to_status IN {_STATUSES}", name=op.f("ck_case_status_events_to_status_valid")),
        sa.ForeignKeyConstraint(
            ["case_id"], ["missing_cases.id"], name=op.f("fk_case_status_events_case_id_missing_cases"), ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["changed_by_id"], ["users.id"], name=op.f("fk_case_status_events_changed_by_id_users"), ondelete="RESTRICT"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_case_status_events")),
    )
    op.create_index(
        op.f("ix_case_status_events_case_id_created_at"), "case_status_events", ["case_id", "created_at"]
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_case_status_events_case_id_created_at"), table_name="case_status_events")
    op.drop_table("case_status_events")
    op.drop_constraint(op.f("ck_missing_cases_rejected_requires_reason"), "missing_cases", type_="check")
    op.drop_constraint(op.f("fk_missing_cases_reviewed_by_id_users"), "missing_cases", type_="foreignkey")
    op.drop_column("missing_cases", "reviewed_at")
    op.drop_column("missing_cases", "reviewed_by_id")
    op.drop_column("missing_cases", "review_reason")
