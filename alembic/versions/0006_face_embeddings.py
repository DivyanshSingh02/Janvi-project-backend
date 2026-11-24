"""face embeddings: embedding status on case_photos, face_embeddings and face_search_logs

Revision ID: 0006
Revises: 0005
Create Date: 2026-11-01
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0006"
down_revision: Union[str, None] = "0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Constraint and index names are written out in full (op.f) and match the naming
    # convention in app/db/base.py, so the database and the models always agree.
    op.add_column(
        "case_photos",
        sa.Column("embedding_status", sa.String(length=20), server_default="pending", nullable=False),
    )
    op.add_column("case_photos", sa.Column("embedding_error", sa.String(length=50), nullable=True))
    op.create_check_constraint(
        op.f("ck_case_photos_embedding_status_valid"),
        "case_photos",
        "embedding_status IN ('pending', 'done', 'failed')",
    )

    op.create_table(
        "face_embeddings",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("photo_id", sa.Uuid(), nullable=False),
        sa.Column("case_id", sa.Uuid(), nullable=False),
        sa.Column("model_name", sa.String(length=100), nullable=False),
        sa.Column("model_version", sa.String(length=50), nullable=False),
        sa.Column("dimension", sa.Integer(), nullable=False),
        sa.Column("vector", sa.LargeBinary(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint("dimension > 0", name=op.f("ck_face_embeddings_dimension_positive")),
        sa.CheckConstraint(
            "octet_length(vector) = dimension * 4", name=op.f("ck_face_embeddings_vector_length_matches_dimension")
        ),
        sa.ForeignKeyConstraint(
            ["photo_id"], ["case_photos.id"], name=op.f("fk_face_embeddings_photo_id_case_photos"), ondelete="CASCADE"
        ),
        sa.ForeignKeyConstraint(
            ["case_id"], ["missing_cases.id"], name=op.f("fk_face_embeddings_case_id_missing_cases"), ondelete="CASCADE"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_face_embeddings")),
        sa.UniqueConstraint(
            "photo_id", "model_name", "model_version", name=op.f("uq_face_embeddings_photo_model_version")
        ),
    )
    op.create_index(op.f("ix_face_embeddings_case_id"), "face_embeddings", ["case_id"])

    op.create_table(
        "face_search_logs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("admin_id", sa.Uuid(), nullable=False),
        sa.Column("searched_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("result_count", sa.Integer(), nullable=False),
        sa.Column("top_case_id", sa.Uuid(), nullable=True),
        sa.Column("top_similarity", sa.Float(), nullable=True),
        sa.Column("model_name", sa.String(length=100), nullable=False),
        sa.Column("model_version", sa.String(length=50), nullable=False),
        sa.CheckConstraint("result_count >= 0", name=op.f("ck_face_search_logs_result_count_non_negative")),
        sa.ForeignKeyConstraint(
            ["admin_id"], ["users.id"], name=op.f("fk_face_search_logs_admin_id_users"), ondelete="RESTRICT"
        ),
        sa.ForeignKeyConstraint(
            ["top_case_id"],
            ["missing_cases.id"],
            name=op.f("fk_face_search_logs_top_case_id_missing_cases"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_face_search_logs")),
    )


def downgrade() -> None:
    # Drops the embeddings and the search log (derived data and an audit trail of searches). The photos, the
    # cases and the stored image files are not touched.
    op.drop_table("face_search_logs")
    op.drop_index(op.f("ix_face_embeddings_case_id"), table_name="face_embeddings")
    op.drop_table("face_embeddings")
    op.drop_constraint(op.f("ck_case_photos_embedding_status_valid"), "case_photos", type_="check")
    op.drop_column("case_photos", "embedding_error")
    op.drop_column("case_photos", "embedding_status")
