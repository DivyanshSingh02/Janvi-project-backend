"""baseline

Revision ID: 0001
Revises:
Create Date: 2026-10-31
"""
from typing import Sequence, Union

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Empty on purpose: creates the alembic_version table so you can verify the
    # migration setup. Real tables arrive in Stage 2.
    pass


def downgrade() -> None:
    pass
