"""stage trace on knowledge jobs

Revision ID: 0007
Revises: 0006
Create Date: 2026-09-26
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # W12: what each card run saw and decided, kept with the job. Nullable:
    # jobs that ran before this carry no trace.
    op.add_column("knowledge_jobs", sa.Column("trace", postgresql.JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("knowledge_jobs", "trace")
