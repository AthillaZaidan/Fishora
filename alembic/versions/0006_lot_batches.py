"""one catch publishes as a batch of lots

Revision ID: 0006
Revises: 0005
Create Date: 2026-09-26
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0006"
down_revision: str | None = "0005"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # The operator now sells a catch as N lots of K kg, each its own auction.
    # Lots already published are a batch of one.
    op.add_column("lots", sa.Column("batch_index", sa.Integer(), nullable=False, server_default="1"))
    op.add_column("lots", sa.Column("batch_size", sa.Integer(), nullable=False, server_default="1"))
    op.create_check_constraint("ck_lots_batch_index", "lots", "batch_index BETWEEN 1 AND batch_size")
    # Still one publish per catch: a second publish collides on lot 1.
    op.drop_constraint("uq_lots_prediction_id", "lots", type_="unique")
    op.create_unique_constraint("uq_lots_prediction_batch", "lots", ["prediction_id", "batch_index"])


def downgrade() -> None:
    # Fails while any catch has more than one lot, which is the point: the old
    # schema cannot represent a batch.
    op.drop_constraint("uq_lots_prediction_batch", "lots", type_="unique")
    op.create_unique_constraint("uq_lots_prediction_id", "lots", ["prediction_id"])
    op.drop_constraint("ck_lots_batch_index", "lots", type_="check")
    op.drop_column("lots", "batch_size")
    op.drop_column("lots", "batch_index")
