"""add migration_lane_id to migration_plans

Revision ID: f1a2b3c4d5e6
Revises: da6b91173954
Create Date: 2026-02-22 18:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'f1a2b3c4d5e6'
down_revision: Union[str, None] = 'da6b91173954'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('migration_plans', sa.Column('migration_lane_id', sa.String(length=100), nullable=True))


def downgrade() -> None:
    op.drop_column('migration_plans', 'migration_lane_id')
