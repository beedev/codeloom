"""add enterprise migration columns (run_id, lane_versions)

Revision ID: b2c3d4e5f6a7
Revises: a7b8c9d0e1f2
Create Date: 2026-02-23 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID


# revision identifiers, used by Alembic.
revision: str = 'b2c3d4e5f6a7'
down_revision: Union[str, None] = 'a7b8c9d0e1f2'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('migration_phases', sa.Column('run_id', UUID(), nullable=True))
    op.add_column('migration_plans', sa.Column('lane_versions', JSONB, nullable=True))


def downgrade() -> None:
    op.drop_column('migration_plans', 'lane_versions')
    op.drop_column('migration_phases', 'run_id')
