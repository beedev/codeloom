"""add asset_strategies to migration_plans

Revision ID: b7a3f9e1c2d4
Revises: 54447161ed20
Create Date: 2026-02-21 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'b7a3f9e1c2d4'
down_revision: Union[str, None] = '54447161ed20'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'migration_plans',
        sa.Column('asset_strategies', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('migration_plans', 'asset_strategies')
