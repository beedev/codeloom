"""add target_manifest to migration_plans

Revision ID: 4801d52f3782
Revises: d22758990d1d
Create Date: 2026-03-27 22:27:10.869601

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '4801d52f3782'
down_revision: Union[str, None] = 'd22758990d1d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'migration_plans',
        sa.Column('target_manifest', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('migration_plans', 'target_manifest')
