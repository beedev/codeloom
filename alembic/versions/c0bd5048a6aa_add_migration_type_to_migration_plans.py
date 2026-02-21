"""add migration_type to migration_plans

Revision ID: c0bd5048a6aa
Revises: a1b2c3d4e5f6
Create Date: 2026-02-18 22:04:31.909608

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = 'c0bd5048a6aa'
down_revision: Union[str, None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'migration_plans',
        sa.Column(
            'migration_type',
            sa.String(length=30),
            nullable=False,
            server_default='framework_migration',
        ),
    )


def downgrade() -> None:
    op.drop_column('migration_plans', 'migration_type')
