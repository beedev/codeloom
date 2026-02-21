"""add framework_docs to migration_plans

Revision ID: 0f8735457718
Revises: e077862f969f
Create Date: 2026-02-18 11:59:00.533858

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '0f8735457718'
down_revision: Union[str, None] = 'e077862f969f'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'migration_plans',
        sa.Column('framework_docs', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('migration_plans', 'framework_docs')
