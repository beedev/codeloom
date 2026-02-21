"""add pipeline_version and mvp_analysis

Revision ID: a1b2c3d4e5f6
Revises: 0f8735457718
Create Date: 2026-02-18 18:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, None] = '0f8735457718'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # MigrationPlan: add pipeline_version (default 2 for new plans, existing plans get 1)
    op.add_column(
        'migration_plans',
        sa.Column('pipeline_version', sa.Integer(), nullable=False, server_default='1'),
    )

    # FunctionalMVP: add on-demand analysis columns
    op.add_column(
        'functional_mvps',
        sa.Column('analysis_output', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.add_column(
        'functional_mvps',
        sa.Column('analysis_at', sa.TIMESTAMP(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column('functional_mvps', 'analysis_at')
    op.drop_column('functional_mvps', 'analysis_output')
    op.drop_column('migration_plans', 'pipeline_version')
