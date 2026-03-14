"""enable AGE extension and add age_graph_status to projects

Revision ID: 3f8e9a0b1c2d
Revises: 2e7dbe92a6af
Create Date: 2026-03-13 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3f8e9a0b1c2d'
down_revision: Union[str, None] = '2e7dbe92a6af'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Enable Apache AGE extension (requires superuser — may already be installed)
    op.execute("CREATE EXTENSION IF NOT EXISTS age")

    # Add graph sync status to projects
    op.add_column(
        'projects',
        sa.Column('age_graph_status', sa.String(20), server_default='pending'),
    )


def downgrade() -> None:
    op.drop_column('projects', 'age_graph_status')
