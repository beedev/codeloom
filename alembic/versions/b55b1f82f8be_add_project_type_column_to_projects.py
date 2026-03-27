"""add project_type column to projects

Revision ID: b55b1f82f8be
Revises: 84c854201518
Create Date: 2026-03-27 11:05:17.739961

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b55b1f82f8be'
down_revision: Union[str, None] = '84c854201518'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        'projects',
        sa.Column('project_type', sa.String(length=20), nullable=False, server_default='code'),
    )


def downgrade() -> None:
    op.drop_column('projects', 'project_type')
