"""add project source tracking fields

Revision ID: 97ccdc52d3e9
Revises: 2d144d33226a
Create Date: 2026-02-16 15:15:45.609528

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '97ccdc52d3e9'
down_revision: Union[str, None] = '2d144d33226a'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('projects', sa.Column('source_type', sa.String(length=20), nullable=True))
    op.add_column('projects', sa.Column('source_url', sa.String(length=2048), nullable=True))
    op.add_column('projects', sa.Column('repo_branch', sa.String(length=255), nullable=True))
    op.add_column('projects', sa.Column('last_synced_at', sa.TIMESTAMP(), nullable=True))


def downgrade() -> None:
    op.drop_column('projects', 'last_synced_at')
    op.drop_column('projects', 'repo_branch')
    op.drop_column('projects', 'source_url')
    op.drop_column('projects', 'source_type')
