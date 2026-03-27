"""add reverse engineering docs table

Revision ID: 84c854201518
Revises: 3f8e9a0b1c2d
Create Date: 2026-03-21 20:27:34.466530

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '84c854201518'
down_revision: Union[str, None] = '3f8e9a0b1c2d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('reverse_engineering_docs',
    sa.Column('doc_id', sa.UUID(), nullable=False),
    sa.Column('project_id', sa.UUID(), nullable=False),
    sa.Column('status', sa.String(length=20), nullable=False),
    sa.Column('chapters', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('chapter_titles', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    sa.Column('progress', sa.Integer(), nullable=True),
    sa.Column('total_chapters', sa.Integer(), nullable=True),
    sa.Column('error', sa.Text(), nullable=True),
    sa.Column('created_at', sa.TIMESTAMP(), nullable=False),
    sa.Column('updated_at', sa.TIMESTAMP(), nullable=True),
    sa.ForeignKeyConstraint(['project_id'], ['projects.project_id'], ondelete='CASCADE'),
    sa.PrimaryKeyConstraint('doc_id')
    )
    op.create_index('idx_reveng_docs_project', 'reverse_engineering_docs', ['project_id'], unique=False)


def downgrade() -> None:
    op.drop_index('idx_reveng_docs_project', table_name='reverse_engineering_docs')
    op.drop_table('reverse_engineering_docs')
