"""add_parent_project_id_and_knowledge_notebook_ids

Revision ID: d22758990d1d
Revises: b55b1f82f8be
Create Date: 2026-03-27 12:51:50.432372

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'd22758990d1d'
down_revision: Union[str, None] = 'b55b1f82f8be'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add parent_project_id to projects (links migration notebook → code project)
    op.add_column('projects', sa.Column('parent_project_id', sa.UUID(), nullable=True))
    op.create_foreign_key(
        'fk_projects_parent_project',
        'projects', 'projects',
        ['parent_project_id'], ['project_id'],
        ondelete='CASCADE'
    )

    # Add knowledge_notebook_ids to migration_plans (reference notebooks for transforms)
    op.add_column('migration_plans', sa.Column(
        'knowledge_notebook_ids',
        postgresql.JSONB(astext_type=sa.Text()),
        nullable=True
    ))


def downgrade() -> None:
    op.drop_constraint('fk_projects_parent_project', 'projects', type_='foreignkey')
    op.drop_column('projects', 'parent_project_id')
    op.drop_column('migration_plans', 'knowledge_notebook_ids')
