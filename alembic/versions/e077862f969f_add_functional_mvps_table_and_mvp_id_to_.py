"""add functional_mvps table and mvp_id to phases

Revision ID: e077862f969f
Revises: 97ccdc52d3e9
Create Date: 2026-02-17 15:18:30.199294

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = 'e077862f969f'
down_revision: Union[str, None] = '97ccdc52d3e9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create functional_mvps table
    op.create_table('functional_mvps',
        sa.Column('mvp_id', sa.Integer(), autoincrement=True, nullable=False),
        sa.Column('plan_id', sa.UUID(), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=True),
        sa.Column('priority', sa.Integer(), nullable=True),
        sa.Column('file_ids', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('unit_ids', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('depends_on_mvp_ids', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('sp_references', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('metrics', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('current_phase', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(), nullable=False),
        sa.Column('updated_at', sa.TIMESTAMP(), nullable=True),
        sa.ForeignKeyConstraint(['plan_id'], ['migration_plans.plan_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('mvp_id')
    )
    op.create_index('idx_functional_mvps_plan', 'functional_mvps', ['plan_id'], unique=False)

    # Add mvp_id to migration_phases + update unique constraint
    op.add_column('migration_phases', sa.Column('mvp_id', sa.Integer(), nullable=True))
    op.drop_constraint('uq_plan_phase', 'migration_phases', type_='unique')
    op.create_index('idx_migration_phases_mvp', 'migration_phases', ['mvp_id'], unique=False)
    op.create_unique_constraint('uq_plan_phase_mvp', 'migration_phases', ['plan_id', 'phase_number', 'mvp_id'])
    op.create_foreign_key('fk_migration_phases_mvp', 'migration_phases', 'functional_mvps', ['mvp_id'], ['mvp_id'], ondelete='CASCADE')

    # Add discovery_metadata to migration_plans
    op.add_column('migration_plans', sa.Column('discovery_metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=True))


def downgrade() -> None:
    op.drop_column('migration_plans', 'discovery_metadata')
    op.drop_constraint('fk_migration_phases_mvp', 'migration_phases', type_='foreignkey')
    op.drop_constraint('uq_plan_phase_mvp', 'migration_phases', type_='unique')
    op.drop_index('idx_migration_phases_mvp', table_name='migration_phases')
    op.create_unique_constraint('uq_plan_phase', 'migration_phases', ['plan_id', 'phase_number'])
    op.drop_column('migration_phases', 'mvp_id')
    op.drop_index('idx_functional_mvps_plan', table_name='functional_mvps')
    op.drop_table('functional_mvps')
