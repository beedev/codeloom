"""add deep understanding tables and project status column

Revision ID: 325b05218d1c
Revises: c0bd5048a6aa
Create Date: 2026-02-19 12:33:46.386122

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '325b05218d1c'
down_revision: Union[str, None] = 'c0bd5048a6aa'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Add deep_analysis_status column to projects
    op.add_column('projects', sa.Column(
        'deep_analysis_status', sa.String(length=20),
        server_default='none', nullable=False,
    ))

    # 2. Create deep_analysis_jobs table
    op.create_table('deep_analysis_jobs',
        sa.Column('job_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('project_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('status', sa.String(length=20), server_default='pending', nullable=False),
        sa.Column('worker_id', sa.String(length=100), nullable=True),
        sa.Column('retry_count', sa.Integer(), server_default='0', nullable=False),
        sa.Column('next_attempt_at', sa.TIMESTAMP(), nullable=True),
        sa.Column('total_entry_points', sa.Integer(), nullable=True),
        sa.Column('completed_entry_points', sa.Integer(), server_default='0', nullable=False),
        sa.Column('created_at', sa.TIMESTAMP(), server_default=sa.func.now(), nullable=False),
        sa.Column('started_at', sa.TIMESTAMP(), nullable=True),
        sa.Column('heartbeat_at', sa.TIMESTAMP(), nullable=True),
        sa.Column('completed_at', sa.TIMESTAMP(), nullable=True),
        sa.Column('error_details', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.ForeignKeyConstraint(['project_id'], ['projects.project_id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['user_id'], ['users.user_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('job_id'),
    )
    op.create_index('idx_deep_jobs_project_status', 'deep_analysis_jobs', ['project_id', 'status'])
    op.create_index('idx_deep_jobs_status_created', 'deep_analysis_jobs', ['status', 'created_at'])

    # 3. Create deep_analyses table
    op.create_table('deep_analyses',
        sa.Column('analysis_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('job_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('project_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('entry_unit_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('entry_type', sa.String(length=50), nullable=False),
        sa.Column('tier', sa.String(length=20), nullable=False),
        sa.Column('total_units', sa.Integer(), nullable=False),
        sa.Column('total_tokens', sa.Integer(), nullable=False),
        sa.Column('confidence_score', sa.Float(), nullable=True),
        sa.Column('coverage_pct', sa.Float(), nullable=True),
        sa.Column('result_json', postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column('narrative', sa.Text(), nullable=True),
        sa.Column('schema_version', sa.Integer(), server_default='1', nullable=False),
        sa.Column('prompt_version', sa.String(length=20), server_default='v1.0', nullable=False),
        sa.Column('analyzed_at', sa.TIMESTAMP(), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(['entry_unit_id'], ['code_units.unit_id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['job_id'], ['deep_analysis_jobs.job_id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['project_id'], ['projects.project_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('analysis_id'),
        sa.UniqueConstraint('project_id', 'entry_unit_id', 'schema_version', name='uq_deep_analysis_entry'),
    )
    op.create_index('idx_deep_analyses_entry', 'deep_analyses', ['entry_unit_id'])
    op.create_index('idx_deep_analyses_project', 'deep_analyses', ['project_id'])

    # 4. Create analysis_units junction table
    op.create_table('analysis_units',
        sa.Column('analysis_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('unit_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('project_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('min_depth', sa.Integer(), server_default='0', nullable=False),
        sa.Column('path_count', sa.Integer(), server_default='1', nullable=False),
        sa.ForeignKeyConstraint(['analysis_id'], ['deep_analyses.analysis_id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['project_id'], ['projects.project_id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['unit_id'], ['code_units.unit_id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('analysis_id', 'unit_id'),
    )
    op.create_index('idx_analysis_units_project_unit', 'analysis_units', ['project_id', 'unit_id'])


def downgrade() -> None:
    op.drop_index('idx_analysis_units_project_unit', table_name='analysis_units')
    op.drop_table('analysis_units')
    op.drop_index('idx_deep_analyses_project', table_name='deep_analyses')
    op.drop_index('idx_deep_analyses_entry', table_name='deep_analyses')
    op.drop_table('deep_analyses')
    op.drop_index('idx_deep_jobs_status_created', table_name='deep_analysis_jobs')
    op.drop_index('idx_deep_jobs_project_status', table_name='deep_analysis_jobs')
    op.drop_table('deep_analysis_jobs')
    op.drop_column('projects', 'deep_analysis_status')
