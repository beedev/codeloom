"""add RAG feedback and adaptive settings tables

Revision ID: a9c9ea721187
Revises: 4801d52f3782
Create Date: 2026-03-28 14:48:49.885199

Tables created:
1. rag_feedback          - User thumbs/star ratings for chat responses (linked to Langfuse traces)
2. rag_feedback_nodes    - Per-node relevance annotations attached to feedback records
3. rag_feedback_insights - Pre-computed analysis results from FeedbackAnalyzer (with TTL)
4. rag_adaptive_settings - Adaptive retrieval parameters derived from feedback (with TTL)
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = 'a9c9ea721187'
down_revision: Union[str, None] = '4801d52f3782'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. rag_feedback
    op.create_table(
        'rag_feedback',
        sa.Column('feedback_id', postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text('gen_random_uuid()')),
        sa.Column('trace_id', sa.String(255), nullable=False,
                  comment='Langfuse trace ID for correlation'),
        sa.Column('query_id', postgresql.UUID(as_uuid=True), nullable=False,
                  comment='Internal query UUID'),
        sa.Column('user_id', postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column('project_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('rating', sa.Integer(), nullable=True, comment='Star rating 1-5'),
        sa.Column('helpful', sa.Boolean(), nullable=True, comment='Thumbs up/down'),
        sa.Column('user_message', sa.Text(), nullable=True),
        sa.Column('feedback_category', sa.String(50), nullable=True,
                  comment='inaccurate | irrelevant | incomplete | helpful | other'),
        sa.Column('created_at', sa.TIMESTAMP(), nullable=False,
                  server_default=sa.func.now()),
        sa.ForeignKeyConstraint(['user_id'], ['users.user_id'], ondelete='SET NULL'),
        sa.ForeignKeyConstraint(['project_id'], ['projects.project_id'], ondelete='CASCADE'),
        sa.CheckConstraint('rating >= 1 AND rating <= 5', name='ck_rag_feedback_rating'),
    )
    op.create_index('ix_rag_feedback_trace', 'rag_feedback', ['trace_id'])
    op.create_unique_constraint('uq_rag_feedback_trace_id', 'rag_feedback', ['trace_id'])
    op.create_index('ix_rag_feedback_project', 'rag_feedback', ['project_id', 'created_at'])
    op.create_index('ix_rag_feedback_user', 'rag_feedback', ['user_id', 'created_at'])

    # 2. rag_feedback_nodes
    op.create_table(
        'rag_feedback_nodes',
        sa.Column('id', sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column('feedback_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('node_id', sa.String(255), nullable=False),
        sa.Column('node_rank', sa.Integer(), nullable=True,
                  comment='Rank position in retrieval result (0-indexed)'),
        sa.Column('was_relevant', sa.Boolean(), nullable=True,
                  comment='User marked this node as relevant/irrelevant'),
        sa.ForeignKeyConstraint(['feedback_id'], ['rag_feedback.feedback_id'], ondelete='CASCADE'),
    )
    op.create_index('ix_rag_feedback_nodes', 'rag_feedback_nodes', ['feedback_id'])

    # 3. rag_feedback_insights
    op.create_table(
        'rag_feedback_insights',
        sa.Column('insight_id', postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text('gen_random_uuid()')),
        sa.Column('project_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('analysis_type', sa.String(50), nullable=True,
                  comment='intent_breakdown | retrieval_quality | overall'),
        sa.Column('metric_name', sa.String(100), nullable=True),
        sa.Column('dimension_value', sa.String(255), nullable=True),
        sa.Column('metric_value', sa.Float(), nullable=True),
        sa.Column('sample_count', sa.Integer(), nullable=True),
        sa.Column('confidence', sa.Float(), nullable=True),
        sa.Column('recommended_action', sa.String(50), nullable=True,
                  comment='increase_top_k | lower_similarity_threshold | adjust_strategy | none'),
        sa.Column('recommended_param_value', sa.Float(), nullable=True),
        sa.Column('created_at', sa.TIMESTAMP(), nullable=False, server_default=sa.func.now()),
        sa.Column('expires_at', sa.TIMESTAMP(), nullable=True),
        sa.ForeignKeyConstraint(['project_id'], ['projects.project_id'], ondelete='CASCADE'),
    )
    op.create_index('ix_rag_feedback_insights_project', 'rag_feedback_insights',
                    ['project_id', 'created_at'])

    # 4. rag_adaptive_settings
    op.create_table(
        'rag_adaptive_settings',
        sa.Column('setting_id', postgresql.UUID(as_uuid=True), primary_key=True,
                  server_default=sa.text('gen_random_uuid()')),
        sa.Column('project_id', postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column('intent', sa.String(50), nullable=True,
                  comment='NULL = applies to all intents'),
        sa.Column('top_k', sa.Integer(), nullable=True),
        sa.Column('similarity_threshold', sa.Float(), nullable=True),
        sa.Column('bm25_weight', sa.Float(), nullable=True),
        sa.Column('vector_weight', sa.Float(), nullable=True),
        sa.Column('reranker_top_k', sa.Integer(), nullable=True),
        sa.Column('source', sa.String(50), nullable=True,
                  comment='feedback_analysis | manual | default'),
        sa.Column('created_at', sa.TIMESTAMP(), nullable=False, server_default=sa.func.now()),
        sa.Column('expires_at', sa.TIMESTAMP(), nullable=True,
                  comment='NULL = permanent; set to 24h normally'),
        sa.ForeignKeyConstraint(['project_id'], ['projects.project_id'], ondelete='CASCADE'),
    )
    op.create_index('ix_rag_adaptive_project', 'rag_adaptive_settings',
                    ['project_id', 'intent', 'created_at'])


def downgrade() -> None:
    op.drop_table('rag_adaptive_settings')
    op.drop_table('rag_feedback_insights')
    op.drop_table('rag_feedback_nodes')
    op.drop_constraint('uq_rag_feedback_trace_id', 'rag_feedback', type_='unique')
    op.drop_table('rag_feedback')
