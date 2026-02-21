"""add diagrams to functional_mvps

Revision ID: 54447161ed20
Revises: 325b05218d1c
Create Date: 2026-02-19 23:32:43.133691

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = '54447161ed20'
down_revision: Union[str, None] = '325b05218d1c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('functional_mvps', sa.Column('diagrams', postgresql.JSONB(astext_type=sa.Text()), nullable=True))


def downgrade() -> None:
    op.drop_column('functional_mvps', 'diagrams')
