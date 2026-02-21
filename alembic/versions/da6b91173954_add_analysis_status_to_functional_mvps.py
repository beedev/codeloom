"""add analysis_status to functional_mvps

Revision ID: da6b91173954
Revises: b7a3f9e1c2d4
Create Date: 2026-02-22 00:08:03.585789

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'da6b91173954'
down_revision: Union[str, None] = 'b7a3f9e1c2d4'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('functional_mvps', sa.Column('analysis_status', sa.String(length=20), nullable=True))
    op.add_column('functional_mvps', sa.Column('analysis_error', sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column('functional_mvps', 'analysis_error')
    op.drop_column('functional_mvps', 'analysis_status')
