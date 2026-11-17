"""Persist account-wide and owner daily image attempt reservations."""

import sqlalchemy as sa
from alembic import op

revision = "003_image_budget"
down_revision = "002_tenant_privacy"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "image_generation_budgets",
        sa.Column("scope", sa.String(128), primary_key=True),
        sa.Column("day", sa.String(10), primary_key=True),
        sa.Column("used", sa.Integer(), nullable=False, server_default="0"),
    )


def downgrade():
    op.drop_table("image_generation_budgets")
