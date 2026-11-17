"""Persist owner-scoped image generation idempotency records."""

import sqlalchemy as sa
from alembic import op

revision = "004_image_idempotency"
down_revision = "003_image_budget"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "image_idempotency",
        sa.Column("owner_id", sa.String(128), primary_key=True),
        sa.Column("idempotency_key", sa.String(128), primary_key=True),
        sa.Column("request_hash", sa.String(64), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("response_json", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )


def downgrade():
    op.drop_table("image_idempotency")
