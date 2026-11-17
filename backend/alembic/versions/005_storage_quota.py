"""Add durable storage quota tracking and image record byte size.

Revision ID: 005_storage_quota
Revises: 004_image_idempotency
Create Date: 2026-10-08
"""

import sqlalchemy as sa
from alembic import op

revision = "005_storage_quota"
down_revision = "004_image_idempotency"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "image_storage_quotas",
        sa.Column("user_id", sa.String(128), primary_key=True),
        sa.Column("image_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_bytes", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )
    # Add byte size to image_records for accurate tracking and quota recovery
    op.add_column("image_records", sa.Column("bytes", sa.Integer(), nullable=False, server_default="0"))

    # Backfill per-user and project quotas from existing image_records if present
    op.execute(
        sa.text(
            "INSERT INTO image_storage_quotas (user_id, image_count, total_bytes) "
            "SELECT user_id, COUNT(*), COALESCE(SUM(bytes), 0) "
            "FROM image_records "
            "GROUP BY user_id "
            "ON CONFLICT (user_id) DO UPDATE SET "
            "image_count = EXCLUDED.image_count, total_bytes = EXCLUDED.total_bytes"
        )
    )
    op.execute(
        sa.text(
            "INSERT INTO image_storage_quotas (user_id, image_count, total_bytes) "
            "SELECT '__project__', COUNT(*), COALESCE(SUM(bytes), 0) "
            "FROM image_records "
            "WHERE TRUE "
            "ON CONFLICT (user_id) DO UPDATE SET "
            "image_count = EXCLUDED.image_count, total_bytes = EXCLUDED.total_bytes"
        )
    )


def downgrade():
    with op.batch_alter_table("image_records") as batch_op:
        batch_op.drop_column("bytes")
    op.drop_table("image_storage_quotas")
