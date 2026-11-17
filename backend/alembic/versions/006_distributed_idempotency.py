"""Distributed image idempotency with worker leasing and fencing tokens.

Revision ID: 006_distributed_idempotency
Revises: 005_storage_quota
Create Date: 2026-10-08
"""

import sqlalchemy as sa
from alembic import op

revision = "006_distributed_idempotency"
down_revision = "005_storage_quota"
branch_labels = None
depends_on = None


def upgrade():
    # Rebuild SQLite tables for CURRENT_TIMESTAMP defaults on populated rows;
    # PostgreSQL retains ordinary ALTER TABLE operations (including offline SQL).
    recreate = "always" if op.get_context().dialect.name == "sqlite" else "auto"
    with op.batch_alter_table("image_idempotency", recreate=recreate) as batch_op:
        batch_op.add_column(sa.Column("worker_id", sa.String(128), nullable=True))
        batch_op.add_column(sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True))
        batch_op.add_column(sa.Column("fencing_token", sa.BigInteger(), nullable=False, server_default="1"))
        batch_op.add_column(sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=True))


def downgrade():
    with op.batch_alter_table("image_idempotency") as batch_op:
        batch_op.drop_column("updated_at")
        batch_op.drop_column("fencing_token")
        batch_op.drop_column("lease_expires_at")
        batch_op.drop_column("worker_id")
