"""Add durable document storage quotas, reservations, and user daily usage governance tables.

Revision ID: 008_document_storage_and_daily_quotas
Revises: 007_operation_reservation_ledger
Create Date: 2026-10-10
"""

import sqlalchemy as sa

from alembic import op

revision = "008_doc_storage_daily_quotas"
down_revision = "007_operation_reservation_ledger"
branch_labels = None
depends_on = None


def upgrade():
    context = op.get_context()
    if context.dialect.name == "postgresql":
        op.execute(sa.text("ALTER TABLE alembic_version ALTER COLUMN version_num TYPE VARCHAR(64)"))

    # 1. Document Storage Quotas
    op.create_table(
        "document_storage_quotas",
        sa.Column("user_id", sa.String(128), primary_key=True),
        sa.Column("file_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("total_bytes", sa.BigInteger(), nullable=False, server_default="0"),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
    )

    # 2. Document Storage Reservations Ledger
    op.create_table(
        "document_storage_reservations",
        sa.Column("reservation_id", sa.String(128), primary_key=True),
        sa.Column("user_id", sa.String(128), nullable=False),
        sa.Column("bytes", sa.BigInteger(), nullable=False),
        sa.Column("state", sa.String(32), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("storage_path", sa.Text(), nullable=True),
        sa.Column("file_id", sa.String(128), unique=True, nullable=True),
        sa.Column("filename", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("bytes > 0", name="ck_doc_reservation_bytes"),
        sa.CheckConstraint(
            "state IN ('active','writing','cleanup','committed','released','abandoned')",
            name="ck_doc_reservation_state",
        ),
    )
    op.create_index(
        "ix_doc_storage_reservation_owner_state",
        "document_storage_reservations",
        ["user_id", "state"],
    )

    # 3. User Daily Free-Tier Governance
    op.create_table(
        "user_daily_usage",
        sa.Column("user_id", sa.String(128), nullable=False),
        sa.Column("usage_date", sa.String(10), nullable=False),
        sa.Column("chat_prompts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("web_searches", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("document_uploads", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("audio_seconds", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("provider_calls", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("provider_errors", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_updated", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.PrimaryKeyConstraint("user_id", "usage_date", name="pk_user_daily_usage"),
    )
    op.create_index("idx_user_daily_usage_date", "user_daily_usage", ["usage_date"])

    # 4. Backfill document quotas from existing library_files if present
    op.execute(
        sa.text(
            "INSERT INTO document_storage_quotas (user_id, file_count, total_bytes) "
            "SELECT user_id, COUNT(*), COALESCE(SUM(file_size), 0) "
            "FROM library_files "
            "WHERE status = 'indexed' "
            "GROUP BY user_id "
            "ON CONFLICT (user_id) DO UPDATE SET "
            "file_count = EXCLUDED.file_count, total_bytes = EXCLUDED.total_bytes"
        )
    )
    op.execute(
        sa.text(
            "INSERT INTO document_storage_quotas (user_id, file_count, total_bytes) "
            "SELECT '__project__', COUNT(*), COALESCE(SUM(file_size), 0) "
            "FROM library_files "
            "WHERE status = 'indexed' "
            "ON CONFLICT (user_id) DO UPDATE SET "
            "file_count = EXCLUDED.file_count, total_bytes = EXCLUDED.total_bytes"
        )
    )


def downgrade():
    op.drop_table("user_daily_usage")
    op.drop_table("document_storage_reservations")
    op.drop_table("document_storage_quotas")
