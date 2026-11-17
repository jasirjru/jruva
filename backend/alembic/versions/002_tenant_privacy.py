"""Owner-linked diagnostics and account deletion state; minimize legacy log content."""
import sqlalchemy as sa
from alembic import op

revision = "002_tenant_privacy"
down_revision = "001_initial_schema"
branch_labels = None
depends_on = None


def upgrade():
    op.add_column("users", sa.Column("google_sub", sa.String(255), nullable=True))
    op.create_index("idx_users_google_sub", "users", ["google_sub"], unique=True)
    op.add_column("users", sa.Column("deletion_pending", sa.Boolean(), nullable=False, server_default=sa.false()))
    with op.batch_alter_table("conversation_logs") as batch:
        batch.add_column(sa.Column("user_id", sa.String(128), nullable=True))
        batch.create_index("idx_conversation_logs_user_id", ["user_id"])
        batch.create_foreign_key("fk_conversation_logs_user", "users", ["user_id"], ["id"], ondelete="CASCADE")
    op.execute("UPDATE conversation_logs SET user_query = '', model_response = '', retrieved_chunks = '[]', shadow_response = NULL")


def downgrade():
    # Previously redacted content is intentionally unrecoverable.
    with op.batch_alter_table("conversation_logs") as batch:
        batch.drop_constraint("fk_conversation_logs_user", type_="foreignkey")
        batch.drop_index("idx_conversation_logs_user_id")
        batch.drop_column("user_id")
    op.drop_column("users", "deletion_pending")
    op.drop_index("idx_users_google_sub", table_name="users")
    op.drop_column("users", "google_sub")
