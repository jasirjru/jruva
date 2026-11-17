"""Durable external outcomes and image storage reservation receipts."""

import sqlalchemy as sa
from alembic import op

revision = "007_operation_reservation_ledger"
down_revision = "006_distributed_idempotency"
branch_labels = None
depends_on = None


def upgrade():
    context = op.get_context()
    legacy_storage = (
        "SELECT q.user_id FROM image_storage_quotas q LEFT JOIN "
        "(SELECT user_id,COUNT(*) AS c,COALESCE(SUM(bytes),0) AS b FROM image_records GROUP BY user_id) r "
        "ON q.user_id=r.user_id WHERE q.user_id!='__project__' "
        "AND (q.image_count>COALESCE(r.c,0) OR q.total_bytes>COALESCE(r.b,0)) "
        "UNION ALL SELECT '__project__' WHERE "
        "(SELECT total_bytes FROM image_storage_quotas WHERE user_id='__project__') "
        "> (SELECT COALESCE(SUM(bytes),0) FROM image_records)"
    )
    if context.dialect.name == "postgresql":
        op.execute(sa.text("LOCK TABLE image_storage_quotas IN SHARE ROW EXCLUSIVE MODE"))
    if context.as_sql:
        if context.dialect.name != "postgresql":
            raise RuntimeError("SQLite ledger upgrade requires online legacy receipt inspection")
        op.execute(
            sa.text(
                "DO $audit$ BEGIN IF EXISTS (" + legacy_storage + ") THEN "
                "RAISE EXCEPTION 'Resolve legacy storage reservations before ledger migration'; "
                "END IF; END $audit$"
            )
        )
    elif op.get_bind().execute(sa.text(legacy_storage)).first():
        raise RuntimeError("Resolve legacy storage reservations with all old writers stopped before ledger migration")
    with op.batch_alter_table("image_idempotency") as batch:
        batch.add_column(sa.Column("operation_id", sa.String(128)))
        batch.add_column(sa.Column("external_state", sa.String(32), nullable=False, server_default="unstarted"))
        batch.add_column(sa.Column("provider_model", sa.String(128)))
        batch.add_column(sa.Column("provider_attempts", sa.Integer(), nullable=False, server_default="0"))
    legacy_id = (
        "'legacy-' || md5(json_build_array(owner_id,idempotency_key)::text)"
        if context.dialect.name == "postgresql"
        else "'legacy-' || lower(hex(randomblob(16)))"
    )
    op.execute(
        sa.text(
            "UPDATE image_idempotency SET status='indeterminate',external_state='indeterminate',operation_id="
            + legacy_id
            + " WHERE status!='completed'"
        )
    )
    op.create_table(
        "image_storage_reservations",
        sa.Column("reservation_id", sa.String(128), primary_key=True),
        sa.Column("user_id", sa.String(128), nullable=False),
        sa.Column("bytes", sa.BigInteger(), nullable=False),
        sa.Column("state", sa.String(32), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("object_path", sa.Text()),
        sa.Column("image_id", sa.String(128), unique=True),
        sa.Column("operation_id", sa.String(128)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint("bytes>0", name="ck_storage_reservation_bytes"),
        sa.CheckConstraint(
            "state IN ('active','writing','cleanup','committed','released','abandoned')",
            name="ck_storage_reservation_state",
        ),
    )
    op.create_index("ix_storage_reservation_owner_state", "image_storage_reservations", ["user_id", "state"])


def downgrade():
    if op.get_context().as_sql:
        raise RuntimeError("Downgrade requires online inspection of unresolved operations")
    if op.get_bind().execute(sa.text("SELECT COUNT(*) FROM image_idempotency WHERE status='indeterminate'")).scalar():
        raise RuntimeError("Resolve indeterminate operations before rollback; old workers can repeat them")
    if (
        op.get_bind()
        .execute(
            sa.text("SELECT COUNT(*) FROM image_storage_reservations WHERE state IN ('active','writing','cleanup')")
        )
        .scalar()
    ):
        raise RuntimeError("Resolve outstanding storage reservations before rollback")
    op.drop_table("image_storage_reservations")
    with op.batch_alter_table("image_idempotency") as batch:
        for name in ("provider_attempts", "provider_model", "external_state", "operation_id"):
            batch.drop_column(name)
