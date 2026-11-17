"""Resolve or fence uncertain image metadata writes before object compensation."""

from app.core.database import run_sync_db
from app.services.image.storage_quota import _quota_transaction, lock_storage_project
from sqlalchemy import text


def resolve_image_metadata_write(owner, reservation_id, path, size):
    async def resolve():
        async with _quota_transaction() as session:
            # Wait for an earlier writer to settle, or fence it before its
            # conditional reservation conversion can commit a future record.
            await lock_storage_project(session)
            row = (await session.execute(text(
                "SELECT bytes,state,object_path,image_id FROM image_storage_reservations "
                "WHERE reservation_id=:r AND user_id=:u"
            ), {"r": reservation_id, "u": owner})).fetchone()
            if not row or row[0] != size or row[2] != path:
                raise RuntimeError("Image metadata receipt cannot be verified")
            if row[1] == "committed":
                image = (await session.execute(text(
                    "SELECT id FROM image_records WHERE id=:i AND user_id=:u AND file_url=:p AND bytes=:b"
                ), {"i": row[3], "u": owner, "p": path, "b": size})).fetchone()
                if not image:
                    raise RuntimeError("Committed image association cannot be verified")
                return {"id": image[0]}
            if row[1] not in ("writing", "cleanup") or row[3] is not None:
                raise RuntimeError("Image metadata outcome requires explicit recovery")
            await session.execute(text(
                "UPDATE image_storage_reservations SET state='cleanup' "
                "WHERE reservation_id=:r AND user_id=:u AND state='writing'"
            ), {"r": reservation_id, "u": owner})
            return None

    return run_sync_db(resolve())
