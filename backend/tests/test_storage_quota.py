"""Tests for durable retained image storage quota enforcement (R06).

Verifies per-user count, per-user bytes, project-wide bytes, and 10 MB object limit.
Covers atomic reservations, deletion recovery, failed upload rollback, and user isolation.
"""

import io
from unittest.mock import patch

import pytest
from app.services.image.genuine_service import save_owned_image
from app.services.image.providers import ImageProviderError
from app.services.image.storage_quota import (
    get_storage_quota_usage,
    reconcile_storage_quota,
    reserve_storage_quota,
)
from app.services.workspace_store import workspace_store
from PIL import Image


@pytest.fixture(autouse=True)
def clean_quota_tables():
    """Ensure a clean state for quota testing."""
    with workspace_store._get_sqlite_connection() as conn:
        conn.execute("DROP TABLE IF EXISTS image_storage_quotas")
        conn.execute("DROP TABLE IF EXISTS image_storage_reservations")
        conn.execute("DELETE FROM image_records WHERE user_id LIKE 'quota_%'")
        conn.commit()
    yield
    with workspace_store._get_sqlite_connection() as conn:
        conn.execute("DROP TABLE IF EXISTS image_storage_quotas")
        conn.execute("DROP TABLE IF EXISTS image_storage_reservations")
        conn.execute("DELETE FROM image_records WHERE user_id LIKE 'quota_%'")
        conn.commit()


def _dummy_png(size=(100, 100), color=(255, 0, 0)) -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", size, color=color)
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_single_object_size_limit():
    """Images exceeding the 10 MB per-object bucket limit are rejected with HTTP 400."""
    eleven_mb = 11 * 1024 * 1024
    with pytest.raises(ImageProviderError) as exc_info:
        reserve_storage_quota(
            "quota_user_1",
            eleven_mb,
            max_object_bytes=10 * 1024 * 1024,
        )
    assert exc_info.value.status == 400
    assert "exceeds the maximum allowed limit" in str(exc_info.value)


def test_per_user_image_count_limit():
    """A user cannot exceed their allowed image count quota."""
    owner = "quota_user_count"
    max_count = 3
    byte_size = 1024

    for _ in range(max_count):
        reserve_storage_quota(owner, byte_size, max_user_count=max_count)

    usage = get_storage_quota_usage(owner)
    assert usage["image_count"] == 3

    # Next attempt should be rejected with 429
    with pytest.raises(ImageProviderError) as exc_info:
        reserve_storage_quota(owner, byte_size, max_user_count=max_count)
    assert exc_info.value.status == 429
    assert "Maximum retained image count (3) reached" in str(exc_info.value)


def test_per_user_image_bytes_limit():
    """A user cannot exceed their allowed byte storage budget."""
    owner = "quota_user_bytes"
    max_bytes = 5 * 1024 * 1024  # 5 MB
    chunk_size = 2 * 1024 * 1024  # 2 MB

    reserve_storage_quota(owner, chunk_size, max_user_bytes=max_bytes)
    reserve_storage_quota(owner, chunk_size, max_user_bytes=max_bytes)

    usage = get_storage_quota_usage(owner)
    assert usage["total_bytes"] == 4 * 1024 * 1024

    # Third 2MB chunk would reach 6MB > 5MB limit
    with pytest.raises(ImageProviderError) as exc_info:
        reserve_storage_quota(owner, chunk_size, max_user_bytes=max_bytes)
    assert exc_info.value.status == 429
    assert "User retained storage quota" in str(exc_info.value)


def test_project_wide_bytes_limit():
    """Shared project capacity cannot be exceeded by multiple users combined."""
    max_proj = 10 * 1024 * 1024  # 10 MB total project capacity
    max_usr = 8 * 1024 * 1024

    reserve_storage_quota("quota_user_a", 6 * 1024 * 1024, max_user_bytes=max_usr, max_project_bytes=max_proj)
    
    # User B attempting 5 MB would make project total 11 MB > 10 MB
    with pytest.raises(ImageProviderError) as exc_info:
        reserve_storage_quota("quota_user_b", 5 * 1024 * 1024, max_user_bytes=max_usr, max_project_bytes=max_proj)
    assert exc_info.value.status == 429
    assert "Total project image storage capacity reached" in str(exc_info.value)


def test_delete_image_releases_quota():
    """Deleting an image releases count and bytes from user and project totals."""
    owner = "quota_user_del"
    png_data = _dummy_png()
    data_len = len(png_data)

    reserve_storage_quota(owner, data_len)
    rec = workspace_store.record_image(
        prompt="quota test",
        file_url="local://quota_test.png",
        model="cf/flux",
        user_id=owner,
        bytes=data_len,
    )
    usage_before = get_storage_quota_usage(owner)
    assert usage_before["image_count"] == 1
    assert usage_before["total_bytes"] == data_len

    # Delete via workspace_store triggers quota release
    deleted = workspace_store.delete_image(rec["id"])
    assert deleted is True

    usage_after = get_storage_quota_usage(owner)
    assert usage_after["image_count"] == 0
    assert usage_after["total_bytes"] == 0


def test_failed_save_releases_reserved_quota():
    """If storage save fails, the pre-reserved quota is safely rolled back."""
    owner = "quota_user_fail"
    from tests.private_accounts import private_account_headers
    private_account_headers(owner)
    png_data = _dummy_png()

    import asyncio

    from app.services.storage import storage_service

    # Simulate storage failure and mock active_account
    with (
        patch("app.services.image.genuine_service.active_account", return_value=True),
        patch.object(storage_service, "save_bytes", side_effect=OSError("Storage disk full")),
        pytest.raises(OSError),
    ):
        asyncio.run(
            save_owned_image(
                png_data,
                owner=owner,
                prompt="fail test",
                enhanced="fail test",
                model="cf/flux",
                tier="B",
                style="photorealistic",
            )
        )

    usage = get_storage_quota_usage(owner)
    assert usage["image_count"] == 0
    assert usage["total_bytes"] == 0


def test_user_quotas_are_strictly_isolated():
    """Exhausting quota for User A does not prevent User B from generating."""
    owner_a = "quota_user_iso_a"
    owner_b = "quota_user_iso_b"
    max_count = 2

    # Exhaust user A
    for _ in range(max_count):
        reserve_storage_quota(owner_a, 1024, max_user_count=max_count)

    with pytest.raises(ImageProviderError):
        reserve_storage_quota(owner_a, 1024, max_user_count=max_count)

    # User B can still successfully reserve
    reserve_storage_quota(owner_b, 1024, max_user_count=max_count)
    usage_b = get_storage_quota_usage(owner_b)
    assert usage_b["image_count"] == 1


def test_reconcile_heals_counter_drift():
    """Reconcile recalculates exact sum from image_records rows."""
    owner = "quota_user_recon"
    
    # Create two records directly without updating quota table
    workspace_store.record_image(
        prompt="p1", file_url="url1", model="m", user_id=owner, bytes=5000
    )
    workspace_store.record_image(
        prompt="p2", file_url="url2", model="m", user_id=owner, bytes=7000
    )

    # Run reconciliation
    reconcile_storage_quota(owner)

    usage = get_storage_quota_usage(owner)
    assert usage["image_count"] == 2
    assert usage["total_bytes"] == 12000


def test_concurrent_writes_cannot_bypass_quota():
    owner = "quota_user_concurrent"
    max_count = 2
    from concurrent.futures import ThreadPoolExecutor

    def attempt_reserve(idx):
        try:
            reserve_storage_quota(owner, 1024, max_user_count=max_count)
            return "success"
        except ImageProviderError as exc:
            assert exc.status == 429
            return "exhausted"

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(attempt_reserve, range(8)))

    assert results.count("success") == 2
    assert results.count("exhausted") == 6
    usage = get_storage_quota_usage(owner)
    assert usage["image_count"] == 2
