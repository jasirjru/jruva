"""Lightweight attachment metadata normalization shared by chat and providers."""

from typing import Any

_VISUAL_EXTENSIONS = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".webp": "image/webp", ".gif": "image/gif", ".bmp": "image/bmp",
    ".tiff": "image/tiff", ".svg": "image/svg+xml", ".ico": "image/x-icon",
    ".pdf": "application/pdf",
}


def attachment_mime_type(attachment: Any) -> str:
    """Recognize accepted visual filenames even with missing or generic MIME.

    Metadata is a routing hint, not proof that a payload is a valid image.
    Conflicting metadata must never demote a visual file to a text-only route.
    """
    if isinstance(attachment, dict):
        mime = attachment.get("mime_type")
        name = attachment.get("name")
    else:
        mime = getattr(attachment, "mime_type", "")
        name = getattr(attachment, "name", "")
    mime = str(mime or "").split(";", 1)[0].strip().lower()
    mime = {"image/jpg": "image/jpeg", "jpg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(mime, mime)
    if mime.startswith("image/") or mime == "application/pdf":
        return mime
    name = str(name or "").strip().lower()
    extension = "." + name.rsplit(".", 1)[-1] if "." in name else ""
    return _VISUAL_EXTENSIONS.get(extension, mime)


def is_visual_attachment(attachment: Any) -> bool:
    mime = attachment_mime_type(attachment)
    return mime.startswith("image/") or mime == "application/pdf"
