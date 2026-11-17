"""Safe responses for original bytes from the authenticated document library."""

import mimetypes
import re
import unicodedata
from urllib.parse import quote

from starlette.responses import Response


def document_download_response(content: bytes, filename: str) -> Response:
    """Preserve private bytes and Unicode names without serving active documents inline."""
    # Both separators matter because metadata can originate on a different OS.
    leaf = filename.replace("\\", "/").rsplit("/", 1)[-1]
    leaf = "".join(char for char in leaf if unicodedata.category(char) != "Cc")
    leaf = leaf.strip(" .") or "document"
    fallback = re.sub(r"[^a-zA-Z0-9._ -]", "_", leaf).strip(" .") or "document"
    encoded = quote(leaf, safe="", encoding="utf-8", errors="replace")

    mime = mimetypes.guess_type(leaf)[0] or "application/octet-stream"
    inline = mime == "application/pdf" and content.startswith(b"%PDF-")
    disposition = "inline" if inline else "attachment"
    headers = {
        "Content-Disposition": f'{disposition}; filename="{fallback}"; filename*=UTF-8\'\'{encoded}',
        "Cache-Control": "private, no-store",
        "X-Content-Type-Options": "nosniff",
    }
    if mime in {"text/html", "text/xml", "application/xml", "image/svg+xml"} or mime.endswith("+xml"):
        headers["Content-Security-Policy"] = "sandbox; default-src 'none'; base-uri 'none'; form-action 'none'"
    return Response(content=content, media_type=mime, headers=headers)
