"""Private document response contracts without configuration or hosted services."""

from urllib.parse import unquote

import pytest


def response(content: bytes, filename: str):
    from app.services.document_download import document_download_response

    return document_download_response(content, filename)


def test_pdf_preview_preserves_original_bytes_and_disables_shared_caching():
    content = b"%PDF-1.4\nsynthetic private document"
    result = response(content, "report.pdf")
    assert result.body == content
    assert result.headers["content-type"] == "application/pdf"
    assert result.headers["content-disposition"].startswith("inline;")
    assert "private" in result.headers["cache-control"]
    assert "no-store" in result.headers["cache-control"]
    assert result.headers["x-content-type-options"] == "nosniff"


@pytest.mark.parametrize(
    ("filename", "content"),
    [
        ("report.html", b"<html><script>window.syntheticMarker=1</script></html>"),
        ("report.HTML", b"<html><script>window.syntheticMarker=1</script></html>"),
        ("report.svg", b'<svg xmlns="http://www.w3.org/2000/svg"><script/></svg>'),
        ("report.xml", b'<?xml-stylesheet href="synthetic.xsl"?><document/>'),
        ("report.xhtml", b'<html xmlns="http://www.w3.org/1999/xhtml"><script/></html>'),
    ],
)
def test_active_documents_are_downloads_with_an_opaque_sandbox(filename, content):
    result = response(content, filename)
    assert result.body == content
    assert result.headers["content-disposition"].startswith("attachment;")
    directives = result.headers["content-security-policy"].split(";")
    assert "sandbox" in [item.strip() for item in directives]
    assert "default-src 'none'" in [item.strip() for item in directives]
    assert result.headers["x-content-type-options"] == "nosniff"
    assert "no-store" in result.headers["cache-control"]


def test_malayalam_filename_round_trips_through_rfc5987():
    filename = "മലയാളം.pdf"
    result = response(b"%PDF-1.4\nsynthetic", filename)
    header = result.headers["content-disposition"]
    header.encode("ascii")
    assert unquote(header.split("filename*=UTF-8''", 1)[1]) == filename
    assert '; filename="' in header


@pytest.mark.parametrize(
    "filename",
    [
        '../synthetic/unsafe"\r\nInjected: yes.pdf',
        '..\\synthetic\\unsafe"\x00\tname.pdf',
        "../synthetic/unsafe\x7fname.pdf",
    ],
)
def test_download_filename_cannot_inject_headers_or_directory_components(filename):
    result = response(b"%PDF-1.4\nsynthetic", filename)
    header = result.headers["content-disposition"]
    header.encode("ascii")
    fallback = header.split('filename="', 1)[1].split('";', 1)[0]
    restored = unquote(header.split("filename*=UTF-8''", 1)[1])
    assert "synthetic" not in fallback and "synthetic" not in restored
    assert '"' not in fallback
    assert not any(char in fallback + restored for char in "\r\n\x00\t\x7f/\\")
    assert "injected" not in result.headers


def test_mislabeled_pdf_is_not_previewed_inline():
    result = response(b"<html><script>window.syntheticMarker=1</script></html>", "report.pdf")
    assert result.headers["content-disposition"].startswith("attachment;")


def test_unknown_type_downloads_without_content_sniffing():
    result = response(b"synthetic", "report.unknown")
    assert result.headers["content-type"] == "application/octet-stream"
    assert result.headers["content-disposition"].startswith("attachment;")
    assert result.headers["x-content-type-options"] == "nosniff"


@pytest.mark.parametrize("filename", ["", "..", "/", "\\", "\r\n\x00"])
def test_empty_or_path_only_names_receive_a_usable_filename(filename):
    result = response(b"synthetic", filename)
    assert 'filename="document"' in result.headers["content-disposition"]


def test_surrogate_filename_does_not_crash_header_encoding():
    result = response(b"synthetic", "invalid\ud800.txt")
    result.headers["content-disposition"].encode("ascii")
