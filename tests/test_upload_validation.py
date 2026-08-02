"""Unit tests for the upload validation module.

The validation module is the security perimeter of /api/files/upload.
Test the wire-protocol shapes: bad filenames, fake MIMEs, quota math.
No DB or app fixtures — `check_quota` takes an explicit aiosqlite
connection so we can hand it an in-memory DB.
"""

from __future__ import annotations

import asyncio

import pytest

from augmentum.vfs.validation import (
    EXECUTABLE_MIMES,
    check_quota,
    get_user_storage_used,
    is_mime_mismatch,
    sanitize_filename,
    sniff_mime,
)

# --- sanitize_filename --------------------------------------------------

class TestSanitizeFilename:
    def test_plain_name_passes_through(self):
        assert sanitize_filename("photo.png") == "photo.png"

    def test_unicode_preserved(self):
        # Japanese + emoji should round-trip; we only strip control chars.
        assert sanitize_filename("写真.png") == "写真.png"
        assert sanitize_filename("party 🎉.txt") == "party 🎉.txt"

    def test_path_traversal_stripped(self):
        # Both unix and windows separators get neutralised — keep the leaf.
        assert sanitize_filename("../../etc/passwd") == "passwd"
        assert sanitize_filename("..\\..\\windows\\system32\\cmd.exe") == "cmd.exe"
        assert sanitize_filename("/abs/path/file.txt") == "file.txt"

    def test_null_byte_dropped(self):
        # Classic null-byte truncation attack — filename must not contain \x00.
        cleaned = sanitize_filename("safe.txt\x00.exe")
        assert "\x00" not in cleaned
        assert cleaned == "safe.txt.exe"

    def test_control_chars_dropped(self):
        # \x01-\x1f and \x7f are stripped wholesale.
        for bad in ("\x01", "\x05", "\x1f", "\x7f"):
            cleaned = sanitize_filename(f"name{bad}thing.txt")
            assert bad not in cleaned

    def test_windows_reserved_chars_dropped(self):
        for bad in ("<", ">", ":", '"', "|", "?", "*"):
            cleaned = sanitize_filename(f"a{bad}b.txt")
            assert bad not in cleaned

    def test_empty_returns_fallback(self):
        assert sanitize_filename("") == "upload"
        assert sanitize_filename(None) == "upload"
        assert sanitize_filename("   ") == "upload"
        assert sanitize_filename("...") == "upload"
        assert sanitize_filename("/") == "upload"

    def test_trailing_dots_stripped(self):
        # Windows silently drops trailing dots — strip them ourselves so
        # downloads round-trip with the same name.
        assert sanitize_filename("file.txt...") == "file.txt"

    def test_reserved_dos_names_prefixed(self):
        # CON, NUL, COM1 etc. cannot exist on Windows; prefix with `_`.
        assert sanitize_filename("CON") == "_CON"
        assert sanitize_filename("nul.txt") == "_nul.txt"
        assert sanitize_filename("COM1.log") == "_COM1.log"
        # Not reserved — leave alone.
        assert sanitize_filename("console.txt") == "console.txt"

    def test_max_length_truncation_keeps_extension(self):
        long_stem = "a" * 300
        result = sanitize_filename(f"{long_stem}.png", max_len=255)
        assert len(result) <= 255
        assert result.endswith(".png")

    def test_no_extension_truncation(self):
        result = sanitize_filename("a" * 300, max_len=64)
        assert len(result) == 64


# --- sniff_mime ---------------------------------------------------------

class TestSniffMime:
    def test_pdf(self):
        assert sniff_mime(b"%PDF-1.4\n%...\n") == "application/pdf"

    def test_png(self):
        png_header = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        assert sniff_mime(png_header) == "image/png"

    def test_jpeg(self):
        assert sniff_mime(b"\xff\xd8\xff\xe0\x00\x10JFIF") == "image/jpeg"

    def test_gif87(self):
        assert sniff_mime(b"GIF87a" + b"\x00" * 50) == "image/gif"

    def test_gif89(self):
        assert sniff_mime(b"GIF89a" + b"\x00" * 50) == "image/gif"

    def test_webp(self):
        # RIFF + 4-byte size + WEBP marker.
        assert sniff_mime(b"RIFF\x00\x00\x00\x00WEBP" + b"\x00" * 50) == "image/webp"

    def test_wav(self):
        assert sniff_mime(b"RIFF\x00\x00\x00\x00WAVE" + b"\x00" * 50) == "audio/wav"

    def test_zip(self):
        # Local file header — covers .docx/.xlsx/.pptx/.epub.
        assert sniff_mime(b"PK\x03\x04" + b"\x00" * 50) == "application/zip"

    def test_gzip(self):
        assert sniff_mime(b"\x1f\x8b\x08" + b"\x00" * 50) == "application/gzip"

    def test_mp3_with_id3(self):
        assert sniff_mime(b"ID3\x04\x00\x00" + b"\x00" * 50) == "audio/mpeg"

    def test_mp3_sync_frame(self):
        # No ID3 tag — raw MPEG audio frame.
        assert sniff_mime(b"\xff\xfb\x90\x00" + b"\x00" * 50) == "audio/mpeg"

    def test_mp4(self):
        # ftyp box at offset 4.
        assert sniff_mime(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 50) == "video/mp4"

    def test_m4a(self):
        assert sniff_mime(b"\x00\x00\x00\x18ftypM4A " + b"\x00" * 50) == "audio/mp4"

    def test_windows_executable_detected(self):
        # Critical: malicious upload claiming to be a PDF must not hide
        # behind the client Content-Type — magic bytes win.
        sniffed = sniff_mime(b"MZ\x90\x00\x03\x00", fallback="application/pdf")
        assert sniffed == "application/x-msdownload"
        assert sniffed in EXECUTABLE_MIMES

    def test_elf_binary_detected(self):
        sniffed = sniff_mime(b"\x7fELF\x02\x01\x01", fallback="image/png")
        assert sniffed == "application/x-executable"
        assert sniffed in EXECUTABLE_MIMES

    def test_plain_text(self):
        assert sniff_mime(b"Hello, world!\nThis is plain text.") == "text/plain"

    def test_html(self):
        assert sniff_mime(b"<!DOCTYPE html><html><body>x</body></html>") == "text/html"

    def test_json_object(self):
        assert sniff_mime(b'{"key": "value"}') == "application/json"

    def test_json_array(self):
        assert sniff_mime(b"[1, 2, 3]") == "application/json"

    def test_xml(self):
        assert sniff_mime(b"<?xml version='1.0'?><root/>") == "application/xml"

    def test_svg(self):
        assert sniff_mime(b"<svg xmlns='http://www.w3.org/2000/svg'></svg>") == "image/svg+xml"

    def test_unknown_falls_back_to_provided(self):
        # Random binary garbage with a fallback hint.
        assert sniff_mime(b"\x00\x01\x02\x03\xff\xfe", fallback="x/y") == "x/y"

    def test_unknown_no_fallback_is_octet_stream(self):
        assert sniff_mime(b"\x00\x01\x02\x03\xff\xfe") == "application/octet-stream"

    def test_empty_input(self):
        assert sniff_mime(b"") == "application/octet-stream"
        assert sniff_mime(b"", fallback="text/plain") == "text/plain"


# --- is_mime_mismatch ---------------------------------------------------

class TestMimeMismatch:
    def test_equal_not_mismatch(self):
        assert not is_mime_mismatch("image/png", "image/png")

    def test_empty_not_mismatch(self):
        # Nothing to compare = no warning.
        assert not is_mime_mismatch("", "image/png")
        assert not is_mime_mismatch("image/png", "")

    def test_charset_suffix_ignored(self):
        # Both sides may carry "; charset=utf-8" — that shouldn't trip the check.
        assert not is_mime_mismatch("text/plain; charset=utf-8", "text/plain")

    def test_office_doc_zip_not_flagged(self):
        # docx legitimately sniffs as application/zip.
        claim = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        assert not is_mime_mismatch(claim, "application/zip")

    def test_epub_zip_not_flagged(self):
        assert not is_mime_mismatch("application/epub+zip", "application/zip")

    def test_exe_claiming_pdf_is_flagged(self):
        assert is_mime_mismatch("application/pdf", "application/x-msdownload")

    def test_png_claiming_jpeg_is_flagged(self):
        assert is_mime_mismatch("image/jpeg", "image/png")


# --- Quota --------------------------------------------------------------

@pytest.fixture
async def db():
    import aiosqlite
    conn = await aiosqlite.connect(":memory:")
    await conn.execute("""
        CREATE TABLE uploads (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            size_bytes INTEGER NOT NULL
        )
    """)
    await conn.commit()
    yield conn
    await conn.close()


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


class TestQuota:
    def test_storage_used_empty(self):
        async def go():
            import aiosqlite
            conn = await aiosqlite.connect(":memory:")
            await conn.execute(
                "CREATE TABLE uploads (id TEXT PRIMARY KEY, user_id TEXT, size_bytes INTEGER)"
            )
            await conn.commit()
            assert await get_user_storage_used(conn, "u1") == 0
            await conn.close()
        _run(go())

    def test_storage_used_sums_across_rows(self):
        async def go():
            import aiosqlite
            conn = await aiosqlite.connect(":memory:")
            await conn.execute(
                "CREATE TABLE uploads (id TEXT PRIMARY KEY, user_id TEXT, size_bytes INTEGER)"
            )
            await conn.executemany(
                "INSERT INTO uploads (id, user_id, size_bytes) VALUES (?, ?, ?)",
                [("a", "u1", 100), ("b", "u1", 250), ("c", "u2", 999)],
            )
            await conn.commit()
            assert await get_user_storage_used(conn, "u1") == 350
            assert await get_user_storage_used(conn, "u2") == 999
            assert await get_user_storage_used(conn, "u3") == 0
            await conn.close()
        _run(go())

    def test_check_quota_disabled_when_zero(self):
        async def go():
            import aiosqlite
            conn = await aiosqlite.connect(":memory:")
            await conn.execute(
                "CREATE TABLE uploads (id TEXT PRIMARY KEY, user_id TEXT, size_bytes INTEGER)"
            )
            ok, used, q = await check_quota(conn, "u1", incoming_bytes=10**12, quota_bytes=0)
            assert ok is True
            assert (used, q) == (0, 0)
            await conn.close()
        _run(go())

    def test_check_quota_allows_under(self):
        async def go():
            import aiosqlite
            conn = await aiosqlite.connect(":memory:")
            await conn.execute(
                "CREATE TABLE uploads (id TEXT PRIMARY KEY, user_id TEXT, size_bytes INTEGER)"
            )
            await conn.execute(
                "INSERT INTO uploads (id, user_id, size_bytes) VALUES ('a', 'u1', 500)"
            )
            await conn.commit()
            ok, used, q = await check_quota(conn, "u1", incoming_bytes=300, quota_bytes=1000)
            assert ok and used == 500 and q == 1000
            await conn.close()
        _run(go())

    def test_check_quota_blocks_over(self):
        async def go():
            import aiosqlite
            conn = await aiosqlite.connect(":memory:")
            await conn.execute(
                "CREATE TABLE uploads (id TEXT PRIMARY KEY, user_id TEXT, size_bytes INTEGER)"
            )
            await conn.execute(
                "INSERT INTO uploads (id, user_id, size_bytes) VALUES ('a', 'u1', 900)"
            )
            await conn.commit()
            ok, used, q = await check_quota(conn, "u1", incoming_bytes=200, quota_bytes=1000)
            assert ok is False
            assert used == 900 and q == 1000
            await conn.close()
        _run(go())
