"""File upload validation: filename sanitization, MIME sniffing, quota.

Pure-function module — no app state, no I/O except the DB-backed quota
helper which takes an explicit connection. Lets us unit-test the security
critical bits without spinning the FastAPI app.

Why a hand-rolled MIME sniffer instead of python-magic / filetype:
adding a dependency for ~30 lines of magic-byte tables isn't worth it,
and python-magic needs libmagic on the host (Windows install pain). The
table covers the formats users actually upload here; anything we miss
falls through to the client-claimed MIME, which the route stores as a
separate column so a mismatch is observable.
"""

from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiosqlite


# --- MIME sniffing -------------------------------------------------------

# Magic-byte table: (offset, prefix, mime).  Order matters only for prefixes
# that share a byte; we keep the dangerous executable signatures last so a
# legitimate document that happens to start with "MZ" elsewhere isn't
# misclassified.
_MAGIC: list[tuple[int, bytes, str]] = [
    (0, b"%PDF-",                "application/pdf"),
    (0, b"\x89PNG\r\n\x1a\n",    "image/png"),
    (0, b"\xff\xd8\xff",         "image/jpeg"),
    (0, b"GIF87a",               "image/gif"),
    (0, b"GIF89a",               "image/gif"),
    (0, b"BM",                   "image/bmp"),
    (0, b"II*\x00",              "image/tiff"),
    (0, b"MM\x00*",              "image/tiff"),
    (0, b"PK\x03\x04",           "application/zip"),       # docx/xlsx/pptx/epub/odt
    (0, b"PK\x05\x06",           "application/zip"),
    (0, b"PK\x07\x08",           "application/zip"),
    (0, b"\x1f\x8b",             "application/gzip"),
    (0, b"7z\xbc\xaf\x27\x1c",   "application/x-7z-compressed"),
    (0, b"Rar!\x1a\x07",         "application/vnd.rar"),
    (0, b"ID3",                  "audio/mpeg"),
    (0, b"OggS",                 "audio/ogg"),
    (0, b"fLaC",                 "audio/flac"),
    (0, b"\x1a\x45\xdf\xa3",     "video/webm"),            # also Matroska
    (0, b"\x7fELF",              "application/x-executable"),
    (0, b"MZ",                   "application/x-msdownload"),
    (0, b"\xca\xfe\xba\xbe",     "application/x-mach-binary"),
    (0, b"\xcf\xfa\xed\xfe",     "application/x-mach-binary"),
    (0, b"\xfe\xed\xfa\xce",     "application/x-mach-binary"),
    (0, b"\xfe\xed\xfa\xcf",     "application/x-mach-binary"),
]

# RIFF containers carry their kind at offset 8 (4 bytes after "RIFF" + 4-byte size).
_RIFF_KINDS = {
    b"WEBP": "image/webp",
    b"WAVE": "audio/wav",
    b"AVI ": "video/x-msvideo",
}

EXECUTABLE_MIMES = frozenset({
    "application/x-executable",
    "application/x-msdownload",
    "application/x-mach-binary",
})


def sniff_mime(data: bytes, fallback: str = "") -> str:
    """Return a server-detected MIME type from the file's leading bytes.

    Falls back to `fallback` (typically the client-supplied Content-Type)
    or "application/octet-stream" when nothing matches.
    """
    if not data:
        return fallback or "application/octet-stream"

    head = data[:512]

    for offset, prefix, mime in _MAGIC:
        if head[offset:offset + len(prefix)] == prefix:
            return mime

    # RIFF family (WebP, WAV, AVI)
    if head[:4] == b"RIFF" and len(head) >= 12:
        kind = head[8:12]
        if kind in _RIFF_KINDS:
            return _RIFF_KINDS[kind]

    # MP4 family — "ftyp" box at offset 4
    if len(head) >= 12 and head[4:8] == b"ftyp":
        brand = head[8:12]
        if brand[:3] in (b"M4A", b"M4B"):
            return "audio/mp4"
        return "video/mp4"

    # MP3 sync frame (no ID3 tag)
    if len(head) >= 2 and head[0] == 0xff and (head[1] & 0xe0) == 0xe0:
        return "audio/mpeg"

    # Text heuristic — printable UTF-8 dominates the head.
    try:
        text = head.decode("utf-8")
    except UnicodeDecodeError:
        return fallback or "application/octet-stream"

    printable = sum(1 for c in text if c.isprintable() or c in "\r\n\t")
    if printable / max(1, len(text)) > 0.95:
        stripped = text.lstrip()
        if stripped.startswith("<svg") or "<svg" in stripped[:200]:
            return "image/svg+xml"
        if stripped.startswith("<?xml"):
            return "application/xml"
        if stripped.startswith("<!DOCTYPE html") or stripped.startswith("<html"):
            return "text/html"
        if stripped[:1] in ("{", "["):
            return "application/json"
        return "text/plain"

    return fallback or "application/octet-stream"


def is_mime_mismatch(claimed: str, sniffed: str) -> bool:
    """Return True when client-claimed and server-sniffed MIMEs disagree
    in a way the operator should care about.

    Empty values are not a mismatch (nothing to compare).  Equal values
    are not a mismatch.  Office documents legitimately sniff as `application/zip`
    so we accept that pairing.  Everything else that differs at the type/
    subtype level is flagged.
    """
    if not claimed or not sniffed:
        return False
    claimed = claimed.split(";", 1)[0].strip().lower()
    sniffed = sniffed.split(";", 1)[0].strip().lower()
    if claimed == sniffed:
        return False
    # OOXML / OpenDocument / EPUB are ZIP containers — sniffer can't see
    # past the outer envelope without unzipping, so trust the client claim
    # when the sniffed type is generic ZIP.
    return not (
        sniffed == "application/zip" and (
            claimed.startswith("application/vnd.openxmlformats-officedocument")
            or claimed.startswith("application/vnd.oasis.opendocument")
            or claimed in {"application/epub+zip", "application/java-archive"}
        )
    )


# --- Filename sanitization ----------------------------------------------

# Control chars (incl. NULL) plus Windows-reserved chars.  Forward/back
# slash is handled separately because we want to drop entire path components,
# not just strip the slash.
_BAD_NAME_CHARS = re.compile(r"[\x00-\x1f\x7f<>:\"|?*]")

# Reserved DOS device names — Windows refuses to create these even with an
# extension.  Case-insensitive comparison against the stem.
_RESERVED_NAMES = frozenset({
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
})


def sanitize_filename(raw: str | None, *, max_len: int = 255) -> str:
    """Return a safe display filename.

    Strips path separators, control bytes, NULLs, and Windows-reserved
    characters; NFC-normalises unicode so visually-identical names compare
    equal; trims surrounding whitespace and trailing dots (Windows silently
    drops them); falls back to "upload" if the result is empty.  Truncates
    to `max_len` while preserving the extension when possible.
    """
    if not raw:
        return "upload"
    name = unicodedata.normalize("NFC", raw)
    name = _BAD_NAME_CHARS.sub("", name)
    # Discard any directory components — last segment wins.
    name = name.replace("\\", "/").rsplit("/", 1)[-1]
    name = name.strip().strip(".").strip()
    if not name:
        return "upload"

    stem, dot, _ext = name.rpartition(".")
    check_against = stem if dot else name
    if check_against.lower() in _RESERVED_NAMES:
        name = f"_{name}"

    if len(name) <= max_len:
        return name
    if "." in name:
        stem, _, ext = name.rpartition(".")
        ext = ext[:32]  # paranoid cap on extension length
        keep = max_len - len(ext) - 1
        if keep > 0:
            return f"{stem[:keep]}.{ext}"
    return name[:max_len]


# --- Per-user storage quota ---------------------------------------------

async def get_user_storage_used(conn: aiosqlite.Connection, user_id: str) -> int:
    """Return the user's current upload byte total.

    Sums `uploads.size_bytes` directly — this over-counts when multiple
    uploads dedup to the same blob, which matches the "I uploaded N bytes"
    user mental model and errs toward enforcing the quota (worst case: we
    refuse a fresh upload that would have deduped, never the reverse).
    """
    if not user_id:
        return 0
    cursor = await conn.execute(
        "SELECT COALESCE(SUM(size_bytes), 0) FROM uploads WHERE user_id = ?",
        (user_id,),
    )
    row = await cursor.fetchone()
    return int(row[0]) if row else 0


async def check_quota(
    conn: aiosqlite.Connection,
    user_id: str,
    *,
    incoming_bytes: int,
    quota_bytes: int,
) -> tuple[bool, int, int]:
    """Check whether `incoming_bytes` more would exceed the user's quota.

    Returns (ok, used_bytes, quota_bytes). When `quota_bytes <= 0` the
    check is disabled and `ok` is always True.
    """
    if quota_bytes <= 0:
        return True, 0, 0
    used = await get_user_storage_used(conn, user_id)
    return (used + incoming_bytes <= quota_bytes), used, quota_bytes
