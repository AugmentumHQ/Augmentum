"""Decompression-bomb guard + magic-byte sniffing for user archives.

Traversal ("zip-slip") is already guarded at every extract site in this
codebase; what was missing (2026-07-07 security triage, closed
2026-07-18) is the OTHER archive attack: a tiny upload that inflates to
terabytes and fills the disk / OOMs the process. ``ensure_zip_sane``
inspects the central directory BEFORE any member is decompressed, so a
bomb is rejected at ~zero cost.

Use it at every site that opens a zip whose bytes a USER (or a fetched
remote) supplied: uploads, library/artifact imports, image-pipeline
imports, epub/cbz readers, provider downloads.

``sniff_kind`` gives magic-byte format detection for import paths that
previously trusted the file extension alone — a ``.epub`` that's really
an executable, or a ``.zip`` that's really a 40GB sparse tar, gets
named for what it is.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Ceilings chosen to clear every legitimate archive this app handles
# (game bundles, comic CBZs, epubs, model-pipeline zips — all well under
# 4GB uncompressed) while making bombs unrepresentable. Callers with
# tighter domains should pass tighter values, never looser ones without
# a reason in a comment.
DEFAULT_MAX_UNCOMPRESSED = 4 * 1024**3  # 4 GiB total inflated size
DEFAULT_MAX_MEMBERS = 100_000
# Ratio fires only above a floor of compressed bytes — tiny files (an
# empty .txt compresses "infinitely") must not trip it.
DEFAULT_MAX_RATIO = 200.0
_RATIO_FLOOR_BYTES = 1024 * 1024  # ratio applies above 1 MiB compressed


class UnsafeArchiveError(ValueError):
    """Raised when an archive's declared contents exceed safety ceilings."""


def ensure_zip_sane(
    zf: zipfile.ZipFile,
    *,
    max_uncompressed: int = DEFAULT_MAX_UNCOMPRESSED,
    max_members: int = DEFAULT_MAX_MEMBERS,
    max_ratio: float = DEFAULT_MAX_RATIO,
    source: str = "",
) -> None:
    """Reject decompression bombs from the central directory, pre-extract.

    Checks (all against DECLARED sizes — no decompression happens):
    * member count ceiling
    * total uncompressed size ceiling
    * compression ratio ceiling (above a compressed-bytes floor)

    Raises :class:`UnsafeArchiveError` with an operator-readable reason.
    Note: a lying central directory (declared size < real inflated size)
    is caught by ``zipfile`` itself at read time, which stops inflating
    at the declared size — so declared-size math is sufficient here.
    """
    infos = zf.infolist()
    if len(infos) > max_members:
        raise UnsafeArchiveError(
            f"archive has {len(infos)} members (limit {max_members})"
        )
    total_uncompressed = sum(i.file_size for i in infos)
    if total_uncompressed > max_uncompressed:
        raise UnsafeArchiveError(
            f"archive inflates to {total_uncompressed / 1024**2:.0f} MiB "
            f"(limit {max_uncompressed / 1024**2:.0f} MiB)"
        )
    total_compressed = sum(i.compress_size for i in infos)
    if (
        total_compressed > _RATIO_FLOOR_BYTES
        and total_uncompressed / max(1, total_compressed) > max_ratio
    ):
        raise UnsafeArchiveError(
            f"archive compression ratio "
            f"{total_uncompressed / max(1, total_compressed):.0f}:1 "
            f"exceeds {max_ratio:.0f}:1 — decompression-bomb shaped"
        )
    if source:
        log.debug(
            "archive_sanity_ok", source=source, members=len(infos),
            uncompressed=total_uncompressed,
        )


# Magic prefixes for the formats this app ingests by extension. Order
# matters only for prefix-of-prefix cases (none currently).
_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"PK\x03\x04", "zip"),      # also epub/cbz/docx/xlsx — zip containers
    (b"PK\x05\x06", "zip"),      # empty zip
    (b"%PDF", "pdf"),
    (b"\x1f\x8b", "gzip"),
    (b"7z\xbc\xaf\x27\x1c", "7z"),
    (b"Rar!\x1a\x07", "rar"),
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"\xff\xd8\xff", "jpeg"),
    (b"GIF8", "gif"),
    (b"SQLite format 3\x00", "sqlite"),
    (b"MZ", "pe-executable"),
    (b"\x7fELF", "elf-executable"),
)


def sniff_kind(data_or_path: bytes | str | Path) -> str:
    """Best-effort container format from magic bytes.

    Returns one of the ``_MAGIC`` kinds, ``"tar"`` (ustar probe at offset
    257), or ``""`` when unrecognized (plain text and unknown binaries).
    Never raises on unreadable paths — returns ``""``.
    """
    if isinstance(data_or_path, str | Path):
        try:
            with open(data_or_path, "rb") as fh:
                head = fh.read(512)
        except OSError:
            return ""
    else:
        head = bytes(data_or_path[:512])
    for magic, kind in _MAGIC:
        if head.startswith(magic):
            return kind
    if len(head) >= 262 and head[257:262] == b"ustar":
        return "tar"
    return ""


def ensure_looks_like_zip(data_or_path: bytes | str | Path, *, label: str = "file") -> None:
    """Raise :class:`UnsafeArchiveError` unless the content IS a zip.

    For import paths that route on file extension — the extension says
    what the USER claims, the magic bytes say what the file IS.
    """
    kind = sniff_kind(data_or_path)
    if kind != "zip":
        raise UnsafeArchiveError(
            f"{label} does not contain zip data "
            f"(detected: {kind or 'unknown'})"
        )
