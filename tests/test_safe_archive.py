"""Decompression-bomb guard + magic sniffing (utils/safe_archive.py)."""

from __future__ import annotations

import io
import zipfile

import pytest

from augmentum.utils.safe_archive import (
    UnsafeArchiveError,
    ensure_looks_like_zip,
    ensure_zip_sane,
    sniff_kind,
)


def _zip_of(entries: dict[str, bytes]) -> zipfile.ZipFile:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    buf.seek(0)
    return zipfile.ZipFile(buf)


def test_normal_zip_passes():
    zf = _zip_of({"a.txt": b"hello", "b/c.txt": b"world" * 100})
    ensure_zip_sane(zf)  # no raise


def test_member_count_ceiling():
    zf = _zip_of({f"f{i}": b"x" for i in range(50)})
    with pytest.raises(UnsafeArchiveError, match="members"):
        ensure_zip_sane(zf, max_members=10)


def test_total_uncompressed_ceiling():
    zf = _zip_of({"big.bin": b"\0" * 2_000_000})
    with pytest.raises(UnsafeArchiveError, match="inflates"):
        ensure_zip_sane(zf, max_uncompressed=1_000_000)


def test_bomb_ratio_rejected():
    # ensure_zip_sane reads only declared sizes from infolist() — feed it
    # a bomb-shaped central directory directly rather than building a
    # multi-GiB archive in RAM.
    from types import SimpleNamespace

    bomb_member = SimpleNamespace(
        file_size=10 * 1024**3,       # declares 10 GiB
        compress_size=2 * 1024**2,    # from 2 MiB — 5120:1
    )
    fake_zf = SimpleNamespace(infolist=lambda: [bomb_member])
    with pytest.raises(UnsafeArchiveError, match="ratio"):
        ensure_zip_sane(fake_zf, max_uncompressed=1024**4)


def test_ratio_floor_spares_tiny_files():
    # An empty-ish file has an extreme ratio but is under the floor.
    zf = _zip_of({"t.txt": b"\0" * 100_000})
    ensure_zip_sane(zf)  # no raise


def test_sniff_zip_and_pdf_and_unknown():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("a", b"x")
    assert sniff_kind(buf.getvalue()) == "zip"
    assert sniff_kind(b"%PDF-1.7 ...") == "pdf"
    assert sniff_kind(b"just some text") == ""
    assert sniff_kind(b"MZ\x90\x00") == "pe-executable"


def test_ensure_looks_like_zip():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("a", b"x")
    ensure_looks_like_zip(buf.getvalue())  # no raise
    with pytest.raises(UnsafeArchiveError, match="does not contain zip"):
        ensure_looks_like_zip(b"plain text", label="upload")


def test_sniff_unreadable_path_returns_empty(tmp_path):
    assert sniff_kind(tmp_path / "missing.bin") == ""
