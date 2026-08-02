"""Tests for store-first BIOS handling and the libretro hash database.

Regression origin (2026-07-25, observed in the live container):

    bulk_import_complete  bios=0 imported=0 junk=2  unknown=38
    bulk_import_complete  bios=0 imported=0 junk=32 unknown=0

38 BIOS files uploaded, zero installed. Identification was GATING
admission, and identification was a 67-entry hand-maintained table
with 15 hashes that demanded an exact canonical filename plus an exact
byte size. RetroArch, EmuDeck and ES-DE all store first and verify
second; these tests pin that behaviour so the gate cannot come back.

Coverage:
  * hash database parses, indexes, and identifies by sha1/md5/crc32
  * files that the old strict path rejected now identify
  * a near-miss explains itself instead of saying 'unrecognised'
  * verify_status is derived honestly from match provenance
  * the store surfaces files that occupy no catalog slot
  * slot-name validation refuses traversal but allows odd real names
"""

from __future__ import annotations

import hashlib
import zlib

import pytest

from augmentum.titles import bios_hashdb
from augmentum.titles.bios_store import _VERIFY_BY_MATCH
from augmentum.titles.file_classifier import classify

# ── Hash database ────────────────────────────────────────────────────


def test_hashdb_loads_a_meaningful_corpus() -> None:
    """The whole point of vendoring libretro's System.dat is breadth.
    The hand-maintained catalog carried 15 hashes; if this collapses
    back to that order of magnitude the DAT failed to parse."""
    s = bios_hashdb.stats()
    assert s["entries"] > 300, f"hash db looks unparsed: {s}"
    assert s["systems"] > 30, f"too few platforms: {s}"
    assert s["with_sha1"] > 300
    assert s["with_md5"] > 300
    assert s["with_crc32"] > 300


def test_hashdb_identifies_by_every_digest() -> None:
    """A known dump must be findable by any of the three digests --
    that redundancy is what lets a file installed today get verified
    tomorrow after a database refresh."""
    psx = next(
        e for e in bios_hashdb.known_for_system("psx")
        if e.basename == "scph5500.bin"
    )
    for kwargs, expect in (
        ({"sha1": psx.sha1}, "sha1"),
        ({"md5": psx.md5}, "md5"),
        ({"crc32": psx.crc32}, "crc32"),
        ({"filename": psx.basename, "size_bytes": psx.size_bytes}, "name_size"),
    ):
        hit, matched_by = bios_hashdb.identify(**kwargs)
        assert hit is not None, f"no hit for {kwargs}"
        assert hit.basename == "scph5500.bin"
        assert matched_by == expect


def test_hashdb_hash_beats_a_misleading_filename() -> None:
    """A renamed dump is the common real-world case. Identification
    must follow the bytes, not the name the user happened to give it."""
    psx = next(
        e for e in bios_hashdb.known_for_system("psx")
        if e.basename == "scph5500.bin"
    )
    hit, matched_by = bios_hashdb.identify(
        sha1=psx.sha1, filename="my_totally_wrong_name.rom", size_bytes=1,
    )
    assert matched_by == "sha1"
    assert hit is not None and hit.system_id == "psx"


def test_hashdb_system_aliases_resolve_both_ways() -> None:
    """Our own catalog carries both 'dc' and 'dreamcast'. Neither was
    renamed (that would strand existing rows), so both must reach the
    same hash-database entries."""
    assert set(bios_hashdb.aliases_for("dc")) == {"dc", "dreamcast"}
    assert bios_hashdb.known_for_system("dc")
    assert (
        {e.basename for e in bios_hashdb.known_for_system("dc")}
        == {e.basename for e in bios_hashdb.known_for_system("dreamcast")}
    )


def test_hashdb_missing_file_degrades_quietly(monkeypatch) -> None:
    """A missing DAT must not take the vault down. Store-first does not
    depend on identification, so the failure mode is 'nothing is
    verified', never 'nothing can be installed'."""
    from pathlib import Path

    bios_hashdb._index.cache_clear()
    monkeypatch.setattr(bios_hashdb, "_DAT_PATH", Path("/nonexistent/x.dat"))
    try:
        assert bios_hashdb.stats()["entries"] == 0
        assert bios_hashdb.identify(sha1="deadbeef") == (None, "")
    finally:
        bios_hashdb._index.cache_clear()


# ── Classifier: files the old gate threw away ────────────────────────


@pytest.mark.parametrize(
    ("filename", "size", "system"),
    [
        ("disksys.rom", 8192, "fds"),          # was: unknown -> discarded
        ("scph5501.bin", 524288, "psx"),
        ("gba_bios.bin", 16384, "gba"),
        ("bios_CD_U.bin", 131072, "segacd"),
        ("dc_boot.bin", 2097152, "dreamcast"),
        ("syscard3.pce", 262144, "pcecd"),
        ("lynxboot.img", 512, "lynx"),
    ],
)
def test_real_bios_names_are_identified(filename, size, system) -> None:
    verdict = classify(filename, size_bytes=size)
    assert verdict.kind == "bios", f"{filename} -> {verdict.kind}: {verdict.reason}"
    assert verdict.bios_file is not None
    assert verdict.bios_file.system_id == system
    assert verdict.matched_by, "a BIOS verdict must record how it matched"


def test_near_miss_explains_the_size_difference() -> None:
    """One byte off a known dump used to produce a bare 'unrecognised'.
    The user needs to know it was close and why, because store-first
    will install it anyway on their say-so."""
    verdict = classify("scph5501.bin", size_bytes=524287)
    assert verdict.kind == "unknown"
    assert "524287" in verdict.reason
    assert "524288" in verdict.reason
    assert "psx" in verdict.reason


def test_verify_status_mapping_is_honest() -> None:
    """Cryptographic digests prove the bytes; a name+size agreement
    proves only the slot; a user assertion proves nothing. Conflating
    them is how a vault ends up claiming it verified something it
    never hashed."""
    assert _VERIFY_BY_MATCH["sha1"] == "verified"
    assert _VERIFY_BY_MATCH["md5"] == "verified"
    assert _VERIFY_BY_MATCH["crc32"] == "verified"
    assert _VERIFY_BY_MATCH["name_size"] == "named"
    assert _VERIFY_BY_MATCH["user_asserted"] == "unverified"


def test_classifier_accepts_all_three_digests() -> None:
    """The route computes md5/crc32 alongside sha1 and hands them
    down; if classify() ignored them the extra work would be silently
    wasted and md5-only entries would never match."""
    data = b"\x00" * 64
    verdict = classify(
        "whatever.bin",
        sha1=hashlib.sha1(data).hexdigest(),
        md5=hashlib.md5(data).hexdigest(),
        crc32=format(zlib.crc32(data) & 0xFFFFFFFF, "08x"),
        size_bytes=len(data),
    )
    # Not a real BIOS -- the assertion is that it parsed the kwargs
    # rather than raising a TypeError.
    assert verdict.kind in {"unknown", "junk", "bios", "rom"}


# ── Slot-name validation ─────────────────────────────────────────────


@pytest.mark.parametrize("name", [
    "scph5501.bin",
    "PS2 BIOS (Europe) v2.20.bin",   # spaces + parens are common
    "bios-ünïcode.rom",
    "a" * 255,
])
def test_safe_slot_names_are_allowed(name) -> None:
    """Store-first means the user names the file. Real BIOS sets are
    full of spaces, parens and unicode; refusing them would recreate
    the bug at a different layer."""
    from augmentum.proxy.titles_bios_routes import _is_safe_slot_name
    assert _is_safe_slot_name(name), name


@pytest.mark.parametrize("name", [
    "", ".", "..",
    "../../etc/passwd",
    "sub/dir.bin",
    "back\\slash.bin",
    "nul\x00byte.bin",
    "bell\x07.bin",
    "a" * 256,
])
def test_unsafe_slot_names_are_refused(name) -> None:
    """The name becomes a path segment on the serve route, so anything
    that could traverse or split a path has to go."""
    from augmentum.proxy.titles_bios_routes import _is_safe_slot_name
    assert not _is_safe_slot_name(name), name


# ── Store: extras must be visible ────────────────────────────────────


def test_status_entry_reports_extras() -> None:
    """A stored file with no catalog slot has to render in the vault.
    If it doesn't, it is on disk, holding a blob refcount, and being
    served to the emulator, while the panel that manages it shows
    nothing -- indistinguishable from a failed upload."""
    from augmentum.titles.bios_store import BiosStatusEntry

    e = BiosStatusEntry(
        system_id="ps2",
        canonical_filename="weird_regional_dump.bin",
        description="Stored by you",
        optional=True,
        present=True,
        matched_by="user_asserted",
        installed_filename="weird_regional_dump.bin",
        verify_status="unverified",
        size_bytes=4194304,
        is_extra=True,
    )
    d = e.to_dict()
    assert d["is_extra"] is True
    assert d["verify_status"] == "unverified"
    assert d["optional"] is True, "an extra must never gate a launch"
    assert d["size_bytes"] == 4194304
