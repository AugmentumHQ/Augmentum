"""Tests for the BIOS catalog and the bulk-import file classifier.

Both modules are pure (no I/O, no DB), so these are fast unit tests
that run as part of the standard suite.

Coverage:
  * catalog: every entry has required keys, no duplicate (sys, name)
    pairs, hash lookups round-trip, name+size matching is strict
  * classifier: known ROM extension -> rom, BIOS hash -> bios,
    .zip -> archive, .DS_Store -> junk, unknown .bin -> unknown,
    BIOS-name-but-wrong-size -> unknown with low confidence,
    SHA1 hit beats extension match, junk extension after ROM check
"""

from __future__ import annotations

from augmentum.titles import bios_catalog
from augmentum.titles.file_classifier import classify

# ── Catalog ──────────────────────────────────────────────────────────


def test_catalog_has_entries() -> None:
    files = bios_catalog.all_files()
    assert len(files) > 30, "EmuDeck-grade catalog should have 30+ entries"


def test_catalog_required_systems_covered() -> None:
    """Every system in rom_systems with bios_required=True must have
    at least one non-optional catalog entry. Missing this is a
    correctness bug -- launch will block on something the catalog
    can't satisfy."""
    from augmentum.titles.rom_systems import list_systems

    bios_required_systems = [s.id for s in list_systems() if s.bios_required]
    for sys_id in bios_required_systems:
        required = bios_catalog.required_for_system(sys_id)
        assert required, (
            f"system {sys_id!r} has bios_required=True but no required "
            "BIOS files in the catalog"
        )


def test_catalog_no_duplicate_canonical_pairs() -> None:
    """A given (system_id, filename) pair must be unique in the
    catalog -- a duplicate would create ambiguity in the BIOS panel
    and the install path."""
    seen: set[tuple[str, str]] = set()
    for f in bios_catalog.all_files():
        key = (f.system_id, f.filename.lower())
        # Aliases are fine if they share the same hash; we only flag
        # a true duplicate when both system + name AND hash collide.
        # Here we require system+name uniqueness, which is the
        # stronger invariant we actually rely on.
        assert key not in seen, (
            f"duplicate catalog entry: {f.system_id}/{f.filename}"
        )
        seen.add(key)


def test_lookup_by_sha1_psx() -> None:
    # scph5500.bin -- well-known canonical SHA1
    f = bios_catalog.lookup_by_sha1(
        "b05def971d8ec59f346f2d9ac21fb742e3eb6917"
    )
    assert f is not None
    assert f.system_id == "psx"
    assert f.filename == "scph5500.bin"


def test_lookup_by_sha1_case_insensitive() -> None:
    f = bios_catalog.lookup_by_sha1(
        "B05DEF971D8EC59F346F2D9AC21FB742E3EB6917"
    )
    assert f is not None
    assert f.system_id == "psx"


def test_lookup_by_sha1_miss() -> None:
    assert bios_catalog.lookup_by_sha1("0" * 40) is None
    assert bios_catalog.lookup_by_sha1("") is None


def test_lookup_by_name_size_strict() -> None:
    # scph5500.bin at the right size matches.
    f = bios_catalog.lookup_by_name_size("scph5500.bin", 524288)
    assert f is not None and f.system_id == "psx"

    # Wrong size at the right name returns None -- we'd rather flag
    # the user's dump than misclassify.
    assert bios_catalog.lookup_by_name_size("scph5500.bin", 100) is None


def test_lookup_loose_by_name() -> None:
    """Loose name lookup returns every catalog entry sharing the
    canonical filename. Used by the classifier to surface 'this
    looks like a BIOS but the size is off' hints."""
    matches = bios_catalog.lookup_loose_by_name("scph5500.bin")
    assert any(m.system_id == "psx" for m in matches)


# ── Classifier ──────────────────────────────────────────────────────


def test_classify_known_rom_extension() -> None:
    v = classify("Tetris.gb", size_bytes=32768)
    assert v.kind == "rom"
    assert v.system is not None
    assert v.system.id == "gb"


def test_classify_known_rom_with_path_prefix() -> None:
    """webkitRelativePath gives names like 'Roms/GB/Tetris.gb'.
    The classifier normalises and matches on the basename."""
    v = classify("Roms/GB/Tetris.gb", size_bytes=32768)
    assert v.kind == "rom"
    assert v.system is not None
    assert v.system.id == "gb"


def test_classify_bios_by_sha1_beats_extension_guess() -> None:
    """A .bin file with a known BIOS hash should classify as BIOS,
    not get sent through the ROM detection path. Hash wins."""
    v = classify(
        "scph5500.bin",
        sha1="b05def971d8ec59f346f2d9ac21fb742e3eb6917",
        size_bytes=524288,
    )
    assert v.kind == "bios"
    assert v.bios_file is not None
    assert v.bios_file.system_id == "psx"
    assert v.confidence == "high"


def test_classify_bios_by_name_size() -> None:
    """No SHA1 catalogued, but the canonical name + size matches.
    Lower-tier confidence but still a BIOS verdict."""
    # ps2-0200a-20040614.bin has SHA1=None in the catalog (we don't
    # have a high-confidence hash); falls back to (name, size).
    v = classify(
        "ps2-0200a-20040614.bin",
        size_bytes=4194304,
    )
    assert v.kind == "bios"
    assert v.bios_file is not None
    assert v.bios_file.system_id == "ps2"


def test_classify_bios_wrong_size_falls_to_unknown() -> None:
    """Right BIOS name but wrong size = 'unknown' with the loose-
    name hint. We never miscategorise as BIOS when the bytes don't
    line up."""
    v = classify("scph5500.bin", size_bytes=100)
    assert v.kind == "unknown"
    assert "scph5500.bin" in v.reason
    assert v.confidence == "low"


def test_classify_archive_zip() -> None:
    v = classify("BIOS_pack.zip", size_bytes=1024 * 1024)
    assert v.kind == "archive"
    assert v.archive_format == "zip"


def test_classify_archive_7z_surfaces_for_caller() -> None:
    """7z support isn't shipped; classifier still returns 'archive'
    so the bulk-import endpoint can decide what to do (currently
    surfaces as unknown for manual extraction)."""
    v = classify("pack.7z", size_bytes=1024 * 1024)
    assert v.kind == "archive"
    assert v.archive_format == "7z"


def test_classify_junk_filenames() -> None:
    for name in ("Thumbs.db", "desktop.ini", "cover.jpg"):
        v = classify(name, size_bytes=1024)
        assert v.kind == "junk", f"{name} should be junk, got {v.kind}"


def test_classify_junk_extensions() -> None:
    for name in ("readme.txt", "notes.nfo", "screenshot.png"):
        v = classify(name, size_bytes=1024)
        assert v.kind == "junk", f"{name} should be junk, got {v.kind}"


def test_classify_unknown_bin_no_match() -> None:
    """A .bin file with no recognised name and no BIOS hash falls
    through to unknown -- the bulk-import path surfaces it for user
    review with override candidates."""
    v = classify("random_data.bin", size_bytes=1024)
    assert v.kind == "unknown"


def test_classify_empty_filename_is_unknown() -> None:
    v = classify("", size_bytes=0)
    assert v.kind == "unknown"


def test_required_for_system_excludes_optional() -> None:
    """required_for_system used by the launch path should never
    include optional entries -- launch shouldn't block on optional
    BIOS like gba_bios.bin."""
    psx_required = bios_catalog.required_for_system("psx")
    assert psx_required, "PSX must have at least one required BIOS"
    for f in psx_required:
        assert not f.optional


def test_systems_with_bios_includes_known_systems() -> None:
    systems = set(bios_catalog.systems_with_bios())
    # Spot-check the painful ones the user actually has trouble with.
    for required in ("psx", "ps2", "saturn"):
        assert required in systems
