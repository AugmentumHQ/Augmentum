"""File classifier for the bulk-import flow.

Given a single file's identity (name + size + hashes + a small
header sample), decide what it is and where it should go:

    - rom(system)         -> InternalRomSource pipeline
    - bios(system, name)  -> BiosStore.install
    - archive(format)     -> caller extracts + recurses
    - junk(reason)        -> silently skip with a count
    - unknown             -> surface to user with override picker

Hash matches always win over filename guesses. The classifier is
pure and synchronous: no I/O, no DB access. The caller is
responsible for hashing the bytes (so the same hash can be reused
for the dedupe pre-flight) and for recursing into archives.

Adding a new junk extension or archive format is a single line in
the constants below; adding a new ROM extension is one line in
``rom_systems.py``; adding a new BIOS file is one line in
``bios_catalog.py``. The classifier glues them together without
embedding policy of its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from augmentum.titles import bios_hashdb
from augmentum.titles.bios_catalog import (
    BiosFile,
    lookup_by_name_size,
    lookup_by_pattern,
    lookup_by_sha1,
    lookup_loose_by_name,
)
from augmentum.titles.rom_systems import SystemSpec, detect_system

# ── Junk filter ──────────────────────────────────────────────────────


# Extensions we always discard. These are companion files that ship
# alongside ROMs and BIOS files but aren't themselves usable. We
# treat them as junk so a "drag your whole ROMs folder onto the page"
# UX doesn't surface 50 files for review every time.
_JUNK_EXTENSIONS: frozenset[str] = frozenset({
    # Documentation / metadata
    ".txt", ".nfo", ".diz", ".readme", ".md",
    # OS detritus
    ".ds_store", ".thumbs", ".desktop", ".lnk",
    # Save data (orphaned -- belongs in the SaveStore, not here)
    ".sav", ".srm", ".state", ".st0", ".st1", ".st2", ".st3",
    ".st4", ".st5", ".st6", ".st7", ".st8", ".st9", ".sgm",
    # Image companions (cover art lives in metadata.thumbnail_url,
    # not as a sibling file in the ROMs folder)
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".ico",
    # Cue/sub sidecars without their .bin/.iso are useless on their own.
    # We DON'T junk .cue here (it's the canonical entry for PSX
    # multi-bin sets) -- the classifier surfaces .bin separately if
    # the user dropped only the .bin without the .cue.
    ".sub", ".ccd", ".sbi", ".m3u",
    # Misc
    ".bak", ".tmp", ".part", ".log", ".xml", ".dat",
})


# Specific filenames that are always junk regardless of extension.
_JUNK_FILENAMES: frozenset[str] = frozenset({
    "thumbs.db", "desktop.ini", ".ds_store",
    "folder.jpg", "folder.png", "albumartsmall.jpg",
    "cover.jpg", "cover.png", "cover.webp",
    "screenshot.png", "icon.png", "preview.png",
})


# ── Archives ─────────────────────────────────────────────────────────


# Extensions that are containers we recurse into. .zip is the only
# format we extract today (stdlib zipfile, no new dep). .7z support
# would need py7zr; we surface .7z as 'archive' so the UI can show
# "we can't open this yet -- extract it manually" rather than
# silently dropping it.
#
# IMPORTANT: arcade ROMs are .zip and MUST stay zipped (FBNeo / MAME
# read the zip directly). The bulk-import flow checks the zip
# contents BEFORE deciding to extract: if every member looks like a
# ROM romset entry (no BIOS hashes match, no top-level recognisable
# files), we treat it as opaque arcade.zip. The pure classifier
# returns 'archive(zip)' and the caller does the contents-aware
# decision -- see bulk_import_routes.
_ARCHIVE_FORMATS: dict[str, str] = {
    ".zip": "zip",
    ".7z": "7z",
    ".tar": "tar",
    ".tar.gz": "tar.gz",
    ".tgz": "tar.gz",
    ".rar": "rar",
}


# ── Result type ──────────────────────────────────────────────────────


ClassificationKind = Literal[
    "rom", "bios", "archive", "junk", "unknown",
]


@dataclass(frozen=True)
class Classification:
    """The classifier's verdict for one file.

    Exactly one of the system/bios_file/archive_format/junk_reason
    fields is populated based on ``kind``. Callers should switch on
    ``kind`` first; the typed accessors are convenience.
    """
    kind: ClassificationKind
    confidence: Literal["high", "medium", "low"]
    # Populated for kind='rom'
    system: SystemSpec | None = None
    # Populated for kind='bios'
    bios_file: BiosFile | None = None
    # Populated for kind='archive'
    archive_format: str = ""
    # Populated for kind='junk'
    junk_reason: str = ""
    # Always populated -- explanation text for the UI
    reason: str = ""
    # Populated for kind='bios': how the identification was made.
    # 'sha1'|'md5'|'crc32' are cryptographic (verified); 'name_size'
    # and 'pattern' identify the slot but not the bytes.
    matched_by: str = ""


# ── Public API ───────────────────────────────────────────────────────


def _known_to_bios_file(known: bios_hashdb.KnownBios) -> BiosFile:
    """Adapt a hash-database hit to the BiosFile shape the install
    path already speaks.

    ``optional`` is True because the hash database is a catalogue of
    what EXISTS, not of what a given core REQUIRES to boot -- libretro
    publishes required-vs-optional per core in the ``.info`` files,
    not in System.dat. Our own ``bios_catalog`` remains the authority
    on which slots gate a launch; treating an identified-but-
    uncatalogued file as required would block launches on firmware
    nothing actually needs.
    """
    return BiosFile(
        system_id=known.system_id,
        filename=known.basename,
        size_bytes=known.size_bytes,
        sha1=known.sha1 or None,
        md5=known.md5 or None,
        optional=True,
        description=f"{known.platform} — libretro BIOS database",
    )


def classify(
    filename: str,
    *,
    sha1: str = "",
    sha256: str = "",
    md5: str = "",
    crc32: str = "",
    size_bytes: int = 0,
    header: bytes | None = None,
) -> Classification:
    """Classify a single file. Pure: no I/O, no DB access.

    Order of operations (first match wins):
      0. any known hash / name+size → libretro BIOS database
      1. SHA1 → BIOS catalog (definitive identification)
      2. (filename, size) → BIOS catalog (canonical name match)
      3. junk filenames (Thumbs.db, cover.jpg, ...)
      4. archive extension (.zip / .7z / ...)
      5. ROM extension / header magic bytes
      6. junk extensions (.txt / .nfo / ...)
      7. loose BIOS filename hint (right name, wrong size)
      8. unknown

    Why archive beats ROM: ``.zip`` is registered as the arcade ROM
    extension, but most ``.zip`` drops are BIOS packs or random
    archives, NOT arcade romsets. The bulk-import endpoint inspects
    the zip's contents and re-classifies as arcade-rom when the
    shape matches (multiple .bin/.rom parts at the top level). The
    classifier itself stays I/O-free.

    Hash matching wins because BIOS files often have generic names
    like ``bios.bin`` or ``rom1.bin`` that collide with ROM
    extensions. A SHA1 hit identifies the file regardless of name.
    """
    name = (filename or "").strip()
    name_lower = name.lower()
    base_name = name_lower.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]

    # 0. libretro known-BIOS database (the broad net: ~500 entries
    # across 60+ platforms, each with CRC32 + MD5 + SHA1). Runs first
    # because it is both the widest and the strongest identifier --
    # the hand-maintained catalog below only carries slot policy
    # (required vs optional) and a handful of hashes.
    known, matched_by = bios_hashdb.identify(
        sha1=sha1, md5=md5, crc32=crc32,
        filename=base_name, size_bytes=size_bytes,
    )
    if known is not None:
        return Classification(
            kind="bios",
            confidence="high" if matched_by != "name_size" else "medium",
            bios_file=_known_to_bios_file(known),
            matched_by=matched_by,
            reason=(
                f"{matched_by.upper()} matches {known.platform} "
                f"{known.basename} in the libretro BIOS database"
            ),
        )

    # 1. SHA1 BIOS hit against our own catalog (covers slots libretro
    # doesn't publish -- PS3/Switch/Wii U firmware and similar).
    if sha1:
        bios = lookup_by_sha1(sha1)
        if bios is not None:
            return Classification(
                kind="bios",
                confidence="high",
                bios_file=bios,
                matched_by="sha1",
                reason=f"SHA1 matches {bios.system_id}/{bios.filename}",
            )

    # 2. (canonical filename + size) BIOS hit
    if base_name and size_bytes > 0:
        bios = lookup_by_name_size(base_name, size_bytes)
        if bios is not None:
            return Classification(
                kind="bios",
                confidence="high",
                bios_file=bios,
                matched_by="name_size",
                reason=(
                    f"Filename '{base_name}' + size {size_bytes} match "
                    f"{bios.system_id}/{bios.filename}"
                ),
            )

    # 2b. Manufacturer-naming pattern hit. Maps Sony's SCPH-XXXXX,
    # Atari's lynxboot, Commodore's kickstart-X.X, etc. onto a
    # canonical catalog entry when the size matches. This is what
    # makes a raw ``scph39001.bin`` dump get recognised as a PS2
    # BIOS even though the catalog stores it under
    # ``ps2-0200a-20040614.bin``. Runs after exact name/size match
    # so a perfectly-named drop never hits the pattern fallback.
    if base_name and size_bytes > 0:
        bios = lookup_by_pattern(base_name, size_bytes)
        if bios is not None:
            return Classification(
                kind="bios",
                confidence="high",
                bios_file=bios,
                matched_by="name_size",
                reason=(
                    f"Filename '{base_name}' matches manufacturer "
                    f"naming pattern for {bios.system_id} (size "
                    f"{size_bytes} matches {bios.filename})"
                ),
            )

    # 3. Hard junk filenames
    if base_name in _JUNK_FILENAMES:
        return Classification(
            kind="junk", confidence="high",
            junk_reason="known_companion_file",
            reason=f"'{base_name}' is a known non-ROM companion file",
        )

    # 4. Archive (precedes ROM detection because .zip is registered
    # for arcade but most .zip drops are BIOS packs or random
    # archives. The bulk-import endpoint inspects the contents and
    # promotes to arcade-rom when the shape matches.)
    archive_fmt = _archive_format_for(base_name)
    if archive_fmt:
        return Classification(
            kind="archive",
            confidence="high",
            archive_format=archive_fmt,
            reason=f"{archive_fmt} archive (members will be reclassified)",
        )

    # 5. ROM extension / header magic
    rom_spec = detect_system(name, header=header)
    if rom_spec is not None:
        return Classification(
            kind="rom",
            confidence="high",
            system=rom_spec,
            reason=(
                f"Extension matches {rom_spec.id} ({rom_spec.label})"
                if not header else
                f"Header bytes match {rom_spec.id} ({rom_spec.label})"
            ),
        )

    # 6. Junk extensions (after ROM/archive checks so .zip arcade
    # ROMs and recognized extensions never get junked)
    for ext in _JUNK_EXTENSIONS:
        if base_name.endswith(ext):
            return Classification(
                kind="junk", confidence="high",
                junk_reason=f"junk_extension:{ext}",
                reason=f"'{ext}' files are companion data, not ROMs",
            )

    # 7. Loose BIOS name hint (canonical name but wrong size). Checked
    # against both the hash database and our catalog, so a near-miss
    # gets a specific explanation ("expected 524288, got 524287")
    # instead of a bare 'unrecognised'. Still 'unknown' -- but the
    # store-first path installs it anyway when the user dropped it on
    # a system row, and this text is what the vault shows next to it.
    if base_name:
        db_near = bios_hashdb.candidates_for_name(base_name)
        if db_near:
            systems = sorted({e.system_id for e in db_near})
            expected = sorted({e.size_bytes for e in db_near})
            return Classification(
                kind="unknown",
                confidence="low",
                reason=(
                    f"'{base_name}' is a known BIOS name for "
                    f"{', '.join(systems)}, but this file is "
                    f"{size_bytes} bytes and the known dumps are "
                    f"{expected}. Likely a different revision or a bad "
                    "dump — install it anyway if you trust it."
                ),
            )
        loose = lookup_loose_by_name(base_name)
        if loose:
            systems = sorted({f.system_id for f in loose})
            expected_sizes = sorted({f.size_bytes for f in loose if f.size_bytes})
            return Classification(
                kind="unknown",
                confidence="low",
                reason=(
                    f"Filename '{base_name}' is a known BIOS for "
                    f"{', '.join(systems)} but the size {size_bytes} "
                    f"doesn't match expected {expected_sizes}. Possibly "
                    "a corrupt or wrong-region dump."
                ),
            )

    # 8. Genuinely unknown
    return Classification(
        kind="unknown",
        confidence="low",
        reason="No known signature, extension, or BIOS hash matched",
    )


def _archive_format_for(filename: str) -> str:
    """Return the archive format string ('zip', '7z', 'tar.gz', ...)
    or empty string if not an archive. Multi-extension formats like
    .tar.gz are checked first so .gz alone doesn't false-match."""
    n = filename.lower()
    # Longest extensions first
    for ext in sorted(_ARCHIVE_FORMATS, key=len, reverse=True):
        if n.endswith(ext):
            return _ARCHIVE_FORMATS[ext]
    return ""
