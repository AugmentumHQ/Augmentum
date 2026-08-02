"""Known-BIOS hash database, sourced from libretro-database.

This is the *identification* layer of BIOS handling. It answers one
question: "I have these bytes -- what BIOS is this, if any?"

It deliberately does NOT decide whether a file may be installed. That
separation is the whole point, and it is how every mature emulation
frontend works:

    RetroArch  -- the System directory is a plain folder. You drop
                  files in it. ``System.dat`` exists so the Core
                  Information screen can render a checklist; it never
                  gates what lands on disk.
    EmuDeck    -- ``~/emulation/bios`` is a flat folder, files are
                  copied in unconditionally, and the "BIOS Checker" is
                  a separate advisory tool run afterwards.
    ES-DE      -- same shape: a BIOS checker that reports, not a
                  gatekeeper that refuses.

The industry converged on store-first / verify-second because BIOS
dumps are messy in the real world: regional revisions, redumps that
differ by a byte, community renames, and firmware that simply has no
canonical hash. A frontend that only accepts hash-perfect files
rejects most of what real users actually own.

Data source
-----------
``data/libretro_system.dat`` is vendored verbatim from
libretro-database (``dat/System.dat``), which is released under
CC-BY-SA-4.0. It is a ClrMamePro-format DAT: a header block, then one
``game`` block whose ``rom`` entries are grouped by ``comment`` lines
naming the platform::

    comment "Sony - PlayStation"
    rom ( name scph5500.bin size 524288 crc 8C93A399 md5 8dd7d5296a650fac7319bce665a6a53c sha1 b05def971d8ec59f346f2d9ac21fb742e3eb6917 )

Every entry carries size + CRC32 + MD5 + SHA1, so we can identify a
dump by any of them. That is a meaningful upgrade over hand-maintained
hashes: the DAT covers hundreds of entries across 60+ platforms and is
kept current upstream.

Refreshing
----------
``python scripts/refresh_bios_hashdb.py`` re-downloads the DAT and
reports what changed. Attribution must stay in the vendored file.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_DAT_PATH = Path(__file__).parent / "data" / "libretro_system.dat"


# ── Platform mapping ─────────────────────────────────────────────────
#
# libretro uses No-Intro's verbose platform names; we use short system
# ids (matching ``rom_systems.SystemSpec.id`` where a runtime exists).
# Platforms absent from this map still parse -- they just get a
# derived id -- so a new upstream platform never silently drops out of
# the database. Map entries exist to align the ones we actually run.
#
# NOTE: our own BIOS catalog historically carries BOTH ``dc`` and
# ``dreamcast`` as system ids. We keep both reachable via the alias
# table below rather than renaming either, so existing installs and
# existing catalog rows keep resolving.
_PLATFORM_TO_SYSTEM: dict[str, str] = {
    "3DO Company, The - 3DO": "3do",
    "Amstrad - CPC": "amstradcpc",
    "Arcade": "arcade",
    "Atari - 400-800": "atari800",
    "Atari - 5200": "atari5200",
    "Atari - 7800": "atari7800",
    "Atari - Lynx": "lynx",
    "Atari - ST": "atarist",
    "Coleco - ColecoVision": "colecovision",
    "Commodore - Amiga": "amiga",
    "Commodore - C128": "c128",
    "Fairchild Channel F": "channelf",
    "Magnavox - Odyssey2": "odyssey2",
    "Mattel - Intellivision": "intellivision",
    "Microsoft - MSX": "msx",
    "NEC - PC Engine - TurboGrafx 16 - SuperGrafx": "pcecd",
    "NEC - PC-98": "pc98",
    "NEC - PC-8801": "pc8801",
    "NEC - PC-FX": "pcfx",
    "Nintendo - Famicom Disk System": "fds",
    "Nintendo - Gameboy": "gb",
    "Nintendo - Game Boy Advance": "gba",
    "Nintendo - Gameboy Color": "gbc",
    "Nintendo - GameCube": "gamecube",
    "Nintendo - Nintendo 64DD": "n64dd",
    "Nintendo - Nintendo DS": "nds",
    "Nintendo - Nintendo DSi": "dsi",
    "Nintendo - Nintendo Entertainment System": "nes",
    "Nintendo - Pokemon Mini": "pokemini",
    "Nintendo - Satellaview": "satellaview",
    "Nintendo - Super Game Boy": "sgb",
    "Nintendo - Super Nintendo Entertainment System": "snes",
    "Philips - CD-i": "cdi",
    "Phillips - Videopac+": "videopac",
    "Sega - Dreamcast": "dreamcast",
    "Sega - Dreamcast-based Arcade": "naomi",
    "Sega - Game Gear": "gg",
    "Sega - Master System - Mark III": "sms",
    "Sega - Mega CD - Sega CD": "segacd",
    "Sega - Mega Drive - Genesis": "genesis",
    "Sega - Saturn": "saturn",
    "Sharp - X1": "x1",
    "Sharp - X68000": "x68000",
    "Sinclair - ZX Spectrum": "zxspectrum",
    "SNK - NeoGeo CD": "neogeocd",
    "Sony - PlayStation": "psx",
    "Sony - PlayStation 2": "ps2",
    "Sony - PlayStation Portable": "psp",
}


# Systems our own catalog names differently from the mapping above.
# Both directions resolve, so a Dreamcast BIOS identified by the DAT
# as ``dreamcast`` still satisfies a catalog slot filed under ``dc``.
_SYSTEM_ALIASES: dict[str, tuple[str, ...]] = {
    "dreamcast": ("dc",),
    "dc": ("dreamcast",),
    "pcecd": ("pce",),
    "pce": ("pcecd",),
}


def aliases_for(system_id: str) -> tuple[str, ...]:
    """Every id that refers to the same physical system, including
    ``system_id`` itself. Callers use this when checking whether an
    identified BIOS satisfies a catalog slot."""
    sid = (system_id or "").strip().lower()
    if not sid:
        return ()
    return (sid, *_SYSTEM_ALIASES.get(sid, ()))


@dataclass(frozen=True)
class KnownBios:
    """One identified BIOS/firmware file from the hash database."""
    system_id: str        # our short id ('psx', 'ps2', ...)
    platform: str         # upstream verbose name, for display
    filename: str         # canonical name emulators expect
    size_bytes: int
    crc32: str            # lowercase hex, no prefix
    md5: str              # lowercase hex
    sha1: str             # lowercase hex

    @property
    def basename(self) -> str:
        """Filename without any directory prefix. Some DAT entries are
        pathed (``dc/naomi2.zip``) because the core expects a
        subdirectory; matching is done on the leaf."""
        return self.filename.rsplit("/", 1)[-1]


# ── DAT parsing ──────────────────────────────────────────────────────


# A rom line. Fields are space-delimited key/value pairs; ``name`` may
# contain spaces (and is NOT always quoted upstream), so we anchor on
# the following ``size`` key rather than greedily consuming.
_ROM_RE = re.compile(
    r"""rom\s*\(\s*
        name\s+(?P<name>.+?)\s+
        size\s+(?P<size>\d+)\s+
        (?:crc\s+(?P<crc>[0-9A-Fa-f]+)\s+)?
        (?:md5\s+(?P<md5>[0-9A-Fa-f]{32})\s+)?
        (?:sha1\s+(?P<sha1>[0-9A-Fa-f]{40})\s*)?
        \)""",
    re.VERBOSE,
)

_COMMENT_RE = re.compile(r'^\s*comment\s+"(?P<text>[^"]*)"\s*$')


def _derive_system_id(platform: str) -> str:
    """Fallback id for a platform we haven't explicitly mapped. Keeps
    new upstream platforms in the database instead of dropping them."""
    tail = platform.split(" - ")[-1]
    return re.sub(r"[^a-z0-9]+", "", tail.lower()) or "unknown"


def _parse_dat(text: str) -> list[KnownBios]:
    out: list[KnownBios] = []
    platform = ""
    for line in text.splitlines():
        m_comment = _COMMENT_RE.match(line)
        if m_comment:
            text_val = m_comment.group("text")
            # The header carries prose comments ("System, firmware,
            # and BIOS files...") and the game block opens with a
            # redundant "System". Neither is a platform heading; real
            # headings are the ones we can map or that contain a
            # vendor separator.
            if text_val and text_val != "System" and not text_val.endswith("."):
                platform = text_val
            continue
        m_rom = _ROM_RE.search(line)
        if not m_rom:
            continue
        name = m_rom.group("name").strip().strip('"')
        system_id = _PLATFORM_TO_SYSTEM.get(platform) or _derive_system_id(platform)
        out.append(KnownBios(
            system_id=system_id,
            platform=platform,
            filename=name,
            size_bytes=int(m_rom.group("size")),
            crc32=(m_rom.group("crc") or "").lower(),
            md5=(m_rom.group("md5") or "").lower(),
            sha1=(m_rom.group("sha1") or "").lower(),
        ))
    return out


@dataclass(frozen=True)
class _Index:
    entries: tuple[KnownBios, ...]
    by_sha1: dict[str, KnownBios]
    by_md5: dict[str, KnownBios]
    by_crc32: dict[str, KnownBios]
    by_name_size: dict[tuple[str, int], KnownBios]
    by_name: dict[str, tuple[KnownBios, ...]]


@lru_cache(maxsize=1)
def _index() -> _Index:
    try:
        text = _DAT_PATH.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        # A missing DAT degrades identification to the hand-maintained
        # catalog; it must never take the BIOS vault down, because
        # store-first install does not depend on identification.
        log.warning("bios_hashdb_unavailable", path=str(_DAT_PATH), error=str(exc))
        return _Index((), {}, {}, {}, {}, {})

    entries = tuple(_parse_dat(text))
    by_sha1: dict[str, KnownBios] = {}
    by_md5: dict[str, KnownBios] = {}
    by_crc32: dict[str, KnownBios] = {}
    by_name_size: dict[tuple[str, int], KnownBios] = {}
    by_name: dict[str, list[KnownBios]] = {}
    for e in entries:
        # First writer wins on collisions -- the DAT is ordered by
        # platform and earlier platforms are the more common ones.
        if e.sha1:
            by_sha1.setdefault(e.sha1, e)
        if e.md5:
            by_md5.setdefault(e.md5, e)
        if e.crc32:
            by_crc32.setdefault(e.crc32, e)
        by_name_size.setdefault((e.basename.lower(), e.size_bytes), e)
        by_name.setdefault(e.basename.lower(), []).append(e)

    log.info(
        "bios_hashdb_loaded",
        entries=len(entries),
        systems=len({e.system_id for e in entries}),
        with_sha1=len(by_sha1), with_md5=len(by_md5),
    )
    return _Index(
        entries=entries,
        by_sha1=by_sha1, by_md5=by_md5, by_crc32=by_crc32,
        by_name_size=by_name_size,
        by_name={k: tuple(v) for k, v in by_name.items()},
    )


# ── Public lookups ───────────────────────────────────────────────────


def identify(
    *,
    sha1: str = "",
    md5: str = "",
    crc32: str = "",
    filename: str = "",
    size_bytes: int = 0,
) -> tuple[KnownBios | None, str]:
    """Identify a file against the hash database.

    Returns ``(entry, matched_by)`` where ``matched_by`` is one of
    ``sha1`` / ``md5`` / ``crc32`` / ``name_size`` / ``""``.

    Hash matches are definitive and are tried strongest-first. The
    ``name_size`` fallback exists because a meaningful slice of the
    DAT's firmware entries are region/revision variants that users
    commonly hold in a byte-differing redump; a name+size agreement is
    strong enough to *suggest* an identity, and the caller records the
    weaker provenance so the UI can show it honestly.
    """
    idx = _index()
    if sha1:
        hit = idx.by_sha1.get(sha1.strip().lower())
        if hit is not None:
            return hit, "sha1"
    if md5:
        hit = idx.by_md5.get(md5.strip().lower())
        if hit is not None:
            return hit, "md5"
    if crc32:
        hit = idx.by_crc32.get(crc32.strip().lower().removeprefix("0x"))
        if hit is not None:
            return hit, "crc32"
    if filename and size_bytes > 0:
        base = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].lower()
        hit = idx.by_name_size.get((base, size_bytes))
        if hit is not None:
            return hit, "name_size"
    return None, ""


def candidates_for_name(filename: str) -> tuple[KnownBios, ...]:
    """Every known entry sharing this filename, regardless of size.
    Used to explain a near-miss ("right name, wrong size -- expected
    524288, got 524287") instead of a bare 'unrecognised'."""
    if not filename:
        return ()
    base = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].lower()
    return _index().by_name.get(base, ())


def known_for_system(system_id: str) -> tuple[KnownBios, ...]:
    """Every known BIOS for a system (and its aliases). Powers the
    'files this system is known to use' reference list in the vault."""
    wanted = set(aliases_for(system_id))
    if not wanted:
        return ()
    return tuple(e for e in _index().entries if e.system_id in wanted)


def stats() -> dict[str, int]:
    """Database size, for the health check and the vault footer."""
    idx = _index()
    return {
        "entries": len(idx.entries),
        "systems": len({e.system_id for e in idx.entries}),
        "with_sha1": len(idx.by_sha1),
        "with_md5": len(idx.by_md5),
        "with_crc32": len(idx.by_crc32),
    }
