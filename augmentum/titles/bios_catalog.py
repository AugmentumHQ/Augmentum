"""BIOS catalog -- per-system list of BIOS / firmware files emulators
need to boot retail games.

EmuDeck-grade scope: every system that has a documented BIOS need,
including ones we don't yet run in-browser (PS3, Switch, Wii U, 3DS).
This lets the bulk-import classifier recognise BIOS dumps the user
drops alongside their ROMs even when the corresponding emulator
runtime is gated behind a future streamed-container build.

Each entry carries the canonical filename emulators expect, the
expected file size, and (when known with high confidence) a SHA1
hash. The classifier matches on hash first, then (filename, size)
when no hash is recorded. SHA1=None entries match by filename+size
only, which is conservative-correct: a wrong file at the right size
fails to boot, but won't be silently miscategorised as something
else.

Hash sources -- the well-known canonical hashes here come from
RetroArch's libretro-database (system .dat files) and No-Intro /
Redump checksum lists, both public. The catalog can be expanded
either by editing this file directly or, for the lazy path, by
running the bulk-import flow with a known-good BIOS dump and
copying the resulting "unrecognised file" SHA1 into a new entry.

Adding a new system = one entry per file in ``_BIOS``. The
``optional`` flag controls whether the launch path warns or
hard-blocks when missing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class BiosFile:
    system_id: str           # matches rom_systems.SystemSpec.id
    filename: str            # canonical name emulators expect
    size_bytes: int          # exact byte length
    sha1: str | None         # None = match by (filename, size) only
    md5: str | None = None   # informational; not used for matching
    optional: bool = False   # True = HLE-able / nice-to-have
    description: str = ""    # shown in the BIOS status panel


# ── BIOS catalog ──────────────────────────────────────────────────────
#
# Ordered by 5th-gen first (PSX/Saturn -- the actively painful ones
# today), then 6th-gen+ (PS2/Dreamcast), then portables, then heavy
# modern systems that will need a streamed runtime.


_BIOS: tuple[BiosFile, ...] = (
    # ── Sony PlayStation ─────────────────────────────────────────────
    BiosFile("psx", "scph5500.bin", 524288,
             "b05def971d8ec59f346f2d9ac21fb742e3eb6917",
             description="PSX BIOS Japan v3.0"),
    BiosFile("psx", "scph5501.bin", 524288,
             "0555c6fae8906f3f09baf5988f00e55f88e9f30b",
             description="PSX BIOS USA v3.0"),
    BiosFile("psx", "scph5502.bin", 524288,
             "f6bc2d1f5eb6593de7d089c425ac681d6fffd3f0",
             description="PSX BIOS Europe v3.0"),
    BiosFile("psx", "scph7001.bin", 524288,
             "14df4f6c1e367ce097c11deae21566b4fe5647a9",
             description="PSX BIOS USA v4.1", optional=True),
    BiosFile("psx", "scph7003.bin", 524288, None,
             description="PSX BIOS USA v4.1 alt", optional=True),
    BiosFile("psx", "scph1001.bin", 524288,
             "10155d8d6e6e832d6ea66db9bc098321fb5b8170",
             description="PSX BIOS USA v2.2", optional=True),
    BiosFile("psx", "scph101.bin", 524288,
             "6e3735ff4c7dc899ee98981385f6f3b0298044d8",
             description="PSone BIOS USA v4.5", optional=True),

    # ── Sony PlayStation 2 ──────────────────────────────────────────
    # PS2 BIOS comes in many regional + revision variants. We list
    # the most common ones; users with a different revision will see
    # it as 'unknown' and can mark-as-bios via the override picker.
    BiosFile("ps2", "ps2-0100a-20000902.bin", 4194304, None,
             description="PS2 BIOS Japan v1.00", optional=True),
    BiosFile("ps2", "ps2-0120a-20010427.bin", 4194304, None,
             description="PS2 BIOS USA v1.20", optional=True),
    BiosFile("ps2", "ps2-0150e-20010427.bin", 4194304, None,
             description="PS2 BIOS Europe v1.50", optional=True),
    BiosFile("ps2", "ps2-0200a-20040614.bin", 4194304, None,
             description="PS2 BIOS USA v2.00"),
    BiosFile("ps2", "ps2-0220e-20040614.bin", 4194304, None,
             description="PS2 BIOS Europe v2.20", optional=True),
    BiosFile("ps2", "rom1.bin", 524288, None, optional=True,
             description="PS2 ROM1 (DVD player)"),
    BiosFile("ps2", "rom2.bin", 524288, None, optional=True,
             description="PS2 ROM2 (Chinese variant)"),
    BiosFile("ps2", "erom.bin", 1835008, None, optional=True,
             description="PS2 EROM (DVD region)"),

    # ── Sega Saturn ─────────────────────────────────────────────────
    BiosFile("saturn", "saturn_bios.bin", 524288,
             "fa27b60be485d1ba43404e5be5fc4fbd0d97ef3a",
             description="Saturn BIOS (sega_101) USA/Europe"),
    BiosFile("saturn", "sega_101.bin", 524288,
             "fa27b60be485d1ba43404e5be5fc4fbd0d97ef3a",
             description="Saturn BIOS USA/Europe (alt name)"),
    BiosFile("saturn", "mpr-17933.bin", 524288,
             "94a9cec53fe8f3e9b4842c5cdf03c2dd8b86c5e2",
             description="Saturn BIOS USA v1.01a"),
    BiosFile("saturn", "mpr-18811-mx.ic1", 524288, None,
             description="Saturn BIOS Japan v1.00", optional=True),
    BiosFile("saturn", "stv_bios.zip", 0, None, optional=True,
             description="Saturn ST-V arcade BIOS pack"),

    # ── Sega CD / Mega CD ───────────────────────────────────────────
    BiosFile("segacd", "bios_CD_U.bin", 131072, None,
             description="Sega CD BIOS USA v1.10"),
    BiosFile("segacd", "bios_CD_E.bin", 131072, None,
             description="Mega CD BIOS Europe v2.00"),
    BiosFile("segacd", "bios_CD_J.bin", 131072, None,
             description="Mega CD BIOS Japan v1.00s"),

    # ── Sega Dreamcast ──────────────────────────────────────────────
    BiosFile("dreamcast", "dc_boot.bin", 2097152, None,
             description="Dreamcast boot ROM"),
    BiosFile("dc", "dc_boot.bin", 2097152, None,
             description="Dreamcast boot ROM (alt id)"),
    BiosFile("dreamcast", "dc_flash.bin", 131072, None,
             description="Dreamcast flash ROM (region/clock)"),
    BiosFile("dc", "dc_flash.bin", 131072, None,
             description="Dreamcast flash (alt id)"),

    # ── 3DO ─────────────────────────────────────────────────────────
    BiosFile("3do", "panafz1.bin", 1048576, None,
             description="Panasonic FZ-1 (most-compatible)"),
    BiosFile("3do", "panafz10.bin", 1048576, None,
             description="Panasonic FZ-10", optional=True),
    BiosFile("3do", "goldstar.bin", 1048576, None,
             description="GoldStar GDO-101", optional=True),

    # ── Atari Lynx ──────────────────────────────────────────────────
    BiosFile("lynx", "lynxboot.img", 512,
             "e4ed47fae31693e016b081c6bda48da5b70d7ccb",
             description="Atari Lynx boot ROM"),

    # ── PC Engine CD / TurboGrafx-CD ────────────────────────────────
    BiosFile("pcecd", "syscard1.pce", 262144, None, optional=True,
             description="PCE-CD System Card 1.0"),
    BiosFile("pcecd", "syscard2.pce", 262144, None, optional=True,
             description="PCE-CD System Card 2.0"),
    BiosFile("pcecd", "syscard3.pce", 262144, None,
             description="PCE-CD System Card 3.0 (most games)"),
    BiosFile("pcecd", "gexpress.pce", 262144, None, optional=True,
             description="Game Express CD card"),

    # ── PC-FX ───────────────────────────────────────────────────────
    BiosFile("pcfx", "pcfx.rom", 1048576, None,
             description="PC-FX BIOS"),

    # ── Neo Geo / Neo Geo CD ────────────────────────────────────────
    # neogeo is fundamentally a zip of multiple BIOS variants; we
    # accept the canonical zip name and let the arcade core pick.
    BiosFile("neogeo", "neogeo.zip", 0, None,
             description="Neo Geo MVS/AES BIOS pack"),
    BiosFile("neogeocd", "neocd.bin", 524288, None, optional=True,
             description="Neo Geo CD BIOS (front/top loader)"),
    BiosFile("neogeocd", "front-sp1.bin", 524288, None, optional=True,
             description="Neo Geo CD front-loader BIOS"),
    BiosFile("neogeocd", "top-sp1.bin", 524288, None, optional=True,
             description="Neo Geo CD top-loader BIOS"),

    # ── Atari 5200 / 7800 ──────────────────────────────────────────
    BiosFile("atari5200", "5200.rom", 2048, None,
             description="Atari 5200 BIOS"),
    BiosFile("atari7800", "7800 BIOS (U).rom", 4096, None, optional=True,
             description="Atari 7800 BIOS USA (HLE-able)"),
    BiosFile("atari7800", "7800 BIOS (E).rom", 16384, None, optional=True,
             description="Atari 7800 BIOS Europe (HLE-able)"),

    # ── ColecoVision ────────────────────────────────────────────────
    BiosFile("colecovision", "colecovision.rom", 8192,
             "45bedc4cbdeac66c7df59e9e599195c778d86a92",
             description="ColecoVision BIOS"),
    BiosFile("colecovision", "coleco.rom", 8192,
             "45bedc4cbdeac66c7df59e9e599195c778d86a92",
             description="ColecoVision BIOS (alt name)"),

    # ── Intellivision ───────────────────────────────────────────────
    BiosFile("intellivision", "exec.bin", 8192, None,
             description="Intellivision EXEC ROM"),
    BiosFile("intellivision", "grom.bin", 2048, None,
             description="Intellivision GROM"),

    # ── Amiga ───────────────────────────────────────────────────────
    BiosFile("amiga", "kick13.rom", 262144, None,
             description="Amiga Kickstart 1.3 (A500 default)"),
    BiosFile("amiga", "kick20.rom", 524288, None, optional=True,
             description="Amiga Kickstart 2.0", ),
    BiosFile("amiga", "kick31.rom", 524288, None, optional=True,
             description="Amiga Kickstart 3.1 (A1200/A4000)"),

    # ── Nintendo handhelds (BIOS optional, HLE works) ──────────────
    BiosFile("gba", "gba_bios.bin", 16384,
             "300c20df6731a33952ded8c436f7f186d25d3492",
             description="GBA BIOS (optional; mGBA HLEs)", optional=True),
    BiosFile("nds", "bios7.bin", 16384,
             "24f67bdea115a2c847c8813a262502ee1607b7df",
             description="NDS ARM7 BIOS (melonDS requires)", optional=True),
    BiosFile("nds", "bios9.bin", 4096,
             "1ba4174f5499e4c3502c324db5247b25b0c54fc4",
             description="NDS ARM9 BIOS (melonDS requires)", optional=True),
    BiosFile("nds", "firmware.bin", 262144, None,
             description="NDS firmware (settings + WiFi)", optional=True),

    # ── Nintendo DSi (extends NDS) ─────────────────────────────────
    BiosFile("dsi", "bios7i.bin", 65536, None, optional=True,
             description="DSi ARM7 BIOS"),
    BiosFile("dsi", "bios9i.bin", 65536, None, optional=True,
             description="DSi ARM9 BIOS"),
    BiosFile("dsi", "nand.bin", 251658240, None, optional=True,
             description="DSi NAND dump (per-console)"),

    # ── Nintendo GameCube ──────────────────────────────────────────
    # Dolphin doesn't require a BIOS dump (HLE IPL by default),
    # but a real IPL gives the boot animation + region check.
    BiosFile("gamecube", "IPL.bin", 2097152, None, optional=True,
             description="GC IPL (boot animation; HLE works)"),

    # ── Nintendo 3DS ────────────────────────────────────────────────
    BiosFile("3ds", "boot9.bin", 65536, None,
             description="3DS ARM9 boot ROM (Citra/Lime3DS requires)"),
    BiosFile("3ds", "boot11.bin", 65536, None,
             description="3DS ARM11 boot ROM"),
    BiosFile("3ds", "aes_keys.txt", 0, None,
             description="3DS AES keys (text file; size varies)"),

    # ── Nintendo Switch (heavy modern; needs streamed runtime) ─────
    BiosFile("switch", "prod.keys", 0, None,
             description="Switch production keys (size varies; ~10KB)"),
    BiosFile("switch", "title.keys", 0, None, optional=True,
             description="Switch title keys"),

    # ── Nintendo Wii U (heavy modern; needs streamed runtime) ──────
    BiosFile("wiiu", "otp.bin", 1024, None,
             description="Wii U OTP (per-console; required by Cemu)"),
    BiosFile("wiiu", "seeprom.bin", 512, None, optional=True,
             description="Wii U SEEPROM (per-console)"),

    # ── Sony PlayStation Portable (optional; PPSSPP HLEs) ──────────
    BiosFile("psp", "PSPBIOS.PBP", 0, None, optional=True,
             description="PSP firmware PBP (PPSSPP HLEs by default)"),

    # ── Sony PlayStation 3 (heavy modern; needs streamed runtime) ──
    # PS3 firmware is delivered as an installable PUP, not raw bytes.
    BiosFile("ps3", "PS3UPDAT.PUP", 0, None,
             description="PS3 firmware (install via RPCS3 → File → "
                         "Install Firmware)"),
)


# ── Indexes ──────────────────────────────────────────────────────────


# Hash → file (highest-confidence match: SHA1 collisions are
# astronomically unlikely for distinct files of the same size).
_BY_SHA1: dict[str, BiosFile] = {}
# (lowercased filename, size) → file. Filename only is too loose;
# size pins it. A dropped "bios.bin" with size 524288 could be many
# things, so we also keep a (filename) loose index for "is this the
# right name for *some* known BIOS" classification confidence.
_BY_NAME_SIZE: dict[tuple[str, int], BiosFile] = {}
_BY_NAME: dict[str, list[BiosFile]] = {}


for _f in _BIOS:
    if _f.sha1:
        _BY_SHA1.setdefault(_f.sha1.lower(), _f)
    if _f.size_bytes:
        _BY_NAME_SIZE.setdefault((_f.filename.lower(), _f.size_bytes), _f)
    _BY_NAME.setdefault(_f.filename.lower(), []).append(_f)


# ── Public API ───────────────────────────────────────────────────────


def lookup_by_sha1(sha1: str) -> BiosFile | None:
    """Strongest-confidence match. SHA1 collisions are not a concern;
    a hit here is definitive identification."""
    if not sha1:
        return None
    return _BY_SHA1.get(sha1.lower())


def lookup_by_name_size(filename: str, size_bytes: int) -> BiosFile | None:
    """Fallback match when SHA1 isn't catalogued. ``filename`` is
    matched case-insensitively against the canonical name; size must
    match exactly. A dropped file with the wrong size for a given
    canonical name returns None (so we don't misclassify a corrupt
    or wrong-region dump as the wrong file)."""
    if not filename or size_bytes <= 0:
        return None
    key = (filename.lower(), int(size_bytes))
    return _BY_NAME_SIZE.get(key)


def lookup_loose_by_name(filename: str) -> list[BiosFile]:
    """Loose name-only match -- used as a hint when classification is
    ambiguous (e.g., user dropped 'scph7001.bin' but the size doesn't
    match the catalogued size; we can still surface 'this looks like a
    PSX BIOS, but the size is off'). Returns all entries that share
    the canonical name (different systems can reuse names like
    'bios.bin', so a list is correct)."""
    if not filename:
        return []
    return list(_BY_NAME.get(filename.lower(), ()))


# ── Manufacturer-naming pattern fallbacks ────────────────────────────
#
# Real-world BIOS dumps come in two filename conventions: libretro's
# canonical format (e.g. ``ps2-0200a-20040614.bin``) and the original
# manufacturer naming (Sony's SCPH-XXXXX, Atari's lynxboot.img,
# Commodore's kickstart.rom, etc.). The catalog uses libretro names
# because that's what the libretro databases match against — but
# every PS2 dump in circulation uses Sony naming, and forcing users
# to rename ``scph39001.bin`` to ``ps2-0190a-20030822.bin`` before
# upload is a terrible UX.
#
# Each rule maps (filename regex, expected size, system_id) to a
# canonical catalog entry. When a match hits, we resolve it back to
# that catalog entry so install / list_status / missing_required all
# see a normal hit. PCSX2 / other emulators ID the BIOS at runtime
# from the bytes, so the canonical name we store under is just
# bookkeeping — what matters is that the byte-content is correct
# and the size matches.
#
# Adding patterns: the regex must match the full filename (anchored
# both ends). Size = exact byte length. Canonical = a filename
# already present in _BIOS.

_PATTERN_RULES: tuple[tuple[re.Pattern[str], int, str, str], ...] = (
    # Sony PlayStation 2: every retail BIOS revision is 4 MiB and
    # follows scph-XXXXX naming. PCSX2 reads the file's internal
    # version blocks at boot, so any valid 4 MiB SCPH dump satisfies
    # the launch path regardless of revision. Map them all to the
    # 2.00 USA entry (the most-common required slot) so list_status
    # marks PS2 ready as soon as ANY SCPH dump lands.
    (re.compile(r"^scph[-_ ]?\d{4,5}[a-z]?\.bin$", re.IGNORECASE),
     4_194_304, "ps2", "ps2-0200a-20040614.bin"),

    # Sony PlayStation 1 (classic): scph BIOS dumps that aren't the
    # exact catalogued revisions still satisfy launch — every PSX
    # BIOS is 512 KiB. Existing scph5500/5501/5502 entries in the
    # catalog have known SHA1s (and win via lookup_by_sha1 first);
    # this catches the dozen+ revisions that don't have hashes
    # recorded yet.
    (re.compile(r"^scph[-_ ]?\d{3,5}[a-z]?\.bin$", re.IGNORECASE),
     524_288, "psx", "scph5501.bin"),

    # Commodore Amiga: Kickstart ROMs follow kickXX.rom or
    # kickstart-XX.rom. Sizes vary by revision (256 KiB / 512 KiB),
    # so we list each size in its own rule pointing at the
    # correspondingly-sized catalog entry.
    (re.compile(r"^kick(start)?[-_ ]?1[\.\-_]?[23][0-9]?\.rom$",
                re.IGNORECASE),
     262_144, "amiga", "kick13.rom"),
    (re.compile(r"^kick(start)?[-_ ]?2[\.\-_]?[0-9]+[a-z]?\.rom$",
                re.IGNORECASE),
     524_288, "amiga", "kick20.rom"),
    (re.compile(r"^kick(start)?[-_ ]?3[\.\-_]?[0-9]+[a-z]?\.rom$",
                re.IGNORECASE),
     524_288, "amiga", "kick31.rom"),
)


def lookup_by_pattern(filename: str, size_bytes: int) -> BiosFile | None:
    """Pattern-based fallback: map manufacturer-naming dumps onto a
    canonical catalog entry when (a) the user's filename matches a
    known pattern AND (b) the byte length matches what the canonical
    expects. Returns the canonical BiosFile (not a synthesized one)
    so install + status code paths see a normal catalog hit.

    This is a fallback after SHA1 and (filename, size) exact lookup
    fail — both of those are stronger signals and shouldn't be
    bypassed. Used by ``classify()`` before the unknown-bucket
    fallthrough."""
    if not filename or size_bytes <= 0:
        return None
    base = filename.lower().rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    for rx, expected_size, system_id, canonical_name in _PATTERN_RULES:
        if size_bytes != expected_size:
            continue
        if not rx.match(base):
            continue
        # Resolve canonical_name back to the actual BiosFile so the
        # caller can install/store consistently. Linear scan over
        # _BIOS is fine — catalog has ~70 entries.
        for f in _BIOS:
            if (
                f.system_id == system_id
                and f.filename.lower() == canonical_name.lower()
            ):
                return f
    return None


def required_for_system(system_id: str) -> list[BiosFile]:
    """All non-optional BIOS files for a system. Used by the launch
    path to validate the user has what's needed before booting."""
    return [f for f in _BIOS if f.system_id == system_id and not f.optional]


def all_for_system(system_id: str) -> list[BiosFile]:
    """Every catalogued BIOS file for a system (required + optional).
    Used by the BIOS status panel to render the full checklist."""
    return [f for f in _BIOS if f.system_id == system_id]


def systems_with_bios() -> list[str]:
    """Sorted unique system IDs that have at least one catalog entry.
    Used by the status panel header and the bulk-import summary."""
    return sorted({f.system_id for f in _BIOS})


def all_files() -> list[BiosFile]:
    """Whole catalog. Mostly for tests / introspection."""
    return list(_BIOS)
