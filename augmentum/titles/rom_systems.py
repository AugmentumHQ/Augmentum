"""ROM system detection -- map an uploaded ROM file to the right
emulator system + libretro core.

Strategy:
  1. Extension match (the common case -- .nes, .smc, .gba)
  2. Magic-byte / header inspection for ambiguous extensions (.bin
     could be PCE/Genesis/PS1 disc; .iso could be PSX or Saturn)
  3. Default to "unknown" with a hint that the user should pick

Adding a new system = one entry in ``_SYSTEMS`` + (optionally) a
header rule in ``_HEADER_RULES``. EmulatorJS expects the system id;
the libretro core id is informational here (the runtime adapter uses
it to populate the LaunchHandle's emulator config).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SystemSpec:
    id: str                 # EmulatorJS / RetroArch system id
    label: str              # human-readable name
    extensions: tuple[str, ...]
    libretro_core: str      # default libretro core for this system
    bios_required: bool = False
    notes: str = ""
    # Honest status flag for the runtime layer. Without it, systems whose
    # cores aren't actually bundled in EmulatorJS (or have no WASM port at
    # all) silently fall into EmulatorJS's "start core" failure UI when
    # the user clicks them — looks like a settings screen, isn't.
    #   "bundled"            — core .data file ships under ui/lib/emulator-js/
    #                          data/cores/ — runtime mounts it normally.
    #   "streaming_required" — no upstream WASM port (PCSX2, Dolphin, RPCS3,
    #                          Yuzu, etc.). Browser runtime refuses to launch
    #                          with a friendly error; will be playable once
    #                          the AGSP-streamed emulator runtime ships.
    #   "experimental"       — core exists upstream but isn't bundled in our
    #                          EmulatorJS build (vectrex's vecx, apple2's
    #                          linapple). Fixable by rebuilding EmulatorJS to
    #                          include them; until then the browser runtime
    #                          refuses to launch.
    core_status: str = "bundled"
    # When core_status == "streaming_required", these tell the
    # AgspStreamedRuntime which AGSP profile + emulator binary to use.
    # Profile id matches augmentum/game_stream/profiles.py registry;
    # emulator name matches the case-statement in the streamed
    # container's entrypoint (entrypoint-emulator-streamed.sh).
    # Both empty for browser-runnable systems.
    streaming_profile: str = ""
    streaming_emulator: str = ""


# ── System catalog ────────────────────────────────────────────────────


_SYSTEMS: tuple[SystemSpec, ...] = (
    # 8-bit
    SystemSpec("nes",       "Nintendo Entertainment System",
               (".nes",),                              "fceumm"),
    SystemSpec("gb",        "Game Boy",
               (".gb",),                               "gambatte"),
    SystemSpec("gbc",       "Game Boy Color",
               (".gbc",),                              "gambatte"),
    SystemSpec("sms",       "Sega Master System",
               (".sms",),                              "genesis_plus_gx"),
    SystemSpec("gg",        "Sega Game Gear",
               (".gg",),                               "genesis_plus_gx"),
    SystemSpec("atari2600", "Atari 2600",
               (".a26",),                              "stella2014"),
    SystemSpec("colecovision", "ColecoVision",
               (".col",),                              "gearcoleco"),
    # Vectrex's vecx core isn't bundled in our EmulatorJS build. Marked
    # experimental until we vendor it; clicking a .vec ROM today gets a
    # clean error toast instead of the broken "start core" UI.
    SystemSpec("vectrex",   "Vectrex",
               (".vec",),                              "vecx",
               core_status="experimental"),
    # 16-bit
    SystemSpec("snes",      "Super Nintendo",
               (".smc", ".sfc"),                       "snes9x"),
    # Note: ``.md`` is also a common Genesis/Mega Drive extension but
    # we deliberately exclude it because it collides with Markdown
    # (false-positive on README.md uploads). Users with .md ROMs can
    # rename to .gen or .smd, or pass system_id=genesis explicitly.
    SystemSpec("genesis",   "Sega Genesis / Mega Drive",
               (".gen", ".smd"),                       "genesis_plus_gx"),
    # Bundled core is named ``mednafen_pce`` in EmulatorJS; the ``_fast``
    # suffix was a stale upstream name and made every TurboGrafx-16 ROM
    # silently 404 the core file at launch.
    SystemSpec("pce",       "TurboGrafx-16 / PC Engine",
               (".pce",),                              "mednafen_pce"),
    SystemSpec("lynx",      "Atari Lynx",
               (".lnx",),                              "handy",
               bios_required=True),
    SystemSpec("ngp",       "Neo Geo Pocket",
               (".ngp", ".ngc"),                       "mednafen_ngp"),
    SystemSpec("ws",        "WonderSwan",
               (".ws", ".wsc"),                        "mednafen_wswan"),
    # 32-bit
    SystemSpec("gba",       "Game Boy Advance",
               (".gba",),                              "mgba"),
    SystemSpec("nds",       "Nintendo DS",
               (".nds",),                              "desmume"),
    SystemSpec("psx",       "PlayStation 1",
               (".cue", ".chd", ".pbp"),               "pcsx_rearmed",
               bios_required=True),
    SystemSpec("saturn",    "Sega Saturn",
               (".cdi",),                              "yabause",
               bios_required=True,
               notes="Performance varies; .iso also accepted via .bin/.cue pair"),
    SystemSpec("3do",       "3DO",
               (".iso", ".cue"),                       "opera",
               bios_required=True),
    SystemSpec("jaguar",    "Atari Jaguar",
               (".jag", ".j64"),                       "virtualjaguar"),
    # Arcade + larger
    SystemSpec("arcade",    "Arcade (FBNeo / MAME 2003)",
               (".zip",),                              "fbneo",
               notes="ZIP must be a recognised romset; FBNeo or MAME 2003 cores"),
    SystemSpec("n64",       "Nintendo 64",
               (".n64", ".v64", ".z64"),               "mupen64plus_next"),
    SystemSpec("psp",       "PlayStation Portable",
               (".iso", ".cso"),                       "ppsspp",
               notes="Performance is variable for PSP -- expect 30fps target on mid-range hardware"),
    # PlayStation 2 / GameCube / Wii: PCSX2 and Dolphin have no upstream
    # libretro WASM ports — they need a native process. Marked
    # streaming_required so the catalog can list them as roadmap, but the
    # browser runtime refuses to launch them with an actionable message.
    # The AGSP-streamed runtime (when wired up) will satisfy these.
    # PS2 disc images are 4.7 GB single-layer / 8.5 GB dual-layer; bump
    # emulator_rom_max_mb past 5000 (single) or 9000 (dual) to fit.
    # ``.bin``/.iso are universal disc-image extensions but collide with
    # other systems above (Genesis .bin, 3DO/PSP .iso) so we use the
    # PS2-specific compressed formats here. ``.bin``/.iso PS2 images can
    # still be uploaded with an explicit system_id=ps2 at the API call
    # site.
    SystemSpec("ps2",       "PlayStation 2",
               (".gz", ".cso", ".chd"),                "pcsx2",
               bios_required=True,
               core_status="streaming_required",
               # PCSX2 launcher not yet wired in the streamed container —
               # see services/game-stream/Dockerfile.emulator-streamed
               # TODO. Setting these now means flipping that single
               # dispatch case in the entrypoint is all that's needed
               # to enable PS2 once we add the binary.
               streaming_profile="emulator-streamed",
               streaming_emulator="pcsx2",
               notes="Streaming runtime required — PCSX2 has no WASM port. "
                     "Single-layer 4.7 GB needs emulator_rom_max_mb >= 5000"),
    # GameCube discs are 1.46 GB single-layer; fits inside the 2 GB
    # default ROM cap. Wii single-layer is 4.7 GB; bump
    # emulator_rom_max_mb past 5000. .iso excluded (collides with
    # 3DO/PSP/PS2); upload with system_id=gamecube|wii to override.
    SystemSpec("gamecube",  "Nintendo GameCube",
               (".gcm", ".gcz", ".rvz", ".ciso"),       "dolphin",
               core_status="streaming_required",
               streaming_profile="emulator-streamed",
               streaming_emulator="dolphin",
               notes="Streaming runtime required — Dolphin has no WASM port"),
    SystemSpec("wii",       "Nintendo Wii",
               (".wbfs", ".wad", ".wia"),               "dolphin",
               core_status="streaming_required",
               streaming_profile="emulator-streamed",
               streaming_emulator="dolphin",
               notes="Streaming runtime required — Dolphin has no WASM port. "
                     "Wii dual-layer (8.5 GB) needs emulator_rom_max_mb >= 9000"),
    # Computer / DOS
    SystemSpec("c64",       "Commodore 64",
               (".d64", ".t64", ".prg"),               "vice_x64"),
    SystemSpec("amiga",     "Amiga",
               (".adf", ".lha"),                       "puae",
               bios_required=True),
    # linapple isn't bundled in our EmulatorJS build. Marked experimental
    # until we vendor it; same as vectrex.
    SystemSpec("apple2",    "Apple II",
               (".dsk", ".woz"),                       "linapple",
               core_status="experimental"),
    SystemSpec("dos",       "DOS",
               (".exe", ".com", ".bat"),               "dosbox_pure",
               notes="Best results with full game folder zipped to .zip"),
)


# Quick lookup: extension -> SystemSpec
_BY_EXTENSION: dict[str, SystemSpec] = {}
for _spec in _SYSTEMS:
    for ext in _spec.extensions:
        # Prefer earlier definitions on collision (e.g. .iso → 3do
        # before psp). This is documented in the catalog ordering;
        # callers can override with explicit system_id at upload time.
        _BY_EXTENSION.setdefault(ext.lower(), _spec)


# Magic-byte rules for ambiguous extensions. Each rule is
# (offset, expected_bytes, system_id). Tested in order; first match wins.
#
# Disambiguates .iso uploads (the worst extension collision in the
# catalog — gamecube + wii + 3do + psp + ps2 all use it). With these
# rules, content sniffing puts each .iso in the right bucket; the
# extension-only fallback only fires when the header doesn't match
# anything (e.g. PSP .iso, which has no fixed early-byte signature).
_HEADER_RULES: tuple[tuple[int, bytes, str], ...] = (
    # NES: 'NES\x1a' at the very start
    (0, b"NES\x1a", "nes"),
    # GameCube: magic word 0xC2339F3D at offset 0x1C in the disc
    # header. Stable across every retail/homebrew GC image format
    # (.iso, .gcm, .rvz, .gcz — all carry the same header at the same
    # offset). See https://wiibrew.org/wiki/Wii_Disc#Header
    (0x1C, b"\xC2\x33\x9F\x3D", "gamecube"),
    # Wii: magic word 0x5D1C9EA3 at offset 0x18. Same source.
    (0x18, b"\x5D\x1C\x9E\xA3", "wii"),
    # GBA: 'GameBoy Advance' header at 0xa0 not stable; rely on extension
    # SNES: no fixed magic; rely on extension
    # PSX disc: 'PLAYSTATION' at offset 0x8001 within ISO's primary
    # volume descriptor. We don't read deeply enough for that here --
    # users with ambiguous .iso/.bin/.cue should specify system_id.
)


# ── Public API ───────────────────────────────────────────────────────


def detect_system(filename: str, header: bytes | None = None) -> SystemSpec | None:
    """Identify the emulator system from filename + optional header bytes.

    Returns None when nothing matches. Callers should treat None as
    "ask the user to pick" rather than guess.
    """
    name = filename.lower()
    # Magic-byte rules first when we have a header sample.
    if header:
        for offset, expected, system_id in _HEADER_RULES:
            if header[offset:offset + len(expected)] == expected:
                return _by_id(system_id)

        # PSX / PS2 disambiguation. Both consoles share the ISO9660
        # Primary Volume Descriptor "system identifier" string
        # ("PLAYSTATION") at offset 0x8008. The disambiguator is
        # SYSTEM.CNF in the disc root:
        #   PSX uses "BOOT = cdrom:\\SLPS_xxx.xx;1"
        #   PS2 uses "BOOT2 = cdrom0:\\SLUS_xxx.xx;1"
        # On retail discs SYSTEM.CNF lands within the first 64 KB of
        # the image (root directory extent is contiguous after the
        # PVD), so a presence-scan for b"BOOT2" inside the header
        # sample is enough — no ISO9660 walk required. This is the
        # only system pair we treat as a chained rule; everything
        # else is a clean fixed-offset match in _HEADER_RULES.
        if (
            len(header) >= 0x8008 + 11
            and header[0x8008:0x8008 + 11] == b"PLAYSTATION"
        ):
            if b"BOOT2" in header:
                return _by_id("ps2")
            return _by_id("psx")

    # Extension fallback.
    for ext, spec in _BY_EXTENSION.items():
        if name.endswith(ext):
            return spec
    return None


def list_systems() -> list[SystemSpec]:
    return list(_SYSTEMS)


def get_system(system_id: str) -> SystemSpec | None:
    return _by_id(system_id)


def supported_extensions() -> tuple[str, ...]:
    return tuple(_BY_EXTENSION.keys())


def _by_id(system_id: str) -> SystemSpec | None:
    for spec in _SYSTEMS:
        if spec.id == system_id:
            return spec
    return None
