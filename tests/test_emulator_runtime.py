"""Tests for the AXF emulator-browser runtime, ROM system detection,
InternalRomSource, and the ROM-upload route.
"""

from __future__ import annotations

import json

import aiosqlite
import pytest

_SCHEMA_SQL = """
CREATE TABLE users (id TEXT PRIMARY KEY);

CREATE TABLE artifacts (
    id              TEXT PRIMARY KEY,
    task_id         TEXT NOT NULL DEFAULT '',
    session_id      TEXT NOT NULL DEFAULT '',
    filename        TEXT NOT NULL DEFAULT '',
    display_name    TEXT NOT NULL DEFAULT '',
    format          TEXT NOT NULL DEFAULT '',
    size_bytes      INTEGER NOT NULL DEFAULT 0,
    path            TEXT NOT NULL DEFAULT '',
    metadata        TEXT NOT NULL DEFAULT '{}',
    user_id         TEXT NOT NULL REFERENCES users(id),
    pinned          INTEGER NOT NULL DEFAULT 0,
    last_opened_at  TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


# ── ROM system detection ───────────────────────────────────────────


def test_detect_system_by_extension():
    from augmentum.titles.rom_systems import detect_system

    spec = detect_system("Super Mario Bros.nes")
    assert spec is not None and spec.id == "nes"

    spec = detect_system("Zelda LTTP.smc")
    assert spec is not None and spec.id == "snes"

    spec = detect_system("Pokemon Red.gb")
    assert spec is not None and spec.id == "gb"


def test_detect_system_nes_magic_bytes():
    """NES magic-byte rule overrides extension when both agree, and
    catches misnamed files."""
    from augmentum.titles.rom_systems import detect_system

    nes_header = b"NES\x1a" + b"\x00" * 12
    spec = detect_system("misnamed.bin", header=nes_header)
    assert spec is not None and spec.id == "nes"


def test_detect_system_returns_none_for_unknown():
    from augmentum.titles.rom_systems import detect_system

    assert detect_system("README.md") is None
    assert detect_system("unknown.xyz", header=b"\x00\x00\x00\x00") is None


def test_get_system_lookup_known_id():
    from augmentum.titles.rom_systems import get_system

    snes = get_system("snes")
    assert snes is not None and snes.libretro_core == "snes9x"


# ── InternalRomSource ──────────────────────────────────────────────


async def _mkdb():
    conn = await aiosqlite.connect(":memory:")
    await conn.executescript(_SCHEMA_SQL)
    await conn.execute("INSERT INTO users (id) VALUES ('u1')")
    await conn.execute("INSERT INTO users (id) VALUES ('u2')")
    await conn.commit()
    return conn


@pytest.mark.asyncio
async def test_internal_rom_source_creates_emulator_artifact():
    from augmentum.titles import InternalRomSource

    conn = await _mkdb()
    src = InternalRomSource(conn)
    artifact_id = await src.import_for_user(
        {
            "rom_sha256": "abc123",
            "rom_size_bytes": 524288,
            "system_id": "nes",
            "title": "Test Game",
            "original_filename": "test.nes",
        },
        user_id="u1",
    )
    assert artifact_id

    cursor = await conn.execute(
        "SELECT metadata FROM artifacts WHERE id = ?", (artifact_id,),
    )
    md = json.loads((await cursor.fetchone())[0])
    assert md["kind"] == "emulator_rom"
    assert md["source"] == "internal-rom"
    assert md["source_id"] == "abc123"           # ROM hash = stable id
    assert md["system_id"] == "nes"
    assert md["libretro_core"] == "fceumm"
    assert md["runtime_preferred"] == "emulator-browser"


@pytest.mark.asyncio
async def test_internal_rom_source_idempotent():
    """Re-importing the same ROM (same sha) returns the existing id."""
    from augmentum.titles import InternalRomSource

    conn = await _mkdb()
    src = InternalRomSource(conn)
    a = await src.import_for_user(
        {"rom_sha256": "abc", "rom_size_bytes": 100,
         "system_id": "nes", "title": "A"},
        user_id="u1",
    )
    b = await src.import_for_user(
        {"rom_sha256": "abc", "rom_size_bytes": 100,
         "system_id": "nes", "title": "A"},
        user_id="u1",
    )
    assert a == b
    cursor = await conn.execute(
        "SELECT COUNT(*) FROM artifacts WHERE user_id = 'u1'",
    )
    assert (await cursor.fetchone())[0] == 1


@pytest.mark.asyncio
async def test_internal_rom_source_rejects_unknown_system():
    from augmentum.titles import InternalRomSource, SourceImportError

    conn = await _mkdb()
    src = InternalRomSource(conn)
    with pytest.raises(SourceImportError):
        await src.import_for_user(
            {"rom_sha256": "x", "rom_size_bytes": 1,
             "system_id": "no-such-system", "title": "X"},
            user_id="u1",
        )


# ── EmulatorBrowserRuntime ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_emulator_browser_supports_emulator_rom():
    from augmentum.titles import EmulatorBrowserRuntime, TitleManifest

    rt = EmulatorBrowserRuntime()
    row = {
        "id": "art_rom",
        "user_id": "u1",
        "metadata": json.dumps({
            "kind": "emulator_rom",
            "source": "internal-rom",
            "source_id": "abc",
            "system_id": "nes",
            "libretro_core": "fceumm",
            "rom_sha256": "abc",
        }),
    }
    manifest = TitleManifest.from_artifact_row(row)
    assert await rt.supports(manifest) is True


@pytest.mark.asyncio
async def test_emulator_browser_does_not_support_other_kinds():
    from augmentum.titles import EmulatorBrowserRuntime, TitleManifest

    rt = EmulatorBrowserRuntime()
    row = {
        "id": "art_web",
        "user_id": "u1",
        "metadata": json.dumps({"kind": "web_app"}),
    }
    manifest = TitleManifest.from_artifact_row(row)
    assert await rt.supports(manifest) is False


@pytest.mark.asyncio
async def test_emulator_browser_launch_handle_shape():
    """The LaunchHandle's metadata must carry every config key the
    frontend EmulatorJS bridge reads -- system, core, rom URL, save
    bridge URL, emulator_js_path."""
    from augmentum.titles import EmulatorBrowserRuntime, TitleManifest

    rt = EmulatorBrowserRuntime()
    row = {
        "id": "art_rom",
        "user_id": "u1",
        "display_name": "Mario",
        "metadata": json.dumps({
            "kind": "emulator_rom",
            "source": "internal-rom",
            "source_id": "abc",
            "system_id": "nes",
            "libretro_core": "fceumm",
            "rom_sha256": "abc",
            "bios_required": False,
            "title": "Mario",
        }),
    }
    manifest = TitleManifest.from_artifact_row(row)
    handle = await rt.launch(manifest, ctx={"user_id": "u1"})
    assert handle.runtime_id == "emulator-browser"
    assert handle.kind == "emulator"
    md = handle.metadata
    assert md["system"] == "nes"
    assert md["core"] == "fceumm"
    assert md["rom_url"] == "/api/titles/art_rom/rom"
    assert md["save_bridge_url"] == "/api/titles/art_rom/saves"
    assert md["emulator_js_path"] == "/ui/lib/emulator-js/data/"
    assert md["bios_required"] is False


@pytest.mark.asyncio
async def test_emulator_browser_launch_rejects_missing_metadata():
    """A title row that's missing system_id or rom_sha256 must error
    cleanly rather than producing a half-baked LaunchHandle."""
    from augmentum.titles import EmulatorBrowserRuntime, TitleManifest

    rt = EmulatorBrowserRuntime()
    row = {
        "id": "art_rom",
        "user_id": "u1",
        "metadata": json.dumps({
            "kind": "emulator_rom",
            "source": "internal-rom",
            # Missing rom_sha256 + system_id
        }),
    }
    manifest = TitleManifest.from_artifact_row(row)
    with pytest.raises(RuntimeError):
        await rt.launch(manifest, ctx={"user_id": "u1"})


# ── Runtime registry resolution ────────────────────────────────────


@pytest.mark.asyncio
async def test_runtime_registry_resolves_emulator_browser_for_rom():
    from augmentum.titles import (
        BrowserIframeRuntime,
        EmulatorBrowserRuntime,
        RuntimeRegistry,
        TitleManifest,
    )

    reg = RuntimeRegistry()
    reg.register(BrowserIframeRuntime())
    reg.register(EmulatorBrowserRuntime())
    row = {
        "id": "art_rom",
        "user_id": "u1",
        "metadata": json.dumps({
            "kind": "emulator_rom",
            "source": "internal-rom",
            "system_id": "nes",
            "rom_sha256": "abc",
            "libretro_core": "fceumm",
            "runtime_preferred": "emulator-browser",
        }),
    }
    manifest = TitleManifest.from_artifact_row(row)
    rt = await reg.resolve_for(manifest)
    assert rt is not None and rt.id == "emulator-browser"
