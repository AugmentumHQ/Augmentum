"""Builder P0 foundation tests.

Covers the genuinely-new P0 surface (spec:
docs/superpowers/specs/2026-06-15-builder-profiles-system-synthesizer-design.md):

- the builder resource tools (the app-builder toolkit exposed as pulled-on
  coder tools), and
- the build_runs profile/target/capabilities/workspace_id persistence
  (migration 269), against the REAL migration files applied to an in-memory
  SQLite so schema drift is impossible, and
- the Frontend App Builder Power manifest parses with the expected shape.
"""
from __future__ import annotations

from pathlib import Path

import aiosqlite
import pytest

from augmentum.builds.store import BuildRunStore
from augmentum.coder.builder_tools import (
    BUILDER_TOOLS,
    create_builder_tools,
)

_MIG_DIR = Path(__file__).resolve().parents[1] / "augmentum" / "state" / "migrations"
_BUILD_RUN_MIGRATIONS = (
    "146_build_runs.sql",
    "217_build_runs_acked.sql",
    "269_build_runs_profile.sql",
)


async def _mkstore() -> tuple[BuildRunStore, aiosqlite.Connection]:
    conn = await aiosqlite.connect(":memory:")
    # The real migration runner creates schema_version first; some legacy
    # migrations (146) self-record into it. Provide it so the files apply
    # in isolation.
    await conn.executescript(
        "CREATE TABLE IF NOT EXISTS schema_version "
        "(version INTEGER PRIMARY KEY, description TEXT);"
    )
    for name in _BUILD_RUN_MIGRATIONS:
        await conn.executescript((_MIG_DIR / name).read_text(encoding="utf-8"))
    return BuildRunStore(conn), conn


# ---------------------------------------------------------------------------
# Resource tools
# ---------------------------------------------------------------------------


def test_builder_tools_factory_shape():
    tools = create_builder_tools()
    assert len(tools) == len(BUILDER_TOOLS)
    names = {t.name for t in tools}
    assert names == {"builder_reference", "builder_design_system", "builder_api_refs"}
    # Builder-internal: reachable by the coder agent, not chat.
    for t in tools:
        assert t.surfaces.coder is True
        assert t.surfaces.chat is False
        assert isinstance(t.input_schema, dict) and t.input_schema.get("type") == "object"


@pytest.mark.asyncio
async def test_design_system_tool_returns_palette():
    tool = next(t for t in create_builder_tools() if t.name == "builder_design_system")
    res = await tool.execute(description="a calm minimalist note taking app", kind="static")
    assert res.success
    # Concrete CSS custom properties the agent is told to reference.
    assert "--surface" in res.output and "--text" in res.output
    assert res.metadata.get("mood")


@pytest.mark.asyncio
async def test_api_refs_tool_detects_game_canvas():
    tool = next(t for t in create_builder_tools() if t.name == "builder_api_refs")
    res = await tool.execute(description="a snake arcade game on a canvas", kind="game")
    assert res.success
    assert "canvas_game" in (res.metadata.get("categories") or [])
    # Verified Canvas idiom should be present (arc, not the hallucinated fillCircle).
    assert "arc(" in res.output


@pytest.mark.asyncio
async def test_api_refs_tool_explicit_categories_override():
    tool = next(t for t in create_builder_tools() if t.name == "builder_api_refs")
    res = await tool.execute(description="anything", kind="static", categories=["charts_dashboard"])
    assert res.success
    assert res.metadata.get("categories") == ["charts_dashboard"]


@pytest.mark.asyncio
async def test_reference_tool_returns_block_for_known_kind():
    tool = next(t for t in create_builder_tools() if t.name == "builder_reference")
    res = await tool.execute(kind="form", query="tip calculator with operations", max_refs=2)
    assert res.success
    # The always-include base reference for a scaffold should always land.
    assert len(res.output.strip()) > 0


@pytest.mark.asyncio
async def test_reference_tool_coerces_unknown_kind():
    tool = next(t for t in create_builder_tools() if t.name == "builder_reference")
    # An unknown kind must not raise — it falls back to the static family.
    res = await tool.execute(kind="not-a-real-kind", query="dashboard")
    assert res.success


# ---------------------------------------------------------------------------
# build_runs profile/target/capabilities/workspace_id persistence (mig 269)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_build_run_profile_fields_roundtrip():
    store, conn = await _mkstore()
    try:
        run = await store.create(
            user_id="u1",
            name="Tip Calculator",
            profile_id="form",
            target="inline",
            capabilities=["controllers"],
            workspace_id="ws-abc",
        )
        assert run["profile_id"] == "form"
        assert run["target"] == "inline"
        assert run["capabilities"] == ["controllers"]
        assert run["workspace_id"] == "ws-abc"

        fetched = await store.get(run["id"], user_id="u1")
        assert fetched is not None
        assert fetched["profile_id"] == "form"
        assert fetched["capabilities"] == ["controllers"]
        assert fetched["workspace_id"] == "ws-abc"
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_build_run_defaults_back_compat():
    store, conn = await _mkstore()
    try:
        # A create with no profile args must land on the historical shape.
        run = await store.create(user_id="u1", name="Legacy")
        assert run["profile_id"] == "static"
        assert run["target"] == "inline"
        assert run["capabilities"] == []
        assert run["workspace_id"] == ""
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_build_run_update_sets_workspace_id():
    store, conn = await _mkstore()
    try:
        run = await store.create(user_id="u1", name="Snake", profile_id="game")
        await store.update(run["id"], user_id="u1", workspace_id="ws-live", target="workspace")
        fetched = await store.get(run["id"], user_id="u1")
        assert fetched["workspace_id"] == "ws-live"
        assert fetched["target"] == "workspace"
        assert fetched["profile_id"] == "game"
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_build_run_user_isolation_unaffected():
    store, conn = await _mkstore()
    try:
        run = await store.create(user_id="u1", name="Mine", profile_id="game")
        # Another user must not see it.
        assert await store.get(run["id"], user_id="u2") is None
    finally:
        await conn.close()


# ---------------------------------------------------------------------------
# Frontend App Builder Power manifest
# ---------------------------------------------------------------------------


def test_frontend_app_power_manifest_parses():
    from augmentum.powers.manifest import discover_manifest_file, parse_power_manifest

    pkg = Path(__file__).resolve().parents[1] / ".augmentum" / "powers" / "frontend-app"
    manifest_path = discover_manifest_file(pkg)
    assert manifest_path is not None, "frontend-app POWER.md must exist"
    m = parse_power_manifest(manifest_path, source_kind="native")
    assert m.id == "frontend-app"
    assert m.kind == "guidance"
    assert m.activation_policy == "explicit_only"
    # The build agent must be steered toward the resource tools + verification loop.
    for needed in ("builder_design_system", "browser_open", "service_start", "finish_task"):
        assert needed in m.preferred_tools, f"{needed} should be a preferred tool"
    assert "post_write" in m.activation_windows


# ---------------------------------------------------------------------------
# Builder facade — pure helpers (no Docker / no backend)
# ---------------------------------------------------------------------------


def _make_targz(members: dict[str, bytes]) -> bytes:
    import io
    import tarfile

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_facade_normalize_member_path():
    from augmentum.builds.facade import _normalize_member_path

    assert _normalize_member_path("workspace/app.js") == "app.js"
    assert _normalize_member_path("./workspace/styles.css") == "styles.css"
    assert _normalize_member_path("/workspace/sub/x.js") == "sub/x.js"
    assert _normalize_member_path("index.html") == "index.html"


def test_facade_extract_text_files_filters_and_flags_entrypoint():
    from augmentum.builds.facade import extract_text_files_from_targz

    targz = _make_targz({
        "workspace/index.html": b"<!doctype html><title>x</title>",
        "workspace/app.js": b"console.log(1)",
        "workspace/node_modules/dep.js": b"// should be excluded",
        "workspace/.git/config": b"[core]",
        "workspace/logo.png": b"\x89PNG\r\n\x1a\n\x00\xff\xfe",  # binary → skipped
    })
    files = extract_text_files_from_targz(targz)
    paths = {f["path"] for f in files}
    assert paths == {"index.html", "app.js"}, paths
    entry = next(f for f in files if f["path"] == "index.html")
    assert entry.get("isEntrypoint") is True


def test_facade_source_json_is_application_with_profile():
    import json as _json

    from augmentum.builds.facade import build_source_json

    raw = build_source_json(
        name="Snake", files=[{"path": "index.html", "content": "<x>"}],
        profile_id="game", target="inline", capabilities=["controllers"],
    )
    data = _json.loads(raw)
    assert data["type"] == "application"  # back-compat detection preserved
    assert data["profile"] == "game"
    assert data["capabilities"] == ["controllers"]
    assert data["files"][0]["path"] == "index.html"


def test_facade_zip_round_trips():
    import io
    import zipfile

    from augmentum.builds.facade import zip_files

    blob = zip_files([{"path": "a/b.txt", "content": "hello"}])
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        assert zf.read("a/b.txt").decode() == "hello"


def test_facade_system_prompt_fallback_mentions_resources():
    from augmentum.builds.facade import build_system_prompt

    # No registry → default prompt, but it must still steer toward the loop + tools.
    prompt = build_system_prompt(None)
    for needed in ("builder_reference", "browser_open", "finish_task", "console"):
        assert needed in prompt


def test_facade_safe_filename():
    from augmentum.builds.facade import _safe_filename

    assert _safe_filename("My Tip Calculator!") == "my-tip-calculator"
    assert _safe_filename("") == "app"
