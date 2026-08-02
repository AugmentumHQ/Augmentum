"""Gated tool capabilities (Phase 2): the model REQUESTS heavy tools; the
server surfaces a confirmation offer instead of firing them.

Slice 1 — the SSOS framework: gated capabilities are advertised to the model,
the marker is recognized as gated (not run), and the lookup path is untouched.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from augmentum.modes.passthrough.orchestrator import SSOSOrchestrator


class _FakeRegistry:
    def get(self, name):
        return None  # forces fallback_hint in the prompt block

    def list_tools(self):
        return []


def _orch(app_state=None, user_id="u1"):
    return SSOSOrchestrator(_FakeRegistry(), user_id=user_id, app_state=app_state)


def test_gated_capabilities_present():
    names = {c.name for c in SSOSOrchestrator.gated_capabilities()}
    assert names == {
        "image_generation", "build_application",
        "create_ebook", "create_presentation", "create_document",
        "create_chart", "create_spreadsheet",
    }
    assert all(c.kind == "gated" for c in SSOSOrchestrator.gated_capabilities())


def test_only_structured_creators_need_a_plan():
    by_name = {c.name: c for c in SSOSOrchestrator.gated_capabilities()}
    assert by_name["create_ebook"].needs_plan is True
    assert by_name["create_presentation"].needs_plan is True
    assert by_name["create_document"].needs_plan is True
    # single-string-arg tools fire directly, no planner
    assert by_name["image_generation"].needs_plan is False
    assert by_name["build_application"].needs_plan is False
    # Never-gated creators are exposed as their REAL tool, so the model fills
    # the real structured schema and there is no brief for a planner to expand.
    # If either is ever re-gated, add a gated_planner PlanSpec first — the
    # single-arg proxy would otherwise hand it {brief: ...}, which does not
    # match these schemas.
    assert by_name["create_chart"].needs_plan is False
    assert by_name["create_spreadsheet"].needs_plan is False


def test_chart_and_spreadsheet_are_never_gated():
    """They must run inline, not surface a confirmation chip.

    A chart the model chose to draw should just appear, the way a generated
    image does. Membership is asserted against the module-level set so this
    fails loudly if someone adds them back to the proposing path.
    """
    from augmentum.modes.passthrough.handler import _NEVER_GATED_CAPABILITIES
    assert sorted(_NEVER_GATED_CAPABILITIES) == ["create_chart", "create_spreadsheet"]
    gated = {c.tool for c in SSOSOrchestrator.gated_capabilities()}
    assert gated >= _NEVER_GATED_CAPABILITIES


def test_never_gated_capability_dep_gate():
    """Chart/spreadsheet are dropped from Auto when their render dep is gone.

    Other inline capabilities (image_generation) are unaffected: a transient
    unhealthy provider must not make the model deny it can generate images.
    """
    from augmentum.modes.passthrough.handler import _capability_dep_available

    assert _capability_dep_available(
        SimpleNamespace(name="create_chart", health_check=lambda: True)) is True
    assert _capability_dep_available(
        SimpleNamespace(name="create_chart", health_check=lambda: False)) is False
    # not a never-gated creator → never consulted, stays exposed
    assert _capability_dep_available(
        SimpleNamespace(name="image_generation", health_check=lambda: False)) is True

    # A throwing probe must not silently remove a working tool.
    def _boom():
        raise RuntimeError("probe failed")
    assert _capability_dep_available(
        SimpleNamespace(name="create_chart", health_check=_boom)) is True


def test_chart_health_check_tracks_matplotlib():
    """The gate is only meaningful because the tools override health_check."""
    from importlib.util import find_spec

    from augmentum.tools.artifact_chart import ChartTool
    from augmentum.tools.artifact_spreadsheet import SpreadsheetTool

    assert ChartTool(MagicMock()).health_check() is (
        find_spec("matplotlib") is not None)
    assert SpreadsheetTool(MagicMock()).health_check() is (
        find_spec("openpyxl") is not None)


def test_chart_prompting_rides_the_description():
    """The description is the ONLY text every model tier receives.

    ``tools_to_native_format`` sends name/description/parameters and DROPS
    model_hint; the hint is appended only for the "small" tier, and
    ``fallback_hint`` reaches the model only on the marker/proxy paths. So the
    when-to-use guidance has to be in the description or most models never see
    it — the bug this asserts against is guidance living only in model_hint.
    """
    from augmentum.modes.analytical.tool_calling import tools_to_native_format
    from augmentum.tools.artifact_chart import ChartTool

    tool = ChartTool(MagicMock())
    desc = tool.description.lower()
    assert "trend" in desc and "compar" in desc
    assert "do not need to be asked" in desc  # the proactive licence
    assert tool.model_hint  # small models get the extra nudge

    schema = tools_to_native_format([tool])[0]
    sent = schema["function"]["description"]
    assert "trend" in sent.lower()  # survives the wire format
    assert tool.model_hint not in sent  # documents WHY description must carry it


def test_spreadsheet_description_routes_away_from_prose_tables():
    """The observed miss is a model exporting an .xlsx for three rows the user
    only wanted to read. The boundary must be in the description, not the hint.
    """
    from augmentum.tools.artifact_spreadsheet import SpreadsheetTool

    tool = SpreadsheetTool(MagicMock())
    desc = tool.description.lower()
    assert "markdown table" in desc
    assert "create_chart" in desc  # points at the sibling for shape/trend data
    assert tool.model_hint


def test_offerable_is_lookups_plus_gated():
    offer = {c.name for c in SSOSOrchestrator.offerable_capabilities()}
    lookups = {c.name for c in SSOSOrchestrator.lookup_capabilities()}
    gated = {c.name for c in SSOSOrchestrator.gated_capabilities()}
    assert offer == lookups | gated
    assert lookups and gated and not (lookups & gated)


def test_match_trigger_recognizes_lookup_and_gated():
    lk = SSOSOrchestrator.match_trigger("[[tool:web_search]] llama news")
    assert lk and lk[0].kind == "lookup" and lk[1] == "llama news"
    gt = SSOSOrchestrator.match_trigger("[[tool:image_generation]] a red fox")
    assert gt and gt[0].kind == "gated" and gt[1] == "a red fox"


def test_parse_trigger_still_lookup_only():
    """Back-compat: the lookup-run path must NOT match a gated marker."""
    assert SSOSOrchestrator.parse_trigger("[[tool:image_generation]] a fox") is None
    assert SSOSOrchestrator.parse_trigger("[[tool:web_search]] x") is not None


def test_match_trigger_rejects_garbage_and_empty_args():
    assert SSOSOrchestrator.match_trigger("just a normal sentence") is None
    assert SSOSOrchestrator.match_trigger("[[tool:image_generation]]") is None  # no args
    assert SSOSOrchestrator.match_trigger("[[tool:nonexistent]] x") is None


def test_hint_advertises_gated_with_confirm_note():
    hint = _orch().build_soft_trigger_hint()
    assert "image_generation" in hint and "build_application" in hint
    assert "confirm before it runs" in hint
    assert "taps Accept" in hint  # the gated framing


@pytest.mark.asyncio
async def test_propose_gated_safe_without_app_state():
    """No app_state / user → graceful False, never raises."""
    cap = SSOSOrchestrator.gated_capabilities()[0]
    assert await _orch(app_state=None).propose_gated(cap, "a fox") is False
    # app_state present but no backend conn → still graceful
    bad_state = SimpleNamespace(state_manager=SimpleNamespace(backend=None))
    assert await _orch(app_state=bad_state).propose_gated(cap, "a fox") is False


@pytest.mark.asyncio
async def test_propose_gated_publishes_offer_end_to_end():
    """A gated capability surfaces a real offer through the dispatcher."""
    import augmentum.offers.catalog  # noqa: F401 — registers gated_tool
    from augmentum.state.backends.sqlite import SQLiteBackend
    backend = SQLiteBackend(":memory:")
    await backend.connect()
    try:
        app_state = SimpleNamespace(
            state_manager=SimpleNamespace(backend=backend),
            notification_hub=None,  # dispatcher persists the row regardless
        )
        orch = _orch(app_state=app_state)
        cap = next(c for c in SSOSOrchestrator.gated_capabilities()
                   if c.name == "image_generation")
        ok = await orch.propose_gated(cap, "a red fox at dusk", mode="passthrough")
        assert ok is True
    finally:
        await backend.close()


def _image_request(registry):
    request = MagicMock()
    request.app.state.tool_registry = registry
    request.scope = {"user": SimpleNamespace(id="u1")}
    return request


@pytest.mark.asyncio
async def test_gated_image_accept_runs_inline_and_returns_deliverable():
    """image_generation is fast (seconds) → Accept runs it INLINE and hands the
    finished image back as a ``deliverable`` so the chat can append it into the
    originating session, NOT detach-and-forget it into the gallery (the
    regression). The session_id must also reach the tool's _context."""
    import augmentum.offers.catalog  # noqa: F401 — registers gated_tool
    from augmentum.offers.catalog.base import get_entry

    entry = get_entry("gated_tool", "image_generation")
    assert entry is not None and entry.accept is not None

    captured: dict = {}

    async def _exec(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            success=True, metadata={"url": "/api/image/abc123"}, error="",
        )

    tool = MagicMock()
    tool.execute = _exec
    tool.long_running = False  # the property that picks the inline path
    registry = MagicMock()
    registry.get.return_value = tool

    payload = {"extra": {
        "args": "a red fox", "primary_arg": "prompt", "session_id": "sess-123",
    }}
    res = await entry.accept(payload, _image_request(registry))

    # Result is delivered inline, not "started" detached.
    assert res["ok"] is True
    assert "started" not in res
    assert res["deliverable"] == {
        "kind": "image", "url": "/api/image/abc123", "session_id": "sess-123",
    }
    # And the chat session reached the tool.
    assert captured.get("prompt") == "a red fox"
    ctx = captured.get("_context") or {}
    assert ctx.get("session_id") == "sess-123"
    assert ctx.get("user_id") == "u1"


@pytest.mark.asyncio
async def test_gated_image_accept_failure_points_at_gallery():
    """A failed inline generation returns ok=False (no phantom deliverable)."""
    import augmentum.offers.catalog  # noqa: F401 — registers gated_tool
    from augmentum.offers.catalog.base import get_entry

    entry = get_entry("gated_tool", "image_generation")

    async def _exec(**kwargs):
        return SimpleNamespace(success=False, metadata={}, error="boom")

    tool = MagicMock()
    tool.execute = _exec
    tool.long_running = False
    registry = MagicMock()
    registry.get.return_value = tool

    payload = {"extra": {"args": "x", "primary_arg": "prompt", "session_id": "s"}}
    res = await entry.accept(payload, _image_request(registry))
    assert res["ok"] is False
    assert "deliverable" not in res


@pytest.mark.asyncio
async def test_gated_longrunning_accept_stays_detached(monkeypatch):
    """build_application is multi-minute → Accept must NOT block: it stays a
    fire-and-forget background task and returns a 'started' ack (the project
    card / library is its own delivery surface)."""
    import augmentum.offers.catalog  # noqa: F401 — registers gated_tool
    from augmentum.offers.catalog.base import get_entry

    entry = get_entry("gated_tool", "build_application")
    assert entry is not None and entry.accept is not None

    captured: dict = {}

    async def _exec(**kwargs):
        captured.update(kwargs)

    tool = MagicMock()
    tool.execute = _exec
    tool.long_running = True  # → detached path
    registry = MagicMock()
    registry.get.return_value = tool

    request = MagicMock()
    request.app.state.tool_registry = registry
    request.scope = {"user": SimpleNamespace(id="u1")}

    spawned: list = []
    real_create_task = asyncio.create_task

    def _capture(coro):
        task = real_create_task(coro)
        spawned.append(task)
        return task

    monkeypatch.setattr(asyncio, "create_task", _capture)

    payload = {"extra": {
        "args": "a todo app", "primary_arg": "description", "session_id": "s1",
    }}
    res = await entry.accept(payload, request)
    assert res["ok"] is True and res["started"] is True
    assert "deliverable" not in res
    await asyncio.gather(*spawned)
    assert captured.get("description") == "a todo app"
    assert (captured.get("_context") or {}).get("session_id") == "s1"
