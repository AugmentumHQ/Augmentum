"""Integration tests for the coder.delegate verb + workspace resolver.

What we're protecting:
  * Registration shape — tier-3 only, costly, not initiatable, required=[prompt]
  * Resolver — confident single match vs. ambiguous offer vs. no-workspaces
  * Handler — anon refusal; confident match enqueues a coder_background_run;
    ambiguity parks candidates + pending_intent and emits companion.candidates
  * Never-auto-select — the offer always includes a "New workspace" tile and
    carries the generic accept-arg metadata so a spoken pick fills workspace_id
  * Model — the run uses the PRIMARY chat model, never a companion utility model
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

# ── Fakes ───────────────────────────────────────────────────────────────────

@dataclass
class _WS:
    id: str
    name: str
    git_url: str = ""
    last_active: float = 0.0
    kind: str = "regular"


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    async def fetchall(self):
        return self._rows


class _FakeConn:
    def __init__(self, owned_ids):
        self._owned = owned_ids

    async def execute(self, _sql, _params=None):
        return _FakeCursor([(i,) for i in self._owned])


class _FakeBackend:
    def __init__(self, conn):
        self.conn = conn


class _FakeStateManager:
    def __init__(self, conn):
        self.backend = _FakeBackend(conn)


class _FakeContainerManager:
    def __init__(self, workspaces):
        self._ws = workspaces

    async def list_workspaces(self):
        return list(self._ws)


class _FakeJobsStore:
    def __init__(self):
        self.created = []

    async def create(self, *, user_id, job_type, payload, priority, max_attempts):
        self.created.append({
            "user_id": user_id, "job_type": job_type, "payload": payload,
            "priority": priority, "max_attempts": max_attempts,
        })
        return f"job_{len(self.created)}"


class _FakeJobRunner:
    def __init__(self):
        self.woken = 0

    def wake(self):
        self.woken += 1


class _AppState:
    def __init__(self, workspaces, owned_ids):
        self.container_manager = _FakeContainerManager(workspaces)
        self.state_manager = _FakeStateManager(_FakeConn(owned_ids))
        self.jobs_store = _FakeJobsStore()
        self.job_runner = _FakeJobRunner()


def _ctx(app, user="u1", sess="s1"):
    from augmentum.intent.action import SessionContext
    return SessionContext(user_id=user, session_id=sess, mode="becca_direct", app_state=app)


@pytest.fixture(autouse=True)
def _primary_model(monkeypatch):
    """Give the server a primary chat model + no heavyweight by default."""
    from augmentum.config import settings
    monkeypatch.setattr(settings, "primary_chat_model", "qwen3.6-35b", raising=False)
    monkeypatch.setattr(settings, "heavyweight_model", "", raising=False)
    return settings


# ── Registration ────────────────────────────────────────────────────────────

def test_registration_shape():
    import augmentum.intent  # noqa: F401 — ensure builtins imported
    from augmentum.intent.registry import REGISTRY

    action = REGISTRY.get("coder.delegate")
    assert action is not None
    assert action.fanout.tier3 is True
    assert action.fanout.tier1 is False and action.fanout.tier2 is False
    assert action.stakes == "costly"
    assert action.companion_initiatable is False
    assert action.required_args == ["prompt"]


def test_in_voice_costly_bucket():
    from augmentum.intent.manifest import VOICE_TOOLS_COSTLY, all_voice_tools
    assert "coder.delegate" in VOICE_TOOLS_COSTLY
    assert "coder.delegate" in all_voice_tools()


# ── Resolver ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_resolver_confident_single_match():
    from augmentum.coder.workspace_resolver import resolve_workspace
    app = _AppState(
        [_WS("ws_ui", "augmentum-ui"), _WS("ws_api", "billing-api")],
        {"ws_ui", "ws_api"},
    )
    res = await resolve_workspace(app, user_id="u1", request_text="add dark mode to the ui")
    assert res.decision == "confident"
    assert res.top.workspace_id == "ws_ui"


@pytest.mark.asyncio
async def test_resolver_offers_when_ambiguous():
    from augmentum.coder.workspace_resolver import resolve_workspace
    app = _AppState(
        [_WS("ws_a", "alpha"), _WS("ws_b", "beta")],
        {"ws_a", "ws_b"},
    )
    # No token overlap with either name → ambiguous → offer.
    res = await resolve_workspace(app, user_id="u1", request_text="build a login screen")
    assert res.decision == "offer"
    assert len(res.candidates) >= 2


@pytest.mark.asyncio
async def test_resolver_none_when_no_workspaces():
    from augmentum.coder.workspace_resolver import resolve_workspace
    app = _AppState([], set())
    res = await resolve_workspace(app, user_id="u1", request_text="build something")
    assert res.decision == "none"


@pytest.mark.asyncio
async def test_resolver_only_returns_owned_regular():
    from augmentum.coder.workspace_resolver import resolve_workspace
    app = _AppState(
        [_WS("ws_ui", "augmentum-ui"), _WS("ws_bf", "augmentum-ui-audit", kind="bug_finder")],
        {"ws_ui"},  # ws_bf not owned
    )
    res = await resolve_workspace(app, user_id="u1", request_text="augmentum-ui tweak")
    ids = {c.workspace_id for c in res.candidates}
    assert "ws_bf" not in ids


# ── Handler ─────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_handler_refuses_anon():
    from augmentum.intent.registry import REGISTRY
    app = _AppState([_WS("ws_ui", "augmentum-ui")], {"ws_ui"})
    action = REGISTRY.get("coder.delegate")
    res = await action.handler("build x", _ctx(app, user=""), {"prompt": "build x"})
    assert res.fulfilled is False
    assert app.jobs_store.created == []


@pytest.mark.asyncio
async def test_handler_confident_enqueues_primary_model():
    from augmentum.intent.registry import REGISTRY
    app = _AppState([_WS("ws_ui", "augmentum-ui")], {"ws_ui"})
    action = REGISTRY.get("coder.delegate")
    res = await action.handler(
        "add dark mode to augmentum-ui", _ctx(app),
        {"prompt": "add dark mode to augmentum-ui"},
    )
    assert res.fulfilled is True
    assert len(app.jobs_store.created) == 1
    job = app.jobs_store.created[0]
    assert job["job_type"] == "coder_background_run"
    assert job["payload"]["workspace_id"] == "ws_ui"
    assert job["payload"]["model"] == "qwen3.6-35b"  # primary, not a utility model
    assert app.job_runner.woken == 1
    # Acknowledgment channel so a voice pick dismisses the candidate dock.
    assert res.surface_emit["channel"] == "coder.delegate"


@pytest.mark.asyncio
async def test_handler_records_trail_for_take_me_there():
    from augmentum.intent.registry import REGISTRY
    app = _AppState([_WS("ws_ui", "augmentum-ui")], {"ws_ui"})
    ctx = _ctx(app)
    action = REGISTRY.get("coder.delegate")
    await action.handler(
        "add dark mode to augmentum-ui", ctx,
        {"prompt": "add dark mode to augmentum-ui"},
    )
    trail = ctx.referents.trail
    assert trail and trail[-1]["kind"] == "coder_run"
    assert trail[-1]["ref"] == "ws_ui"  # jump target for "take me there"


@pytest.mark.asyncio
async def test_handler_ambiguous_parks_and_offers():
    from augmentum.intent.registry import REGISTRY
    app = _AppState([_WS("ws_a", "alpha"), _WS("ws_b", "beta")], {"ws_a", "ws_b"})
    ctx = _ctx(app)
    action = REGISTRY.get("coder.delegate")
    res = await action.handler("build a login screen", ctx, {"prompt": "build a login screen"})

    assert res.fulfilled is False
    assert app.jobs_store.created == []  # nothing dispatched until the user picks
    assert res.surface_emit["channel"] == "companion.candidates"
    payload = res.surface_emit["payload"]
    assert payload["intent"] == "coder.delegate"
    assert payload["delegation"]["model"] == "qwen3.6-35b"
    # Always includes a New-workspace tile (never force an existing pick).
    assert any(c.get("is_new") for c in payload["candidates"])

    refs = ctx.referents
    assert refs.pending_intent["action_id"] == "coder.delegate"
    assert refs.pending_intent["missing"] == ["workspace_id"]
    # Generic accept metadata so the router fills workspace_id, not file_id.
    assert refs.pending_candidates_intent == "coder.delegate"
    assert refs.pending_candidates_id_field == "workspace_id"


@pytest.mark.asyncio
async def test_handler_explicit_pick_enqueues():
    from augmentum.intent.registry import REGISTRY
    app = _AppState([_WS("ws_a", "alpha"), _WS("ws_b", "beta")], {"ws_a", "ws_b"})
    action = REGISTRY.get("coder.delegate")
    res = await action.handler(
        "", _ctx(app),
        {"prompt": "build a login screen", "workspace_id": "ws_b"},
    )
    assert res.fulfilled is True
    assert app.jobs_store.created[0]["payload"]["workspace_id"] == "ws_b"


@pytest.mark.asyncio
async def test_handler_new_workspace_routes_to_create():
    from augmentum.intent.registry import REGISTRY
    app = _AppState([_WS("ws_a", "alpha")], {"ws_a"})
    action = REGISTRY.get("coder.delegate")
    res = await action.handler(
        "", _ctx(app),
        {"prompt": "build a login screen", "workspace_id": "__new__"},
    )
    assert res.surface_emit["channel"] == "coder.new_workspace"
    assert app.jobs_store.created == []  # creation happens in the UI, not here


@pytest.mark.asyncio
async def test_handler_heavyweight_tier_uses_pinned_model(monkeypatch):
    from augmentum.config import settings
    monkeypatch.setattr(settings, "heavyweight_model", "deepseek-v4-pro", raising=False)
    from augmentum.intent.registry import REGISTRY
    app = _AppState([_WS("ws_ui", "augmentum-ui")], {"ws_ui"})
    action = REGISTRY.get("coder.delegate")
    res = await action.handler(
        "add dark mode to augmentum-ui", _ctx(app),
        {"prompt": "add dark mode to augmentum-ui", "tier": "heavyweight"},
    )
    assert res.fulfilled is True
    assert app.jobs_store.created[0]["payload"]["model"] == "deepseek-v4-pro"
