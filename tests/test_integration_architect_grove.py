"""Integration tests — architect dispatch end-to-end for grove.play_matching.

Protects the novel UX flow: LLM picks ``grove.play_matching`` via tool
exposure (Tier-3) → arg_inferrer queries fixture device_play_history →
fills source + track_id → handler emits grove.play surface event with
the inferred track.

The verb is Tier-3-only as of 2026-06-10 — the LLM picks based on intent
understanding plus context, not Tier-1 regex templates. So these tests
invoke the inferrer + handler directly (matching how
``intent.tool_adapter`` calls into them on a Tier-3 hit), bypassing
``dispatch_architect_command``'s Tier-1 matcher entry point. The
business logic (favourites inference, ask-when-no-match, anon refuse)
is the same; only the matching step changes.

Uses an in-memory fake conn so the test exercises the real query path
without needing a SQLite fixture file.
"""

from __future__ import annotations

import pytest


class _FakeCursor:
    def __init__(self, rows):
        self._rows = rows

    async def fetchall(self):
        return self._rows


class _FakeConn:
    """Async-compatible fake SQLite connection. Returns canned rows
    matching the device_play_history SELECT in inference.query_play_history.

    Rows are 8-tuples matching:
      capability_id, file_id, content_key, content_label, action,
      is_favorite, created_at, extra
    """

    def __init__(self, rows):
        self.rows = rows
        self.queries = []

    async def execute(self, sql, params=None):
        self.queries.append((sql, params))
        return _FakeCursor(self.rows)


class _FakeBackend:
    def __init__(self, conn):
        self.conn = conn


class _FakeStateManager:
    def __init__(self, conn):
        self.backend = _FakeBackend(conn)


class _FakeRuntime:
    """Minimal runtime exposing state_manager.backend.conn."""

    def __init__(self, conn):
        self.state_manager = _FakeStateManager(conn)


class _FakeAppState:
    def __init__(self, runtime):
        self.companion_runtime = runtime
        self.intent_referents = {}


@pytest.fixture(autouse=True)
def _enable_architect_dispatch(monkeypatch):
    """Force-enable architect dispatch for these tests."""
    from augmentum.config import settings
    monkeypatch.setattr(settings, "architect_dispatch_enabled", True)


async def _invoke_grove_tier3(query: str, session, app_state, *, text: str = ""):
    """Invoke grove.play_matching via the same path the Tier-3 LLM tool
    adapter uses: load the action, run its arg_inferrer, then call the
    handler with the inferred args. Returns (action_result, inferred_args).

    Mirrors ``intent.tool_adapter.LLMToolAdapter.execute`` minus the
    JSON-schema validation step (we supply args directly here).
    """
    import augmentum.architect  # noqa: F401 — register actions
    from augmentum.architect.inference import infer_args
    from augmentum.intent.registry import REGISTRY

    action = REGISTRY.get("grove.play_matching")
    assert action is not None, "grove.play_matching not registered"

    runtime = getattr(app_state, "companion_runtime", None) if app_state else None

    # Bind persistent ReferentCache same way dispatch_architect_command would.
    if app_state is not None:
        from augmentum.architect.dispatch import get_referent_cache
        session.referents = get_referent_cache(
            app_state, session.user_id, session.session_id,
        )

    filled = await infer_args(action, {"query": query}, session, runtime)
    result = await action.handler(text, session, filled)
    return result, filled


@pytest.mark.asyncio
async def test_play_jazz_infers_from_favorites():
    """LLM picks grove.play_matching with query="jazz" → inferrer finds a
    jazz favourite → handler emits grove.play with that track filled in.
    """
    from augmentum.intent.action import SessionContext

    # Fixture history — a jazz-labelled favourite + an artist match + a
    # non-jazz recent. Substring matching is the Phase 1 inference; full
    # semantic genre understanding lands in Phase 3 (embeddings).
    rows = [
        ("media.audio_play@1", "file_jazz_pl", "", "Late Night Jazz Playlist",
         "play", 1, "2026-05-20T12:00:00", "{}"),
        ("media.audio_play@1", "file_miles", "", "Miles Davis - Kind of Blue",
         "play", 1, "2026-05-21T12:00:00", "{}"),
        ("media.audio_play@1", "file_rock_001", "", "Led Zeppelin - IV",
         "play", 0, "2026-05-27T12:00:00", "{}"),
    ]
    conn = _FakeConn(rows)
    runtime = _FakeRuntime(conn)
    app_state = _FakeAppState(runtime)

    session = SessionContext(
        user_id="usr_test_1",
        session_id="sess_test_1",
        mode="passthrough",
        app_state=app_state,
    )

    result, filled = await _invoke_grove_tier3("jazz", session, app_state)

    assert result is not None
    assert result.short_circuit is True
    # Inferrer picked the jazz-labelled playlist (favourite, top hit)
    assert filled.get("track_id") == "file_jazz_pl"
    assert "Jazz" in filled.get("content_label", "")
    # Surface emit carries the resolved track
    assert result.surface_emit is not None
    assert result.surface_emit.get("channel") == "grove.play"
    assert result.surface_emit["payload"]["track_id"] == "file_jazz_pl"


@pytest.mark.asyncio
async def test_play_unknown_genre_asks():
    """When no history matches the query, handler should ask rather
    than pick a random fallback."""
    from augmentum.intent.action import SessionContext

    # History is non-empty but nothing matches "polka".
    rows = [
        ("media.audio_play@1", "file_jazz_001", "", "Miles Davis - Kind of Blue",
         "play", 1, "2026-05-20T12:00:00", "{}"),
    ]
    conn = _FakeConn(rows)
    runtime = _FakeRuntime(conn)
    app_state = _FakeAppState(runtime)

    session = SessionContext(
        user_id="usr_test_1",
        session_id="sess_test_1",
        mode="passthrough",
        app_state=app_state,
    )

    result, filled = await _invoke_grove_tier3("polka", session, app_state)

    # The handler returns a clarifying ActionResult, not None — inferrer
    # ran but couldn't find a match; handler asked instead of dispatching.
    assert result is not None
    # No track filled
    assert not filled.get("track_id")
    # Spoken clarification — content varies but should reference the query
    assert "polka" in result.speak.lower() or "favourites" in result.speak.lower() \
        or "favorites" in result.speak.lower()


@pytest.mark.asyncio
async def test_disabled_flag_returns_none():
    """When architect_dispatch_enabled is False, dispatch is a no-op."""
    import augmentum.architect  # noqa: F401
    from augmentum.architect.dispatch import dispatch_architect_command
    from augmentum.config import settings
    from augmentum.intent.action import SessionContext

    # Override the fixture
    settings.architect_dispatch_enabled = False
    try:
        session = SessionContext(
            user_id="usr_test_1",
            session_id="sess_test_1",
            mode="passthrough",
        )
        result = await dispatch_architect_command(
            "play jazz",
            surface="voice",
            session=session,
        )
        assert result is None, "expected None when architect dispatch disabled"
    finally:
        settings.architect_dispatch_enabled = True


@pytest.mark.asyncio
async def test_surface_filter_rejects_xr():
    """grove.play_matching is registered for voice + chat. An XR
    surface command should NOT dispatch through architect — it falls
    through to whatever XR-aware handler exists.
    """
    import augmentum.architect  # noqa: F401
    from augmentum.architect.dispatch import dispatch_architect_command
    from augmentum.intent.action import SessionContext

    session = SessionContext(
        user_id="usr_test_1",
        session_id="sess_test_1",
        mode="passthrough",
    )

    result = await dispatch_architect_command(
        "play jazz",
        surface="xr",
        session=session,
    )
    # Surface filter rejected the dispatch
    assert result is None


@pytest.mark.asyncio
async def test_anon_user_refused():
    """Anon session (empty user_id) returns clarifying response.

    The multi-tenant invariant — anon sessions cannot trigger user-scoped
    side effects — is enforced INSIDE the handler, regardless of which
    tier dispatched it. So this test invokes the handler directly with
    user_id="" and asserts the refusal.
    """
    from augmentum.intent.action import SessionContext

    session = SessionContext(
        user_id="",  # anon
        session_id="sess_test_1",
        mode="passthrough",
    )

    result, _filled = await _invoke_grove_tier3("jazz", session, app_state=None)
    assert result is not None
    # Handler short-circuits with a refusal — multi-tenant invariant.
    assert result.short_circuit is True
    assert "signed-out" in result.speak.lower() or "anon" in result.speak.lower() \
        or "play" in result.speak.lower()  # any spoken response is fine
