"""End-of-turn invariants for the Cardsmith /turn route.

These tests lock in the contracts the streaming generator must uphold so a
future refactor doesn't silently re-introduce a stranded lock or a missing
``[DONE]``:

  - Lock acquired across the turn, released on completion (so the next
    /turn for the same session can run immediately).
  - Visible reply persisted on ``sess.messages`` so the model sees its own
    last turn on the next /turn.
  - SSE stream terminates with ``data: [DONE]``.
  - Concurrent /turn while another is in flight returns 409 (not queued).
  - Unknown / cross-tenant session ids are 404.
  - /cancel does not drop a session belonging to a different tenant.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import pytest

from augmentum.models.base import InternalStreamChunk

# ── Shared helpers ────────────────────────────────────────────────────────


def _wire_resolver(
    app,
    mock_backend,
    deltas: list[str],
    *,
    thinking: list[str] | None = None,
) -> None:
    """Make ``provider_registry.resolve_model_for_role`` return a backend
    whose ``chat_stream`` yields ``thinking`` deltas (if any), then the
    visible ``deltas``, then a done-chunk.
    """
    thinking_chunks = list(thinking or [])

    async def _stream(_req) -> AsyncIterator[InternalStreamChunk]:
        for t in thinking_chunks:
            yield InternalStreamChunk(content_delta="", thinking_delta=t, done=False)
        for d in deltas:
            yield InternalStreamChunk(content_delta=d, done=False)
        yield InternalStreamChunk(content_delta="", done=True, finish_reason="stop")

    mock_backend.chat_stream = _stream
    app.state.provider_registry.resolve_model_for_role = AsyncMock(
        return_value=(mock_backend, "test-model")
    )


def _start_session(client, **overrides) -> str:
    body = {"card_type": "single", "source": "describe", "seed_prompt": "A medic."}
    body.update(overrides)
    resp = client.post("/api/characters/cardsmith/start", json=body)
    assert resp.status_code == 200, resp.text
    return resp.json()["session_id"]


def _consume_sse(client, session_id: str, user_message: str = "") -> bytes:
    """Drive a /turn request and return the full SSE response body."""
    with client.stream(
        "POST",
        "/api/characters/cardsmith/turn",
        json={"session_id": session_id, "user_message": user_message},
    ) as resp:
        assert resp.status_code == 200, resp.read()
        return b"".join(resp.iter_bytes())


# ── End-of-turn invariants ────────────────────────────────────────────────


def test_turn_emits_done_and_releases_lock(app, client, mock_backend):
    """Happy path: turn streams to completion, lock is free afterwards, and
    the assistant reply lands on sess.messages."""
    from augmentum.modes.narrative.cardsmith import get_session

    _wire_resolver(app, mock_backend, ["Hi! ", "What's their name?"])
    session_id = _start_session(client)

    body = _consume_sse(client, session_id, user_message="")
    assert b"data: [DONE]" in body, "stream must terminate with [DONE]"

    sess = get_session(session_id, user_id="usr_test")
    assert sess is not None, "session should still exist after a non-finalizing turn"
    assert not sess.lock.locked(), "lock must be released at end of turn"

    roles = [m["role"] for m in sess.messages]
    assert "user" in roles and "assistant" in roles
    assert sess.messages[-1]["role"] == "assistant"
    assert "Hi! What's their name?" in sess.messages[-1]["content"]


def test_two_sequential_turns_run_back_to_back(app, client, mock_backend):
    """Regression: a stranded lock from turn N would 409 turn N+1."""
    from augmentum.modes.narrative.cardsmith import get_session

    _wire_resolver(app, mock_backend, ["Got it. "])
    session_id = _start_session(client)

    _consume_sse(client, session_id, user_message="")
    _consume_sse(client, session_id, user_message="Her name is Lyra.")

    sess = get_session(session_id, user_id="usr_test")
    assert sess is not None
    assert not sess.lock.locked()
    user_turns = [m for m in sess.messages if m["role"] == "user"]
    assert len(user_turns) == 2  # seed + "Her name is Lyra."


def test_concurrent_turn_returns_409(app, client, mock_backend):
    """A concurrent /turn while another holds the lock must 409 — not queue.

    We simulate the in-flight turn by directly acquiring the session's lock
    on a private event loop (the route's ``locked()`` check is sync, so it
    sees the held state regardless of which loop owns it).
    """
    import asyncio

    from augmentum.modes.narrative.cardsmith import get_or_create_session

    _wire_resolver(app, mock_backend, ["Hello"])
    sess = get_or_create_session(
        user_id="usr_test", card_type="single", source="describe",
    )

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(sess.lock.acquire())
        try:
            resp = client.post(
                "/api/characters/cardsmith/turn",
                json={"session_id": sess.session_id, "user_message": "hi"},
            )
            assert resp.status_code == 409
            assert "in progress" in resp.json()["error"].lower()
        finally:
            sess.lock.release()
    finally:
        loop.close()


def test_turn_404_for_unknown_session(client):
    resp = client.post(
        "/api/characters/cardsmith/turn",
        json={"session_id": "cs_does_not_exist", "user_message": "hi"},
    )
    assert resp.status_code == 404


def test_turn_404_for_other_users_session(app, client, mock_backend):
    """A session created by user A must not be drivable by user B."""
    from augmentum.modes.narrative.cardsmith import get_or_create_session

    _wire_resolver(app, mock_backend, ["Hi"])
    other_sess = get_or_create_session(
        user_id="usr_other", card_type="single", source="describe",
    )
    resp = client.post(
        "/api/characters/cardsmith/turn",
        json={"session_id": other_sess.session_id, "user_message": "hi"},
    )
    assert resp.status_code == 404


def test_turn_releases_lock_when_provider_unavailable(app, client):
    """If the provider registry has no backends, /turn 503s — and the lock
    must be released on the way out so the session isn't stranded."""
    from augmentum.modes.narrative.cardsmith import get_session

    # No resolver wiring — we want resolve_model_for_role to error. Easier
    # to just empty the backends dict so the early-503 branch fires.
    app.state.provider_registry.backends = {}

    session_id = _start_session(client)
    resp = client.post(
        "/api/characters/cardsmith/turn",
        json={"session_id": session_id, "user_message": "hi"},
    )
    assert resp.status_code == 503

    sess = get_session(session_id, user_id="usr_test")
    assert sess is not None
    assert not sess.lock.locked(), "lock must be released on early-503 paths too"


def test_cancel_does_not_drop_other_users_session(app, client, mock_backend):
    """Cross-tenant /cancel must be a no-op (returns ok but session stays)."""
    from augmentum.modes.narrative.cardsmith import get_or_create_session, get_session

    other_sess = get_or_create_session(
        user_id="usr_other", card_type="single", source="describe",
    )
    resp = client.post(
        "/api/characters/cardsmith/cancel",
        json={"session_id": other_sess.session_id},
    )
    assert resp.status_code == 200  # idempotent, doesn't leak existence

    still_there = get_session(other_sess.session_id, user_id="usr_other")
    assert still_there is not None, "other tenant's session must not be dropped"


def test_turn_forwards_thinking_deltas_as_sse(app, client, mock_backend):
    """Reasoning models silently burned time on turn 2+ before this fix —
    no progress was visible until visible content arrived. The route now
    forwards ``thinking_delta`` chunks as ``{"type":"thinking",...}`` SSE
    events so the UI can show reasoning activity."""
    _wire_resolver(
        app,
        mock_backend,
        deltas=["Got it. "],
        thinking=["Considering tone... ", "deciding on a name..."],
    )
    session_id = _start_session(client)
    body = _consume_sse(client, session_id, user_message="").decode()

    assert '"type": "thinking"' in body
    assert "Considering tone..." in body
    # Visible content also lands.
    assert '"type": "delta"' in body
    assert "Got it." in body
    assert "data: [DONE]" in body


def test_cancel_drops_own_session(app, client, mock_backend):
    """Sanity: when the uid matches, /cancel does drop the session."""
    from augmentum.modes.narrative.cardsmith import get_session

    session_id = _start_session(client)
    resp = client.post(
        "/api/characters/cardsmith/cancel",
        json={"session_id": session_id},
    )
    assert resp.status_code == 200
    assert get_session(session_id, user_id="usr_test") is None


# ── pytest entry sanity ───────────────────────────────────────────────────

if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
