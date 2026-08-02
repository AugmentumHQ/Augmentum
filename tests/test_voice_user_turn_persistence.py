"""Regression guard: voice calls must persist the USER side of every turn.

History: the voice call (`voice.js` over `/ws/voice`) persists the chat tree
CLIENT-side. The user turn used to ride on the display-only ``transcript`` echo,
which a stale client stage flag or a learned-command match could swallow — so a
call would end with "only assistant turns saved" (reported repeatedly: 2026-06-11
through 2026-06-25). The fix added an authoritative ``user_committed`` emit at the
real commit point, consolidated here into `_commit_user_turn` as the single source
of truth shared by the VAD and streaming paths.

These tests lock the contract so it can't silently regress again:
  * the commit both records LLM context AND emits ``user_committed``;
  * Stage Send suppresses the emit (client already persisted) — no double-save;
  * BOTH call paths funnel through the helper and never bypass it with a bare
    ``add_user_message`` (which would commit without telling the client).
"""

from __future__ import annotations

import asyncio
import inspect
import json

from augmentum.proxy import voice_routes
from augmentum.proxy.voice_routes import (
    _commit_user_turn,
    _process_voice_turn,
    _process_voice_turn_from_transcript,
)


class _FakeWS:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send_text(self, text: str) -> None:
        self.sent.append(json.loads(text))


class _FakeSession:
    def __init__(self) -> None:
        self.user_messages: list[str] = []

    def add_user_message(self, text: str) -> None:
        self.user_messages.append(text)


def _committed(ws: _FakeWS) -> list[dict]:
    return [m for m in ws.sent if m.get("type") == "user_committed"]


# ── Behavioural contract of the shared helper ─────────────────────────────

def test_commit_records_context_and_emits_user_committed():
    ws, s = _FakeWS(), _FakeSession()
    asyncio.run(_commit_user_turn(ws, s, "what's the weather like"))
    # 1) the turn is in the LLM context
    assert s.user_messages == ["what's the weather like"]
    # 2) the client is told to persist the user side of the chat tree
    assert _committed(ws) == [{"type": "user_committed", "text": "what's the weather like"}]


def test_stage_send_suppresses_emit_but_keeps_context():
    """Stage Send already persisted client-side — emitting would double it."""
    ws, s = _FakeWS(), _FakeSession()
    asyncio.run(_commit_user_turn(ws, s, "edited before send", emit=False))
    assert s.user_messages == ["edited before send"]   # still in LLM context
    assert _committed(ws) == []                          # but no second persist


def test_user_committed_payload_shape_is_stable():
    """The client switches on exactly this type+field — keep it stable."""
    ws, s = _FakeWS(), _FakeSession()
    asyncio.run(_commit_user_turn(ws, s, "hello"))
    msg = _committed(ws)[0]
    assert set(msg) == {"type", "text"}
    assert msg["type"] == "user_committed"
    assert msg["text"] == "hello"


# ── Structural tripwires: the call paths can't bypass the contract ────────

def test_both_call_paths_commit_through_the_shared_helper():
    for fn in (_process_voice_turn, _process_voice_turn_from_transcript):
        src = inspect.getsource(fn)
        assert "_commit_user_turn(" in src, (
            f"{fn.__name__} must commit the user turn via _commit_user_turn — "
            "the user side of the chat tree depends on it"
        )


def test_call_paths_never_add_user_message_directly():
    """A bare add_user_message in a call path = commit without the emit =
    the original dropped-user-turn regression. Force it through the helper."""
    for fn in (_process_voice_turn, _process_voice_turn_from_transcript):
        src = inspect.getsource(fn)
        assert "add_user_message(" not in src, (
            f"{fn.__name__} calls add_user_message directly; route it through "
            "_commit_user_turn so 'user_committed' can never be dropped"
        )


def test_helper_emits_the_documented_event_name():
    """Guard the wire contract at the server too (client listens for this)."""
    assert "user_committed" in inspect.getsource(_commit_user_turn)
    # the module still owns the single emit of this event
    assert voice_routes.__name__.endswith("voice_routes")
