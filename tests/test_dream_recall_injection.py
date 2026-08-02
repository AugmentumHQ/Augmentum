"""Tests for dream recall injection into chat prompts.

Covers ``resolve_and_inject_dream_context`` and
``apply_dream_injection_to_request`` in augmentum.memory.integration.

Why this file exists: the route-level inline version shipped with three
bugs since Stage B multi-tenancy — persona_id/user_id positional
swap, missing per-user opt-in gate, and ignored per-user recall
toggles. This suite pins the corrected behaviour so a refactor can't
silently regress the "dreams reach chat prompts" contract.
"""
from __future__ import annotations

from types import SimpleNamespace

import aiosqlite

from augmentum.memory.integration import (
    apply_dream_injection_to_request,
    resolve_and_inject_dream_context,
)
from augmentum.models.base import InternalChatRequest, Message
from augmentum.state.settings_store import SettingsStore

# ---------------------------------------------------------------------------
# Fixtures & fakes
# ---------------------------------------------------------------------------


async def _make_store() -> tuple[aiosqlite.Connection, SettingsStore]:
    conn = await aiosqlite.connect(":memory:")
    await conn.execute(
        "CREATE TABLE app_settings ("
        "  key TEXT PRIMARY KEY, value TEXT NOT NULL,"
        "  updated_at TEXT DEFAULT (datetime('now')))"
    )
    await conn.execute(
        "CREATE TABLE user_settings ("
        "  user_id TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,"
        "  updated_at TEXT DEFAULT (datetime('now')),"
        "  PRIMARY KEY (user_id, key))"
    )
    await conn.commit()
    return conn, SettingsStore(conn)


class _FakePortrait:
    def __init__(self, voice="V", threads="T", impressions="I"):
        self.voice_notes = voice
        self.active_threads = threads
        self.impressions = impressions


class _FakeEntry:
    def __init__(self, content: str, expires_at: str | None = None):
        self.content = content
        self.expires_at = expires_at


def _app_state(
    *, store=None, portrait=None, entries=None,
    portrait_mgr=True, journal=True,
):
    """Build a fake ``app.state`` with the dream subsystem wired.

    Records get_current/list_entries calls on .calls so tests can verify
    the persona_id / user_id contract.
    """
    calls: dict = {"get_current": [], "list_entries": []}

    async def _get_current(persona_id, *, user_id=""):
        calls["get_current"].append({"persona_id": persona_id, "user_id": user_id})
        return portrait

    async def _list_entries(persona_id, *, limit=50, user_id="", **_):
        calls["list_entries"].append({
            "persona_id": persona_id, "user_id": user_id, "limit": limit,
        })
        return (entries or []), len(entries or [])

    state = SimpleNamespace(
        settings_store=store,
        dream_portrait_manager=(
            SimpleNamespace(get_current=_get_current) if portrait_mgr else None
        ),
        dream_journal=(
            SimpleNamespace(list_entries=_list_entries) if journal else None
        ),
    )
    state.calls = calls  # type: ignore[attr-defined]
    return state


# ---------------------------------------------------------------------------
# resolve_and_inject_dream_context — the policy layer
# ---------------------------------------------------------------------------


class TestResolveAndInjectPolicy:

    async def test_noop_when_subsystem_not_running(self):
        """No portrait_mgr and no dream_journal on state → nothing happens."""
        state = _app_state(portrait_mgr=False, journal=False)
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        await resolve_and_inject_dream_context(messages, state, "alice")
        assert messages[0]["content"] == "sys"

    async def test_noop_when_user_opted_out(self):
        """Even if portrait exists, opted-out users get no injection."""
        conn, store = await _make_store()
        # alice has dream OFF
        await store.set_user("alice", "ui.dreamEnabled", "false")
        state = _app_state(store=store, portrait=_FakePortrait())
        messages = [{"role": "system", "content": "sys"}]
        await resolve_and_inject_dream_context(messages, state, "alice")
        assert messages[0]["content"] == "sys"
        assert state.calls["get_current"] == [], \
            "portrait should not even be fetched for opted-out users"
        await conn.close()

    async def test_injects_when_user_opted_in(self):
        conn, store = await _make_store()
        await store.set_user("alice", "ui.dreamEnabled", "true")
        state = _app_state(
            store=store, portrait=_FakePortrait(voice="curious"),
        )
        messages = [{"role": "system", "content": "sys"}]
        await resolve_and_inject_dream_context(messages, state, "alice")
        assert "<evolved_self>" in messages[0]["content"]
        assert "curious" in messages[0]["content"]

    async def test_persona_id_is_default_not_user_id(self):
        """The canonical bug: passing user_id positionally matches
        persona_id. Verify the call uses persona_id='default'."""
        conn, store = await _make_store()
        await store.set_user("alice", "ui.dreamEnabled", "true")
        state = _app_state(
            store=store, portrait=_FakePortrait(),
            entries=[_FakeEntry("r1")],
        )
        await resolve_and_inject_dream_context(
            [{"role": "system", "content": "sys"}], state, "alice",
        )
        assert state.calls["get_current"] == [
            {"persona_id": "default", "user_id": "alice"},
        ]
        assert state.calls["list_entries"][0]["persona_id"] == "default"
        assert state.calls["list_entries"][0]["user_id"] == "alice"
        await conn.close()

    async def test_per_user_recall_disabled_skips_entries(self):
        """Portrait still injected; recent_reflections block suppressed."""
        conn, store = await _make_store()
        await store.set_user("alice", "ui.dreamEnabled", "true")
        await store.set_user("alice", "ui.dreamRecallEnabled", "false")
        state = _app_state(
            store=store, portrait=_FakePortrait(),
            entries=[_FakeEntry("should not appear")],
        )
        messages = [{"role": "system", "content": "sys"}]
        await resolve_and_inject_dream_context(messages, state, "alice")
        assert "<evolved_self>" in messages[0]["content"]
        assert "recent_reflections" not in messages[0]["content"]
        assert state.calls["list_entries"] == [], \
            "entries must not be fetched when recall disabled"
        await conn.close()

    async def test_per_user_recall_limit_honoured(self):
        conn, store = await _make_store()
        await store.set_user("alice", "ui.dreamEnabled", "true")
        await store.set_user("alice", "ui.dreamRecallLimit", "7")
        state = _app_state(
            store=store,
            entries=[_FakeEntry("r")],
        )
        await resolve_and_inject_dream_context(
            [{"role": "system", "content": "sys"}], state, "alice",
        )
        assert state.calls["list_entries"][0]["limit"] == 7
        await conn.close()

    async def test_malformed_recall_limit_falls_back(self):
        conn, store = await _make_store()
        await store.set_user("alice", "ui.dreamEnabled", "true")
        await store.set_user("alice", "ui.dreamRecallLimit", "not-a-number")
        state = _app_state(store=store, entries=[_FakeEntry("r")])
        # Should not raise; limit falls back to global default
        await resolve_and_inject_dream_context(
            [{"role": "system", "content": "sys"}], state, "alice",
        )
        assert state.calls["list_entries"], "entries still fetched despite malformed limit"
        await conn.close()

    async def test_expired_entries_filtered(self):
        conn, store = await _make_store()
        await store.set_user("alice", "ui.dreamEnabled", "true")
        state = _app_state(
            store=store,
            entries=[
                _FakeEntry("alive", expires_at=None),
                _FakeEntry("dead", expires_at="2025-01-01"),
            ],
        )
        messages = [{"role": "system", "content": "sys"}]
        await resolve_and_inject_dream_context(messages, state, "alice")
        assert "alive" in messages[0]["content"]
        assert "dead" not in messages[0]["content"]
        await conn.close()

    async def test_no_store_preserves_legacy_behaviour(self):
        """Tests running without a settings store (rare) should still inject
        if data is present — we don't want to silently drop portraits on
        an old single-tenant install path."""
        state = _app_state(store=None, portrait=_FakePortrait())
        messages = [{"role": "system", "content": "sys"}]
        await resolve_and_inject_dream_context(messages, state, "alice")
        assert "<evolved_self>" in messages[0]["content"]

    async def test_opt_in_lookup_error_fails_closed(self):
        """If settings_store raises, we do NOT inject — better to drop
        context than leak across the gate."""

        class _ExplodingStore:
            async def get_user_or_global(self, *_a, **_kw):
                raise RuntimeError("db unavailable")

        state = _app_state(store=_ExplodingStore(), portrait=_FakePortrait())
        messages = [{"role": "system", "content": "sys"}]
        await resolve_and_inject_dream_context(messages, state, "alice")
        assert messages[0]["content"] == "sys"


# ---------------------------------------------------------------------------
# apply_dream_injection_to_request — the Pydantic-bridging wrapper
# ---------------------------------------------------------------------------


class TestApplyToRequest:

    @staticmethod
    def _req(messages: list[Message]) -> InternalChatRequest:
        return InternalChatRequest(model="x", messages=messages)

    async def test_no_system_message_gets_one_prepended(self):
        """Regression for pre-existing bug: when no system message exists,
        the inline route code would overwrite the user's first message
        with the dream block. This verifies the wrapper inserts cleanly."""
        conn, store = await _make_store()
        await store.set_user("alice", "ui.dreamEnabled", "true")
        state = _app_state(store=store, portrait=_FakePortrait(voice="V"))

        req = self._req([
            Message(role="user", content="hello"),
        ])
        await apply_dream_injection_to_request(req, state, "alice")

        assert len(req.messages) == 2
        assert req.messages[0].role == "system"
        assert "<evolved_self>" in req.messages[0].content
        assert req.messages[1].role == "user"
        assert req.messages[1].content == "hello", \
            "user's message must not be clobbered by prepended system"
        await conn.close()

    async def test_existing_system_message_gets_extended(self):
        conn, store = await _make_store()
        await store.set_user("alice", "ui.dreamEnabled", "true")
        state = _app_state(store=store, portrait=_FakePortrait())

        req = self._req([
            Message(role="system", content="you are helpful"),
            Message(role="user", content="hi"),
        ])
        await apply_dream_injection_to_request(req, state, "alice")

        assert len(req.messages) == 2
        assert "<evolved_self>" in req.messages[0].content
        assert "you are helpful" in req.messages[0].content
        assert req.messages[1].content == "hi"
        await conn.close()

    async def test_preserves_message_side_fields(self):
        """Injection must not drop images/thinking/tool_calls on untouched messages."""
        conn, store = await _make_store()
        await store.set_user("alice", "ui.dreamEnabled", "true")
        state = _app_state(store=store, portrait=_FakePortrait())

        req = self._req([
            Message(role="system", content="sys"),
            Message(
                role="user", content="look at this",
                images=["data:image/png;base64,xyz"],
                thinking="user's chain of thought",
            ),
        ])
        await apply_dream_injection_to_request(req, state, "alice")

        user_msg = req.messages[1]
        assert user_msg.content == "look at this"
        assert user_msg.images == ["data:image/png;base64,xyz"]
        assert user_msg.thinking == "user's chain of thought"
        await conn.close()

    async def test_no_op_when_opted_out(self):
        conn, store = await _make_store()
        await store.set_user("alice", "ui.dreamEnabled", "false")
        state = _app_state(store=store, portrait=_FakePortrait())

        req = self._req([
            Message(role="system", content="sys"),
            Message(role="user", content="hi"),
        ])
        await apply_dream_injection_to_request(req, state, "alice")

        assert len(req.messages) == 2
        assert req.messages[0].content == "sys"
        await conn.close()
