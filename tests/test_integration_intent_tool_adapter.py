"""Integration tests for the ActionTool adapter.

When the LLM invokes an action via tool-calling, the call flows
through augmentum.tools.chain into ActionTool.execute. This test
suite exercises that path with realistic kwargs and verifies:

  * user_id is plumbed through from chain (``_user_id`` kwarg)
  * The handler sees clean args (no leading underscore framework kwargs)
  * ActionResult.surface_emit lands on the per-session pending queue
  * ToolResult exposes the right output / metadata shape
"""

from __future__ import annotations

import pytest


class _FakeNotesStore:
    def __init__(self):
        self.notes = {}

    async def create(self, note, *, user_id):
        if not user_id:
            raise ValueError("user_id required")
        self.notes[note["id"]] = {**note, "user_id": user_id}
        return self.notes[note["id"]]

    async def get(self, note_id, *, user_id):
        n = self.notes.get(note_id)
        return dict(n) if n and n.get("user_id") == user_id else None

    async def update(self, note_id, updates, *, user_id):
        n = self.notes.get(note_id)
        if not n or n.get("user_id") != user_id:
            return None
        n.update(updates)
        return dict(n)


class _AppState:
    def __init__(self):
        self.notes_store = _FakeNotesStore()
        self.intent_referents = {}


@pytest.mark.asyncio
class TestActionToolExecute:
    async def test_user_id_threaded_through(self):
        from augmentum.intent import REGISTRY
        from augmentum.intent.tool_adapter import ActionTool

        app = _AppState()
        action = REGISTRY.get("note.create")
        tool = ActionTool(action, app_state=app)

        # Simulate what augmentum.tools.chain passes to a tool's execute()
        # when ``cache_user_id`` is provided and the tool is flagged
        # ``needs_user_context``.
        result = await tool.execute(
            _user_id="user_abc",
            _context={"user_id": "user_abc", "session_id": "sess_1"},
            title="Test note",
            content="hello",
        )
        assert result.success is True
        assert "Sure" in result.output or "here" in result.output.lower()
        # Note actually landed in the store under the right user
        assert len(app.notes_store.notes) == 1
        nid = next(iter(app.notes_store.notes))
        assert app.notes_store.notes[nid]["user_id"] == "user_abc"

    async def test_surface_emit_queued_on_referent_cache(self):
        from augmentum.intent import REGISTRY
        from augmentum.intent.dispatch import get_referent_cache
        from augmentum.intent.tool_adapter import ActionTool

        app = _AppState()
        action = REGISTRY.get("note.create")
        tool = ActionTool(action, app_state=app)

        await tool.execute(
            _user_id="user_xyz",
            _context={"session_id": "sess_99"},
            title="X", content="Y",
        )

        refs = get_referent_cache(app, "user_xyz", "sess_99")
        # Side-channel queue populated for the voice route to drain.
        assert len(refs.pending_surface_events) == 1
        ev = refs.pending_surface_events[0]
        assert ev["type"] == "intent_action"
        assert ev["action"] == "note.create"
        assert ev["surface"]["channel"] == "note.open_sticky"
        assert ev["surface"]["payload"]["title"] == "X"

    async def test_metadata_carries_intent_action(self):
        from augmentum.intent import REGISTRY
        from augmentum.intent.tool_adapter import ActionTool

        app = _AppState()
        action = REGISTRY.get("note.create")
        tool = ActionTool(action, app_state=app)

        result = await tool.execute(
            _user_id="u", _context={"session_id": "s"},
            title="t", content="c",
        )
        # metadata mirrors the queued event so a future in-stream
        # emitter can also pick it up.
        assert "intent_action" in result.metadata
        assert result.metadata["intent_action"]["action"] == "note.create"

    async def test_filter_underscore_kwargs_from_handler_args(self):
        from augmentum.intent import REGISTRY
        from augmentum.intent.tool_adapter import ActionTool

        # Use a recall action where args validate cleanly.
        action = REGISTRY.get("memory.recall")

        called_with = {}

        async def _spy(text, session, args):
            called_with.update(args)
            return None  # no-op for test

        # Monkeypatch the handler to capture args
        original_handler = action.handler
        try:
            action.handler = _spy
            tool = ActionTool(action, app_state=_AppState())
            await tool.execute(
                _user_id="u",
                _context={"session_id": "s"},
                _request_context={"any": "thing"},
                query="my novel",
            )
        finally:
            action.handler = original_handler

        # Handler only saw the clean args.
        assert "query" in called_with
        assert "_user_id" not in called_with
        assert "_context" not in called_with
        assert "_request_context" not in called_with

    async def test_handler_exception_returns_failure(self):
        from augmentum.intent import REGISTRY
        from augmentum.intent.tool_adapter import ActionTool

        action = REGISTRY.get("note.create")

        async def _broken(text, session, args):
            raise RuntimeError("boom")

        original = action.handler
        try:
            action.handler = _broken
            tool = ActionTool(action, app_state=_AppState())
            result = await tool.execute(_user_id="u", title="x")
        finally:
            action.handler = original

        assert result.success is False
        assert "note.create failed" in result.error or "failed" in result.error.lower()

    async def test_needs_user_context_flag(self):
        from augmentum.intent import REGISTRY
        from augmentum.intent.tool_adapter import ActionTool

        # Chain.py uses this attribute to decide whether to inject
        # ``_user_id`` — every action primitive MUST carry it.
        action = REGISTRY.get("memory.save")
        tool = ActionTool(action, app_state=None)
        assert getattr(tool, "needs_user_context", False) is True


@pytest.mark.asyncio
class TestRegisterActionTools:
    async def test_registers_tier3_actions(self):
        from augmentum.intent import REGISTRY, register_action_tools
        from augmentum.tools.registry import ToolRegistry

        reg = ToolRegistry()
        count = register_action_tools(reg, app_state=_AppState())

        # Should register all tier3 actions
        tier3_count = sum(1 for a in REGISTRY.all() if a.fanout.tier3)
        assert count == tier3_count
        assert reg.get("note.create") is not None
        # Control actions are tier3=False, NOT registered as tools
        assert reg.get("control.stop") is None

    async def test_idempotent_re_register(self):
        from augmentum.intent import register_action_tools
        from augmentum.tools.registry import ToolRegistry

        reg = ToolRegistry()
        first = register_action_tools(reg, app_state=_AppState())
        # Re-register against the same registry — no new tools added.
        second = register_action_tools(reg, app_state=_AppState())
        assert second == 0
        assert first > 0

    async def test_handles_missing_registry(self):
        from augmentum.intent import register_action_tools
        # Should silently return 0 rather than raise.
        assert register_action_tools(None, app_state=None) == 0
