"""Smoke tests — import the intent package + verify the registry shape.

Verifies:
  * Every builtin action module imports + registers its actions
  * The Tier 1 matcher returns the expected hits for canonical phrases
  * The schema generator produces well-formed JSON Schema
  * ActionTool wrapping preserves action metadata
"""

from __future__ import annotations

import pytest

# Architect primitives are registered as Actions in the shared REGISTRY
# via @register_action decorators that fire at module import. Without
# this import, primitives like ``companion.today_recap`` are absent
# from the registry and any smoke test that asserts against them will
# fail with no match. Production paths (voice_routes.py, chat handlers)
# import architect transitively, so this only matters for tests.
import augmentum.architect  # noqa: F401


class TestIntentPackageImports:
    """Verify the package + every builtin module is importable."""

    def test_package_imports(self):
        from augmentum.intent import (
            Action,
            match_intent,
            register_action,
        )
        # Smoke: every export is a real symbol
        assert Action is not None
        assert callable(register_action)
        assert callable(match_intent)

    def test_builtins_register_on_import(self):
        from augmentum.intent import REGISTRY
        actions_by_id = {a.id for a in REGISTRY.all()}
        # Control set — at minimum the 6 fast-path actions are registered.
        for expected in (
            "control.stop", "control.repeat", "control.slower",
            "control.louder", "control.goodbye", "control.nevermind",
        ):
            assert expected in actions_by_id, f"missing {expected}"
        # Navigation
        assert "navigate.open_surface" in actions_by_id
        assert "navigate.back" in actions_by_id
        # Notes / memory primitives
        for expected in (
            "note.create", "note.append", "note.show_sticky",
            "note.start_capture", "note.end_capture",
            "memory.save", "memory.recall",
        ):
            assert expected in actions_by_id, f"missing {expected}"


class TestTier1Matcher:
    """Canonical phrase → action id, sanity-checking the patterns."""

    @pytest.mark.parametrize("text,expected_id", [
        ("stop", "control.stop"),
        ("stop.", "control.stop"),
        ("shut up", "control.stop"),
        ("what?", "control.repeat"),
        ("say that again", "control.repeat"),
        ("slow down", "control.slower"),
        ("louder", "control.louder"),
        ("bye becca", "control.goodbye"),
        ("thanks bye", "control.goodbye"),
        ("never mind", "control.nevermind"),
        ("nvm", "control.nevermind"),
        ("open browse", "navigate.open_surface"),
        ("show me my files", "navigate.open_surface"),
        ("open the notes panel", "navigate.open_surface"),
        ("pull up settings", "navigate.open_surface"),
        ("go back", "navigate.back"),
        ("close that", "navigate.back"),
        ("open a new note", "note.create"),
        ("open up a new note", "note.create"),
        ("make me a quick note", "note.create"),
        ("jot something down", "note.create"),
        ("translate my thoughts", "note.start_capture"),
        ("take notes on this", "note.start_capture"),
        ("save this", "note.end_capture"),
        ("stop noting", "note.end_capture"),
        ("that's enough", "note.end_capture"),
        ("remember that", "memory.save"),
        ("save this to memory", "memory.save"),
    ])
    def test_pattern_match(self, text, expected_id):
        from augmentum.intent import match_intent
        m = match_intent(text)
        assert m is not None, f"no match for {text!r}"
        assert m.action_id == expected_id, (
            f"expected {expected_id} for {text!r}, got {m.action_id}"
        )

    def test_no_match_for_unrelated(self):
        from augmentum.intent import match_intent
        for text in (
            "hello world",
            "what's the weather like",
            "do you have any thoughts on this design",
        ):
            m = match_intent(text)
            # NB: "do you have any thoughts on" doesn't trigger
            # capture mode — that requires the more specific
            # "take notes on" / "translate my thoughts" phrasing.
            assert m is None or m.action_id != "note.start_capture", (
                f"false positive on {text!r}: {m and m.action_id}"
            )

    def test_open_surface_extracts_arg(self):
        from augmentum.intent import match_intent
        m = match_intent("open the coder panel")
        assert m is not None
        assert m.action_id == "navigate.open_surface"
        assert m.args.get("surface") == "coder"


class TestActionToolSchema:
    """Verify the JSON Schema generator output for LLM consumption."""

    def test_required_args_propagated(self):
        from augmentum.intent import REGISTRY
        from augmentum.intent.tool_adapter import ActionTool

        save = REGISTRY.get("memory.save")
        assert save is not None
        tool = ActionTool(save, app_state=None)
        schema = tool.input_schema
        assert schema["type"] == "object"
        assert "content" in schema["properties"]
        assert "required" in schema
        assert "content" in schema["required"]

    def test_optional_args_not_required(self):
        from augmentum.intent import REGISTRY
        from augmentum.intent.tool_adapter import ActionTool

        create = REGISTRY.get("note.create")
        assert create is not None
        tool = ActionTool(create, app_state=None)
        schema = tool.input_schema
        # title / content are optional → no "required" key
        assert schema.get("required") in (None, [])

    def test_navigate_enum_preserved(self):
        from augmentum.intent import REGISTRY
        from augmentum.intent.tool_adapter import ActionTool

        nav = REGISTRY.get("navigate.open_surface")
        assert nav is not None
        tool = ActionTool(nav, app_state=None)
        schema = tool.input_schema
        surface = schema["properties"]["surface"]
        assert "enum" in surface
        assert "browse" in surface["enum"]
        assert "coder" in surface["enum"]
        assert "required" in schema
        assert "surface" in schema["required"]

    def test_action_tool_metadata(self):
        from augmentum.intent import REGISTRY
        from augmentum.intent.tool_adapter import ActionTool
        from augmentum.tools.base import ToolCategory

        stop = REGISTRY.get("control.stop")
        tool = ActionTool(stop, app_state=None)
        assert tool.name == "control.stop"
        assert tool.category == ToolCategory.EXECUTE
        # Side-effecting verbs aren't cacheable.
        assert tool.cacheable is False
        # Description includes a sample phrasing for the LLM.
        assert "stop" in tool.description.lower()


class TestReferentCache:
    """Per-session cache survives across dispatch() calls."""

    def test_isolated_by_user_session(self):
        from augmentum.intent.dispatch import get_referent_cache

        class _AppState:
            pass

        app = _AppState()
        a = get_referent_cache(app, "user_a", "sess_1")
        b = get_referent_cache(app, "user_b", "sess_1")
        c = get_referent_cache(app, "user_a", "sess_1")  # same key

        a.active_note_id = "note_a"
        b.active_note_id = "note_b"

        assert c.active_note_id == "note_a"
        assert b.active_note_id == "note_b"
        assert a is c
        assert a is not b

    def test_handles_missing_app_state(self):
        from augmentum.intent.action import ReferentCache
        from augmentum.intent.dispatch import get_referent_cache

        cache = get_referent_cache(None, "u", "s")
        # Returns a fresh ReferentCache rather than raising.
        assert isinstance(cache, ReferentCache)


class TestReferentCacheEviction:
    """Lazy TTL sweep drops idle caches without touching active ones."""

    def test_stale_cache_evicted_after_ttl(self, monkeypatch):
        # Disambiguation: ``augmentum.intent`` re-exports the
        # ``dispatch`` function from this module, so the bare alias
        # ``import augmentum.intent.dispatch as dispatch_mod`` resolves
        # to the function, not the module. Go through importlib so we
        # get the actual module object (needed for monkeypatch).
        import importlib
        dispatch_mod = importlib.import_module("augmentum.intent.dispatch")
        from augmentum.intent.action import ReferentCache
        from augmentum.intent.dispatch import (
            REFERENT_TTL_SECONDS,
            _evict_stale_referents,
        )

        store: dict = {}
        now = 1000.0

        stale = ReferentCache()
        stale.last_touched = now - REFERENT_TTL_SECONDS - 1.0
        store[("u_old", "s_old")] = stale

        fresh = ReferentCache()
        fresh.last_touched = now - 10.0
        store[("u_new", "s_new")] = fresh

        evicted = _evict_stale_referents(store, now)
        assert evicted == 1
        assert ("u_old", "s_old") not in store
        assert ("u_new", "s_new") in store

    def test_get_touches_last_touched(self):
        from augmentum.intent.dispatch import get_referent_cache

        class _AppState:
            pass

        app = _AppState()
        cache = get_referent_cache(app, "u", "s")
        first_touched = cache.last_touched
        assert first_touched > 0
        # Touch again — last_touched advances (or stays at the same
        # monotonic tick on very fast hardware; >= rules both out).
        cache2 = get_referent_cache(app, "u", "s")
        assert cache2 is cache
        assert cache2.last_touched >= first_touched

    def test_sweep_is_rate_limited(self, monkeypatch):
        # Disambiguation: ``augmentum.intent`` re-exports the
        # ``dispatch`` function from this module, so the bare alias
        # ``import augmentum.intent.dispatch as dispatch_mod`` resolves
        # to the function, not the module. Go through importlib so we
        # get the actual module object (needed for monkeypatch).
        import importlib
        dispatch_mod = importlib.import_module("augmentum.intent.dispatch")

        calls: list[float] = []

        def fake_evict(store, now):
            calls.append(now)
            return 0

        monkeypatch.setattr(dispatch_mod, "_evict_stale_referents", fake_evict)

        class _AppState:
            pass

        app = _AppState()
        # First call triggers a sweep (last_sweep is 0).
        dispatch_mod.get_referent_cache(app, "u", "s")
        # Second call within the sweep interval should NOT trigger.
        dispatch_mod.get_referent_cache(app, "u", "s2")
        assert len(calls) == 1, calls


class TestVoiceToolManifest:
    """Voice-tool manifest filters allowlists by ambient policy."""

    def test_full_policy_returns_all_buckets(self):
        from augmentum.intent.manifest import (
            all_voice_tools,
            voice_tools_for,
        )

        all_set = all_voice_tools()
        # Foreground (ambient=False) always returns the full set.
        assert voice_tools_for(ambient=False, policy="safe") == all_set
        # Explicit full policy in ambient returns the full set too.
        assert voice_tools_for(ambient=True, policy="full") == all_set

    def test_safe_policy_drops_disruptive_and_costly(self):
        from augmentum.intent.manifest import (
            VOICE_TOOLS_CORE,
            VOICE_TOOLS_COSTLY,
            VOICE_TOOLS_DISRUPTIVE,
            VOICE_TOOLS_INTERACTIVE,
            voice_tools_for,
        )

        safe = voice_tools_for(ambient=True, policy="safe")
        assert safe == VOICE_TOOLS_CORE | VOICE_TOOLS_INTERACTIVE
        # Disruptive + costly are filtered out.
        assert not (safe & VOICE_TOOLS_DISRUPTIVE)
        assert not (safe & VOICE_TOOLS_COSTLY)

    def test_minimal_policy_is_core_only(self):
        from augmentum.intent.manifest import (
            VOICE_TOOLS_CORE,
            voice_tools_for,
        )

        assert voice_tools_for(ambient=True, policy="minimal") == VOICE_TOOLS_CORE

    def test_custom_policy_with_unknown_drops_silently(self):
        from augmentum.intent.manifest import voice_tools_for

        # "note.create" is real, "fake.tool" is not.
        out = voice_tools_for(
            ambient=True,
            policy="custom",
            custom_allowlist=["note.create", "fake.tool"],
        )
        assert "note.create" in out
        assert "fake.tool" not in out

    def test_custom_policy_with_empty_list_falls_back_to_minimal(self):
        from augmentum.intent.manifest import (
            VOICE_TOOLS_CORE,
            voice_tools_for,
        )

        out = voice_tools_for(ambient=True, policy="custom", custom_allowlist=[])
        # Empty custom collapses to minimal — never silently expose
        # the full universe from a misconfigured allowlist.
        assert out == VOICE_TOOLS_CORE

    def test_unknown_policy_falls_back_to_default(self):
        from augmentum.intent.manifest import (
            DEFAULT_AMBIENT_POLICY,
            voice_tools_for,
        )

        bad = voice_tools_for(ambient=True, policy="banana")
        good = voice_tools_for(ambient=True, policy=DEFAULT_AMBIENT_POLICY)
        assert bad == good

    def test_today_recap_in_core_bucket(self):
        from augmentum.intent.manifest import VOICE_TOOLS_CORE

        # Companion's daily reflection is always-safe.
        assert "companion.today_recap" in VOICE_TOOLS_CORE


class TestTodayRecapPatterns:
    """today_recap Tier 1 patterns cover the direct phrasings."""

    @pytest.mark.parametrize("text", [
        "today recap",
        "today's recap",
        "recap today",
        "recap of today",
        "summarize my day",
        "summarize today",
        "daily summary",
        "what did I do today",
        "what have I done today",
    ])
    def test_today_recap_direct_phrasing(self, text):
        from augmentum.intent import match_intent

        m = match_intent(text)
        assert m is not None, f"no Tier 1 match for {text!r}"
        assert m.action_id == "companion.today_recap", (
            f"expected companion.today_recap for {text!r}, got {m.action_id}"
        )


class TestCaptureCleanup:
    """Capture-mode cleanup degrades gracefully on every failure path."""

    @pytest.mark.asyncio
    async def test_empty_text_returns_unchanged(self):
        from augmentum.intent.capture_cleanup import cleanup_captured_text

        assert await cleanup_captured_text("", app_state=None) == ""
        assert await cleanup_captured_text("   \n\t  ", app_state=None) == "   \n\t  "

    @pytest.mark.asyncio
    async def test_disabled_setting_returns_raw(self, monkeypatch):
        from augmentum.config import settings
        from augmentum.intent.capture_cleanup import cleanup_captured_text

        monkeypatch.setattr(
            settings, "companion_note_capture_cleanup", False, raising=False,
        )
        raw = "this is what i said and i kept saying it"
        out = await cleanup_captured_text(raw, app_state=None)
        assert out == raw

    @pytest.mark.asyncio
    async def test_no_registry_returns_raw(self, monkeypatch):
        from augmentum.config import settings
        from augmentum.intent.capture_cleanup import cleanup_captured_text

        monkeypatch.setattr(
            settings, "companion_note_capture_cleanup", True, raising=False,
        )

        class _AppState:
            pass

        raw = "something the user said"
        # provider_registry is None on _AppState — function returns raw.
        out = await cleanup_captured_text(raw, app_state=_AppState())
        assert out == raw

    @pytest.mark.asyncio
    async def test_apply_cleanup_skips_when_baseline_past_end(self):
        from augmentum.intent.capture_cleanup import apply_cleanup_to_note

        class _Store:
            async def get(self, _id, *, user_id):  # noqa: ARG002
                return {"id": "n", "content": "abc"}

        # baseline of 10 with content length 3 — nothing captured.
        changed, new_content = await apply_cleanup_to_note(
            _Store(),
            "n",
            user_id="u",
            baseline_chars=10,
            app_state=None,
        )
        assert changed is False
        assert new_content == ""
