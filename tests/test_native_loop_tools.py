"""Native-loop tool payload hardening (2026-06-13 voice crash).

DeepSeek 400'd a companion voice turn with 'Tool names must be unique'
and the whole turn died after STT had already succeeded. The assembly
deduped by REQUESTED name while the registry's fuzzy resolver can map
two different requests onto one tool (or onto colliding names after a
strict backend's normalization).
"""

from __future__ import annotations

from types import SimpleNamespace

from augmentum.companion_runtime.native_loop import (
    _tool_name_collision_key,
    assemble_native_tools,
)


class FakeRegistry:
    def __init__(self, mapping: dict):
        self._mapping = mapping

    def resolve(self, name):
        return self._mapping.get(name)


def _tool(name: str):
    return SimpleNamespace(name=name)


def test_two_requests_resolving_to_one_tool_dedupe():
    shared = _tool("web")
    reg = FakeRegistry({"browse": shared, "web": shared})
    tools = assemble_native_tools(reg, ["browse", "web"])
    assert [t.name for t in tools] == ["web"]


def test_dot_underscore_collision_deduped():
    # A strict backend normalizes punctuation — 'memory.recall' and
    # 'memory_recall' are the same function name to it.
    reg = FakeRegistry({
        "recall": _tool("memory.recall"),
        "memory_recall": _tool("memory_recall"),
    })
    tools = assemble_native_tools(reg, ["recall", "memory_recall"])
    assert len(tools) == 1


def test_distinct_tools_all_kept_in_order():
    reg = FakeRegistry({
        "a": _tool("alpha"), "b": _tool("beta"), "c": _tool("gamma"),
    })
    tools = assemble_native_tools(reg, ["a", "b", "c", "a"])
    assert [t.name for t in tools] == ["alpha", "beta", "gamma"]


def test_unresolvable_names_skipped():
    reg = FakeRegistry({"a": _tool("alpha")})
    tools = assemble_native_tools(reg, ["nope", "a", "missing"])
    assert [t.name for t in tools] == ["alpha"]


def test_collision_key_normalization():
    assert _tool_name_collision_key("Memory.Recall") == "memory_recall"
    assert _tool_name_collision_key("media-recommend") == "media_recommend"
    assert (
        _tool_name_collision_key("media.recommend")
        != _tool_name_collision_key("media_recommendations")
    )


def test_core_tool_names_has_no_removed_datetime():
    # DateTimeTool was removed from the registry (date/time is prompt-injected);
    # leaving "datetime" in CORE_TOOL_NAMES produced a tool_resolve_failed
    # warning every companion turn.
    from augmentum.companion_runtime.native_loop import CORE_TOOL_NAMES
    assert "datetime" not in CORE_TOOL_NAMES


def test_browse_is_headless_no_auto_snapshot():
    # The browse lookup must NOT auto-mount a UI panel: it fired
    # `browse.snapshot_ready` on every call, so a plain "what's the news" popped
    # the browse panel and the companion briefed thinly off the open page instead
    # of gathering and answering in words (headless-first doctrine). Showing a
    # page is opt-in via the explicit screen verbs. Guard against re-adding it.
    from augmentum.companion_runtime.tools import TOOL_CATALOG
    browse = TOOL_CATALOG["browse"]
    assert "ui_effect" not in browse, "browse must stay headless — no auto-mount"
    # first sentence carries the delivery keyword so the roster marks it a gather tool
    assert "silently" in browse["description"].lower()
    assert "nothing opens" in browse["description"].lower()


def test_select_companion_tools_filters_subagent_keys():
    # analytical/passthrough are subagent handoffs, not global ToolRegistry
    # entries — they must NOT be fed to the resolver (which would warn). The
    # roster still advertises them via the prompt; this only governs the FC
    # tool schema.
    import unittest.mock as mock

    from augmentum.companion_runtime import native_loop
    from augmentum.companion_runtime import tools as tool_bridge

    captured = {}

    def _fake_assemble(registry, names):
        captured["names"] = list(names)
        return []

    def _fake_enumerate(text="", pin=(), *, context_budget_chars=None):  # noqa: ARG001
        # Mimic enumerate_tools: catalogue keys (incl. subagents) as "name".
        return [{"name": k} for k in ("recall", "files_read", "analytical", "passthrough")]

    with mock.patch.object(native_loop, "assemble_native_tools", _fake_assemble), \
         mock.patch.object(tool_bridge, "enumerate_tools", _fake_enumerate), \
         mock.patch.object(tool_bridge, "pending_pin", lambda *a, **k: ()):
        native_loop.select_companion_tools(
            FakeRegistry({}),
            intent=SimpleNamespace(text="hi", metadata={}),
            app_state=SimpleNamespace(),
            user_id="u1",
            session_id="s1",
        )

    names = captured["names"]
    assert "analytical" not in names
    assert "passthrough" not in names
    # Non-subagent catalogue keys survive.
    assert "recall" in names and "files_read" in names
