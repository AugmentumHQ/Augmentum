"""Scheduling substrate is reachable from all three entrypoints (2026-07-07).

The 2026-07-02 scheduling-generalization policy: chat, voice, and the
companion are ENTRYPOINTS to timed action, so the substrate tools must be
*always inferable* by the model — the regex tier in ``tools/filter.py`` is a
fast-path, never the gate. These tests pin the class-level fix:

* chat: ``filter_tools_for_query`` never strips substrate tools from the
  pool, even on messages with zero scheduling vocabulary (the original
  regression: "set a update tracker for bitcoin for every 15 minutes"
  matched no pattern and the model received no scheduling schema at all).
* voice: the substrate opted into ``Tool.surfaces.voice``, and the manifest
  union picks registry opt-ins up at LOOKUP time (voice_routes' module-level
  ``all_voice_tools()`` snapshot predated ``bind_registry`` and permanently
  dropped every registry opt-in — fixed to a live lookup).
* companion: the substrate names ride ``CORE_TOOL_NAMES`` (the tools already
  declared ``surfaces.companion=True``; the loop now honors it).
"""

from __future__ import annotations

from augmentum.tools.filter import (
    _SCHEDULE_INJECTION_ORDER,
    _SCHEDULE_SUBSTRATE_TOOLS,
    filter_tools_for_query,
)


class _FakeTool:
    def __init__(self, name: str):
        self.name = name


_NO_SCHEDULE_VOCAB = "set a update tracker for bitcoin for every 15 minutes please"


def _pool(*names: str) -> list[_FakeTool]:
    return [_FakeTool(n) for n in names]


class TestChatFilterRideAlong:
    def test_substrate_survives_unsignalled_message_large_pool(self):
        # >6 tools + no regex match used to collapse to _safe_defaults,
        # which contains no scheduling tool.
        pool = _pool(
            "web_search", "wikipedia", "youtube", "image_search",
            "python_exec", "image_generation", "create_document",
            *_SCHEDULE_INJECTION_ORDER,
        )
        kept = {t.name for t in filter_tools_for_query(_NO_SCHEDULE_VOCAB, pool)}
        assert _SCHEDULE_SUBSTRATE_TOOLS <= kept, (
            f"substrate stripped on unsignalled turn: missing "
            f"{_SCHEDULE_SUBSTRATE_TOOLS - kept}"
        )

    def test_substrate_survives_signalled_nonschedule_message(self):
        # A message that matches a DIFFERENT pattern group (math) must not
        # evict the substrate either.
        pool = _pool(
            "calculator", "web_search", "wikipedia", "python_exec",
            "image_generation", "youtube", "image_search",
            *_SCHEDULE_INJECTION_ORDER,
        )
        kept = {t.name for t in filter_tools_for_query("calculate 15% of 2840", pool)}
        assert _SCHEDULE_SUBSTRATE_TOOLS <= kept

    def test_absent_substrate_is_not_invented(self):
        # Filter only keeps what the pool offers — installs without the
        # scheduling tools registered are unaffected.
        pool = _pool("web_search", "wikipedia")
        kept = {t.name for t in filter_tools_for_query(_NO_SCHEDULE_VOCAB, pool)}
        assert not (kept & _SCHEDULE_SUBSTRATE_TOOLS)


class TestVoiceManifestLiveLookup:
    def test_substrate_declares_voice_tier(self):
        from augmentum.tools.manage_briefings import (
            CancelBriefingTool,
            ListBriefingsTool,
        )
        from augmentum.tools.schedule_briefing import ScheduleBriefingTool

        for cls in (ScheduleBriefingTool, ListBriefingsTool, CancelBriefingTool):
            tool = cls.__new__(cls)  # surfaces is a pure property
            s = tool.surfaces
            assert s.voice == "disruptive", cls.__name__
            assert s.voice_capability_line, cls.__name__
            assert s.companion is True, cls.__name__

    def test_voice_routes_allowlist_is_live_not_snapshot(self):
        # The legacy import surface must resolve at access time.
        import augmentum.proxy.voice_routes as vr
        from augmentum.intent import manifest

        before = vr._VOICE_TOOLS
        assert isinstance(before, frozenset)

        class _Surf:
            voice = "disruptive"
            voice_capability_line = "test line"

        class _VoiceTool:
            name = "schedule_briefing"
            surfaces = _Surf()

        class _Reg:
            def list_tools(self):
                return [_VoiceTool()]

        old = manifest._registry
        try:
            manifest.bind_registry(_Reg())
            assert "schedule_briefing" in vr._VOICE_TOOLS, (
                "voice allowlist froze at import — registry opt-ins invisible"
            )
        finally:
            manifest._registry = old


class TestCompanionRoster:
    def test_substrate_in_core_tool_names(self):
        from augmentum.companion_runtime.native_loop import CORE_TOOL_NAMES

        missing = _SCHEDULE_SUBSTRATE_TOOLS - set(CORE_TOOL_NAMES)
        assert not missing, f"companion roster missing substrate: {missing}"
