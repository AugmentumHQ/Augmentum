"""Tests for the 2026-07 flow UX pass.

Covers:
- word-boundary keyword scoring in the flow resolver (the "form" ∈
  "transformers" misroute class) + the shared scorer contract
- SurfaceExposure.flow: action verbs hidden from flow surfaces, category
  expansion in the executor filtered, explicit tool_names pins honored
- the three new builtin templates (Explainer / Live Lookup / Summarize)
  and their verdict/streaming protocol
"""

from __future__ import annotations

from augmentum.reasoning.models import FlowStep, ReasoningFlow
from augmentum.reasoning.resolver import (
    MIN_AUTO_ROUTE_SCORE,
    score_flow_for_query,
)
from augmentum.reasoning.templates import (
    BUILTIN_TEMPLATES,
    explainer_flow,
    live_lookup_flow,
    summarize_flow,
)
from augmentum.tools.base import SurfaceExposure, Tool, ToolCategory, ToolResult


def _flow(keywords: list[str], domains: list[str] | None = None) -> ReasoningFlow:
    return ReasoningFlow(
        name="t", trigger_keywords=keywords, trigger_domains=domains or [],
    )


class TestKeywordScoring:
    def test_word_boundary_blocks_midword_match(self):
        # The observed misroute: Application's "form" fired on "transformers".
        score, matched = score_flow_for_query(
            _flow(["form"]), "help me learn how transformers work",
        )
        assert score == 0
        assert matched == []

    def test_word_boundary_blocks_app_in_happen(self):
        score, _ = score_flow_for_query(_flow(["app"]), "what happened today")
        assert score == 0

    def test_exact_word_still_matches(self):
        score, matched = score_flow_for_query(_flow(["form"]), "fill out this form")
        assert score == 2
        assert matched == ["form"]

    def test_phrases_match_with_boundaries(self):
        score, matched = score_flow_for_query(
            _flow(["current events"]), "any current events in europe?",
        )
        assert score == 2
        assert matched == ["current events"]

    def test_domains_score_one(self):
        score, matched = score_flow_for_query(
            _flow([], domains=["news"]), "news about rates",
        )
        assert score == 1
        assert matched == ["news"]

    def test_phrase_superset_outscores_content_noun(self):
        # "tldr this" + "tldr" both match an explicit summarize ask, so the
        # intent verb beats a flow keyword appearing inside pasted content.
        summarize = _flow(["tldr", "tldr this"])
        report = _flow(["report"])
        q = "tldr this: quarterly report says revenue grew"
        s_sum, _ = score_flow_for_query(summarize, q)
        s_rep, _ = score_flow_for_query(report, q)
        assert s_sum > s_rep
        assert s_sum >= MIN_AUTO_ROUTE_SCORE

    def test_punctuation_keywords_do_not_crash(self):
        score, _ = score_flow_for_query(_flow(["tl;dr"]), "tl;dr this article")
        assert score == 2

    def test_empty_keyword_ignored(self):
        score, _ = score_flow_for_query(_flow(["", "  "]), "anything")
        assert score == 0


class _StubTool(Tool):
    def __init__(self, name: str, category: ToolCategory, flow: bool) -> None:
        self._name = name
        self._category = category
        self._flow = flow

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "stub"

    @property
    def category(self) -> ToolCategory:
        return self._category

    @property
    def input_schema(self) -> dict:
        return {"type": "object", "properties": {}}

    @property
    def surfaces(self) -> SurfaceExposure:
        return SurfaceExposure(flow=self._flow)

    async def execute(self, **kwargs) -> ToolResult:
        return ToolResult(success=True, output="ok")


class _StubRegistry:
    def __init__(self, tools):
        self._tools = {t.name: t for t in tools}

    def get(self, name):
        return self._tools.get(name)

    def get_for_phase(self, phase, *, exclude=None, allowed_names=None):
        return [
            t for t in self._tools.values()
            if t.category.value == phase and (not exclude or t.name not in exclude)
        ]


class TestFlowSurfaceFiltering:
    def test_action_verbs_declare_flow_false(self):
        from augmentum.intent.tool_adapter import ActionTool
        surfaces = ActionTool.surfaces.fget(object.__new__(ActionTool))
        assert surfaces.flow is False
        assert surfaces.chat is True  # unchanged historical reach

    def test_default_exposure_keeps_flow_true(self):
        assert SurfaceExposure().flow is True

    def test_category_expansion_filters_flow_false(self):
        from augmentum.reasoning.executor import _resolve_tools_for_step
        reg = _StubRegistry([
            _StubTool("web_search", ToolCategory.SEARCH, flow=True),
            _StubTool("media.play", ToolCategory.SEARCH, flow=False),
        ])
        step = FlowStep(name="s", tool_categories=["search"])
        names = {t.name for t in _resolve_tools_for_step(step, reg)}
        assert "web_search" in names
        assert "media.play" not in names

    def test_explicit_pin_bypasses_flow_filter(self):
        from augmentum.reasoning.executor import _resolve_tools_for_step
        reg = _StubRegistry([
            _StubTool("media.play", ToolCategory.SEARCH, flow=False),
        ])
        step = FlowStep(name="s", tool_names=["media.play"])
        names = {t.name for t in _resolve_tools_for_step(step, reg)}
        assert "media.play" in names

    def test_tool_host_surface_filter(self):
        from types import SimpleNamespace

        from augmentum.capabilities.frontdesk import CapabilityContext
        from augmentum.capabilities.tool_host import ToolHost

        reg = SimpleNamespace(list_tools=lambda: [
            _StubTool("web_search", ToolCategory.SEARCH, flow=True),
            _StubTool("note.create", ToolCategory.EXECUTE, flow=False),
        ])
        ctx = CapabilityContext(SimpleNamespace(tool_registry=reg))
        full = ToolHost().list_ui_tools(ctx)
        flow_only = ToolHost().list_ui_tools(ctx, surface="flow")
        assert {t["name"] for t in full["tools"]} == {"web_search", "note.create"}
        assert {t["name"] for t in flow_only["tools"]} == {"web_search"}


class TestNewTemplates:
    def test_registered(self):
        for key in ("explainer", "live_lookup", "summarize"):
            assert key in BUILTIN_TEMPLATES

    def test_final_step_streams(self):
        for factory in (explainer_flow, live_lookup_flow, summarize_flow):
            flow = factory()
            assert flow.is_builtin is True
            assert flow.steps[-1].stream_to_user is True
            assert flow.steps[-1].role == "respond"

    def test_explainer_verify_is_verdict_shaped(self):
        flow = explainer_flow()
        verify = next(s for s in flow.steps if s.role == "verify")
        for marker in ("ERRORS_FOUND", "VERIFIED", "CONFIDENCE"):
            assert marker in verify.system_prompt
        # Grounding-aware: search results outrank training priors.
        assert "NEWER than your training data" in verify.system_prompt
        # Tight cap so a chatty model can't append a restatement.
        assert 0 < verify.output_cap <= 300

    def test_live_lookup_search_has_fetch_fallback(self):
        flow = live_lookup_flow()
        search = flow.steps[0]
        assert "search" in search.tool_categories
        assert "fetch" in search.tool_categories

    def test_summarize_uses_no_search_tools(self):
        flow = summarize_flow()
        for step in flow.steps:
            assert "search" not in step.tool_categories

    def test_no_exact_keyword_collision_with_other_builtins(self):
        # Overlapping keywords make routing tie (creation-order roulette).
        # New templates must not collide with any other builtin.
        new = {"Explainer", "Live Lookup", "Summarize"}
        keyword_owners: dict[str, str] = {}
        for factory in BUILTIN_TEMPLATES.values():
            flow = factory()
            for kw in flow.trigger_keywords:
                k = kw.lower()
                prior = keyword_owners.get(k)
                if prior and (prior in new or flow.name in new):
                    raise AssertionError(
                        f'keyword "{k}" owned by both {prior} and {flow.name}'
                    )
                keyword_owners.setdefault(k, flow.name)
