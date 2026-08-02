"""Tests for agentic flow templates and flow resolution (Phase D)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from augmentum.models.base import (
    InternalChatRequest,
    InternalChatResponse,
    InternalStreamChunk,
    Message,
)
from augmentum.reasoning.models import FlowStep, ReasoningFlow


# ---------------------------------------------------------------------------
# Template structure tests
# ---------------------------------------------------------------------------


class TestReportFlow:
    def test_report_flow_structure(self):
        from augmentum.reasoning.templates import agentic_report_flow

        flow = agentic_report_flow()
        assert flow.name == "Report"
        assert "agentic" in flow.trigger_domains
        assert flow.is_builtin is True
        assert flow.autonomy_level == 3

    def test_report_flow_has_correct_roles(self):
        from augmentum.reasoning.templates import agentic_report_flow

        flow = agentic_report_flow()
        roles = [s.role for s in flow.steps]
        assert roles == ["plan", "search", "analyze", "draft", "review", "create", "deliver"]

    def test_report_flow_step_count(self):
        from augmentum.reasoning.templates import agentic_report_flow

        flow = agentic_report_flow()
        assert len(flow.steps) == 7

    def test_report_flow_has_search_tools(self):
        from augmentum.reasoning.templates import agentic_report_flow

        flow = agentic_report_flow()
        research_step = [s for s in flow.steps if s.role == "search"][0]
        assert "search" in research_step.tool_categories
        assert "fetch" in research_step.tool_categories

    def test_report_flow_has_artifact_tools(self):
        from augmentum.reasoning.templates import agentic_report_flow

        flow = agentic_report_flow()
        create_step = [s for s in flow.steps if s.role == "create"][0]
        assert "create_document" in create_step.tool_names
        assert "artifact" in create_step.tool_categories

    def test_report_flow_only_deliver_streams(self):
        from augmentum.reasoning.templates import agentic_report_flow

        flow = agentic_report_flow()
        streaming = [s.name for s in flow.steps if s.stream_to_user]
        assert streaming == ["Deliver"], f"Only Deliver should stream, got {streaming}"
        internal = [s.name for s in flow.steps if not s.stream_to_user]
        assert "Plan" in internal
        assert "Research" in internal
        assert "Draft" in internal

    def test_report_flow_keywords(self):
        from augmentum.reasoning.templates import agentic_report_flow

        flow = agentic_report_flow()
        assert "report" in flow.trigger_keywords
        assert "document" in flow.trigger_keywords
        assert "whitepaper" in flow.trigger_keywords

    def test_report_flow_auto_search_enabled(self):
        from augmentum.reasoning.templates import agentic_report_flow

        flow = agentic_report_flow()
        assert flow.auto_search is True


class TestPresentationFlow:
    def test_presentation_flow_structure(self):
        from augmentum.reasoning.templates import agentic_presentation_flow

        flow = agentic_presentation_flow()
        assert flow.name == "Presentation"
        assert "agentic" in flow.trigger_domains
        assert flow.is_builtin is True
        assert flow.autonomy_level == 3

    def test_presentation_flow_has_correct_roles(self):
        from augmentum.reasoning.templates import agentic_presentation_flow

        flow = agentic_presentation_flow()
        roles = [s.role for s in flow.steps]
        assert roles == [
            "plan", "search", "analyze", "draft",
            "illustrate", "create", "review", "deliver",
        ]

    def test_presentation_flow_step_count(self):
        from augmentum.reasoning.templates import agentic_presentation_flow

        flow = agentic_presentation_flow()
        assert len(flow.steps) == 8

    def test_presentation_flow_has_illustrate_step(self):
        from augmentum.reasoning.templates import agentic_presentation_flow

        flow = agentic_presentation_flow()
        illustrate = [s for s in flow.steps if s.role == "illustrate"]
        assert len(illustrate) == 1

    def test_presentation_flow_has_image_tools(self):
        from augmentum.reasoning.templates import agentic_presentation_flow

        flow = agentic_presentation_flow()
        illustrate_step = [s for s in flow.steps if s.role == "illustrate"][0]
        # New Illustrate Slides step: image_search + post-render picker.
        # Legacy image_generation path is gone — Generate Instead is an
        # in-picker escape hatch, not a flow step.
        assert "image_search" in illustrate_step.tool_names
        assert "image_generation" not in illustrate_step.tool_names
        assert "image" in illustrate_step.tool_categories

    def test_presentation_flow_has_artifact_tools(self):
        from augmentum.reasoning.templates import agentic_presentation_flow

        flow = agentic_presentation_flow()
        create_step = [s for s in flow.steps if s.role == "create"][0]
        assert "create_presentation" in create_step.tool_names
        assert "artifact" in create_step.tool_categories

    def test_presentation_flow_keywords(self):
        from augmentum.reasoning.templates import agentic_presentation_flow

        flow = agentic_presentation_flow()
        assert "presentation" in flow.trigger_keywords
        assert "slides" in flow.trigger_keywords
        assert "pptx" in flow.trigger_keywords

    def test_presentation_flow_only_deliver_streams(self):
        from augmentum.reasoning.templates import agentic_presentation_flow

        flow = agentic_presentation_flow()
        streaming = [s.name for s in flow.steps if s.stream_to_user]
        assert streaming == ["Deliver"]


class TestStorybookFlow:
    def test_storybook_flow_structure(self):
        from augmentum.reasoning.templates import agentic_storybook_flow

        flow = agentic_storybook_flow()
        assert flow.name == "Storybook"
        assert "agentic" in flow.trigger_domains
        assert flow.is_builtin is True
        assert flow.autonomy_level == 3

    def test_storybook_flow_has_correct_roles(self):
        from augmentum.reasoning.templates import agentic_storybook_flow

        # The flow was collapsed from 6 steps to 3: illustration is now
        # handled inside create_ebook._auto_illustrate, and there is no
        # separate review/deliver step — the "create" step streams the
        # final download link directly (see templates.py:1043 docstring).
        flow = agentic_storybook_flow()
        roles = [s.role for s in flow.steps]
        assert roles == ["plan", "draft", "create"]

    def test_storybook_flow_step_count(self):
        from augmentum.reasoning.templates import agentic_storybook_flow

        flow = agentic_storybook_flow()
        assert len(flow.steps) == 3

    def test_storybook_flow_no_search(self):
        from augmentum.reasoning.templates import agentic_storybook_flow

        flow = agentic_storybook_flow()
        assert flow.auto_search is False
        # No search role step
        search_steps = [s for s in flow.steps if s.role == "search"]
        assert len(search_steps) == 0

    def test_storybook_flow_auto_illustrates_via_ebook_tool(self):
        """Illustration is no longer a standalone step — create_ebook handles
        cover + per-chapter images via _auto_illustrate(). Verify the flow
        wires the ebook tool rather than invoking image_generation manually.
        """
        from augmentum.reasoning.templates import agentic_storybook_flow

        flow = agentic_storybook_flow()
        create_step = [s for s in flow.steps if s.role == "create"][0]
        assert "create_ebook" in create_step.tool_names
        # The flow intentionally does NOT call image_generation itself —
        # _auto_illustrate is the single entry point for illustrations.
        all_tools = {t for s in flow.steps for t in s.tool_names}
        assert "image_generation" not in all_tools

    def test_storybook_flow_create_step_wires_ebook(self):
        from augmentum.reasoning.templates import agentic_storybook_flow

        flow = agentic_storybook_flow()
        create_step = [s for s in flow.steps if s.role == "create"][0]
        assert "create_ebook" in create_step.tool_names

    def test_storybook_flow_keywords(self):
        from augmentum.reasoning.templates import agentic_storybook_flow

        flow = agentic_storybook_flow()
        assert "storybook" in flow.trigger_keywords
        assert "story" in flow.trigger_keywords
        assert "fairy tale" in flow.trigger_keywords

    def test_storybook_flow_only_create_streams(self):
        """Only the final create_ebook step streams to the user — the plan
        and draft steps build working memory silently."""
        from augmentum.reasoning.templates import agentic_storybook_flow

        flow = agentic_storybook_flow()
        streaming = [s.name for s in flow.steps if s.stream_to_user]
        assert streaming == ["Create Book"]


class TestDataComparisonFlow:
    def test_data_comparison_flow_structure(self):
        from augmentum.reasoning.templates import agentic_data_comparison_flow

        flow = agentic_data_comparison_flow()
        assert flow.name == "Data & Comparison"
        assert "agentic" in flow.trigger_domains
        assert flow.is_builtin is True
        assert flow.autonomy_level == 3

    def test_data_comparison_flow_has_correct_roles(self):
        from augmentum.reasoning.templates import agentic_data_comparison_flow

        flow = agentic_data_comparison_flow()
        roles = [s.role for s in flow.steps]
        assert roles == ["plan", "search", "analyze", "create", "create", "deliver"]

    def test_data_comparison_flow_step_count(self):
        from augmentum.reasoning.templates import agentic_data_comparison_flow

        flow = agentic_data_comparison_flow()
        assert len(flow.steps) == 6

    def test_data_comparison_flow_has_chart_tools(self):
        from augmentum.reasoning.templates import agentic_data_comparison_flow

        flow = agentic_data_comparison_flow()
        chart_step = [s for s in flow.steps if "create_chart" in s.tool_names][0]
        assert "create_chart" in chart_step.tool_names
        assert "artifact" in chart_step.tool_categories

    def test_data_comparison_flow_has_spreadsheet_tools(self):
        from augmentum.reasoning.templates import agentic_data_comparison_flow

        flow = agentic_data_comparison_flow()
        sheet_step = [s for s in flow.steps if "create_spreadsheet" in s.tool_names][0]
        assert "create_spreadsheet" in sheet_step.tool_names
        assert "artifact" in sheet_step.tool_categories

    def test_data_comparison_flow_has_python_exec(self):
        from augmentum.reasoning.templates import agentic_data_comparison_flow

        flow = agentic_data_comparison_flow()
        analyze_step = [s for s in flow.steps if s.role == "analyze"][0]
        assert "python_exec" in analyze_step.tool_names

    def test_data_comparison_flow_keywords(self):
        from augmentum.reasoning.templates import agentic_data_comparison_flow

        flow = agentic_data_comparison_flow()
        assert "chart" in flow.trigger_keywords
        assert "data analysis" in flow.trigger_keywords
        assert "visualization" in flow.trigger_keywords

    def test_data_analysis_flow_only_deliver_streams(self):
        from augmentum.reasoning.templates import agentic_data_comparison_flow

        flow = agentic_data_comparison_flow()
        streaming = [s.name for s in flow.steps if s.stream_to_user]
        assert streaming == ["Deliver"]

    def test_data_analysis_flow_auto_search_enabled(self):
        from augmentum.reasoning.templates import agentic_data_comparison_flow

        flow = agentic_data_comparison_flow()
        assert flow.auto_search is True


# ---------------------------------------------------------------------------
# Template registry tests
# ---------------------------------------------------------------------------


class TestTemplateRegistry:
    def test_agentic_templates_in_registry(self):
        from augmentum.reasoning.templates import BUILTIN_TEMPLATES

        assert "agentic_report" in BUILTIN_TEMPLATES
        assert "agentic_presentation" in BUILTIN_TEMPLATES
        assert "agentic_storybook" in BUILTIN_TEMPLATES

    def test_all_agentic_templates_are_builtin(self):
        from augmentum.reasoning.templates import BUILTIN_TEMPLATES

        for key in ("agentic_report", "agentic_presentation", "agentic_storybook", "agentic_data_comparison"):
            flow = BUILTIN_TEMPLATES[key]()
            assert flow.is_builtin is True

    def test_agentic_templates_have_unique_ids(self):
        from augmentum.reasoning.templates import BUILTIN_TEMPLATES

        ids = set()
        for key in ("agentic_report", "agentic_presentation", "agentic_storybook", "agentic_data_comparison"):
            flow = BUILTIN_TEMPLATES[key]()
            assert flow.id not in ids
            ids.add(flow.id)

    def test_get_template_returns_agentic(self):
        from augmentum.reasoning.templates import get_template

        flow = get_template("agentic_report")
        assert flow is not None
        assert flow.name == "Report"

    def test_list_templates_includes_agentic(self):
        from augmentum.reasoning.templates import list_templates

        templates = list_templates()
        names = {t["name"] for t in templates}
        assert "agentic_report" in names
        assert "agentic_presentation" in names
        assert "agentic_storybook" in names
        assert "agentic_data_comparison" in names
        assert "agentic_fact_checker" in names
        assert "agentic_tutorial" in names
        assert "agentic_application" in names
        # agentic_competitive_analysis was merged into agentic_data_comparison

    def test_new_templates_in_registry(self):
        from augmentum.reasoning.templates import BUILTIN_TEMPLATES

        assert "agentic_data_comparison" in BUILTIN_TEMPLATES
        assert "agentic_fact_checker" in BUILTIN_TEMPLATES
        assert "agentic_tutorial" in BUILTIN_TEMPLATES
        assert "agentic_application" in BUILTIN_TEMPLATES
        # agentic_competitive_analysis merged into agentic_data_comparison

    def test_total_template_count(self):
        from augmentum.reasoning.templates import BUILTIN_TEMPLATES

        # 7 base + 7 agentic = 14
        assert len(BUILTIN_TEMPLATES) == 14


class TestTutorialFlowFormatBranch:
    """The Tutorial Builder serves any topic, not just coding tutorials.

    Regression guard for the bug where physical/conceptual tutorials (e.g.
    "how to change a tire") came out with Python ``print()`` statements and
    fake "expected console output", because the draft step hardcoded
    "COMPLETE code → expected output" into every step.
    """

    def _flow(self):
        from augmentum.reasoning.templates import agentic_tutorial_flow

        return agentic_tutorial_flow()

    def test_plan_emits_format_decision(self):
        plan = next(s for s in self._flow().steps if s.role == "plan")
        # The plan must label the topic so downstream steps can branch.
        assert "FORMAT" in plan.system_prompt
        assert "procedure" in plan.system_prompt.lower()
        assert "code" in plan.system_prompt.lower()

    def test_draft_branches_on_format_and_forbids_code_for_procedures(self):
        draft = next(s for s in self._flow().steps if s.role == "draft")
        sp = draft.system_prompt.lower()
        # Must offer a plain-language path, not demand code for every step.
        assert "procedure" in sp
        assert "plain language" in sp or "plain-language" in sp
        # Must explicitly forbid wrapping real-world actions in code.
        assert "print(" in draft.system_prompt
        # The old unconditional "COMPLETE code → expected output" per-step
        # structure must be gone.
        assert "complete code → expected output" not in sp

    def test_verify_does_not_force_code_on_procedures(self):
        verify = next(s for s in self._flow().steps if s.role == "verify")
        # Verify must no-op (not fabricate code) when there's nothing to run.
        assert "no code to verify" in verify.system_prompt.lower()
        assert verify.system_prompt.strip().lower() != "test all code examples."

    def test_illustrate_offers_both_photo_and_generated_sources(self):
        illus = next(s for s in self._flow().steps if s.role == "illustrate")
        # Must offer real-photo search alongside generation, not generate-only.
        assert "image_search" in illus.tool_names
        assert "image_generation" in illus.tool_names
        sp = illus.system_prompt.lower()
        # FORMAT-aware: procedures lead with real photos, never a stylised model.
        assert "procedure" in sp
        assert "photograph" in sp
        assert "never" in sp  # "a stylised/anime model is NEVER used..."


class TestFlowTopicShapeBranches:
    """Each builder serves a versatile range of topics, not one shape.

    Regression guard for the class of bug where a flow hardcodes one topic
    shape (code / cited-data / data-assertions / fiction) and breaks on the
    wider range its trigger keywords admit — e.g. a personal essay forced to
    carry fake [Source] citations, or a conceptual comparison charted with
    invented numbers. The Plan step must emit a shape label and the
    downstream draft/create steps must branch on it.
    """

    def _step(self, flow, role):
        return next(s for s in flow.steps if s.role == role)

    def test_report_branches_evidence_vs_composition(self):
        from augmentum.reasoning.templates import agentic_report_flow

        flow = agentic_report_flow()
        plan = self._step(flow, "plan").system_prompt.lower()
        assert "format" in plan and "composition" in plan and "evidence" in plan
        # Draft must offer a no-fabrication composition path.
        draft = self._step(flow, "draft").system_prompt.lower()
        assert "composition" in draft
        assert "do not fabricate" in draft or "not fabricate" in draft
        # Review must not ding a composition for missing citations.
        review = self._step(flow, "review").system_prompt.lower()
        assert "fabricated statistic" in review or "fake citation" in review

    def test_presentation_branches_persuasive_vs_educational(self):
        from augmentum.reasoning.templates import agentic_presentation_flow

        flow = agentic_presentation_flow()
        plan = self._step(flow, "plan").system_prompt.lower()
        assert "educational" in plan and "persuasive" in plan
        # The slide-quality anchor (in the draft step) must allow concept titles.
        draft = self._step(flow, "draft").system_prompt.lower()
        assert "educational" in draft
        assert "do not invent data" in draft or "fabricated" in draft

    def test_data_comparison_skips_charts_when_qualitative(self):
        from augmentum.reasoning.templates import agentic_data_comparison_flow

        flow = agentic_data_comparison_flow()
        plan = self._step(flow, "plan").system_prompt.lower()
        assert "qualitative" in plan and "quantitative" in plan
        # The Create Charts step must refuse to invent numbers.
        charts = next(
            s for s in flow.steps
            if "create_chart" in s.tool_names
        ).system_prompt.lower()
        assert "no quantitative data to chart" in charts
        assert "do not invent numbers" in charts

    def test_storybook_branches_fiction_vs_retelling(self):
        from augmentum.reasoning.templates import agentic_storybook_flow

        flow = agentic_storybook_flow()
        plan = self._step(flow, "plan").system_prompt.lower()
        assert "mode" in plan and "retelling" in plan
        # A retelling must not invent appearances/quotes.
        draft = self._step(flow, "draft").system_prompt.lower()
        assert "retelling" in draft
        assert "do not invent" in draft


class TestAgenticArtifactDelivery:
    def test_delivery_cards_accept_artifact_id_key(self):
        from augmentum.modes.agentic.handler import _build_artifact_cards
        from augmentum.modes.agentic.working_memory import WorkingMemory

        wmem = WorkingMemory("deliver app artifact")
        wmem.record_artifact({
            "artifact_id": "art_app",
            "download_url": "/api/artifacts/art_app/download",
            "display_name": "Built App",
            "format": "zip",
        })

        cards = _build_artifact_cards(wmem)

        assert cards[0]["id"] == "art_app"
        assert cards[0]["download_url"] == "/api/artifacts/art_app/download"

    def test_delivery_cards_preserve_raw_card_kind(self):
        from augmentum.modes.agentic.handler import _build_artifact_cards
        from augmentum.modes.agentic.working_memory import WorkingMemory

        wmem = WorkingMemory("deliver chart artifact")
        raw_card = {"kind": "image", "preview": {"image_url": "/api/artifacts/chart/download"}}
        wmem.record_artifact({
            "id": "chart_1",
            "download_url": "/api/artifacts/chart_1/download",
            "display_name": "Chart",
            "format": "png",
            "card": raw_card,
        })

        cards = _build_artifact_cards(wmem)

        assert cards[0]["card"] == raw_card

    def test_delivery_cards_read_nested_ebook_page_type(self):
        from augmentum.modes.agentic.handler import _build_artifact_cards
        from augmentum.modes.agentic.working_memory import WorkingMemory

        wmem = WorkingMemory("deliver ebook artifact")
        wmem.record_artifact({
            "id": "epub_1",
            "download_url": "/api/artifacts/epub_1/download",
            "display_name": "Story.epub",
            "format": "epub",
            "metadata": {"page_type": "ebook", "chapter_count": 3},
            "card": {
                "kind": "artifact",
                "artifact_id": "epub_1",
                "preview": {"format": "epub"},
            },
        })

        cards = _build_artifact_cards(wmem)

        assert cards[0]["page_type"] == "ebook"
        assert cards[0]["format"] == "epub"


# ---------------------------------------------------------------------------
# Flow resolution tests
# ---------------------------------------------------------------------------


class TestFlowResolution:
    @pytest.fixture
    def mock_flow_store(self):
        from augmentum.reasoning.templates import (
            agentic_presentation_flow,
            agentic_report_flow,
            agentic_storybook_flow,
        )

        flows = [
            agentic_report_flow(),
            agentic_presentation_flow(),
            agentic_storybook_flow(),
        ]

        store = AsyncMock()
        store.list_all = AsyncMock(return_value=flows)
        return store

    @pytest.fixture
    def mock_backend(self):
        backend = AsyncMock()
        backend.chat = AsyncMock(return_value=InternalChatResponse(
            message=Message(role="assistant", content="## Task: Test\n\n- [ ] 1. Step"),
            model="test-model",
        ))

        async def fake_stream(req):
            yield InternalStreamChunk(content_delta="Done.", model="test-model")
            yield InternalStreamChunk(content_delta="", model="test-model", done=True)

        backend.chat_stream = fake_stream
        return backend

    @pytest.fixture
    def mock_task_store(self):
        store = AsyncMock()
        store.create = AsyncMock(side_effect=lambda t: t)
        store.update = AsyncMock()
        store.get_incomplete_for_session = AsyncMock(return_value=None)
        return store

    @pytest.mark.asyncio
    async def test_resolves_report_for_report_query(
        self, mock_backend, mock_task_store, mock_flow_store,
    ):
        from augmentum.modes.agentic.handler import AgenticHandler

        handler = AgenticHandler(
            backend=mock_backend,
            session_id="ses_test",
            task_store=mock_task_store,
            flow_store=mock_flow_store,
        )

        # Use the private method directly
        flow = await handler._resolve_agentic_flow(
            "Write a detailed report on renewable energy", "test-model"
        )
        assert flow is not None
        assert flow.name == "Report"

    @pytest.mark.asyncio
    async def test_resolves_presentation_for_slides_query(
        self, mock_backend, mock_task_store, mock_flow_store,
    ):
        from augmentum.modes.agentic.handler import AgenticHandler

        handler = AgenticHandler(
            backend=mock_backend,
            session_id="ses_test",
            task_store=mock_task_store,
            flow_store=mock_flow_store,
        )

        flow = await handler._resolve_agentic_flow(
            "Create a presentation about space exploration", "test-model"
        )
        assert flow is not None
        assert flow.name == "Presentation"

    @pytest.mark.asyncio
    async def test_resolves_storybook_for_story_query(
        self, mock_backend, mock_task_store, mock_flow_store,
    ):
        from augmentum.modes.agentic.handler import AgenticHandler

        handler = AgenticHandler(
            backend=mock_backend,
            session_id="ses_test",
            task_store=mock_task_store,
            flow_store=mock_flow_store,
        )

        flow = await handler._resolve_agentic_flow(
            "Write a fairy tale about a brave little dragon", "test-model"
        )
        assert flow is not None
        assert flow.name == "Storybook"

    @pytest.mark.asyncio
    async def test_resolves_none_if_no_keyword_match(
        self, mock_backend, mock_task_store, mock_flow_store,
    ):
        from augmentum.modes.agentic.handler import AgenticHandler

        handler = AgenticHandler(
            backend=mock_backend,
            session_id="ses_test",
            task_store=mock_task_store,
            flow_store=mock_flow_store,
        )

        flow = await handler._resolve_agentic_flow(
            "Do something interesting", "test-model"
        )
        # No keyword match → returns None (falls back to ad-hoc path)
        assert flow is None

    @pytest.mark.asyncio
    async def test_resolves_none_without_flow_store(
        self, mock_backend, mock_task_store,
    ):
        from augmentum.modes.agentic.handler import AgenticHandler

        handler = AgenticHandler(
            backend=mock_backend,
            session_id="ses_test",
            task_store=mock_task_store,
        )

        flow = await handler._resolve_agentic_flow("Write a report", "test-model")
        assert flow is None

    @pytest.mark.asyncio
    async def test_best_match_with_multiple_keywords(
        self, mock_backend, mock_task_store, mock_flow_store,
    ):
        from augmentum.modes.agentic.handler import AgenticHandler

        handler = AgenticHandler(
            backend=mock_backend,
            session_id="ses_test",
            task_store=mock_task_store,
            flow_store=mock_flow_store,
        )

        # "slides" + "presentation" = 2 matches for presentation flow
        flow = await handler._resolve_agentic_flow(
            "Make slides for a presentation about AI", "test-model"
        )
        assert flow.name == "Presentation"


# ---------------------------------------------------------------------------
# Step role validation
# ---------------------------------------------------------------------------


class TestStepRoles:
    def test_all_agentic_roles_are_recognized(self):
        from augmentum.modes.agentic.handler import _AGENTIC_ROLES

        assert "plan" in _AGENTIC_ROLES
        assert "draft" in _AGENTIC_ROLES
        assert "create" in _AGENTIC_ROLES
        assert "illustrate" in _AGENTIC_ROLES
        assert "review" in _AGENTIC_ROLES
        assert "deliver" in _AGENTIC_ROLES

    def test_report_roles_are_valid(self):
        from augmentum.modes.agentic.handler import _ALL_ROLES
        from augmentum.reasoning.templates import agentic_report_flow

        flow = agentic_report_flow()
        for step in flow.steps:
            assert step.role in _ALL_ROLES, f"Unknown role: {step.role}"

    def test_presentation_roles_are_valid(self):
        from augmentum.modes.agentic.handler import _ALL_ROLES
        from augmentum.reasoning.templates import agentic_presentation_flow

        flow = agentic_presentation_flow()
        for step in flow.steps:
            assert step.role in _ALL_ROLES, f"Unknown role: {step.role}"

    def test_storybook_roles_are_valid(self):
        from augmentum.modes.agentic.handler import _ALL_ROLES
        from augmentum.reasoning.templates import agentic_storybook_flow

        flow = agentic_storybook_flow()
        for step in flow.steps:
            assert step.role in _ALL_ROLES, f"Unknown role: {step.role}"

    def test_data_analysis_roles_are_valid(self):
        from augmentum.modes.agentic.handler import _ALL_ROLES
        from augmentum.reasoning.templates import agentic_data_comparison_flow

        flow = agentic_data_comparison_flow()
        for step in flow.steps:
            assert step.role in _ALL_ROLES, f"Unknown role: {step.role}"


# ---------------------------------------------------------------------------
# Sort order validation
# ---------------------------------------------------------------------------


class TestSortOrder:
    def test_report_sort_order_sequential(self):
        from augmentum.reasoning.templates import agentic_report_flow

        flow = agentic_report_flow()
        orders = [s.sort_order for s in flow.steps]
        assert orders == list(range(len(flow.steps)))

    def test_presentation_sort_order_sequential(self):
        from augmentum.reasoning.templates import agentic_presentation_flow

        flow = agentic_presentation_flow()
        orders = [s.sort_order for s in flow.steps]
        assert orders == list(range(len(flow.steps)))

    def test_storybook_sort_order_sequential(self):
        from augmentum.reasoning.templates import agentic_storybook_flow

        flow = agentic_storybook_flow()
        orders = [s.sort_order for s in flow.steps]
        assert orders == list(range(len(flow.steps)))

    def test_data_analysis_sort_order_sequential(self):
        from augmentum.reasoning.templates import agentic_data_comparison_flow

        flow = agentic_data_comparison_flow()
        orders = [s.sort_order for s in flow.steps]
        assert orders == list(range(len(flow.steps)))


# ---------------------------------------------------------------------------
# Autonomy integration with templates
# ---------------------------------------------------------------------------


class TestTemplateAutonomy:
    def test_create_steps_are_high_impact(self):
        """Create and illustrate steps should trigger approval at level 2."""
        from augmentum.modes.agentic.autonomy import needs_step_approval
        from augmentum.reasoning.templates import agentic_storybook_flow

        flow = agentic_storybook_flow()
        create_step = [s for s in flow.steps if s.role == "create"][0]
        assert needs_step_approval(2, create_step.role) is True

    def test_deliver_steps_not_high_impact(self):
        from augmentum.modes.agentic.autonomy import needs_step_approval
        from augmentum.reasoning.templates import agentic_presentation_flow

        flow = agentic_presentation_flow()
        deliver_step = [s for s in flow.steps if s.role == "deliver"][0]
        # Deliver only presents results — no approval needed at level 2
        assert needs_step_approval(2, deliver_step.role) is False

    def test_search_steps_no_approval_at_level_2(self):
        from augmentum.modes.agentic.autonomy import needs_step_approval
        from augmentum.reasoning.templates import agentic_report_flow

        flow = agentic_report_flow()
        search_step = [s for s in flow.steps if s.role == "search"][0]
        assert needs_step_approval(2, search_step.role) is False

    def test_draft_steps_no_approval_at_level_2(self):
        from augmentum.modes.agentic.autonomy import needs_step_approval
        from augmentum.reasoning.templates import agentic_report_flow

        flow = agentic_report_flow()
        draft_step = [s for s in flow.steps if s.role == "draft"][0]
        assert needs_step_approval(2, draft_step.role) is False

    def test_illustrate_is_high_impact(self):
        """The autonomy contract: 'illustrate' is a high-impact role even
        though the storybook flow no longer uses it (illustrations moved
        into create_ebook._auto_illustrate). The role is still used by
        the PPTX flow and must still require approval at level 2.
        """
        from augmentum.modes.agentic.autonomy import needs_step_approval

        assert needs_step_approval(2, "illustrate") is True


# ---------------------------------------------------------------------------
# PPTX slide-marker parser
# ---------------------------------------------------------------------------

class TestParseSlideDraft:
    """Cover _parse_slide_draft. The parser must accept the marker variants
    LLMs actually emit (#, ##, ###, plain "Slide N:", **Slide N:**) and
    must surface a warning when a large draft collapses to one slide —
    that's the silent-failure mode flagged in the artifact-pipeline audit.
    """

    def test_canonical_h3_marker(self):
        from augmentum.modes.agentic.handler import _parse_slide_draft
        draft = (
            "Here's the deck:\n\n"
            "### Slide 1: Intro\nFirst slide body\n"
            "### Slide 2: Details\nSecond slide body\n"
        )
        slides, warning = _parse_slide_draft(draft)
        assert warning == ""
        assert [s["title"] for s in slides] == ["Intro", "Details"]
        assert slides[0]["body"] == "First slide body"

    def test_h2_marker_accepted(self):
        from augmentum.modes.agentic.handler import _parse_slide_draft
        draft = "## Slide 1: Alpha\nA\n## Slide 2: Beta\nB\n"
        slides, warning = _parse_slide_draft(draft)
        assert warning == ""
        assert len(slides) == 2

    def test_h1_marker_accepted(self):
        from augmentum.modes.agentic.handler import _parse_slide_draft
        draft = "# Slide 1: One\nbody one\n# Slide 2: Two\nbody two\n"
        slides, warning = _parse_slide_draft(draft)
        assert warning == ""
        assert [s["title"] for s in slides] == ["One", "Two"]

    def test_bare_slide_prefix_accepted(self):
        from augmentum.modes.agentic.handler import _parse_slide_draft
        draft = "Slide 1: First\nbody one\nSlide 2: Second\nbody two\n"
        slides, warning = _parse_slide_draft(draft)
        assert warning == ""
        assert [s["title"] for s in slides] == ["First", "Second"]

    def test_bolded_slide_prefix_accepted(self):
        from augmentum.modes.agentic.handler import _parse_slide_draft
        draft = "**Slide 1: A**\nbody A\n**Slide 2: B**\nbody B\n"
        slides, warning = _parse_slide_draft(draft)
        assert warning == ""
        assert [s["title"] for s in slides] == ["A", "B"]

    def test_h3_without_slide_prefix_accepted(self):
        from augmentum.modes.agentic.handler import _parse_slide_draft
        draft = "### Title One\nbody\n### Title Two\nbody\n"
        slides, warning = _parse_slide_draft(draft)
        assert warning == ""
        assert [s["title"] for s in slides] == ["Title One", "Title Two"]

    def test_notes_block_extracted(self):
        from augmentum.modes.agentic.handler import _parse_slide_draft
        draft = (
            "### Slide 1: Title\n"
            "- bullet one\n"
            "- bullet two\n"
            "**Notes:** Speaker says this.\n"
            "Continued speaker note.\n"
        )
        slides, _ = _parse_slide_draft(draft)
        assert slides[0]["body"] == "- bullet one\n- bullet two"
        assert "Speaker says this" in slides[0]["notes"]
        assert "Continued speaker note" in slides[0]["notes"]

    def test_large_draft_with_no_markers_warns(self):
        """The collapse case the audit flagged: LLM emits a long unstructured
        draft, parser falls back to one slide, but must SURFACE the issue."""
        from augmentum.modes.agentic.handler import _parse_slide_draft
        draft = "This is a long unstructured presentation. " * 40  # ~1700 chars
        slides, warning = _parse_slide_draft(draft, fallback_title="Talk")
        assert len(slides) == 1
        assert slides[0]["title"] == "Talk"
        assert warning  # non-empty
        assert "no slide markers" in warning

    def test_short_draft_with_no_markers_no_warn(self):
        """Short drafts may legitimately be one slide; don't cry wolf."""
        from augmentum.modes.agentic.handler import _parse_slide_draft
        draft = "Short note for a single slide."
        slides, warning = _parse_slide_draft(draft, fallback_title="Note")
        assert len(slides) == 1
        assert warning == ""

    def test_mixed_marker_styles_in_one_draft(self):
        """LLMs sometimes start canonical and drift mid-draft."""
        from augmentum.modes.agentic.handler import _parse_slide_draft
        draft = (
            "### Slide 1: First\nbody A\n"
            "## Slide 2: Second\nbody B\n"
            "Slide 3: Third\nbody C\n"
        )
        slides, warning = _parse_slide_draft(draft)
        assert warning == ""
        assert [s["title"] for s in slides] == ["First", "Second", "Third"]

    def test_case_insensitive_marker(self):
        from augmentum.modes.agentic.handler import _parse_slide_draft
        draft = "### slide 1: lower\nbody\n### SLIDE 2: UPPER\nbody\n"
        slides, _ = _parse_slide_draft(draft)
        assert len(slides) == 2
