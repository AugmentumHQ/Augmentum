"""Smoke tests — every module under modes/ imports and key classes construct."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest


class TestBaseImports:
    """Base handler module imports."""

    def test_import_base(self):
        from augmentum.modes.base import ModeHandler
        assert ModeHandler is not None

    def test_import_v_command(self):
        from augmentum.modes.v_command import extract_v_command, generate_direct_image
        assert extract_v_command is not None
        assert generate_direct_image is not None


class TestPassthroughImports:
    """Passthrough mode module imports."""

    def test_import_handler(self):
        from augmentum.modes.passthrough.handler import PassthroughHandler
        assert PassthroughHandler is not None

    def test_import_orchestrator(self):
        from augmentum.modes.passthrough.orchestrator import SSOSOrchestrator
        assert SSOSOrchestrator is not None

    def test_construct_handler(self):
        from augmentum.modes.passthrough.handler import PassthroughHandler
        backend = MagicMock()
        handler = PassthroughHandler(backend=backend)
        assert handler._backend is backend

    def test_construct_orchestrator(self):
        from augmentum.modes.passthrough.orchestrator import SSOSOrchestrator
        registry = MagicMock()
        orch = SSOSOrchestrator(tool_registry=registry)
        assert orch._registry is registry


class TestAnalyticalImports:
    """Analytical mode module imports."""

    def test_import_handler(self):
        from augmentum.modes.analytical.handler import AnalyticalHandler
        assert AnalyticalHandler is not None

    def test_import_engine(self):
        from augmentum.modes.analytical.engine import AnalyticalEngine
        assert AnalyticalEngine is not None

    def test_import_state(self):
        from augmentum.modes.analytical.state import (
            AnalyticalPhase,
            AnalyticalResult,
            AnalyticalState,
            PhaseResult,
            ToolCallRecord,
        )
        assert AnalyticalPhase.ASSESS.value == "assess"

    def test_import_prompts(self):
        from augmentum.modes.analytical.prompts import get_phase_prompt
        assert callable(get_phase_prompt)

    def test_import_tool_calling(self):
        from augmentum.modes.analytical.tool_calling import (
            ToolCallingTier,
            select_tier,
            tools_to_native_format,
            parse_native_tool_call,
            parse_structured_output,
            extract_structured_text,
            parse_python_style_tool_call,
            coerce_tool_params,
        )
        assert ToolCallingTier.NATIVE.value == "native"

    def test_import_auto_verify(self):
        from augmentum.modes.analytical.auto_verify import (
            run_auto_verification,
            extract_math_expressions,
            extract_code_blocks,
            VerificationCheck,
            AutoVerifyResult,
        )
        assert callable(run_auto_verification)

    def test_construct_state(self):
        from augmentum.modes.analytical.state import AnalyticalState
        state = AnalyticalState(query="test")
        assert state.query == "test"
        assert state.complexity == "moderate"


class TestAgenticImports:
    """Agentic mode module imports."""

    def test_import_handler(self):
        from augmentum.modes.agentic.handler import AgenticHandler
        assert AgenticHandler is not None

    def test_import_planner(self):
        from augmentum.modes.agentic.planner import (
            parse_plan,
            update_plan_step,
            mark_current_step,
            plan_to_context,
            PLAN_SYSTEM_PROMPT,
        )
        assert callable(parse_plan)

    def test_import_working_memory(self):
        from augmentum.modes.agentic.working_memory import WorkingMemory
        assert WorkingMemory is not None

    def test_import_task_state(self):
        from augmentum.modes.agentic.task_state import TaskState, TaskStatus, TaskStore
        assert TaskStatus.PLANNING.value == "planning"

    def test_import_autonomy(self):
        from augmentum.modes.agentic.autonomy import (
            needs_plan_approval,
            needs_step_approval,
            build_approval_chunk,
            build_inform_chunk,
        )
        assert callable(needs_plan_approval)

    def test_construct_handler(self):
        from augmentum.modes.agentic.handler import AgenticHandler
        backend = MagicMock()
        handler = AgenticHandler(backend=backend)
        assert handler._backend is backend

    def test_construct_working_memory(self):
        from augmentum.modes.agentic.working_memory import WorkingMemory
        wm = WorkingMemory(goal="test task")
        assert wm.goal == "test task"

    def test_construct_task_state(self):
        from augmentum.modes.agentic.task_state import TaskState, TaskStatus
        ts = TaskState(title="Test Task", total_steps=3)
        assert ts.title == "Test Task"
        assert ts.status == TaskStatus.PLANNING


class TestNarrativeImports:
    """Narrative mode module imports."""

    def test_import_handler(self):
        from augmentum.modes.narrative.handler import NarrativeHandler
        assert NarrativeHandler is not None

    def test_import_engine(self):
        from augmentum.modes.narrative.engine import NarrativeEngine
        assert NarrativeEngine is not None

    def test_import_character_tracker(self):
        from augmentum.modes.narrative.character_tracker import CharacterTracker, CharacterUpdate
        assert CharacterTracker is not None

    def test_import_world_tracker(self):
        from augmentum.modes.narrative.world_tracker import WorldTracker, SceneState
        assert WorldTracker is not None

    def test_import_plot_tracker(self):
        from augmentum.modes.narrative.plot_tracker import PlotTracker, PlotUpdate
        assert PlotTracker is not None

    def test_import_relationship_tracker(self):
        from augmentum.modes.narrative.relationship_tracker import RelationshipTracker, Relationship
        assert RelationshipTracker is not None

    def test_import_branch_tracker(self):
        from augmentum.modes.narrative.branch_tracker import BranchTracker, BranchDetection
        assert BranchTracker is not None

    def test_import_group_manager(self):
        from augmentum.modes.narrative.group_manager import CharacterGroup, GroupTurnManager, GroupStore
        assert GroupTurnManager is not None

    def test_import_llm_extractor(self):
        from augmentum.modes.narrative.llm_extractor import NarrativeExtraction
        assert NarrativeExtraction is not None

    def test_import_card_parser(self):
        from augmentum.modes.narrative.card_parser import CardParser, CharacterCard
        assert CardParser is not None

    def test_import_regex_transformer(self):
        from augmentum.modes.narrative.regex_transformer import RegexScript, apply_regex_scripts
        assert callable(apply_regex_scripts)

    def test_import_regex_presets(self):
        import augmentum.modes.narrative.regex_presets
        assert augmentum.modes.narrative.regex_presets is not None

    def test_import_prompt_presets(self):
        from augmentum.modes.narrative.prompt_presets import PromptPreset
        assert PromptPreset is not None

    def test_import_macro_expander(self):
        from augmentum.modes.narrative.macro_expander import expand_macros, expand_messages
        assert callable(expand_macros)

    def test_import_memory_settings(self):
        from augmentum.modes.narrative.memory_settings import (
            SessionMemorySettings,
            resolve_memory_setting,
        )
        assert SessionMemorySettings is not None

    def test_import_world_info_buffer(self):
        from augmentum.modes.narrative.world_info_buffer import WorldInfoBuffer
        assert WorldInfoBuffer is not None

    def test_import_world_info_groups(self):
        from augmentum.modes.narrative.world_info_groups import filter_by_groups
        assert callable(filter_by_groups)

    def test_import_context_builder(self):
        from augmentum.modes.narrative.context_builder import ContextBuilder, BuiltContext
        assert ContextBuilder is not None

    def test_import_lore_engine(self):
        from augmentum.modes.narrative.lore_engine import LoreEngine, match_keywords
        assert LoreEngine is not None

    def test_import_memory(self):
        from augmentum.modes.narrative.memory import (
            CardType,
            MemoryEntry,
            StateSnapshot,
            SummaryMode,
            detect_card_type,
        )
        assert CardType is not None

    def test_construct_engine(self):
        from augmentum.modes.narrative.engine import NarrativeEngine
        engine = NarrativeEngine(session_id="test-session")
        assert engine._session_id == "test-session"


class TestCoderImports:
    """Coder mode module imports."""

    def test_import_handler(self):
        from augmentum.modes.coder.handler import CoderHandler
        assert CoderHandler is not None

    def test_construct_handler(self):
        from augmentum.modes.coder.handler import CoderHandler
        backend = MagicMock()
        handler = CoderHandler(backend=backend, session_id="test")
        assert handler._backend is backend
        assert handler._session_id == "test"

    def test_construct_handler_accepts_provider_registry(self):
        """``CoderHandler.__init__`` accepts the ``provider_registry`` kwarg
        the factory passes from app_state. Single-model setups pass ``None``;
        the handler must not blow up — plan-phase role resolution falls back
        to the bound backend in that case.
        """
        from augmentum.modes.coder.handler import CoderHandler
        backend = MagicMock()
        registry = MagicMock()
        handler = CoderHandler(
            backend=backend,
            session_id="test",
            provider_registry=registry,
        )
        assert handler._provider_registry is registry
        # None must also work (older construction paths, tests)
        handler_no_reg = CoderHandler(backend=backend, session_id="test")
        assert handler_no_reg._provider_registry is None


class TestCoderPromptMetadata:
    """PromptMeta substrate — versioning, role mapping, allow/deny drift.

    These tests guard the Week-1 wire-up: ``PromptMeta`` declares which
    tools the dispatcher should keep out of plan-phase schemas, and the
    deny-list must stay complementary to the long-standing ``READ_ONLY_TOOLS``
    allow-list (also used by phase_act for parallelism classification).
    Drift between the two surfaces is a classification bug we want to
    catch in CI, not in production.
    """

    def test_prompt_registry_populated(self):
        """Every documented prompt has metadata. Version + name set."""
        from augmentum.coder.prompts import PROMPT_REGISTRY
        expected = {
            "coder.workspace_guide",
            "coder.tool_reference",
            "coder.plan",
            "coder.mission_plan",
            "coder.mission_replan",
            "coder.mission_act",
            "coder.act",
            "coder.edit_format",
            "coder.native",
        }
        assert set(PROMPT_REGISTRY.keys()) == expected
        for meta in PROMPT_REGISTRY.values():
            assert meta.version, f"{meta.name} has empty version"
            assert "." in meta.name, f"{meta.name} doesn't follow coder.* convention"

    def test_plan_meta_disallows_writes_and_executes(self):
        """Plan phase must reject the canonical write/execute tools.

        These are the verbs whose absence in the schema makes plan
        structurally read-only — the prompt's prose ban is reinforcement,
        not the only line of defense.
        """
        from augmentum.coder.prompts import PLAN_META
        for tool in ("file_write", "code_edit", "shell_exec", "test_run", "git"):
            assert tool in PLAN_META.disallowed_tools, (
                f"{tool} should be denied during plan phase"
            )

    def test_plan_meta_uses_utility_role(self):
        """Plan resolves through the smarter ``utility`` model tier."""
        from augmentum.coder.prompts import PLAN_META, MISSION_PLAN_META
        assert PLAN_META.model_role == "utility"
        assert MISSION_PLAN_META.model_role == "utility"

    def test_act_meta_uses_bound_backend(self):
        """Act phase does not remap — uses the user's primary chat model.

        Empty ``model_role`` is the explicit "no remap" signal for the
        dispatcher; act stays on the request's bound backend so the user's
        active chat model is what runs iteration-heavy work.
        """
        from augmentum.coder.prompts import ACT_META, NATIVE_META, MISSION_ACT_META
        assert ACT_META.model_role == ""
        assert NATIVE_META.model_role == ""
        assert MISSION_ACT_META.model_role == ""

    def test_disallowed_and_read_only_are_complementary(self):
        """Plan-phase deny-list and the read-only allow-list must not overlap.

        Both surfaces describe plan-phase tool safety from opposite sides
        (allow vs deny). Overlap means a tool was classified as
        ``read-only`` AND ``forbidden during plan`` — a contradiction that
        signals classification drift. The deny still wins at runtime
        (defense in depth), but the contradiction should be cleaned up
        rather than papered over.
        """
        from augmentum.coder.prompts import PLAN_META
        from augmentum.coder.tools import READ_ONLY_TOOLS
        overlap = set(PLAN_META.disallowed_tools) & READ_ONLY_TOOLS
        assert not overlap, (
            f"PLAN_META.disallowed_tools and READ_ONLY_TOOLS overlap on: "
            f"{sorted(overlap)}. Each tool must be in at most one list."
        )

    def test_prompt_profile_for_strategy_shapes(self):
        """The ledger ``prompt_profile`` field carries strategy-specific
        ``name@version`` pairs separated by ``;`` — analytics joins on this."""
        from augmentum.coder.prompts import prompt_profile_for_strategy

        # Default strategies use plan + act
        for strategy in ("hybrid", "canonical", "", "unknown-strategy"):
            profile = prompt_profile_for_strategy(strategy)
            assert "coder.workspace_guide@" in profile
            assert "coder.plan@" in profile
            assert "coder.act@" in profile
            assert "coder.native@" not in profile

        # Native strategy skips plan
        native = prompt_profile_for_strategy("native")
        assert "coder.native@" in native
        assert "coder.plan@" not in native
        assert "coder.act@" not in native

        # Legacy uses mission prompts
        legacy = prompt_profile_for_strategy("legacy")
        assert "coder.mission_plan@" in legacy
        assert "coder.mission_act@" in legacy

    def test_dispatch_fork_prompt_registered(self):
        """The orchestrator-spawn dispatch prompt has metadata and the
        renderer's expected variables are declared."""
        from augmentum.coder.prompts import (
            DISPATCH_FORK_META, DISPATCH_FORK_SYSTEM, PROMPT_REGISTRY,
        )
        assert "coder.dispatch_fork" in PROMPT_REGISTRY
        assert DISPATCH_FORK_META.agent_role == "fork"
        # Every placeholder declared in variables must appear in the prompt
        for var in DISPATCH_FORK_META.variables:
            assert f"${{{var}}}" in DISPATCH_FORK_SYSTEM, (
                f"declared variable {var!r} not present in DISPATCH_FORK_SYSTEM"
            )


class TestCoderDispatch:
    """Orchestrator → coder dispatch contract.

    Today there's no production caller — the user-direct path runs
    through ``CoderHandler.process_stream`` and bypasses dispatch.
    These tests pin the contract's shape so future Becca-side wiring
    composes against a stable target.
    """

    def test_direct_user_dispatch_round_trip(self):
        """The minimal direct-user dispatch renders without external context.

        Matches today's no-orchestrator behavior: empty success criteria,
        no brief, default tiers. The TQG handles termination the way it
        does for direct-user turns.
        """
        from augmentum.coder import dispatch as d
        from augmentum.coder.prompts import DISPATCH_FORK_SYSTEM
        disp = d.CoderDispatch.for_direct_user_turn(
            workspace_id="ws_x", user_id="u_x", task="Fix the login bug",
        )
        rendered = d.render_dispatch_system(disp, fork_prompt=DISPATCH_FORK_SYSTEM)
        assert "Fix the login bug" in rendered
        assert "(none specified)" in rendered  # empty constraints
        assert "falls back to user-demand heuristic" in rendered  # empty criteria
        assert "(no relational context passed through" in rendered
        assert "cost_tier: balanced" in rendered

    def test_dispatch_renders_promise_criteria(self):
        """Each Promise in success_criteria surfaces description + verify kind."""
        from augmentum.coder import dispatch as d
        from augmentum.coder.prompts import DISPATCH_FORK_SYSTEM
        from augmentum.promises.models import (
            Promise, Verification, VerificationKind,
        )
        disp = d.CoderDispatch(
            workspace_id="ws", user_id="u", task="Build a thing",
            success_criteria=(
                Promise(
                    description="file written",
                    verify=Verification(
                        kind=VerificationKind.FILE,
                        spec={"path": "/workspace/thing.py"},
                    ),
                ),
                Promise(
                    description="smoke shell passes",
                    verify=Verification(
                        kind=VerificationKind.SHELL,
                        spec={"cmd": "python /workspace/thing.py"},
                    ),
                ),
            ),
        )
        rendered = d.render_dispatch_system(disp, fork_prompt=DISPATCH_FORK_SYSTEM)
        assert "file written" in rendered
        assert "verify: file(path='/workspace/thing.py')" in rendered
        assert "smoke shell passes" in rendered
        assert "verify: shell(cmd=" in rendered

    def test_dispatch_renders_brief_with_conventional_and_custom_keys(self):
        """Brief surfaces conventional keys nicely and falls through unknown keys."""
        from augmentum.coder import dispatch as d
        from augmentum.coder.prompts import DISPATCH_FORK_SYSTEM
        disp = d.CoderDispatch(
            workspace_id="ws", user_id="u", task="anything",
            context_brief={
                "user_mood": "focused",
                "recent_decisions": ["picked FastAPI"],
                "custom_field": "arbitrary",
            },
        )
        rendered = d.render_dispatch_system(disp, fork_prompt=DISPATCH_FORK_SYSTEM)
        assert "user_mood" in rendered
        assert "focused" in rendered
        assert "picked FastAPI" in rendered
        assert "custom_field" in rendered  # non-conventional keys fall through

    def test_dispatch_renders_runtime_parameters(self):
        """cost_tier / parallelism / permission_mode all surface in the prompt."""
        from augmentum.coder import dispatch as d
        from augmentum.coder.prompts import DISPATCH_FORK_SYSTEM
        disp = d.CoderDispatch(
            workspace_id="ws", user_id="u", task="anything",
            cost_tier="thorough", parallelism=3,
            permission_mode="confirm_mutations",
        )
        rendered = d.render_dispatch_system(disp, fork_prompt=DISPATCH_FORK_SYSTEM)
        assert "cost_tier: thorough" in rendered
        assert "parallelism: 3" in rendered
        assert "permission_mode: confirm_mutations" in rendered

    def test_dispatch_braces_in_task_survive_substitution(self):
        """Task text containing ``{x}`` must not be eaten by string formatting.

        We use ``str.replace``-with-``${VAR}``-markers rather than
        ``str.format`` exactly so user-provided text (which routinely
        contains braces in code / template references) doesn't trigger
        format-string parsing.
        """
        from augmentum.coder import dispatch as d
        from augmentum.coder.prompts import DISPATCH_FORK_SYSTEM
        disp = d.CoderDispatch.for_direct_user_turn(
            workspace_id="ws", user_id="u",
            task="Replace {foo} placeholders in templates",
        )
        rendered = d.render_dispatch_system(disp, fork_prompt=DISPATCH_FORK_SYSTEM)
        assert "{foo}" in rendered  # literal braces preserved
