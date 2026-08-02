"""Mode-handler settings — coder, UARF (analytical), passthrough chains,
agentic, narrative-adjacent analytical, and the architect dispatch layer.

Grouped here because each is a thin pocket of settings (3-20 each) and
they share the 'how a mode behaves' framing.
"""

from __future__ import annotations

from augmentum.registry.registry import SettingsRegistry
from augmentum.registry.settings import Setting

_CODER = ("coder",)
_CODER_ADV = ("coder", "advanced")
_UARF = ("uarf", "analytical")
_UARF_ADV = ("uarf", "analytical", "advanced")
_PT = ("passthrough", "advanced")
_AGENTIC = ("agentic",)
_ANALYTICAL = ("analytical",)
_ARCHITECT = ("architect", "advanced")


def register(r: SettingsRegistry) -> None:
    # ============== Coder ==============
    r.register(
        Setting(
            key="coder_subagents_enabled",
            kind="bool",
            default=True,
            label="Coder subagents",
            description=(
                "Allow the coder handler to spawn task-dispatch subagents "
                "(the Claude Code-style Task tool path)."
            ),
            section="coder.subagents",
            tags=_CODER,
            voice_aliases=("coder subagents", "subagent dispatch"),
            trust_tier="local_significant",
        )
    )
    r.register(
        Setting(
            key="coder_subagent_max_concurrent",
            kind="int",
            default=4,
            label="Subagent concurrency",
            description=(
                "How many subagents the coder can run in parallel per turn."
            ),
            section="coder.subagents",
            min_value=1,
            max_value=16,
            tags=_CODER_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="coder_subagent_max_depth",
            kind="int",
            default=1,
            label="Subagent max depth",
            description=(
                "How deeply subagents can recursively dispatch. 1 = no "
                "nested subagent calls. Caps fan-out."
            ),
            section="coder.subagents",
            min_value=1,
            max_value=4,
            tags=_CODER_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="coder_next_speaker_check_enabled",
            kind="bool",
            default=True,
            label="Next-speaker check",
            description=(
                "Run the next-speaker classifier so the coder knows whether "
                "to expect a user reply or keep working."
            ),
            section="coder.behavior",
            tags=_CODER_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )

    # ---- Compaction ----
    r.register(
        Setting(
            key="coder_compaction_auto_enabled",
            kind="bool",
            default=True,
            label="Auto-compaction",
            description=(
                "Auto-compact coder turn history when context fills up. "
                "Disabling forces manual /compact."
            ),
            section="coder.compaction",
            tags=_CODER,
            trust_tier="local_significant",
        )
    )
    r.register(
        Setting(
            key="coder_compaction_threshold",
            kind="float",
            default=0.65,
            label="Compaction trigger",
            description=(
                "Fraction of context filled before auto-compaction triggers. "
                "Lower = compacts earlier (loses less detail per pass)."
            ),
            section="coder.compaction",
            min_value=0.3,
            max_value=0.95,
            tags=_CODER_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="coder_compaction_keep_recent",
            kind="int",
            default=0,
            label="Compaction keep-recent",
            description=(
                "How many recent turns are kept verbatim through compaction. "
                "0 = use the engine's default heuristic."
            ),
            section="coder.compaction",
            min_value=4,
            max_value=40,
            tags=_CODER_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )

    # ---- Archive ----
    r.register(
        Setting(
            key="coder_archive_enabled",
            kind="bool",
            default=True,
            label="Turn archive",
            description=(
                "Archive coder turns to disk for later inspection / rewind. "
                "Required for the rewind feature to work."
            ),
            section="coder.archive",
            tags=_CODER,
            trust_tier="local_significant",
        )
    )
    r.register(
        Setting(
            key="coder_archive_max_turns_per_workspace",
            kind="int",
            default=0,
            label="Archive cap per workspace",
            description=(
                "Maximum archived turns retained per workspace. 0 = unlimited "
                "(the recommended default — rewind needs depth)."
            ),
            section="coder.archive",
            min_value=0,
            max_value=1_000_000,
            tags=_CODER_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )

    # ---- Breakers ----
    breakers: list[tuple[str, str, int, str, str]] = [
        (
            "coder_breaker_validation_error_streak",
            "Validation-error streak break",
            100,
            "Halt the coder when N consecutive turns produce a validation error.",
            "Trips loops where the same lint/typecheck error recurs without progress.",
        ),
        (
            "coder_breaker_same_validation_error_repeat",
            "Same-validation-error repeat",
            100,
            "Halt when the SAME validation error repeats N times.",
            "Catches loops on a specific error string.",
        ),
        (
            "coder_breaker_action_stagnation_break",
            "Action stagnation break",
            200,
            "Halt when N actions produce no visible state change.",
            "Stops infinite no-op loops.",
        ),
        (
            "coder_breaker_test_failure_streak",
            "Test-failure streak break",
            100,
            "Halt when tests fail for N consecutive turns.",
            "Stops the coder from grinding on a test that's actually impossible to make pass.",
        ),
        (
            "coder_breaker_same_file_edit_break",
            "Same-file-edit break",
            200,
            "Halt when the same file is edited N consecutive turns without progress.",
            "Detects file-thrash on one location.",
        ),
        (
            "coder_breaker_no_write_progress_break",
            "No-write-progress break",
            200,
            "Halt when N turns pass with no file writes at all.",
            "Stops planning-loop chatter that never produces code.",
        ),
        (
            "coder_breaker_inspection_loop_nudge",
            "Inspection-loop nudge",
            100,
            "Nudge the coder after N read-only turns in a row.",
            "Reads only, no writes — the nudge says 'time to act'.",
        ),
        (
            "coder_breaker_inspection_loop_break",
            "Inspection-loop break",
            100,
            "Halt after N read-only turns if the nudge doesn't unblock.",
            "Hard ceiling on read-only inspection loops.",
        ),
    ]
    for key, label, max_v, line1, line2 in breakers:
        r.register(
            Setting(
                key=key,
                kind="int",
                default=0,
                label=label,
                description=f"{line1} 0 = breaker disabled. {line2}",
                section="coder.breakers",
                min_value=0,
                max_value=max_v,
                tags=_CODER_ADV,
                advanced=True,
                trust_tier="admin_only",
            )
        )

    # ---- Iteration caps ----
    r.register(
        Setting(
            key="coder_hybrid_max_iters",
            kind="int",
            default=0,
            label="Hybrid max iterations",
            description=(
                "Per-turn iteration cap for the hybrid strategy. 0 = use the "
                "engine default (recommended)."
            ),
            section="coder.iteration",
            min_value=0,
            max_value=5000,
            tags=_CODER_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="coder_hybrid_max_iters_ungated",
            kind="int",
            default=0,
            label="Hybrid max iterations (ungated)",
            description=(
                "Iteration cap for hybrid when the next-speaker gate is off. "
                "Higher than the gated cap because no breakpoint is forced."
            ),
            section="coder.iteration",
            min_value=0,
            max_value=10000,
            tags=_CODER_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="coder_native_nudge_max",
            kind="int",
            default=0,
            label="Native-strategy nudge cap",
            description=(
                "How many corrective nudges the native strategy emits per "
                "turn. 0 = engine default."
            ),
            section="coder.iteration",
            min_value=0,
            max_value=10,
            tags=_CODER_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )

    # ---- Token budgets ----
    r.register(
        Setting(
            key="coder_context_reserve_pct",
            kind="float",
            default=0.10,
            label="Context reserve",
            description=(
                "Fraction of context window reserved for output + tool "
                "schemas + reasoning. Lower = more window for input, "
                "higher = safer margin."
            ),
            section="coder.budget",
            min_value=0.02,
            max_value=0.40,
            tags=_CODER_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="coder_file_write_max_tokens",
            kind="int",
            default=0,
            label="File-write max tokens",
            description=(
                "Per-call cap on file_write output tokens. 0 = uncapped "
                "(matches Claude Code / Codex CLI). Raise only for weak "
                "local models with tiny output budgets."
            ),
            section="coder.budget",
            min_value=0,
            max_value=32000,
            tags=_CODER_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="coder_local_max_tokens_pct",
            kind="int",
            default=25,
            label="Local backend output %",
            description=(
                "For local backends, output budget as a percentage of context. "
                "0 = fall back to flat mode-hint default."
            ),
            section="coder.budget",
            min_value=0,
            max_value=90,
            tags=_CODER_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="coder_local_max_tokens_cap",
            kind="int",
            default=32768,
            label="Local output cap",
            description=(
                "Absolute ceiling on the computed local output budget. 0 = "
                "no absolute cap (use the % directly)."
            ),
            section="coder.budget",
            min_value=0,
            max_value=262144,
            tags=_CODER_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )

    # ---- Pause / idle ----
    r.register(
        Setting(
            key="coder_pause_idle",
            kind="bool",
            default=True,
            label="Auto-pause idle workspaces",
            description=(
                "Pause coder workspaces that have been idle to lift active-"
                "credit pressure without losing state."
            ),
            section="coder.lifecycle",
            tags=_CODER,
            trust_tier="local_significant",
        )
    )
    r.register(
        Setting(
            key="coder_pause_stop_after_seconds",
            kind="int",
            default=21600,
            label="Auto-stop after pause (s)",
            description=(
                "After this many seconds paused, the workspace is fully "
                "stopped (state archived to disk). Default 6h."
            ),
            section="coder.lifecycle",
            min_value=0,
            max_value=7 * 86_400,
            tags=_CODER_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="coder_max_paused_seconds",
            kind="int",
            default=1800,
            label="Max paused (s)",
            description=(
                "Maximum continuous paused time before auto-resume or stop "
                "logic kicks in. Caps zombie paused workspaces."
            ),
            section="coder.lifecycle",
            min_value=0,
            max_value=86_400,
            tags=_CODER_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="coder_paused_sweep_interval_s",
            kind="int",
            default=60,
            label="Paused sweep interval (s)",
            description=(
                "How often the lifecycle sweeper checks paused workspaces "
                "for resume/stop conditions."
            ),
            section="coder.lifecycle",
            min_value=0,
            max_value=600,
            tags=_CODER_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )

    # ---- Workspace PID guard ----
    r.register(
        Setting(
            key="coder_workspace_pids_limit",
            kind="int",
            default=1024,
            label="Workspace PID cap",
            description=(
                "Per-workspace PID hard cap inside the coder container. "
                "Prevents fork-bombs from taking down the host."
            ),
            section="coder.security",
            min_value=256,
            max_value=16_384,
            tags=("coder", "security"),
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="coder_workspace_pids_warn_pct",
            kind="float",
            default=0.80,
            label="Workspace PID warn %",
            description=(
                "Warn when PID usage crosses this fraction of the cap."
            ),
            section="coder.security",
            min_value=0.0,
            max_value=0.99,
            tags=("coder", "security", "advanced"),
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="coder_workspace_pids_check_interval_s",
            kind="int",
            default=120,
            label="Workspace PID check interval (s)",
            description=(
                "How often the PID-guard checks workspace processes. Cheap."
            ),
            section="coder.security",
            min_value=0,
            max_value=3600,
            tags=("coder", "security", "advanced"),
            advanced=True,
            trust_tier="admin_only",
        )
    )

    # ============== UARF (analytical) ==============
    r.register(
        Setting(
            key="uarf_auto_search",
            kind="bool",
            default=True,
            label="Auto web search",
            description=(
                "Let the analytical mode autonomously decide to search the web "
                "during a turn."
            ),
            section="uarf.search",
            tags=_UARF,
            voice_aliases=("auto search", "web search"),
            trust_tier="local_reversible",
        )
    )
    r.register(
        Setting(
            key="uarf_auto_search_queries",
            kind="int",
            default=5,
            label="Auto-search queries per turn",
            description=(
                "How many distinct search queries the analytical mode can "
                "issue per turn."
            ),
            section="uarf.search",
            min_value=1,
            max_value=10,
            tags=_UARF,
            trust_tier="local_reversible",
        )
    )
    r.register(
        Setting(
            key="uarf_auto_search_results_per_query",
            kind="int",
            default=5,
            label="Results per query",
            description=(
                "How many search results to consume per query. Higher = more "
                "context, more tokens."
            ),
            section="uarf.search",
            min_value=1,
            max_value=10,
            tags=_UARF,
            trust_tier="local_reversible",
        )
    )
    r.register(
        Setting(
            key="uarf_auto_search_max_context_chars",
            kind="int",
            default=24000,
            label="Search context cap (chars)",
            description=(
                "Total character cap on injected search context per turn. "
                "Cuts oldest results first when exceeded."
            ),
            section="uarf.search",
            min_value=1000,
            max_value=128000,
            tags=_UARF_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="uarf_search_retry_max",
            kind="int",
            default=1,
            label="Search retry cap",
            description=(
                "Maximum retries when a search returns too few results."
            ),
            section="uarf.search",
            min_value=0,
            max_value=5,
            tags=_UARF_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="uarf_search_retry_min_results",
            kind="int",
            default=2,
            label="Search retry threshold",
            description=(
                "Trigger a retry when a query returns fewer than N results."
            ),
            section="uarf.search",
            min_value=0,
            max_value=10,
            tags=_UARF_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="uarf_auto_verify",
            kind="bool",
            default=True,
            label="Auto-verify",
            description=(
                "Run a verification pass (cross-model check) before the final "
                "synthesis. Catches hallucinations; adds latency."
            ),
            section="uarf.verify",
            tags=_UARF,
            voice_aliases=("auto verify", "fact check"),
            trust_tier="local_reversible",
        )
    )
    r.register(
        Setting(
            key="uarf_verify_model",
            kind="str",
            default="",
            label="Verification model",
            description=(
                "Model used for the cross-verification pass. Empty = use the "
                "default backend model."
            ),
            section="uarf.verify",
            max_length=256,
            tags=_UARF_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="uarf_proactive_search",
            kind="bool",
            default=True,
            label="Proactive search",
            description=(
                "Surface a search even when the user didn't explicitly ask "
                "for one, when the topic warrants."
            ),
            section="uarf.proactive",
            tags=_UARF,
            trust_tier="local_reversible",
        )
    )
    r.register(
        Setting(
            key="uarf_proactive_math",
            kind="bool",
            default=True,
            label="Proactive math",
            description=(
                "Auto-invoke the calculator / math eval tool when arithmetic "
                "appears."
            ),
            section="uarf.proactive",
            tags=_UARF,
            trust_tier="local_reversible",
        )
    )
    r.register(
        Setting(
            key="uarf_proactive_code",
            kind="bool",
            default=True,
            label="Proactive code execution",
            description=(
                "Auto-run sandboxed code when the model emits code blocks "
                "that would benefit from running."
            ),
            section="uarf.proactive",
            tags=_UARF,
            trust_tier="local_significant",
        )
    )
    r.register(
        Setting(
            key="uarf_heuristic_assess",
            kind="bool",
            default=True,
            label="Heuristic assess",
            description=(
                "Run the lightweight pre-LLM heuristic assessor that chooses "
                "which UARF phases to run for this query."
            ),
            section="uarf.routing",
            tags=_UARF_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="uarf_max_tool_calls_per_phase",
            kind="int",
            default=3,
            label="Tool calls per phase",
            description=(
                "Cap on tool invocations within a single UARF phase."
            ),
            section="uarf.routing",
            min_value=1,
            max_value=10,
            tags=_UARF_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="uarf_conversation_max_chars",
            kind="int",
            default=4000,
            label="Conversation max chars",
            description=(
                "Character cap on the conversation context UARF injects per "
                "turn. Cuts oldest first."
            ),
            section="uarf.routing",
            min_value=500,
            max_value=32000,
            tags=_UARF_ADV,
            advanced=True,
            trust_tier="admin_only",
        )
    )

    # ============== Memory ==============
    r.register(
        Setting(
            key="memory_dedup_threshold",
            kind="float",
            default=0.88,
            label="Memory dedup threshold",
            description=(
                "Semantic similarity above which two memory candidates are "
                "considered duplicates and merged. Higher = stricter."
            ),
            section="memory.dedup",
            min_value=0.5,
            max_value=1.0,
            tags=("memory", "advanced"),
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="memory_contradiction_threshold",
            kind="float",
            default=0.78,
            label="Contradiction threshold",
            description=(
                "Semantic similarity above which a new fact is checked for "
                "contradiction with existing facts."
            ),
            section="memory.contradiction",
            min_value=0.3,
            max_value=1.0,
            tags=("memory", "advanced"),
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="memory_llm_extraction_model",
            kind="str",
            default="",
            label="Memory extraction model",
            description=(
                "LLM used to extract facts from chat turns. Empty = default "
                "backend model."
            ),
            section="memory.extraction",
            max_length=256,
            tags=("memory", "advanced"),
            advanced=True,
            trust_tier="admin_only",
        )
    )

    # ============== Passthrough chains ==============
    r.register(
        Setting(
            key="passthrough_chain_enabled",
            kind="bool",
            default=True,
            label="Passthrough chains",
            description=(
                "Allow passthrough mode to compose multi-step chains "
                "(plan-and-execute pipelines)."
            ),
            section="passthrough.chain",
            tags=_PT,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="passthrough_chain_max_steps",
            kind="int",
            default=10,
            label="Chain max steps",
            description=(
                "Hard cap on chain length per turn."
            ),
            section="passthrough.chain",
            min_value=1,
            max_value=20,
            tags=_PT,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="passthrough_chain_max_parallel",
            kind="int",
            default=3,
            label="Chain parallel branches",
            description=(
                "Maximum branches running in parallel within a chain step."
            ),
            section="passthrough.chain",
            min_value=1,
            max_value=10,
            tags=_PT,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="passthrough_chain_max_flows",
            kind="int",
            default=50,
            label="Chain max flows",
            description=(
                "Cap on distinct flows allowed in a single chain plan."
            ),
            section="passthrough.chain",
            min_value=1,
            max_value=200,
            tags=_PT,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="passthrough_chain_max_retries",
            kind="int",
            default=2,
            label="Chain step retries",
            description=(
                "Per-step retry cap before the chain falls through to error "
                "handling."
            ),
            section="passthrough.chain",
            min_value=0,
            max_value=10,
            tags=_PT,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="passthrough_chain_timeout",
            kind="float",
            default=120.0,
            label="Chain step timeout (s)",
            description=(
                "Per-step wall-clock timeout. Crosses fall through to retries."
            ),
            section="passthrough.chain",
            min_value=10.0,
            max_value=600.0,
            tags=_PT,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="passthrough_chain_synthesis_timeout",
            kind="float",
            default=120.0,
            label="Chain synthesis timeout (s)",
            description=(
                "Timeout for the final synthesis step (after all branches "
                "complete)."
            ),
            section="passthrough.chain",
            min_value=10.0,
            max_value=600.0,
            tags=_PT,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="passthrough_chain_attention_anchor",
            kind="bool",
            default=True,
            label="Attention anchor",
            description=(
                "Re-anchor the chain plan in the synthesis prompt — helps "
                "models that drift from the original goal."
            ),
            section="passthrough.chain",
            tags=_PT,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="passthrough_chain_error_as_observation",
            kind="bool",
            default=True,
            label="Errors as observations",
            description=(
                "Pipe step errors back into the plan as observations the "
                "model can react to, instead of failing the chain."
            ),
            section="passthrough.chain",
            tags=_PT,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="passthrough_chain_plan_mutation",
            kind="bool",
            default=True,
            label="Plan mutation",
            description=(
                "Let the model rewrite the chain plan mid-execution based on "
                "what it learned from earlier steps."
            ),
            section="passthrough.chain",
            tags=_PT,
            advanced=True,
            trust_tier="admin_only",
        )
    )

    # ============== Agentic ==============
    r.register(
        Setting(
            key="agentic_default_autonomy",
            kind="int",
            default=2,
            label="Default autonomy",
            description=(
                "Default autonomy level for agentic mode. 1 = ask before every "
                "tool call; 4 = autonomous until task completion."
            ),
            section="agentic",
            min_value=1,
            max_value=4,
            tags=_AGENTIC,
            voice_aliases=("autonomy", "agentic autonomy"),
            trust_tier="local_significant",
        )
    )
    r.register(
        Setting(
            key="agentic_max_steps",
            kind="int",
            default=20,
            label="Agentic max steps",
            description=(
                "Hard cap on plan-execute iterations per turn in agentic mode."
            ),
            section="agentic",
            min_value=1,
            max_value=100,
            tags=_AGENTIC,
            trust_tier="local_significant",
        )
    )
    r.register(
        Setting(
            key="agentic_artifact_theme",
            kind="enum",
            default="slate",
            label="Artifact theme",
            description=(
                "Visual theme for artifacts (presentations / docs) generated "
                "in agentic mode."
            ),
            section="agentic",
            enum_values=("slate", "ivory", "midnight", "warm", "neutral"),
            max_length=32,
            tags=_AGENTIC,
            trust_tier="local_reversible",
        )
    )
    r.register(
        Setting(
            key="agentic_image_model",
            kind="str",
            default="",
            label="Agentic image model",
            description=(
                "Image model used when agentic mode generates images. Empty "
                "= default image backend."
            ),
            section="agentic",
            max_length=256,
            tags=_AGENTIC,
            trust_tier="admin_only",
        )
    )

    # ============== Analytical ==============
    r.register(
        Setting(
            key="analytical_confidence_threshold",
            kind="float",
            default=0.5,
            label="Confidence threshold",
            description=(
                "Confidence floor for analytical mode to declare a phase done."
            ),
            section="analytical",
            min_value=0.0,
            max_value=1.0,
            tags=_ANALYTICAL,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="analytical_max_backtracks",
            kind="int",
            default=3,
            label="Max backtracks",
            description=(
                "How many times an analytical phase can revise its plan "
                "before committing."
            ),
            section="analytical",
            min_value=0,
            max_value=10,
            tags=_ANALYTICAL,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="analytical_max_phase_retries",
            kind="int",
            default=2,
            label="Phase retries",
            description=(
                "Per-phase retry cap before the analytical loop terminates "
                "with the best-effort answer."
            ),
            section="analytical",
            min_value=0,
            max_value=5,
            tags=_ANALYTICAL,
            trust_tier="admin_only",
        )
    )

    # ============== Architect ==============
    r.register(
        Setting(
            key="architect_dispatch_enabled",
            kind="bool",
            default=False,
            label="Architect dispatch",
            description=(
                "Use the architect (companion-as-orchestrator) layer to "
                "infer routing defaults before falling through to mode "
                "selection."
            ),
            section="architect",
            tags=_ARCHITECT,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="architect_router_enabled",
            kind="bool",
            default=True,
            label="Architect LLM router",
            description=(
                "Use a confidence-tier LLM router to replace template-based "
                "gate selection. Architect-dispatch dependency."
            ),
            section="architect",
            tags=_ARCHITECT,
            advanced=True,
            trust_tier="admin_only",
        )
    )
    r.register(
        Setting(
            key="architect_router_timeout_ms",
            kind="int",
            default=4000,
            label="Architect router timeout (ms)",
            description=(
                "Max time the LLM router can take before falling back to the "
                "template gate."
            ),
            section="architect",
            min_value=500,
            max_value=10000,
            tags=_ARCHITECT,
            advanced=True,
            trust_tier="admin_only",
        )
    )
