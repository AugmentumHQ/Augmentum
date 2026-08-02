# Coder Mode — Module Guide

Plan/Act coding agent. Routes user messages through a plan phase
(numbered-plan / question / informational classification), an act
phase (one of three strategies), and back to the user as streaming
chat events.

## File layout

```
augmentum/modes/coder/
├── handler.py              # orchestrator, shared helpers, router
├── chat_egress.py          # single emit() / emit_relay() for all yield sites
├── phase_plan.py           # _plan_phase (plan generation + marker-gated streaming)
├── phase_act.py            # _act_canonical + _act_hybrid (production strategies)
├── _legacy.py              # _act_phase_legacy + mission/architect/react/decompose
│                           # (frozen rollback — env-gated only)
└── README.md               # this file
```

**handler.py** is the orchestrator. It owns:
- `CoderHandler` class (inherits `LegacyStrategyMixin`, `PlanPhaseMixin`,
  `ActPhaseMixin`, `ModeHandler`)
- Routing in `_handle_stream` (dispatch to plan / act / conversational
  / passthrough)
- Cross-phase helpers: `_stream_and_parse`, `_run_tool_tracked`,
  `_run_tools_parallel`, `_build_messages`, `_maybe_compact_messages`,
  `_inject_sticky_reminder`, `_build_sticky_reminder`,
  `_render_fallback_summary`, `_build_turn_summary`
- Module-level helpers: `_strip_tool_json`, `_strip_cot_tokens`,
  `_tool_to_schema`, `_preview_len`, etc. — these are late-bound
  into the mixin modules' namespaces (see below).

**chat_egress.py** is the single egress point for chat. Every
`yield InternalStreamChunk(...)` call in this package routes through
`emit()` (originated content) or `emit_relay()` (wrapping a backend
chunk). Both validate `phase` and `status` against the exhaustive
`Phase` / `Status` Literals. `AUGMENTUM_STRICT_METADATA=0` downgrades
validation from raise to warning for emergency bypass.

**phase_plan.py** runs the plan phase. Weak models often emit reasoning
prose before the required `Plan:` / `Question:` marker; this module
buffers each delta until the marker appears and drops pre-marker
content. When a stream ends with no marker, `coder_plan_missing_marker`
is logged.

**phase_act.py** runs the production strategies. `_act_native` is **the
shipped strategy and the single source of truth for loop guards.** Despite
the "minimal Claude-Code/Qwen-Code parity loop" framing, it has absorbed
the full guard set over time: Termination Quality Gate, nudge-streak cap,
silent-success-fog detector, stagnation→buddy-model escalation, plus the
goal-judge and Arbor verify gates. New guard work lands HERE.

`_act_canonical` (consensus loop) and `_act_hybrid` (consensus backbone +
four innovations + five streak-break detectors) are **FROZEN** —
comparison/rollback only, reachable via `AUGMENTUM_CODER_STRATEGY=canonical|
hybrid` or the strategy header. They were the proving ground for native's
guards but are no longer hand-synced. **Do not port new guards into them**
(the triplicate hand-sync across three loops was a real maintenance hazard
and a likely source of silent guard-drift). Expect their guards to lag
native; they're kept loadable purely for rollback + A-B comparison.

**_legacy.py** holds the pre-hybrid strategies: ReAct, mission-with-
verified-promises, architect-then-editor, decompose-and-step-through,
direct ReWOO. Reachable only via `AUGMENTUM_CODER_STRATEGY=legacy`.
**Do not extend.** Any new feature goes in `_act_hybrid` or
`_act_canonical`. The file is preserved for regression comparison
and rollback safety.

## Late-bind pattern

Each mixin module (`_legacy.py`, `phase_plan.py`, `phase_act.py`)
exposes `_bind_handler_helpers()`. Handler.py calls all three at its
module bottom, injecting handler-level helpers (`_strip_tool_json`,
`_tool_to_schema`, constants like `_MUTATING_TOOLS`) into each mixin's
`__dict__` so the mixin methods can reference them by bare name.

This sidesteps the circular import: `handler.py` imports the mixin
classes at its top, so mixin modules must not `from handler import X`
at their tops. By late-binding at the bottom of `handler.py`, both
modules are fully loaded before any mixin method runs.

Names that tests monkeypatch on `handler` (`create_coder_tools`,
`select_tier`) are registered as `_LiveProxy` objects instead of
cached references. Each invocation resolves `handler.<name>` fresh,
so a `monkeypatch.setattr("augmentum.modes.coder.handler.X", ...)`
propagates into the mixin methods.

## Project digest

`augmentum/coder/digest.py` builds an optional "masterfile" — every
source file in `/workspace` inlined into one boundary-delimited block.
When the total fits under `AUGMENTUM_CODER_DIGEST_BUDGET` (explicit
override) or the dynamic digest budget derived from the active model
context, it's prepended to the system prompt and `repo_map` is skipped
entirely. When over budget, returns `None` and the caller falls through
to `WorkspaceSnapshot` + `repo_map` + on-demand
`file_read`. **All-or-nothing** — no truncation, ever: a truncated
file with an "authoritative" preamble would give the model a
confidently-wrong mental model.

## Phase 6 — final-turn synthesis (opt-in)

`AUGMENTUM_CODER_SYNTHESIZE_HYBRID=1` makes `_act_hybrid` call
`_synthesize_response` on termination when the model did real work
(`total_writes > 0 OR tool_calls_made > 0`) but narrated little
(<80 chars). Replaces the deterministic `_render_fallback_summary`
with an LLM-generated summary. Falls back to deterministic if
synthesis errors or emits nothing.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `AUGMENTUM_CODER_STRATEGY` | `native` | `native` / `hybrid` / `canonical` / `legacy` |
| `AUGMENTUM_CODER_MAX_ITERS` | `150` | Hybrid iteration cap |
| `AUGMENTUM_CODER_VALIDATION_STREAK` | `5` | Malformed-tool-calls break |
| `AUGMENTUM_CODER_TEST_FAILURE_STREAK` | `8` | Test-run failure break |
| `AUGMENTUM_CODER_SAME_FILE_CAP` | `15` | Same-path edit cap |
| `AUGMENTUM_CODER_ACTION_STAGNATION` | `20` | Same-tool streak break |
| `AUGMENTUM_CODER_INSPECTION_NUDGE` | `5` | Inspection-only streak (nudge) |
| `AUGMENTUM_CODER_INSPECTION_BREAK_DELTA` | `3` | Inspection streak (break after nudge) |
| `AUGMENTUM_CODER_NO_WRITE_PROGRESS` | `10` | Attempted-writes-all-failing break |
| `AUGMENTUM_CODER_SILENT_SUCCESS_STREAK` | `3` | Silent-success shell nudge threshold |
| `AUGMENTUM_CODER_IDENTICAL_RESULT_STREAK` | `3` | Identical (tool, args, output) repeated N iters → one-shot nudge |
| `AUGMENTUM_CODER_FAILING_SHELL_STREAK` | `4` | Failing-shell-without-edit nudge threshold |
| `AUGMENTUM_CODER_READ_REPEAT_CAP` | `5` | Per-path read-repeat refusal |
| `AUGMENTUM_CODER_COMPACT_TOKENS` | dynamic | Override auto-compaction token threshold |
| `AUGMENTUM_CODER_DIGEST_BUDGET` | dynamic | Override masterfile token budget |
| `AUGMENTUM_CODER_SYNTHESIZE_HYBRID` | unset | Enable LLM final-turn synthesis |
| `AUGMENTUM_STRICT_METADATA` | `1` | Strict phase/status validation (set `0` to downgrade to warning) |

## Testing

Core tests, all required-green on main:

- `test_coder_handler.py` — CoderHandler end-to-end, strategy dispatch
- `test_coder_context_preservation.py` — fanout, dedup, compaction-preserves-content
- `test_coder_cross_turn_persistence.py` — turn_summaries injection
- `test_coder_loop_guards.py` — stagnation, inspection-streak, dedup
- `test_coder_loop_wiring.py` — nudge formatting, tasks_completed, content-loop
- `test_coder_plan_file.py` — plan.md attention anchor
- `test_coder_hybrid_synthesis.py` — Phase 6 synthesis (flag on/off, failure modes)
- `test_coder_structured_compaction.py` — opencode-style header in compacted block
- `test_coder_chat_egress.py` — Phase/Status Literal enforcement
- `test_coder_workspace_snapshot.py` — snapshot + empty-marker
- `test_coder_digest.py` — project-digest all-or-nothing contract
