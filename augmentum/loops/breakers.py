"""Soft-breaker registry — single source of truth for loop thresholds.

Phase 2 / PR-2.3 of the Integrated Coding Nervous System spec.

The coder loop today has 14+ soft circuit-breakers (validation streak,
test-failure streak, action stagnation, inspection loop, etc.) plus a
hard iteration ceiling. Each one is a small dataclass-shaped policy:
*what counter does it watch, what threshold trips it, what tier set
does it belong to?*

PR-2.3 extracts the policy table. The actual *check logic* stays in
``phase_act.py`` for now — PR-2.4 wires the LoopRunner to drive the
loop and consult the registry directly. By pulling the table out
first, the runner can read the same source of truth the existing
coder code uses, and tests can monkey-patch the registry once instead
of poking individual constants in phase_act.

Tier mapping
------------
Each :class:`Breaker` declares which :data:`BreakerSet` bucket
(`"minimal" | "standard" | "full"`) it belongs to. :data:`MINIMAL`
contains only the hard iteration ceiling. :data:`STANDARD` adds the
high-signal stop conditions (same-validation repeat, action
stagnation, failing-shell nudge, termination quality gate).
:data:`FULL` is the complete coder suite — every breaker present
today.

A :class:`BreakerRegistry` filters the table by tier so the runner
can ask "what breakers apply at Medium intensity?" and get back only
the relevant ones.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from augmentum.loops.tier import BreakerSet, Intensity


# ── Env helper (matches phase_act._env_int contract) ──────────────────


def _env_int(name: str, default: int) -> int:
    """Read a positive int from env, falling back to ``default``.

    Same semantics as :func:`augmentum.modes.coder.phase_act._env_int`
    so production deploys with the existing ``AUGMENTUM_CODER_*`` env
    overrides keep working without rename. Negative / zero / unparsable
    values fall back to the default — protects against accidental
    "AUGMENTUM_CODER_MAX_ITERS=0" disabling the ceiling entirely.
    """
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        v = int(raw)
        return v if v > 0 else default
    except ValueError:
        return default


# ── Breaker definition ────────────────────────────────────────────────


@dataclass(frozen=True)
class Breaker:
    """One soft-breaker policy.

    Attributes
    ----------
    name:
        Identifier the runner emits to telemetry and the user-facing
        "[Stopped: <name>]" copy. Matches the ``termination_reason``
        strings phase_act already emits so log dashboards keep working.
    threshold:
        How many iterations / repeats trigger the breaker. Resolved
        from env if ``env_var`` is set, otherwise the dataclass default.
    env_var:
        Optional environment-variable name for the deploy-time override.
        Defaults to ``""`` when no env override is wanted.
    kind:
        ``"break"`` stops the loop; ``"nudge"`` injects a one-shot
        reminder into the conversation and lets the loop continue. The
        spec's "Verify-gate before Stop" (PR-2.5) only applies to
        ``break`` breakers — nudges don't terminate.
    bucket:
        Which :data:`BreakerSet` this breaker belongs to. The runner
        filters by intensity tier at start of run.
    description:
        Short human-readable explanation. Surfaced in audit logs and
        the upcoming Workshop UX surface (Phase 5).
    """

    name: str
    threshold: int = 0
    env_var: str = ""
    kind: str = "break"  # "break" | "nudge"
    bucket: BreakerSet = "standard"
    description: str = ""

    @property
    def resolved_threshold(self) -> int:
        """The effective threshold respecting any env override.

        Read once at registry-construction time; later env mutations
        don't take effect until the next process restart. Mirrors the
        coder's existing pattern (constants resolved at import).
        """
        if not self.env_var:
            return self.threshold
        return _env_int(self.env_var, self.threshold)


# ── Tool-name sets (shared by breakers + LoopRunner act loop) ─────────


MUTATING_TOOL_NAMES: frozenset[str] = frozenset({
    "code_edit", "code_edit_batch", "file_write", "apply_patch",
})
"""Tools that count as a write attempt. The ``no_write_progress`` and
``same_file_edit`` breakers watch the success ratio of calls in this
set."""


INSPECTION_TOOLS: frozenset[str] = frozenset({
    "file_read", "file_list", "dir_tree",
    "code_grep", "find_files", "code_search",
    "shell_read",
    "doc_search", "doc_fetch", "pack_search",
    "env_info", "container_info", "git",
})
"""Tools whose presence-only flags an iteration as "pure inspection".
Note: ``shell_exec`` was removed 2026-04-22 — it's the legitimate
build/install/deploy surface and counting it as inspection mis-fires
the breaker on real work."""


NATIVE_SERIAL_TOOL_NAMES: frozenset[str] = MUTATING_TOOL_NAMES | frozenset({
    "shell_exec", "git", "publish_ports", "ask_user", "test_run",
    "service_start", "service_list", "service_logs", "service_stop",
    "service_probe", "profile_update",
    "browser_open", "browser_snapshot", "browser_verify",
    "browser_click", "browser_type", "browser_screenshot",
    "browser_evaluate", "browser_wait", "browser_extract",
    "browser_fill_form",
    # Sidecar-native verbs share the same persistent browser session —
    # ordering matters (a click changes what the next get sees).
    "browser_interact", "browser_navigate", "browser_get",
    "browser_console", "browser_tabs", "browser_find",
    "observe",
})
"""Tools the native loop runs strictly serially — either because they
mutate workspace state, contend on a shared resource (Chromium, GPU),
or have ordering requirements (service lifecycle)."""


PARALLEL_READ_TOOLS: frozenset[str] = frozenset({
    "file_read",
    "code_grep",
    "find_files",
    "code_search",
    "doc_search",
    "doc_fetch",
    "pack_search",
    "env_info",
    "container_info",
    "profile_read",
    "http_request",
    "db_inspect",
})
"""Side-effect-free reads safe to fan out in parallel. Hybrid loop
batches calls from this set into one parallel wave per iteration."""


# ── The registry ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class BreakerRegistry:
    """Filtered view of breakers active at a given intensity.

    Construct via :meth:`for_intensity` rather than instantiating
    directly — that's the only path that resolves env overrides + filters
    by tier bucket in one go.
    """

    intensity: Intensity
    breakers: tuple[Breaker, ...] = field(default_factory=tuple)
    max_iterations: int = 0
    """Hard ceiling. ``Intensity.max_iterations`` wins unless the
    ``coder_max_iters`` env override raises it (used by the
    ungated-safeguards path)."""

    def by_name(self, name: str) -> Breaker | None:
        for b in self.breakers:
            if b.name == name:
                return b
        return None

    def names(self) -> tuple[str, ...]:
        return tuple(b.name for b in self.breakers)

    def filter(self, kind: str) -> tuple[Breaker, ...]:
        """Return breakers whose ``kind`` matches (``"break"`` or
        ``"nudge"``)."""
        return tuple(b for b in self.breakers if b.kind == kind)

    @classmethod
    def for_intensity(
        cls,
        intensity: Intensity,
        *,
        max_iterations_override: int | None = None,
    ) -> "BreakerRegistry":
        """Build a registry for one intensity tier.

        Resolves env overrides on each breaker, filters by the tier's
        ``breakers`` bucket, and picks the right hard ceiling. The
        ``max_iterations_override`` knob lets the coder's
        ``safeguards_enabled=False`` path raise the ceiling without
        flipping intensity.
        """
        buckets_active = _ACTIVE_BUCKETS_BY_NAME.get(intensity.breakers, ())
        breakers = tuple(
            b for b in ALL_BREAKERS if b.bucket in buckets_active
        )
        max_iters = (
            max_iterations_override
            if max_iterations_override is not None
            else intensity.max_iterations
        )
        return cls(
            intensity=intensity,
            breakers=breakers,
            max_iterations=max_iters,
        )


# ── Bucket inclusion table ────────────────────────────────────────────


_ACTIVE_BUCKETS_BY_NAME: dict[BreakerSet, tuple[BreakerSet, ...]] = {
    # "minimal" only loads the iteration-ceiling-level safety. Used by
    # LIGHT tier (just-act).
    "minimal": ("minimal",),
    # "standard" adds the high-signal stop conditions for the App
    # Builder Medium tier — same-validation repeat, action stagnation,
    # failing-shell nudge, termination quality gate.
    "standard": ("minimal", "standard"),
    # "full" loads the complete coder suite — everything in the table.
    "full": ("minimal", "standard", "full"),
}


# ── The breaker table ────────────────────────────────────────────────


ALL_BREAKERS: tuple[Breaker, ...] = (
    # ── minimal: hard runaway protection only ──────────────────────────
    Breaker(
        name="termination_quality_gate",
        threshold=0,  # gate, not counter
        env_var="",
        kind="nudge",
        bucket="minimal",
        description=(
            "Inspect the model's stop signal and nudge if it looks "
            "premature (single-sentence excuse, near-empty prose, or "
            "explicit user-insistence on completion)."
        ),
    ),
    # ── standard: high-signal stop conditions ──────────────────────────
    Breaker(
        name="validation_error_streak",
        threshold=5,
        env_var="AUGMENTUM_CODER_VALIDATION_STREAK",
        kind="break",
        bucket="standard",
        description=(
            "N consecutive iterations where every tool call fails "
            "schema validation. Default 5."
        ),
    ),
    Breaker(
        name="same_validation_error_repeat",
        threshold=2,
        env_var="AUGMENTUM_CODER_SAME_VALIDATION_BREAK",
        kind="break",
        bucket="standard",
        description=(
            "Tighter sibling of validation_error_streak — same tool, "
            "same error signature, N attempts in a row. Default 2. "
            "Runtime-tunable via ``coder_breaker_same_validation_error_repeat``."
        ),
    ),
    Breaker(
        name="action_stagnation_break",
        threshold=20,
        env_var="AUGMENTUM_CODER_ACTION_STAGNATION",
        kind="break",
        bucket="standard",
        description=(
            "Same tool name every iteration for N iters — model is "
            "spinning. Default 20."
        ),
    ),
    Breaker(
        name="failing_shell_nudge",
        threshold=4,
        env_var="AUGMENTUM_CODER_FAILING_SHELL_STREAK",
        kind="nudge",
        bucket="standard",
        description=(
            "Failing shell_exec N times without an edit between "
            "attempts — classic 'retry expecting different result' "
            "trap. Default 4."
        ),
    ),
    # ── full: complete coder suite ─────────────────────────────────────
    Breaker(
        name="test_failure_streak",
        threshold=8,
        env_var="AUGMENTUM_CODER_TEST_FAILURE_STREAK",
        kind="break",
        bucket="full",
        description=(
            "N consecutive test_run failures with no successes in "
            "between. Default 8."
        ),
    ),
    Breaker(
        name="same_file_edit_break",
        threshold=15,
        env_var="AUGMENTUM_CODER_SAME_FILE_CAP",
        kind="break",
        bucket="full",
        description=(
            "N consecutive edits to the same single file — usually "
            "indicates a stuck refactor. Default 15."
        ),
    ),
    Breaker(
        name="same_file_edit_nudge",
        threshold=5,
        env_var="AUGMENTUM_CODER_SAME_FILE_NUDGE",
        kind="nudge",
        bucket="full",
        description=(
            "N successful mutations of the same single path this turn "
            "— early rung of the same_file_edit_break ladder. Fires a "
            "one-shot prescriptive nudge (re-read, hypothesis, "
            "surgical edit, failing check) well before the hard break "
            "so the model gets a chance to change approach. Motivated "
            "by a 2026-07-06 9B native run that rewrote one file 20+ "
            "times: every write 'succeeded', so no other breaker "
            "fired. Default 5."
        ),
    ),
    Breaker(
        name="duplicate_call_nudge",
        threshold=4,
        env_var="AUGMENTUM_CODER_DUPLICATE_CALL_NUDGE",
        kind="nudge",
        bucket="full",
        description=(
            "N successful runs of the SAME read-shaped tool call "
            "(identical tool + input) within one turn — windowed, so "
            "rotating cycles of 3-4 distinct calls are caught where "
            "the consecutive-identical detector is blind. Motivated "
            "by a 2026-07-04 deepseek native run cancelled at 146 "
            "calls, 101 of them code_grep cycling the same three "
            "(pattern, path) pairs; interleaved edits kept resetting "
            "inspection_loop_nudge. Default 4."
        ),
    ),
    Breaker(
        name="duplicate_call_reorient",
        threshold=3,
        env_var="AUGMENTUM_CODER_DUPLICATE_CALL_REORIENT_DELTA",
        kind="nudge",
        bucket="full",
        description=(
            "Additional identical runs after duplicate_call_nudge "
            "before the loop REPAIRS the context in place: duplicate "
            "tool results are stubbed (first kept as ground truth) and "
            "a reorientation note preserves the lesson — recovery, not "
            "a break. Same key advancing further, or a second key "
            "reaching this rung after one repair, escalates to the "
            "buddy model instead. Default 3 (reorient at nudge+3)."
        ),
    ),
    Breaker(
        name="inspection_loop_nudge",
        threshold=5,
        env_var="AUGMENTUM_CODER_INSPECTION_NUDGE",
        kind="nudge",
        bucket="full",
        description=(
            "N read-only iterations with no mutations attempted. "
            "Default 5 — break N+3 iters after if ignored."
        ),
    ),
    Breaker(
        name="inspection_loop_break",
        threshold=3,
        env_var="AUGMENTUM_CODER_INSPECTION_BREAK_DELTA",
        kind="break",
        bucket="full",
        description=(
            "Additional iterations after inspection_loop_nudge "
            "before breaking. Default 3 — i.e. break at nudge+3."
        ),
    ),
    Breaker(
        name="no_write_progress_break",
        threshold=10,
        env_var="AUGMENTUM_CODER_NO_WRITE_PROGRESS",
        kind="break",
        bucket="full",
        description=(
            "N iterations attempting mutating tool calls (code_edit / "
            "file_write / apply_patch) with NONE succeeding. Default 10. "
            "Runtime-tunable via ``coder_breaker_no_write_progress_break``."
        ),
    ),
    Breaker(
        name="silent_success_nudge",
        threshold=3,
        env_var="AUGMENTUM_CODER_SILENT_SUCCESS_STREAK",
        kind="nudge",
        bucket="full",
        description=(
            "N consecutive shell_exec '(exit 0, no stdout)' results — "
            "no diagnostic signal. Default 3."
        ),
    ),
    Breaker(
        name="identical_tool_result_nudge",
        threshold=3,
        env_var="AUGMENTUM_CODER_IDENTICAL_RESULT_STREAK",
        kind="nudge",
        bucket="full",
        description=(
            "Same (tool, arguments, output) byte-for-byte across N "
            "consecutive iterations — the model is re-issuing a call "
            "whose result won't change. Catches the 'successful' loop "
            "the no_progress / silent_success / validation breakers miss "
            "(a repeated read or shell that succeeds every time). "
            "Default 3."
        ),
    ),
    Breaker(
        name="probe_no_signal_nudge",
        threshold=3,
        env_var="AUGMENTUM_CODER_PROBE_NO_SIGNAL",
        kind="nudge",
        bucket="full",
        description=(
            "Same shell probe re-run N times with byte-identical output "
            "despite file edits landing in between — the model's "
            "verification signal cannot detect its own changes (the "
            "always-green print-script pattern). Default 3."
        ),
    ),
    Breaker(
        name="command_carousel_nudge",
        threshold=4,
        env_var="AUGMENTUM_CODER_COMMAND_CAROUSEL_NUDGE",
        kind="nudge",
        bucket="full",
        description=(
            "Same NORMALIZED shell command (output-shaping pipe tail + "
            "redirections stripped) re-run N times this turn with no "
            "improvement in its meaningful signal (pytest pass/fail "
            "counts, error signature). Catches the test/probe re-run "
            "carousel that dodges duplicate_calls (excludes shell), "
            "probe_no_signal (needs byte-identical output), and "
            "action_stagnation (needs same tool name). Motivated by "
            "three 2026-07-07 Qwen3.6-35B runs of 147-150 iterations "
            "re-running one pytest command dozens of ways. Default 4."
        ),
    ),
    Breaker(
        name="command_carousel_reorient",
        threshold=3,
        env_var="AUGMENTUM_CODER_COMMAND_CAROUSEL_REORIENT_DELTA",
        kind="nudge",
        bucket="full",
        description=(
            "Additional no-signal re-runs after command_carousel_nudge "
            "before the loop REPAIRS context in place: duplicate shell "
            "results stubbed (first kept), reorientation note appended. "
            "Same command advancing further, or a second command "
            "reaching this rung after one repair, escalates to the "
            "buddy. Default 3 (reorient at nudge+3)."
        ),
    ),
    Breaker(
        name="progress_stall_nudge",
        threshold=25,
        env_var="AUGMENTUM_CODER_PROGRESS_STALL_NUDGE",
        kind="nudge",
        bucket="full",
        description=(
            "Coarse superset backstop: N iterations with NO measurable "
            "progress — the changed-file set didn't grow AND no "
            "additional test started passing. Catches carousels that "
            "vary enough to dodge every narrow breaker yet accomplish "
            "nothing globally. Resets on any genuine step so long "
            "legitimate builds don't trip it. Default 25."
        ),
    ),
    Breaker(
        name="progress_stall_break",
        threshold=35,
        env_var="AUGMENTUM_CODER_PROGRESS_STALL_BREAK",
        kind="break",
        bucket="full",
        description=(
            "Hard stop when the coarse progress ledger has seen no "
            "measurable forward step for N iterations — the floor that "
            "makes runaway turn length structurally impossible. Default "
            "35 (10 past the nudge)."
        ),
    ),
    Breaker(
        name="task_stale_nudge",
        threshold=8,
        env_var="AUGMENTUM_CODER_TASK_STALE_STREAK",
        kind="nudge",
        bucket="full",
        description=(
            "Task list set but not updated for N iters — model lost "
            "track of its own plan. Default 8."
        ),
    ),
    Breaker(
        name="coordination_only_nudge",
        threshold=3,
        env_var="AUGMENTUM_CODER_COORDINATION_ONLY_STREAK",
        kind="nudge",
        bucket="full",
        description=(
            "N consecutive iterations using only meta-coordination "
            "tools (task_list / ask_user) without concrete progress. "
            "Default 3."
        ),
    ),
)


# ── Convenience: legacy threshold aliases ─────────────────────────────


# phase_act.py imports these names directly today. Re-exporting the
# resolved threshold integers from the registry keeps the existing
# code working with zero touch — when PR-2.4 wires LoopRunner.run(),
# the act-loop check logic moves and these aliases get retired.


def _threshold(name: str) -> int:
    for b in ALL_BREAKERS:
        if b.name == name:
            return b.resolved_threshold
    raise KeyError(f"unknown breaker: {name}")


def live_threshold(name: str) -> int:
    """Runtime-effective threshold for breaker ``name``.

    Reads ``config.settings.coder_breaker_<name>`` on every call so a
    settings_store update (POST /api/config/setting) takes effect on
    the very next breaker check without a server restart. Treats 0 /
    unset as "use the registered default" — the env-var path and
    ``_threshold(name)`` still work for deploy-time tuning, this just
    layers a runtime override on top.

    Call this instead of the module-level constants
    (VALIDATION_ERROR_STREAK_BREAK, etc.) at any check site you want
    the user to be able to tune live.
    """
    try:
        from augmentum.config import settings as _settings
        override = getattr(_settings, f"coder_breaker_{name}", 0) or 0
        if int(override) > 0:
            return int(override)
    except Exception:
        # Defensive: a malformed setting must never crash a turn.
        pass
    return _threshold(name)


def live_max_iters(*, ungated: bool = False) -> int:
    """Runtime-effective hybrid iteration cap.

    Same pattern as ``live_threshold``: reads
    ``settings.coder_hybrid_max_iters`` (or
    ``coder_hybrid_max_iters_ungated``) and falls back to the env-var
    default when zero/unset.
    """
    key = "coder_hybrid_max_iters_ungated" if ungated else "coder_hybrid_max_iters"
    try:
        from augmentum.config import settings as _settings
        override = getattr(_settings, key, 0) or 0
        if int(override) > 0:
            return int(override)
    except Exception:
        pass
    return HYBRID_MAX_ITERS_UNGATED if ungated else HYBRID_MAX_ITERS


VALIDATION_ERROR_STREAK_BREAK: int = _threshold("validation_error_streak")
SAME_VALIDATION_REPEAT_BREAK: int = _threshold("same_validation_error_repeat")
ACTION_STAGNATION_BREAK: int = _threshold("action_stagnation_break")
TEST_FAILURE_STREAK_BREAK: int = _threshold("test_failure_streak")
SAME_FILE_EDIT_BREAK: int = _threshold("same_file_edit_break")
SAME_FILE_EDIT_NUDGE_AT: int = _threshold("same_file_edit_nudge")
NO_WRITE_PROGRESS_BREAK: int = _threshold("no_write_progress_break")
SILENT_SUCCESS_NUDGE_AT: int = _threshold("silent_success_nudge")
IDENTICAL_TOOL_RESULT_NUDGE_AT: int = _threshold("identical_tool_result_nudge")
FAILING_SHELL_NUDGE_AT: int = _threshold("failing_shell_nudge")
PROBE_NO_SIGNAL_NUDGE_AT: int = _threshold("probe_no_signal_nudge")
COMMAND_CAROUSEL_NUDGE_AT: int = _threshold("command_carousel_nudge")
COMMAND_CAROUSEL_REORIENT_DELTA: int = _threshold("command_carousel_reorient")
PROGRESS_STALL_NUDGE_AT: int = _threshold("progress_stall_nudge")
PROGRESS_STALL_BREAK_AT: int = _threshold("progress_stall_break")
TASK_STALE_NUDGE_AT: int = _threshold("task_stale_nudge")
COORDINATION_ONLY_NUDGE_AT: int = _threshold("coordination_only_nudge")
INSPECTION_STREAK_NUDGE: int = _threshold("inspection_loop_nudge")
INSPECTION_STREAK_BREAK_AFTER_NUDGE: int = _threshold("inspection_loop_break")

# Constants the runner needs that aren't soft-breakers themselves but
# share the threshold-policy shape. Kept here so deploy-time tuning
# happens in one place.
HYBRID_MAX_ITERS: int = _env_int("AUGMENTUM_CODER_MAX_ITERS", 150)
HYBRID_MAX_ITERS_UNGATED: int = _env_int(
    "AUGMENTUM_CODER_MAX_ITERS_UNGATED", 500,
)
HYBRID_STAGNATION_REPEATS: int = 2
HYBRID_CONTINUATION_LOOKBACK: int = 3
HYBRID_MIN_TURN_PROSE_CHARS: int = 80
INSPECTION_COLD_START_GRACE: int = 2
# Native-strategy nudge cap. Pre-2026-05-31 the TQG accepted the stop
# on the second prose-no-tools response (one nudge then accept).
# Chatty local models (Qwen-3.6) regularly emit two short preambles
# before reaching a tool call, so the cap moves to 2 by default.
# Tunable via ``coder_native_nudge_max`` (0 = use this default).
NATIVE_NUDGE_MAX_DEFAULT: int = _env_int("AUGMENTUM_CODER_NATIVE_NUDGE_MAX", 2)


def live_native_nudge_max() -> int:
    """Runtime-effective native-strategy nudge cap.

    Reads ``settings.coder_native_nudge_max``; falls back to
    :data:`NATIVE_NUDGE_MAX_DEFAULT` when zero/unset.
    """
    try:
        from augmentum.config import settings as _settings
        override = int(getattr(_settings, "coder_native_nudge_max", 0) or 0)
        if override > 0:
            return override
    except Exception:
        pass
    return NATIVE_NUDGE_MAX_DEFAULT


__all__ = [
    "ACTION_STAGNATION_BREAK",
    "ALL_BREAKERS",
    "Breaker",
    "BreakerRegistry",
    "COMMAND_CAROUSEL_NUDGE_AT",
    "COMMAND_CAROUSEL_REORIENT_DELTA",
    "COORDINATION_ONLY_NUDGE_AT",
    "FAILING_SHELL_NUDGE_AT",
    "PROGRESS_STALL_BREAK_AT",
    "PROGRESS_STALL_NUDGE_AT",
    "HYBRID_CONTINUATION_LOOKBACK",
    "HYBRID_MAX_ITERS",
    "HYBRID_MAX_ITERS_UNGATED",
    "HYBRID_MIN_TURN_PROSE_CHARS",
    "HYBRID_STAGNATION_REPEATS",
    "IDENTICAL_TOOL_RESULT_NUDGE_AT",
    "INSPECTION_COLD_START_GRACE",
    "INSPECTION_STREAK_BREAK_AFTER_NUDGE",
    "INSPECTION_STREAK_NUDGE",
    "INSPECTION_TOOLS",
    "MUTATING_TOOL_NAMES",
    "NATIVE_NUDGE_MAX_DEFAULT",
    "NATIVE_SERIAL_TOOL_NAMES",
    "NO_WRITE_PROGRESS_BREAK",
    "PARALLEL_READ_TOOLS",
    "PROBE_NO_SIGNAL_NUDGE_AT",
    "SAME_FILE_EDIT_BREAK",
    "SAME_FILE_EDIT_NUDGE_AT",
    "SAME_VALIDATION_REPEAT_BREAK",
    "SILENT_SUCCESS_NUDGE_AT",
    "TASK_STALE_NUDGE_AT",
    "TEST_FAILURE_STREAK_BREAK",
    "VALIDATION_ERROR_STREAK_BREAK",
    "live_max_iters",
    "live_native_nudge_max",
    "live_threshold",
]
