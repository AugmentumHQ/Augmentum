"""Coder mode handler — Plan/Act coding agent loop.

Phase 2: full agent loop with plan generation, native tool calling,
streaming metadata, and phase transitions.

Phase 1 passthrough is preserved when no container_manager is provided.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import uuid
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

import structlog

from augmentum.coder.context_tokens import (
    coder_context_token_limit as _coder_context_token_limit,
)
from augmentum.coder.context_tokens import (
    coder_digest_token_budget as _coder_digest_token_budget,
)
from augmentum.coder.prompts import (
    ACT_SYSTEM,
    ACT_SYSTEM_WITH_TOOLS,
    EDIT_FORMAT_INSTRUCTIONS,  # noqa: F401 — re-exported for callers
    MISSION_ACT_SYSTEM,
    MISSION_ACT_SYSTEM_WITH_TOOLS,
    NATIVE_SYSTEM,
)
from augmentum.coder.state import CoderPhase, CoderState
from augmentum.coder.tools import (
    READ_ONLY_TOOLS,  # noqa: F401
    # Re-exported so tests monkeypatching ``handler.create_coder_tools``
    # work; phase_act / _legacy / phase_plan resolve it via _LiveProxy
    # at call time.
    create_coder_tools,  # noqa: F401
)
from augmentum.models.base import (
    InternalChatRequest,
    InternalChatResponse,
    InternalStreamChunk,
    Message,
)
from augmentum.modes.analytical.tool_calling import (
    ToolCallingTier,
    # Re-exported so tests that monkeypatch ``handler.select_tier`` work.
    # phase_act looks it up via _LiveProxy at call time.
    select_tier,  # noqa: F401
)
from augmentum.modes.base import ModeHandler

# Gated import: ``_legacy`` pulls ``augmentum.promises`` (MissionRunner,
# Promise, Verification, parse_mission_json, etc.) at module-load time.
# The legacy strategy is only reachable via ``AUGMENTUM_CODER_STRATEGY=
# legacy`` (see dispatcher below). When the user hasn't opted in, install
# a stub mixin so those heavy imports never happen. Re-enable by setting
# the env var and restarting.
_LEGACY_STRATEGY_ENABLED = (
    os.environ.get("AUGMENTUM_CODER_STRATEGY", "").strip().lower() == "legacy"
)
if _LEGACY_STRATEGY_ENABLED:
    from augmentum.modes.coder._legacy import LegacyStrategyMixin
else:
    class LegacyStrategyMixin:
        """Stub. The real ``LegacyStrategyMixin`` in
        ``augmentum.modes.coder._legacy`` only loads when
        ``AUGMENTUM_CODER_STRATEGY=legacy``. The strategy dispatcher
        guards against reaching the methods below; the stub raises a
        clear error if anything ever calls into it.
        """

        async def _act_phase_legacy(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise RuntimeError(
                "Legacy coder strategy not loaded. Set "
                "AUGMENTUM_CODER_STRATEGY=legacy and restart to enable."
            )

from augmentum.modes.coder.chat_egress import (
    RUNTIME_CARRIER_HEADER,
    emit,
    emit_relay,
)
from augmentum.modes.coder.intent import (
    TIER_LIMITS,  # noqa: F401 — re-exported for phase_act
    Tier,
    TierClassification,
    TurnIntent,
    TurnIntentKind,
    classify_tier,
    classify_turn_intent,
    explicitly_requests_execution,
)
from augmentum.modes.coder.phase_act import (
    # Re-export for tests that import constants from handler. Each is
    # kept live so a ``monkeypatch.setattr("handler._X", ...)`` propagates
    # through ``_cfg()`` in phase_act. noqa F401 = ruff would strip.
    _ACTION_STAGNATION_BREAK,  # noqa: F401
    _HYBRID_MAX_ITERS,  # noqa: F401
    _HYBRID_MIN_TURN_PROSE_CHARS,  # noqa: F401
    _INSPECTION_COLD_START_GRACE,  # noqa: F401
    _SAME_FILE_EDIT_BREAK,  # noqa: F401
    _TEST_FAILURE_STREAK_BREAK,  # noqa: F401
    _VALIDATION_ERROR_STREAK_BREAK,  # noqa: F401
    ActPhaseMixin,
)
from augmentum.modes.coder.phase_plan import (
    PlanPhaseMixin,
    _parse_plan_steps,  # noqa: F401  — re-exported for tests importing from handler
    _plan_is_question,
)
from augmentum.modes.coder.runtime_truth import RuntimeTruth
from augmentum.modes.coder.turn_context import TurnContext, build_turn_context
from augmentum.promises import (
    Promise,
    PromiseStatus,
)
from augmentum.utils.datetime_context import get_datetime_context

if TYPE_CHECKING:
    from augmentum.coder.dispatch import CoderDispatch
    from augmentum.powers.registry import PowerRegistry
    from augmentum.state.manager import StateManager

log = structlog.get_logger(__name__)

# --- Canonical + Hybrid loop constants ------------------------------------
# `_act_canonical` is the consensus pattern (Codex / Claude Code / opencode
# / qwen-code / cline all converge on the same skeleton): one loop, model
# decides when to stop, iteration cap is a pure fail-safe. No budget
# machinery, no repeat detection, no planner. Intended for strong models
# where trust-the-model is the right default.

# Test-monkeypatchable act constants. The strategies live in phase_act.py
# but tests monkeypatch via the handler module, so we keep the source
# of truth here and have phase_act look these up fresh at call time.
_CANONICAL_MAX_ITERS = 50
_HYBRID_OBSERVATION_EVERY = 8
_HYBRID_READ_FANOUT = 5

# `_act_hybrid` keeps the consensus backbone but adds four weaker-model
# resilience mechanisms: observation refresh, stagnation nudge, parallel
# read fan-out, and a heuristic continuation judge. Rebuttal to the pure
# trust-the-model stance — still model-driven, but with ground truth and
# soft error recovery.
# All circuit-breaker thresholds live behind AUGMENTUM_CODER_* env vars
# so deployments can dial them per-model-tier without code edits. The
# defaults were bumped 2026-04-20 (second-pass tuning) after feedback
# that original values were firing on coherent-but-ongoing work. If you
# see a breaker fire incorrectly, raise the corresponding env var; if
# you see pathological loops running too long, lower it.
def _env_int(name: str, default: int) -> int:
    """Read an int env var with sane fallback; guards against bad input."""
    import os as _os
    raw = _os.environ.get(name, "")
    if not raw:
        return default
    try:
        v = int(raw)
        return v if v > 0 else default
    except ValueError:
        return default


# Backoff schedule between transient-failure retries — total budget ~17s
# across 3 attempts (initial + 2 retries). Long enough that a Cerebras
# queue surge or Anthropic 503 wave usually clears inside the window;
# short enough that the user isn't watching a dead screen for a minute.
_RETRY_BACKOFF_S: tuple[float, ...] = (2.0, 5.0, 10.0)


# Sentinel for the streaming-truncation failure mode: the model emitted
# a tool-call header but the arguments JSON never arrived because the
# response was cut off by max_tokens (provider returns
# finish_reason="length"). Distinguished from ``_parse_error_raw``
# (which carries an unparseable raw string) so the dispatcher can
# emit a different error message — "your output was truncated" is the
# actionable diagnosis; "your JSON was malformed" is not, because no
# JSON arrived at all.
_TRUNCATION_REASON_EMPTY_ARGS = "args_missing"


def _build_truncation_error(
    *, tool_name: str, finish_reason: str,
) -> str:
    """Compose the model-facing error for a truncated tool call.

    The model has just emitted a ``file_write`` / ``code_edit`` /
    ``apply_patch`` header that ran out of output budget before the
    arguments JSON could arrive. The error has to be actionable —
    telling the model "path is required" makes it retry identically.
    Instead, name the failure mode and point at the recovery path
    (smaller code_edit blocks).
    """
    reason = finish_reason or "unknown"
    return (
        f"Your previous response was truncated by the provider "
        f"(finish_reason={reason!r}) before the {tool_name!r} tool's "
        "arguments JSON arrived on the wire. The output budget can't "
        "hold a tool call this large. Do NOT retry with the same plan "
        "— the truncation will recur. Instead: use code_edit with "
        "small targeted SEARCH/REPLACE blocks (under ~200 lines per "
        "block), or split a full-file rewrite into multiple file_write "
        "calls with shorter content each."
    )


# Synthetic stand-in for a tool result that no longer exists in history —
# inserted by ``_reconcile_tool_pairing`` so an assistant ``tool_calls``
# message is never sent to the backend with an unanswered ``tool_call_id``.
# The original output was already gone (compaction dropped it, or the batch
# was interrupted), so this loses nothing new; it just makes the absence
# explicit instead of letting a strict provider (DeepSeek) 400 the whole
# turn. Worded so the model can't misread it as a successful result.
_PAIRING_STUB = (
    "[tool result unavailable — the original output was dropped during "
    "context compaction. Continue without it; re-run the tool if you "
    "still need this information.]"
)


def _reconcile_tool_pairing(messages: list) -> list:
    """Return a copy of ``messages`` with tool-call/result pairing repaired.

    OpenAI-compatible backends require that every assistant message bearing
    ``tool_calls`` is immediately followed by one ``role="tool"`` message
    per ``tool_call_id`` — and that no ``tool`` message appears without a
    preceding assistant call declaring its id. Most providers tolerate
    violations silently; DeepSeek's direct API enforces it strictly and
    returns ``400 "An assistant message with 'tool_calls' must be followed
    by tool messages…"``, killing the turn.

    Orphans arise from non-pairing-aware context compaction
    (``_maybe_compact_messages`` slices at a fixed keep-recent boundary) or
    a turn that persisted a trailing unanswered assistant call. This walks
    the list once and guarantees the pairing invariant:

      * For each assistant ``tool_calls`` group, emit the assistant message
        then exactly one tool message per declared id, in declared order —
        reusing the real result where present, synthesizing ``_PAIRING_STUB``
        where missing.
      * Drop ``tool`` messages whose id isn't declared by the assistant call
        they follow (leading/dangling orphans).

    Pure: never mutates the input list or its ``Message`` objects. A
    well-formed history round-trips to an equivalent list (no-op in the
    happy path). Logs a single ``warning`` when it actually changes
    something, so orphaning frequency is observable in the logs.
    """
    out: list = []
    n = len(messages)
    i = 0
    synthesized = 0
    dropped = 0
    while i < n:
        m = messages[i]
        role = getattr(m, "role", "")
        tcs = (getattr(m, "tool_calls", None) or []) if role == "assistant" else []
        if role == "assistant" and tcs:
            out.append(m)
            declared: list[str] = []
            for tc in tcs:
                tcid = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
                if tcid:
                    declared.append(tcid)
            declared_set = set(declared)
            # Consume the run of tool messages that immediately follows.
            seen: dict[str, object] = {}
            j = i + 1
            while j < n and getattr(messages[j], "role", "") == "tool":
                tmid = getattr(messages[j], "tool_call_id", None)
                if tmid in declared_set and tmid not in seen:
                    seen[tmid] = messages[j]
                else:
                    dropped += 1  # unknown id or duplicate within the group
                j += 1
            for tcid in declared:
                if tcid in seen:
                    out.append(seen[tcid])
                else:
                    out.append(Message(
                        role="tool", content=_PAIRING_STUB, tool_call_id=tcid,
                    ))
                    synthesized += 1
            i = j
            continue
        if role == "tool":
            # A tool message not consumed by a preceding assistant group is
            # a dangling orphan (its declaring call was compacted away).
            dropped += 1
            i += 1
            continue
        out.append(m)
        i += 1

    if synthesized or dropped:
        log.warning(
            "coder.tool_pairing_reconciled",
            synthesized=synthesized, dropped=dropped,
            messages_before=n, messages_after=len(out),
        )
    return out


# Status codes that map to "the same request will succeed if we wait".
# Includes 408 Request Timeout, 425 Too Early, 429 Too Many Requests,
# and the 5xx server-side errors. Connection-level failures (httpx
# ReadError, ConnectError, ReadTimeout) classify as transient too —
# see _classify_backend_error below.
_TRANSIENT_HTTP_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

# Status codes that map to "this request will keep failing — bail now".
# 4xx auth / not-found / validation errors — retrying just wastes time
# and burns the user's quota / rate-limit budget.
_PERMANENT_HTTP_STATUS = frozenset({400, 401, 403, 404, 405, 410, 422})

# Regex to pull the HTTP status from openai_compat's RuntimeError shape:
# ``RuntimeError("Backend returned 429: ...")``. The raw string is the
# only signal we get today; openai_compat raises plain RuntimeError
# rather than a typed exception. Captured group is the integer status.
_BACKEND_STATUS_RE = re.compile(r"Backend returned (\d{3})")

# Substrings that mark a 429 as a per-minute token quota rather than a
# transient queue/QPS surge. Quota windows reset on the minute boundary,
# so the in-loop retry budget (~17s across 3 attempts) is almost never
# enough to ride them out — the user just watches a frozen screen for
# 17s before seeing the same error. We bail after the first attempt and
# let the user decide whether to wait or switch model. Case-insensitive
# substring match — providers phrase these differently:
#   ChatGPT-bridge / OpenAI: "Tokens per minute limit exceeded"
#                            "token_quota_exceeded"
#   Anthropic:               "tokens_per_minute" (in rate-limit type)
#   Groq / Mistral:          "rate_limit_exceeded" + "tokens"
_QUOTA_MARKERS = (
    "token_quota_exceeded",
    "tokens per minute",
    "tokens_per_minute",
    "tpm",
    "tokens.*minute",  # informational only — substring search below
)


def _classify_backend_error(exc: Exception) -> tuple[str, int | None]:
    """Classify a backend exception as ``transient``, ``permanent``, or ``quota``.

    Returns ``(kind, status)``. ``kind`` is:

      * ``"transient"`` — retrying has a reasonable chance of succeeding
        (429 QPS surge, 502/503/504 upstream, network blip).
      * ``"permanent"`` — same request will fail identically
        (4xx validation / auth / not-found). Bail immediately.
      * ``"quota"`` — a 429 that's specifically a per-minute token
        quota. Bails after the first attempt without burning the
        full 17s budget, since the window won't reset in time. Surfaces
        as a permanent error to the user with a quota-specific hint.
      * Default ``"transient"`` for unknown errors — better to spend
        ~17s retrying an unknown class once than to declare it
        permanent and leave the user with a hard-stop.
    """
    # httpx connection-layer failures don't carry an HTTP status; they're
    # always transient by definition (the request never reached a server
    # that would have given a status code).
    exc_module = type(exc).__module__ or ""
    exc_name = type(exc).__name__
    if exc_module.startswith("httpx") and exc_name in (
        "ReadTimeout", "ConnectTimeout", "ReadError", "ConnectError",
        "RemoteProtocolError", "PoolTimeout",
    ):
        return "transient", None

    raw = str(exc) or ""

    # RuntimeError("Backend returned NNN: ...") — pull the status code.
    m = _BACKEND_STATUS_RE.search(raw)
    if m:
        status = int(m.group(1))
        if status == 429:
            # Distinguish per-minute token quota from QPS surge. Quota
            # markers in the body → bail after one attempt instead of
            # burning 17s. Match is case-insensitive substring; the
            # entries in _QUOTA_MARKERS are normalised to lowercase
            # already.
            body_lower = raw.lower()
            if any(marker in body_lower for marker in (
                "token_quota_exceeded", "tokens per minute",
                "tokens_per_minute", "tpm_limit", "tpm exceeded",
            )):
                return "quota", status
            return "transient", status
        if status in _TRANSIENT_HTTP_STATUS:
            return "transient", status
        if status in _PERMANENT_HTTP_STATUS:
            return "permanent", status
        # Unrecognised 4xx → permanent (retrying validation errors is
        # always wrong). Unrecognised 5xx → transient (server fault).
        if 400 <= status < 500:
            return "permanent", status
        return "transient", status

    # Unknown exception shape — default to transient so we retry once
    # rather than killing the turn on first failure.
    return "transient", None


# Circuit breaker for malformed tool calls. After this many consecutive
# iterations where every tool call in the batch failed input validation
# (missing required args, malformed JSON), break the loop with a
# user-visible explanation. Observed 2026-04-20 on weaker models:
# without this, the model rotates across tools emitting empty params
# (dir_tree, shell_exec, file_read) — each batch has a different
# signature so the stagnation nudge never fires. Bumped from 3 → 5
# 2026-04-20 after "firing too early on coherent work" feedback —
# legitimate schema recovery can take 3-4 retries on weaker models.

# Break when `test_run` has failed this many iterations in a row with
# NO successes in between. Observed 2026-04-20 on a session where the
# agent ran ~40 failed test_run calls back-to-back. A single passing
# test_run resets the streak. Bumped from 5 → 8 2026-04-20 — real
# debug cycles legitimately run 5-6 test iterations before going green
# (reproduce → diagnose → fix → test → fix → test → green).

# Cap on consecutive edits to the SAME file within one turn. Catches
# the "agent thrashes on one file, never makes progress" pattern.
# Bumped from 8 → 15 2026-04-20 — a real refactor of a 500-line file
# can legitimately touch it 10+ times (add import, add helper, call
# helper, fix tests, update imports, fix type error, rename…). 15
# still catches pathological thrashing; 8 was punishing real work.

# Preemptive repeat-read refusal threshold. Observed 2026-04-20 (Qwen
# 3.6 35B A3B): a vague "improve it" prompt made the model read
# snake.html four times in five iterations. The soft sticky-reminder
# signals ("Already inspected: file_read /snake.html ×3") get ignored
# by weak models. At this threshold the 3rd and later identical reads
# are REFUSED with a synthetic tool_result pointing at the earlier
# content — unambiguous, unlike the soft signal.
#
# Only applies to pure-read tools (file_read, file_list, dir_tree,
# code_grep, find_files, code_search, shell_read). Writes / shell_exec
# are allowed to repeat — they're not information-gathering, and
# forcing a refusal there would block legitimate retries.
#
# The counter is CLEARED for a path when that path is mutated (see
# CoderState.clear_tool_calls_for_path), so "edit → read to verify"
# workflows are not blocked.
#
# Bumped from 3 → 5 2026-04-20 — 3 was firing on legitimate re-reads
# of a file the model was actively analysing (different excerpts, same
# path). 5 still catches the pathological case where a model forgets
# it already has the content.
_READ_REPEAT_REFUSAL_CAP = _env_int(
    "AUGMENTUM_CODER_READ_REPEAT_CAP", 5,
)

_PREEMPTIVE_REFUSAL_TOOLS = frozenset({
    "file_read", "file_list", "dir_tree",
    "code_grep", "find_files", "code_search",
    "shell_read",
})

# Cold-start grace for the inspection-only streak detector. Ported from
# qwen-code's ``hasSeenNonReadTool`` latch (LoopDetectionService.ts).
# Observation: even on a "make me a snake game" prompt (a creation goal),
# the agent legitimately needs a few iterations of ``dir_tree`` + README
# reads before writing — that's not thrashing, that's orientation. The
# streak detector should not count these first iterations against the
# agent. Grace of 2 iterations means the streak starts ticking at
# iteration 3, after the agent has had a chance to look around.

# Same-tool-regardless-of-args streak breaker. Ported from qwen-code's
# ``checkActionStagnation`` (LoopDetectionService.ts). Our existing
# stagnation detector is IDENTICAL-batch (same tool + same args, breaks
# at 2) and the validation-error streak fires on malformed calls. Neither
# catches "model rotates through variations on the same tool":
#
#   shell_exec(cmd='make')
#   shell_exec(cmd='make 2>&1')
#   shell_exec(cmd='make clean; make')
#   shell_exec(cmd='make 2>&1 | head -50')
#   ...
#
# All different args, same tool, parameter-thrashing. Break after N
# consecutive iterations where every tool call is the same tool name.
# Bumped from 8 → 20 2026-04-20 — long debug cycles legitimately loop
# on shell_exec with different invocations; 8 cut some real sessions
# off. Kept strictly ABOVE _SAME_FILE_EDIT_BREAK so the more specific
# "thrashing on /path" message wins on overlapping patterns. Read-only
# exploration tools are EXEMPT from this detector (see the iter-update
# block in _act_hybrid).

# Minimum assistant-prose length (chars, after CoT/tool-JSON stripping) that
# counts as "the model has delivered a substantive answer" when it stops
# emitting tool calls. Pre-2026-04-20 the continuation judge only looked at
# recent writes — so an informational query like "what's in this project?"
# (agent reads → answers → stops, zero writes) would get NUDGED instead of
# terminating. That nudge put the agent back into an action-hunting loop
# with no task to action on, and it thrashed. 40 chars ≈ one full sentence;
# anything shorter is plausibly the model trailing off without a real answer
# and warrants the original nudge.

# Minimum cumulative prose (across ALL iterations) that a turn must emit
# before we treat it as self-narrating. Observed 2026-04-20: a Qwen 3.6
# 35B A3B turn ran 15 shell_exec calls trying to build nsnake, failed
# quietly, emitted zero narrative prose, and showed the user an empty
# "Done" banner. The tool output is invisible to the user (only tool
# NAMES show in the UI), so when the model doesn't narrate, the user
# has nothing. Below this threshold we synth a fallback summary at turn
# end so the user isn't left with silence. 80 chars ≈ two sentences.

# Mid-turn compaction threshold. This is the fallback used before a turn
# has probed the active backend/model. Each real turn refreshes a dynamic
# value from ``backend.get_context_length(model)`` while preserving
# ``AUGMENTUM_CODER_COMPACT_TOKENS`` as an override.
_COMPACT_AT_TOKENS = _coder_context_token_limit()
# How many trailing messages to keep uncompacted. Bumped 6 → 12 on
# 2026-05-31 alongside the auto-compaction work — gives the model
# more working context after the summary lands. Live-tunable via
# ``coder_compaction_keep_recent`` setting (0 sentinel reads this
# value so the test suite's monkeypatch path still works).
_COMPACT_KEEP_RECENT = 12

# Hydrated recency buffer — how many recent COMPLETED turns to replay in
# full in-format (assistant tool_calls + tool results + thinking) at the
# head of the next turn's history, instead of the client's collapsed
# ``getMessagesForLLM`` form (which strips tool_calls/results/thinking).
# Without this the model never sees its own recent tool-using behavior on
# a follow-up turn and stops emulating it — reasons, then answers in
# prose without calling tools (observed live 2026-07-03, 9B DeepSeek:
# turn 1 interleaved 12 tools cleanly, turn 2 answered directly).
# Older-than-window turns stay represented by the <prior_turns> summary.
# Bounded small: recency is what anchors imitation; 1-2 turns suffices,
# and the chains feed into compaction which bounds total tokens.
_RECENCY_BUFFER_TURNS = 2

# Immutable opening of the auto-compaction block. Deliberately carries
# NO counts or other mutating text: the block extends append-only
# across compaction passes (see ``_maybe_compact_messages``), and the
# llama-server prefix cache can only reuse up to the first changed
# byte — a "N earlier messages condensed" header rewrote itself every
# pass and invalidated the whole packed window (measured 2026-07-02,
# stable_pct 0.13).
_COMPACT_WRAPPER_OPEN = (
    "<compacted reason=\"context-pressure\">\n"
    "Earlier messages were condensed to save context. Segments are "
    "ordered oldest first; messages after this block are verbatim. "
    "Use this as memory of the earlier work; inspect files or rerun "
    "tools only when exact current state matters.\n\n"
)

# Per-tool permission policy. Values:
#   "auto"               (default) — every tool call proceeds without a gate
#   "confirm_mutations"  — tools in _APPROVAL_REQUIRED_TOOLS ask the user
#                          (via the handler's permission_callback) before
#                          executing. Denied calls inject a synthetic
#                          tool_result so the model can react.
_CODER_PERMISSIONS = __import__("os").environ.get(
    "AUGMENTUM_CODER_PERMISSIONS", "auto",
).strip().lower() or "auto"

# Strategy dispatch override. `native` is the production default — the
# minimal Claude-Code/Qwen-Code parity loop, which empirically performs
# best across day-to-day coding turns. `hybrid` (consensus backbone plus
# four rebuttal innovations) and `canonical` (minimalist baseline) remain
# available for comparison; `legacy` is kept behind an explicit opt-in
# for rollback safety and regression testing.
#
# Values:
#   "native"    (default) → minimal Claude-Code/Qwen-Code style loop
#                           (skips plan, TQG, nudges, sticky reminders,
#                           snapshot/digest injection — see
#                           ``phase_act._act_native`` for the contract).
#   "hybrid"              → 4-innovation rebuttal loop (_act_hybrid)
#   "canonical"           → always use _act_canonical
#   "legacy"              → use the pre-hybrid heuristic dispatcher that
#                           routes to _act_mission / _act_decompose /
#                           _act_architect / _act_direct
_CODER_STRATEGY_OVERRIDE = __import__("os").environ.get(
    "AUGMENTUM_CODER_STRATEGY", "native",
).strip().lower() or "native"

# Recognised strategy values. Anything else from the request header /
# env var falls back to hybrid (with a warning) rather than raising —
# protects users against typos in the URL/env without crashing the turn.
_CODER_KNOWN_STRATEGIES = frozenset({"hybrid", "canonical", "native", "legacy"})

# Tools that mutate workspace state. These are executed sequentially AFTER
# the batch's read-only calls complete, which (a) guarantees state.files_read
# is populated before any code_edit's read-before-edit guard checks it, and
# (b) prevents two code_edits on the same path from racing.
_MUTATING_TOOLS = frozenset({
    "code_edit", "code_edit_batch", "file_write", "apply_patch",
    "publish_ports",
})

# Total digest budget per compacted "A: called ..." line. Keeps segments
# lean while still carrying WHAT changed / WHICH command ran.
_DIGEST_CAP = 300


def _one_line(text: str, cap: int) -> str:
    """Flatten to one line and bound to ``cap`` chars."""
    flat = " ".join(str(text).split())
    return flat[:cap]


def _tool_call_digest(tool_calls) -> str:
    """Compact argument digest for write-shaped + shell tool calls.

    Used by compaction's fold rendering. Pre-2026-07-09, folding an
    assistant tool-call turn discarded its ARGUMENTS entirely — a
    ``code_edit`` or ``shell_exec`` collapsed to "A: called code_edit",
    so the model could no longer know WHAT it changed or WHICH command
    produced a given output. This carries a bounded preview of those
    arguments into the fold.

    Never-truncate boundary note: these are explicitly-marked DIGESTS of
    environment-bound tool arguments inside a fold segment — the fold
    already drops the full arguments by design; this RESTORES a bounded
    view rather than silently cutting content. The grounded truth lives
    in the workspace and can be re-read.
    """
    parts: list[str] = []
    for tc in tool_calls or []:
        if isinstance(tc, dict):
            fn = tc.get("function", {}) or {}
            name = fn.get("name", "") or tc.get("name", "") or ""
            args_str = fn.get("arguments", "")
        else:
            name = getattr(tc, "name", "") or ""
            fn_obj = getattr(tc, "function", None)
            args_str = (
                getattr(fn_obj, "arguments", "") if fn_obj is not None
                else getattr(tc, "arguments", "")
            ) or ""
        if isinstance(args_str, dict):
            args = args_str
        else:
            try:
                args = json.loads(args_str) if args_str else {}
            except (json.JSONDecodeError, TypeError):
                args = {}
        if not isinstance(args, dict):
            continue

        piece = ""
        if name == "shell_exec":
            cmd = args.get("command") or args.get("cmd") or ""
            if cmd:
                piece = f"$ {_one_line(cmd, 160)}"
        elif name == "code_edit":
            path = args.get("path") or args.get("file_path") or ""
            old = str(args.get("old_string") or "")
            new = str(args.get("new_string") or "")
            if len(old) + len(new) > 400:
                # Huge args: shape only (-old/+new line counts).
                piece = (
                    f"{path} (-{len(old.splitlines()) or 1}"
                    f"/+{len(new.splitlines()) or 1} lines)"
                )
            elif old or new:
                piece = f"{path} {_one_line(f'{old} -> {new}', 120)}"
            elif path:
                piece = str(path)
        elif name == "apply_patch":
            path = args.get("path") or args.get("file_path") or ""
            patch = str(args.get("patch") or args.get("diff") or "")
            piece = f"{path} {_one_line(patch, 120)}".strip()
        elif name == "file_write":
            path = args.get("path") or args.get("file_path") or ""
            body = str(args.get("content") or "")
            piece = f"{path} ({len(body)} chars) {_one_line(body, 100)}".strip()
        elif name == "code_edit_batch":
            edits = args.get("edits") or []
            if isinstance(edits, list) and edits:
                paths = []
                for e in edits:
                    if isinstance(e, dict):
                        p = e.get("path") or e.get("file_path") or ""
                        if p and p not in paths:
                            paths.append(str(p))
                piece = f"{len(edits)} edits: {_one_line(', '.join(paths), 140)}"
        if piece:
            parts.append(piece)

    digest = "; ".join(parts)
    if len(digest) > _DIGEST_CAP:
        digest = digest[: _DIGEST_CAP - 1] + "…"
    return digest

# Tools that only inspect the workspace without changing it. Feeds the
# "creation-task stuck in inspection" detector — if the user's goal has
# a creation verb and all we're calling are these, the model is stuck
# exploring when it should be writing. Excludes task_list + ask_user
# (which ARE progress-like actions, not surveillance).

# Creation verbs that imply the model should be WRITING files, not just
# reading. Observed 2026-04-20 on a "make me a snake game" request
# where the model spent 5 iterations running wc/tail/head on a file and
# never wrote anything. Matching is case-insensitive whole-word; false
# positives (e.g. "write me a summary") are acceptable risk in coder
# mode — any "write" in this context is going to be about code.
_CREATION_VERB_RE = re.compile(
    r"\b(make|create|build|add|implement|write|generate|scaffold|setup|"
    r"set\s+up|initialize|init|refactor|port|migrate|convert|rewrite)\b",
    re.IGNORECASE,
)

# Inspection-streak threshold for forceful nudge. Bumped from 3 → 5
# 2026-04-20 — weak models legitimately need 4-5 read iterations to
# orient before writing; 3 was firing too early. Cold-start grace
# (_INSPECTION_COLD_START_GRACE) is subtracted separately, so the
# nudge fires at iteration N+cold_start (default 7).
# How many iterations we let the model have AFTER a forceful nudge to
# actually begin creating. Bumped from 1 → 3 2026-04-20 — giving the
# nudge a real chance to land before killing the turn.


def _intent_key(tc: dict) -> tuple:
    """Deterministic exact-args dedup key for one tool call.

    Keys on ``(tool_name, sorted input items)``. Two calls with the
    same tool and identical input dict produce the same key. Used by
    the within-batch dedup in _act_hybrid so a model that emits
    ``file_read(path=X)`` twice in one turn only hits the container
    once — the second call gets a synthetic "duplicate of canonical"
    tool_result pointing at the first.

    NOTE: this is EXACT-args dedup. Rotating variants like
    ``head -5 X && tail -5 X`` vs ``wc -l X && tail -5 X`` produce
    different keys and run separately — different-spelling dedup is a
    follow-up (Fix B on the 2026-04-20 plan).
    """
    name = tc.get("name") or tc.get("function", {}).get("name", "")
    input_ = tc.get("input") or tc.get("function", {}).get("arguments", {})
    if isinstance(input_, str):
        try:
            input_ = json.loads(input_)
        except (json.JSONDecodeError, TypeError):
            input_ = {}
    if not isinstance(input_, dict):
        return (name, ())
    # Sort items and json-serialise values so nested dicts / lists key
    # deterministically. `sort_keys=True` on json.dumps handles nested
    # objects; sorting the outer items handles key order drift.
    items = tuple(sorted(
        (k, json.dumps(v, sort_keys=True, default=str))
        for k, v in input_.items()
    ))
    return (name, items)

# Tools that require explicit user approval under
# `permissions=confirm_mutations`. Includes workspace mutators plus
# shell/git because those can reach outside the current task scope.
_APPROVAL_REQUIRED_TOOLS = _MUTATING_TOOLS | {
    "shell_exec", "git", "service_start", "service_stop", "profile_update",
}
_ALWAYS_CONFIRM_TOOLS = frozenset({"publish_ports"})

# Universal backstop on tool_result size. Individual tools truncate their own
# output (tools._truncate caps at 50 KB, http_request at 50 KB, doc_fetch at
# max_chars, …), but a path that BYPASSES those caps used to produce a result
# big enough to blow the outer transport limit ("Response too large: 614794
# bytes (limit 500000)") — a HARD fetch failure the model got NO usable result
# from. This clamp runs at the single chokepoint every tool_result passes
# through (``_append_tool_result_to_history``), so NO tool, regardless of which
# one or what args the model passed, can ever produce a dead fetch: it always
# degrades to a usable truncated result plus a hint to use code_search /
# paginate. Measured in BYTES (the transport limit is bytes) and held well
# under 500 KB to leave room for the message/JSON envelope.
_TOOL_RESULT_HARD_CAP_BYTES = 256_000


def _clamp_tool_result_bytes(text: str, tool_name: str) -> str:
    """Clamp ``text`` to ``_TOOL_RESULT_HARD_CAP_BYTES`` UTF-8 bytes, appending
    a recourse hint when it fires. Returns ``text`` unchanged when it fits."""
    if not text:
        return text
    encoded = text.encode("utf-8")
    total = len(encoded)
    if total <= _TOOL_RESULT_HARD_CAP_BYTES:
        return text
    # Truncate on a byte budget, then decode tolerantly (a multi-byte char may
    # straddle the cut — ``errors='ignore'`` drops the partial trailing byte).
    head = encoded[:_TOOL_RESULT_HARD_CAP_BYTES].decode("utf-8", "ignore")
    log.warning(
        "tool_result_hard_clamped",
        tool=tool_name, total_bytes=total, cap_bytes=_TOOL_RESULT_HARD_CAP_BYTES,
    )
    return (
        head
        + f"\n\n[TRUNCATED by Augmentum: {tool_name} returned {total} bytes, "
        f"over the {_TOOL_RESULT_HARD_CAP_BYTES // 1000} KB tool-result limit. "
        f"Showing the first {_TOOL_RESULT_HARD_CAP_BYTES // 1000} KB. To get "
        f"what you need without the overflow: use `code_search` (semantic — "
        f"returns only the relevant chunks), narrow the query/path, or "
        f"paginate (e.g. file_read with offset/limit).]"
    )


# Serve-side tool-result cap, scaled to the compaction ceiling. Individual
# tools clip their own output at 50 KB (``coder.tools._truncate`` et al.) —
# sized for big-window models. On a small-window deployment (e.g.
# ``AUGMENTUM_CODER_COMPACT_TOKENS=16384`` for a trained 16k-window model),
# a single 50k-char result (~12.5k tokens) blows the whole window, so the
# chokepoint below (``_append_tool_result_to_history``) re-clips each result
# to at most ~25% of the effective window (chars ≈ tokens × 4), floored at
# 8k chars so tiny limits don't starve results, ceiled at the 50k default so
# big-window deployments are byte-identical to pre-scaling behavior.
_MAX_OUTPUT_CHARS = 50_000
_OUTPUT_CAP_FLOOR_CHARS = 8_000
_OUTPUT_CAP_WINDOW_FRACTION = 0.25
_OUTPUT_CAP_CHARS_PER_TOKEN = 4


def _scaled_output_cap_chars(limit_tokens: int | None) -> int:
    """Effective per-result char cap for a given compaction token limit."""
    try:
        tokens = int(limit_tokens or 0)
    except (TypeError, ValueError):
        tokens = 0
    if tokens <= 0:
        return _MAX_OUTPUT_CHARS
    scaled = int(tokens * _OUTPUT_CAP_CHARS_PER_TOKEN * _OUTPUT_CAP_WINDOW_FRACTION)
    return min(_MAX_OUTPUT_CHARS, max(_OUTPUT_CAP_FLOOR_CHARS, scaled))


def _truncate_output(text: str, max_chars: int = _MAX_OUTPUT_CHARS) -> str:
    """Both-ends clip in the same shape as ``coder.tools._truncate``.

    No-op whenever ``max_chars`` is at (or above) the 50k default — the
    tools already clip at that size, and skipping entirely guarantees
    byte-identical behavior for big-window deployments. Only fires when
    the compaction ceiling scales the cap DOWN (small-window models).
    Like ``tools._truncate``, the notice is prepended AND appended so
    the signal survives compaction's short tool_result previews.
    """
    total = len(text)
    if max_chars >= _MAX_OUTPUT_CHARS or total <= max_chars:
        return text
    header = (
        f"[TRUNCATED — showing first {max_chars} of {total} chars. "
        "Result clipped to fit this model's context window. For "
        "file_read, use its own paging hint for the next line offset; "
        "for search results (code_grep / find_files), narrow your "
        "pattern or use a smaller `limit`.]\n\n"
    )
    return (
        header
        + text[:max_chars]
        + f"\n\n... (truncated, {total} total chars)"
    )


# Internal workspace paths created by coder infrastructure itself. These
# should never surface in user-facing turn review because they're
# implementation detail (scratch spillover, transient agent state).
_INTERNAL_REVIEW_PREFIXES = (
    "/workspace/.augmentum/",
)

# Read-only / explanatory turns should stay on inspection tools unless the
# user explicitly asks to run something. ``shell_read`` is intentionally
# excluded even though it is nominally "read-only" because it can still run
# programs like ``python app.py`` and create noisy side effects in the UX.
_EXPLANATORY_SAFE_TOOLS = frozenset(
    set(READ_ONLY_TOOLS) - {"shell_read"} | {"finish_task", "ask_user"}
)
_EXPLANATORY_GIT_ACTIONS = frozenset({"status", "diff", "log"})
_EXPLANATORY_EXECUTION_RE = re.compile(
    r"\b("
    r"run|execute|launch|start|boot|spin\s+up|open|test|pytest|"
    r"build|compile|install|benchmark|profile|debug"
    r")\b",
    re.IGNORECASE,
)
_EXPLICIT_SETUP_RE = re.compile(
    r"\b("
    r"install|setup|bootstrap|dependencies|dependency|deps|"
    r"requirements|venv|virtualenv|environment|tooling"
    r")\b",
    re.IGNORECASE,
)
_ENV_BOOTSTRAP_INSTALL_RE = re.compile(
    r"\b(?:apt-get|apt)\s+(?:update|install)\b"
    r"|\bpip3?\s+install\b"
    r"|\bpython3?\s+-m\s+pip\s+install\b"
    r"|\bnpm\s+install\b"
    r"|\byarn\s+install\b"
    r"|\bpnpm\s+install\b"
    r"|\buv\s+pip\s+install\b",
    re.IGNORECASE,
)
_ENV_DISCOVERY_SHELL_RE = re.compile(
    r"^\s*(?:"
    r"which\s+python(?:3)?\b"
    r"|command\s+-v\s+python(?:3)?\b"
    r"|type\s+python(?:3)?\b"
    r"|python(?:3)?\s+--version\b"
    r"|pytest\s+--version\b"
    r"|python(?:3)?\s+-m\s+pytest\s+--version\b"
    r"|find\s+/(?:usr|opt|home|root)\b.*\bpython\b"
    r"|ls\s+-la\s+/(?:usr|bin)/python[^\n]*"
    r"|ps\s+aux\s+\|\s+grep\b.*(?:apt|python)"
    r")",
    re.IGNORECASE,
)
_ENVIRONMENT_AUDIT_RE = re.compile(
    r"\b("
    r"environment|workspace|container|runtime|runtimes|tooling|tools|"
    r"languages|packages|installed|pre-?installed|what can you do"
    r")\b",
    re.IGNORECASE,
)
_TEST_EXEC_SHELL_RE = re.compile(
    r"\b(?:pytest|python3?\s+-m\s+pytest|npm\s+test|npx\s+(?:jest|vitest)|go\s+test)\b",
    re.IGNORECASE,
)
_BROWSER_DISCOVERY_SHELL_RE = re.compile(
    r"^\s*(?:"
    r"which\s+(?:chromium|chromium-browser|google-chrome)\b"
    r"|command\s+-v\s+(?:chromium|chromium-browser|google-chrome)\b"
    r"|pip3?\s+list\b.*\b(?:playwright|selenium|httpx)\b"
    r"|python3?\s+-m\s+pip\s+show\b.*\b(?:playwright|selenium)\b"
    r"|(?:playwright|chromium|google-chrome)\s+--version\b"
    r")",
    re.IGNORECASE,
)
_BROWSER_EXEC_SHELL_RE = re.compile(
    r"\b(?:xvfb-run|chromium|chromium-browser|google-chrome|playwright|selenium)\b"
    r"|file:///workspace/",
    re.IGNORECASE,
)
_ROOT_WORKSPACE_SHELL_RE = re.compile(
    r"^\s*(?:"
    r"(?:ls|find|tree|du)\b[^\n]*\s/workspace(?:\s|$)"
    r"|cd\s+/workspace\s*&&\s*(?:ls|find|tree|du)\b"
    r"|pwd\s*&&\s*ls\b[^\n]*\s/workspace(?:\s|$)"
    r")",
    re.IGNORECASE,
)
_POPULATED_REPO_EMPTY_QUESTION_RE = re.compile(
    r"\b(?:"
    r"empty\s+workspace"
    r"|workspace\s+(?:appears|is)\s+(?:to\s+be\s+)?empty"
    r"|workspace\s+appears\s+empty"
    r"|workspace\s+only\s+contains"
    r"|only\s+contains\s+a\s+workspace\s+guide"
    r"|no\s+(?:existing\s+)?(?:repository|repo|codebase)\b"
    r"|no\s+(?:repository|repo)\s+has\s+been\s+cloned(?:\s+yet)?"
    r"|what\s+would\s+you\s+like\s+me\s+to\s+(?:create|build|work\s+on|improve)"
    r"|clone\s+(?:an?\s+)?(?:existing\s+)?repo"
    r"|clone\s+a\s+specific\s+(?:repository|repo)"
    r"|provide\s+a\s+url"
    r"|paste\s+repository\s+contents"
    r"|sample\s+project"
    r"|different\s+environment"
    r"|python\s+web\s+application"
    r"|node(?:\.js)?\s+rest\s+api"
    r"|cli\s+tool"
    r")\b",
    re.IGNORECASE,
)
_POPULATED_REPO_FALSE_EMPTY_RE = re.compile(
    r"\b(?:"
    r"empty\s+workspace"
    r"|workspace\s+(?:appears|is)\s+(?:to\s+be\s+)?empty"
    r"|no\s+(?:existing\s+)?(?:repository|repo|codebase)"
    r"|no\s+(?:repository|repo)\s+has\s+been\s+cloned(?:\s+yet)?"
    r"|only\s+contains\s+(?:documentation|workspace\.md|plan\.md|\.augmentum)"
    r"|only\s+contains\s+a\s+workspace\s+guide"
    r"|provide\s+(?:the\s+)?url"
    r"|paste\s+repository\s+contents"
    r"|clone\s+a\s+specific\s+(?:repository|repo)"
    r"|clone\s+(?:a\s+specific|an?\s+existing)?\s*(?:repository|repo)"
    r"|create\s+(?:a\s+)?sample\s+project"
    r"|different\s+environment"
    r"|what\s+would\s+you\s+like\s+me\s+to\s+work\s+on"
    r")\b",
    re.IGNORECASE,
)
_REPOISH_ROOT_NAMES = frozenset({
    ".git",
    "README",
    "README.md",
    "README.rst",
    "README.txt",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "package.json",
    "Cargo.toml",
    "go.mod",
    "Makefile",
    "src",
    "tests",
    "docs",
})
_ROOT_PROBE_CANDIDATES = (
    ".git",
    ".github",
    "README.md",
    "README.rst",
    "README.txt",
    "README",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "package.json",
    "Cargo.toml",
    "go.mod",
    "Makefile",
    "src",
    "tests",
    "docs",
    "examples",
)

# Maximum characters in a tool result preview included in metadata chunks
_PREVIEW_CHARS = 200

# Expanded preview for tools where the output IS the deliverable
_EXPANDED_PREVIEW_TOOLS = frozenset({
    "dir_tree", "file_read", "code_grep", "find_files", "code_search",
    "env_info", "git", "test_run", "shell_exec", "shell_read", "doc_search",
    "pack_search", "container_info", "apply_patch",
})
_EXPANDED_PREVIEW_CHARS = 800

# Per-tool overrides for tools whose output is a discrete enumerable
# list and every item matters to the user. The UI render path
# (``ui/scripts/coder.js`` task_list branch) caps at 40 lines, so we
# size to fit ~40 tasks at ~80 chars each (status marker + content).
# The default 200-char cap clipped to ~3 tasks before the preview ever
# reached the frontend; bumping at the tool-name granularity avoids
# bloating unrelated tool previews.
_PREVIEW_OVERRIDES: dict[str, int] = {
    "task_list": 3200,
}


# Failed-result previews need their own cap. The raw error + the
# tool's ``error_hints`` recovery suggestion (appended by
# ``Tool.enrich_error``) together routinely run 600-1500 chars — the
# hint is the actionable bit, so clipping at 200 hides exactly what
# the user needs to read to understand "why did file_write fail and
# what should the model do next?". Bumped 2026-05-31 after the
# file_write missing-path hint was reaching the UI as
# "...Required: path (string) + content (string)…" with the entire
# recovery suggestion ("most common cause: model output budget ran
# out — switch to code_edit_batch") clipped off mid-sentence.
_ERROR_PREVIEW_CHARS = 2000


def _preview_len(tool_name: str, *, success: bool = True) -> int:
    """Return preview char limit for a tool — expanded for output-centric
    tools, and further expanded for error results so the recovery hint
    isn't clipped off the end."""
    if not success:
        # Errors get a generous cap so the enriched recovery hint
        # always lands intact. Still bounded — a runaway shell stderr
        # shouldn't blow the meta chunk size.
        base = _PREVIEW_OVERRIDES.get(tool_name) or (
            _EXPANDED_PREVIEW_CHARS if tool_name in _EXPANDED_PREVIEW_TOOLS
            else _PREVIEW_CHARS
        )
        return max(_ERROR_PREVIEW_CHARS, base)
    if tool_name in _PREVIEW_OVERRIDES:
        return _PREVIEW_OVERRIDES[tool_name]
    return _EXPANDED_PREVIEW_CHARS if tool_name in _EXPANDED_PREVIEW_TOOLS else _PREVIEW_CHARS


# Tool groups for soft-failure target extraction. Kept in sync with the
# analogous sets in CoderState for ``hit_repeat_cap`` — different constants
# so they can evolve independently (state tracks the "already inspected"
# surface; this tracks the "kept failing" surface).
_FAILURE_TARGETS_BY_PATH = frozenset({
    "file_read", "file_write", "code_edit", "code_edit_batch",
    "file_list", "dir_tree",
})
_FAILURE_TARGETS_BY_COMMAND = frozenset({
    "shell_read", "shell_exec", "test_run",
})
_FAILURE_TARGETS_BY_QUERY = frozenset({
    "code_grep", "find_files", "code_search",
})


# "Action-like" code fence languages — markdown fences that a weak model
# uses as a substitute for actual tool calls. Observed 2026-04-20 on
# Qwen 3.6: model wrote ``` ```bash curl http://... ``` ``` in prose and
# stopped, with zero shell_exec calls. Our continuation judge saw the
# long prose and fired ``model_stop_with_answer`` — but nothing ran.
# Neither Codex nor Claude Code has a defense for this failure mode;
# this regex is the first half of ours (detection), paired with a
# forced-nudge override on the acceptance side.
_ACTION_CODE_FENCE_RE = re.compile(
    r"```(?:bash|sh|shell|zsh|console|terminal|powershell|pwsh)\b",
    re.IGNORECASE,
)


def _has_unclaimed_code_block(text: str) -> bool:
    """True if ``text`` contains an action-like fenced code block.

    "Action-like" means bash/sh/shell/zsh/console/terminal — the kinds
    of fences a model writes when describing a command it should have
    called via `shell_exec`. Python / JS / other fences are fine in
    prose (code snippets, explanations) and don't trip this.
    """
    if not text:
        return False
    return bool(_ACTION_CODE_FENCE_RE.search(text))


_INLINE_TOOL_MARKUP_RE = re.compile(
    r"(?im)^\s*(?:<tool_call>\s*)?([a-z_][a-z0-9_]*)\s*:\s*(.+?)\s*(?:</tool_call>)?\s*$",
)
_INLINE_TEXT_TOOL_NAMES = frozenset({
    "shell_exec", "shell_read", "test_run",
    "file_read", "file_write", "file_list", "dir_tree",
    "code_edit", "code_edit_batch", "apply_patch",
    "code_search", "find_files", "code_grep",
    "doc_search", "doc_fetch", "pack_search",
    "git", "env_info", "container_info", "task_list", "ask_user", "finish_task",
})
_INLINE_TOOL_SINGLE_ARG_KEYS = {
    "shell_exec": "command",
    "shell_read": "command",
    "test_run": "command",
    "file_read": "path",
    "file_list": "path",
    "dir_tree": "path",
    "apply_patch": "patch",
    "doc_fetch": "url",
    "doc_search": "query",
    "pack_search": "query",
    "code_search": "query",
    "find_files": "pattern",
    "code_grep": "pattern",
}


def _has_unclaimed_tool_markup(text: str) -> bool:
    """True if ``text`` contains pseudo-tool-call markup in prose.

    Weak/open models sometimes narrate intended tool calls as plain text like:

        <tool_call>shell_exec: pytest -q

    or:

        shell_exec: curl -I http://localhost:8080

    If the fallback parser fails to recover these, the continuation judge must
    not accept them as a substantive answer because nothing actually ran.
    """
    if not text:
        return False
    for m in _INLINE_TOOL_MARKUP_RE.finditer(text):
        name = (m.group(1) or "").strip()
        if name in _INLINE_TEXT_TOOL_NAMES:
            return True
    return False


_LEAKED_TOOL_MARKUP_RE = re.compile(
    r"<tool_call>|<function=([a-zA-Z0-9_]+)",
)


def _has_leaked_tool_markup(text: str) -> bool:
    """True if ``text`` contains RAW XML-style tool-call markup.

    Distinct from ``_has_unclaimed_tool_markup`` (one-line ``name: args``
    narration): this catches the Qwen-XML form emitted in the WRONG
    channel —

        <tool_call>
        <function=file_read>
        <parameter=path>...</parameter>
        </function>
        </tool_call>

    Live failure 2026-07-02 (Qwythos-9B, run …0d0d4a6ebc): the model
    wrote two such calls INSIDE its thinking block. llama.cpp's tool
    parser only fires on markup in the content channel after the think
    closer, so nothing executed — then the model reasoned "I've already
    gathered a lot of information" (it hadn't) and stopped the turn
    early. Small merges trained on interleaved-reasoning transcripts
    reproduce the format without the channel discipline; the stop gate
    must not accept a wrap-up built on actions that never ran.

    A bare ``<tool_call>`` counts even without a recognizable function
    name — the marker itself has no legitimate reason to appear verbatim
    in reasoning or prose. Named ``<function=`` markup is checked
    against the known coder tool names to avoid flagging quoted
    third-party code.
    """
    if not text:
        return False
    for m in _LEAKED_TOOL_MARKUP_RE.finditer(text):
        name = m.group(1)
        if name is None or name in _INLINE_TEXT_TOOL_NAMES:
            return True
    return False


def _has_resumable_objective_state(state) -> bool:
    """True iff the handler has real in-progress objective state to resume.

    A prior clarification turn stores ``Question: ...`` in ``state.plan`` but
    that is not executable objective state. Treating it as resumable causes a
    later "continue" to revive stale clarification text instead of starting a
    fresh plan/act cycle.
    """
    plan = (getattr(state, "plan", "") or "").strip()
    has_real_plan = bool(plan) and not _plan_is_question(plan)
    return bool(
        has_real_plan
        or getattr(state, "tasks", None)
        or getattr(state, "mission", None)
        or getattr(state, "pending_objective_contract", None)
    )


def _is_compact_command(text: str) -> bool:
    """True for the Coder manual compaction slash command."""
    return (text or "").strip().lower() in {"/compact", "/compact now"}


# Within-response content-loop detector. qwen-code's checkContentLoop
# uses 50-char sliding chunks with markdown-structure resets; we use
# simpler token-based n-gram deduplication because our prose is
# post-processed (tool-call JSON stripped, CoT tokens stripped, code
# fences collapsed) before it reaches this check — the qwen-code
# resets aren't needed for a heuristic this coarse. Observed pattern:
# "Let me run these now. Running diagnostics: ```...``` Let me run
# these now. Running diagnostics: ```...``` Let me run these..." —
# 8-token window catches that verbatim.
_CONTENT_LOOP_WINDOW = 8
_CONTENT_LOOP_MIN_REPEATS = 3
_CONTENT_LOOP_MIN_TEXT_CHARS = 200


# Detect plan-phase output that's a clarification QUESTION (per the
# VAGUE branch of PLAN_SYSTEM) rather than an actual Plan. When the
# model emits "Question: ..." with options, the act phase should be
# skipped — the user needs to answer before any work can be planned.
# Observed 2026-04-20 on "Write tests for the main functionality":
# model correctly produced a Question, but act phase ran anyway and
# the model burned 6 iterations exploring before inspection-only-loop
# stopped it. Matcher is permissive: allow leading whitespace / up to
# ~200 chars of preamble before the "Question:" marker, because
# weaker models sometimes emit a one-line "Let me think..." header.
# ``_plan_is_question`` and ``_PLAN_QUESTION_RE`` moved to
# ``phase_plan.py``. Re-imported below for in-handler callers.


def _has_content_loop(
    text: str,
    *,
    window: int = _CONTENT_LOOP_WINDOW,
    min_repeats: int = _CONTENT_LOOP_MIN_REPEATS,
) -> bool:
    """True if any ``window``-token substring repeats ``min_repeats``+ times.

    Short text (< _CONTENT_LOOP_MIN_TEXT_CHARS chars) always returns
    False — a tight response can't be degenerate by construction, and
    we don't want false positives on terse outputs. Tokenisation is
    whitespace-split (not syntactic); that's cheap and matches how the
    observed failures manifest.
    """
    if not text or len(text) < _CONTENT_LOOP_MIN_TEXT_CHARS:
        return False
    tokens = text.split()
    if len(tokens) < window * min_repeats:
        return False
    counts: dict[tuple[str, ...], int] = {}
    for i in range(len(tokens) - window + 1):
        gram = tuple(tokens[i:i + window])
        counts[gram] = counts.get(gram, 0) + 1
        if counts[gram] >= min_repeats:
            return True
    return False


def _soft_failure_target(tool_name: str, tool_input: dict) -> str:
    """Pick the argument that identifies WHERE a soft failure happened.

    Used as the dedup key for ``CoderState.record_tool_failure`` so the
    sticky reminder shows one row per (tool, target) rather than one per
    failure. e.g. 5 mtime-guard rejections on ``/snake.html`` becomes
    ``code_edit /snake.html × 5`` instead of five separate lines.

    Returns ``""`` for tools with no obvious single-target argument —
    those get bucketed together under the tool name, which is still
    better than being untracked.
    """
    if not isinstance(tool_input, dict):
        return ""
    if tool_name in _FAILURE_TARGETS_BY_PATH:
        return str(tool_input.get("path") or "")[:160]
    if tool_name in _FAILURE_TARGETS_BY_COMMAND:
        return str(tool_input.get("command") or "")[:160]
    if tool_name in _FAILURE_TARGETS_BY_QUERY:
        return str(
            tool_input.get("pattern")
            or tool_input.get("query")
            or tool_input.get("text")
            or "",
        )[:160]
    return ""


# Chain-of-thought delimiter tokens that various model families leak into
# their output (Gemini, Qwen, DeepSeek, GPT-harmony). These aren't tool
# calls and aren't meant for the user — strip them from any displayed prose.
_COT_TOKEN_RE = re.compile(r"<\|[^|>\n]{1,40}\|>")
_THINK_BLOCK_RE = re.compile(
    r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE,
)

# ``_PLAN_MARKER_RE`` moved to ``phase_plan.py`` along with
# ``_plan_phase``. Not re-imported — handler no longer references it.


def _strip_cot_tokens(text: str) -> str:
    """Remove model-internal CoT delimiters and think blocks from displayed prose.

    Handles:
      - ``<|mask_start|>``, ``<|mask_end|>``, ``<|channel|>``, ``<|message|>``,
        ``<|thinking|>``, ``<|begin_of_thought|>``, etc. — anything matching
        the ``<|...|>`` shape under 40 chars.
      - ``<think>...</think>`` / ``<thinking>...</thinking>`` full blocks.

    Applied per streamed delta; a token split across two deltas will leak,
    but in practice tokenizers emit these whole.
    """
    if not text:
        return text
    text = _THINK_BLOCK_RE.sub("", text)
    text = _COT_TOKEN_RE.sub("", text)
    return text


# Gemini-style ``tool_code[N] ```python ... ```` blocks. Some models trained
# with Google's code-execution protocol emit these instead of our structured
# tool-call format. We convert to a ``shell_exec`` with a base64-encoded
# Python payload so the model's intent is honoured inside the sandbox.
_TOOL_CODE_RE = re.compile(
    r"tool_code\s*\[?\s*\d*\s*\]?\s*```(?:python|py)?\s*\n(.*?)\n\s*```",
    re.DOTALL | re.IGNORECASE,
)


def _extract_tool_code_blocks(text: str) -> list[dict]:
    """Convert Gemini-style ``tool_code[N] ```python ... ```` blocks to shell_exec calls.

    Returns a list of normalised tool-call dicts ({id, name, input}).
    Empty list if no such blocks are present.
    """
    import base64

    results: list[dict] = []
    for match in _TOOL_CODE_RE.finditer(text):
        code = match.group(1).strip()
        if not code:
            continue
        code_b64 = base64.b64encode(code.encode("utf-8")).decode("ascii")
        # Base64 wrapping sidesteps all quoting issues with inline Python
        command = (
            f"python3 -c \"import base64;"
            f"exec(base64.b64decode('{code_b64}').decode())\""
        )
        results.append({
            "id": str(uuid.uuid4()),
            "name": "shell_exec",
            "input": {"command": command},
        })
    return results


def _strip_tool_json(text: str) -> str:
    """Strip tool-call JSON objects from text using brace-depth matching.

    Handles nested braces (e.g. file_write with code content containing
    braces, escaped quotes, etc.).  The simple regex [^}]* fails on these.
    """
    if "<task_list>" in text:
        text = re.sub(r"<task_list>[\s\S]*?</task_list>", "", text, flags=re.IGNORECASE)
    if '"tool"' not in text:
        return text
    result: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] == "{" and text[i:i + 7].lstrip().startswith('{"tool"'):
            # Walk forward counting braces
            depth = 0
            j = i
            in_string = False
            escape = False
            while j < n:
                ch = text[j]
                if escape:
                    escape = False
                    j += 1
                    continue
                if ch == "\\" and in_string:
                    escape = True
                    j += 1
                    continue
                if ch == '"' and not escape:
                    in_string = not in_string
                if not in_string:
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            j += 1
                            break
                j += 1
            i = j
        else:
            result.append(text[i])
            i += 1
    return "".join(result)


class CoderHandler(LegacyStrategyMixin, PlanPhaseMixin, ActPhaseMixin, ModeHandler):
    """Plan/Act coding agent with streaming metadata.

    Lifecycle
    ---------
    WAITING  → Plan phase (LLM generates step list)
             → Act phase  (LLM calls tools to execute steps)
             → WAITING    (ready for next user message)

    Falls back to Phase-1 passthrough when ``container_manager`` is ``None``.
    """

    def __init__(
        self,
        backend,
        *,
        session_id: str = "",
        tool_registry=None,
        container_manager=None,
        executor=None,
        workspace_id: str = "",
        permission_callback=None,
        question_callback=None,
        review_registry=None,
        user_id: str = "",
        state_manager: StateManager | None = None,
        power_registry: PowerRegistry | None = None,
        settings_store=None,
        mcp_client=None,
        coder_strategy: str = "",
        provider_registry=None,
        coder_run_broker=None,
        jobs_store=None,
        vision_router=None,
        dispatch: CoderDispatch | None = None,
    ) -> None:
        self._backend = backend
        # Provider registry for role-based model resolution. When set, the
        # plan phase resolves its model via ``resolve_model_for_role`` using
        # the ``model_role`` declared on :class:`PromptMeta` (currently
        # ``utility`` for plan/replan prompts). When ``None`` (tests, older
        # construction sites), every phase uses the bound ``backend`` — the
        # single-model parity behaviour we had before this wire.
        self._provider_registry = provider_registry
        self._session_id = session_id
        self._tool_registry = tool_registry
        self._container_manager = container_manager
        self._workspace_id = workspace_id or session_id
        # The controller depends ONLY on the WorkspaceExecutor CONTRACT for
        # workspace I/O + metadata + capabilities — it must never reach
        # self._container_manager directly (that coupling is exactly what leaked
        # the loop into one surface and broke the editor path). When no executor
        # was injected (normal container construction), wrap the container
        # manager so the container path stays byte-identical; the ACP editor
        # path injects a RemoteEditorExecutor. Either way self._executor is the
        # single surface the loop talks to — the seam a future harness plugs.
        from augmentum.coder.executors import ContainerExecutor
        self._executor = executor
        if self._executor is None and container_manager is not None:
            self._executor = ContainerExecutor(container_manager, self._workspace_id)
        # Optional async callback for per-tool approval when the
        # permissions policy is "confirm_mutations". Signature:
        # ``async (tool_name: str, tool_input: dict) -> bool``. When None
        # and the policy requires confirmation, the call is denied so the
        # model gets a cleanly-formatted refusal to react to.
        self._permission_callback = permission_callback
        # Optional async callback for mid-task user questions raised by
        # the ``ask_user`` tool. Signature:
        # ``async (questions: list[dict]) -> list[str | dict] | None``.
        # Returning None signals the user cancelled; the tool surfaces
        # that as a clean error so the model can proceed or give up.
        # Kept distinct from permission_callback because questions carry
        # rich structured options and always have a user-facing answer,
        # whereas permission is boolean and typically silent on consent.
        self._question_callback = question_callback
        # Optional ReviewRegistry the handler publishes turn bundles into
        # when an agent turn ends. None in tests or deployments without
        # the review flow wired — the act loops no-op the publish step
        # in that case. See augmentum/coder/reviews.py.
        self._review_registry = review_registry
        # User id captured at handler build time so the review bundle
        # can scope to the right tenant. Tests default to "" — matches
        # the PermissionRegistry single-tenant-dev convention.
        self._user_id = user_id
        self._state_manager = state_manager
        self._power_registry = power_registry
        self._settings_store = settings_store
        self._mcp_client = mcp_client
        # Background-job queue reference so the bug_finder_run coder tool
        # can enqueue audit runs. None in tests / older construction
        # paths — the tool is then omitted from the registry.
        self._jobs_store = jobs_store
        # Substrate vision (Slot C classifier+mmproj → SmolVLM CPU
        # fallback) — captions browser screenshots for NON-vision coder
        # models so they aren't blind either. See
        # _maybe_screenshot_image_message.
        self._vision_router = vision_router
        # Orchestrator contract (CoderDispatch). Set when a non-user
        # orchestrator (companion, scheduled job, external CLI) composed a
        # spec instead of a raw user turn: its ``success_criteria`` seed the
        # turn's mission (giving the P3 verifier gate an inbound contract) and
        # its rendered fork-system block is injected as a stable system part.
        # None on the direct-user path — behaviour is then byte-identical.
        self._dispatch = dispatch
        self._dispatch_system_block = ""
        if dispatch is not None:
            from augmentum.coder.dispatch import render_dispatch_system
            from augmentum.coder.prompts import DISPATCH_FORK_SYSTEM
            try:
                self._dispatch_system_block = render_dispatch_system(
                    dispatch, fork_prompt=DISPATCH_FORK_SYSTEM,
                )
            except Exception:
                self._dispatch_system_block = ""
        # Iterative visual comparison (verification-spine follow-on):
        # remember the previous screenshot's path + vision caption so the
        # NEXT screenshot's feed can say "here's what iteration N-1
        # looked like" — turning isolated glances into a compare loop.
        # (path, caption) for the non-VL caption path; a counter for the
        # VL pixel path (the previous image is already in message
        # history, the model just needs to be told to compare).
        self._last_screenshot_note: tuple[str, str] | None = None
        self._screenshot_pixel_feeds = 0
        self._profile_store = None
        self._service_store = None
        # Lazy-built SubagentDispatcher — None until first
        # ``_get_subagent_dispatcher`` call, then memoized. Only
        # constructed when ``settings.coder_subagents_enabled`` is True
        # AND the app exposed ``coder_agent_registry`` + provider
        # registry. Wired into ``create_coder_tools`` via the
        # ``subagent_dispatcher=`` kwarg so the task_dispatch tool
        # surfaces only in supported sessions.
        self._subagent_dispatcher = None
        # Background-task pool for fire-and-forget archive writes from
        # sync code paths (e.g., ``_record_interruption_summary``).
        # Holding references prevents asyncio from GC-ing in-flight
        # tasks. Cleared automatically via done_callback.
        self._interruption_archive_tasks: set = set()
        try:
            conn = getattr(getattr(state_manager, "backend", None), "conn", None)
            if conn is not None:
                from augmentum.coder.profile import CoderProfileStore
                from augmentum.coder.services import CoderServiceStore

                self._profile_store = CoderProfileStore(conn)
                self._service_store = CoderServiceStore(conn)
        except Exception:
            # Not fatal — both stores stay None and callers guard on
            # truthiness — but it silently disables profile/service
            # features (per-workspace memory, dev-server tools), so the
            # operator needs to see it. warning, not debug.
            log.warning("coder_runtime_store_init_failed", exc_info=True)
        # Per-request strategy override (from X-Augmentum-Coder-Strategy
        # header). Empty falls back to the env-var default at dispatch
        # time. Stored raw — sanitised by ``_resolve_strategy``.
        self._coder_strategy_override = (coder_strategy or "").strip().lower()
        # Optional broker for detaching the agent loop from the HTTP
        # request lifecycle. When set and ``settings.coder_background_runs``
        # is True, ``_handle_stream`` registers the run with the broker
        # and yields chunks via subscription — so a client disconnect
        # (mobile screen sleep, tab switch) leaves the agent task
        # running and the UI can reattach via the /stream route. When
        # None (tests, legacy callers), behavior matches the original
        # in-request execution path.
        self._coder_run_broker = coder_run_broker

        # In-memory state (Phase 3 will persist to SQLite)
        self._state = CoderState(
            session_id=session_id,
            workspace_id=self._workspace_id,
        )
        self._cached_guide: str | None = None
        # Rendered <workspace_facts> block (objective + identity +
        # observations). Refreshed at turn-start by
        # ``_refresh_kernel_facts``; consumed by ``_build_messages``
        # (canonical/hybrid/plan) and ``_act_native``. Single source
        # of truth so the same block surfaces consistently across
        # every strategy without each one re-rendering.
        self._cached_facts_block: str = ""
        # Byte-stable repo-map block (code_intel.render_repo_map) —
        # refreshed at turn start, rides the STABLE prefix in every
        # strategy's prompt. No line numbers/timestamps by design.
        self._repo_map_context_block: str = ""
        self._turn_start_workspace_paths: set[str] | None = None
        self._active_power_context_block = ""
        self._active_power_summary: dict | None = None
        self._controller_power_context_block = ""
        self._controller_power_summary: dict | None = None
        self._controller_power_windows_fired: set[str] = set()
        # System-driven explore dispatch (subagent-router Power → run
        # explore_codebase ourselves). Findings block injected into the plan
        # prompt; fired-once guard so a single turn dispatches at most one.
        self._auto_explore_context_block = ""
        self._auto_explore_fired = False
        self._controller_edited_paths: list[str] = []
        self._pending_power_activation_events: list[dict[str, object]] = []
        self._power_followup_nudges_seen: set[tuple[str, str, str]] = set()
        self._workspace_tree_authoritative_for_turn: bool = False
        self._workspace_tree_file_count_for_turn: int = 0
        self._workspace_git_url_for_turn: str = ""
        self._workspace_root_probe_populated_for_turn: bool = False
        self._workspace_root_probe_context_block: str = ""
        self._turn_intent_for_turn = TurnIntent(TurnIntentKind.UNKNOWN)
        # Tier classification for the current turn. Defaults to COMPOSED
        # (current hybrid-loop behavior) until classify_tier runs in
        # _handle_stream_body. Read by phase_act._act_hybrid to bound
        # the iteration cap per tier.
        self._turn_tier_for_turn: TierClassification = TierClassification(
            tier=Tier.COMPOSED, reason="default",
        )
        self._runtime_truth_for_turn: RuntimeTruth | None = None
        self._runtime_truth_context_block = ""
        # Live-preview console/error events surfaced at turn-start (push). The
        # watermark advances per turn so each event is injected once. See
        # preview_console.py + _render_preview_console_block.
        self._preview_console_context_block = ""
        self._preview_console_seen = 0
        self._turn_ledger = None
        self._model_context_window_for_turn: int = 0
        self._coder_compact_token_limit: int = int(_COMPACT_AT_TOKENS)
        self._coder_digest_token_budget: int = _coder_digest_token_budget()
        # Tracks the (tasks, mission) signature at last DB persist. Used by
        # ``_persist_state_if_dirty`` so mid-iter persists are cheap when
        # nothing the inspector cares about has changed.
        self._last_persisted_state_signature: str = ""

        # Auto-refreshing workspace tree — the model sees file listings
        # without having to ``dir_tree`` / ``ls`` for them. Marked stale
        # after every successful mutation tool, refreshed lazily at
        # injection points. None in Phase 1 (no container) so ``render``
        # short-circuits.
        from augmentum.coder.snapshot import WorkspaceSnapshot
        self._workspace_snapshot: WorkspaceSnapshot | None = (
            WorkspaceSnapshot(container_manager, self._workspace_id)
            if container_manager is not None else None
        )

        # Filesystem-as-scratchpad (Manus "ultimate context" pattern).
        # Externalises oversized tool outputs to files so the inline
        # conversation body stays under compaction thresholds while the
        # full content remains recoverable via ``file_read``. Handler
        # hooks into ``_append_tool_result_to_history``; no IO happens
        # until a result actually crosses the size threshold.
        from augmentum.coder.scratch import ScratchStore
        self._scratch_store: ScratchStore | None = (
            ScratchStore(container_manager, self._workspace_id)
            if container_manager is not None else None
        )

        # Workspace-kernel v2 (docs/superpowers/specs/2026-05-16-workspace-
        # kernel-design.md). Maintains /workspace/.augmentum/ files the
        # model reads on demand instead of having content re-framed into
        # the message stream every turn. Constructed unconditionally so
        # tests + the flag-gated migrations can rely on its presence;
        # the refresh + reader calls are themselves gated by the
        # coder_kernel_v2 setting in config.
        from augmentum.coder.workspace_kernel import WorkspaceKernel
        self._workspace_kernel: WorkspaceKernel | None = (
            WorkspaceKernel(container_manager, self._workspace_id)
            if container_manager is not None else None
        )

    # ------------------------------------------------------------------
    # ModeHandler interface
    # ------------------------------------------------------------------

    async def _handle(
        self, request: InternalChatRequest,
    ) -> InternalChatResponse:
        parts: list[str] = []
        async for chunk in self._handle_stream(request):
            if chunk.content_delta:
                parts.append(chunk.content_delta)
        return InternalChatResponse(
            message=Message(role="assistant", content="".join(parts)),
            model=request.model,
            finish_reason="stop",
        )

    async def _restore_state(self) -> None:
        """Reload persisted coder state before processing a user turn."""
        if not self._state_manager or not self._session_id:
            return
        try:
            restored = await self._state_manager.load_coder_state(
                self._session_id,
                user_id=self._user_id,
            )
            if restored is None:
                return
            self._state = restored
            self._state.session_id = self._session_id
            self._state.workspace_id = self._workspace_id
            log.info(
                "coder_state_restored",
                session_id=self._session_id,
                workspace_id=self._workspace_id,
                tasks=len(self._state.tasks or []),
                turn_summaries=len(self._state.turn_summaries or []),
            )
        except Exception as exc:
            log.warning(
                "coder_state_load_failed",
                session_id=self._session_id,
                workspace_id=self._workspace_id,
                error=str(exc),
            )

    async def _persist_state(self) -> None:
        """Persist coder state after a completed request."""
        if not self._state_manager or not self._session_id:
            return
        try:
            self._state.session_id = self._session_id
            self._state.workspace_id = self._workspace_id
            await self._state_manager.save_coder_state(
                self._session_id,
                self._state,
                user_id=self._user_id,
            )
            log.info(
                "coder_state_persisted",
                session_id=self._session_id,
                workspace_id=self._workspace_id,
                tasks=len(self._state.tasks or []),
                turn_summaries=len(self._state.turn_summaries or []),
            )
            # Surface to the inspector panel so it refreshes without
            # waiting for the next ~5s poll cycle. Server-scoped event
            # (workspace_id in payload; the inspector filters by its
            # own workspace context).
            try:
                from augmentum.proxy import system_events
                system_events.publish(
                    "coder.state_updated",
                    {
                        "workspace_id": self._workspace_id,
                        "session_id": self._session_id,
                        "tasks_count": len(self._state.tasks or []),
                        "mission_count": len(self._state.mission or []),
                    },
                    user_id=self._user_id or "",
                )
            except Exception:
                log.debug("coder_state_publish_failed", exc_info=True)
        except Exception as exc:
            log.warning(
                "coder_state_persist_failed",
                session_id=self._session_id,
                workspace_id=self._workspace_id,
                error=str(exc),
            )

    async def _persist_state_if_dirty(self) -> bool:
        """Persist only when (tasks, mission) have changed since last write.

        Called from mid-turn hot paths (after each act iteration) so the
        inspector panel can see live progress. Cheap when nothing
        material changed — just an O(N) JSON hash compare, no DB hit.
        Returns True if a persist actually fired.
        """
        if not self._state_manager or not self._session_id:
            return False
        try:
            signature = json.dumps(
                {
                    "t": self._state.tasks or [],
                    "m": [
                        p.to_dict() if hasattr(p, "to_dict") else p
                        for p in (self._state.mission or [])
                    ],
                },
                sort_keys=True, default=str,
            )
        except Exception:
            # Hash failure shouldn't block the turn; skip the optimisation.
            await self._persist_state()
            return True
        if signature == self._last_persisted_state_signature:
            return False
        self._last_persisted_state_signature = signature
        await self._persist_state()
        return True

    # ------------------------------------------------------------------
    # Cooperative turn handling — pause gate + inbox drain
    # ------------------------------------------------------------------

    def _current_broker_run_id(self) -> str:
        """The active run_id the broker entry is keyed by, or empty string.

        Cooperative drain + pause logic only does anything when there's
        an active ledger AND broker — i.e. the detached background-runs
        path. The in-request legacy path never has a broker entry and
        the helpers below all short-circuit to no-op.
        """
        if self._turn_ledger is None:
            return ""
        return self._turn_ledger.run_id or ""

    def _format_steer_content(self, msg: dict) -> str:
        """Render a steer inbox entry as user-message content.

        Steer messages are added to the agent's in-flight message list
        as ``role=user``. We wrap them in an explicit "user interjected
        mid-turn" framing so the model treats them as new instructions
        rather than continuations of the original turn (which they
        deliberately are not — the user changed direction). Empty-
        content steers (attachments only) still get the wrapper so the
        model knows the user just dropped files in.
        """
        body = (msg.get("content") or "").strip()
        atts = msg.get("attachments") or []
        att_lines = ""
        if atts:
            names = [
                str(a.get("name") or a.get("path") or "")
                for a in atts if isinstance(a, dict)
            ]
            names = [n for n in names if n]
            if names:
                att_lines = "\n[Attached: " + ", ".join(names[:6]) + "]"
        if not body and not att_lines:
            return "[User interjected — no content]"
        return f"[User interjected mid-turn]\n{body}{att_lines}"

    async def _coop_iteration_check(self) -> list[dict]:
        """Pause gate + steer drain. Called at each strategy's iter top.

        Returns the list of drained steer messages (possibly empty) so
        the caller can append them to its message list and emit a
        chunk noting the inject. Awaiting the pause gate yields the
        loop cleanly when paused — no busy-wait, no polling. When
        unpaused (the common case) the await returns immediately.

        Returns an empty list when there's no broker (legacy in-
        request path) or no active run_id — preserves the existing
        loop shape with zero behavioural change in those modes.
        """
        # User-controlled request pacing, applied BEFORE the broker check so
        # it covers every act loop (broker-backed AND legacy in-request) —
        # this method is the single per-iteration choke-point all three loops
        # share. No-op unless the user turned pacing on. See
        # ``_maybe_pace_request``.
        await self._maybe_pace_request()
        if self._coder_run_broker is None:
            return []
        run_id = self._current_broker_run_id()
        if not run_id:
            return []
        # Pause first — if the user paused mid-turn, the steer drain
        # happens AFTER they resume so all the pending messages land
        # in one batch on resume. Reverse order would leave drained
        # messages waiting through a long pause.
        await self._coder_run_broker.await_pause_gate(run_id)
        drained = self._coder_run_broker.drain_user_messages(
            run_id, mode="steer",
        )
        if drained:
            log.info(
                "coder.coop_steer_delivered",
                run_id=run_id,
                count=len(drained),
            )
        return drained

    async def _maybe_pace_request(self) -> None:
        """Sleep before a loop iteration when the user enabled request pacing.

        Fast cloud models (Gemini flash-lite, Groq, etc.) can fire tool-call
        turns quickly enough to trip a provider rate limit (429), which the
        loop then wastes retries on. When ``coder_request_delay_enabled`` is
        set, sleep ``coder_request_delay_seconds`` before each model request so
        the cadence stays under the limit. Off by default (no sleep). The value
        is clamped to a sane ceiling so a mistyped number can't wedge a turn.
        """
        from augmentum.config import settings as _settings
        if not getattr(_settings, "coder_request_delay_enabled", False):
            return
        try:
            secs = float(getattr(_settings, "coder_request_delay_seconds", 0.0) or 0.0)
        except (TypeError, ValueError):
            return
        if secs <= 0:
            return
        secs = min(secs, 120.0)  # ceiling: never wedge the loop on a fat finger
        log.debug("coder.request_pacing_sleep", seconds=secs)
        await asyncio.sleep(secs)

    def _steer_interrupt_pending(self) -> bool:
        """True when a mid-generation steer is waiting AND mid-reasoning
        interrupt is enabled.

        Cheap non-destructive peek (dict get + len) — safe to call per
        streamed reasoning chunk. Always False on the legacy in-request
        path (no broker) or when ``coder_steer_interrupt_reasoning`` is
        off, so those flows are byte-for-byte unchanged.
        """
        from augmentum.config import settings as _settings
        if not _settings.coder_steer_interrupt_reasoning:
            return False
        if self._coder_run_broker is None:
            return False
        run_id = self._current_broker_run_id()
        if not run_id:
            return False
        return self._coder_run_broker.inbox_depth(run_id, mode="steer") > 0

    def _drain_steer_for_interrupt(self) -> list[dict]:
        """Pop pending steer entries for the mid-reasoning interrupt path.

        Mirrors :meth:`_coop_iteration_check`'s drain but WITHOUT the pause
        gate — we're mid-generation, not at a boundary. Safe no-op on the
        no-broker path.
        """
        if self._coder_run_broker is None:
            return []
        run_id = self._current_broker_run_id()
        if not run_id:
            return []
        return self._coder_run_broker.drain_user_messages(run_id, mode="steer")

    def _plan_mode_addendum(self) -> str:
        """System prompt prefix when planning_mode is ``plan``.

        Soft guidance — every tool remains available (hard filter was
        retired in migration 208). The model has agency to act, but is
        framed toward propose-first collaboration. Trust the model to
        read this and adjust; if you wanted enforcement, you'd use a
        deny rule in .augmentum/permissions.toml instead.

        Empty string in non-plan modes (no-op concatenation).
        """
        if getattr(self, "_planning_mode", "auto") != "plan":
            return ""
        return (
            "\n\n## Plan Mode\n\n"
            "The user has set this workspace to plan mode. They want "
            "to collaborate on the approach before changes land. "
            "Naturally bias toward:\n"
            "  1. Reading the relevant code first to ground your "
            "thinking in what's actually there.\n"
            "  2. Outlining your proposed approach in clear steps "
            "(which files, what changes, why).\n"
            "  3. Inviting the user to confirm or adjust before you "
            "start editing.\n\n"
            "All tools remain available — use your judgment. If the "
            "request is small and obvious (a typo fix, a one-line "
            "rename) it's fine to propose-then-do in the same turn. "
            "If the request is ambiguous, multi-file, or has design "
            "trade-offs, pause after the proposal and wait. The user "
            "will cycle out of plan mode (Shift+Tab) to signal "
            "\"proceed without checking in.\""
        )

    def _coop_drain_dropped_inbox(self) -> list[dict]:
        """Flush every pending interjection (queue + steer) on cancel/error.

        Called from the handler's exception paths so messages queued
        against an interrupted turn don't sit in their "queued" /
        "steering" badge state forever. The route layer's UI handler
        flips those badges to "dropped" so the user sees the message
        wasn't delivered and can retype.

        Returns the drained entries (in FIFO order) so the caller can
        emit a ``queue_dropped`` chunk listing the ids. Safe no-op
        when there's no broker / no run_id (legacy in-request path
        doesn't have an inbox).
        """
        if self._coder_run_broker is None:
            return []
        run_id = self._current_broker_run_id()
        if not run_id:
            return []
        # Drain everything, mode-unfiltered. Both queue + steer
        # messages get the same "dropped" treatment because both are
        # waiting for a turn that's no longer going to complete.
        return self._coder_run_broker.drain_user_messages(run_id, mode=None)

    def _coop_drain_queued_followups(self) -> list[dict]:
        """End-of-turn drain. Pops EVERY undelivered inbox entry.

        Called by the broker-path wrapper AFTER the agent loop exits
        naturally (NOT on cancel/error — queued follow-ups from a
        rewound or errored turn would land in a state the user
        doesn't expect). Returns the drained list so the caller can
        emit a stream chunk telling the frontend to chain into a new
        turn with these as the user prompt.

        Drains ``mode=None`` (everything), not just ``queue``:
        a ``steer`` entry still in the inbox here never hit an
        iteration boundary — the user interjected while the model was
        writing its final response (or the loop exited before the next
        boundary). Before 2026-07-03 those entries were stranded: the
        end-of-turn drain skipped them, no chunk ever resolved them,
        the UI badge stayed "steering" forever, and the content
        evicted with the broker entry — the user had to retype.
        Promoting them into the follow-up chain preserves the intent
        (the model never saw the instruction; the next turn acts on
        it) without retyping.

        The frontend-chaining design avoids reentering the agent loop
        inside the same broker task — each drained queue entry
        becomes a separate ledger turn with its own run_id, preserving
        the run-per-turn invariant the ledger + rewind paths depend on.
        """
        if self._coder_run_broker is None:
            return []
        run_id = self._current_broker_run_id()
        if not run_id:
            return []
        drained = self._coder_run_broker.drain_user_messages(
            run_id, mode=None,
        )
        if drained:
            log.info(
                "coder.coop_queue_drained",
                run_id=run_id,
                count=len(drained),
                promoted_steers=sum(
                    1 for m in drained if m.get("mode") == "steer"
                ),
            )
        return drained

    async def _refresh_coder_token_budgets(self, model: str) -> None:
        """Derive per-turn Coder budgets from the active model window."""
        context_window = 0
        get_context_length = getattr(self._backend, "get_context_length", None)
        if callable(get_context_length):
            try:
                context_window = int(await get_context_length(model) or 0)
            except Exception:
                log.debug(
                    "coder_context_length_probe_failed",
                    model=model,
                    exc_info=True,
                )
                context_window = 0

        self._model_context_window_for_turn = max(0, context_window)
        self._coder_compact_token_limit = _coder_context_token_limit(
            self._model_context_window_for_turn,
        )
        self._coder_digest_token_budget = _coder_digest_token_budget(
            self._model_context_window_for_turn,
        )

    async def _capture_turn_workspace_baseline(self) -> None:
        """Record the pre-turn workspace tree for review diffs."""
        self._turn_start_workspace_paths = None
        if self._workspace_snapshot is None:
            return
        try:
            await self._workspace_snapshot.refresh_if_stale(force=True)
            self._turn_start_workspace_paths = self._workspace_snapshot.current_paths
        except Exception:
            log.debug("workspace_turn_baseline_failed", exc_info=True)

    async def _load_active_power_for_turn(self) -> None:
        """Resolve the current workspace's active Power into one prompt block."""
        self._active_power_context_block = ""
        self._active_power_summary = None
        if self._power_registry is None or self._settings_store is None:
            return
        try:
            from augmentum.powers import PowerStateStore

            state = PowerStateStore(self._settings_store)
            active = await state.get_active_power(
                self._user_id,
                workspace_id=self._workspace_id,
            )
            if active is None:
                return
            manifest = self._power_registry.get_power(active.power_id)
            if manifest is None:
                return
            if manifest.mode_scope and "coder" not in manifest.mode_scope:
                return
            if not await state.is_enabled(self._user_id, manifest.id):
                return
            health = self._power_registry.evaluate_health(
                manifest,
                mcp_client=self._mcp_client,
                tool_registry=self._tool_registry,
            )
            if health.status != "ready":
                return
            self._active_power_context_block = manifest.render_prompt_block()
            self._active_power_summary = manifest.to_summary_dict(
                enabled=True,
                health=health,
                active=active,
            )
            self._queue_power_activation_event(
                {
                    "id": manifest.id,
                    "display_name": manifest.display_name,
                    "kind": manifest.kind,
                    "activation_policy": manifest.activation_policy,
                    "activation_windows": list(manifest.activation_windows),
                    "checkpoint": "pre_plan",
                    "reason": active.reason or "workspace power pinned by user",
                    "source": active.source or "manual",
                    "scope": active.scope or "workspace",
                    "transient": False,
                },
            )
        except Exception:
            log.debug("coder_active_power_load_failed", exc_info=True)

    async def _maybe_activate_controller_power(
        self,
        checkpoint: str,
        *,
        latest_user_text: str = "",
        edited_paths: list[str] | None = None,
        force: bool = False,
    ) -> bool:
        """Select a controller-managed Power for a loop checkpoint.

        Controller powers are transient to the current request turn.
        They do not overwrite the user's pinned workspace Power.
        """
        if checkpoint in self._controller_power_windows_fired and not force:
            return False
        if self._power_registry is None or self._settings_store is None:
            return False
        try:
            from augmentum.powers import PowerActivation, PowerStateStore
            from augmentum.powers.controller import select_controller_power

            state = PowerStateStore(self._settings_store)
            manifests = []
            health_by_id: dict[str, object] = {}
            for manifest in self._power_registry.list_powers():
                if manifest.mode_scope and "coder" not in manifest.mode_scope:
                    continue
                if not await state.is_enabled(self._user_id, manifest.id):
                    continue
                health = self._power_registry.evaluate_health(
                    manifest,
                    mcp_client=self._mcp_client,
                    tool_registry=self._tool_registry,
                )
                if health.status != "ready":
                    continue
                manifests.append(manifest)
                health_by_id[manifest.id] = health

            selection = select_controller_power(
                manifests,
                checkpoint=checkpoint,
                latest_user_text=latest_user_text or self._last_user_message_for_summary(),
                edited_paths=edited_paths if edited_paths is not None else list(self._controller_edited_paths),
                current_controller_power_id=(self._controller_power_summary or {}).get("id", ""),
                manual_power_id=(self._active_power_summary or {}).get("id", ""),
            )
            if selection is None:
                return False

            manifest = selection.manifest
            previous_id = (self._controller_power_summary or {}).get("id", "")
            activation = PowerActivation(
                power_id=manifest.id,
                workspace_id=self._workspace_id,
                source="controller",
                scope="turn",
                reason=selection.reason,
            )
            summary = manifest.to_summary_dict(
                enabled=True,
                health=health_by_id[manifest.id],
                active=activation,
            )
            summary["active_checkpoint"] = checkpoint
            self._controller_power_context_block = (
                f'<controller_power checkpoint="{checkpoint}" policy="{manifest.activation_policy}">\n'
                f"{manifest.render_prompt_block()}\n"
                f"Activation reason: {selection.reason}\n"
                "</controller_power>"
            )
            self._controller_power_summary = summary
            self._controller_power_windows_fired.add(checkpoint)
            if manifest.id != previous_id:
                self._queue_power_activation_event({
                    "id": manifest.id,
                    "display_name": manifest.display_name,
                    "kind": manifest.kind,
                    "activation_policy": manifest.activation_policy,
                    "activation_windows": list(manifest.activation_windows),
                    "checkpoint": checkpoint,
                    "reason": selection.reason,
                    "source": "controller",
                    "scope": "turn",
                    "transient": True,
                })
            log.info(
                "coder_controller_power_activated",
                checkpoint=checkpoint,
                power_id=manifest.id,
                reason=selection.reason,
                previous_id=previous_id,
            )
            return manifest.id != previous_id
        except Exception:
            log.debug(
                "coder_controller_power_activate_failed",
                checkpoint=checkpoint,
                exc_info=True,
            )
            return False

    async def _maybe_auto_dispatch_explore(self, *, latest_user_text: str, model: str):
        """System-driven delegation: dispatch ``explore_codebase`` ourselves.

        Local/open models reliably DON'T call the delegation tool even when
        it's offered and the prompt coaches them to (validated 2026-06-19 in
        a live coder turn: Qwen3-Coder used ``code_grep``/``file_list``, never
        delegated). The ``subagent-router`` Power already DETECTS the
        explore-shaped signal deterministically at ``pre_plan`` — so when it's
        the active controller pick, we stop nudging and run the explore
        subagent for the model, injecting its findings into the plan context.

        Async generator: yields meta chunks for the live UI; its real output
        is the side effect ``self._auto_explore_context_block`` that
        ``_render_dynamic_context`` folds into the plan prompt. Fires at most
        once per turn. Gated on ``coder_subagent_auto_explore`` +
        ``coder_subagents_enabled`` (the dispatcher is None otherwise).
        """
        if self._auto_explore_fired:
            return
        from augmentum.config import settings as _settings
        if not getattr(_settings, "coder_subagent_auto_explore", False):
            return
        # Only when the subagent-router Power is THIS turn's controller pick —
        # that's the deterministic "an explore would help here" signal.
        if (self._controller_power_summary or {}).get("id") != "subagent-router":
            return
        if not _text_is_explore_shaped(latest_user_text):
            return
        dispatcher = self._get_subagent_dispatcher()
        if dispatcher is None:
            return
        query = (latest_user_text or "").strip()
        if not query:
            return

        self._auto_explore_fired = True
        yield self._meta_chunk(
            phase="planning", status="auto_explore_started", model=model,
            extra={"auto_explore": {"query": query[:200]}},
        )
        try:
            from augmentum.coder.tools import ExploreCodebaseTool
            tool = ExploreCodebaseTool(
                container_manager=self._container_manager,
                workspace_id=self._workspace_id,
                state=self._state,
                dispatcher=dispatcher,
                profile_store=getattr(self, "_profile_store", None),
                service_store=getattr(self, "_service_store", None),
                user_id=self._user_id,
            )
            result = await tool.execute(query=query)
        except Exception:
            log.warning("coder_auto_explore_failed", exc_info=True)
            yield self._meta_chunk(
                phase="planning", status="auto_explore_failed", model=model,
            )
            return

        findings = (getattr(result, "output", "") or "").strip()
        if findings:
            self._auto_explore_context_block = (
                '<auto_exploration note="The system ran an explore subagent on '
                'your request BEFORE planning. Ground your plan in these '
                'findings; do not re-grep what is already covered here.">\n'
                f"{findings}\n"
                "</auto_exploration>"
            )
        meta = getattr(result, "metadata", {}) or {}
        yield self._meta_chunk(
            phase="planning", status="auto_explore_done", model=model,
            extra={"auto_explore": {
                "ok": bool(getattr(result, "success", False)),
                "subagent_id": meta.get("subagent_id", ""),
            }},
        )

    def _queue_power_activation_event(self, event: dict[str, object]) -> None:
        """Queue a power-activation event for later streaming emission."""
        event_id = str(event.get("id") or "")
        checkpoint = str(event.get("checkpoint") or "")
        source = str(event.get("source") or "")
        for existing in self._pending_power_activation_events:
            if (
                str(existing.get("id") or "") == event_id
                and str(existing.get("checkpoint") or "") == checkpoint
                and str(existing.get("source") or "") == source
            ):
                return
        self._pending_power_activation_events.append(dict(event))

    def _drain_pending_power_activation_events(self) -> list[dict[str, object]]:
        """Return and clear queued power-activation events in FIFO order."""
        events = list(self._pending_power_activation_events)
        self._pending_power_activation_events = []
        return events

    def _take_pending_power_activation_event(self) -> dict | None:
        """Return the next queued power-activation event, if any."""
        if not self._pending_power_activation_events:
            return None
        return self._pending_power_activation_events.pop(0)

    def _active_power_summaries(self) -> list[dict]:
        """Return active Power summaries in precedence order.

        Controller-selected transient Powers come first because they are
        the most contextually relevant specialist for the current
        checkpoint; manually pinned workspace Powers come second.
        """
        summaries: list[dict] = []
        for summary in (self._controller_power_summary, self._active_power_summary):
            if isinstance(summary, dict) and summary.get("id"):
                summaries.append(summary)
        return summaries

    def _active_verifier_power_ids(self) -> list[str]:
        """Return the currently active verifier Power ids without duplicates."""
        seen: set[str] = set()
        ordered: list[str] = []
        for summary in self._active_power_summaries():
            if str(summary.get("kind") or "") != "verifier":
                continue
            power_id = str(summary.get("id") or "").strip()
            if power_id and power_id not in seen:
                seen.add(power_id)
                ordered.append(power_id)
        return ordered

    def _render_power_closeout_contract(self) -> str:
        """Return verifier-specific close-out instructions for final answers."""
        verifier_ids = self._active_verifier_power_ids()
        if not verifier_ids:
            return ""

        bullets: list[str] = []
        if "failure-triage" in verifier_ids:
            bullets.append(
                "state the user-visible symptom, the first failing signal, and the root cause",
            )
        if "test-author" in verifier_ids:
            bullets.append(
                "name the focused test or validation you added or ran, and whether it passed",
            )
        if "browser-verification" in verifier_ids:
            bullets.append(
                "name the exact page, route, or flow you checked and one concrete browser/runtime signal",
            )
        if "release-review" in verifier_ids:
            bullets.append(
                "say plainly whether the work is release-ready, or what still blocks release",
            )
        if not bullets:
            return ""

        joined = "\n".join(f"- {item}" for item in bullets)
        return (
            "Do not end with a generic wrap-up. In your final answer, use short "
            "evidence-backed bullets that:\n"
            f"{joined}"
        )

    @staticmethod
    def _tool_result_brief(tool_result: dict | None) -> str:
        """Return the first useful preview/error line from a tool_result payload."""
        if not isinstance(tool_result, dict):
            return ""
        raw = str(
            tool_result.get("output_preview")
            or tool_result.get("error")
            or "",
        ).strip()
        if not raw:
            return ""
        for line in raw.splitlines():
            line = re.sub(r"\s+", " ", line).strip()
            if line:
                return line[:240]
        return re.sub(r"\s+", " ", raw)[:240]

    def _find_tool_result_evidence(
        self,
        tool_results: list[dict] | None,
        *,
        tools: tuple[str, ...] = (),
        keywords: tuple[str, ...] = (),
        prefer_failures: bool = False,
        prefer_success: bool = False,
    ) -> str:
        """Pick one relevant tool-result preview for user-facing evidence lines."""
        lowered_tools = {t.lower() for t in tools if t}
        lowered_keywords = tuple(k.lower() for k in keywords if k)
        candidates: list[tuple[dict, str]] = []
        for tr in tool_results or []:
            if not isinstance(tr, dict):
                continue
            tool_name = str(tr.get("tool") or "").lower()
            preview = self._tool_result_brief(tr)
            haystack = f"{tool_name}\n{preview}".lower()
            matched = bool(
                (lowered_tools and tool_name in lowered_tools)
                or (
                    lowered_keywords
                    and any(keyword in haystack for keyword in lowered_keywords)
                )
            )
            if matched and preview:
                candidates.append((tr, preview))

        if not candidates:
            return ""
        if prefer_failures:
            for tr, preview in candidates:
                if not tr.get("success", True):
                    return preview
        if prefer_success:
            for tr, preview in candidates:
                if tr.get("success", False):
                    return preview
        return candidates[0][1]

    def _render_verifier_fallback_lines(
        self,
        *,
        tool_results: list[dict] | None,
        writes: list[str],
        last_err: str,
    ) -> list[str]:
        """Add specialist evidence lines when a verifier Power was active."""
        verifier_ids = set(self._active_verifier_power_ids())
        if not verifier_ids:
            return []

        test_evidence = self._find_tool_result_evidence(
            tool_results,
            tools=("test_run",),
            keywords=(
                " passed",
                " failed",
                "pytest",
                "jest",
                "vitest",
                "go test",
                "tests:",
                "fail ",
                "pass ",
            ),
            prefer_failures=True,
            prefer_success=True,
        )
        browser_evidence = self._find_tool_result_evidence(
            tool_results,
            keywords=(
                "playwright",
                "browser",
                "page ",
                "route",
                "http://",
                "https://",
                "localhost",
                "signup",
                "login",
                "rendered",
            ),
            prefer_failures=True,
            prefer_success=True,
        )
        diagnosis_evidence = self._find_tool_result_evidence(
            tool_results,
            keywords=(
                "traceback",
                "exception",
                "error",
                "failed",
                "failing",
                "assert",
                "not found",
                "cannot",
            ),
            prefer_failures=True,
        )

        lines: list[str] = []
        if "failure-triage" in verifier_ids:
            if diagnosis_evidence:
                lines.append(
                    f"**Diagnosis:** First concrete failure signal: {diagnosis_evidence}",
                )
            elif last_err:
                lines.append(
                    f"**Diagnosis:** The clearest captured failure signal was: {last_err}",
                )
            else:
                lines.append(
                    "**Diagnosis:** The turn ended without one crisp failing signal or root-cause statement.",
                )

        if "browser-verification" in verifier_ids:
            if browser_evidence:
                lines.append(f"**Browser evidence:** {browser_evidence}")
            else:
                lines.append(
                    "**Browser evidence:** No concrete browser/runtime signal was captured in this turn.",
                )

        if "test-author" in verifier_ids or "release-review" in verifier_ids:
            if test_evidence:
                lines.append(f"**Validation evidence:** {test_evidence}")
            elif writes:
                lines.append(
                    "**Validation evidence:** No focused passing test was confirmed in this turn.",
                )

        if "release-review" in verifier_ids:
            if last_err or diagnosis_evidence:
                lines.append(
                    "**Release gate:** The work is not yet proven release-ready; resolve the failing signal above or rerun the most relevant validation.",
                )
            elif test_evidence or browser_evidence:
                lines.append(
                    "**Release gate:** One relevant verification signal was captured this turn; only the remaining risk needs another check.",
                )
            elif writes:
                lines.append(
                    "**Release gate:** Changes landed, but release readiness was not cleanly proven before the turn ended.",
                )

        return lines

    def _is_environment_audit_turn(self, user_goal: str) -> bool:
        """True when the current inspect turn is primarily about the environment."""
        if self._turn_intent_for_turn.kind != TurnIntentKind.INSPECT:
            return False
        return bool(_ENVIRONMENT_AUDIT_RE.search((user_goal or "").strip()))

    def _render_environment_provenance_lines(self) -> list[str]:
        """Render observed vs intended environment facts for inspect summaries."""
        truth = self._runtime_truth_for_turn
        if truth is None:
            return []

        observed_bits: list[str] = []
        for name in ("python3", "node", "go", "rustc", "java"):
            value = (truth.observed_runtimes.get(name) or "").strip()
            if value and value.lower() != "missing":
                observed_bits.append(f"{name}: {value}")
        for name in ("pip", "npm", "cargo"):
            value = (truth.observed_package_managers.get(name) or "").strip()
            if value and value.lower() != "missing":
                observed_bits.append(f"{name}: {value}")

        if truth.workspace_mode == "fallback":
            baseline_desc = (
                f"{truth.workspace_image or 'ubuntu:24.04'} fallback workspace with "
                "the guide-critical bootstrap subset"
            )
        elif truth.workspace_mode == "prebaked":
            baseline_desc = (
                f"{truth.workspace_image or 'augmentum-workspace'} prebaked workspace image"
            )
        else:
            baseline_desc = "workspace baseline unavailable from container metadata"

        if truth.missing_baseline:
            missing_desc = ", ".join(truth.missing_baseline)
            if truth.missing_optional:
                missing_desc += (
                    f" (optional extras also absent: {', '.join(truth.missing_optional)})"
                )
        elif truth.missing_optional:
            missing_desc = (
                "no core baseline gaps detected; optional extras absent: "
                + ", ".join(truth.missing_optional)
            )
        else:
            missing_desc = "no core baseline gaps detected in the direct probe"

        lines = [
            (
                "**Observed now:** "
                + ("; ".join(observed_bits) if observed_bits else "No direct runtime probe succeeded this turn.")
            ),
            f"**Intended baseline:** {baseline_desc}.",
            f"**Missing / not observed:** {missing_desc}.",
            (
                "**Interpretation:** Treat only the observed section as confirmed present right now. "
                "Use the intended baseline as design intent, not proof."
            ),
        ]
        return lines

    @staticmethod
    def _messages_include_tool_call(
        messages: list[Message],
        tool_name: str,
    ) -> bool:
        """Return True when the conversation history already includes `tool_name`."""
        wanted = (tool_name or "").strip()
        if not wanted:
            return False
        for message in messages:
            if message.role != "assistant":
                continue
            for tc in (getattr(message, "tool_calls", None) or []):
                if isinstance(tc, dict):
                    name = (
                        tc.get("name")
                        or tc.get("function", {}).get("name", "")
                    )
                    if name == wanted:
                        return True
        return False

    @staticmethod
    def _messages_include_shell_command_matching(
        messages: list[Message],
        pattern: re.Pattern[str],
    ) -> bool:
        """Return True when prior assistant tool calls include a matching shell command."""
        for message in messages:
            if message.role != "assistant":
                continue
            for tc in (getattr(message, "tool_calls", None) or []):
                if not isinstance(tc, dict):
                    continue
                name = (
                    tc.get("name")
                    or tc.get("function", {}).get("name", "")
                )
                if name != "shell_exec":
                    continue
                args = tc.get("input") or tc.get("function", {}).get("arguments") or {}
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except Exception:
                        args = {}
                if not isinstance(args, dict):
                    continue
                command = str(args.get("command") or "")
                if command and pattern.search(command):
                    return True
        return False

    def _maybe_refuse_verifier_shell_exec(
        self,
        *,
        command: str,
        latest_input: str,
        goal_text: str,
        messages: list[Message],
    ) -> str:
        """Return a refusal reason for shell bootstrap churn during verifier turns."""
        verifier_ids = set(self._active_verifier_power_ids())
        if not verifier_ids:
            return ""
        cmd = (command or "").strip()
        if not cmd:
            return ""

        user_text = f"{latest_input}\n{goal_text}"
        if (
            _ENV_BOOTSTRAP_INSTALL_RE.search(cmd)
            and not _EXPLICIT_SETUP_RE.search(user_text)
        ):
            extra = ""
            if "pytest" in cmd.lower():
                extra = (
                    " For tiny Python regressions, prefer unittest from the "
                    "standard library over installing pytest."
                )
            return (
                "Refused: a verifier Power is active and this shell command is "
                "trying to install or update tooling just to prove the change. "
                "Do not apt-get / pip install / npm install unless the user "
                "explicitly asked for environment setup. Use env_info or test_run "
                f"to inspect the runtime, or report the missing dependency clearly.{extra}"
            )

        if (
            self._messages_include_tool_call(messages, "env_info")
            and _ENV_DISCOVERY_SHELL_RE.search(cmd)
        ):
            return (
                "Refused: env_info already captured the runtime basics this turn. "
                "Reuse that signal or move to one concrete proof step (prefer "
                "test_run or the smallest relevant command) instead of more "
                "Python/pytest discovery shell commands."
            )
        if _TEST_EXEC_SHELL_RE.search(cmd):
            return (
                "Refused: a verifier Power is active and this looks like a raw "
                "test command inside shell_exec. Use test_run instead so the loop "
                "gets structured pass/fail feedback. If pytest is unavailable for "
                "a tiny Python regression, fall back to unittest rather than "
                "installing packages."
            )
        if "browser-verification" in verifier_ids:
            if _BROWSER_DISCOVERY_SHELL_RE.search(cmd):
                return (
                    "Refused: Browser Verification is active and this command only "
                    "discovers local browser tooling. Do not burn turns on `which "
                    "chromium` / `pip list playwright` probes. Either do one direct "
                    "browser-facing proof step, or switch to static validation and "
                    "state that runtime browser execution was not available."
                )
            if (
                _BROWSER_EXEC_SHELL_RE.search(cmd)
                and self._messages_include_shell_command_matching(
                    messages, _BROWSER_EXEC_SHELL_RE,
                )
            ):
                return (
                    "Refused: Browser Verification already attempted one browser "
                    "execution step this turn. Do not keep retrying browser shell "
                    "commands. Reuse the captured runtime signal, fix the page, or "
                    "fall back to one static validation pass and report the remaining "
                    "uncertainty explicitly."
                )
        return ""

    def _maybe_refuse_root_workspace_reinspection(
        self,
        *,
        tool_name: str,
        tool_input: dict,
    ) -> str:
        """Refuse broad /workspace rediscovery when an authoritative tree already exists."""
        if not self._workspace_has_populated_repo_for_turn():
            return ""
        has_cloned_repo_signal = bool(self._workspace_git_url_for_turn)
        has_root_probe_signal = self._workspace_root_probe_populated_for_turn
        has_large_authoritative_tree = (
            self._workspace_tree_authoritative_for_turn
            and self._workspace_tree_file_count_for_turn >= 40
        )
        if not has_cloned_repo_signal and not has_root_probe_signal and not has_large_authoritative_tree:
            return ""

        if tool_name in {"dir_tree", "file_list"}:
            path = str(tool_input.get("path") or "/workspace").strip() or "/workspace"
            if path.rstrip("/") != "/workspace":
                return ""
        elif tool_name in {"shell_exec", "shell_read"}:
            command = str(tool_input.get("command") or "")
            if not _ROOT_WORKSPACE_SHELL_RE.search(command):
                return ""
        else:
            return ""

        if has_cloned_repo_signal:
            repo_descriptor = "cloned existing repository"
        elif has_root_probe_signal:
            repo_descriptor = "existing repository detected at /workspace root"
        else:
            repo_descriptor = (
                f"populated repo ({self._workspace_tree_file_count_for_turn} files)"
            )
        return (
            "Refused: this turn already has an authoritative workspace tree for a "
            f"{repo_descriptor}. "
            "Do not re-list /workspace root with dir_tree, file_list, or ls/find. "
            "Use the existing workspace_tree in the system prompt, then inspect a "
            "narrower subdirectory or read likely entry files (for example README, "
            "pyproject, src/, tests/, docs/) to move the task forward."
        )

    async def _probe_workspace_root_for_turn(self) -> None:
        """Run one cheap controller-side /workspace probe for repo-shaped roots."""
        self._workspace_root_probe_populated_for_turn = False
        self._workspace_root_probe_context_block = ""
        if self._container_manager is None or not self._workspace_id:
            return

        visible_entries: list[str] = []
        for name in _ROOT_PROBE_CANDIDATES:
            try:
                await self._container_manager._run_command(
                    self._workspace_id,
                    ["test", "-e", f"/workspace/{name}"],
                    timeout=3.0,
                )
            except Exception as exc:
                log.debug("root_probe_failed", name=name, error=str(exc))
                continue
            display = name
            try:
                await self._container_manager._run_command(
                    self._workspace_id,
                    ["test", "-d", f"/workspace/{name}"],
                    timeout=3.0,
                )
                display = f"{name}/"
            except Exception as exc:
                # test -d returns non-zero (or run_command raises) when
                # `name` isn't a directory — leave display as the bare
                # name and let the visible listing show "f" without a
                # trailing slash.
                log.debug(
                    "coder_workspace_probe_isdir_failed",
                    entry=name,
                    error=str(exc),
                )
            visible_entries.append(display)

        if not visible_entries:
            try:
                entries = await self._container_manager.file_list(
                    self._workspace_id,
                    "/workspace",
                )
            except Exception:
                log.debug("coder_workspace_root_probe_failed", exc_info=True)
                return

            repoish = False
            for entry in entries or []:
                name = str(getattr(entry, "name", "") or "").strip()
                if not name or name in {".", "..", ".augmentum"}:
                    continue
                is_dir = bool(getattr(entry, "is_dir", False))
                visible_entries.append(f"{name}/" if is_dir else name)
                if name in _REPOISH_ROOT_NAMES:
                    repoish = True

            if not visible_entries:
                return

            if not repoish and len(visible_entries) < 3:
                return

        self._workspace_root_probe_populated_for_turn = True
        preview = ", ".join(visible_entries[:8])
        if len(visible_entries) > 8:
            preview += ", ..."
        self._workspace_root_probe_context_block = (
            "<workspace_root_probe>\n"
            "Controller root probe saw existing repository paths in /workspace: "
            f"{preview}\n"
            "Treat this as an existing repository. Do NOT ask for another repo URL "
            "or what to build from scratch. Use these paths to pick one or two "
            "high-signal inspections and identify a small real improvement.\n"
            "</workspace_root_probe>"
        )

    def _workspace_has_populated_repo_for_turn(self) -> bool:
        """Return True when the current turn is grounded on an existing repo."""
        return (
            bool(self._workspace_git_url_for_turn)
            or self._workspace_root_probe_populated_for_turn
            or (
                self._workspace_tree_authoritative_for_turn
                and self._workspace_tree_file_count_for_turn >= 40
            )
        )

    def _populated_repo_descriptor_for_turn(self) -> str:
        """Human-readable summary of the current populated repo signal."""
        if self._workspace_git_url_for_turn:
            return "a cloned existing repository"
        if self._workspace_root_probe_populated_for_turn:
            return "an existing repository detected at /workspace root"
        if self._workspace_tree_file_count_for_turn >= 40:
            return f"a populated existing repository ({self._workspace_tree_file_count_for_turn} files)"
        return "an existing repository"

    def _response_contradicts_populated_repo(self, text: str) -> bool:
        """Return True when assistant prose falsely treats a populated repo as empty."""
        if not self._workspace_has_populated_repo_for_turn():
            return False
        body = (text or "").strip()
        if not body:
            return False
        return bool(_POPULATED_REPO_FALSE_EMPTY_RE.search(body))

    def _maybe_refuse_populated_repo_ask_user(self, *, tool_input: dict) -> str:
        """Refuse generic 'what should I build?' questions on populated repos."""
        if not self._workspace_has_populated_repo_for_turn():
            return ""
        questions = tool_input.get("questions")
        if not isinstance(questions, list) or not questions:
            return ""

        for question in questions:
            if not isinstance(question, dict):
                continue
            prompt = str(question.get("prompt") or "")
            options = question.get("options") or []
            option_text = "\n".join(str(opt or "") for opt in options)
            combined = f"{prompt}\n{option_text}".strip()
            if combined and _POPULATED_REPO_EMPTY_QUESTION_RE.search(combined):
                return (
                    "Refused: this workspace already looks like "
                    f"{self._populated_repo_descriptor_for_turn()}, so "
                    "do not ask the user what to build from scratch or claim the repo is "
                    "empty. Use the workspace_tree plus one or two high-signal reads "
                    "(README, package manifest, src/, tests/, docs/) to identify a small "
                    "real improvement first. Only ask the user a narrower question if the "
                    "existing repo presents multiple plausible directions after inspection."
                )
        return ""

    def _append_power_followup_nudge(
        self,
        messages: list[Message],
        event: dict[str, object],
        *,
        goal_text: str = "",
    ) -> None:
        """Append a concrete follow-up nudge after a Power activates.

        Late verifier activations help most when the loop immediately
        steers into the next concrete proof step. We keep these nudges
        narrow, one-shot, and checkpoint-specific so they reinforce the
        active specialist without becoming another layer of general
        controller chatter.
        """
        power_id = str(event.get("id") or "")
        checkpoint = str(event.get("checkpoint") or "")
        source = str(event.get("source") or "")
        key = (power_id, checkpoint, source)
        if key in self._power_followup_nudges_seen:
            return

        goal_hint = goal_text.strip()
        body = ""
        if power_id == "test-author" and checkpoint in {"post_write", "verify_failed"}:
            body = (
                "Test Author is active. Stop broad exploration and add or run the "
                "smallest regression that proves the changed behavior. Prefer one "
                "targeted test first, not repeated task_list updates or broad suite churn. "
                "Prefer test_run over raw shell probing, and do not install tooling "
                "unless the user explicitly asked for environment setup."
            )
        elif power_id == "browser-verification" and checkpoint in {"post_write", "verify_failed"}:
            body = (
                "Browser Verification is active. Do one tight browser-facing check now: "
                "confirm the exact route/flow, gather one concrete runtime signal, then "
                "either fix the issue or report the verified result. Choose exactly one "
                "proof path: one real browser execution attempt, or one static validation "
                "pass if browser execution is unavailable. Avoid repeated shell status "
                "polling or browser-tool discovery without new evidence."
            )
        elif power_id == "failure-triage" and checkpoint in {"pre_plan", "verify_failed"}:
            body = (
                "Failure Triage is active. Treat this as diagnosis work: create or run the "
                "smallest failing reproduction, identify the first meaningful failure signal, "
                "then explain the root cause before widening the fix."
            )

        if not body:
            return

        if goal_hint:
            body += f" Keep the work tightly scoped to: {goal_hint[:220]}."
        messages.append(Message(role="user", content=f"<nudge>{body}</nudge>"))
        self._power_followup_nudges_seen.add(key)

    async def _maybe_request_pre_finish_review(
        self,
        *,
        request,
        messages: list,
        total_writes: int,
        latest_input: str,
        user_goal: str,
    ) -> bool:
        """Inject one final review iteration before termination when useful."""
        if total_writes <= 0:
            return False
        if "pre_finish" in self._controller_power_windows_fired:
            return False
        changed = bool(self._controller_edited_paths)
        activated = await self._maybe_activate_controller_power(
            "pre_finish",
            latest_user_text=user_goal or latest_input,
            edited_paths=list(self._controller_edited_paths),
        )
        if not activated:
            return False
        display_name = (self._controller_power_summary or {}).get("display_name", "final review")
        closeout_contract = self._render_power_closeout_contract()
        closeout_tail = (
            f"\nUse this exact close-out shape:\n{closeout_contract}"
            if closeout_contract else ""
        )
        messages.append(Message(
            role="user",
            content=(
                "<nudge>"
                f"Before finishing, do one brief pass with {display_name}. "
                "Review the changed files, run the most relevant validation if needed, "
                "and then either finish cleanly or explain what still blocks release. "
                f"{'Changes landed this turn; treat this as a real quality gate.' if changed else ''}"
                f"{closeout_tail}"
                "</nudge>"
            ),
        ))
        return True

    def _reset_for_new_request(self, *, preserve_objective: bool = False) -> None:
        """Clear per-request state so prior turns don't leak into this one.

        Handlers are cached by (user_id, session_id) across requests, so
        the same ``CoderState`` instance serves every user message in a
        session. Without this reset, a prior request's task list, plan,
        and recent blockers would render in the sticky reminder of the
        next request — misleading the model into continuing the old
        goal. Session-long invariants (files_read for the read-before-
        edit guard, cumulative tool_calls_made) are preserved; only
        per-request scratchpads reset.
        """
        if not preserve_objective:
            self._state.plan = ""
            self._state.plan_steps = []
            self._state.current_step = 0
            self._state.step_outputs = {}
            self._state.mission = []
            self._state.tasks = []
            self._state.clear_pending_objective_contract()
        # Phase 2 / PR-2.2: clear in-place so the loops.ObservationLedger
        # sharing this list reference stays in sync (reassignment to a
        # new list would orphan the ledger's reference).
        self._state.recent_validation_errors.clear()
        # Phase 2.2: cross-turn failure ledger. Don't wipe — recurring
        # failures should persist so the model sees the pattern across
        # turns, not just within one. Stale entries (>30 min since
        # last_at) age out automatically here; recurring ones keep
        # refreshing themselves via record_tool_failure's dedupe path.
        self._state.prune_stale_tool_failures()
        self._state.recent_tool_calls.clear()
        self._state.consecutive_failures = 0
        self._state.error = None
        # Clear finish_task signal so a prior turn's completion doesn't
        # short-circuit the new request's very first iteration.
        self._state.finish_requested = False
        # Clear model-initiated compaction state for the same reason.
        self._state.compact_requested = False
        self._state.compact_note = ""
        self._state.compact_tool_uses = 0
        self._state.finish_summary = ""
        # Reset turn-intent classification so this request's priming
        # tree picks up the new user message. Re-set below at the
        # classify_turn_intent site.
        self._state.current_intent = None
        self._active_power_context_block = ""
        self._active_power_summary = None
        self._controller_power_context_block = ""
        self._controller_power_summary = None
        self._controller_power_windows_fired = set()
        self._auto_explore_context_block = ""
        self._auto_explore_fired = False
        self._controller_edited_paths = []
        self._pending_power_activation_events = []
        self._power_followup_nudges_seen = set()
        # Reviewable-turn flow: fresh turn_id + fresh snapshot. The
        # mutating tools read ``active_turn_snapshot`` via the state
        # object, so setting it here makes the capture path light up
        # automatically on this turn's first write. Handler MUST have
        # a container_manager for the snapshot to function; tests that
        # build a handler with container_manager=None will skip the
        # snapshot but all other tool behaviour still works (the
        # snapshot_before_write helper bails cleanly on None).
        import uuid

        from augmentum.coder.turn_snapshot import TurnSnapshot
        self._state.active_turn_id = uuid.uuid4().hex[:12]
        # Reset per-turn auto-observation budget so the cap applies
        # afresh each turn rather than accumulating across the session.
        self._state._auto_observe_used_this_turn = 0
        if self._container_manager is not None:
            self._state.active_turn_snapshot = TurnSnapshot(
                turn_id=self._state.active_turn_id,
                workspace_id=self._workspace_id,
                container_manager=self._container_manager,
            )
        else:
            self._state.active_turn_snapshot = None
        # Phase resets to WAITING; the loop re-enters PLANNING / EXECUTING
        # naturally. Leave files_read + working_set + tool_calls_made
        # alone — they're session-level, not request-level. Also leave
        # turn_summaries alone — that IS the cross-turn memory; wiping
        # it would defeat the 2026-04-20 persistence fix.
        self._state.phase = CoderPhase.WAITING

    async def _start_turn_ledger(self, request: InternalChatRequest) -> None:
        """Start a persisted run ledger for this Coder turn, best-effort."""
        if self._state_manager is None:
            return
        try:
            conn = getattr(getattr(self._state_manager, "backend", None), "conn", None)
            if conn is None:
                return
            from augmentum.coder.ledger import CoderTurnLedger, CoderTurnLedgerStore

            tooling_profile = ""
            if self._container_manager is not None and self._workspace_id:
                try:
                    info = await self._container_manager._get_workspace(self._workspace_id)
                    tooling_profile = getattr(info, "tooling_profile", "") or ""
                except Exception:
                    tooling_profile = ""
            provider = self._backend.__class__.__name__ if self._backend is not None else ""
            # ``prompt_profile`` carries the names + versions of the
            # prompts actually in play this turn (workspace guide + plan
            # + act / native / mission act depending on strategy). Lets
            # retrospective analysis group runs by prompt version when a
            # prose edit regresses a metric — no schema migration needed,
            # the field already exists on ``coder_turn_runs``.
            from augmentum.coder.prompts import prompt_profile_for_strategy as _pprofile
            _strategy = self._resolve_strategy()
            self._turn_ledger = await CoderTurnLedger.start(
                CoderTurnLedgerStore(conn),
                user_id=self._user_id,
                workspace_id=self._workspace_id,
                session_id=self._session_id,
                model=request.model,
                provider=provider,
                strategy=_strategy,
                prompt_profile=_pprofile(_strategy),
                tooling_profile=tooling_profile,
            )
            log.info(
                "coder.turn_run_started",
                run_id=self._turn_ledger.run_id,
                workspace_id=self._workspace_id,
                strategy=self._turn_ledger.strategy,
                model=request.model,
            )
        except Exception:
            self._turn_ledger = None
            log.debug("coder.turn_ledger_start_failed", exc_info=True)

    async def _record_turn_ledger_chunk(self, chunk: InternalStreamChunk) -> None:
        ledger = self._turn_ledger
        if ledger is None:
            return
        try:
            if chunk.augmentum is None:
                chunk.augmentum = {}
            if isinstance(chunk.augmentum, dict):
                chunk.augmentum.setdefault("run_id", ledger.run_id)
            await ledger.observe_chunk(chunk)
        except Exception:
            log.debug("coder.turn_ledger_record_failed", exc_info=True)

    def _get_subagent_dispatcher(self):
        """Lazy-build the per-handler SubagentDispatcher.

        Returns ``None`` when ``settings.coder_subagents_enabled`` is
        False, or when the lifespan didn't wire the agent registry /
        provider registry. The first successful resolution memoizes so
        every phase_act/phase_plan call gets the same instance.

        Tool-list provider is a thunk that re-materializes the live
        tool registry on every spawn — so role files that opt into
        narrower / wider subsets see the current registry state, not a
        snapshot taken at construction time.
        """
        if self._subagent_dispatcher is not None:
            return self._subagent_dispatcher

        from augmentum.config import settings as _settings
        if not getattr(_settings, "coder_subagents_enabled", False):
            return None
        if self._provider_registry is None:
            return None

        # Reach the app state via the state_manager → backend chain.
        # The lifespan attaches ``coder_agent_registry`` +
        # ``coder_subagent_store`` so we read them off the app state.
        registry = None
        store = None
        try:
            from augmentum.proxy import server as _server
            app = getattr(_server, "app", None) or getattr(_server, "_app", None)
            if app is None:
                # Tests may construct the handler without an app — fall
                # through to constructing a minimal registry inline.
                pass
            else:
                registry = getattr(app.state, "coder_agent_registry", None)
                store = getattr(app.state, "coder_subagent_store", None)
        except Exception:
            registry = None
            store = None

        if registry is None:
            from augmentum.agents.presets import BUILTIN_ROLES
            from augmentum.agents.registry import AgentRegistry
            registry = AgentRegistry(builtins=BUILTIN_ROLES)

        from augmentum.agents.dispatch import SubagentDispatcher

        # Provide live tool registry + coder state via thunks so the
        # dispatcher sees current state on every spawn, not stale
        # snapshots taken at construction.
        def _live_tools():
            from augmentum.coder.tools import create_coder_tools
            return create_coder_tools(
                self._container_manager,
                self._workspace_id,
                self._state,
                executor=getattr(self, "_executor", None),
                tool_registry=self._tool_registry,
                profile_store=self._profile_store,
                service_store=self._service_store,
                user_id=self._user_id,
                # Important: subagent_dispatcher=None inside the thunk to
                # prevent recursion (a subagent's tool list shouldn't
                # include task_dispatch by default; role files opt in
                # via permissions.can_spawn_subagents).
                subagent_dispatcher=None,
                # Subagents share the parent's archive — same workspace,
                # same user. Recall over past turns helps an explore /
                # research subagent see prior work in the same scope.
                db_conn=self._resolve_archive_conn(),
                # Don't expose bug_finder dispatch to subagents — the
                # full audit is a 5-30 minute heavyweight job that
                # should be initiated by the lead, not by a fan-out
                # subagent.
                jobs_store=None,
            )

        def _live_state():
            return self._state

        self._subagent_dispatcher = SubagentDispatcher(
            registry=registry,
            provider_registry=self._provider_registry,
            store=store,
            tool_registry_provider=_live_tools,
            coder_state_provider=_live_state,
        )
        return self._subagent_dispatcher

    async def _refresh_kernel_facts(
        self, request: InternalChatRequest,
    ) -> None:
        """Turn-start hook: auto-seed objective.md + render facts block.

        Stores the rendered ``<workspace_facts>`` block on
        ``self._cached_facts_block`` so every strategy's prompt
        construction (``_build_messages`` for canonical/hybrid/plan,
        inline read for native) gets the same content without
        re-computing.

        Gated on:
          - ``coder_kernel_v2`` (the kernel itself must be on)
          - ``coder_kernel_inline_facts`` (the in-prompt rendering)
          - kernel instance exists (skipped when container_manager is None)

        Auto-seeding obeys ``coder_kernel_auto_seed_objective`` and
        only fires when the substantive user message (continuation-
        filtered) clears the 30-char length gate. Idempotent — never
        overwrites an existing objective.md.

        Best-effort throughout: any failure clears the cached block
        and logs at debug. A kernel failure must never block a turn;
        the kernel hint still tells the model the files exist.
        """
        self._cached_facts_block = ""
        if self._workspace_kernel is None:
            return

        from augmentum.config import settings as _settings_for_facts

        if not _settings_for_facts.coder_kernel_v2:
            return

        # Auto-seed first, then render — that way the freshly-seeded
        # objective shows up in the same turn it was created (not
        # one turn later).
        if _settings_for_facts.coder_kernel_auto_seed_objective:
            try:
                _, substantive_goal = _extract_goal_split(request.messages)
                seed_text = (substantive_goal or "").strip()
                # Visible diagnostic — without this, every silent-skip
                # case (no substantive ask, file already present, length
                # gate, container write failure) looked identical to
                # "auto-seed working fine". The inspector panel shows
                # "No objective pinned yet" regardless, so users can't
                # tell whether the issue is on the seed side or the
                # render side. Logged at INFO so it shows up in normal
                # container logs without bumping levels.
                seeded = False
                skip_reason = ""
                if not seed_text:
                    skip_reason = "no_substantive_goal"
                elif len(seed_text) < 12:
                    skip_reason = f"text_below_min_chars (len={len(seed_text)})"
                else:
                    seeded = await self._workspace_kernel.seed_objective_if_missing(
                        seed_text,
                    )
                    if not seeded:
                        skip_reason = "file_already_present_or_seed_failed"
                log.info(
                    "coder.objective_auto_seed",
                    workspace_id=self._workspace_id,
                    seeded=bool(seeded),
                    seed_text_len=len(seed_text),
                    skip_reason=skip_reason,
                )
            except Exception:
                log.warning(
                    "coder.kernel_facts_seed_failed",
                    workspace_id=self._workspace_id,
                    exc_info=True,
                )

        if not _settings_for_facts.coder_kernel_inline_facts:
            return

        try:
            self._cached_facts_block = await self._workspace_kernel.render_facts_block(
                budget_chars=600,
            )
        except Exception:
            log.debug(
                "kernel_facts_render_failed",
                workspace_id=self._workspace_id,
                exc_info=True,
            )
            self._cached_facts_block = ""

        # Bridge the rendered facts onto the live CoderState so the
        # SubagentDispatcher's context bridge can read them when spawning
        # task_dispatch children. The dispatcher reads
        # state.kernel_facts_text / state.orientation_text via
        # context_bridge.extract_*; without this assignment every subagent
        # launches context-blind (the objective never crosses the boundary).
        # Best-effort and turn-transient — never serialized.
        state = getattr(self, "_state", None)
        if state is not None:
            try:
                state.kernel_facts_text = self._cached_facts_block
                state.orientation_text = (
                    await self._workspace_kernel.render_orientation(budget_chars=240)
                )
            except Exception:
                log.debug(
                    "kernel_orientation_render_failed",
                    workspace_id=self._workspace_id,
                    exc_info=True,
                )

    async def _render_maker_agreements_block(self) -> str:
        """Render this user's accrued Working Agreements for the system prompt.

        Durable, model-agnostic "how this maker wants to be worked with"
        principles (mig 273) — injected once per turn so any local model
        inherits the accrued relationship. Returns "" when disabled, when
        there's no DB conn / user, or when the user hasn't accrued any
        (the common first-run case), so injecting it is a pure no-op until
        the relationship has actually grown something. Never raises — a
        prompt-enrichment hook must not break a turn.
        """
        try:
            from augmentum.config import settings as _settings
            if not getattr(_settings, "coder_maker_agreements_enabled", True):
                return ""
            conn = self._resolve_archive_conn()
            user_id = getattr(self, "_user_id", "") or ""
            if conn is None or not user_id:
                return ""
            from augmentum.coder.maker_agreements import MakerAgreements
            return await MakerAgreements(conn).render_for_prompt(user_id=user_id)
        except Exception:
            log.warning("maker_agreements_render_failed", exc_info=True)
            return ""

    async def _finish_turn_ledger(self, *, status: str) -> None:
        ledger = self._turn_ledger
        if ledger is None:
            return
        self._turn_ledger = None
        try:
            # Priming tree (Sprint 1): forward per-branch token telemetry
            # captured by _build_act_system into the ledger so it lands
            # in coder_turn_runs.priming_telemetry for later analysis.
            await ledger.finish(
                status=status,
                priming_telemetry=self._state.last_priming_telemetry or None,
            )
            log.info("coder.turn_run_finished", run_id=ledger.run_id, status=status)
        except Exception:
            # warning, not debug: a swallowed finish leaves a live-process
            # zombie (status='running' until the next restart's sweep).
            log.warning("coder.turn_ledger_finish_failed", exc_info=True)

    # Cap on screenshot bytes fed to the model — bigger than this and the
    # base64 payload starts crowding out the conversation itself.
    _SCREENSHOT_IMAGE_MAX_BYTES = 3_000_000

    async def _model_supports_vision(self, model: str) -> bool:
        """Whether the CURRENT backend+model can consume image parts.

        Reads the backend's own capability report (list_models →
        ModelInfo.vision — the field the game-agent fix made honest).
        Cached per (handler, model): one probe per turn's first
        screenshot, not per screenshot.
        """
        cache = getattr(self, "_vision_capable_cache", None)
        if isinstance(cache, tuple) and cache[0] == model:
            return cache[1]
        ok = False
        try:
            list_models = getattr(self._backend, "list_models", None)
            if callable(list_models):
                for mi in await list_models() or []:
                    name = getattr(mi, "name", "") or getattr(mi, "model", "")
                    if name == model or getattr(mi, "model", "") == model:
                        ok = bool(getattr(mi, "vision", False))
                        break
        except Exception:
            log.debug("coder_vision_probe_failed", exc_info=True)
        self._vision_capable_cache = (model, ok)
        return ok

    async def _maybe_screenshot_image_message(self, tool_result, model: str):
        """Build the follow-up user message carrying the screenshot pixels.

        Returns None (never raises) when: no workspace file path in the
        result, model isn't vision-capable, file too large/missing —
        the text-only flow is unchanged in every one of those cases.
        """
        try:
            if self._container_manager is None:
                return None
            meta = tool_result.metadata or {}
            browser = meta.get("browser") or {}
            path = str(browser.get("path") or "")
            if not path.startswith("/workspace") or ".." in path.split("/"):
                return None
            data = await self._container_manager.file_download(
                self._workspace_id, path,
            )
            if not data or len(data) > self._SCREENSHOT_IMAGE_MAX_BYTES:
                return None
            from augmentum.models.base import Message
            if await self._model_supports_vision(model):
                import base64
                uri = "data:image/png;base64," + base64.b64encode(data).decode()
                # Comparison discipline: the previous screenshot's pixels
                # are already in message history (we put them there), so
                # a repeat capture just needs the instruction to diff —
                # design iteration converges on compare, not re-glance.
                self._screenshot_pixel_feeds += 1
                if self._screenshot_pixel_feeds > 1:
                    note = (
                        f"[attached: the browser_screenshot you just "
                        f"captured ({path}). Compare it against the "
                        f"previous screenshot earlier in this conversation: "
                        f"what changed, did it improve, and what visual "
                        f"defects remain? Don't conclude the page is "
                        f"correct without naming what you checked.]"
                    )
                else:
                    note = (
                        f"[attached: the browser_screenshot you just "
                        f"captured ({path}). Inspect it visually before "
                        f"concluding the page is correct.]"
                    )
                return Message(
                    role="user",
                    content=[
                        {"type": "text", "text": note},
                        {"type": "image_url", "image_url": {"url": uri}},
                    ],
                )
            # Non-vision coder model: caption via the substrate vision
            # tier (Slot C classifier+mmproj → SmolVLM CPU fallback) with
            # a coder-specific prompt, and feed the TEXT back. Same
            # tiering doctrine as vision/router.py's matrix — the coder
            # model never needs to be vision-capable to stop being blind.
            router = self._vision_router
            if router is None:
                return None
            caption = await router.caption(
                data,
                prompt=(
                    "This is a screenshot of a web page a coding agent just "
                    "built or modified. Describe it for that agent: overall "
                    "layout and structure, visible headings/buttons/text "
                    "(quote exact strings), colors/theme, and ANY visual "
                    "defects — blank regions, overlapping or clipped "
                    "elements, misalignment, unstyled HTML, error messages "
                    "or stack traces on the page. Be concrete and terse."
                ),
                max_tokens=256,
                timeout_s=30.0,
            )
            caption = (caption or "").strip()
            if not caption:
                return None
            # Comparison discipline for the caption path: replay the
            # PREVIOUS screenshot's caption (cached text — no second
            # vision call, no fabricated diff) so the agent can compare
            # iterations itself instead of judging each shot in
            # isolation.
            prev = self._last_screenshot_note
            self._last_screenshot_note = (path, caption)
            body = (
                f"[browser_screenshot {path} — described by the local "
                f"vision model, since your model can't view images "
                f"directly]\n{caption}"
            )
            if prev is not None:
                body += (
                    f"\n\n[for comparison, the previous screenshot "
                    f"({prev[0]}) was described as:]\n{prev[1]}\n"
                    f"[compare the two descriptions: what changed, did it "
                    f"improve, and what visual defects remain?]"
                )
            return Message(role="user", content=body)
        except Exception:
            log.warning("coder_screenshot_vision_feed_failed", exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Interruption summaries (cancellation / mid-turn error)
    #
    # The normal end-of-turn summary in phase_act.py (~line 3028) only
    # fires when the act loop reaches its natural end. CancelledError
    # and unhandled exceptions abort the generator before that code is
    # ever reached, leaving a silent gap in ``turn_summaries`` while
    # the partial assistant message persists in the chat tree. The
    # next turn's ``<prior_turns>`` block then says "nothing
    # happened" — but the model still reads the trailing partial
    # message and tries to continue it.
    #
    # ``_record_interruption_summary`` writes a turn_summary from the
    # CancelledError / Exception handler in ``_run_agent_with_ledger``
    # so the next turn sees a coherent "previous turn was interrupted"
    # stanza with the reason and roughly where the agent was when it
    # stopped.
    # ------------------------------------------------------------------

    def _resolve_cancel_reason(self) -> str:
        """Read the cancel reason set by ``broker.cancel(reason=...)``.

        Returns ``"user_cancel"`` when no broker entry is found (e.g.
        legacy in-request path, ledger never started) so the renderer
        always has a non-empty reason to display.
        """
        broker = self._coder_run_broker
        run_id = self._turn_ledger.run_id if self._turn_ledger is not None else ""
        if broker is None or not run_id:
            return "user_cancel"
        try:
            entry = broker.get(run_id)
        except Exception:
            return "user_cancel"
        if entry is None:
            return "user_cancel"
        return (entry.cancel_reason or "").strip() or "user_cancel"

    def _classify_runtime_error(self, exc: BaseException) -> str:
        """Bucket an unhandled exception into a reason the model can read.

        The categories match the labels in ``_render_cancelled_stanza``
        so the model gets a consistent vocabulary across cancel and
        error paths. Conservative — anything we can't confidently
        classify becomes ``"backend_error"`` rather than guessing.
        """
        name = exc.__class__.__name__.lower()
        msg = str(exc).lower()
        if "timeout" in name or "timeout" in msg or "timed out" in msg:
            return "timeout"
        if (
            "ratelimit" in name
            or "rate_limit" in msg
            or "rate limit" in msg
            or "429" in msg
        ):
            return "rate_limit"
        if (
            "connectionerror" in name
            or "connectreset" in name
            or "connection" in msg
            or "network" in msg
            or "disconnect" in msg
        ):
            return "network"
        return "backend_error"

    def _record_interruption_summary(
        self,
        request: InternalChatRequest,
        *,
        reason: str,
        kind: str = "cancelled",
        error_text: str = "",
    ) -> None:
        """Append a turn_summary describing how the turn was interrupted.

        ``kind`` is ``"cancelled"`` (user-driven Stop / slash command /
        shutdown) or ``"errored"`` (backend timeout / rate-limit /
        network failure). The schema is a superset of the normal
        ``_build_turn_summary`` shape — the renderer dispatches on
        ``s.get("cancelled")`` / ``s.get("errored")`` to swap stanzas.

        Best-effort: any exception inside this method is swallowed
        with a debug log so we never mask the original cancellation
        / error during cleanup.
        """
        import time as _time

        try:
            user_goal = ""
            for msg in reversed(request.messages):
                if msg.role == "user":
                    # De-wrap the UI envelope — interruption summaries land in
                    # the same ring + archive as normal ones, so the raw
                    # [Terminal context] buffer would pollute recall/prior_turns
                    # exactly the same way (audit 2026-06-25). Clean at this seam
                    # too, not just _extract_goal_split.
                    user_goal = _clean_user_text(
                        (msg.content or "").strip(), single_line=False,
                    )
                    break

            # Last tool the model actually ran this turn. Populated
            # opportunistically by record_tool_call from inside the act
            # loop; missing for turns cancelled before any tool fired.
            last_tool: dict = {}
            calls = self._state.recent_tool_calls or []
            if calls:
                entry = calls[-1]
                last_tool = {
                    "tool":      (entry.get("tool") or "")[:40],
                    "target":    (entry.get("key") or "")[:160],
                    "iteration": int(entry.get("last_iter") or 0),
                }
            active = self._state.active_task() or {}
            interrupted_at = {
                "phase":            self._state.phase.value,
                "tool_calls_made":  int(self._state.tool_calls_made or 0),
                "current_step":     int(self._state.current_step or 0),
                "total_steps":      int(self._state.total_steps or 0),
                "last_tool":        last_tool,
                "active_task":      (
                    {"content": (active.get("content") or "")[:200]}
                    if active else {}
                ),
            }

            # working_set is session-cumulative so this overstates the
            # files read THIS turn — but it's a strict superset and
            # the model can cross-reference against earlier prior_turns
            # to disambiguate. Cap to keep the prompt cheap.
            recent_reads = list(self._state.working_set)[:12]

            summary = {
                "turn_idx":    len(self._state.turn_summaries) + 1,
                # Stamp the originating turn id so the rewind endpoint can
                # pop only the summary belonging to the run being rewound.
                # Older persisted summaries lack this field; rewind falls
                # back to a "last entry" heuristic when it's missing.
                "turn_id":     getattr(self._state, "active_turn_id", "") or "",
                "user_goal":   user_goal[:300],
                "files_read":  recent_reads,
                "files_edited": [],
                "edits":        [],
                "shell_commands": [],
                "outcome":     "cancelled" if kind == "cancelled" else "errored",
                "blockers":    (error_text or "")[:200],
                "cancelled":   kind == "cancelled",
                "errored":     kind == "errored",
                "cancel_reason": reason or ("user_cancel" if kind == "cancelled" else "backend_error"),
                "interrupted_at": interrupted_at,
                "created_at":  _time.time(),
            }
            self._state.add_turn_summary(summary)
            # Fire-and-forget archive write — this helper is sync but
            # called from async-generator context, so we can schedule
            # the awaitable on the running loop without blocking.
            # ``get_running_loop`` raises if no loop is active; we
            # swallow that case because interruptions during teardown
            # are valid (the archive write just won't happen).
            try:
                import asyncio as _asyncio
                _loop = _asyncio.get_running_loop()
                # Stash the Task so it isn't GC'd mid-flight. The
                # callback removes it after completion. Same pattern
                # async-libs recommend instead of bare create_task.
                _task = _loop.create_task(self._archive_turn_summary(summary))
                self._interruption_archive_tasks.add(_task)
                _task.add_done_callback(
                    self._interruption_archive_tasks.discard,
                )
            except RuntimeError:
                # No running loop — interrupted during shutdown. Skip.
                log.debug("coder.interruption_summary_archive_no_loop")
            except Exception:
                log.debug("coder.interruption_summary_archive_failed", exc_info=True)
        except Exception:
            log.debug("coder.interruption_summary_failed", exc_info=True)

    async def _handle_stream(
        self, request: InternalChatRequest,
    ) -> AsyncIterator[InternalStreamChunk]:
        # New user turn → purge per-request scratchpads so prior goals
        # don't haunt the sticky reminder. Session-level state (files
        # already read, cumulative tool count) is preserved.
        await self._restore_state()

        user_msg = ""
        for msg in reversed(request.messages):
            if msg.role == "user":
                user_msg = msg.content.strip()
                break
        has_prior_state = _has_resumable_objective_state(self._state)
        self._reset_for_new_request(
            preserve_objective=(
                _is_conversational_greeting(user_msg)
                or (
                    _is_continuation_request(user_msg)
                    and has_prior_state
                )
            ),
        )

        # Refresh per-workspace safeguards toggle from the DB at turn
        # start so user flips via PUT /api/coder/workspaces/{id}/safeguards
        # take effect on the next turn without a process restart. Best-
        # effort: any failure leaves the prior value (default True) intact.
        if self._container_manager is not None and self._workspace_id:
            try:
                info = await self._container_manager._get_workspace(
                    self._workspace_id,
                )
                self._state.safeguards_enabled = bool(info.safeguards_enabled)
                # Refresh planning_mode every turn so a Shift+Tab cycle
                # the user just hit takes effect immediately without a
                # full session restart. Stored on the handler instance
                # (not CoderState) because it's a per-handler operating
                # mode, not part of the persisted agent state.
                mode = (getattr(info, "planning_mode", "") or "auto").strip().lower()
                if mode not in ("default", "plan", "auto"):
                    mode = "auto"
                self._planning_mode = mode
            except Exception as exc:
                log.debug("coder_safeguards_refresh_failed", workspace_id=self._workspace_id, error=str(exc))
                self._planning_mode = "auto"
        else:
            self._planning_mode = "auto"

        await self._start_turn_ledger(request)

        # Branch: detached (background) vs in-request (legacy). The
        # broker path lets the client disconnect without killing the
        # run; the UI reattaches via GET /api/coder/runs/{id}/stream.
        # The legacy path is preserved behind the feature flag so we
        # can flip back if regressions surface.
        from augmentum.config import settings as _settings
        run_id = self._turn_ledger.run_id if self._turn_ledger is not None else ""
        use_broker = (
            bool(_settings.coder_background_runs)
            and self._coder_run_broker is not None
            and bool(run_id)
        )

        if not use_broker:
            async for chunk in self._run_agent_with_ledger(request):
                yield chunk
            return

        broker = self._coder_run_broker
        # Wrap the existing agent iterator so the broker pumps it in a
        # detached task. ``self`` captured by closure keeps the handler
        # alive past the request lifetime — single-use, freed when the
        # task ends.
        handler = self

        async def _agent_factory(_entry):
            async for chunk in handler._run_agent_with_ledger(request):
                yield chunk

        await broker.start_run(
            run_id=run_id,
            user_id=self._user_id,
            workspace_id=self._workspace_id,
            agent=_agent_factory,
        )

        # Hand the freshly-built TurnSnapshot to the broker so the
        # /rewind route can restore files without poking into this
        # detached handler instance. _reset_for_new_request already
        # populated active_turn_snapshot earlier in this method;
        # tests that build a handler with container_manager=None will
        # have a None snapshot and this is a no-op.
        snapshot = getattr(self._state, "active_turn_snapshot", None)
        if snapshot is not None:
            broker.attach_snapshot(run_id, snapshot)

        # Subscribe and stream back. Client disconnect raises
        # CancelledError here — we let it propagate so FastAPI
        # closes the response, but the broker task keeps running.
        # The UI calls POST /api/coder/runs/{id}/cancel to actually
        # stop a run; aborting the fetch is no longer enough.
        try:
            async for buffered in broker.subscribe(run_id, since_seq=0):
                yield buffered.chunk
        except asyncio.CancelledError:
            log.info(
                "coder.request_disconnected_run_continues",
                run_id=run_id,
                workspace_id=self._workspace_id,
            )
            raise

    async def _run_agent_with_ledger(
        self, request: InternalChatRequest,
    ) -> AsyncIterator[InternalStreamChunk]:
        """Drive ``_handle_stream_body`` with ledger recording + cancel cleanup.

        Extracted from the old ``_handle_stream`` so the broker path
        can hand the same generator to its detached task. CancelledError
        semantics are preserved: it still means "tear down execs", just
        triggered now by ``broker.cancel(run_id)`` instead of an HTTP
        client disconnect.

        Both CancelledError (user / slash command / shutdown) and
        unhandled Exception (timeout / rate-limit / network) write an
        interruption entry to ``turn_summaries`` before propagating.
        Without that entry, the next turn's ``<prior_turns>`` block
        has a silent gap where the cancelled work *should* be — the
        model then reads the leftover partial assistant message in
        history and tries to continue it.
        """
        ledger_status = "error"
        # Visible assistant text, accumulated so the reply can be persisted
        # server-side at completion. Without this, the saved conversation
        # only ever gets the assistant message when an attached CLIENT
        # renders it and POSTs /api/coder/conversation — a user who left
        # the coder surface mid-run (mobile screen off, mode switch) came
        # back to an unanswered prompt even though the run completed fine
        # (the broker detach worked; the result just had no path home).
        _visible_parts: list[str] = []
        try:
            async for chunk in self._handle_stream_body(request):
                await self._record_turn_ledger_chunk(chunk)
                if chunk.content_delta:
                    _visible_parts.append(chunk.content_delta)
                yield chunk
            # Natural completion — drain any cooperative "queue" mode
            # interjections that landed during the turn. The frontend
            # auto-chains them as a new /chat request so each drained
            # entry becomes its own turn with its own ledger row +
            # rewind snapshot. NOT done on the cancel/error paths
            # below because queued follow-ups from an interrupted turn
            # would land in a state the user doesn't expect (they were
            # waiting to apply AFTER the work they cancelled).
            _queued = self._coop_drain_queued_followups()
            if _queued:
                yield self._meta_chunk(
                    phase="completing", status="queue_followup",
                    model=request.model,
                    extra={
                        "messages": [
                            {
                                "id": m.get("id", ""),
                                "content": m.get("content", ""),
                                "attachments": m.get("attachments", []),
                                "queued_at": m.get("queued_at", 0.0),
                                "delivered_at": m.get("delivered_at", 0.0),
                            }
                            for m in _queued
                        ],
                    },
                )
            ledger_status = "completed"
            # Persist the reply if no client was attached to render+save it
            # (run survived a disconnect — the result must too). Best-effort:
            # a persistence hiccup shouldn't fail a completed run.
            try:
                await self._persist_reply_if_orphaned(
                    request, "".join(_visible_parts),
                )
            except Exception:
                log.warning(
                    "coder.orphaned_reply_persist_failed",
                    workspace_id=self._workspace_id,
                    exc_info=True,
                )
        except asyncio.CancelledError:
            ledger_status = "cancelled"
            reason = self._resolve_cancel_reason()
            self._record_interruption_summary(
                request, reason=reason, kind="cancelled",
            )
            # Drain queued/steered messages out of the inbox so the
            # UI can flip their badges to "dropped" and the user knows
            # to retype. Without this, queued messages would stay in
            # "queued" state forever on a cancelled turn — the same
            # symptom class as the validator bug fixed earlier today.
            # Yield BEFORE re-raising so the chunk lands in the broker
            # buffer for the subscriber to see.
            try:
                _dropped = self._coop_drain_dropped_inbox()
                if _dropped:
                    yield self._meta_chunk(
                        phase="executing", status="queue_dropped",
                        model=request.model,
                        extra={
                            "reason": reason or "user_cancel",
                            "dropped_msg_ids": [
                                m.get("id", "") for m in _dropped
                            ],
                        },
                    )
            except Exception as drop_exc:
                # Defensive: don't let inbox cleanup mask the original
                # cancellation. Logged so operators can see if the
                # drain itself is misbehaving.
                log.warning(
                    "coder.cancel_inbox_drain_failed",
                    error=str(drop_exc),
                )
            raise
        except Exception as exc:
            # Backend timeout, rate limit, network blip — anything the
            # ledger would otherwise just tag ``status='error'`` with
            # nothing in the cross-turn memory. Classify into the same
            # reason vocabulary the cancel renderer uses so the model
            # sees a consistent shape on the next turn.
            kind = self._classify_runtime_error(exc)
            self._record_interruption_summary(
                request, reason=kind, kind="errored", error_text=str(exc),
            )
            # Stamp the WHY onto the run row. Before this, every errored
            # run landed as bare status='error' with finish_reason empty —
            # 74 error rows in one week, zero diagnosable causes once the
            # container logs rotated (the terminal event is never recorded
            # either; the exception kills the stream before it can be).
            if self._turn_ledger is not None:
                if not self._turn_ledger.finish_reason:
                    self._turn_ledger.finish_reason = kind or "exception"
                if not self._turn_ledger.fallback_reason:
                    self._turn_ledger.fallback_reason = str(exc)[:300]
            # Same inbox drain as the CancelledError path — drop any
            # queued/steered messages and flip their UI badges to
            # "dropped" so the user can retype rather than wait
            # forever for an answer that won't come.
            try:
                _dropped = self._coop_drain_dropped_inbox()
                if _dropped:
                    yield self._meta_chunk(
                        phase="executing", status="queue_dropped",
                        model=request.model,
                        extra={
                            "reason": kind or "backend_error",
                            "dropped_msg_ids": [
                                m.get("id", "") for m in _dropped
                            ],
                        },
                    )
            except Exception as drop_exc:
                log.warning(
                    "coder.error_inbox_drain_failed",
                    error=str(drop_exc),
                )
            raise
        finally:
            await self._finish_turn_ledger(status=ledger_status)
            if self._container_manager is not None:
                try:
                    killed = await self._container_manager.cancel_workspace_execs(
                        self._workspace_id,
                    )
                    if killed:
                        log.info(
                            "coder.cancel_killed_execs",
                            workspace=self._workspace_id, count=killed,
                        )
                except Exception:
                    log.warning(
                        "coder.cancel_cleanup_failed",
                        exc_info=True,
                    )

    async def _handle_stream_body(
        self, request: InternalChatRequest,
    ) -> AsyncIterator[InternalStreamChunk]:
        """Core routing — see ``_handle_stream`` for the wrapping
        cancel-cleanup ``finally``. Separated so the cleanup wrap
        has one single reliable exit point."""

        # Phase 1 fallback: no container, just pass through
        if self._container_manager is None:
            async for chunk in self._passthrough_stream(request):
                yield chunk
            return

        await self._refresh_coder_token_budgets(request.model)

        # Check for vague requests BEFORE hitting the model
        user_msg = ""
        for msg in reversed(request.messages):
            if msg.role == "user":
                user_msg = msg.content.strip()
                break

        yield self._token_budget_chunk(
            request.messages,
            scope="incoming_request",
            phase="planning",
            model=request.model,
        )

        if _is_compact_command(user_msg):
            async for chunk in self._handle_manual_compact_command(request):
                yield chunk
            return

        # Conversational short-circuit. "hey there", "thanks", "ok cool" —
        # small-talk that shouldn't enter the plan→act machinery. Route
        # through a thin conversational passthrough with a system hint
        # so the model responds briefly and invites the real task.
        # Added 2026-04-20 after a "hey there" greeting triggered a full
        # plan phase and a 40+ tool-call thrash on an empty workspace.
        if _is_conversational_greeting(user_msg):
            yield self._meta_chunk(
                phase="conversational", status="short_circuit",
                model=request.model,
            )
            async for chunk in self._conversational_stream(request):
                yield chunk
            return

        if _is_vague_request(user_msg):
            question = _generate_clarification(user_msg)
            yield emit(
                question,
                phase="planning", status="question",
                model=request.model,
            )
            return

        await self._capture_turn_workspace_baseline()
        self._workspace_tree_authoritative_for_turn = False
        self._workspace_tree_file_count_for_turn = 0
        self._workspace_git_url_for_turn = ""
        self._workspace_root_probe_populated_for_turn = False
        self._workspace_root_probe_context_block = ""
        turn_context = await build_turn_context(handler=self, request=request)
        self._turn_intent_for_turn = _classify_turn_intent(
            turn_context.latest_input,
            goal_text=turn_context.user_goal or turn_context.latest_input,
        )
        # Mirror onto state so downstream prompt builders (exemplar
        # loader, tool shortlist, sticky-reminder trim) can read intent
        # without re-classifying or having access to the handler attr.
        self._state.current_intent = self._turn_intent_for_turn
        # Orchestrator dispatch: seed the turn's mission from the inbound
        # success_criteria so the plan/act loop and the P3 verifier gate have
        # a real contract to work against. Only when the orchestrator supplied
        # criteria AND the state has no mission yet (never clobber an in-flight
        # cross-turn mission). No-op on the direct-user path.
        if (
            self._dispatch is not None
            and self._dispatch.success_criteria
            and not self._state.mission
        ):
            self._state.mission = list(self._dispatch.success_criteria)
        self._turn_tier_for_turn = classify_tier(
            latest_text=turn_context.latest_input,
            goal_text=turn_context.user_goal or "",
            workspace_file_count=(
                self._workspace_tree_file_count_for_turn
                if self._workspace_tree_file_count_for_turn > 0
                else None
            ),
        )
        # Eval-visible tier metadata. Emitted once per turn right after
        # classification so test harnesses + UI traces can read which
        # execution path the handler took.
        yield self._meta_chunk(
            phase="planning", status="tier_classified",
            model=request.model,
            extra={
                "tier": self._turn_tier_for_turn.tier.value,
                "tier_reason": self._turn_tier_for_turn.reason,
                "tier_signals": list(self._turn_tier_for_turn.signals),
            },
        )

        # Workspace-kernel v2: refresh .augmentum/ files now that we
        # know the tier. Tier-conditional — REFLEX is a no-op. Best-
        # effort: a kernel failure must never block the turn.
        from augmentum.config import settings as _settings
        if (
            _settings.coder_kernel_v2
            and self._workspace_kernel is not None
        ):
            await self._workspace_kernel.refresh(
                tier=self._turn_tier_for_turn.tier,
            )
            # Refresh-then-cache the rendered facts block so every
            # strategy's prompt construction reads the same content
            # via ``self._cached_facts_block``. Mirrors the
            # ``_cached_guide`` pattern.
            await self._refresh_kernel_facts(request)
        # Code-intel repo map: render from the CURRENT index (fast host-side
        # read, no container exec), then fire a background incremental build
        # so drift from shell/git mutations lands by the NEXT turn. Rendering
        # before building keeps turn latency at zero.
        self._repo_map_context_block = ""
        if (
            getattr(_settings, "coder_code_intel_enabled", True)
            and getattr(_settings, "coder_repo_map_enabled", True)
            and self._container_manager is not None
        ):
            try:
                from augmentum.coder import code_intel as _ci
                self._repo_map_context_block = await _ci.render_repo_map(
                    self._workspace_id,
                    max_chars=int(getattr(_settings, "coder_repo_map_max_chars", 4000)),
                )
                _t = asyncio.create_task(
                    _ci.build_code_intel(self._container_manager, self._workspace_id),
                )
                _t.add_done_callback(lambda t: t.cancelled() or t.exception())
            except Exception:
                log.warning(
                    "coder_repo_map_refresh_failed",
                    workspace_id=self._workspace_id, exc_info=True,
                )
                self._repo_map_context_block = ""
        self._runtime_truth_for_turn = turn_context.runtime_truth
        self._runtime_truth_context_block = (
            turn_context.runtime_truth.render_block()
            if turn_context.runtime_truth is not None
            else ""
        )
        # Fold in NEW console/error events from the user's live preview since
        # last turn (push). Dynamic per-turn content → rides the runtime
        # carrier, not the system prefix, so the KV prefix stays cache-hot.
        self._preview_console_context_block = self._render_preview_console_block()
        self._workspace_tree_authoritative_for_turn = bool(
            turn_context.tree_is_authoritative,
        )
        if self._container_manager is not None:
            try:
                info = await self._container_manager._get_workspace(self._workspace_id)
                self._workspace_git_url_for_turn = str(info.git_url or "")
            except Exception:
                log.debug(
                    "coder_workspace_git_url_lookup_failed",
                    workspace_id=self._workspace_id,
                    exc_info=True,
                )
        await self._probe_workspace_root_for_turn()
        if self._workspace_snapshot is not None:
            if self._workspace_tree_authoritative_for_turn:
                try:
                    await self._workspace_snapshot.refresh_if_stale(force=True)
                except Exception:
                    log.debug(
                        "coder_workspace_snapshot_count_refresh_failed",
                        exc_info=True,
                    )
            self._workspace_tree_file_count_for_turn = len(
                self._workspace_snapshot.current_paths,
            )
        await self._load_active_power_for_turn()
        await self._maybe_activate_controller_power(
            "pre_plan",
            latest_user_text=turn_context.user_goal or turn_context.latest_input,
        )
        for power_event in self._drain_pending_power_activation_events():
            self._append_power_followup_nudge(
                request.messages,
                power_event,
                goal_text=turn_context.user_goal or turn_context.latest_input,
            )
            yield self._meta_chunk(
                phase="planning",
                status="power_activated",
                model=request.model,
                extra={"power_activation": power_event},
            )

        # System-driven delegation: if the subagent-router Power fired on an
        # explore-shaped ask, run the explore subagent OURSELVES now (local
        # models won't) and inject its findings into the plan context below.
        async for _ev in self._maybe_auto_dispatch_explore(
            latest_user_text=turn_context.user_goal or turn_context.latest_input,
            model=request.model,
        ):
            yield _ev

        # Continuation short-circuit. "continue please" / "keep going" /
        # "monitor the download" — the user wants us to resume prior
        # work, not kick off a fresh plan→act cycle. If we have a
        # cached plan or task list from a previous turn, go straight
        # to act with that state intact. Without this, the plan phase
        # re-derives intent from scratch every continuation turn,
        # producing redundant project-description prose (observed
        # 2026-04-22 with Qwen 3.5 on a RetroArch setup session).
        #
        # Resumption requires SOMETHING to resume — if the state is
        # empty (e.g., first message in a session was "continue"),
        # fall through to the normal plan→act flow so we don't
        # silently no-op.
        has_prior_state = _has_resumable_objective_state(self._state)
        if _is_continuation_request(user_msg) and has_prior_state:
            yield self._meta_chunk(
                phase="executing", status="continuation",
                model=request.model,
            )
            self._state.phase = CoderPhase.EXECUTING
            async for chunk in self._act_phase(request, turn_context):
                yield chunk
            return

        # Native strategy bypasses the plan phase entirely. The whole
        # point of native is "trust the model the way Claude Code /
        # Qwen Code do" — pre-deriving a numbered plan teaches Qwen
        # 3.6-class models to mark items "done" after 1-2 iterations.
        # See ``_act_native`` for the full minimal-loop contract.
        if self._resolve_strategy() == "native":
            self._state.phase = CoderPhase.EXECUTING
            async for chunk in self._act_phase(request, turn_context):
                yield chunk
            return

        # Plan/Act split. Plan phase runs before act on fresh turns so the
        # resulting step list is injected into the hybrid loop's system
        # prompt. Legacy harness-based plan-skip (rewoo/architect) is gone:
        # under the hybrid default, every fresh turn gets the same
        # plan→act lifecycle.
        #
        # Question short-circuit (2026-04-20): when the plan phase
        # emits "Question: ..." (the VAGUE branch of PLAN_SYSTEM) rather
        # than a "Plan: ..." action list, skip the act phase entirely
        # and end the turn with the question streamed to the user.
        # Without this short-circuit, weak models sprint into
        # exploration on the ambiguous task and burn iterations until
        # a circuit breaker fires. Observed on "Write tests for the
        # main functionality" — model correctly asked "which kind of
        # tests?", then act phase ran anyway and broke on inspection-
        # only-loop at iter 6.
        # REFLEX short-circuit (Phase 1.3): trivial single-action edits
        # don't benefit from the plan phase. Skipping it saves an entire
        # LLM call's worth of tokens + latency. The hybrid loop's tier
        # cap (Phase 1.2) bounds the act path to 2 iters for REFLEX, so
        # the savings compound.
        if (
            self._turn_tier_for_turn.tier == Tier.REFLEX
            and self._state.phase in (CoderPhase.WAITING, CoderPhase.REVIEWING)
        ):
            yield self._meta_chunk(
                phase="planning", status="skipped_reflex",
                model=request.model,
                extra={"tier": self._turn_tier_for_turn.tier.value},
            )
            self._state.phase = CoderPhase.EXECUTING
            async for chunk in self._act_phase(request, turn_context):
                yield chunk
            return

        if self._state.phase in (CoderPhase.WAITING, CoderPhase.REVIEWING):
            async for chunk in self._plan_phase(request, turn_context):
                yield chunk
            if _plan_is_question(self._state.plan or ""):
                yield self._meta_chunk(
                    phase="planning", status="question",
                    model=request.model,
                )
                # A clarification question is not resumable objective
                # state. Clear it so a later "continue" doesn't revive
                # stale question text instead of starting a fresh turn.
                self._state.plan = ""
                self._state.plan_steps = []
                self._state.current_step = 0
                self._state.step_outputs = {}
                self._state.tasks = []
                self._state.mission = []
                # Park the session state back at WAITING so the user's
                # next message re-enters plan→act cleanly.
                self._state.phase = CoderPhase.WAITING
                return
            async for chunk in self._act_phase(request, turn_context):
                yield chunk
        elif self._state.phase == CoderPhase.EXECUTING:
            async for chunk in self._act_phase(request, turn_context):
                yield chunk
        else:
            # PLANNING — unusual re-entry, passthrough
            async for chunk in self._passthrough_stream(request):
                yield chunk

    async def _handle_manual_compact_command(
        self,
        request: InternalChatRequest,
    ) -> AsyncIterator[InternalStreamChunk]:
        """Handle a direct `/compact` request that reaches the backend.

        The normal UI intercepts `/compact` and compacts the persisted
        conversation via REST so the visible history is replaced. This
        backend fallback keeps API/terminal callers from accidentally
        sending `/compact` to the model.
        """
        from augmentum.coder.context_tokens import compact_conversation_messages

        raw_messages = [
            {"role": m.role, "content": m.content or ""}
            for m in request.messages
            if getattr(m, "role", "") in {"system", "user", "assistant"}
        ]
        result = compact_conversation_messages(
            raw_messages,
            force=True,
            limit=self._coder_compact_token_limit,
        )
        if result.compacted:
            yield self._meta_chunk(
                phase="planning",
                status="compaction",
                model=request.model,
                extra={
                    "manual": True,
                    "tokens_before": result.tokens_before,
                    "tokens_after": result.tokens_after,
                    "messages_after": len(result.messages),
                    "dropped_messages": result.dropped_messages,
                },
            )
            summary = (
                "Compacted this request context from "
                f"{result.tokens_before:,} to {result.tokens_after:,} "
                "estimated tokens. In the Coder UI, `/compact` also "
                "updates the saved visible conversation history."
            )
        else:
            summary = (
                f"Context is already compact enough "
                f"({result.tokens_before:,} estimated tokens); nothing "
                "needed to be condensed."
            )
        yield emit(
            summary,
            phase="planning",
            status="streaming",
            model=request.model,
        )
        yield self._meta_chunk(
            phase="planning",
            status="complete",
            model=request.model,
            extra={
                "manual_compact": True,
                "compacted": result.compacted,
                "tokens_before": result.tokens_before,
                "tokens_after": result.tokens_after,
            },
        )

    # ------------------------------------------------------------------
    # Phase 1 fallback
    # ------------------------------------------------------------------

    async def _passthrough_stream(
        self, request: InternalChatRequest,
    ) -> AsyncIterator[InternalStreamChunk]:
        """Pass the request straight to the backend (Phase 1 behaviour)."""
        async for chunk in self._backend.chat_stream(request):
            yield emit_relay(
                chunk,
                phase="passthrough", status="streaming",
                model_fallback=request.model,
            )

    async def _conversational_stream(
        self, request: InternalChatRequest,
    ) -> AsyncIterator[InternalStreamChunk]:
        """Thin greeting/small-talk path.

        Used by `handle_stream` when the user's message is detected as
        conversational (greeting, thanks, acknowledgement). We still go
        through the backend so the reply is natural, but we prepend a
        tight system prompt so the model stays brief and invites the
        real task instead of hallucinating work to do. No tools, no plan,
        no act phase.
        """
        convo_system = (
            "You are the coder mode assistant for Augmentum. The user "
            "has just greeted you or sent a short acknowledgement — they "
            "haven't asked for code work yet. Reply in ONE short sentence "
            "(friendly, not verbose). Then in a second short line, invite "
            "them to describe what they'd like to build, fix, or explore. "
            "Do NOT call any tools. Do NOT inspect the workspace. Do NOT "
            "assume a task."
        )
        convo_request = InternalChatRequest(
            model=request.model,
            messages=[
                Message(role="system", content=convo_system),
                *request.messages,
            ],
            stream=True,
            temperature=request.temperature,
            tools=None,
        )
        async for chunk in self._backend.chat_stream(convo_request):
            yield emit_relay(
                chunk,
                phase="conversational", status="streaming",
                model_fallback=request.model,
            )


    # ------------------------------------------------------------------
    # Shared tool execution helper
    # ------------------------------------------------------------------

    async def _execute_tool_with_verification(
        self,
        tool_name: str,
        tool_input: dict,
        tool_map: dict,
    ) -> tuple:
        """Execute a tool call with post-write verification, lint, and checkpoint.

        Returns (tool_result, checkpoint_hash, tool_id).
        """
        from augmentum.tools.base import ToolResult as _TR
        tool_id = str(uuid.uuid4())

        tool_result = await _execute_tool(
            tool_map=tool_map,
            tool_name=tool_name,
            tool_input=tool_input,
            workspace_id=self._workspace_id,
        )
        self._state.tool_calls_made += 1

        # ── Post-write verification ──────────────────────────────────
        # Small models sometimes report success but the write didn't
        # land (truncated content, encoding issues, docker errors).
        # Verify the file actually exists and has content.
        if tool_result.success and tool_name in ("file_write", "code_edit", "code_edit_batch"):
            path = tool_input.get("path", "")
            if path and self._container_manager:
                try:
                    verify_out = await self._container_manager._run_command(
                        self._workspace_id,
                        ["bash", "-c", f"wc -c < {path} 2>&1"],
                        timeout=5.0,
                    )
                    verify_out = verify_out.strip()
                    if "No such file" in verify_out or verify_out == "0":
                        tool_result = _TR(
                            success=False,
                            error=f"File verification failed: {path} {'does not exist' if 'No such' in verify_out else 'is empty'}",
                            metadata=tool_result.metadata,
                        )
                except Exception as exc:
                    log.debug(
                        "coder_post_mutation_verify_failed",
                        path=path,
                        error=str(exc),
                    )

        # ── Syntax check ─────────────────────────────────────────────
        # Applies to all three mutating tools. Pre-2026-04-20 this list
        # omitted ``code_multi_edit`` — a bug, not a decision — so multi-
        # file edits silently skipped lint and could checkpoint broken
        # syntax. code_multi_edit shapes ``tool_input`` as either a
        # single path or a list of edits; ``_mutation_paths`` normalises
        # both shapes so the post-hook fires for every touched file.
        if tool_result.success and tool_name in ("code_edit", "file_write", "code_edit_batch"):
            paths = self._mutation_paths(tool_name, tool_input)
            lint_fragments: list[str] = []
            any_error = False
            for path in paths:
                lint_output = await self._run_lint_check(path)
                if not lint_output:
                    continue
                has_error = any(
                    kw in lint_output.lower()
                    for kw in ("syntaxerror", "error:", "invalid syntax", "unexpected")
                )
                any_error = any_error or has_error
                lint_fragments.append(f"{path}:\n{lint_output}")
            if lint_fragments:
                combined = "\n\n".join(lint_fragments)
                tool_result = _TR(
                    success=not any_error,
                    output=(tool_result.output or "") + f"\n\n[Syntax check]\n{combined}",
                    error=(
                        f"Syntax errors in {', '.join(paths)}: {combined[:200]}"
                        if any_error else None
                    ),
                    metadata=tool_result.metadata,
                )

        # ── Focused test run ─────────────────────────────────────────
        # When the model edits a test file, run just that file against
        # pytest (short timeout, not the full suite — model can call
        # `test_run` for that). Result appended to output so the model
        # sees pass/fail without burning an extra tool call. Test FAIL
        # does not flip the write's success=False — the write was
        # valid; the failure is a separate signal the model should react
        # to (fix the test or fix the source). Success-flipping only
        # happens on lint errors, which ARE write problems.
        if tool_result.success and tool_name in ("code_edit", "file_write", "code_edit_batch"):
            paths = self._mutation_paths(tool_name, tool_input)
            test_outputs: list[str] = []
            for path in paths:
                if not self._is_test_file(path):
                    continue
                test_out = await self._run_focused_test(path)
                if test_out:
                    test_outputs.append(f"{path}:\n{test_out}")
            if test_outputs:
                combined = "\n\n".join(test_outputs)
                tool_result = _TR(
                    success=tool_result.success,
                    output=(tool_result.output or "") + f"\n\n[Test run]\n{combined}",
                    error=tool_result.error,
                    metadata=tool_result.metadata,
                )

        # ── Checkpoint ────────────────────────────────────────────────
        checkpoint_hash = None
        if tool_result.success and tool_name in ("code_edit", "file_write", "code_edit_batch"):
            paths = self._mutation_paths(tool_name, tool_input)
            short = ", ".join(p.replace("/workspace/", "") for p in paths[:3])
            if len(paths) > 3:
                short += f", +{len(paths) - 3} more"
            checkpoint_hash = await self._container_manager.git_checkpoint(
                self._workspace_id,
                f"Agent: {tool_name} {short}" if short else f"Agent: {tool_name}",
            )
            # Keep find_symbol/file_outline fresh MID-turn: targeted
            # background re-extraction of just the mutated files. Full
            # drift (shell/git writes) is caught by the turn-start build.
            from augmentum.config import settings as _ci_settings
            if paths and getattr(_ci_settings, "coder_code_intel_enabled", True):
                from augmentum.coder import code_intel as _ci
                _t = asyncio.create_task(
                    _ci.reindex_paths(self._container_manager, self._workspace_id, paths),
                )
                _t.add_done_callback(lambda t: t.cancelled() or t.exception())

        return tool_result, checkpoint_hash, tool_id

    # ------------------------------------------------------------------
    # Post-task verification — confirm the goal was met
    # ------------------------------------------------------------------

    async def _verify_task_results(
        self,
        tool_results: list[dict],
        tool_map: dict,
    ) -> list[dict]:
        """Run post-task verification on files that were written/edited.

        Checks that files exist and passes syntax validation.  Appends
        verification results to the tool_results list for synthesis.
        Returns the (possibly extended) list.
        """
        if not self._container_manager:
            return tool_results

        # Collect paths that were written/edited
        written_paths = set()
        for tr in tool_results:
            tool = tr.get("tool", "")
            if tool in ("file_write", "code_edit") and tr.get("success"):
                # Extract path from output_preview or error
                preview = tr.get("output_preview", "")
                if "to /workspace" in preview:
                    path = preview.split("to ")[-1].split()[0]
                    written_paths.add(path)

        if not written_paths:
            return tool_results

        # Verify each written file exists and run tests if test files exist
        test_files = [p for p in written_paths if "test" in p.lower()]
        src_files = [p for p in written_paths if "test" not in p.lower()]

        # Quick existence check for all files
        for path in written_paths:
            try:
                check = await self._container_manager._run_command(
                    self._workspace_id,
                    ["bash", "-c", f"test -f {path} && echo EXISTS || echo MISSING"],
                    timeout=5.0,
                )
                if "MISSING" in check:
                    tool_results.append({
                        "tool": "verification",
                        "success": False,
                        "output_preview": f"⚠ {path} was not created — file is missing",
                    })
            except Exception as exc:
                log.debug("coder_post_write_existence_check_failed", path=path, error=str(exc))

        # Run test files if any were created
        if test_files and src_files:
            for tf in test_files:
                try:
                    test_out = await self._container_manager._run_command(
                        self._workspace_id,
                        ["bash", "-c", f"cd /workspace && python3 -m pytest {tf} -x --tb=short 2>&1 | tail -20"],
                        timeout=30.0,
                    )
                    passed = "passed" in test_out.lower() and "failed" not in test_out.lower()
                    tool_results.append({
                        "tool": "test_run",
                        "success": passed,
                        "output_preview": test_out.strip()[:500],
                    })
                except Exception as exc:
                    log.debug("coder_post_write_test_run_failed", test_file=tf, error=str(exc))

        return tool_results

    # ------------------------------------------------------------------
    # Synthesis — generate a user-facing summary from tool results
    # ------------------------------------------------------------------

    async def _publish_turn_review(self, user_message: str) -> None:
        """Collect diffs from the active turn snapshot and publish a
        ReviewBundle to the registry.

        Called by ``_act_hybrid`` / ``_act_canonical`` just before they
        return. Silent no-op when:

        * No review_registry is wired (tests, non-review deployments).
        * No active_turn_snapshot exists (no container manager, or the
          turn was aborted before initialisation).
        * The snapshot yields zero diffs (agent did no mutating work
          — pure inspection/conversation turn; nothing to review).

        Failures inside ``collect_diffs`` are logged but don't raise —
        a broken review must never tear down the turn's final message
        to the user.
        """
        if self._review_registry is None:
            return
        snap = getattr(self._state, "active_turn_snapshot", None)
        if snap is None:
            return
        if (
            self._workspace_snapshot is not None
            and self._turn_start_workspace_paths is not None
        ):
            try:
                await self._workspace_snapshot.refresh_if_stale(force=True)
                for rel_path in sorted(
                    self._workspace_snapshot.current_paths
                    - self._turn_start_workspace_paths
                ):
                    path = f"/workspace/{rel_path}"
                    if _is_reviewable_turn_path(path):
                        snap.register_created_path(path)
            except Exception:
                log.debug("coder.review_shell_created_paths_failed", exc_info=True)
        try:
            diffs = await snap.collect_diffs()
        except Exception as exc:
            log.warning(
                "coder.review_collect_failed",
                turn_id=self._state.active_turn_id,
                error=str(exc),
            )
            return
        diffs = [d for d in diffs if _is_reviewable_turn_path(d.path)]
        if not diffs:
            return

        from augmentum.coder.reviews import ReviewBundle
        bundle = ReviewBundle(
            turn_id=self._state.active_turn_id,
            user_id=self._user_id,
            workspace_id=self._workspace_id,
            session_id=self._session_id,
            user_message=(user_message or "").strip()[:200],
            files=diffs,
            snapshot=snap,
        )
        self._review_registry.publish(bundle)

    async def _reflect_on_streak_break(
        self,
        request: InternalChatRequest,
        *,
        break_kind: str,
        streak: int,
        extra_context: str = "",
    ) -> AsyncIterator[InternalStreamChunk]:
        """Stream a one-off self-critique before terminating a streak break.

        Reflexion (arxiv:2303.11366) — generalises the handcrafted
        nudge / break message into a model-generated explanation of
        what went wrong. Asks the model three things:
          1. What it assumed that turned out to be wrong
          2. What signal in the tool results contradicted that
          3. What it would do FUNDAMENTALLY differently next time

        Failures are silent — if the backend errors or returns
        nothing, the loop's existing ``[Stopped: ...]`` terse
        message is the only user-visible artifact, identical to
        pre-Reflexion behaviour. Capped via
        ``coder_reflexion_max_tokens`` so a verbose model can't
        keep streaming after termination.
        """
        from augmentum.config import settings as _settings

        if not getattr(_settings, "coder_reflexion_on_break", True):
            return

        max_tokens = int(
            getattr(_settings, "coder_reflexion_max_tokens", 220),
        )

        prompt = (
            f"Your last attempt hit a {break_kind} after {streak} "
            "iterations. Before stopping, take 3 sentences MAX to "
            "answer:\n"
            "1. What did you assume that turned out to be wrong?\n"
            "2. What signal from your tool results should have warned you?\n"
            "3. If you had one more attempt, what would you do "
            "FUNDAMENTALLY differently — not retry, change approach?\n\n"
            "Be terse. No preamble. No apology. Just the 3 answers."
        )
        if extra_context:
            prompt += f"\n\nContext: {extra_context}"

        reflect_request = InternalChatRequest(
            model=request.model,
            messages=[Message(role="user", content=prompt)],
            stream=True,
            temperature=0.4,  # slight diversity, not random
            max_tokens=max_tokens,
        )

        try:
            async for chunk in self._backend.chat_stream(reflect_request):
                if chunk.content_delta:
                    yield emit(
                        chunk.content_delta,
                        phase="executing", status="reflection",
                        model=request.model,
                    )
        except Exception as exc:
            log.warning(
                "coder.reflection_failed",
                break_kind=break_kind, error=str(exc),
            )

    async def _synthesize_response(
        self,
        request: InternalChatRequest,
        user_query: str,
        tool_results: list[dict],
    ) -> AsyncIterator[InternalStreamChunk]:
        """Ask the LLM to summarize tool results into a helpful response.

        Called after direct/architect/decompose strategies complete their
        tool execution.  Streams the response token-by-token so the user
        sees it appear progressively.
        """
        # Build a compact context of what tools returned
        results_text = []
        for tr in tool_results:
            status = "OK" if tr.get("success") else "FAILED"
            preview = (tr.get("output_preview") or tr.get("error") or "")[:400]
            results_text.append(f"[{tr.get('tool', '?')}] {status}\n{preview}")

        if not results_text:
            return

        # Check if there were failures
        had_failures = any(not tr.get("success") for tr in tool_results)

        synth_system = (
            "You are a coding assistant. The user asked a question and tools were "
            "executed. Summarize what happened concisely for the user.\n"
            "Rules:\n"
            "- Be direct: say what was created, what worked, what failed\n"
            "- If files were created, list them with a one-line description\n"
            "- If tests ran, report pass/fail with key details\n"
            "- If something failed, explain what went wrong and suggest a fix\n"
            "- Do NOT list tool names or internal mechanics\n"
            "- Do NOT re-describe the project or re-summarize what this "
            "repo is. The user already knows what the project is — they "
            "are mid-conversation with you. Answer about THIS TURN's "
            "actions and their outcome, nothing else.\n"
            "- If the user's latest message was 'continue' / 'keep going' "
            "/ 'monitor X', they want an update on in-flight work, not a "
            "project explainer. Report progress, not provenance.\n"
            "- Use markdown formatting when it improves readability"
        )
        if self._is_environment_audit_turn(user_query):
            synth_system += (
                "\n- This was an environment/workspace audit. Structure the response with "
                "short markdown sections named exactly: `Observed now`, "
                "`Intended baseline`, `Missing or not observed`, and "
                "`What this workspace supports`."
                "\n- Only put items in `Observed now` if they were directly confirmed by "
                "runtime truth or tool output."
                "\n- Use `Intended baseline` for design-time defaults or fallback bootstrap intent."
                "\n- Put anything absent from direct observation under `Missing or not observed`."
            )
        closeout_contract = self._render_power_closeout_contract()
        if closeout_contract:
            synth_system += (
                "\n- A specialist verifier was active for this turn. Honor this "
                f"close-out contract exactly:\n{closeout_contract}"
            )
        runtime_truth_block = ""
        if self._is_environment_audit_turn(user_query) and self._runtime_truth_for_turn is not None:
            runtime_truth_block = (
                f"Runtime truth:\n{self._runtime_truth_for_turn.render_block()}\n\n"
            )
        synth_messages = [
            Message(role="system", content=synth_system),
            Message(
                role="user",
                content=(
                    f"User's request: {user_query}\n\n"
                    f"{runtime_truth_block}"
                    f"Tool results:\n{'---'.join(results_text)}\n\n"
                    + ("Some steps failed — explain what went wrong clearly.\n" if had_failures else "")
                    + "Provide a clear, concise response to the user."
                ),
            ),
        ]

        synth_request = InternalChatRequest(
            model=request.model,
            messages=synth_messages,
            stream=True,
            temperature=request.temperature,
        )

        from augmentum.modes.coder.chat_egress import StreamProgressTracker
        progress = StreamProgressTracker()
        yield progress.begin(phase="executing", model=request.model)
        try:
            async for chunk in self._backend.chat_stream(synth_request):
                ev = progress.update(
                    chunk, phase="executing", model=request.model,
                )
                if ev is not None:
                    yield ev
                if chunk.content_delta:
                    yield emit(
                        chunk.content_delta,
                        phase="executing", status="streaming",
                        model=request.model,
                    )
        except Exception as exc:
            log.warning("coder.synthesis_failed", error=str(exc))

    # ------------------------------------------------------------------
    # Act phase — strategy router
    # ------------------------------------------------------------------

    async def _act_phase(
        self,
        request: InternalChatRequest,
        turn_context: TurnContext,
    ) -> AsyncIterator[InternalStreamChunk]:
        """Execute using the best strategy for the current model."""
        self._state.phase = CoderPhase.EXECUTING

        yield self._meta_chunk(
            phase="executing",
            status="started",
            model=request.model,
        )

        # Ensure workspace guide is cached (used by _build_messages)
        await self._get_workspace_guide()

        # Production dispatch: native is THE shipped strategy — the pure
        # Claude-Code/Qwen-Code parity loop, and the single source of truth
        # for loop guards (TQG, silent-success, stagnation, verify/goal
        # gates). hybrid, canonical, and legacy are all FROZEN: reachable
        # only via AUGMENTUM_CODER_STRATEGY / the strategy header, kept
        # loadable for rollback + A-B comparison, NOT hand-synced with
        # native's guards. New guard work goes in _act_native only (see the
        # frozen-method docstrings). Every production path flows through this
        # switch — no caller bypasses it.
        strategy = self._resolve_strategy()
        if strategy == "native":
            # Native skips the heavy act context (semantic search and
            # sticky scaffolding), but still gets a bounded orientation
            # prelude: runtime truth, recent turn memory, workspace
            # profile, and project grounding when already available.
            # prior_turns intentionally NOT passed — it gets carried in
            # the user-role runtime carrier inserted by _act_native, same
            # as canonical/hybrid. Keeping it out of the system prefix
            # lets the slot prefix-cache survive across turns.
            native_context = turn_context.to_native_context(
                runtime_truth_block=self._runtime_truth_context_block,
            )
            # Query-dependent grounding (auto-recall) is kept OUT of the
            # system prefix and handed to the runtime carrier instead —
            # it re-ranks per user query, and in the system prompt it
            # invalidated the whole slot cache every turn (measured
            # 2026-07-02, kv_prefix_stability contract=violated at
            # message 0).
            self._turn_dynamic_context_block = (
                turn_context.to_native_dynamic_context()
            )
            log.info("coder.strategy_selected", model=request.model,
                     strategy="native",
                     context_chars=len(native_context))
            async for chunk in self._act_native(
                request,
                workspace_context=native_context,
            ):
                yield chunk
            return

        # Build workspace context (project digest OR workspace snapshot
        # + repo map + semantic search). The project digest is a
        # small-workspace optimisation — when every file fits under
        # the token budget, inlining them all up-front eliminates the
        # iteration-burning dir_tree/file_read dance. Returns None on
        # larger projects, falling through to the existing path.
        # Shared per-turn workspace grounding is built once in
        # _handle_stream_body and rendered here for the act phase.
        workspace_context = await turn_context.to_act_context()
        # Dynamic half (semantic hits + auto-recall) — carried in the
        # per-turn runtime carrier, NOT the system prefix. See
        # ``TurnContext.to_act_dynamic_context`` for the cache rationale.
        self._turn_dynamic_context_block = (
            await turn_context.to_act_dynamic_context()
        )
        if strategy == "canonical":
            log.info("coder.strategy_selected", model=request.model,
                     strategy="canonical")
            async for chunk in self._act_canonical(request, workspace_context):
                yield chunk
            return
        if strategy == "legacy":
            async for chunk in self._act_phase_legacy(request, workspace_context):
                yield chunk
            return

        # Hybrid is no longer the default but remains reachable via env
        # var / header override.
        log.info("coder.strategy_selected", model=request.model,
                 strategy="hybrid")
        async for chunk in self._act_hybrid(request, workspace_context):
            yield chunk


    def _resolve_strategy(self) -> str:
        """Resolve effective coder strategy for this turn.

        Priority: per-request override (constructor arg from the
        ``X-Augmentum-Coder-Strategy`` header) → env var
        (``AUGMENTUM_CODER_STRATEGY``) → ``native`` default.

        Unknown values warn and fall back to ``native`` so a typo in
        the header (UI bug, scripted client mistake) degrades to the
        default loop instead of crashing the turn.
        """
        candidate = self._coder_strategy_override or _CODER_STRATEGY_OVERRIDE
        if candidate not in _CODER_KNOWN_STRATEGIES:
            log.warning(
                "coder.strategy_unknown",
                requested=candidate, fallback="native",
            )
            return "native"
        return candidate

    def _extract_user_goal(self, request: InternalChatRequest) -> str:
        """Thin wrapper over ``_last_user_message`` for summary callers.

        Both strategies (``_act_canonical`` and ``_act_hybrid``) need
        "the latest user turn, stripped and stringy" for turn-summary
        book-keeping. Centralising here keeps them in sync if the
        terminal-wrapper parsing ever evolves.
        """
        return self._last_user_message(request) or ""

    def _last_user_message(self, request: InternalChatRequest) -> str:
        """Return the user's latest message, stripped of the UI's context wrappers.

        The coder UI prepends ``[Terminal context]\\n<buffer>\\n// <intent>``
        before sending, so the raw content is not a clean prompt. We
        extract the final ``// ...`` intent line when present, otherwise
        fall back to the raw content trimmed to a single line.
        """
        for m in reversed(request.messages):
            if m.role != "user":
                continue
            raw = (m.content or "").strip()
            if not raw:
                continue
            # Shared de-wrapping (the // intent line, else strip the
            # [Terminal context] preamble, else first line). See
            # _clean_user_text — the single seam for goal cleaning.
            return _clean_user_text(raw, single_line=True)
        return ""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_messages(
        self,
        request: InternalChatRequest,
        extra_system: str,
    ) -> list[Message]:
        """Build a message list with a stable system prefix + dynamic tail.

        Layout (designed for llama-server slot prefix-cache reuse):

          system (stable parts that change rarely across turns):
            guide → kernel_v2_hint → cached_facts_block
            → runtime_truth_context_block
            → workspace_root_probe_context_block
            → user_sys (datetime de-duped) → extra_system

          …conversation history…

          [CARRIER injected just before the latest user message]:
            role=user, framed as "[Augmentum runtime context — not
            user dialogue]" containing prior_turns + power blocks +
            per-query dynamic grounding (semantic hits / auto-recall)
            + dt_context LAST (most volatile at the extreme tail)

          latest user message

        The datetime / prior_turns / power blocks rotated every turn
        (datetime by the minute) when they lived in the system prompt,
        invalidating llama-server's prefix-cache for everything below
        them on every single turn. Hoisting them into a carrier at the
        tail keeps the long stable prefix byte-identical so the slot
        cache hits up through the prior assistant response; only the
        carrier + new user message need re-tokenisation.

        The carrier uses role=user with explicit framing (same pattern
        as ``LlamaCppBackend._late_system_context_carrier``) so it
        survives ``_normalize_system_messages`` without coupling coder
        to the kv_stable_messages checkpoint machinery, and works
        uniformly across backends (not just llama-server).
        """
        guide = self._cached_guide or ""
        messages = list(request.messages)

        # Workspace-kernel v2 hint — one sentence telling the model
        # that /workspace/.augmentum/ exists as a scratch directory
        # with files the kernel maintains. Replaces per-iteration
        # sticky-reminder injection of plan content with on-demand
        # `file_read`. Empty in v1; only rendered when the flag is on.
        # The kernel owns the hint text — single-sourced so native's
        # sys_text (built in phase_act._act_native) renders the same
        # block. Static method so the unit-test path (which doesn't
        # construct a real kernel) still gets the hint.
        from augmentum.coder.workspace_kernel import WorkspaceKernel as _Kernel
        kernel_v2_hint = _Kernel.hint_text()

        # Stable system parts — must stay byte-identical across turns
        # for the prefix cache to hit. Facts block sits adjacent to the
        # kernel hint — both are kernel-layer content and reading them
        # as a pair makes the model's mental model cleaner.
        stable_parts = [
            p
            for p in [
                guide,
                # Orchestrator fork contract — stable across the run (the
                # dispatch is fixed at handler construction), so it belongs in
                # the prefix-cached stable prefix. Empty on direct-user turns.
                self._dispatch_system_block,
                kernel_v2_hint,
                self._cached_facts_block,
                self._repo_map_context_block,
                self._runtime_truth_context_block,
                self._workspace_root_probe_context_block,
            ]
            if p
        ]

        if messages and messages[0].role == "system":
            user_sys = messages[0].content
            # Strip the datetime block if ModeHandler already injected it
            user_sys_clean = re.sub(r"<current_time>.*?</current_time>\n*", "", user_sys, flags=re.DOTALL).strip()
            if user_sys_clean:
                stable_parts.append(user_sys_clean)
            stable_parts.append(extra_system)
            messages[0] = Message(role="system", content="\n\n".join(stable_parts))
        else:
            stable_parts.append(extra_system)
            messages.insert(0, Message(role="system", content="\n\n".join(stable_parts)))

        carrier = self._build_runtime_carrier_message()
        if carrier is not None:
            insert_at = self._last_user_index(messages)
            if insert_at is None:
                messages.append(carrier)
            else:
                messages.insert(insert_at, carrier)

        return messages

    def _render_preview_console_block(self) -> str:
        """Render NEW live-preview console/error events since last turn, or ''.

        Advances the per-workspace watermark so each captured event is injected
        exactly once. The events come from the user's REAL preview session
        (``preview_console.py``), which the headless browser tools structurally
        miss. Best-effort — never raises into the turn path; gated by
        ``coder_preview_console_capture``.
        """
        try:
            from augmentum.config import settings as _settings
            if not getattr(_settings, "coder_preview_console_capture", True):
                return ""
            from augmentum.coder import preview_console as _pc
            new = _pc.snapshot(
                self._workspace_id, since=self._preview_console_seen, limit=15,
            )
            if not new:
                return ""
            self._preview_console_seen = _pc.high_water(self._workspace_id)
            lines = [
                "<preview_console>",
                "Console/error events from the user's LIVE preview since your "
                "last turn. They may or may not relate to the current request "
                "— investigate before assuming.",
            ]
            for e in new:
                loc = f" ({e['url']}:{e['line']})" if e.get("line") else ""
                lines.append(
                    f"  [{e.get('type', 'error')}] {e.get('text', '')}{loc}"
                )
            lines.append("</preview_console>")
            return "\n".join(lines)
        except Exception:
            return ""

    def _build_runtime_carrier_message(self) -> Message | None:
        """Build the per-turn dynamic context carrier, or None if empty.

        Aggregates every block whose content changes faster than turn
        boundaries (datetime mutates by the minute) or that mutates
        with state the user can flip mid-session (powers activating,
        prior_turns growing past the FIFO cap). Kept out of the system
        prefix so the slot's KV stays cache-hot across turns.

        Returns None when nothing dynamic needs to be conveyed so the
        caller skips insertion and we don't pollute the message list
        with an empty carrier.

        Ordering inside the carrier is deliberate — most-stable first,
        most-volatile last — so the token LCP into the carrier itself
        is as long as possible on the next turn:

          prior_turns (append-mostly ring) → power blocks (rarely flip)
          → semantic hits + auto-recall (re-ranked per query)
          → <current_time> LAST (mutates by the minute).

        Timestamps inside the earlier blocks are ABSOLUTE ("Jul 2
        14:32"), never relative ("7m ago") — relative phrasing would
        re-render every minute and defeat the ordering. The model
        computes recency from the current-time block at the end.
        """
        dynamic_parts = [
            p
            for p in [
                self._render_prior_turns(),
                self._controller_power_context_block,
                self._auto_explore_context_block,
                self._active_power_context_block,
                getattr(self, "_turn_dynamic_context_block", "") or "",
                self._preview_console_context_block,
                get_datetime_context(),
            ]
            if p
        ]
        if not dynamic_parts:
            return None
        body = "\n\n".join(dynamic_parts)
        return Message(
            role="user",
            content=(
                f"{RUNTIME_CARRIER_HEADER}\n"
                "Treat the following block as authoritative state at "
                "the start of this turn. Do not answer it; use it when "
                "relevant to the user message that follows.\n\n"
                f"{body}"
            ),
        )

    @staticmethod
    def _last_user_index(messages: list[Message]) -> int | None:
        """Return the index of the latest user message, or None."""
        for i in range(len(messages) - 1, -1, -1):
            if messages[i].role == "user":
                return i
        return None

    # ------------------------------------------------------------------
    # Hydrated recency buffer — replay recent turns' FULL tool chains
    # ------------------------------------------------------------------

    @staticmethod
    def _fresh_turn_input(user_messages: list[Message]) -> list[Message]:
        """This turn's genuinely-new input from the client payload.

        The client re-sends the whole (collapsed) conversation each turn;
        the only new part is whatever follows the last assistant reply.
        The recency buffer already represents everything up to and
        including that reply, so this is the seam where hydrated history
        meets fresh input. On the first turn (no prior assistant) the
        whole payload is fresh.
        """
        last_asst = -1
        for i, m in enumerate(user_messages):
            if m.role == "assistant":
                last_asst = i
        return user_messages[last_asst + 1:] if last_asst >= 0 else list(user_messages)

    def _apply_recency_buffer(
        self, user_messages: list[Message], fresh: list[Message],
    ) -> list[Message]:
        """Swap the client's collapsed prior turns for hydrated chains.

        Returns ``hydrated_recent_turns + fresh`` when the buffer is
        populated; otherwise returns ``user_messages`` unchanged (safe
        fallback on a fresh handler / first turn, so nothing is ever
        dropped). Every buffered turn is a pairing-complete unit
        (assistant tool_calls captured together with their tool results),
        so injecting them can't orphan a ``tool_call_id``.
        """
        buffer = getattr(self, "_recent_turn_chains", None)
        if not buffer:
            return user_messages
        # Guard: only substitute when we could isolate a fresh user turn,
        # else fall back to the client payload rather than risk dropping
        # the user's actual message.
        if not any(m.role == "user" for m in fresh):
            return user_messages
        hydrated = [m for turn in buffer for m in turn]
        return hydrated + fresh

    def _current_turn_chain(self, messages: list[Message]) -> list[Message] | None:
        """Isolate THIS turn's chain from the full seeded ``messages`` list.

        Returns the fresh user input (``self._pending_turn_input``, set at
        seed time) + every assistant/tool message appended during the loop.
        Synthetic user-role injections (nudges, the runtime carrier, steer
        messages) are excluded. Returns ``None`` when the seam can't be
        resolved (no pending input, or the anchor object isn't found).

        Critical for two consumers that must NOT see the hydrated recency
        buffer's prior-turn tool messages: the turn-summary builder (else
        prior turns' files/commands get re-attributed to this turn) and
        the buffer capture itself. Located by object identity so it
        survives mid-turn compaction (which rebuilds the list but keeps
        the recent-tail Message objects by reference).
        """
        pending = getattr(self, "_pending_turn_input", None) or []
        if not pending:
            return None
        anchor = pending[-1]
        idx = None
        for i in range(len(messages) - 1, -1, -1):
            if messages[i] is anchor:
                idx = i
                break
        if idx is None:
            return None
        tail = [
            m for m in messages[idx + 1:]
            if m.role in ("assistant", "tool")
        ]
        return [*pending, *tail]

    def _capture_recency_turn(self, messages: list[Message]) -> None:
        """Stash THIS turn's full in-format chain for the next turn's seed.

        ONLY tool-using turns are buffered. A prose-only exchange in the
        hydrated window is an anti-exemplar: it shows the model
        "user asks → I answer without tools," pulling it toward the exact
        prose-only regression the buffer exists to fix. Conversational
        turns fall back to the ``<prior_turns>`` summary like older turns.

        Best-effort: any miss leaves the buffer untouched. Bounded to
        ``_RECENCY_BUFFER_TURNS`` (newest kept).
        """
        if _RECENCY_BUFFER_TURNS <= 0:
            return
        chain = self._current_turn_chain(messages)
        if not chain:
            return
        if not any(m.role == "assistant" for m in chain):
            return
        # Tool-only gate — the whole point of the buffer is a tool exemplar.
        if not any(m.role == "tool" for m in chain):
            return
        buf = getattr(self, "_recent_turn_chains", None)
        if buf is None:
            buf = []
            self._recent_turn_chains = buf
        buf.append(chain)
        if len(buf) > _RECENCY_BUFFER_TURNS:
            del buf[: len(buf) - _RECENCY_BUFFER_TURNS]

    def _render_prior_turns(self) -> str:
        """Render the persisted turn-summary ring as a <prior_turns> block.

        Returns an empty string when there's nothing to render — that
        makes the caller's join-with-blank-lines logic no-op naturally.
        The block is designed to fit ~500 tokens at cap (10 summaries ×
        ~50 tokens each) so it's cheap to re-inject every turn.

        Each summary renders as a fixed-shape stanza:

            ## Turn N (outcome)
            Goal: "..."
            Read: path1, path2
            Edited: path1 (what), path2 (what)
            Blockers: short note

        Empty fields are elided. "outcome" is the termination reason
        mapped to a human label (done / incomplete / stopped).
        """
        summaries = self._state.turn_summaries
        if not summaries:
            return ""

        stanzas: list[str] = []
        for s in summaries:
            try:
                idx = s.get("turn_idx", 0)
                goal = (s.get("user_goal") or "").strip()
                reads = s.get("files_read") or []
                edits = s.get("files_edited") or []
                shells = s.get("shell_commands") or []
                outcome = (s.get("outcome") or "unknown").strip()
                blockers = (s.get("blockers") or "").strip()
            except AttributeError:
                continue

            # Cancellation / error path: render a distinct stanza so
            # the model has an explicit signal that the turn ended
            # because the user pressed Stop / a slash command fired /
            # the backend errored — NOT because the agent itself
            # decided the work was done. Without this branch, a
            # cancelled turn renders as ``(cancelled)`` with no reason
            # and no "where it stopped" context, and the model on the
            # next turn tends to silently resume the abandoned plan.
            if s.get("cancelled") or s.get("errored"):
                stanzas.append(self._render_interruption_stanza(s))
                continue

            lines = [f"## Turn {idx} ({outcome})"]
            if goal:
                # Clip long goals — a 2000-char first-user-msg would bust
                # our prompt budget before we got to anything useful.
                clipped = goal[:300] + ("…" if len(goal) > 300 else "")
                lines.append(f'Goal: "{clipped}"')
            if reads:
                lines.append(f"Read: {', '.join(reads[:12])}")
            if edits:
                # Each edit is {path, summary}; fall back gracefully.
                edit_parts: list[str] = []
                for e in edits[:8]:
                    if isinstance(e, dict):
                        p = e.get("path", "")
                        note = (e.get("summary") or "").strip()
                        edit_parts.append(f"{p} ({note})" if note else p)
                    elif isinstance(e, str):
                        edit_parts.append(e)
                if edit_parts:
                    lines.append(f"Edited: {', '.join(edit_parts)}")
            if shells:
                # Build / install / deploy / run turns do their work via
                # shell_exec rather than file writes. Without this the
                # <prior_turns> block says "nothing happened" and the
                # next turn re-runs env discovery. Cap at 6 so a busy
                # turn doesn't dominate the prompt budget.
                shell_display = [
                    (s if len(s) <= 90 else s[:87] + "…")
                    for s in shells[:6]
                    if isinstance(s, str)
                ]
                if shell_display:
                    lines.append(f"Ran: {'; '.join(shell_display)}")
            if blockers:
                lines.append(f"Blockers: {blockers[:200]}")
            stanzas.append("\n".join(lines))

        body = "\n\n".join(stanzas)
        # If any of the summaries is an interruption, extend the
        # block header so the model has a clear directive on how to
        # treat them — the marker alone isn't enough, weak models
        # otherwise read "cancelled" as "almost done, finish it".
        has_interruption = any(
            s.get("cancelled") or s.get("errored") for s in summaries
        )
        intro = (
            "The model's own memory of recent work in this session. "
            "Use this to avoid re-reading files or re-running searches "
            "you already did. Do not ignore — these are grounded facts "
            "from earlier turns."
        )
        if has_interruption:
            intro += (
                "\n\nTurns marked INTERRUPTED ended because the user "
                "or the system stopped them — not because the work was "
                "finished. Treat the current user message as a fresh "
                "instruction; do not silently resume what was interrupted "
                "unless the user explicitly asks you to continue."
            )
        # No count="N" attribute on the opening tag — it mutated every
        # turn at the very START of the runtime carrier body, cutting
        # the token-LCP before the (append-mostly) stanzas began. The
        # model can count stanzas itself; the cache can't un-see a
        # changed prefix byte.
        return (
            f"<prior_turns>\n"
            f"{intro}\n\n{body}\n</prior_turns>"
        )

    # ------------------------------------------------------------------
    # Interruption stanza renderer (cancellation / mid-turn error)
    # ------------------------------------------------------------------

    # Reason → human label map. Anything not in this table renders as
    # the raw token (e.g. ``"timeout"`` becomes ``the turn exceeded its
    # time budget``, while a hypothetical ``"custom_thing"`` would just
    # appear as ``custom_thing``). Keeping this in one place lets the
    # /cancel route and the renderer share a vocabulary without a
    # circular import.
    _INTERRUPTION_REASON_LABELS = {
        "user_cancel":       "user pressed Stop",
        "slash_clear":       "user ran /clear",
        "slash_compact":     "user ran /compact",
        "new_turn_started":  "user started a new turn",
        "page_unload":       "user closed the page",
        "server_shutdown":   "server was shutting down",
        "timeout":           "the turn exceeded its time budget",
        "rate_limit":        "the backend rate-limited the request",
        "network":           "the network connection failed",
        "backend_error":     "the backend returned an error",
    }

    def _render_interruption_stanza(self, s: dict) -> str:
        """Render a cancelled / errored turn stanza for ``<prior_turns>``.

        Sibling of the main loop in ``_render_prior_turns`` — kept
        separate so the normal "done / incomplete" rendering stays
        compact. Schema is the superset written by
        ``_record_interruption_summary``.
        """
        idx = s.get("turn_idx", 0)
        reason = (s.get("cancel_reason") or "").strip() or "user_cancel"
        reason_label = self._INTERRUPTION_REASON_LABELS.get(reason, reason)
        kind = "ERROR" if s.get("errored") else "CANCELLED"
        goal = (s.get("user_goal") or "").strip()
        blockers = (s.get("blockers") or "").strip()
        reads = s.get("files_read") or []
        interrupted = s.get("interrupted_at") or {}
        last_tool = interrupted.get("last_tool") or {}
        active_task = interrupted.get("active_task") or {}

        lines = [f"## Turn {idx} (INTERRUPTED — {kind}: {reason_label})"]
        if goal:
            clipped = goal[:300] + ("…" if len(goal) > 300 else "")
            lines.append(f'Goal: "{clipped}"')

        # Where the agent was when it stopped. Anchor the model so it
        # knows the cancellation interrupted *specific* work — not the
        # whole session. Helps it judge whether the user wants that
        # work continued or replaced.
        if last_tool.get("tool"):
            target = (last_tool.get("target") or "").strip()
            if target:
                lines.append(
                    f"Stopped while running: {last_tool['tool']} ({target})",
                )
            else:
                lines.append(f"Stopped while running: {last_tool['tool']}")

        phase = (interrupted.get("phase") or "").strip()
        calls_made = int(interrupted.get("tool_calls_made") or 0)
        cur_step = int(interrupted.get("current_step") or 0)
        total_steps = int(interrupted.get("total_steps") or 0)
        state_bits: list[str] = []
        if phase and phase != "waiting":
            state_bits.append(f"phase={phase}")
        if calls_made:
            state_bits.append(f"tool_calls_so_far={calls_made}")
        if total_steps:
            state_bits.append(f"step={cur_step}/{total_steps}")
        if state_bits:
            lines.append(f"State at stop: {', '.join(state_bits)}")

        active_content = (active_task.get("content") or "").strip()
        if active_content:
            lines.append(
                f"Task in progress at stop: {active_content[:160]}",
            )

        if reads:
            lines.append(f"Files touched this turn: {', '.join(reads[:8])}")

        if blockers:
            # For the errored variant this carries the exception text;
            # for the cancelled variant it's usually empty.
            lines.append(f"Error: {blockers[:200]}")

        # The trailing directive is the load-bearing piece. Without
        # it, the model reads "CANCELLED" and tries to "complete" the
        # interrupted plan as a helpful gesture. Be explicit.
        if s.get("errored"):
            lines.append(
                "→ This turn failed before it could complete. "
                "Re-attempt only the part the user's new message asks "
                "you to.",
            )
        else:
            lines.append(
                "→ Do not silently resume the work above. The user's "
                "new message is the current instruction; if they want "
                "you to continue what was interrupted, they will say so.",
            )
        return "\n".join(lines)

    def _build_turn_summary(
        self,
        *,
        messages: list,
        user_goal: str,
        termination_reason: str,
    ) -> dict:
        """Scan the final in-turn ``messages`` list to produce a summary dict.

        Purely algorithmic — no LLM call. Walks tool/assistant messages
        to collect file reads, edits, and the last blocking error.

        Returns a dict with a stable schema (so ``to_dict`` /
        ``_render_prior_turns`` don't have to guess):

            {turn_idx, user_goal, files_read, files_edited,
             outcome, blockers, created_at}
        """
        import time as _time

        # Defensive: callers pass turn_context.user_goal (already cleaned via
        # _extract_goal_split), but this is the single chokepoint feeding both
        # the prior-turns ring AND the durable archive, so re-clean here so no
        # path can persist a raw [Terminal context] goal. Idempotent on clean
        # input.
        user_goal = _clean_user_text(user_goal, single_line=False)

        # Outcome mapping — user-facing, short.
        #
        # Phase 3.6 introduced colon-suffix variants of ``model_stop``
        # (e.g., ``model_stop:already_nudged``, ``model_stop:recent_progress``)
        # so traces preserve the granular gate verdict. The user-facing
        # outcome stays simple — strip the suffix before lookup so any
        # ``model_stop:*`` maps to ``done`` and the full reason still
        # lives in the chunk-level ``reason`` extra for log readers.
        outcome_map = {
            "model_stop":              "done",
            "model_stop_after_nudge":  "done",
            "model_stop_with_answer":  "done",
            "tasks_completed":         "done",
            "finish_task":             "done",
            "max_iterations_reached":  "incomplete",
            "validation_error_streak": "stopped (tool errors)",
            "backend_error":           "stopped (backend error)",
        }
        base_reason = (termination_reason or "").split(":", 1)[0]
        outcome = outcome_map.get(
            base_reason, termination_reason or "unknown",
        )
        # Verdict suffix (after the colon) carries the granular TQG
        # decision tag — ``already_nudged``, ``substantive_active``,
        # ``recent_progress``, etc. Surface it alongside the outcome
        # so the inspector can distinguish a clean "done" from a
        # "done because we hit the nudge cap" without parsing the
        # full reason string.
        verdict_reason = ""
        if ":" in (termination_reason or ""):
            verdict_reason = termination_reason.split(":", 1)[1].strip()

        files_read: list[str] = []
        files_edited: list[dict] = []
        # Phase 2.1: per-edit annotations carrying enough context that a
        # later turn (or the retro loop) can reason about what was
        # changed and why. Heuristic-only — the user_goal in the turn
        # summary plus these snippets give the model a "what + why"
        # view without an extra LLM call. ~200 bytes per edit; turn
        # summary cap of 10 keeps this bounded.
        edits: list[dict] = []
        shell_commands: list[str] = []
        last_error: str = ""

        def _snippet(text: object, *, limit: int = 80) -> str:
            """Compact a search/replace value for the summary block.

            Newlines and runs of whitespace become single spaces so a
            multi-line edit doesn't dominate the rendered prior_turns
            block. Non-string inputs (e.g. tools that pass dicts in
            args) collapse to empty so the entry shape stays stable.
            """
            if not isinstance(text, str):
                return ""
            collapsed = " ".join(text.split())
            return collapsed if len(collapsed) <= limit else collapsed[: limit - 1] + "…"

        # Walk assistant messages in order; each may carry tool_calls.
        # Pair them with the *next* tool message that has the matching
        # tool_call_id so we only record successful reads/edits.
        tool_results_by_id: dict[str, str] = {}
        for m in messages:
            if m.role == "tool":
                tcid = getattr(m, "tool_call_id", None) or ""
                if tcid:
                    tool_results_by_id[tcid] = (m.content or "")

        seen_reads: set[str] = set()
        seen_edits: set[str] = set()
        for m in messages:
            if m.role != "assistant":
                continue
            tcs = getattr(m, "tool_calls", None) or []
            for tc in tcs:
                if isinstance(tc, dict):
                    name = tc.get("function", {}).get("name", "") or tc.get("name", "")
                    args_raw = tc.get("function", {}).get("arguments") or tc.get("input") or {}
                    tc_id = tc.get("id", "")
                else:
                    name = getattr(tc, "name", "")
                    args_raw = getattr(tc, "input", {}) or {}
                    tc_id = getattr(tc, "id", "")
                if isinstance(args_raw, str):
                    try:
                        args = json.loads(args_raw)
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                else:
                    args = args_raw or {}
                if not isinstance(args, dict):
                    continue

                result = tool_results_by_id.get(tc_id, "")
                is_error = result.startswith("ERROR:")

                if name == "file_read" and not is_error:
                    path = (args.get("path") or "").strip()
                    if path and path not in seen_reads:
                        seen_reads.add(path)
                        files_read.append(path)
                elif name == "shell_exec" and not is_error:
                    # Record successful shell invocations so the next
                    # turn's ``<prior_turns>`` block shows what was run
                    # — without this, the build / install / deploy work
                    # a turn did is invisible to the next turn, and the
                    # model tends to re-run env discovery it already did.
                    # Dedup on exact-command + trim to 120 chars so a
                    # multi-line pipe doesn't dominate the block.
                    cmd = (args.get("command") or "").strip()
                    if cmd and cmd not in shell_commands:
                        display = cmd if len(cmd) <= 120 else cmd[:117] + "…"
                        shell_commands.append(display)
                elif name in ("code_edit", "code_edit_batch", "file_write") and not is_error:
                    path = (args.get("path") or "").strip()
                    if path and path not in seen_edits:
                        seen_edits.add(path)
                        # Brief note — "wrote N lines" for file_write,
                        # "edited" for SEARCH/REPLACE. Keep under 40
                        # chars so the rendered block stays compact.
                        if name == "file_write":
                            content = args.get("content") or ""
                            note = f"wrote {len(content.splitlines())} lines"
                        elif name == "code_edit_batch":
                            blocks = args.get("blocks") or args.get("edits") or []
                            note = f"{len(blocks)} edits"
                        else:
                            note = "edited"
                        files_edited.append({"path": path, "summary": note})

                        # Phase 2.1: capture snippets for the edit-history
                        # annotation. Stable per-tool shape so the retro
                        # loop / cross-turn reader can reason about what
                        # changed without re-walking the message log.
                        if name == "file_write":
                            content = args.get("content") or ""
                            edits.append({
                                "path": path,
                                "tool": name,
                                "search_snippet": "",
                                "replace_snippet": _snippet(content),
                                "block_count": 1,
                                "lines_written": len(content.splitlines()),
                            })
                        elif name == "code_edit_batch":
                            blocks = args.get("blocks") or args.get("edits") or []
                            first = blocks[0] if blocks and isinstance(blocks[0], dict) else {}
                            edits.append({
                                "path": path,
                                "tool": name,
                                "search_snippet": _snippet(first.get("search")),
                                "replace_snippet": _snippet(first.get("replace")),
                                "block_count": len(blocks),
                                "lines_written": 0,
                            })
                        else:  # code_edit
                            edits.append({
                                "path": path,
                                "tool": name,
                                "search_snippet": _snippet(args.get("search")),
                                "replace_snippet": _snippet(args.get("replace")),
                                "block_count": 1,
                                "lines_written": 0,
                            })
                elif is_error:
                    # Capture a short version of the most-recent blocker.
                    # Assistant messages are walked in order, so the last
                    # error seen wins — that's what we want in the summary.
                    last_error = result[len("ERROR:"):].strip()[:200]

        return {
            "turn_idx": len(self._state.turn_summaries) + 1,
            # Stamp the originating turn id so the rewind endpoint can
            # pop only the summary belonging to the run being rewound.
            # Older persisted summaries lack this field; rewind falls
            # back to a "last entry" heuristic when it's missing.
            "turn_id": getattr(self._state, "active_turn_id", "") or "",
            "user_goal": user_goal.strip(),
            "files_read": files_read,
            "files_edited": files_edited,
            "edits": edits,
            "shell_commands": shell_commands,
            "outcome": outcome,
            "verdict_reason": verdict_reason,
            "blockers": last_error,
            "created_at": _time.time(),
        }

    def _render_fallback_summary(
        self,
        *,
        iteration: int,
        total_writes: int,
        termination_reason: str,
        same_file_edits: dict[str, int],
        messages: list,
        user_goal: str = "",
        tool_results: list[dict] | None = None,
    ) -> str:
        """Build a user-facing narrative when the model stopped silently.

        Called at turn end when ``total_prose_chars`` is below the
        threshold — i.e. the model did work but never narrated it.
        Walks the same counters / state the turn summary uses and
        produces a short Markdown block. Purely mechanical; no LLM call.

        Goal is NOT to be a great summary — it's to be better than
        silence. If a model is narrating well this path never fires.
        """
        # Reason → plain-English tag. Keep in sync with
        # ``_build_turn_summary``'s outcome_map but user-facing here.
        reason_map = {
            "model_stop":              "stopped on its own",
            "model_stop_with_answer":  "stopped on its own",
            "model_stop_after_nudge":  "stopped after a prompt to continue",
            "tasks_completed":         "marked all tasks complete",
            "max_iterations_reached":  "hit the iteration limit",
            "validation_error_streak": "broke on repeated malformed tool calls",
            "test_failure_streak":     "broke on repeated failing tests",
            "same_file_edit_break":    "broke on thrashing one file",
            "inspection_loop_break":   "broke on an inspection-only loop",
            "no_write_progress":       "broke with attempted writes all failing",
            "backend_error":           "hit a backend error",
        }
        reason_text = reason_map.get(
            termination_reason, termination_reason or "stopped",
        )

        # Collect file reads/writes from message history (same walk as
        # _build_turn_summary, condensed for display). Keeps this helper
        # free of coupling to _build_turn_summary's dict shape.
        reads: list[str] = []
        writes: list[str] = []
        seen_reads: set[str] = set()
        seen_writes: set[str] = set()
        last_err: str = ""
        for m in messages:
            if m.role == "tool":
                content = m.content or ""
                if content.startswith("ERROR:"):
                    last_err = content[len("ERROR:"):].strip()[:200]
            if m.role != "assistant":
                continue
            for tc in (getattr(m, "tool_calls", None) or []):
                name = ""
                args: dict = {}
                if isinstance(tc, dict):
                    name = tc.get("function", {}).get("name", "") or tc.get("name", "")
                    args = tc.get("function", {}).get("arguments") or tc.get("input") or {}
                    if isinstance(args, str):
                        try:
                            import json as _json
                            args = _json.loads(args)
                        except Exception:
                            args = {}
                if name == "file_read":
                    p = (args or {}).get("path", "")
                    if p and p not in seen_reads:
                        reads.append(p)
                        seen_reads.add(p)
                elif name in ("file_write", "code_edit", "code_edit_batch"):
                    p = (args or {}).get("path", "")
                    if p and p not in seen_writes:
                        writes.append(p)
                        seen_writes.add(p)

        lines: list[str] = []
        lines.append(
            "\n\n---\n_(I didn't narrate this turn. Here's what I can "
            "reconstruct from my tool calls.)_\n"
        )
        lines.append(f"**Result:** {reason_text}.")
        lines.append(
            f"**Activity:** {iteration} iteration(s), "
            f"{self._state.tool_calls_made} tool call(s), "
            f"{total_writes} file write(s)."
        )
        if self._is_environment_audit_turn(user_goal):
            lines.extend(self._render_environment_provenance_lines())
        if reads:
            shown = reads[:5]
            suffix = f" (+{len(reads) - 5} more)" if len(reads) > 5 else ""
            lines.append(f"**Read:** {', '.join(shown)}{suffix}")
        if writes:
            lines.append(f"**Edited:** {', '.join(writes[:5])}")
        if same_file_edits:
            thrashed = [
                f"{p} (×{n})" for p, n in same_file_edits.items() if n >= 3
            ]
            if thrashed:
                lines.append(f"**Repeated edits:** {', '.join(thrashed[:3])}")

        # Repeated soft failures from state — rendered same as the
        # sticky reminder section but flattened into one line here.
        failures = [
            f for f in (self._state.recent_tool_failures or [])
            if (f.get("count") or 0) >= 2
        ]
        if failures:
            fail_bits = [
                f"{f.get('tool', '?')} {f.get('target') or ''} "
                f"(×{int(f.get('count') or 1)})"
                for f in failures[:3]
            ]
            lines.append(f"**Repeated failures:** {'; '.join(fail_bits)}")

        if last_err:
            lines.append(f"**Last error:** {last_err}")

        lines.extend(
            self._render_verifier_fallback_lines(
                tool_results=tool_results,
                writes=writes,
                last_err=last_err,
            ),
        )

        # Termination-reason-specific recovery hint. For inspection-loop
        # and same-file thrashing breaks the user needs to know what
        # action would unstick the agent — not just "try again harder".
        # Keep these short and concrete; they're the last thing the
        # user reads before deciding whether to retry.
        suggest_map = {
            "inspection_loop_break": (
                "The task looked like a CREATE/BUILD/RUN action, but "
                "the agent kept inspecting instead of executing. Try a "
                "more explicit phrasing — name the specific command or "
                "file, e.g. 'run `docker build -t X .`' or 'create a "
                "file X.py containing Y'."
            ),
            "same_file_edit_break": (
                "The agent thrashed repeatedly on one file. Re-read the "
                "file's full current state yourself before retrying, or "
                "describe the specific change (start/end lines, exact "
                "replacement) rather than leaving interpretation to the "
                "model."
            ),
            "validation_error_streak": (
                "The model kept emitting malformed tool calls. This is "
                "usually a model-capability issue — try a larger / "
                "stronger model (Claude / GPT-4 class), or break the "
                "task into smaller, more explicit steps."
            ),
            "test_failure_streak": (
                "Tests kept failing with no passes. Look at the last "
                "test output above; the fix usually isn't in the code "
                "under test but in the test setup / fixtures / "
                "environment. Consider running the failing test by "
                "hand first."
            ),
            "action_stagnation": (
                "The agent kept calling the same tool with slightly "
                "different arguments. It's stuck exploring the "
                "parameter space. Switch to a different approach: name "
                "the outcome you want, not the command you think should "
                "produce it."
            ),
            "no_write_progress": (
                "The agent attempted edits (code_edit / file_write) "
                "repeatedly but none landed — search-blocks went stale, "
                "idempotence guards fired, or validation errors blocked "
                "every attempt. Re-read the target file(s) yourself to "
                "see their actual current state, then either hand the "
                "exact text to replace or describe the outcome you want "
                "(not the edit) and let the agent plan fresh."
            ),
            "max_iterations_reached": (
                "Hit the iteration cap. The task may be larger than "
                "one turn should cover; break it into phases and run "
                "them sequentially. Or bump "
                "``AUGMENTUM_CODER_MAX_ITERS`` if the work is "
                "legitimately this long."
            ),
            "backend_error": (
                "The backend raised mid-turn. Check the logs for the "
                "underlying cause (timeout / rate-limit / server "
                "error) and retry."
            ),
        }
        suggestion = suggest_map.get(termination_reason)
        if suggestion:
            lines.append(f"\n**Suggested next step:** {suggestion}")
        else:
            lines.append(
                "\nThe model didn't produce a narrative summary. Consider "
                "rerunning with a stronger model, rephrasing the task more "
                "concretely, or asking me to explain what it tried."
            )
        return "\n".join(lines) + "\n"

    @staticmethod
    def _mutation_paths(tool_name: str, tool_input: dict) -> list[str]:
        """Return the list of file paths a mutation tool touched.

        All three mutating tools (``file_write``, ``code_edit``,
        ``code_multi_edit``) today take a single ``path`` — returning
        ``[path]`` is correct for all three. Kept as a helper so if we
        later add a true multi-file edit tool (patch-style), callers
        don't have to change: this function absorbs the shape drift.
        """
        path = (tool_input or {}).get("path", "")
        return [path] if path else []

    @staticmethod
    def _is_test_file(path: str) -> bool:
        """Heuristic: does ``path`` look like a test file worth auto-running?

        Covers the conventions we see in real repos:

          * ``tests/...`` / ``test/...`` — directory convention
          * ``test_*.py`` / ``*_test.py`` — pytest / unittest
          * ``*.test.js`` / ``*.test.ts`` / ``*.spec.js`` / ``*.spec.ts``
            — Jest / Vitest convention
          * ``*_test.go`` — Go testing
          * ``*.spec.rb`` — RSpec

        Returns False for anything that doesn't match — no speculative
        runs, no false positives that would slow down the loop.
        """
        if not path:
            return False
        # Normalise slashes for Windows paths that may leak in via tests.
        p = path.replace("\\", "/").lower()
        basename = p.rsplit("/", 1)[-1]

        # Directory-based conventions. The "/" or startswith checks both
        # handle root-relative (``tests/foo.py``) and nested paths
        # (``pkg/tests/foo.py``). Without startswith we'd miss root
        # test dirs, which is the common case in small projects.
        in_test_dir = (
            "/tests/" in p or p.startswith("tests/")
            or "/test/" in p or p.startswith("test/")
            or "/__tests__/" in p or p.startswith("__tests__/")
        )
        if in_test_dir:
            # Exclude fixtures and conftest — pytest collects those
            # implicitly so running them directly adds nothing.
            if basename in ("conftest.py", "__init__.py"):
                return False
            if "/fixtures/" in p or "/data/" in p:
                return False
            return True

        # Name-based conventions
        if basename.startswith("test_") and basename.endswith(".py"):
            return True
        if basename.endswith("_test.py"):
            return True
        if basename.endswith("_test.go"):
            return True
        if basename.endswith(".spec.rb"):
            return True
        for ext in (".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"):
            if basename.endswith(f".test{ext}") or basename.endswith(f".spec{ext}"):
                return True
        return False

    async def _run_focused_test(self, file_path: str) -> str:
        """Run just the tests in ``file_path`` and return a short report.

        Scoped to the single file by design — the full ``test_run`` tool
        is still available for suite-wide runs. Keeping this narrow
        means we can default the timeout to ~30s instead of 5min, so
        the auto-run doesn't stall the loop when a test hangs.

        Returns ``""`` when nothing worth showing came back. A genuine
        failure returns the summary line + up to ~800 chars of context
        so the model can react without a separate tool call.
        """
        if not self._container_manager or not file_path:
            return ""

        ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
        cmd: str = ""

        if ext == "py":
            # --tb=short keeps tracebacks readable; -x stops on first
            # failure so we don't drown the output; 2>&1 merges stderr.
            # ``command -v pytest`` gate means we fail closed on
            # containers that don't have pytest installed (no noise).
            cmd = (
                f"command -v pytest >/dev/null 2>&1 && "
                f"cd /workspace && "
                f"timeout 30 python3 -m pytest {file_path} -x --tb=short "
                f"--no-header 2>&1 | tail -40 || true"
            )
        elif ext in ("js", "jsx", "ts", "tsx", "mjs", "cjs"):
            # Jest pattern: npx jest <path>. Vitest: npx vitest run <path>.
            # Fall through to the first one that exists. ``|| true`` at
            # the end keeps exit code 0 so bash doesn't propagate a
            # failure that's legitimately a test failure.
            cmd = (
                f"cd /workspace && "
                f"if command -v npx >/dev/null 2>&1; then "
                f"  timeout 30 npx jest --testPathPattern='{file_path}' 2>&1 | tail -40 "
                f"    || timeout 30 npx vitest run {file_path} 2>&1 | tail -40 "
                f"    || true; "
                f"fi"
            )
        elif ext == "go":
            # Run only tests in the file's package, filtered by the
            # file's basename so we don't run the whole package.
            pkg_dir = file_path.rsplit("/", 1)[0] if "/" in file_path else "."
            cmd = (
                f"cd /workspace && "
                f"timeout 30 go test -short {pkg_dir} 2>&1 | tail -30 || true"
            )
        else:
            return ""

        try:
            output = await self._container_manager._run_command(
                self._workspace_id, ["bash", "-c", cmd], timeout=35.0,
            )
        except Exception:
            # Test run is best-effort — never break the tool-call chain.
            log.debug("coder.focused_test_failed", exc_info=True)
            return ""

        output = (output or "").strip()
        if not output:
            return ""

        # Summary heuristic: surface pytest/jest/go summary lines.
        # Pytest: "1 passed in 0.12s" / "1 failed, 2 passed".
        # Jest:   "Tests:       2 passed, 1 failed".
        # Go:     "ok  pkg  0.123s" / "FAIL pkg  0.123s".
        summary_line = ""
        for line in output.splitlines():
            low = line.strip().lower()
            if any(tok in low for tok in (
                "passed", "failed", "error", " ok ", "=== run",
                "PASS", "FAIL",
            )):
                if any(k in low for k in ("passed", "failed", "error", "ok ")):
                    summary_line = line.strip()
                    break

        prefix = f"{summary_line}\n" if summary_line else ""
        return (prefix + output)[:800]

    async def _run_lint_check(self, file_path: str) -> str:
        """Run a quick lint check on a file after editing.

        Auto-detects the appropriate linter based on file extension.
        Returns lint output if there are errors, empty string if clean.
        """
        if not self._container_manager or not file_path:
            return ""

        ext = file_path.rsplit(".", 1)[-1] if "." in file_path else ""
        lint_cmd = ""

        if ext == "py":
            # Try ruff first (fast), fall back to python -m py_compile (always available)
            lint_cmd = (
                f"ruff check --no-fix --output-format text {file_path} 2>/dev/null || "
                f"python3 -m py_compile {file_path} 2>&1"
            )
        elif ext in ("js", "jsx", "ts", "tsx"):
            lint_cmd = f"npx --yes eslint --no-eslintrc --format compact {file_path} 2>/dev/null || node -c \"require('fs').readFileSync('{file_path}','utf8')\" 2>&1"
        elif ext in ("json",):
            lint_cmd = f"python3 -c \"import json; json.load(open('{file_path}'))\" 2>&1"
        elif ext in ("yaml", "yml"):
            lint_cmd = f"python3 -c \"import yaml; yaml.safe_load(open('{file_path}'))\" 2>&1"
        elif ext in ("rs",):
            lint_cmd = f"rustc --edition 2021 --crate-type lib {file_path} 2>&1 | head -20"
        elif ext in ("c", "cpp", "h"):
            lint_cmd = f"gcc -fsyntax-only {file_path} 2>&1 | head -20"

        if not lint_cmd:
            return ""

        try:
            output = await self._container_manager._run_command(
                self._workspace_id, ["bash", "-c", lint_cmd], timeout=10.0,
            )
            # Only return if there are actual errors
            output = output.strip()
            if output and ("error" in output.lower() or "warning" in output.lower()
                          or "syntaxerror" in output.lower() or "traceback" in output.lower()):
                return output[:500]  # Cap at 500 chars
        except Exception as exc:
            log.debug("coder_lint_check_failed", error=str(exc))

        return ""

    async def _get_workspace_guide(self) -> str:
        """Read the workspace guide and detect project characteristics.

        Returns the guide text (from .augmentum/workspace.md) plus a brief
        dynamic context block describing the detected project type, test
        framework, and build system.  Result is cached for the session.
        """
        if self._cached_guide is not None:
            return self._cached_guide

        if not self._container_manager:
            from augmentum.coder.prompts import WORKSPACE_GUIDE
            self._cached_guide = WORKSPACE_GUIDE
            return self._cached_guide

        parts: list[str] = []

        # Read the workspace guide (user may have customized it)
        try:
            content = await self._container_manager._run_command(
                self._workspace_id,
                ["cat", "/workspace/.augmentum/workspace.md"],
                timeout=3.0,
            )
            if content.strip():
                parts.append(content.strip())
        except Exception:
            # Fallback to default if file doesn't exist
            from augmentum.coder.prompts import WORKSPACE_GUIDE
            parts.append(WORKSPACE_GUIDE)

        # Dynamic project detection — append brief context
        try:
            detect_script = (
                "cd /workspace && "
                "echo '## Project Detection' && "
                # Test framework
                "if [ -f pytest.ini ] || [ -f pyproject.toml ] || [ -d tests ]; then "
                "  echo '- Python tests detected. Run: `python3 -m pytest`'; "
                "elif [ -f package.json ] && grep -q '\"test\"' package.json 2>/dev/null; then "
                "  echo '- Node tests detected. Run: `npm test`'; "
                "elif [ -f Cargo.toml ]; then "
                "  echo '- Rust project. Run: `cargo test`'; "
                "elif [ -f go.mod ]; then "
                "  echo '- Go project. Run: `go test ./...`'; "
                "fi && "
                # Package manager
                "if [ -f requirements.txt ]; then echo '- Python deps: `pip install -r requirements.txt`'; "
                "elif [ -f pyproject.toml ]; then echo '- Python project: `pip install -e .`'; "
                "elif [ -f package.json ]; then echo '- Node deps: `npm install`'; "
                "fi && "
                # Build system
                "if [ -f Makefile ]; then echo '- Has Makefile. Run `make` for targets.'; "
                "elif [ -f justfile ]; then echo '- Has justfile. Run `just` for targets.'; "
                "elif [ -f docker-compose.yml ] || [ -f compose.yaml ]; then echo '- Has Docker Compose.'; "
                "fi && "
                # Empty workspace check
                "if [ $(ls -A /workspace 2>/dev/null | grep -v '^\\.augmentum$' | grep -v '^\\.git$' | wc -l) -eq 0 ]; then "
                "  echo '- Empty workspace. Ready for a new project.'; "
                "fi"
            )
            detection = await self._container_manager._run_command(
                self._workspace_id,
                ["bash", "-c", detect_script],
                timeout=5.0,
            )
            detection = detection.strip()
            if detection and "## Project Detection" in detection:
                parts.append(detection)
        except Exception:
            log.debug("workspace_detection_failed", exc_info=True)

        # Priming tree (Sprint 1): workspace profile facts.
        # The coder_profile store accumulates conventions, runtime
        # facts, and recurring failure patterns over the life of a
        # workspace. Inject the highest-signal subset so the model
        # sees workspace-specific knowledge that the static guide and
        # bash detection block can't provide. Falls through silently
        # when the store is missing, the user_id is empty, or no
        # entries exist (fresh workspace).
        try:
            user_id = getattr(self, "_user_id", "")
            if self._profile_store and user_id and self._workspace_id:
                entries = await self._profile_store.query_for_workspace(
                    user_id=user_id, workspace_id=self._workspace_id,
                )
                if entries:
                    from augmentum.coder.profile import render_profile_block
                    block = render_profile_block(entries)
                    if block:
                        parts.append(block)
        except Exception:
            log.debug("workspace_profile_inject_failed", exc_info=True)

        self._cached_guide = "\n\n".join(parts)
        return self._cached_guide

    async def _get_workspace_context(self) -> str:
        """Gather lightweight workspace context for the agent.

        Returns a brief summary: top-level files + git status (if a repo).
        Fast — two quick container exec calls.
        """
        if not self._container_manager:
            return ""

        parts: list[str] = []
        try:
            # Top-level file listing
            entries = await self._container_manager.file_list(
                self._workspace_id, "/workspace",
            )
            if entries:
                names = [
                    (f"{e.name}/" if e.is_dir else e.name)
                    for e in entries
                    if not e.name.startswith('.') or e.name in ('.gitignore', '.env.example')
                ]
                if names:
                    parts.append("Files in /workspace:\n" + ", ".join(names))
        except Exception as exc:
            log.debug("coder_workspace_files_list_failed", workspace_id=self._workspace_id, error=str(exc))

        try:
            # Git status (if it's a repo)
            git_status = await self._container_manager._run_command(
                self._workspace_id,
                ["bash", "-c", "cd /workspace && git rev-parse --is-inside-work-tree 2>/dev/null && git log --oneline -3 2>/dev/null && echo '---' && git status -s 2>/dev/null"],
            )
            if git_status.strip().startswith("true"):
                parts.append("Git:\n" + git_status.strip().replace("true\n", ""))
        except Exception as exc:
            log.debug("coder_workspace_git_status_failed", workspace_id=self._workspace_id, error=str(exc))

        return "\n\n".join(parts)


    # ------------------------------------------------------------------
    # Shared helpers used by _act_canonical and _act_hybrid
    # ------------------------------------------------------------------

    async def _get_workspace_buddy_model(self) -> str:
        """Look up the heavyweight escalation model.

        Resolution order:
          1. Per-workspace ``bug_finder_verifier_model`` (HVY button
             on the coder toolbar). Workspace-scoped — overrides the
             global slot for projects that want a different escalation
             target than the user's normal one.
          2. Global ``settings.heavyweight_model`` (Settings → Models
             → Heavyweight, added 2026-05-31). Acts as the default so
             a user who configures it once gets escalation in every
             new workspace without per-workspace setup. Same slot
             consumed by ``resolve_model_for_role("heavyweight", ...)``
             everywhere else (Bug Finder fallback, future
             ``/second-opinion``, narrative summariser escalation).

        Returns the resolved model name (empty string when neither is
        set — the caller's contract is that an empty return disables
        stagnation auto-escalation; the existing recoverable-error
        pill still surfaces so the user keeps control).
        """
        if self._container_manager is not None and self._workspace_id:
            try:
                info = await self._container_manager._get_workspace(self._workspace_id)
                per_workspace = (
                    getattr(info, "bug_finder_verifier_model", "") or ""
                ).strip()
                if per_workspace:
                    return per_workspace
            except Exception as exc:
                log.debug(
                    "coder.buddy_lookup_failed",
                    workspace_id=self._workspace_id, error=str(exc),
                )
                # Fall through to the global slot — even on lookup
                # failure, the global default is still useful.

        # Live-read so a UI change applies on the next stagnation
        # event without restarting the handler.
        try:
            from augmentum.config import settings as _settings
            return (getattr(_settings, "heavyweight_model", "") or "").strip()
        except Exception:
            return ""

    async def _stream_and_parse(
        self,
        request: InternalChatRequest,
        messages: list,
        tool_schemas,
        tool_map: dict,
        tier,
        iteration: int,
        *,
        progress_out: list | None = None,
        progress_phase: str = "executing",
        tool_choice: str | dict | None = None,
        chat_template_kwargs: dict | None = None,
    ) -> tuple[str, list[dict], str, str, int | None, str]:
        """Run one LLM call; return (full_content, normalized_tool_calls, error_kind, thinking, status_code, error_message).

        Normalised tool calls are dicts shaped ``{id, name, input}``.
        ``error_kind`` is one of:

          * ``""`` — success
          * ``"transient"`` — retried per ``_RETRY_BACKOFF_S`` but the
            backend kept failing with a recoverable status (429, 5xx,
            connection blip). Caller should surface a Try Again
            affordance to the user rather than dead-ending the turn —
            the next attempt is likely to succeed.
          * ``"permanent"`` — backend returned a 4xx that retrying won't
            help (auth, validation, not-found). Surface as a hard error;
            Try Again would just hit the same wall.

        ``error_message`` is ``str(exc)[:500]`` from the last failed
        attempt — the exact backend reply text. Empty on success. The
        caller surfaces this verbatim to the user so they can act on it
        without grepping logs (e.g. "Backend returned 400: Messages with
        role 'tool' must be a response to a preceding message…").

        ``thinking`` is the concatenated reasoning trace the model emitted
        on this turn (empty string if the model didn't think). Required
        for DeepSeek-style providers that 400 in iteration N+1 if the
        prior assistant turn omits ``reasoning_content`` — see
        ``provider_profiles.py:accepts_reasoning_content``. Captured here
        so the caller can attach it to the assistant ``Message`` it
        appends to history.

        Never raises — backend failures populate ``error_kind``.

        ``progress_out``: if provided, streaming sub-state transition
        chunks (``awaiting_first_token`` → ``thinking`` → ``responding``)
        are appended in order. Caller must yield them to surface the
        transitions to the UI. Left None for synthesis / internal calls
        where the UI doesn't need to see this level of detail.

        ``tool_choice``: forwarded to the backend payload. ``None`` lets
        the model decide ("auto"). Native strategy normally leaves this
        unset, matching CLI-agent loops where the model chooses whether
        to call tools or finish with prose.
        """
        from dataclasses import replace as dataclass_replace

        from augmentum.modes.coder.chat_egress import (
            ReasoningRelay,
            StreamProgressTracker,
        )

        # dataclass_replace — NOT a fresh InternalChatRequest(...) — so
        # every field on the source request flows through automatically.
        # The explicit-field-list pattern that used to live here silently
        # dropped ``kv_session_key`` and ``kv_mode`` (set by the route
        # layer to the workspace_id / "coder") on every iteration,
        # defeating slot-affinity and forcing llama-server to re-prefill
        # from cold on each tool round-trip. Same bug class as
        # ``apply_preset`` (commit 731a96d) — see the warning in
        # ``models/base.py:78-82``. Per-iteration overrides
        # (``messages``, ``stream``, ``tools``, ``tool_choice``,
        # ``chat_template_kwargs``) are the only fields that legitimately
        # need to change per call. preserve_thinking deliberately flows
        # through from the source request rather than being overridden
        # here (see the 2026-05-30 Qwen 3.6 regression note in git).
        # Guarantee tool-call/result pairing before the request leaves the
        # building. Non-pairing-aware compaction or a persisted trailing
        # assistant call can orphan a ``tool_call_id``; strict providers
        # (DeepSeek direct) 400 the whole turn on that. ``_reconcile_tool_
        # pairing`` is a no-op on well-formed history, so currently-working
        # turns are unaffected — see its docstring + the 2026-06-27 log dig.
        act_request = dataclass_replace(
            request,
            messages=_reconcile_tool_pairing(messages),
            stream=True,
            tools=tool_schemas,
            tool_choice=tool_choice,
            chat_template_kwargs=chat_template_kwargs,
        )

        content_parts: list[str] = []
        thinking_parts: list[str] = []
        _tc_acc: dict[int, dict] = {}
        # Track the LAST finish_reason seen on the stream. Backends
        # populate this on the terminal chunk (``done=True``). We watch
        # for ``"length"`` specifically — that's the provider's signal
        # that the response was cut off by max_tokens, which is the
        # ground truth for the truncation-during-tool-call failure
        # mode the assembly logic below has to handle.
        final_finish_reason: str | None = None

        # Multi-attempt retry with classifier-aware backoff. Transient
        # failures (429 queue full, 502/503/504 upstream, connection
        # blips) get the full 2s/5s/10s schedule — usually enough to
        # ride out a Cerebras queue surge or Anthropic 503 wave.
        # Permanent failures (4xx auth/validation/not-found) bail on
        # the first attempt — same request will keep failing.
        # See ``_classify_backend_error`` for the kind matrix.
        last_error: Exception | None = None
        last_error_kind: str = ""
        last_status: int | None = None
        max_attempts = len(_RETRY_BACKOFF_S)
        # Live reasoning relay: coalesced ``reasoning_delta`` chunks so the
        # user can watch the model think instead of staring at a status
        # pill. Relayed on EVERY attempt (unlike the transition tracker,
        # which is attempt-0 only) — the ``retrying`` chunk demarcates
        # attempts so the UI can close the interrupted block. Shared
        # across attempts; flush() resets its pending buffer.
        relay = (
            ReasoningRelay(phase=progress_phase, model=request.model)
            if progress_out is not None else None
        )
        for attempt in range(max_attempts):
            try:
                content_parts = []
                thinking_parts = []
                _tc_acc = {}
                final_finish_reason = None
                if progress_out is not None and attempt == 0:
                    progress = StreamProgressTracker()
                    progress_out.append(progress.begin(
                        phase=progress_phase, model=request.model,
                    ))
                else:
                    progress = None
                async for chunk in self._backend.chat_stream(act_request):
                    if progress is not None and progress_out is not None:
                        ev = progress.update(
                            chunk,
                            phase=progress_phase,
                            model=request.model,
                        )
                        if ev is not None:
                            # Pending reasoning text belongs BEFORE the
                            # sub-state transition it precedes (e.g. the
                            # last thought before "responding").
                            if relay is not None:
                                flushed = relay.flush()
                                if flushed is not None:
                                    progress_out.append(flushed)
                            progress_out.append(ev)
                    # Engine stage lifecycle (model_load / model_swap /
                    # slot_restore / prefill) — forward to the client so
                    # coder-stream.js → coder-progress.js can render a
                    # REAL progress bar (it polls /api/engine/v2/
                    # prefill_progress while a ``prefill`` stage is
                    # active — same pipeline chat uses). These chunks
                    # carry neither content_delta nor thinking_delta,
                    # so before this branch they were silently dropped
                    # here and the user stared at a frozen label for
                    # the whole prefill window (measured 2026-07-02,
                    # run …284079a9: 328s of dead air on a ~97k-token
                    # prompt while the engine emitted progress lines
                    # nobody consumed).
                    if progress_out is not None and chunk.augmentum and (
                        "stage_start" in chunk.augmentum
                        or "stage_complete" in chunk.augmentum
                    ):
                        stage_extra = {
                            k: v
                            for k, v in chunk.augmentum.items()
                            if k in ("stage_start", "stage_complete")
                        }
                        progress_out.append(emit(
                            phase=progress_phase,
                            status="stage",
                            model=request.model,
                            extra=stage_extra,
                        ))
                    if chunk.content_delta:
                        content_parts.append(chunk.content_delta)
                    # Capture the reasoning stream so it can round-trip
                    # back to the provider on the next iteration —
                    # DeepSeek-style APIs 400 the request otherwise.
                    if chunk.thinking_delta:
                        thinking_parts.append(chunk.thinking_delta)
                        if relay is not None and progress_out is not None:
                            ev = relay.add(chunk.thinking_delta)
                            if ev is not None:
                                progress_out.append(ev)
                    if chunk.finish_reason:
                        final_finish_reason = chunk.finish_reason
                    if chunk.augmentum and "tool_calls" in chunk.augmentum:
                        for tc_delta in chunk.augmentum["tool_calls"]:
                            idx = tc_delta.get("index", 0)
                            if idx not in _tc_acc:
                                _tc_acc[idx] = {
                                    "id": tc_delta.get("id", str(uuid.uuid4())),
                                    "name": "",
                                    "arguments_parts": [],
                                }
                            acc = _tc_acc[idx]
                            if tc_delta.get("id"):
                                acc["id"] = tc_delta["id"]
                            fn = tc_delta.get("function", {})
                            if fn.get("name"):
                                acc["name"] = fn["name"]
                            if fn.get("arguments"):
                                acc["arguments_parts"].append(fn["arguments"])
                if relay is not None and progress_out is not None:
                    flushed = relay.flush()
                    if flushed is not None:
                        progress_out.append(flushed)
                last_error = None
                last_error_kind = ""
                break
            except Exception as exc:
                # Surface whatever reasoning was mid-buffer before the
                # failure marker — batched, never dropped.
                if relay is not None and progress_out is not None:
                    flushed = relay.flush()
                    if flushed is not None:
                        progress_out.append(flushed)
                last_error = exc
                last_error_kind, last_status = _classify_backend_error(exc)
                log.warning(
                    "coder.stream_failed_attempt",
                    iteration=iteration, attempt=attempt + 1,
                    kind=last_error_kind, status=last_status,
                    error=str(exc),
                )
                # Permanent + quota failures: no point retrying. Permanent
                # 4xx will keep failing; quota windows don't reset within
                # the 17s budget. Bail immediately so the user gets the
                # error in <1s instead of after a wasted retry cycle.
                if last_error_kind in ("permanent", "quota"):
                    break
                # Transient — backoff before next attempt (unless this
                # WAS the last attempt). Emit a progress event so the UI
                # can surface "retrying in Ns..." instead of dead air.
                if attempt < max_attempts - 1:
                    wait_s = _RETRY_BACKOFF_S[attempt]
                    if progress_out is not None:
                        progress_out.append(InternalStreamChunk(
                            model=request.model,
                            augmentum={
                                "phase": progress_phase,
                                "status": "retrying",
                                "retry_wait_s": wait_s,
                                "retry_attempt": attempt + 2,
                                "retry_max": max_attempts,
                                "retry_status_code": last_status,
                            },
                        ))
                    await asyncio.sleep(wait_s)
        if last_error is not None:
            log.warning(
                "coder.stream_failed",
                iteration=iteration,
                kind=last_error_kind or "transient",
                status=last_status,
                error=str(last_error),
            )
            return "", [], (last_error_kind or "transient"), "", last_status, str(last_error)[:500]

        full_content = "".join(content_parts)
        full_thinking = "".join(thinking_parts)

        assembled: list[dict] = []
        for idx in sorted(_tc_acc.keys()):
            acc = _tc_acc[idx]
            if acc["name"]:
                args_str = "".join(acc["arguments_parts"])
                parse_error_raw: str | None = None
                truncation_reason: str | None = None
                if not args_str.strip():
                    # The model emitted a tool-call header (name set) but
                    # NO arguments JSON ever arrived on the wire. This is
                    # the truncation-mid-tool-call failure: the response
                    # ran out of output budget between the call header
                    # and the arguments body. The bare downstream error
                    # ("called without a 'path' argument") misleads the
                    # model into thinking it forgot a field, so it
                    # retries identically and loops. We flag it here so
                    # the dispatcher can short-circuit with a structured
                    # truncation error instead of running the tool with
                    # empty args. See docstring on
                    # ``_TRUNCATION_REASON_EMPTY_ARGS`` below.
                    args = {}
                    truncation_reason = _TRUNCATION_REASON_EMPTY_ARGS
                else:
                    try:
                        args = json.loads(args_str)
                    except json.JSONDecodeError:
                        # Fall back to {} so the tool still receives *something*
                        # (its validation layer will surface the missing args),
                        # but stash the raw unparseable string so the dispatch
                        # layer can prepend a "your JSON was malformed" hint.
                        # Without this surfacing, the model sees "path is
                        # required" when the real problem was truncated JSON —
                        # a misleading cue that burns iterations.
                        args = {}
                        parse_error_raw = args_str[:240]
                entry: dict = {
                    "id": acc["id"], "name": acc["name"], "input": args,
                }
                if parse_error_raw is not None:
                    entry["_parse_error_raw"] = parse_error_raw
                if truncation_reason is not None:
                    entry["_truncation_reason"] = truncation_reason
                    entry["_finish_reason"] = final_finish_reason or ""
                assembled.append(entry)

        tool_calls = assembled
        if not tool_calls and full_content:
            from augmentum.modes.analytical.tool_calling import parse_json_tool_calls
            parsed = parse_json_tool_calls(full_content, known_tools=set(tool_map))
            if parsed:
                tool_calls = [
                    {"id": str(uuid.uuid4()), "name": n, "input": a}
                    for n, a in parsed
                ]
            else:
                tool_calls = _extract_tool_calls_from_text(full_content)

        return full_content, tool_calls, "", full_thinking, None, ""

    async def _stream_and_parse_live(
        self,
        request: InternalChatRequest,
        messages: list,
        tool_schemas,
        tool_map: dict,
        tier,
        iteration: int,
        *,
        result_out: list,
        progress_phase: str = "executing",
        tool_choice: str | dict | None = None,
        chat_template_kwargs: dict | None = None,
    ):
        """Live-relay wrapper around :meth:`_stream_and_parse`.

        ``_stream_and_parse`` buffers its progress events in a list and
        the caller yields them AFTER the LLM call returns — fine for
        transition markers, useless for the live reasoning relay (the
        whole point is watching the model think WHILE it generates).
        This wrapper runs the call as a task with a queue-backed sink so
        events reach the client as they happen, without touching
        ``_stream_and_parse``'s contract (signature, 6-tuple return,
        retry semantics — tests await it directly and still can).

        Yields each progress/reasoning chunk as it is produced; when the
        underlying call completes, appends its result tuple to
        ``result_out`` (a caller-provided list — async generators can't
        return values). If the consumer is closed early (client
        disconnect), the inner task is cancelled so the backend stream
        isn't leaked.

        **Mid-reasoning steer interrupt** (``coder_steer_interrupt_reasoning``):
        while the model is still streaming reasoning (no committal
        content/tool output yet), a steer landing in the run inbox cancels
        the in-flight generation, folds the steer into ``messages`` as a
        ``[User interjected mid-turn]`` user turn, and re-runs the call so
        the model addresses the new content immediately — instead of the
        steer waiting for the next iteration boundary, which a stuck
        reasoning loop never reaches. The looping partial reasoning is
        discarded (never appended to history). Bounded per turn. No-op on
        the legacy in-request path (no broker) — those flows are unchanged.
        """
        # "Still thinking" (interruptible) vs "already producing output"
        # (leave alone — don't guillotine real work).
        _REASONING = frozenset({"reasoning_delta", "thinking", "awaiting_first_token"})
        _COMMITTAL = frozenset({"responding", "streaming", "tool_call", "tool_result"})
        _MAX_REENTRIES = 4  # cap so a steer storm can't spin the loop forever
        reentries = 0

        while True:
            queue: asyncio.Queue = asyncio.Queue()
            sentinel = object()

            class _QueueSink:
                """List-shaped adapter: ``progress_out.append`` → queue push."""
                __slots__ = ()

                @staticmethod
                def append(ev) -> None:
                    queue.put_nowait(ev)

            task = asyncio.create_task(self._stream_and_parse(
                request, messages, tool_schemas, tool_map, tier, iteration,
                progress_out=_QueueSink(),
                progress_phase=progress_phase,
                tool_choice=tool_choice,
                chat_template_kwargs=chat_template_kwargs,
            ))
            # done_callback fires on success, exception, AND cancellation —
            # the drain loop below always terminates. Bind q/s as defaults so
            # a late-firing callback from a cancelled task can't push into the
            # NEXT re-entry's queue (would look like a premature end-of-stream).
            task.add_done_callback(
                lambda _t, q=queue, s=sentinel: q.put_nowait(s)
            )
            interrupted = False
            committed = False
            try:
                while True:
                    ev = await queue.get()
                    if ev is sentinel:
                        break
                    _status = getattr(ev, "status", "")
                    if _status in _COMMITTAL:
                        committed = True
                    yield ev
                    # Interrupt ONLY on a reasoning chunk, only before any
                    # committal output, only when a steer just landed.
                    if (
                        not committed
                        and _status in _REASONING
                        and reentries < _MAX_REENTRIES
                        and self._steer_interrupt_pending()
                    ):
                        interrupted = True
                        break
                if not interrupted:
                    # Task is done here; .result() propagates any exception
                    # (contractually none — _stream_and_parse never raises).
                    result_out.append(task.result())
                    return
            finally:
                if not task.done():
                    task.cancel()
                    with contextlib.suppress(BaseException):
                        await task

            # Interrupted mid-reasoning. The generation is cancelled (finally,
            # above) so the model slot is freed and the looping partial
            # reasoning is discarded. Drain the steer(s), fold them into
            # history as user turns, tell the UI the steer landed, and re-run
            # generation with the redirected context.
            drained = self._drain_steer_for_interrupt()
            reentries += 1
            if not drained:
                # Raced: the steer was drained elsewhere between peek and here.
                # Re-run cleanly (no history change); the loop settles.
                continue
            for _s in drained:
                messages.append(Message(
                    role="user",
                    content=self._format_steer_content(_s),
                ))
            yield self._meta_chunk(
                phase=progress_phase, status="steer_delivered",
                model=request.model,
                extra={
                    "count": len(drained),
                    "iteration": iteration,
                    "mid_reasoning": True,
                },
            )

    def _append_assistant_to_history(
        self,
        messages: list,
        full_content: str,
        tool_calls: list[dict],
        tier,
        *,
        thinking: str = "",
    ) -> None:
        """Append the assistant turn to history in the shape the tier expects.

        ``thinking`` is the model's reasoning trace for this turn (only
        meaningful in NATIVE tier — pseudo-tier replays don't preserve
        it). Required so the next iteration's request carries
        ``reasoning_content`` back to providers like DeepSeek that
        enforce it (otherwise: HTTP 400, agent loop aborts).
        """
        if tier == ToolCallingTier.NATIVE:
            formatted_tc = None
            if tool_calls:
                formatted_tc = [{
                    "id": tc.get("id", str(uuid.uuid4())),
                    "type": "function",
                    "function": {
                        "name": tc.get("name", ""),
                        "arguments": json.dumps(tc.get("input", {})),
                    },
                } for tc in tool_calls]
            messages.append(Message(
                role="assistant",
                content=full_content or "",
                tool_calls=formatted_tc,
                thinking=thinking or None,
            ))
        else:
            final_content = (
                _strip_tool_json(full_content).strip()
                or full_content.strip()
                or "(executing tool)"
            )
            messages.append(Message(role="assistant", content=final_content))

    def _resolve_archive_conn(self):
        """Return the aiosqlite conn used by the turn archive, or None.

        Same lookup as ``_archive_turn_summary`` but pull-out so the
        per-strategy ``create_coder_tools`` callsites can pass it
        through to recall tools without duplicating the chain.
        Returns ``None`` when the state manager isn't wired (tests,
        cli paths) — recall tools then gracefully refuse.
        """
        if self._state_manager is None:
            return None
        return getattr(
            getattr(self._state_manager, "backend", None), "conn", None,
        )

    async def _persist_reply_if_orphaned(
        self, request: InternalChatRequest, final_text: str,
    ) -> None:
        """Write the completed turn's reply into the saved conversation
        when no attached client did it.

        Conversation persistence is client-driven (the coder UI POSTs its
        rendered history), which orphans the reply whenever the user left
        the surface mid-run: the broker keeps the run alive through the
        disconnect, but on return the saved conversation still ends at
        the unanswered user message and the turn LOOKS timed out.

        Orphan detection: the stored conversation is empty or its tail is
        a ``user`` message. When a client IS attached it renders the reply
        and saves the full rich history (tool cards and all) moments
        later — its save simply overwrites this plain-text version, so
        the append is safe in both worlds (last writer wins, and the last
        writer with a richer view is the client).
        """
        text = (final_text or "").strip()
        user_id = getattr(self, "_user_id", "") or ""
        workspace_id = getattr(self, "_workspace_id", "") or ""
        if not text or not user_id or not workspace_id:
            return
        conn = self._resolve_archive_conn()
        if conn is None:
            return
        from augmentum.state.coder_persistence import CoderPersistence
        persistence = CoderPersistence(conn)
        convo = await persistence.load_conversation(
            workspace_id, user_id=user_id,
        )
        # This turn's user message, in case even the send-time save never
        # landed (user left before the debounce fired).
        turn_user = ""
        for m in reversed(request.messages or []):
            if (m.get("role") if isinstance(m, dict) else getattr(m, "role", "")) == "user":
                turn_user = (
                    m.get("content") if isinstance(m, dict)
                    else getattr(m, "content", "")
                ) or ""
                break
        if isinstance(turn_user, list):  # multimodal part-list — text legs only
            turn_user = " ".join(
                p.get("text", "") for p in turn_user if isinstance(p, dict)
            ).strip()
        # Orphan test: is THIS turn already answered in the store? Scan the
        # recent tail for this turn's user message; "answered" means a
        # later assistant message follows it (an attached client rendered
        # and saved — its rich version may format the reply differently,
        # so we match on the user message, not the reply text). Tails:
        # (a) this turn's user msg is the LAST message — user save landed,
        # reply didn't → append assistant; (b) user msg found with an
        # assistant after it → skip; (c) user msg absent (empty store, or
        # a PRIOR turn's tail — the left-before-the-debounce case) →
        # append both.
        found_user_at = -1
        if turn_user:
            for i in range(len(convo) - 1, max(-1, len(convo) - 8), -1):
                m = convo[i] or {}
                if m.get("role") == "user" and (m.get("content") or "") == turn_user:
                    found_user_at = i
                    break
        if found_user_at >= 0:
            answered = any(
                (convo[j] or {}).get("role") == "assistant"
                for j in range(found_user_at + 1, len(convo))
            )
            if answered:
                return  # a client already saved the answered turn
        elif turn_user:
            convo.append({"role": "user", "content": turn_user})
        convo.append({"role": "assistant", "content": text})
        await persistence.save_conversation(
            workspace_id, convo, user_id=user_id,
        )
        log.info(
            "coder.orphaned_reply_persisted",
            workspace_id=workspace_id,
            chars=len(text),
        )

    async def _archive_turn_summary(self, summary: dict) -> None:
        """Persist a finished turn to ``coder_turn_archive``.

        Mirror call sits next to ``_state.add_turn_summary(summary)``
        in each strategy (native/hybrid/canonical/legacy). FIFO is for
        the in-prompt ``<prior_turns>`` block; this is the durable
        long-tail so future sessions and the inspector timeline can
        reach beyond the cap-10 horizon.

        Best-effort — failures log and continue. Empty user_id or
        workspace_id short-circuit (Phase 1 is workspace-scoped only).
        """
        from augmentum.config import settings as _settings
        if not bool(getattr(_settings, "coder_archive_enabled", True)):
            return

        if not isinstance(summary, dict):
            return

        user_id = getattr(self, "_user_id", "") or ""
        workspace_id = (
            getattr(self._state, "workspace_id", "")
            or getattr(self, "_workspace_id", "")
            or ""
        )
        if not user_id or not workspace_id:
            return

        # Pull the SQLite conn from the state manager — same pattern
        # other persistence hooks use (``_start_turn_ledger`` etc).
        if self._state_manager is None:
            return
        conn = getattr(
            getattr(self._state_manager, "backend", None), "conn", None,
        )
        if conn is None:
            log.debug("coder_turn_archive.skip_no_conn")
            return

        try:
            from augmentum.coder.turn_archive import append_turn
            await append_turn(
                conn,
                user_id=user_id,
                workspace_id=workspace_id,
                run_id=getattr(self._state, "active_run_id", "") or "",
                turn_id=summary.get("turn_id", "") or "",
                user_goal=str(summary.get("user_goal", "") or ""),
                outcome=str(summary.get("outcome", "") or ""),
                verdict_reason=str(summary.get("verdict_reason", "") or ""),
                blockers=str(summary.get("blockers", "") or ""),
                files_read=list(summary.get("files_read", []) or []),
                files_edited=list(summary.get("files_edited", []) or []),
                shell_commands=list(summary.get("shell_commands", []) or []),
                edits=list(summary.get("edits", []) or []),
                summary=str(summary.get("compaction_digest", "") or summary.get("user_goal", "") or ""),
                tokens_in=int(summary.get("tokens_in", 0) or 0),
                tokens_out=int(summary.get("tokens_out", 0) or 0),
                event_time=int(summary.get("created_at", 0) or 0) or None,
            )
        except Exception as exc:
            log.debug(
                "coder_turn_archive.write_hook_failed",
                error=str(exc)[:160],
            )

        # Ingest-all-work: mirror an APPLIED turn (files were edited) into the
        # self-edit archive as a source='coder' row, so the never-pruned
        # lineage learns from coder work too — not only the engine's own
        # autonomous attempts. Flag-gated (default OFF); writes to the isolated
        # growth.db, never this handler's main conn. Best-effort, independent
        # of the turn-archive write above.
        try:
            if (bool(getattr(_settings, "selfedit_ingest_coder_enabled", False))
                    and summary.get("files_edited")):
                from augmentum.proxy import server as _server
                _app = getattr(_server, "app", None) or getattr(_server, "_app", None)
                if _app is not None:
                    from augmentum.selfedit.growth_db import get_growth_conn
                    gconn = await get_growth_conn(_app.state)
                    if gconn is not None:
                        from augmentum.selfedit.ingest import ingest_coder_turn
                        await ingest_coder_turn(
                            gconn, user_id=user_id,
                            turn_id=str(summary.get("turn_id", "") or ""),
                            user_goal=str(summary.get("user_goal", "") or ""),
                            outcome=str(summary.get("outcome", "") or ""),
                            files_edited=list(summary.get("files_edited", []) or []),
                            workspace_id=workspace_id,
                        )
        except Exception as exc:
            log.warning(
                "selfedit_ingest_coder_failed",
                error=str(exc)[:160],
            )

    async def _maybe_auto_observe(
        self,
        *,
        tool_name: str,
        tool_input: dict,
        tool_result,
    ) -> None:
        """Extract + persist pattern-derived observations.

        Wraps ``augmentum.coder.auto_observer.extract_auto_observations``
        with per-turn cap enforcement and best-effort persistence — any
        failure logs and returns; never raises. The agent-facing
        ``observe`` tool keeps working in parallel; this is an
        independent pump so an empty ledger isn't the default for
        sessions where the model never reached for the tool.
        """
        # Per-turn cap so a fan-out of 20 same-shape calls (e.g., an
        # apply_patch that edits 20 config files) can't dominate the
        # ledger. Counter lives on the state so it resets per turn.
        cap = 5
        used = getattr(self._state, "_auto_observe_used_this_turn", 0)
        if used >= cap:
            return

        try:
            from augmentum.coder.auto_observer import extract_auto_observations
        except Exception:
            return

        # Pull the workspace id from the active turn's context; the
        # observer module only needs the container manager to persist,
        # which we have on the handler. Fall back to ``self._workspace_id``
        # set at handler construction in case ``state.workspace_id``
        # wasn't mirrored yet on the first turn.
        workspace_id = (
            getattr(self._state, "workspace_id", "")
            or getattr(self, "_workspace_id", "")
            or ""
        )
        cm = getattr(self, "_container_manager", None)
        if not workspace_id or cm is None:
            return

        active_turn_id = getattr(self._state, "active_turn_id", "") or "turn"
        source_tag = f"auto:{active_turn_id}"

        try:
            observations = extract_auto_observations(
                tool_name=tool_name,
                tool_input=tool_input,
                tool_result_success=bool(tool_result.success),
                tool_result_output=tool_result.output or "",
                tool_result_metadata=tool_result.metadata or {},
                source_tag=source_tag,
            )
        except Exception as exc:
            log.debug("coder_auto_observe_extract_failed", error=str(exc)[:160])
            return

        if not observations:
            return

        from augmentum.coder.observations import append_observation

        budget = cap - used
        for obs in observations[:budget]:
            try:
                ok = await append_observation(cm, workspace_id, obs)
            except Exception as exc:
                log.debug(
                    "coder_auto_observe_append_failed",
                    category=obs.category, error=str(exc)[:160],
                )
                ok = False
            if ok:
                used += 1
                # Mirror to in-memory state cache if the kernel exposes
                # one (best-effort — failure to mirror is fine because
                # the next turn re-reads the ledger from disk).
                try:
                    self._state._auto_observe_used_this_turn = used
                except Exception:
                    log.debug("auto_observe_counter_mirror_failed", exc_info=True)

    def _append_tool_result_to_history(
        self,
        messages: list,
        tool_id: str,
        tool_name: str,
        tool_result,
        tier,
    ) -> None:
        """Append a tool_result in the shape the tier expects."""
        result_content = (
            tool_result.output if tool_result.success
            else f"ERROR: {tool_result.error}"
        )
        # Window-scaled char cap (see _scaled_output_cap_chars): on small
        # compaction ceilings a tool's own 50k-char clip is still ~12.5k
        # tokens — enough to blow a 16k window — so re-clip here to ~25% of
        # the effective window. No-op (byte-identical) on big-window
        # deployments where the derived cap sits at the 50k default.
        result_content = _truncate_output(
            result_content or "",
            _scaled_output_cap_chars(
                getattr(self, "_coder_compact_token_limit", 0),
            ),
        )
        # Universal hard byte-cap (see _clamp_tool_result_bytes): the single
        # point every tool_result passes through, so a result that escaped its
        # tool's own truncation can never reach the transport limit as a dead
        # fetch — it degrades to a usable truncated result + recourse hint.
        result_content = _clamp_tool_result_bytes(result_content or "", tool_name)
        if tier == ToolCallingTier.NATIVE:
            messages.append(Message(
                role="tool",
                content=result_content or "",
                tool_call_id=tool_id,
            ))
        else:
            text = f"[Tool result: {tool_name}]\n{result_content}"
            if messages and messages[-1].role == "user":
                messages[-1] = Message(
                    role="user",
                    content=messages[-1].content + "\n\n" + text,
                )
            else:
                messages.append(Message(role="user", content=text))

    def _normalize_tool_call(self, tc: dict) -> tuple[str, str, dict]:
        """Normalise a raw tool call to (id, name, input)."""
        tool_id = tc.get("id") or str(uuid.uuid4())
        tool_name = (
            tc.get("name")
            or tc.get("function", {}).get("name", "")
        )
        raw_input = tc.get("input") or tc.get("function", {}).get("arguments", {})
        if isinstance(raw_input, str):
            try:
                tool_input = json.loads(raw_input)
            except json.JSONDecodeError:
                tool_input = {}
        else:
            tool_input = raw_input or {}
        return tool_id, tool_name, tool_input

    async def _check_tool_permission(
        self, tool_name: str, tool_input: dict,
    ) -> tuple[bool, str]:
        """Gate potentially-destructive tools behind the permission policy.

        Returns ``(allowed, denial_reason)``. Always True under the default
        ``auto`` policy. Under ``confirm_mutations``, the handler's
        ``permission_callback`` is awaited for tools in
        ``_APPROVAL_REQUIRED_TOOLS``; a missing callback means deny
        (safe-by-default) so the model still gets a structured refusal
        rather than silently proceeding.
        """
        requires_confirmation = (
            tool_name in _ALWAYS_CONFIRM_TOOLS
            or (
                _CODER_PERMISSIONS == "confirm_mutations"
                and tool_name in _APPROVAL_REQUIRED_TOOLS
            )
        )
        if not requires_confirmation:
            return True, ""
        if self._permission_callback is None:
            return False, (
                f"Tool {tool_name!r} requires user approval but no "
                f"permission_callback is registered; denied by default."
            )
        try:
            approved = await self._permission_callback(tool_name, tool_input)
        except Exception as exc:
            log.warning("coder.permission_callback_error",
                        tool_name=tool_name, error=str(exc))
            return False, f"Permission check raised: {exc}"
        if approved:
            return True, ""
        return False, f"User denied {tool_name!r} for this invocation."

    async def _run_tool_tracked(
        self,
        tc: dict,
        tool_map: dict,
        tier,
        messages: list,
        model: str,
        counters: dict,
    ) -> AsyncIterator[InternalStreamChunk]:
        """Execute one tool with full post-hooks; stream meta chunks.

        Writes are tallied into ``counters['writes']`` so the hybrid
        continuation judge can tell whether the last N iterations made
        progress. Uses the full verification chain
        (``_execute_tool_with_verification``): existence check, lint,
        syntax check, git checkpoint.

        Honours the per-tool permission policy — when denied, synthesises
        a tool_result pointing at the refusal reason so the model can
        react (ask the user for more context, try a read-only approach,
        or stop) without the call silently failing.
        """
        from augmentum.tools.base import ToolResult as _TR

        tool_id, tool_name, tool_input = self._normalize_tool_call(tc)

        yield self._meta_chunk(
            phase="executing", status="tool_call", model=model,
            extra={"tool_call": {
                "id": tool_id, "tool": tool_name, "input": tool_input,
            }},
        )

        latest_input, user_goal = _extract_goal_split(messages)
        goal_text = user_goal or latest_input
        turn_intent = _classify_turn_intent(latest_input, goal_text=goal_text)
        # Mirror onto state so downstream prompt builders see the same
        # classification this read-only nudge path uses.
        self._state.current_intent = turn_intent
        execution_requested = turn_intent.explicit_execution
        # Read-only NUDGE (was a hard refusal pre-PR-2). When the turn
        # classifies as INSPECT/REVIEW and the model reaches for a
        # mutating tool without an explicit "run/build/test" signal in
        # the user's text, emit an advisory system-reminder and LET THE
        # CALL PROCEED. Three reasons we don't block anymore:
        #  1. The classifier overfits — substring matches like "explain"
        #     in "explain what your recommendations do" don't actually
        #     mean the user wants a read-only turn for the WHOLE session.
        #  2. The model already self-corrects when it sees the nudge in
        #     the next iteration's reminder block, same as it self-
        #     corrected on the old hard refusal text.
        #  3. The container is sandboxed; blast radius is bounded. The
        #     three sibling refusals that defend against pathological
        #     loops (populated_repo_ask_user, root_workspace_reinspection,
        #     read_repeat) remain HARD and are intentionally untouched.
        # Counters and record_validation_error are NOT touched: this is
        # a nudge, not a malformed call, and the sticky "Recent blockers"
        # rollup should stay focused on real schema violations.
        if (
            goal_text
            and turn_intent.read_only_by_default
            and not execution_requested
            and not _is_explanatory_safe_tool_call(tool_name, tool_input)
        ):
            intent_label = (
                "review/audit"
                if turn_intent.kind == TurnIntentKind.REVIEW
                else "explanatory"
            )
            nudge = (
                f"Note: this turn started as a {intent_label} request and "
                f"you're calling {tool_name}, which mutates state. If the "
                "user explicitly asked you to run / build / test, proceed. "
                "Otherwise prefer inspection tools (file_read, code_grep, "
                "git status/diff) so the answer stays grounded in "
                "observation rather than side effects."
            )
            yield self._meta_chunk(
                phase="executing", status="read_only_nudge", model=model,
                extra={
                    "reminder": nudge,
                    "read_only_nudge": True,
                    "tool": tool_name,
                    "intent": intent_label,
                },
            )
        if tool_name == "ask_user":
            populated_repo_refusal = self._maybe_refuse_populated_repo_ask_user(
                tool_input=tool_input,
            )
            if populated_repo_refusal:
                tool_result = _TR(
                    success=False,
                    validation_error=True,
                    error=populated_repo_refusal,
                )
                yield self._meta_chunk(
                    phase="executing",
                    status="tool_result",
                    model=model,
                    extra={"tool_result": {
                        "id": tool_id,
                        "tool": tool_name,
                        "success": False,
                        "output_preview": populated_repo_refusal[:_preview_len(tool_name)],
                        "populated_repo_guard": True,
                    }},
                )
                self._append_tool_result_to_history(
                    messages, tool_id, tool_name, tool_result, tier,
                )
                counters["validation_errors"] = (
                    counters.get("validation_errors", 0) + 1
                )
                counters["tool_calls"] = counters.get("tool_calls", 0) + 1
                self._state.record_validation_error(
                    tool_name=tool_name,
                    error=populated_repo_refusal,
                )
                return

        if tool_name == "shell_exec":
            root_tree_refusal = self._maybe_refuse_root_workspace_reinspection(
                tool_name=tool_name,
                tool_input=tool_input,
            )
            if root_tree_refusal:
                tool_result = _TR(
                    success=False,
                    validation_error=True,
                    error=root_tree_refusal,
                )
                yield self._meta_chunk(
                    phase="executing",
                    status="tool_result",
                    model=model,
                    extra={"tool_result": {
                        "id": tool_id,
                        "tool": tool_name,
                        "success": False,
                        "output_preview": root_tree_refusal[:_preview_len(tool_name)],
                        "workspace_tree_guard": True,
                    }},
                )
                self._append_tool_result_to_history(
                    messages, tool_id, tool_name, tool_result, tier,
                )
                counters["validation_errors"] = (
                    counters.get("validation_errors", 0) + 1
                )
                counters["tool_calls"] = counters.get("tool_calls", 0) + 1
                self._state.record_validation_error(
                    tool_name=tool_name,
                    error=root_tree_refusal,
                )
                return

            verifier_refusal = self._maybe_refuse_verifier_shell_exec(
                command=str(tool_input.get("command") or ""),
                latest_input=latest_input,
                goal_text=goal_text,
                messages=messages,
            )
            if verifier_refusal:
                tool_result = _TR(
                    success=False,
                    validation_error=True,
                    error=verifier_refusal,
                )
                yield self._meta_chunk(
                    phase="executing",
                    status="tool_result",
                    model=model,
                    extra={"tool_result": {
                        "id": tool_id,
                        "tool": tool_name,
                        "success": False,
                        "output_preview": verifier_refusal[:_preview_len(tool_name)],
                        "verifier_guard": True,
                    }},
                )
                self._append_tool_result_to_history(
                    messages, tool_id, tool_name, tool_result, tier,
                )
                counters["validation_errors"] = (
                    counters.get("validation_errors", 0) + 1
                )
                counters["tool_calls"] = counters.get("tool_calls", 0) + 1
                self._state.record_validation_error(
                    tool_name=tool_name,
                    error=verifier_refusal,
                )
                return
        elif tool_name in {"dir_tree", "file_list", "shell_read"}:
            root_tree_refusal = self._maybe_refuse_root_workspace_reinspection(
                tool_name=tool_name,
                tool_input=tool_input,
            )
            if root_tree_refusal:
                tool_result = _TR(
                    success=False,
                    validation_error=True,
                    error=root_tree_refusal,
                )
                yield self._meta_chunk(
                    phase="executing",
                    status="tool_result",
                    model=model,
                    extra={"tool_result": {
                        "id": tool_id,
                        "tool": tool_name,
                        "success": False,
                        "output_preview": root_tree_refusal[:_preview_len(tool_name)],
                        "workspace_tree_guard": True,
                    }},
                )
                self._append_tool_result_to_history(
                    messages, tool_id, tool_name, tool_result, tier,
                )
                counters["validation_errors"] = (
                    counters.get("validation_errors", 0) + 1
                )
                counters["tool_calls"] = counters.get("tool_calls", 0) + 1
                self._state.record_validation_error(
                    tool_name=tool_name,
                    error=root_tree_refusal,
                )
                return

        allowed, denial_reason = await self._check_tool_permission(
            tool_name, tool_input,
        )
        if not allowed:
            tool_result = _TR(success=False, error=denial_reason)
            yield self._meta_chunk(
                phase="executing", status="tool_result",
                model=model,
                extra={"tool_result": {
                    "id": tool_id, "tool": tool_name,
                    "success": False,
                    "output_preview": denial_reason[:_preview_len(tool_name)],
                    "denied": True,
                }},
            )
            self._append_tool_result_to_history(
                messages, tool_id, tool_name, tool_result, tier,
            )
            return

        # Preemptive repeat-read refusal. Observed 2026-04-20 on Qwen 3.6:
        # weak models ignore the soft "Already inspected" sticky-reminder
        # signal and re-read the same file N times looking for traction
        # on a vague prompt. Return a synthetic tool_result instead of
        # running the container round-trip — unambiguous signal to act
        # on the data or ask the user. Cleared for a path on mutation
        # (see the mutation branch below), so "edit → verify" loops
        # aren't blocked.
        if (
            self._state.safeguards_enabled
            and tool_name in _PREEMPTIVE_REFUSAL_TOOLS
        ):
            prior = self._state.repeat_count(
                tool_name=tool_name, tool_input=tool_input,
            )
            if prior >= _READ_REPEAT_REFUSAL_CAP - 1:  # >=2 means this is the 3rd+
                refusal = (
                    f"Refused: you have already called {tool_name} with "
                    f"these arguments {prior}× this session without the "
                    "target file being modified. The result is in your "
                    "earlier tool_results — scroll back to reuse it. "
                    "If you need a different view, change the arguments "
                    "(offset, pattern, path). If you're stuck, explain "
                    "what you're trying to find or ask the user for "
                    "specifics. Running the same read again will NOT "
                    "return new information."
                )
                tool_result = _TR(
                    success=False,
                    validation_error=True,
                    error=refusal,
                )
                yield self._meta_chunk(
                    phase="executing", status="tool_result",
                    model=model,
                    extra={"tool_result": {
                        "id": tool_id, "tool": tool_name,
                        "success": False,
                        "output_preview": refusal[:_preview_len(tool_name)],
                        "preemptive_refusal": True,
                        "prior_count": prior,
                    }},
                )
                self._append_tool_result_to_history(
                    messages, tool_id, tool_name, tool_result, tier,
                )
                # Record this refusal the same way schema validation
                # errors are recorded so the sticky reminder and streak
                # breakers both respond to repeated thrashing.
                counters["validation_errors"] = (
                    counters.get("validation_errors", 0) + 1
                )
                counters["tool_calls"] = counters.get("tool_calls", 0) + 1
                self._state.record_validation_error(
                    tool_name=tool_name, error=refusal,
                )
                return

        # Truncation short-circuit: the assembly layer flagged that
        # this tool call's arguments JSON never arrived (response was
        # cut off mid-tool-call). Don't dispatch the tool — it would
        # only return a misleading "missing required arg" error that
        # the model misreads as "I forgot a field" and retries
        # identically, looping forever. Synthesize a structured
        # truncation result directly. See D1 in the cascade.
        truncation_reason = (
            tc.get("_truncation_reason") if isinstance(tc, dict) else None
        )
        if truncation_reason == _TRUNCATION_REASON_EMPTY_ARGS:
            from augmentum.tools.base import ToolResult as _TR
            tool_result = _TR(
                success=False,
                validation_error=True,
                error=_build_truncation_error(
                    tool_name=tool_name,
                    finish_reason=tc.get("_finish_reason", ""),
                ),
            )
            checkpoint_hash = None
        else:
            tool_result, checkpoint_hash, _ = await self._execute_tool_with_verification(
                tool_name=tool_name, tool_input=tool_input, tool_map=tool_map,
            )

        # If the upstream tool-call JSON was malformed, tell the model
        # that directly — otherwise it only sees the downstream "missing
        # required arg" error and retries the same broken emission.
        parse_error_raw = tc.get("_parse_error_raw") if isinstance(tc, dict) else None
        if parse_error_raw is not None and not tool_result.success:
            from augmentum.tools.base import ToolResult as _TR
            tool_result = _TR(
                success=False,
                validation_error=True,
                error=(
                    f"Your tool-call arguments were not valid JSON and were "
                    f"treated as empty. Raw: {parse_error_raw!r}. "
                    f"Downstream error: {tool_result.error}"
                ),
                metadata=tool_result.metadata,
            )

        # Filesystem-as-scratchpad: externalise oversized successful
        # outputs so compaction doesn't clip the real content later.
        # Happens BEFORE the meta chunk so the UI preview + metadata
        # reflect the scratch path.
        tool_result, scratch_extras = await self._maybe_externalise_result(
            tool_result, tool_name,
        )

        preview = (tool_result.output or tool_result.error or "")[
            :_preview_len(tool_name, success=tool_result.success)
        ]
        tr_extra = {"tool_result": {
            "id": tool_id, "tool": tool_name,
            "success": tool_result.success, "output_preview": preview,
            **scratch_extras,
        }}
        if checkpoint_hash:
            tr_extra["tool_result"]["checkpoint"] = checkpoint_hash
        # Browser tools: forward the minimal metadata the UI's inline
        # screenshot embed reads (coder-conversation.js checks
        # result.metadata.browser.path). Wire tool_result events never
        # carried metadata, so the embed was dead code on the live path —
        # the user never saw the screenshots the agent captured.
        if tool_name.startswith("browser_") and tool_result.metadata:
            _b = tool_result.metadata.get("browser")
            if isinstance(_b, dict) and _b.get("path"):
                tr_extra["tool_result"]["metadata"] = {"browser": {
                    "path": str(_b.get("path") or ""),
                    "title": str(_b.get("title") or "")[:200],
                    "url": str(_b.get("url") or "")[:500],
                }}

        yield self._meta_chunk(
            phase="executing", status="tool_result",
            model=model, extra=tr_extra,
        )

        # Phase 3.2: trust signal. ``_maybe_run_post_write_verify``
        # in tools.py flags a successful write that produced an
        # unparseable file (Python SyntaxError, JSON decode error, etc.)
        # by setting ``metadata['verification_failed'] = True``. The
        # parse-error message is already appended to ``tool_result.output``
        # — this chunk surfaces the failure to the UI as a discrete event
        # so the iteration card can render a verification-failed badge
        # instead of looking like a clean success. Path is included so
        # the UI can deep-link to the file the agent broke.
        if (
            tool_result.metadata
            and tool_result.metadata.get("verification_failed")
        ):
            yield self._meta_chunk(
                phase="executing", status="verification_failed",
                model=model,
                extra={
                    "tool":         tool_name,
                    "tool_call_id": tool_id,
                    "path":         tool_result.metadata.get("path", ""),
                },
            )

        self._append_tool_result_to_history(
            messages, tool_id, tool_name, tool_result, tier,
        )

        # Agent vision: a successful browser_screenshot on a vision-
        # capable model feeds the actual PIXELS back as a follow-up user
        # message (image_url part — the one multimodal shape every
        # backend already serializes). Without this the agent captured
        # screenshots it could never see and reasoned about its UIs
        # blind — the same severed-feed class as the game-agent fix.
        # Tool-role messages can't carry images on most providers, hence
        # the separate user message AFTER the tool result.
        if tool_name == "browser_screenshot" and tool_result.success:
            _img_msg = await self._maybe_screenshot_image_message(
                tool_result, model,
            )
            if _img_msg is not None:
                messages.append(_img_msg)

        # Auto-observation hook — extract durable facts from the just-
        # finished tool call and append to the observation ledger
        # without the model needing to explicitly call ``observe``.
        # Pattern-only; no LLM call. Per-turn cap stops accidental
        # ledger spam from fan-out turns. See [[project_coder_auto_observer]].
        if tool_result.success:
            await self._maybe_auto_observe(
                tool_name=tool_name,
                tool_input=tool_input,
                tool_result=tool_result,
            )

        # Silent-success tracking for shell_exec. The tool returns the
        # literal string "(exit 0, command succeeded with no stdout)"
        # when a successful command produced no output — honest, but
        # when it repeats 3+ times the model loses its grip on what
        # actually happened and spirals. Counter feeds the nudge at
        # iteration level. ``shell_read`` uses the same marker; tracked
        # together.
        if (
            tool_result.success
            and tool_name in ("shell_exec", "shell_read")
        ):
            counters["shell_exec_calls"] = counters.get(
                "shell_exec_calls", 0,
            ) + 1
            if (tool_result.output or "").strip() == (
                "(exit 0, command succeeded with no stdout)"
            ):
                counters["shell_exec_silent"] = counters.get(
                    "shell_exec_silent", 0,
                ) + 1
        # Failing-shell tracking. Counts shell_exec calls that returned
        # success=False (non-validation errors — those are already in
        # validation_errors). Feeds ``_FAILING_SHELL_NUDGE_AT`` so the
        # model gets nudged to change something before the Nth retry.
        if (
            not tool_result.success
            and not getattr(tool_result, "validation_error", False)
            and tool_name == "shell_exec"
        ):
            counters["shell_exec_failed"] = counters.get(
                "shell_exec_failed", 0,
            ) + 1

        if tool_result.success and tool_name in _MUTATING_TOOLS:
            counters["writes"] = counters.get("writes", 0) + 1
            # Track per-path edit count for the thrashing detector
            # (2026-04-20). Feeds _SAME_FILE_EDIT_BREAK logic.
            paths = self._mutation_paths(tool_name, tool_input)
            for path in paths:
                if not path:
                    continue
                counters.setdefault("edited_paths", []).append(path)
                if path not in self._controller_edited_paths:
                    self._controller_edited_paths.append(path)
                # Clear the repeat-read counter for this path so the
                # preemptive refusal doesn't block a legitimate "read
                # to verify my edit" follow-up. Stays in effect for
                # other paths; only THIS path's read history resets.
                self._state.clear_tool_calls_for_path(path)
            # Let the workspace snapshot know the tree may have changed
            # — refresh is lazy, so this just flips a flag. shell_exec
            # also marks stale (cheap over-invalidation) because a
            # shelled-out mv/rm/mkdir won't have gone through
            # _MUTATING_TOOLS but still rearranges files.
            if self._workspace_snapshot is not None:
                self._workspace_snapshot.mark_stale()
            await self._maybe_activate_controller_power(
                "post_write",
                latest_user_text=goal_text,
                edited_paths=list(self._controller_edited_paths),
            )
            for power_event in self._drain_pending_power_activation_events():
                self._append_power_followup_nudge(
                    messages,
                    power_event,
                    goal_text=goal_text,
                )
                yield self._meta_chunk(
                    phase="executing",
                    status="power_activated",
                    model=model,
                    extra={"power_activation": power_event},
                )
        elif tool_result.success and tool_name == "shell_exec":
            if self._workspace_snapshot is not None:
                self._workspace_snapshot.mark_stale()

        # Track test_run pass/fail separately from validation errors —
        # a passing test_run is real progress; a failing one counts
        # toward the _TEST_FAILURE_STREAK_BREAK counter. test_run's
        # tool result success flag mirrors the test outcome (set by
        # TestRunTool.execute: success iff failed==0 and errors==0).
        if tool_name == "test_run":
            if tool_result.success:
                counters["test_passed"] = counters.get("test_passed", 0) + 1
            else:
                counters["test_failed"] = counters.get("test_failed", 0) + 1
                await self._maybe_activate_controller_power(
                    "verify_failed",
                    latest_user_text=goal_text,
                    edited_paths=list(self._controller_edited_paths),
                )
                for power_event in self._drain_pending_power_activation_events():
                    self._append_power_followup_nudge(
                        messages,
                        power_event,
                        goal_text=goal_text,
                    )
                    yield self._meta_chunk(
                        phase="executing",
                        status="power_activated",
                        model=model,
                        extra={"power_activation": power_event},
                    )
        # Track malformed calls (empty required args, missing schema fields).
        # The agent loop uses this to break degenerate retry cycles where the
        # model keeps emitting tool calls with empty params — observed on
        # weaker models, which otherwise loop until _HYBRID_MAX_ITERS.
        if getattr(tool_result, "validation_error", False):
            counters["validation_errors"] = counters.get("validation_errors", 0) + 1
            # Feed into state so the sticky reminder can surface recent
            # blockers across iterations (history alone loses them after
            # compaction truncates tool_result contents to 160 chars).
            self._state.record_validation_error(
                tool_name=tool_name, error=tool_result.error or "",
            )
        elif not tool_result.success:
            # Soft failure — tool ran but returned success=False for
            # non-schema reasons. Observed 2026-04-20: mtime guard
            # repeatedly rejecting an edit, shell_exec with ENOENT on
            # a wrong path, file_read on a non-existent file. Without
            # tracking, the model sees the error in history but — once
            # compaction clips old tool_results or the model's attention
            # narrows — keeps retrying the same call. The sticky reminder
            # renders these with per-target dedup so "code_edit
            # /snake.html × 5" is visible every turn.
            self._state.record_tool_failure(
                tool_name=tool_name,
                target=_soft_failure_target(tool_name, tool_input),
                error=tool_result.error or "",
            )
        if tool_result.success:
            # Intent-level dedup tracker — surfaced to the model via the
            # sticky reminder's "Already inspected" section so it stops
            # re-reading the same file / re-running the same grep 5×.
            # Observed fix for the "rotating fetch variants" loop pattern.
            self._state.record_tool_call(
                tool_name=tool_name,
                tool_input=tool_input,
                iteration=counters.get("iteration", 0),
            )
            # Defensive: if the same (tool, key) has now been called
            # enough times that the reminder clearly wasn't landing,
            # retroactively flag the call as a validation error. The
            # model still got the data (tool_result was appended with
            # tool_result.success=True), but the validation-error streak
            # counter ticks up, so the circuit breaker can end the
            # session if this keeps happening across iterations.
            if (
                self._state.safeguards_enabled
                and self._state.hit_repeat_cap(
                    tool_name=tool_name, tool_input=tool_input, cap=5,
                )
            ):
                counters["validation_errors"] = (
                    counters.get("validation_errors", 0) + 1
                )
                self._state.record_validation_error(
                    tool_name=tool_name,
                    error=(
                        f"You have called {tool_name} with these same "
                        "arguments 5+ times. The content is in your earlier "
                        "tool_results; re-fetching won't reveal new info. "
                        "Either act on what you have or stop and explain "
                        "what's blocking."
                    ),
                )
        counters["tool_calls"] = counters.get("tool_calls", 0) + 1

    async def _maybe_externalise_result(
        self,
        tool_result,
        tool_name: str,
    ) -> tuple:
        """Send oversized successful tool outputs to the scratch store.

        Returns ``(updated_tool_result, extra_dict)``. When a write
        happens, the updated result's ``output`` becomes a summary +
        preview + scratch path; the extra dict carries
        ``scratch_path`` / ``scratch_size`` for the meta chunk. When
        nothing externalises, the result is returned unchanged and
        the extra dict is empty. Errors pass through unchanged —
        they're short (compaction cap 400) and include recovery
        hints the model needs verbatim.
        """
        if (
            not tool_result.success
            or self._scratch_store is None
            or not tool_result.output
        ):
            return tool_result, {}
        try:
            ref = await self._scratch_store.maybe_externalise(
                content=tool_result.output,
                source_tool=tool_name,
            )
        except Exception:
            # Externalisation is the safety net that keeps multi-MB tool
            # outputs from poisoning message history. Silent failure is
            # exactly the case we never want to be deaf to.
            log.warning(
                "scratch.maybe_externalise_failed",
                tool=tool_name,
                size=len(tool_result.output or ""),
                exc_info=True,
            )
            return tool_result, {}
        if ref is None:
            return tool_result, {}

        from augmentum.coder.scratch import render_scratch_message
        from augmentum.tools.base import ToolResult as _TR
        updated = _TR(
            success=tool_result.success,
            output=render_scratch_message(ref),
            error=tool_result.error,
            metadata={
                **(tool_result.metadata or {}),
                "scratch_path": ref.path,
                "scratch_size": ref.original_size,
            },
        )
        extras = {
            "scratch_path": ref.path,
            "scratch_size": ref.original_size,
        }
        return updated, extras

    async def _run_tools_parallel(
        self,
        tool_calls: list[dict],
        tool_map: dict,
        tier,
        messages: list,
        model: str,
        counters: dict,
    ) -> AsyncIterator[InternalStreamChunk]:
        """Execute a batch of read-only tools in parallel via asyncio.gather.

        Streams meta chunks in call order (tool_call events first, then
        tool_result events once the gather completes). Appends results to
        message history in call order too so downstream indexing stays
        deterministic. Read-only tools never increment ``counters['writes']``.
        """
        from augmentum.tools.base import ToolResult as _TR

        normalized = [self._normalize_tool_call(tc) for tc in tool_calls]

        for tool_id, tool_name, tool_input in normalized:
            yield self._meta_chunk(
                phase="executing", status="tool_call", model=model,
                extra={"tool_call": {
                    "id": tool_id, "tool": tool_name, "input": tool_input,
                }},
            )

        # Preemptive repeat-read refusal, parallel edition. For each
        # call in the batch, decide whether to run it for real or short-
        # circuit with a synthetic refusal tool_result. We build a
        # parallel list of "refusal messages or None"; the gather skips
        # real dispatch for refused slots by awaiting an already-resolved
        # _TR. Keeps the downstream zip over (normalized, results, tool_calls)
        # intact — same indexing, same schema contract.
        preflight_refusals: list[str | None] = []
        _refusal_active = self._state.safeguards_enabled
        for _id, n, inp in normalized:
            if _refusal_active and n in _PREEMPTIVE_REFUSAL_TOOLS:
                prior = self._state.repeat_count(
                    tool_name=n, tool_input=inp,
                )
                if prior >= _READ_REPEAT_REFUSAL_CAP - 1:
                    preflight_refusals.append(
                        f"Refused: you have already called {n} with "
                        f"these arguments {prior}× this session without "
                        "the target file being modified. The result is "
                        "in your earlier tool_results — scroll back to "
                        "reuse it. If you need a different view, change "
                        "the arguments. If you're stuck, explain what "
                        "you're trying to find or ask the user."
                    )
                    continue
            preflight_refusals.append(None)

        async def _dispatch_or_refuse(
            idx: int, name: str, inp: dict,
        ):
            if preflight_refusals[idx] is not None:
                return _TR(
                    success=False,
                    validation_error=True,
                    error=preflight_refusals[idx],
                )
            return await _execute_tool(
                tool_map=tool_map, tool_name=name, tool_input=inp,
                workspace_id=self._workspace_id,
            )

        results = await asyncio.gather(
            *[
                _dispatch_or_refuse(i, n, inp)
                for i, (_id, n, inp) in enumerate(normalized)
            ],
            return_exceptions=True,
        )

        for zip_index, ((tool_id, tool_name, _tool_input), res, raw_tc) in enumerate(
            zip(normalized, results, tool_calls, strict=True),
        ):
            self._state.tool_calls_made += 1
            if isinstance(res, BaseException):
                tool_result = _TR(
                    success=False,
                    error=f"Tool {tool_name!r} raised: {res}",
                )
            else:
                tool_result = res

            # Truncation short-circuit: same defense as the serial path.
            # In practice the parallel path is for reads (file_read,
            # code_grep) which rarely truncate, but the surfacing is
            # cheap and prevents this code path from regressing if a
            # truncating tool is ever added to _HYBRID_PARALLEL_READ_TOOLS.
            truncation_reason = (
                raw_tc.get("_truncation_reason")
                if isinstance(raw_tc, dict) else None
            )
            if truncation_reason == _TRUNCATION_REASON_EMPTY_ARGS:
                tool_result = _TR(
                    success=False,
                    validation_error=True,
                    error=_build_truncation_error(
                        tool_name=tool_name,
                        finish_reason=raw_tc.get("_finish_reason", ""),
                    ),
                )

            # Malformed-JSON surfacing (see _run_tool_tracked for rationale).
            parse_error_raw = (
                raw_tc.get("_parse_error_raw")
                if isinstance(raw_tc, dict) else None
            )
            if parse_error_raw is not None and not tool_result.success:
                tool_result = _TR(
                    success=False,
                    validation_error=True,
                    error=(
                        f"Your tool-call arguments were not valid JSON and "
                        f"were treated as empty. Raw: {parse_error_raw!r}. "
                        f"Downstream error: {tool_result.error}"
                    ),
                    metadata=tool_result.metadata,
                )

            # Filesystem-as-scratchpad — same treatment as the serial
            # path. Large reads (which is most of what the parallel
            # path runs) are the primary offenders for compaction clip.
            tool_result, scratch_extras = await self._maybe_externalise_result(
                tool_result, tool_name,
            )

            preview = (tool_result.output or tool_result.error or "")[
                :_preview_len(tool_name, success=tool_result.success)
            ]
            tr_extra_inner: dict = {
                "id": tool_id, "tool": tool_name,
                "success": tool_result.success,
                "output_preview": preview,
                **scratch_extras,
            }
            # Tag preemptively-refused calls in the UI meta chunk so
            # the frontend can render them distinctly from a real tool
            # failure. ``preflight_refusals`` is index-aligned to
            # ``normalized`` by construction above.
            _idx = zip_index  # set by the outer loop's enumerate
            if preflight_refusals[_idx] is not None:
                tr_extra_inner["preemptive_refusal"] = True
                tr_extra_inner["prior_count"] = self._state.repeat_count(
                    tool_name=tool_name, tool_input=_tool_input,
                )
            yield self._meta_chunk(
                phase="executing", status="tool_result", model=model,
                extra={"tool_result": tr_extra_inner},
            )

            self._append_tool_result_to_history(
                messages, tool_id, tool_name, tool_result, tier,
            )
            # test_run outcome tracking — parallel path too, for the
            # thrashing detector. Same classification as the serial
            # path in _run_tool_tracked.
            if tool_name == "test_run":
                if tool_result.success:
                    counters["test_passed"] = counters.get("test_passed", 0) + 1
                else:
                    counters["test_failed"] = counters.get("test_failed", 0) + 1
            # Silent-success shell tracking — see _run_tool_tracked for
            # rationale. Parallel path hits this more often since
            # shell_exec / shell_read fan out through here.
            if (
                tool_result.success
                and tool_name in ("shell_exec", "shell_read")
            ):
                counters["shell_exec_calls"] = counters.get(
                    "shell_exec_calls", 0,
                ) + 1
                if (tool_result.output or "").strip() == (
                    "(exit 0, command succeeded with no stdout)"
                ):
                    counters["shell_exec_silent"] = counters.get(
                        "shell_exec_silent", 0,
                    ) + 1
            if (
                not tool_result.success
                and not getattr(tool_result, "validation_error", False)
                and tool_name == "shell_exec"
            ):
                counters["shell_exec_failed"] = counters.get(
                    "shell_exec_failed", 0,
                ) + 1
            # Read-only tools don't count as writes, by construction.
            # Validation errors are still tracked so the loop can break
            # if the model keeps fan-outing empty-param reads.
            if getattr(tool_result, "validation_error", False):
                counters["validation_errors"] = counters.get("validation_errors", 0) + 1
                self._state.record_validation_error(
                    tool_name=tool_name, error=tool_result.error or "",
                )
            elif not tool_result.success:
                # Soft failure — see _run_tool_tracked for rationale.
                self._state.record_tool_failure(
                    tool_name=tool_name,
                    target=_soft_failure_target(tool_name, _tool_input),
                    error=tool_result.error or "",
                )
            if tool_result.success:
                # Dedup tracker — see _run_tool_tracked for rationale.
                # Parallel path feeds the same buffer so fan-out of
                # redundant reads shows up immediately.
                self._state.record_tool_call(
                    tool_name=tool_name,
                    tool_input=_tool_input,
                    iteration=counters.get("iteration", 0),
                )
                if (
                    self._state.safeguards_enabled
                    and self._state.hit_repeat_cap(
                        tool_name=tool_name, tool_input=_tool_input, cap=5,
                    )
                ):
                    counters["validation_errors"] = (
                        counters.get("validation_errors", 0) + 1
                    )
                    self._state.record_validation_error(
                        tool_name=tool_name,
                        error=(
                            f"You have called {tool_name} with these same "
                            "arguments 5+ times. Stop re-fetching — act on "
                            "what you have or explain what's blocking."
                        ),
                    )
            counters["tool_calls"] = counters.get("tool_calls", 0) + 1

    async def _canonical_observation(self) -> str:
        """Build a fresh workspace observation for hybrid refresh cycle.

        Prefers the auto-refreshing ``WorkspaceSnapshot`` — full tree +
        [NEW]/[MOD]/[DEL] deltas since the last refresh — so observations
        are tree-complete and expose mid-turn mutations automatically.
        Falls back to the legacy top-level ``ls`` + ``git status`` when
        the snapshot isn't available (Phase 1 passthrough) or the scan
        fails. Best-effort throughout: a shell error returns ``""`` so
        the caller's nudge cadence stays unaffected.
        """
        if not self._container_manager:
            return ""

        # Prefer the snapshot — it diffs against the previous state so
        # mid-turn file creations/edits appear automatically, which the
        # legacy ls/git-status observation missed.
        if self._workspace_snapshot is not None:
            try:
                refreshed = await self._workspace_snapshot.refresh_if_stale()
                # If the tree didn't change since last refresh, fall
                # through to the git-status snapshot — refreshing a
                # non-stale tree doesn't tell the model anything new.
                if refreshed:
                    rendered = self._workspace_snapshot.render()
                    if rendered:
                        return rendered
            except Exception:
                log.debug("coder.snapshot_observation_failed", exc_info=True)

        try:
            ctx = await self._get_workspace_context()
            return ctx
        except Exception:
            log.debug("coder.observation_failed", exc_info=True)
            return ""

    async def _read_plan_md(self) -> str:
        """Best-effort read of /workspace/.augmentum/plan.md.

        Returns the file's content (trimmed) or ``""`` on any failure.
        Plan.md is the attention-anchor artifact written at the end
        of ``_plan_phase`` and editable by the agent via ``file_write``
        / ``code_edit``. Sticky reminder reads this every iteration
        so the plan content always lives in the context tail.
        """
        if self._container_manager is None:
            return ""
        try:
            raw = await self._container_manager.file_read(
                self._workspace_id, "/workspace/.augmentum/plan.md",
            )
        except Exception:
            return ""
        return (raw or "").strip()

    def _build_sticky_reminder(
        self,
        *,
        goal: str,
        iteration: int,
        max_iters: int,
        writes: int,
        plan_md: str = "",
        latest_input: str = "",
    ) -> str:
        """Render a Claude Code-style ``<system-reminder>`` block.

        Re-rendered every iteration and injected as a user turn so the
        model always sees: the original goal, its own task list (if
        set), recent validation errors (the signal that history-based
        context alone loses after compaction), and where it is in the
        iteration budget. Sections are omitted when empty so the
        reminder stays tight in the common case.

        ``latest_input`` — when different from ``goal``, included as a
        second line so the model sees BOTH the substantive objective
        it's working on AND the exact phrasing the user just used to
        prompt this turn. Used on continuation messages ("keep going",
        "monitor the download") where the latest input is a weak
        signal and relying on it alone causes the model to
        re-summarise context instead of doing work.
        """
        # Trim both — long prompts otherwise dominate the reminder.
        goal_one_line = " ".join((goal or "").split())[:280]
        latest_one_line = " ".join((latest_input or "").split())[:200]

        # Priming tree (Sprint 1): suppress empty/irrelevant section
        # fallbacks for read-only intents. Non-empty content always
        # surfaces regardless of intent — actual signal beats trim. Only
        # the "Tasks: (empty — call task_list…)" hint is suppressed for
        # INSPECT/REVIEW/RESEARCH, where multi-step planning is the
        # wrong nudge for an answer-shaped turn.
        intent = self._state.current_intent
        is_read_only_intent = intent is not None and intent.kind in {
            TurnIntentKind.INSPECT,
            TurnIntentKind.REVIEW,
            TurnIntentKind.RESEARCH,
        }

        sections: list[str] = []
        sections.append(f"Goal: {goal_one_line}" if goal_one_line else "Goal: (unspecified)")
        if latest_one_line and latest_one_line != goal_one_line:
            # Second line surfaces the user's just-typed message so the
            # model understands "they said X" without confusing X with
            # the overall goal.
            sections.append(f'Latest message: "{latest_one_line}"')

        # Plan.md content — the Manus attention-anchor artifact. Shown
        # AFTER the goal so the model sees both. Clipped to 2000 chars
        # so a verbose multi-page plan doesn't dominate the reminder;
        # the full plan is available at /workspace/.augmentum/plan.md
        # via file_read if needed. Plan.md is also editable by the
        # agent via ``file_write`` / ``code_edit``, so this content
        # reflects whatever the agent most recently wrote.
        if plan_md:
            clipped = plan_md[:2000]
            suffix = (
                "\n... (plan truncated; full content at "
                "/workspace/.augmentum/plan.md)"
                if len(plan_md) > 2000 else ""
            )
            sections.append(f"Plan (plan.md):\n{clipped}{suffix}")

        tasks = self._state.tasks or []
        if tasks:
            lines = ["Tasks:"]
            for t in tasks:
                if not isinstance(t, dict):
                    continue
                status = t.get("status", "pending")
                marker = {
                    "completed":   "[x]",
                    "in_progress": "[~]",
                    "pending":     "[ ]",
                }.get(status, "[ ]")
                suffix = "  ← current" if status == "in_progress" else ""
                lines.append(f"  {marker} {t.get('content', '(untitled)')}{suffix}")
            sections.append("\n".join(lines))
        elif not is_read_only_intent:
            # Suppress the multi-step planning hint on read-only intents
            # (INSPECT/REVIEW/RESEARCH) — those turns answer a question
            # or summarize, not execute a multi-step task list.
            sections.append(
                "Tasks: (empty — call the `task_list` tool to plan your "
                "work if this is multi-step)"
            )

        errors = self._state.recent_validation_errors or []
        if errors:
            lines = ["Recent blockers (malformed tool calls):"]
            for e in errors[-3:]:
                if not isinstance(e, dict):
                    continue
                tool = e.get("tool", "?")
                count = int(e.get("count") or 1)
                suffix = f" (×{count})" if count > 1 else ""
                msg = (e.get("error") or "").split(".")[0][:140]
                lines.append(f"  • {tool}{suffix}: {msg}")
            lines.append(
                "  → Fix by including ALL required arguments in your next "
                "tool call. Read the tool's schema in the error message."
            )
            sections.append("\n".join(lines))

        # Repeated soft failures — mtime-guard rejections, ENOENT, etc.
        # Separate section from "malformed tool calls" because the fix is
        # different: malformed needs a schema re-read; repeated soft-fail
        # needs a fundamentally different approach (re-read the file,
        # check the path exists, install the missing tool, etc.).
        failures = self._state.recent_tool_failures or []
        # Only show ones that have fired at least twice — a single
        # failure is noise; a repeat is a pattern.
        repeated = [f for f in failures if (f.get("count") or 0) >= 2]
        if repeated:
            lines = ["Recent repeated failures:"]
            for e in repeated[-4:]:
                if not isinstance(e, dict):
                    continue
                tool = e.get("tool", "?")
                target = e.get("target") or ""
                count = int(e.get("count") or 1)
                label = f"{tool} {target}".strip()
                msg = (e.get("error") or "").split("\n")[0][:140]
                lines.append(f"  • {label} (×{count}): {msg}")
            lines.append(
                "  → Retrying the same call won't help. Re-read the file "
                "if it's a stale-read error, check the path if it's a "
                "not-found error, or pick a different approach."
            )
            sections.append("\n".join(lines))

        pending_contract = self._state.pending_objective_contract or {}
        if pending_contract:
            lines = ["Pending objective contract:"]
            summary = " ".join(
                str(pending_contract.get("summary") or "").split()
            )[:220]
            if summary:
                lines.append(f"  • Pending proof: {summary}")
            required_next = " ".join(
                str(pending_contract.get("required_next") or "").split()
            )[:220]
            if required_next:
                lines.append(f"  • Required next: {required_next}")
            latest_signal = " ".join(
                str(pending_contract.get("latest_signal") or "").split()
            )[:180]
            if latest_signal:
                lines.append(f"  • Latest signal: {latest_signal}")
            lines.append(
                "  → Do not declare success until this contract is satisfied "
                "or you explain the blocker plainly."
            )
            sections.append("\n".join(lines))

        # Background processes — commands the agent spawned with ``&`` /
        # ``nohup`` / ``disown`` / ``setsid`` this turn. Shown BEFORE
        # "Already inspected" because the failure mode is more severe:
        # re-running a bg command without killing the first one causes
        # ``Address already in use`` errors, which in turn trigger the
        # kill/restart/check spiral. Surfacing "you already started X"
        # steers the agent toward ``pkill`` + reconfigure instead of
        # a second launch.
        bg_procs = self._state.background_processes or []
        if bg_procs:
            lines = ["Background processes started this turn:"]
            for e in bg_procs[-6:]:
                if not isinstance(e, dict):
                    continue
                cmd = (e.get("command") or "").strip()
                count = int(e.get("count") or 1)
                suffix = f" (started ×{count})" if count > 1 else ""
                lines.append(f"  • {cmd}{suffix}")
            lines.append(
                "  → These are still running. Before starting the same "
                "server/daemon again, use `pkill -f <name>` or similar. "
                "Use `ps aux | grep <name>` to check if one is live."
            )
            sections.append("\n".join(lines))

        # "Already inspected" — pure information to prevent the redundant
        # fetch loop where weak models re-read the same file 5× because
        # nothing in their context says "you've seen this." Not a block;
        # the model still decides, but now with the right signal.
        inspected = self._state.recent_tool_calls or []
        if inspected:
            lines = ["Already inspected this turn:"]
            for e in inspected[-6:]:
                if not isinstance(e, dict):
                    continue
                tool = e.get("tool", "?")
                key = (e.get("key") or "").strip()
                count = int(e.get("count") or 1)
                suffix = f" (×{count})" if count > 1 else ""
                # Trim long shell commands so they don't dominate
                display = key if len(key) <= 80 else key[:77] + "…"
                lines.append(f"  • {tool}: {display}{suffix}")
            lines.append(
                "  → The results are in earlier tool_result messages or, "
                "if history was compacted, condensed in the <compacted> "
                "block. Don't re-fetch what you can still see. Re-reading "
                "is legitimate ONLY when content you need was compacted "
                "down to a preview. If you can't find the build command or "
                "answer, explain what's blocking and ask the user — "
                "running the same inspection again won't reveal new info."
            )
            sections.append("\n".join(lines))

        status_line = f"Iteration {iteration}/{max_iters} · {writes} writes so far"
        # Context meter — only rendered when the model-initiated
        # ``compact`` tool is exposed (it's the signal the tool's
        # placement judgment needs; without the tool it's noise).
        # ``_sticky_context_pct`` is set by _inject_sticky_reminder
        # from the live message list each iteration.
        pct = getattr(self, "_sticky_context_pct", None)
        if pct is not None:
            status_line += f" · context {pct}% of budget"
        sections.append(status_line)

        body = "\n\n".join(sections)
        return (
            "<system-reminder>\n"
            "This is a sticky reminder re-rendered every turn. The message "
            "history may have been compacted, so treat this block as the "
            "authoritative view of goal + progress + blockers.\n\n"
            f"{body}\n"
            "</system-reminder>"
        )

    def _inject_sticky_reminder(
        self,
        messages: list,
        *,
        goal: str,
        iteration: int,
        max_iters: int,
        writes: int,
        plan_md: str = "",
        latest_input: str = "",
    ) -> None:
        """Inject (or refresh) the sticky reminder as a user message.

        If the trailing user message is already a sticky reminder,
        replace it in place so history doesn't grow unboundedly. Otherwise
        append. Reminders carry no ``tool_call_id`` so replacement never
        breaks native-tier tool_call chains.

        ``latest_input`` is forwarded to :meth:`_build_sticky_reminder`;
        see that method's docstring for rationale on the goal/latest split.
        """
        from augmentum.config import settings as _settings

        # Context meter for the model-initiated ``compact`` tool: the
        # tool's placement judgment ("am I approaching pressure?")
        # needs a live gauge. Computed here because this is the one
        # site that has the message list; rendered (or not) by
        # _build_sticky_reminder via the transient attribute.
        self._sticky_context_pct = None
        if bool(getattr(_settings, "coder_compact_tool_enabled", False)):
            try:
                from augmentum.utils.tokenizer import count_tokens_messages
                limit = int(getattr(
                    self, "_coder_compact_token_limit", _COMPACT_AT_TOKENS,
                ) or _COMPACT_AT_TOKENS)
                self._sticky_context_pct = min(
                    999, round(100 * count_tokens_messages(messages) / max(1, limit)),
                )
            except Exception:  # noqa: BLE001 — meter is advisory, never blocks
                self._sticky_context_pct = None

        reminder = self._build_sticky_reminder(
            goal=goal, iteration=iteration, max_iters=max_iters,
            writes=writes, plan_md=plan_md, latest_input=latest_input,
        )
        # Detect an existing reminder at the tail (we emit all of ours
        # with the opening sentinel).
        if messages and messages[-1].role == "user":
            tail_content = messages[-1].content or ""
            if tail_content.startswith("<system-reminder>"):
                messages[-1] = Message(role="user", content=reminder)
                return
        messages.append(Message(role="user", content=reminder))

    def _maybe_compact_messages(
        self, messages: list,
    ) -> tuple[bool, int, int]:
        """Collapse the middle of `messages` into a summary if over the
        token threshold.

        Preserves:
          - The system message (always `messages[0]`)
          - The first user message (the task definition)
          - The last ``_COMPACT_KEEP_RECENT`` messages verbatim

        Replaces everything in between with a single synthesized user
        message ``<compacted>…</compacted>`` that condenses each dropped
        turn to one line (tool name + truncated preview, or "called: a, b"
        for assistant turns with tool_calls). Tool_call_id linkage is
        severed for the compacted region, so native-tier models won't see
        dangling tool messages pointing at deleted tool_uses.

        Split into ``_compaction_plan`` (pure analysis) +
        ``_apply_compaction`` (segment build + mutation) so the async
        synthesis path (``_compact_messages_with_synthesis``) can run an
        LLM call between the two without duplicating either half.

        Returns ``(compacted: bool, tokens_before: int, tokens_after: int)``.
        Mutates ``messages`` in place.
        """
        plan, before = self._compaction_plan(messages)
        if plan is None:
            return False, before, before
        return self._apply_compaction(messages, plan, before)

    async def _compact_messages_with_synthesis(
        self, messages: list, request: InternalChatRequest,
    ) -> tuple[bool, int, int]:
        """Compaction with an LLM-written handoff synthesis in the segment.

        Same trigger/region/mutation semantics as
        ``_maybe_compact_messages``, plus one cheap second-model call
        (goal_judge plumbing: no KV affinity, think off, fail-open
        LOUDLY) that turns the mechanical Details lines into a dense
        handoff note — state, decisions + why, learnings, open threads —
        written once into the NEW segment only, so the append-stable
        prefix contract is untouched. Any synthesis failure degrades to
        the plain mechanical segment; compaction itself never fails or
        blocks on the synthesis path.

        ``messages`` must not be mutated by other coroutines between the
        plan and apply steps (the await sits between them); in the coder
        loops the list is iteration-local, so this holds.
        """
        from augmentum.config import settings as _settings

        # Model-initiated fold (the ``compact`` tool): consume the
        # request exactly once, force the plan past the threshold gate,
        # and use the model's own handoff note as the synthesis segment
        # — no second-model call. The flag is consumed even when the
        # fold turns out to be structurally impossible (too few
        # messages): the tool already acked, and leaving the flag set
        # would make the NEXT automatic pass silently claim the note.
        model_note = ""
        if getattr(self._state, "compact_requested", False):
            model_note = (getattr(self._state, "compact_note", "") or "").strip()
            self._state.compact_requested = False
            self._state.compact_note = ""

        plan, before = self._compaction_plan(messages, force=bool(model_note))
        if plan is None:
            if model_note:
                log.info(
                    "coder_compact_tool_noop",
                    reason="no_compactable_region",
                    messages=len(messages),
                )
            return False, before, before
        if model_note:
            return self._apply_compaction(
                messages, plan, before, synthesis=model_note,
            )

        synthesis: str | None = None
        if bool(getattr(_settings, "coder_compaction_synthesis_enabled", True)):
            try:
                from augmentum.coder.compaction_synthesis import (
                    synthesize_compaction_segment,
                )

                lines, structured_header = self._build_segment_parts(
                    messages, plan,
                )
                goal_idx = plan["first_user_idx"]
                user_goal = (
                    (messages[goal_idx].content or "")
                    if 0 <= goal_idx < len(messages) else ""
                )
                synthesis = await synthesize_compaction_segment(
                    self._backend,
                    source_request=request,
                    segment_preview=structured_header + "\n".join(lines),
                    user_goal=user_goal,
                )
            except Exception as exc:  # noqa: BLE001
                # Fail open, loudly — a broken synthesis path must never
                # block compaction (the alternative is a context overflow
                # mid-iteration), and must never fail silently either.
                log.warning(
                    "coder_compaction_synthesis_error", error=str(exc)[:200],
                )
                synthesis = None
        return self._apply_compaction(messages, plan, before, synthesis=synthesis)

    def _compaction_plan(
        self, messages: list, *, force: bool = False,
    ) -> tuple[dict | None, int]:
        """Decide whether compaction should fire and over which region.

        Pure analysis — never mutates ``messages``. Returns
        ``(plan, tokens_before)``; ``plan`` is ``None`` when nothing
        should be compacted (disabled, under threshold, too few
        messages, or no new middle past the existing block).

        ``force=True`` (the model-initiated ``compact`` tool) bypasses
        the enable gate and the token threshold — the model judged the
        semantic seam, so pressure is not the trigger. Structural
        checks (enough messages, a real middle region) still apply.
        """
        from augmentum.config import settings as _settings
        from augmentum.utils.tokenizer import count_tokens_messages

        # Live-tunable: opt-out via settings without restart. Default
        # on — the cost of skipping compaction at the ceiling is a
        # request that overflows the context window mid-iteration.
        if not force and not bool(
            getattr(_settings, "coder_compaction_auto_enabled", True)
        ):
            return None, count_tokens_messages(messages)

        before = count_tokens_messages(messages)
        compact_limit = int(getattr(
            self, "_coder_compact_token_limit", _COMPACT_AT_TOKENS,
        ) or _COMPACT_AT_TOKENS)

        # Trigger at ``threshold * compact_limit`` instead of waiting for
        # the hard ceiling. Pre-2026-05-31 this fired at 100% of the
        # limit, which meant the model was already running with a packed
        # window for the iteration that finally tripped the check. The
        # threshold (default 0.65) creates headroom so the model still
        # has budget for its next response + a tool roundtrip after the
        # compacted prompt is sent.
        threshold = float(getattr(_settings, "coder_compaction_threshold", 0.65) or 0.65)
        threshold = max(0.3, min(0.95, threshold))
        trigger_at = max(1_000, int(compact_limit * threshold))
        if not force and before < trigger_at:
            return None, before

        # Live-tunable keep-recent. Fall back to the module-level default
        # so a misconfigured 0 doesn't strip the entire conversation.
        keep_recent = int(getattr(_settings, "coder_compaction_keep_recent", 0) or 0)
        if keep_recent <= 0:
            keep_recent = _COMPACT_KEEP_RECENT
        keep_recent = max(4, min(40, keep_recent))

        # Need at least: system + first user + keep_recent + 2 middle msgs
        # to compact meaningfully.
        min_for_compact = 2 + keep_recent + 2
        if len(messages) < min_for_compact:
            return None, before

        # Locate anchors. The first-user anchor is the TASK DEFINITION
        # and must skip the per-turn runtime carrier (role=user but
        # scaffolding, not the task) — otherwise the carrier steals the
        # preserved slot and the real task gets condensed to a 160-char
        # ``U:`` line.
        system_idx = 0 if messages and messages[0].role == "system" else -1
        first_user_idx = -1
        start_scan = system_idx + 1 if system_idx >= 0 else 0
        for i in range(start_scan, len(messages)):
            m = messages[i]
            if m.role == "user" and not (
                (m.content or "").startswith(RUNTIME_CARRIER_HEADER)
            ):
                first_user_idx = i
                break

        if first_user_idx < 0:
            return None, before

        # Append-stable extension: if a compacted block already sits
        # right after the task definition (ours, or the /compact
        # path's — both start with ``<compacted``), NEVER re-render
        # it. Re-rendering had two live-measured costs (2026-07-02,
        # session …4dd25611f713): the block's bytes churned at the
        # head of history and invalidated the llama-server prefix
        # cache (stable_pct 0.13 → ~87% re-prefill of a packed
        # window), and the old summary was itself re-condensed into
        # ONE 160-char ``U:`` line — 452 messages of condensed memory
        # crushed on the spot. Instead we condense only the messages
        # AFTER the existing block and append them as a new segment
        # inside its wrapper: everything up to the append point stays
        # byte-identical.
        summary_idx = -1
        existing_summary = None
        if first_user_idx + 1 < len(messages):
            cand = messages[first_user_idx + 1]
            if cand.role in ("user", "assistant") and (
                (cand.content or "").lstrip().startswith("<compacted")
            ):
                summary_idx = first_user_idx + 1
                existing_summary = cand

        # Keep the last N messages verbatim — N comes from the live
        # setting resolved above, not the static _COMPACT_KEEP_RECENT.
        region_start = (
            summary_idx if summary_idx >= 0 else first_user_idx
        ) + 1
        tail_start = max(region_start, len(messages) - keep_recent)
        dropped = messages[region_start:tail_start]
        if len(dropped) < 2:
            # Not enough NEW middle to compact.
            return None, before

        return {
            "compact_limit": compact_limit,
            "first_user_idx": first_user_idx,
            "existing_summary": existing_summary,
            "dropped": dropped,
            "tail_start": tail_start,
        }, before

    def _build_segment_parts(
        self, messages: list, plan: dict,
    ) -> tuple[list[str], str]:
        """Render a plan's dropped region into (Details lines, Summary header).

        Pure — mutates nothing. Shared by ``_apply_compaction`` (which
        writes the segment) and the synthesis path (which feeds the same
        rendering to the second model as its input).
        """
        dropped = plan["dropped"]
        tail_start = plan["tail_start"]

        # ---- Supersession map for file_read results -------------------
        # A read whose file is read again or edited LATER (later in the
        # dropped region, or anywhere in the verbatim tail) is a stale
        # copy — the newest touch is the grounded truth the model should
        # reason from. Stale copies get a one-line tombstone instead of
        # a 1500-char preview; in long runs these dead copies are the
        # bulk of the compacted block's weight.
        def _iter_calls(msg):
            for tc in (getattr(msg, "tool_calls", None) or []):
                if isinstance(tc, dict):
                    fn = tc.get("function", {}) or {}
                    name = fn.get("name", "") or tc.get("name", "") or ""
                    tid = str(tc.get("id", "") or "")
                    args_str = fn.get("arguments", "")
                else:
                    name = getattr(tc, "name", "") or ""
                    tid = str(getattr(tc, "id", "") or "")
                    args_str = ""
                try:
                    args = json.loads(args_str) if args_str else {}
                except (json.JSONDecodeError, TypeError):
                    args = {}
                path = (
                    args.get("path")
                    or args.get("file_path")
                    or args.get("filename")
                    or ""
                )
                yield name, tid, (path if isinstance(path, str) else "")

        read_call_pos: dict[str, tuple[str, int]] = {}
        touch_pos: dict[str, list[int]] = {}
        for pos, m in enumerate(dropped):
            if m.role != "assistant":
                continue
            for name, tid, path in _iter_calls(m):
                if not path:
                    continue
                if name == "file_read" or name in _MUTATING_TOOLS:
                    touch_pos.setdefault(path, []).append(pos)
                if name == "file_read" and tid:
                    read_call_pos[tid] = (path, pos)

        tail_touched: set[str] = set()
        for m in messages[tail_start:]:
            if m.role != "assistant":
                continue
            for name, _tid, path in _iter_calls(m):
                if path and (name == "file_read" or name in _MUTATING_TOOLS):
                    tail_touched.add(path)

        def _superseded_read(msg) -> str:
            """Path if ``msg`` is a stale file_read result, else ''."""
            tid = str(getattr(msg, "tool_call_id", "") or "")
            if not tid or tid not in read_call_pos:
                return ""
            path, pos = read_call_pos[tid]
            if path in tail_touched:
                return path
            if any(p > pos for p in touch_pos.get(path, ())):
                return path
            return ""

        # Build a compact summary of the dropped region
        lines: list[str] = []
        for m in dropped:
            role = m.role
            content = (m.content or "").strip()
            # Skip stale sticky reminders entirely — the CURRENT reminder
            # (rendered fresh each turn) is authoritative; historical
            # copies are pure redundancy and just bloat the summary.
            if role == "user" and content.startswith("<system-reminder>"):
                continue
            # Skip stale runtime carriers — per-turn scaffolding
            # (datetime, prior_turns, recall), not conversation. The
            # next turn re-renders a fresh one; condensing the stale
            # copy would bloat the summary with dead state.
            if role == "user" and content.startswith(RUNTIME_CARRIER_HEADER):
                continue
            # Defensive: never one-line an existing compacted block.
            # The extend path above excludes it from ``dropped`` by
            # position; this guards restructured histories where a
            # block drifted deeper into the list.
            if content.lstrip().startswith("<compacted"):
                continue
            if role == "assistant":
                tc = getattr(m, "tool_calls", None)
                if tc:
                    names = ", ".join(
                        (x.get("function", {}).get("name") if isinstance(x, dict) else "")
                        or getattr(x, "name", "?")
                        for x in tc
                    )
                    # Narration-and-call turns keep BOTH: the content is
                    # the model's stated reasoning for the call — dropping
                    # it (pre-2026-07-09 `elif`) erased the WHY of every
                    # action from the fold.
                    # Write-shaped + shell calls also carry a bounded
                    # argument DIGEST (see _tool_call_digest) so the fold
                    # keeps WHAT changed / WHICH command ran, not just
                    # the tool name. Appends after the narration shape —
                    # never replaces it.
                    digest = _tool_call_digest(tc)
                    suffix = f" [{digest}]" if digest else ""
                    if content:
                        lines.append(
                            f"A: {content[:120]} — called {names}{suffix}"
                        )
                    else:
                        lines.append(f"A: called {names}{suffix}")
                elif content:
                    lines.append(f"A: {content[:120]}")
            elif role == "tool":
                # Tool messages lose their tool_call_id here — that's OK
                # because we also drop the matching assistant tool_use.
                # Successful results get 1500 chars (was 160) because the
                # content IS the grounded fact the model is reasoning from;
                # clipping at 160 wiped most file reads and left the model
                # re-emitting the same file_read every iteration "to see
                # what's in there". ERRORs stay at 400 — the recovery
                # hint fits comfortably and bloating it would crowd out
                # real content. When truncated we keep newlines so
                # line-prefixed outputs (file_read, grep) stay readable.
                is_error = content.startswith("ERROR:")
                stale_path = "" if is_error else _superseded_read(m)
                if stale_path:
                    # Stale copy — a newer read/edit of the same file
                    # exists later in history; that one carries the
                    # grounded content.
                    lines.append(
                        f"T: [file_read {stale_path} — stale copy "
                        "dropped; the file was read or edited again "
                        "later]"
                    )
                    continue
                cap = 400 if is_error else 1500
                total_len = len(content)
                if total_len > cap:
                    preview = content[:cap]
                    if not is_error:
                        # Preserve newlines for grounded content. The note
                        # must be honest about where the rest went: this
                        # segment REPLACES the history, so "full result in
                        # earlier history" (pre-2026-07-09 wording) was a
                        # phantom pointer — the model paged back to nothing,
                        # while the sticky reminder's "don't re-fetch" told
                        # it not to re-read either.
                        preview = (
                            f"{preview}\n... [truncated, {total_len - cap} "
                            f"more chars; the full result was dropped in "
                            f"compaction — re-run the tool if the rest is "
                            f"needed]"
                        )
                    else:
                        preview = preview.replace("\n", " ")
                else:
                    preview = content if not is_error else content.replace("\n", " ")
                lines.append(f"T: {preview}")
            elif role == "user":
                # Observations, nudges, or interjected user turns.
                preview = content[:160].replace("\n", " ")
                lines.append(f"U: {preview}")

        # Structured header — opencode-style categorisation on top of
        # the existing per-message lines. Extracts files touched and
        # tool-call counts from the dropped range so the model can see
        # "what happened" at a glance without re-reading every T: line.
        # The per-message ``lines`` block below remains so grounded
        # content (file contents, grep matches) stays accessible.
        files_read: set[str] = set()
        files_edited: set[str] = set()
        tool_counts: dict[str, int] = {}
        test_pass = 0
        test_fail = 0

        for m in dropped:
            if m.role == "assistant":
                tc_list = getattr(m, "tool_calls", None) or []
                for tc in tc_list:
                    if isinstance(tc, dict):
                        fn = tc.get("function", {})
                        name = fn.get("name", "") or tc.get("name", "")
                        args_str = fn.get("arguments", "")
                    else:
                        name = getattr(tc, "name", "") or ""
                        args_str = ""
                    if not name:
                        continue
                    tool_counts[name] = tool_counts.get(name, 0) + 1
                    try:
                        args = json.loads(args_str) if args_str else {}
                    except (json.JSONDecodeError, TypeError):
                        args = {}
                    path = (
                        args.get("path")
                        or args.get("file_path")
                        or args.get("filename")
                        or ""
                    )
                    if path and isinstance(path, str):
                        if name in ("file_read",):
                            files_read.add(path)
                        elif name in _MUTATING_TOOLS:
                            files_edited.add(path)
            elif m.role == "tool":
                # Infer test outcomes from tool result content. The
                # matching assistant call's name is already counted;
                # here we just tally pass/fail.
                content_lower = (m.content or "").lower()
                if "test" in content_lower:
                    if "passed" in content_lower or "ok" in content_lower:
                        test_pass += 1
                    elif "failed" in content_lower or "error" in content_lower:
                        test_fail += 1

        structured_parts: list[str] = []
        # State-derived fields the dropped-message walk can't infer.
        # ``Active`` is the model's own most recent self-declared focus
        # (in_progress task from task_list); ``Blocker`` surfaces the
        # most recent repeated soft-failure (mtime guard, ENOENT,
        # validation error). Both were previously accessible via the
        # sticky reminder but weren't carried into the compacted block,
        # so a model whose first attended message AFTER compaction was
        # the compacted summary lost the orientation cues.
        active = self._state.active_task() if hasattr(self._state, "active_task") else None
        if active and isinstance(active, dict):
            content = (active.get("content") or "").strip()
            if content:
                structured_parts.append(f"Active task: {content}")
        recent_failures = getattr(self._state, "recent_tool_failures", None) or []
        repeated = [f for f in recent_failures if (f.get("count") or 0) >= 2]
        if repeated:
            last = repeated[-1]
            tool = last.get("tool", "?")
            target = last.get("target") or ""
            count = int(last.get("count") or 1)
            label = f"{tool} {target}".strip()
            structured_parts.append(
                f"Recent blocker: {label} (×{count})"
            )
        if files_edited:
            structured_parts.append(
                "Edited: " + ", ".join(sorted(files_edited))
            )
        if files_read:
            structured_parts.append(
                "Read: " + ", ".join(sorted(files_read))
            )
        if tool_counts:
            # Top 5 tools by call count — gives the model a shape of
            # what the dropped turns did without enumerating trivia.
            top = sorted(tool_counts.items(), key=lambda kv: -kv[1])[:5]
            structured_parts.append(
                "Tools: " + ", ".join(f"{n}×{c}" for n, c in top)
            )
        if test_pass or test_fail:
            structured_parts.append(
                f"Tests: {test_pass} passed, {test_fail} failed"
            )

        structured_header = (
            "### Summary\n" + "\n".join(f"- {p}" for p in structured_parts) + "\n\n"
            if structured_parts else ""
        )
        return lines, structured_header

    def _apply_compaction(
        self,
        messages: list,
        plan: dict,
        before: int,
        synthesis: str | None = None,
    ) -> tuple[bool, int, int]:
        """Materialize a compaction plan into ``messages`` (mutates in place).

        ``synthesis``, when provided, is written into the NEW segment as
        a ``### Synthesis`` section between the mechanical Summary and
        the Details lines. It only ever lands in freshly-appended bytes,
        so the append-stable prefix contract holds regardless of what
        the second model produced.
        """
        from augmentum.utils.tokenizer import count_tokens_messages

        dropped = plan["dropped"]
        compact_limit = plan["compact_limit"]
        first_user_idx = plan["first_user_idx"]
        tail_start = plan["tail_start"]
        existing_summary = plan["existing_summary"]

        lines, structured_header = self._build_segment_parts(messages, plan)

        synthesis_block = ""
        if synthesis and synthesis.strip():
            synthesis_block = "### Synthesis\n" + synthesis.strip() + "\n\n"

        # One immutable segment per compaction pass. The count lives in
        # the SEGMENT header (written once, never edited); the block
        # wrapper below carries no counters at all — a mutating "N
        # earlier messages" header at the head of the block was itself
        # a measured prefix-cache breaker (same class as the
        # ``<prior_turns count=>`` attribute removed 2026-07-02).
        segment_text = (
            f"## Condensed segment ({len(dropped)} messages)\n"
            + structured_header
            + synthesis_block
            + "### Details\n"
            + "\n".join(lines)
        )

        if existing_summary is not None:
            # Extend in place: strip the closer, append the new
            # segment, re-close. Every byte of the existing block up
            # to the old closer is preserved, so the slot cache keeps
            # system + task + all prior condensed history.
            base = (existing_summary.content or "").rstrip()
            if base.endswith("</compacted>"):
                base = base[: -len("</compacted>")].rstrip("\n")
            new_content = f"{base}\n\n{segment_text}\n</compacted>"
            new_content = self._cap_compacted_block(
                new_content, compact_limit,
            )
            summary_msg = Message(
                role=existing_summary.role, content=new_content,
            )
        else:
            summary_msg = Message(
                role="user",
                content=(
                    _COMPACT_WRAPPER_OPEN
                    + segment_text
                    + "\n</compacted>"
                ),
            )

        # Everything before the task definition (system prompt, a
        # runtime carrier riding ahead of it) is preserved verbatim —
        # the old explicit [system, first_user] rebuild silently
        # dropped any such messages.
        new_messages = list(messages[: first_user_idx + 1])
        new_messages.append(summary_msg)
        new_messages.extend(messages[tail_start:])

        after = count_tokens_messages(new_messages)
        # Guard against pathological compactions where the summary is
        # bigger than what it replaced (happens when the dropped region
        # is small and each tool_result sits comfortably under the cap,
        # so we pay the "T: " / per-line overhead without the clip
        # savings). Rollback leaves `messages` untouched and returns
        # compacted=False so the caller's meta chunk isn't misleading.
        if after >= before:
            return False, before, before

        messages.clear()
        messages.extend(new_messages)
        return True, before, after

    @staticmethod
    def _cap_compacted_block(content: str, compact_limit: int) -> str:
        """Bound an append-only compacted block by dropping OLDEST segments.

        The extend path grows the block one segment per compaction
        pass; unchecked, the summary itself would eventually dominate
        the window. When the block exceeds ~25% of the compact token
        limit (chars ≈ tokens × 4, so the char cap is numerically the
        token limit), rebuild it keeping the NEWEST segments that fit
        plus a one-line tombstone. This rewrites the block head — a
        one-time full re-prefill — but it's rare and bounded, unlike
        the every-pass rewrite it replaces.
        """
        cap_chars = max(12_000, int(compact_limit or 0))
        if len(content) <= cap_chars:
            return content

        closer = "\n</compacted>"
        body = content.rstrip()
        if body.endswith("</compacted>"):
            body = body[: -len("</compacted>")].rstrip("\n")

        marker = "\n## Condensed segment "
        first = body.find(marker)
        tombstone = (
            "## Condensed segment (oldest history dropped)\n"
            "- Older condensed segments were removed to stay within "
            "the context budget."
        )
        if first < 0:
            # No segment structure (e.g. a /compact-authored block
            # being extended for the first time while already huge):
            # keep the wrapper's first line + the newest tail.
            nl = body.find("\n")
            head = body[: nl + 1] if nl >= 0 else ""
            budget = max(2_000, cap_chars - len(head) - len(tombstone) - 32)
            return f"{head}{tombstone}\n\n…{body[-budget:]}{closer}"

        head = body[:first]
        segments = [
            "## Condensed segment " + part
            for part in body[first:].split(marker)
            if part.strip()
        ]
        kept: list[str] = []
        budget = cap_chars - len(head) - len(tombstone) - len(closer) - 8
        used = 0
        for seg in reversed(segments):  # newest last → walk backwards
            if used + len(seg) + 2 > budget and kept:
                break
            kept.insert(0, seg)
            used += len(seg) + 2
        return head + "\n\n".join([tombstone, *kept]) + closer

    def _token_budget_chunk(
        self,
        messages: list,
        *,
        scope: str,
        phase: str = "executing",
        model: str = "",
        iteration: int | None = None,
        compacted: bool = False,
    ) -> InternalStreamChunk:
        """Emit approximate prompt-message token usage for Coder traces/UI."""
        from augmentum.coder.context_tokens import token_budget_payload

        payload = token_budget_payload(
            messages,
            scope=scope,
            iteration=iteration,
            limit=getattr(self, "_coder_compact_token_limit", None),
            context_window=getattr(self, "_model_context_window_for_turn", 0),
            compacted=compacted,
        )
        return self._meta_chunk(
            phase=phase,
            status="budget",
            model=model,
            extra={"tokens": payload},
        )

    @staticmethod
    def _meta_chunk(
        *,
        phase: str,
        status: str,
        model: str = "",
        extra: dict | None = None,
    ) -> InternalStreamChunk:
        """Build a metadata-only chunk with no content delta.

        Routes through :func:`chat_egress._validate_metadata` so unknown
        phase/status values fail loud (or log, depending on the
        AUGMENTUM_STRICT_METADATA env var). Keeps the same one-line
        call site that's used ~40× across the handler.
        """
        from augmentum.modes.coder.chat_egress import _validate_metadata
        _validate_metadata(phase, status)
        augmentum: dict = {"mode": "coder", "phase": phase, "status": status}
        if extra:
            augmentum.update(extra)
        return InternalStreamChunk(
            content_delta="",
            model=model,
            augmentum=augmentum,
        )


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

# ``_parse_plan_steps`` moved to ``phase_plan.py``. Not re-imported —
# handler no longer references it directly.


_TRANSIENT_ERROR_MARKERS = (
    "429",
    "too many requests",
    "rate limit",
    "rate-limited",
    "503",
    "service unavailable",
    "502",
    "bad gateway",
    "504",
    "gateway timeout",
    "timeout",
    "temporarily",
)


def _is_transient_backend_error(exc: BaseException) -> bool:
    """Detect rate-limit / server-side transient errors by message content.

    The openai_compat backend raises ``RuntimeError`` with the HTTP status
    embedded in the message string, so we match on that. Network timeouts
    from httpx surface with "timeout" in the message.
    """
    msg = str(exc).lower()
    return any(marker in msg for marker in _TRANSIENT_ERROR_MARKERS)


def _short_error_reason(exc: BaseException) -> str:
    """Produce a UI-safe one-line reason for retry messages."""
    msg = str(exc)
    if "429" in msg or "too many requests" in msg.lower():
        return "rate limited (429)"
    if "503" in msg or "unavailable" in msg.lower():
        return "service unavailable (503)"
    if "timeout" in msg.lower():
        return "request timeout"
    return msg.split("\n", 1)[0][:120]


def _promise_summary(promise: Promise | None) -> dict:
    """Compact dict representation of a Promise for UI payloads.

    Only fields the UI needs — the full Promise (including raw verify
    spec and evidence) is still in the session state if inspection is
    required.
    """
    if promise is None:
        return {}
    return {
        "id": promise.id,
        "description": promise.description,
        "status": promise.status.value,
        "attempts": promise.attempts,
        "max_attempts": promise.max_attempts,
        "verify_kind": promise.verify.kind.value,
        "evidence": (promise.evidence or "")[:200] if promise.evidence else None,
    }


def _tool_fingerprint(tool_name: str, tool_input: dict) -> str:
    """Short stable fingerprint of a tool call for identical-retry detection.

    Trims to the name plus the first ~120 chars of canonicalised input so
    we don't reject minor paraphrases but do catch literal re-runs. Used
    by the mission act_fn to detect "model called shell_exec(curl ...)
    again after it already failed" and force a strategy change.
    """
    try:
        canonical = json.dumps(tool_input or {}, sort_keys=True)[:160]
    except (TypeError, ValueError):
        canonical = str(tool_input)[:160]
    return f"{tool_name}({canonical})"


def _render_mission_observations(
    mission: list[Promise], current: Promise,
) -> str:
    """Render resolved-promise evidence for injection into the act prompt.

    The mission log shows status icons, but the per-promise evidence is
    truncated to one line. When later promises need to know "what actually
    got cloned" or "what error did step 2 see", this block surfaces the
    full recent evidence so the model doesn't re-discover facts by
    re-calling env_info / dir_tree every attempt.
    """
    lines: list[str] = []
    for p in mission:
        if p is current:
            break
        if p.status not in (PromiseStatus.FULFILLED, PromiseStatus.REJECTED):
            continue
        tag = "DONE" if p.status == PromiseStatus.FULFILLED else "FAILED"
        evidence = (p.evidence or "").strip().replace("\n", " ")
        if len(evidence) > 400:
            evidence = evidence[:397] + "..."
        lines.append(f"- [{tag}] {p.description} — {evidence or '(no evidence)'}")
    if not lines:
        return ""
    return "## Mission Observations\n" + "\n".join(lines)


def _tool_to_schema(tool) -> dict:
    """Convert a Tool instance to an OpenAI-style function schema dict."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        },
    }


def _act_system_for_tier(tier) -> str:
    """Pick the right system prompt for the current tool-calling tier.

    Native tier: tools are passed to the model structurally via the
    request's ``tools`` field — every tool's ``description`` and
    ``input_schema`` reach the model's decoder natively. Inlining the
    TOOL_REFERENCE prose in the system prompt duplicates ~1.5k tokens
    of the same information. We drop it to reclaim the budget.

    Text / structured tier: no native schema passes through — the
    model needs the tool catalog in-prompt to know what's available.
    """
    if tier == ToolCallingTier.NATIVE:
        return ACT_SYSTEM
    return ACT_SYSTEM_WITH_TOOLS


def _mission_act_system_for_tier(tier) -> str:
    """Same tier split as `_act_system_for_tier`, for mission strategy."""
    if tier == ToolCallingTier.NATIVE:
        return MISSION_ACT_SYSTEM
    return MISSION_ACT_SYSTEM_WITH_TOOLS


# ---------------------------------------------------------------------------
# Intent-aware system prompt assembly (priming tree, Sprint 1).
#
# The legacy ``_act_system_for_tier`` picks a flat system prompt based
# only on tool-calling tier. That gives every task shape the same prose,
# which (a) wastes tokens on tool descriptions that don't apply and
# (b) leaves a fresh model without a worked example of the shape it's
# being asked to produce.
#
# ``_build_act_system`` adds two branches:
#   - intent-filtered tool shortlist (text/structured tier only —
#     native tier already gets tool schemas via the ``tools=`` arg)
#   - a single matching exemplar (any tier)
# and conditionally includes EDIT_FORMAT_INSTRUCTIONS only for intents
# that may write to the workspace.
# ---------------------------------------------------------------------------


# Intents that may mutate the workspace and therefore benefit from
# EDIT_FORMAT_INSTRUCTIONS. INSPECT/REVIEW/RESEARCH don't edit and
# don't need the SEARCH/REPLACE 4-tier matching reference.
_EDIT_CAPABLE_INTENTS = frozenset({
    TurnIntentKind.IMPLEMENT,
    TurnIntentKind.DEBUG,
    TurnIntentKind.OPERATE,
    TurnIntentKind.UNKNOWN,  # safe superset for unclassified turns
})


def _build_act_system(
    tier,
    intent: TurnIntent | None,
    *,
    state: CoderState | None = None,
) -> str:
    """Assemble the act-phase system prompt for a given tier + intent.

    Branches:
      - NATIVE tier: NATIVE_SYSTEM (already self-contained) + exemplar.
      - TEXT/STRUCTURED tier: ACT_SYSTEM rules + intent-filtered tool
        shortlist + (EDIT_FORMAT_INSTRUCTIONS if edit-capable) +
        exemplar + ACT_SYSTEM completion footer.

    A missing exemplar (loader returns "") degrades gracefully — the
    prompt is still well-formed.

    When ``state`` is supplied, populates ``state.last_priming_telemetry``
    with per-branch token counts. The ledger reads this at turn close
    and persists it to ``coder_turn_runs.priming_telemetry``.
    """
    from augmentum.coder.exemplar_loader import load_exemplar
    from augmentum.coder.tool_shortlist import (
        render_native_intent_hint,
        render_tool_shortlist,
    )
    from augmentum.utils.tokenizer import count_tokens

    intent_kind = intent.kind if intent else TurnIntentKind.UNKNOWN
    exemplar = load_exemplar(intent_kind)
    exemplar_block = f"## Example turn (shape match)\n\n{exemplar}" if exemplar else ""

    branch_tokens: dict[str, int] = {}

    if tier == ToolCallingTier.NATIVE:
        # Native tier: tools delivered via request.tools schema. Prompt
        # adds rules + a one-line preferred-tools hint + exemplar.
        # Hint is soft framing ("others remain available") to keep the
        # model's full toolbox accessible — schema filtering would risk
        # cutting a tool a frontier model legitimately needs, and the
        # exemplar already teaches the intent's shape concretely.
        parts = [NATIVE_SYSTEM]
        branch_tokens["rules"] = count_tokens(NATIVE_SYSTEM)
        intent_hint = render_native_intent_hint(intent_kind)
        if intent_hint:
            parts.append(intent_hint)
            branch_tokens["intent_hint"] = count_tokens(intent_hint)
        else:
            branch_tokens["intent_hint"] = 0
        if exemplar_block:
            parts.append(exemplar_block)
            branch_tokens["exemplar"] = count_tokens(exemplar_block)
        else:
            branch_tokens["exemplar"] = 0
        result = "\n\n".join(parts)
    else:
        # Text / structured tier: splice the tool shortlist + exemplar
        # BEFORE ACT_SYSTEM's "## Completion" footer so the completion
        # contract stays the last thing the model reads.
        rules, sep, completion = ACT_SYSTEM.partition("## Completion")
        completion_block = (sep + completion).rstrip() if sep else ""
        shortlist = render_tool_shortlist(intent_kind)

        parts: list[str] = [rules.rstrip()]
        parts.append(shortlist)
        branch_tokens["rules"] = count_tokens(rules)
        branch_tokens["tool_short"] = count_tokens(shortlist)
        if intent_kind in _EDIT_CAPABLE_INTENTS:
            parts.append(EDIT_FORMAT_INSTRUCTIONS)
            branch_tokens["edit_format"] = count_tokens(EDIT_FORMAT_INSTRUCTIONS)
        else:
            branch_tokens["edit_format"] = 0
        if exemplar_block:
            parts.append(exemplar_block)
            branch_tokens["exemplar"] = count_tokens(exemplar_block)
        else:
            branch_tokens["exemplar"] = 0
        if completion_block:
            parts.append(completion_block)
            branch_tokens["completion"] = count_tokens(completion_block)
        result = "\n\n".join(parts)

    if state is not None:
        state.last_priming_telemetry = {
            "intent": intent_kind.value if hasattr(intent_kind, "value") else str(intent_kind),
            "tier": (
                tier.value if hasattr(tier, "value") else str(tier)
            ),
            "branches": branch_tokens,
            "total_priming_tokens": sum(branch_tokens.values()),
            "exemplar_loaded": bool(exemplar),
        }

    return result


def _walk_json_objects(text: str) -> list[str]:
    """Yield every top-level JSON object substring in ``text`` using a
    depth-aware brace walker.

    Unlike a ``{…}`` regex, this correctly handles nested objects of
    arbitrary depth and strings containing ``{`` / ``}``. Used by the
    text-tier tool-call extractor so models that emit 2-level-nested
    inputs (e.g. ``code_edit`` with structured block arrays) parse
    correctly instead of silently dropping.
    """
    spans: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        ch = text[i]
        if ch != "{":
            i += 1
            continue
        depth = 0
        in_string = False
        escape = False
        j = i
        while j < n:
            c = text[j]
            if escape:
                escape = False
                j += 1
                continue
            if c == "\\" and in_string:
                escape = True
                j += 1
                continue
            if c == '"' and not escape:
                in_string = not in_string
            if not in_string:
                if c == "{":
                    depth += 1
                elif c == "}":
                    depth -= 1
                    if depth == 0:
                        spans.append(text[i : j + 1])
                        i = j + 1
                        break
            j += 1
        else:
            # Unbalanced — bail so we don't infinite-loop.
            break
    return spans


def _extract_tool_calls_from_text(text: str) -> list[dict]:
    """Best-effort extraction of tool calls from LLM text output.

    Handles models that emit tool calls as inline JSON blocks rather than
    structured tool_calls. Looks for patterns like:
        {"tool": "file_read", "input": {"path": "/workspace/main.py"}}
    or OpenAI-style:
        {"name": "file_read", "arguments": {"path": "/workspace/main.py"}}

    Returns a list of normalised tool-call dicts (with "id", "name", "input").
    Returns an empty list on any parse failure — callers treat no tool calls
    as a signal to stop the agent loop.

    Uses a depth-aware brace walker so nested tool inputs (``code_edit``
    with arrays of blocks, ``shell_exec`` with a dict of env vars, etc.)
    parse correctly regardless of nesting depth.

    Also recognises Gemini-style ``tool_code[N] ```python ... ```` blocks
    (converted to a ``shell_exec`` with a base64-wrapped Python payload) so
    models trained with Google's code-execution protocol don't silently
    fail when we can't parse their preferred format.
    """
    results: list[dict] = []
    for raw in _walk_json_objects(text):
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue

        if not isinstance(obj, dict):
            continue

        name = obj.get("tool") or obj.get("name") or obj.get("function", {}).get("name", "")
        inp = obj.get("input") or obj.get("arguments") or obj.get("parameters") or {}
        if isinstance(inp, str):
            try:
                inp = json.loads(inp)
            except json.JSONDecodeError:
                inp = {}

        if name:
            results.append({
                "id": str(uuid.uuid4()),
                "name": name,
                "input": inp,
            })

    if not results:
        for m in _INLINE_TOOL_MARKUP_RE.finditer(text):
            name = (m.group(1) or "").strip()
            raw = (m.group(2) or "").strip()
            if not name or not raw:
                continue
            if name not in _INLINE_TEXT_TOOL_NAMES:
                continue

            args: dict
            if raw.startswith("{"):
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if not isinstance(parsed, dict):
                    continue
                args = parsed
            else:
                arg_key = _INLINE_TOOL_SINGLE_ARG_KEYS.get(name)
                if not arg_key:
                    continue
                args = {arg_key: raw}

            results.append({
                "id": str(uuid.uuid4()),
                "name": name,
                "input": args,
            })

    if not results:
        results.extend(_extract_tool_code_blocks(text))

    return results


def _batch_signature(tool_calls: list[dict]) -> tuple[str, ...]:
    """Stable signature of a tool-call batch for stagnation detection.

    Sorts on ``(name, JSON-stable input)`` so order-invariant batches hash
    to the same value (``file_read a + file_read b`` is the same regardless
    of the order the model emitted them). Used by the hybrid loop's
    soft-nudge path.
    """
    parts = []
    for tc in tool_calls:
        name = tc.get("name") or ""
        raw = tc.get("input") or {}
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                raw = {}
        try:
            parts.append(f"{name}:{json.dumps(raw, sort_keys=True)}")
        except TypeError:
            parts.append(f"{name}:{raw!r}")
    return tuple(sorted(parts))


def _extract_steps_from_request(text: str) -> list[str]:
    """Extract ordered steps from a user request.

    Splits on numbered items, 'then' chains, or sentence boundaries.
    """
    if not text:
        return [text or "Execute the task"]

    # Try numbered steps first: "1) ... 2) ..." or "1. ... 2. ..."
    numbered = re.split(r'(?:^|\n)\s*\d+[.)]\s*', text)
    numbered = [s.strip() for s in numbered if s.strip()]
    if len(numbered) >= 2:
        return numbered

    # Try 'then' splitting: "create X, then run Y, then test Z"
    if re.search(r'\bthen\b', text, re.IGNORECASE):
        parts = re.split(r'[,;.]\s*then\s+', text, flags=re.IGNORECASE)
        parts = [s.strip().rstrip('.') for s in parts if s.strip()]
        if len(parts) >= 2:
            return parts

    # Try sentence splitting for multi-sentence requests
    sentences = re.split(r'(?<=[.!])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 10]
    if len(sentences) >= 2:
        return sentences

    # Single step
    return [text]


def _needs_observation_loop(text: str) -> bool:
    """Detect tasks that require seeing tool output before deciding next steps.

    These tasks involve running commands where the output determines what
    to do next — install/debug/test/deploy/run cycles.  They need the
    ReAct loop (observe result → decide → act again) rather than
    one-shot planning.
    """
    if not text:
        return False
    t = text.lower()

    # Strong signals: tasks where success/failure changes the next action
    _OBSERVATION_PATTERNS = [
        r'\binstall\b.*\b(package|dep|require|librar)',
        r'\b(pip|npm|yarn|cargo|apt|brew|go get)\b',
        r'\b(run|execute|start|launch)\b.*\b(test|server|script|app|build)\b',
        r'\b(debug|troubleshoot|diagnose|investigate)\b',
        r'\b(fix|repair)\b.*\b(error|bug|issue|crash|fail)',
        r'\b(try|retry|attempt)\b',
        r'\b(set\s*up|setup|configure|bootstrap|init)\b.*\b(project|env|environment)\b',
        r'\b(deploy|publish|release)\b',
        r'\b(download|fetch|clone|pull)\b.*\b(and|then)\b',
        r'\b(upgrade|update|migrate)\b.*\b(dep|package|version)',
        r'\bif\s+(it|that|this)\s+(fail|work|succeed)',
        r'\bmake\s+(sure|it\s+work)',
    ]

    for pattern in _OBSERVATION_PATTERNS:
        if re.search(pattern, t):
            return True

    # Weaker signal: multiple sequential verbs implying observe-then-act
    sequential_verbs = len(re.findall(
        r'\b(install|run|test|check|verify|fix|retry|build|start|deploy)\b', t
    ))
    if sequential_verbs >= 3:
        return True

    return False


def _estimate_complexity(text: str) -> int:
    """Estimate the number of distinct steps a request requires.

    Used to decide between direct (1-3 steps) and decompose (4+).
    Counts explicit numbered steps, 'then' chains, and action verbs.
    """
    if not text:
        return 1

    steps = 0

    # Explicit numbered steps: "1) ... 2) ..." or "1. ... 2. ..."
    numbered = re.findall(r'(?:^|\n)\s*\d+[.)]\s', text)
    if numbered:
        return len(numbered)

    # Count 'then' chains: "create X, then run Y, then test Z"
    then_count = len(re.findall(r'\bthen\b', text, re.IGNORECASE))
    steps = then_count + 1  # N 'then's = N+1 steps

    # Count distinct action keywords
    actions = set()
    for pattern in [
        r'\bcreate\b', r'\bwrite\b', r'\bmake\b', r'\bbuild\b',
        r'\bread\b', r'\bcheck\b', r'\brun\b', r'\bexecute\b', r'\btest\b',
        r'\binstall\b', r'\bfind\b', r'\bsearch\b', r'\bgrep\b',
        r'\bedit\b', r'\bmodify\b', r'\bchange\b', r'\brename\b', r'\brefactor\b',
        r'\bfix\b', r'\bdelete\b', r'\bremove\b', r'\badd\b',
        r'\bverify\b', r'\bshow\b', r'\blist\b', r'\bprint\b',
    ]:
        if re.search(pattern, text, re.IGNORECASE):
            actions.add(pattern)

    return max(steps, len(actions))


_GREETING_TOKENS = frozenset({
    "hi", "hello", "hey", "yo", "sup", "hiya", "howdy",
    "gm", "morning", "afternoon", "evening",
    "thanks", "thank", "thx", "ty", "cheers",
    "ok", "okay", "k", "kk", "cool", "nice", "great", "awesome",
    "lol", "haha", "sure", "yeah", "yep", "nah", "no", "yes",
    "bye", "goodbye", "later", "cya",
})

_GREETING_SHORT_PHRASES = frozenset({
    "hey there", "hi there", "hello there",
    "what's up", "whats up", "sup dude",
    "thanks!", "thank you", "ty!",
    "good morning", "good afternoon", "good evening",
    "how are you", "how's it going", "hows it going",
})


# Continuation phrases. A message matching this regex is a signal to
# resume prior work, NOT a new task to plan for. Keys:
#   * Anchored ``^...$`` — only the ENTIRE message (minus punctuation)
#     qualifies, so a long instruction like "continue to add X" never
#     trips the detector even though it starts with "continue".
#   * Includes pure continuations ("continue", "keep going"), polite
#     forms ("please continue"), status queries ("how's it going"), and
#     monitor/check requests ("monitor the download") which semantically
#     also presuppose an in-flight task.
# Observed 2026-04-22 with Qwen 3.5: on "continue please" / "continue
# monitoring the download", the plan phase fires, re-derives intent
# from scratch, and the model parrots the project summary from the
# original "what is this?" turn. Short-circuiting continuation phrases
# past the plan phase (going straight to act with the existing plan)
# fixes the parroting at its source.
_CONTINUATION_RE = re.compile(
    r"""
    ^\s*
    (?:
        continue
      | continue \s+ please
      | please \s+ continue
      | keep \s+ going
      | go \s+ ahead
      | go \s+ on
      | proceed
      | carry \s+ on
      | resume
      | resume \s+ please
      | (?:please \s+)? (?:monitor|watch|check \s+ on) \s+ (?:the \s+)? [\w.-]{1,40}
      | (?:any \s+)? update s? \??
      | (?:what'?s|how \s+ is|how'?s) \s+ (?:the \s+)? (?:status|progress|(?:it \s+)? going)
      | (?:is \s+ it \s+ )? done \??
      | status \??
    )
    \s* [.!]? \s* $
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _is_continuation_request(text: str) -> bool:
    """Detect messages that ask the agent to resume prior work.

    See :data:`_CONTINUATION_RE` for rationale and scope. Returns
    False on empty input. Strips a trailing "please" / "thanks" /
    "thx" politeness suffix before matching so "continue thanks"
    and "continue please" both qualify — these are identical
    intents with extra social padding.
    """
    if not text:
        return False
    stripped = text.strip().lower()
    if not stripped:
        return False
    # Peel off a trailing politeness suffix. Must be a WORD boundary —
    # "please" alone IS a continuation (the regex catches it via the
    # "please continue" branch, handled after peeling).
    stripped = re.sub(
        r"\s*(?:please|thanks|thank\s+you|thx|ty)\s*[.!?,]*\s*$",
        "", stripped,
    ).strip()
    if not stripped:
        # The entire message was "please" / "thanks" — a standalone
        # politeness token. Treat as continuation: the user is
        # prompting us to act on the prior context, not starting
        # fresh.
        return True
    return bool(_CONTINUATION_RE.match(stripped))


def _clean_user_text(raw: str, *, single_line: bool = True) -> str:
    """Strip the coder UI's ``[Terminal context]\\n<buffer>\\n// <intent>`` wrapper
    down to the user's actual intent.

    The UI prepends a terminal buffer + a trailing ``// <intent>`` line before
    sending. Left raw, that wrapper pollutes EVERY memory surface keyed off the
    user message — the turn archive's ``user_goal`` (→ recall embeddings + the
    recalled-context block), the ``<prior_turns>`` ring, and sticky reminders
    (audit 2026-06-25: ~50% of archived goals were terminal-context noise).
    Cleaning ONCE here, applied at each seam where a raw user message becomes a
    goal, fixes the class for all of them at the source.

    Resolution order:
      1) the trailing ``// <intent>`` line when present (always one line);
      2) else, with a leading ``[Terminal context]`` preamble stripped, the
         first remaining line (the rest of the buffer is noise);
      3) else (an un-wrapped message) the whole text — or, when
         ``single_line``, just its first line (sticky-reminder use).
    Never returns None.
    """
    raw = (raw or "").strip()
    if not raw:
        return ""
    intent_lines = [
        line.strip().lstrip("/ ").strip()
        for line in raw.splitlines()
        if line.strip().startswith("//") and line.strip() != "//"
    ]
    if intent_lines:
        return intent_lines[-1]
    without_ctx = re.sub(
        r"^\[Terminal context\].*?(?:\n\n|\Z)", "", raw, flags=re.DOTALL,
    ).strip()
    if without_ctx != raw:
        # A preamble was stripped — the remainder's first line is the best signal.
        return without_ctx.splitlines()[0].strip() if without_ctx else ""
    # No wrapper at all: keep the whole message for goal use; collapse to one
    # line for sticky-reminder use (preserves _last_user_message's behavior).
    return raw if not single_line else raw.splitlines()[0].strip()


def _extract_goal_split(messages: list) -> tuple[str, str]:
    """Return ``(latest_input, substantive_goal)`` for sticky-reminder use.

    ``latest_input`` is always the most recent user message (or the
    empty string). ``substantive_goal`` is the most recent user
    message that is NOT a continuation phrase — walking backward
    through history until we find one. When the latest message is
    already a substantive request, both values are identical.

    Why two values: on a "continue please" turn, the model needs to
    see BOTH signals. "Latest: continue please" tells it why it was
    invoked (no new task). "Goal: <prior substantive message>"
    anchors it in the actual work it's resuming. Without the split,
    vague latest-message text becomes the only goal and the model
    parrots the original turn's project summary because that's the
    next-most-salient thing in view.
    """
    user_contents: list[str] = []
    for m in messages:
        if getattr(m, "role", "") == "user":
            content = (getattr(m, "content", "") or "").strip()
            if not content:
                continue
            # Skip our own sticky-reminder injections — those contain
            # echoes of the goal and would short-circuit the walk
            # back to the true prior message.
            if content.startswith("<system-reminder>"):
                continue
            # De-wrap the UI's [Terminal context]/// envelope so the goal that
            # flows to the archive / recall / prior_turns / sticky reminders is
            # the user's real intent, not terminal noise. Continuation detection
            # below then runs on the clean text too.
            cleaned = _clean_user_text(content, single_line=False)
            if cleaned:
                user_contents.append(cleaned)

    if not user_contents:
        return "", ""

    latest = user_contents[-1]
    if not _is_continuation_request(latest):
        return latest, latest

    for prior in reversed(user_contents[:-1]):
        if not _is_continuation_request(prior):
            return latest, prior

    # Every user message so far has been a continuation — use the
    # latest as both values. Rare but possible on a session that
    # opens with "keep going".
    return latest, latest


def _is_conversational_greeting(text: str) -> bool:
    """Detect greetings / small-talk that shouldn't enter plan→act.

    Observed 2026-04-20: "hey there" triggered the full plan phase,
    which emitted a 3-step inventory plan, which drove the act phase
    into a 40+ tool-call thrash. A greeting doesn't need tools — the
    model just needs to greet back and invite the user to describe
    their task. Route these to passthrough-with-conversational-system
    instead of the plan/act machinery.

    Heuristic: messages under 6 words that are dominated by greeting /
    acknowledgement tokens. Conservative on purpose — a false negative
    (greeting → plan) is what we already had; a false positive
    (actionable request → conversational reply) is a new failure mode,
    so we bias toward the known-safe behaviour.
    """
    if not text:
        return False
    stripped = text.strip().lower()
    if not stripped:
        return False
    # Strip common trailing punctuation so "hey!" matches "hey"
    stripped = stripped.rstrip(".!?,;:~")
    # Direct match on short phrases (handles "hey there", etc.)
    if stripped in _GREETING_SHORT_PHRASES:
        return True
    words = [w.strip(".!?,;:~") for w in stripped.split()]
    words = [w for w in words if w]
    # Single word: must be a greeting token.
    if len(words) == 1:
        return words[0] in _GREETING_TOKENS
    # 2-5 words: must be ALL greeting tokens or all-but-one (for things
    # like "hey claude", "thanks dude", "ok cool thanks").
    if 2 <= len(words) <= 5:
        hits = sum(1 for w in words if w in _GREETING_TOKENS)
        return hits >= len(words) - 1
    return False


def _is_reviewable_turn_path(path: str) -> bool:
    """Return True when a turn-review diff path should be shown to users."""
    clean = (path or "").strip()
    if not clean:
        return False
    return not any(
        clean.startswith(prefix)
        for prefix in _INTERNAL_REVIEW_PREFIXES
    )


def _is_explanatory_request(text: str) -> bool:
    """Detect descriptive / inventory requests that should stay read-only."""
    return _classify_turn_intent(text).kind == TurnIntentKind.INSPECT


def _classify_turn_intent(text: str, *, goal_text: str = "") -> TurnIntent:
    """Thin wrapper kept in handler for tests and call-site stability."""
    return classify_turn_intent(latest_text=text, goal_text=goal_text)


def _is_read_only_request(text: str, *, goal_text: str = "") -> bool:
    """True for inspect/review turns that should stay read-only by default."""
    intent = _classify_turn_intent(text, goal_text=goal_text)
    return intent.read_only_by_default and not intent.explicit_execution


def _explicitly_requests_execution(text: str) -> bool:
    """True when the user clearly asked to run / build / test something."""
    return explicitly_requests_execution(text)


def _is_explanatory_safe_tool_call(tool_name: str, tool_input: dict) -> bool:
    """Allow only inspection or control-flow tools on explanatory turns."""
    if tool_name in _EXPLANATORY_SAFE_TOOLS:
        return True
    if tool_name != "git":
        return False
    action = str((tool_input or {}).get("action", "")).strip().lower()
    return action in _EXPLANATORY_GIT_ACTIONS


# Vague-improvement patterns. Observed 2026-04-20 (Qwen 3.6): "can you
# improve it for me?" sent the model flailing across files without a
# concrete target — reads, failed edits, re-reads of the same file.
# These patterns all share the same structural flaw: the subject is a
# pronoun or elided entirely, with no direction (what KIND of improvement?).
# Fires on every turn because `handle_stream` calls `_is_vague_request`
# per user turn, not just first-turn.
_VAGUE_IMPROVEMENT_PATTERNS = [
    # Generic improvement requests without direction
    r"^improve(?:\s+(?:it|them|this|that|everything))?[.!?]*$",
    r"^make\s+(?:it|them|this|that)\s+better[.!?]*$",
    r"^fix\s+(?:it|them|this|that|stuff|things)[.!?]*$",
    r"^clean(?:\s+(?:it|them|this|that))?(?:\s+up)?[.!?]*$",
    r"^refactor(?:\s+(?:it|them|this|that))?[.!?]*$",
    r"^optimi[sz]e(?:\s+(?:it|them|this|that))?[.!?]*$",
    r"^enhance(?:\s+(?:it|them|this|that))?[.!?]*$",
    r"^polish(?:\s+(?:it|them|this|that))?[.!?]*$",
    # "Can you improve it for me?" / "Can you fix this up please?" forms.
    # Trailing adverbs (up / for me / please) can stack in any order so
    # we loop the group instead of anchoring a single slot.
    r"^(?:can|could|would)\s+you\s+"
    r"(?:improve|fix|clean|refactor|optimi[sz]e|enhance|polish)"
    r"(?:\s+(?:it|them|this|that|up|for\s+me|please))*[.!?]*$",
    # "Help me improve" / "Help me make it better"
    r"^help\s+me\s+"
    r"(?:improve|fix|clean|refactor|optimi[sz]e|enhance|polish)"
    r"(?:\s+(?:it|them|this|that))?[.!?]*$",
    # "Make improvements" — objectless
    r"^make\s+(?:some\s+)?improvements?[.!?]*$",
]


def _is_vague_request(text: str) -> bool:
    """Detect requests too vague to act on without clarification.

    Returns True for:
      - Initial-turn vagueness: "create a file", "make something",
        "build an app".
      - Follow-up vagueness: "improve it", "make it better", "fix this",
        "refactor it" — no concrete target or direction. Weak models
        flail on these because there's nothing to plan around.

    Returns False for actionable commands like "list files", "run tests",
    "show git log", "improve the error messages in auth.py" (specific).
    """
    if not text:
        return True

    text_lower = text.lower().strip()

    # Known vague patterns (must match exactly)
    vague_patterns = [
        r"^create a file$",
        r"^make (?:a|an|some) \w+$",
        r"^build (?:a|an|some) \w+$",
        r"^set up (?:a|an) \w+$",
        r"^write (?:a|an|some) \w+$",
        r"^create (?:a|an|some) \w+$",
        r"^help$",
        r"^help me$",
    ]
    for pattern in vague_patterns:
        if re.match(pattern, text_lower):
            return True

    # Vague-improvement follow-ups
    for pattern in _VAGUE_IMPROVEMENT_PATTERNS:
        if re.match(pattern, text_lower):
            return True

    # Short requests that ARE actionable (don't flag these)
    actionable_short = [
        "list", "ls", "show", "find", "search", "grep", "run", "test",
        "check", "git", "status", "log", "diff", "install", "clean",
    ]
    words = text_lower.split()
    if words and words[0] in actionable_short:
        return False

    return False


def _is_vague_improvement(text: str) -> bool:
    """True iff the text matches an improvement-specific vague pattern.

    Used by `_generate_clarification` to pick the improvement-specific
    question instead of the generic "be more specific" catchall.
    """
    if not text:
        return False
    text_lower = text.lower().strip()
    for pattern in _VAGUE_IMPROVEMENT_PATTERNS:
        if re.match(pattern, text_lower):
            return True
    return False


def _generate_clarification(text: str) -> str:
    """Generate a clarification question for a vague request."""
    text_lower = text.lower().strip()

    # Vague-improvement follow-ups get a direction-picking question
    # instead of the generic "be more specific" catchall. Tested on the
    # Qwen 3.6 "can you improve it for me?" trace — clear direction
    # gives the next turn something concrete to plan around.
    if _is_vague_improvement(text):
        return (
            "What would you like me to improve?\n"
            "- A specific file (tell me which, and what aspect)\n"
            "- Refactor for clarity / readability\n"
            "- Add tests or improve test coverage\n"
            "- Fix bugs (tell me symptoms, or point to a broken behavior)\n"
            "- Add a feature (which one?)\n"
            "- Polish style / formatting\n"
            "- Performance optimizations (where it matters)\n"
            "- Something else (describe it)\n"
        )

    if "file" in text_lower or "script" in text_lower:
        return (
            "What kind of file would you like?\n"
            "- A Python script\n"
            "- A web page (HTML/CSS/JS)\n"
            "- A configuration file (JSON/YAML)\n"
            "- Something else (describe it)\n"
        )
    if any(w in text_lower for w in ["app", "application", "project"]):
        return (
            "What type of project?\n"
            "- Python package with tests\n"
            "- Node.js/Express API\n"
            "- React frontend\n"
            "- Flask/FastAPI web app\n"
            "- Something else (describe it)\n"
        )
    if any(w in text_lower for w in ["test", "testing"]):
        return (
            "What would you like to test?\n"
            "- An existing file (which one?)\n"
            "- Write new tests for a function\n"
            "- Run the existing test suite\n"
        )
    return (
        "Could you be more specific? For example:\n"
        "- \"create a Python script that calculates fibonacci\"\n"
        "- \"list all files in the workspace\"\n"
        "- \"read main.py and add error handling\"\n"
    )


async def _execute_tool(
    *,
    tool_map: dict,
    tool_name: str,
    tool_input: dict,
    workspace_id: str = "",
):
    """Execute a named tool with error handling.

    Returns a ToolResult in all cases — never raises.

    ``workspace_id`` flows into ``_context['workspace_id']`` so tools
    that operate on the current workspace (notably the offer
    substrate's power activation) can target it without a separate
    lookup. Empty falls through without stamping.
    """
    # local import avoids cycle at module load
    from dataclasses import replace

    from augmentum.tools.base import ToolResult, invoke_tool

    tool = tool_map.get(tool_name)
    if tool is None:
        log.warning("coder.unknown_tool", tool_name=tool_name)
        # Flagging as validation_error lets the agent-loop circuit
        # breaker count this toward the malformed-call streak. Without
        # this flag, a model that keeps hallucinating tool names (e.g.
        # 'search_web' instead of 'code_search') loops to _HYBRID_MAX_ITERS.
        return ToolResult(
            success=False,
            validation_error=True,
            error=(
                f"Unknown tool: {tool_name!r}. This tool does not exist in "
                f"the current toolset. Available: {sorted(tool_map)}. "
                f"Pick one of those exact names — don't invent new tools."
            ),
        )

    # Mode + workspace stamps for the offer substrate's gate +
    # workspace-targeting. Defensive setdefault so an inner caller
    # that already stamped these (e.g. test injection) isn't
    # overwritten.
    ctx = dict(tool_input.get("_context") or {})
    ctx.setdefault("mode", "coder")
    if workspace_id:
        ctx.setdefault("workspace_id", workspace_id)
    tool_input = dict(tool_input)
    tool_input["_context"] = ctx

    try:
        import asyncio

        result = await asyncio.wait_for(
            invoke_tool(tool, tool_input),
            timeout=tool.timeout,
        )
    except TimeoutError:
        log.warning("coder.tool_timeout", tool_name=tool_name, timeout=tool.timeout)
        result = ToolResult(
            success=False,
            error=f"Tool {tool_name!r} timed out after {tool.timeout}s",
        )
    except Exception as exc:
        log.warning("coder.tool_error", tool_name=tool_name, error=str(exc))
        result = ToolResult(
            success=False,
            error=f"Tool {tool_name!r} raised: {exc}",
        )

    # Diagnostic — when a tool reports a validation_error (missing
    # required arg, empty path, wrong type), log the SHAPE of what the
    # model actually sent so we can tell schema confusion ("model
    # emitted `filename` instead of `path`") from output-budget
    # truncation ("model emitted only `content` and ran out") from
    # field-name drift ("path key present but value is empty string").
    # Without this, repeating "called without a 'path' argument" tells
    # us nothing about WHY — same symptom, three different root causes.
    # Bodies are summarised (keys + value lengths + 80-char previews)
    # so we never dump a full file body into the container log.
    if getattr(result, "validation_error", False):
        try:
            shape = {}
            for k, v in (tool_input or {}).items():
                if isinstance(v, str):
                    shape[k] = {"type": "str", "len": len(v), "preview": v[:80]}
                elif isinstance(v, (list, tuple)):
                    shape[k] = {"type": type(v).__name__, "len": len(v)}
                elif isinstance(v, dict):
                    shape[k] = {"type": "dict", "keys": sorted(v.keys())[:10]}
                else:
                    shape[k] = {"type": type(v).__name__, "value": repr(v)[:80]}
            log.warning(
                "coder.tool_validation_error",
                tool_name=tool_name,
                arg_keys=sorted((tool_input or {}).keys()),
                arg_shape=shape,
                error_preview=(result.error or "")[:200],
            )
        except Exception:
            # Diagnostic must never mask a real tool error.
            pass

    # Enrich failed results with the tool's declared error_hints.
    # ``Tool.error_hints`` + ``Tool.enrich_error`` exist on the base
    # class (augmentum/tools/base.py:161-227) but no tool called
    # enrich_error itself — each tool just returned a bare error
    # string. Centralising here means every tool's hints surface
    # automatically without each tool having to remember to wrap its
    # own error paths. Hint text is appended as "\n\nHint: <text>"
    # by ``Tool.enrich_error`` so the model sees both the raw error
    # and the actionable recovery suggestion together.
    if not result.success and result.error:
        try:
            enriched = tool.enrich_error(result.error, tool_input)
        except Exception:
            # Defensive — a buggy enrich_error override must not mask
            # the underlying tool failure from the model.
            enriched = result.error
        if enriched != result.error:
            # Copy EVERY field. This rebuild predates ``failure_kind`` and
            # ``card``; a field-by-field constructor silently drops whatever
            # was added to ToolResult after it was written, so the coder was
            # about to lose the crash-vs-empty signal on exactly the failing
            # calls that need it. Mutate a copy instead of re-listing fields.
            result = replace(result, error=enriched)
    return result


# Tools that benefit from live-streamed stdout (long-running container
# execs whose UX wait is dominated by silent build/install/test output).
# Wider than shell_exec alone so any future shell-shaped tool inherits
# the live path without an explicit opt-in.
_SHELL_STREAMING_TOOLS = frozenset({"shell_exec", "shell_read"})

# Tools that emit a live inner-loop activity feed (subagent progress
# events). Uses a parallel streaming wrapper so the UI sees iteration/
# tool-call boundaries inside the subagent as they happen, instead of
# blank-card waiting until the subagent returns.
_SUBAGENT_STREAMING_TOOLS = frozenset({"task_dispatch", "explore_codebase"})

# Explore-shaped wide-search/understanding phrasings that warrant a
# system-driven ``explore_codebase`` dispatch when the subagent-router Power
# is the active controller pick (see ``_maybe_auto_dispatch_explore``). Kept
# narrower than the Power's full trigger set (which also covers audit /
# security / review roles) — we only auto-run the EXPLORE role here.
_AUTO_EXPLORE_MARKERS = (
    "find every", "find all", "where does", "where is", "where are",
    "every call site", "every caller", "every reference", "every usage",
    "all the places", "all places", "across the codebase", "across the repo",
    "throughout the codebase", "trace how", "trace the flow", "how does",
    "map out", "list all", "list every", "understand how",
)


def _text_is_explore_shaped(text: str) -> bool:
    """True when the user's ask is a wide-search / cross-file understanding
    request that an ``explore`` subagent should handle (vs a single-file edit
    or a narrow grep the lead can do inline)."""
    low = (text or "").lower()
    return any(marker in low for marker in _AUTO_EXPLORE_MARKERS)


async def _execute_tool_with_shell_stream(
    *,
    tool_map: dict,
    tool_name: str,
    tool_input: dict,
    model: str,
    workspace_id: str = "",
):
    """Streaming variant of :func:`_execute_tool` for shell tools.

    Async generator that yields:
      - ``InternalStreamChunk`` events carrying live stdout/stderr
        chunks tagged ``aug.status='shell_output'`` AS the docker exec
        produces them. The coder UI's ``onShellOutput`` callback
        appends each chunk into the active tool card's terminal body,
        so the user sees ``pytest`` / ``npm install`` / ``docker
        build`` output landing in real time instead of waiting for
        process exit.
      - A final ``ToolResult`` (always last).

    For non-shell tools or unknown tools, yields just the result —
    same contract as :func:`_execute_tool`.
    """
    import asyncio as _asyncio
    from augmentum.modes.coder.chat_egress import emit

    tool = tool_map.get(tool_name)
    if tool is None or tool_name not in _SHELL_STREAMING_TOOLS:
        # Non-shell path: fall through to the standard executor with
        # no streaming overhead. Yields exactly one value (the result).
        result = await _execute_tool(
            tool_map=tool_map, tool_name=tool_name, tool_input=tool_input,
            workspace_id=workspace_id,
        )
        yield result
        return

    queue: _asyncio.Queue[bytes] = _asyncio.Queue()

    async def _sink(data: bytes) -> None:
        # Bounded by the consumer's drain rate via the bare Queue —
        # if the UI is slow the docker read loop awaits put() and
        # exerts natural backpressure. Acceptable here: stdout
        # rates rarely exceed UI redraw rate, and a stall in the
        # UI shouldn't be allowed to outrun the producer either.
        await queue.put(data)

    tool._on_chunk = _sink

    exec_task = _asyncio.create_task(_execute_tool(
        tool_map=tool_map, tool_name=tool_name, tool_input=tool_input,
        workspace_id=workspace_id,
    ))
    try:
        # Drain queue while exec runs. Each tick races a queue get
        # against task-completion so we wake immediately on either.
        # When exec_task finishes we still drain any leftover bytes
        # that landed between the last poll and task completion.
        get_task: _asyncio.Task | None = None
        while True:
            if get_task is None:
                get_task = _asyncio.create_task(queue.get())
            done, _ = await _asyncio.wait(
                {get_task, exec_task},
                return_when=_asyncio.FIRST_COMPLETED,
            )
            if get_task in done:
                data = get_task.result()
                get_task = None
                if data:
                    text = data.decode("utf-8", errors="replace")
                    yield emit(
                        text,
                        phase="executing", status="shell_output",
                        model=model,
                    )
            if exec_task in done:
                # Drain anything the producer queued between the
                # last poll and now before yielding the result.
                while not queue.empty():
                    leftover = queue.get_nowait()
                    if leftover:
                        text = leftover.decode("utf-8", errors="replace")
                        yield emit(
                            text,
                            phase="executing", status="shell_output",
                            model=model,
                        )
                if get_task is not None and not get_task.done():
                    get_task.cancel()
                break
    except BaseException:
        # User cancel / unexpected error: ensure the exec task is
        # cancelled so the docker exec gets a kill signal via the
        # CancelledError handler in containers._run_command.
        if not exec_task.done():
            import contextlib as _ctx
            exec_task.cancel()
            with _ctx.suppress(BaseException):
                await exec_task
        raise
    finally:
        tool._on_chunk = None

    yield exec_task.result()


async def _execute_tool_with_subagent_stream(
    *,
    tool_map: dict,
    tool_name: str,
    tool_input: dict,
    model: str,
    workspace_id: str = "",
):
    """Streaming variant of :func:`_execute_tool` for ``task_dispatch``.

    Mirrors :func:`_execute_tool_with_shell_stream` but the producer
    is the subagent's inner loop (one ``SubagentProgress`` event per
    iteration / tool call / tool result), not docker stdout. Yields
    one ``InternalStreamChunk`` per progress event tagged
    ``aug.status="subagent_progress"`` so the coder UI can update the
    task_dispatch card's mini activity log live; ends with a final
    ``ToolResult`` (always last). For non-subagent tools or unknown
    tools, falls through to one-shot ``_execute_tool``.
    """
    import asyncio as _asyncio
    from augmentum.modes.coder.chat_egress import emit

    tool = tool_map.get(tool_name)
    if tool is None or tool_name not in _SUBAGENT_STREAMING_TOOLS:
        result = await _execute_tool(
            tool_map=tool_map, tool_name=tool_name, tool_input=tool_input,
            workspace_id=workspace_id,
        )
        yield result
        return

    queue: _asyncio.Queue = _asyncio.Queue()

    async def _sink(progress) -> None:
        # Inner-loop callback. SubagentProgress arrives ~1× per
        # iteration + 2× per tool call (call boundary + result
        # boundary). At an optimistic 50 tools/turn that's ~150 events
        # — well within an unbounded Queue's headroom.
        await queue.put(progress)

    # Install the sink. Cleared in ``finally`` so a second dispatch
    # against the same TaskDispatchTool instance doesn't reuse it.
    tool._on_progress = _sink

    exec_task = _asyncio.create_task(_execute_tool(
        tool_map=tool_map, tool_name=tool_name, tool_input=tool_input,
        workspace_id=workspace_id,
    ))
    try:
        get_task: _asyncio.Task | None = None
        while True:
            if get_task is None:
                get_task = _asyncio.create_task(queue.get())
            done, _ = await _asyncio.wait(
                {get_task, exec_task},
                return_when=_asyncio.FIRST_COMPLETED,
            )
            if get_task in done:
                progress = get_task.result()
                get_task = None
                if progress is not None:
                    yield emit(
                        "",
                        phase="executing", status="subagent_progress",
                        model=model,
                        extra={"subagent_progress": {
                            "instance_id": progress.instance_id,
                            "role": progress.role,
                            "iteration": progress.iteration,
                            "phase": progress.phase,
                            "tool_name": progress.tool_name,
                            "text_preview": progress.text_preview,
                            "tokens_in": progress.tokens_in,
                            "tokens_out": progress.tokens_out,
                            "wallclock_ms": progress.wallclock_ms,
                        }},
                    )
            if exec_task in done:
                # Final drain — pick up anything queued between the
                # last poll and exec completion.
                while not queue.empty():
                    leftover = queue.get_nowait()
                    if leftover is not None:
                        yield emit(
                            "",
                            phase="executing", status="subagent_progress",
                            model=model,
                            extra={"subagent_progress": {
                                "instance_id": leftover.instance_id,
                                "role": leftover.role,
                                "iteration": leftover.iteration,
                                "phase": leftover.phase,
                                "tool_name": leftover.tool_name,
                                "text_preview": leftover.text_preview,
                                "tokens_in": leftover.tokens_in,
                                "tokens_out": leftover.tokens_out,
                                "wallclock_ms": leftover.wallclock_ms,
                            }},
                        )
                if get_task is not None and not get_task.done():
                    get_task.cancel()
                break
    except BaseException:
        if not exec_task.done():
            import contextlib as _ctx
            exec_task.cancel()
            with _ctx.suppress(BaseException):
                await exec_task
        raise
    finally:
        tool._on_progress = None

    yield exec_task.result()


# Late-bind module-level helpers from this file into the mixin modules'
# namespaces so their extracted methods can reference them by bare
# name. See ``_legacy._bind_handler_helpers`` for why this is needed.
# The legacy bind is gated to match the import gate above — same env
# var, same restart-to-toggle semantics.
from augmentum.modes.coder.phase_act import (  # noqa: E402
    _bind_handler_helpers as _bind_act_helpers,
)
from augmentum.modes.coder.phase_plan import (  # noqa: E402
    _bind_handler_helpers as _bind_plan_helpers,
)

if _LEGACY_STRATEGY_ENABLED:
    from augmentum.modes.coder._legacy import (  # noqa: E402
        _bind_handler_helpers as _bind_legacy_helpers,
    )
    _bind_legacy_helpers()
_bind_plan_helpers()
_bind_act_helpers()
