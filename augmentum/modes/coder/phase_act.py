"""Production act-phase strategies for Coder mode.

VERIFIED — 2026-05-10. Originally reconstructed 2026-04-21 from grep +
memory after a botched extraction script. The reconstructed sections
(test_failure_streak, same_file_edit_break, action_stagnation,
inspection_loop_break, plus the upstream same_tool_streak tracker)
have since been confirmed by:

- Structural review against the HIGH-confidence ``validation_error_streak``
  template (counter + reset + break check + ``_reflect_on_streak_break``).
- Behavioral unit tests in ``tests/test_coder_thrashing_breakers.py``,
  ``tests/test_coder_loop_guards.py``, and ``tests/test_coder_loop_wiring.py``
  exercising each detector's positive trigger, reset condition, and
  false-positive guards (17 tests, all passing).

Run ``pytest tests/test_coder_thrashing_breakers.py
tests/test_coder_loop_guards.py tests/test_coder_loop_wiring.py
-k 'streak or thrash or inspection or action_stagnation'`` to re-verify.
"""
from __future__ import annotations

import collections
import json
import re
import uuid
from collections.abc import AsyncIterator

import structlog

from augmentum.coder.prompts import (
    ACT_SYSTEM,  # noqa: F401  — used by _build_act_system (forwarded helper)
    ACT_SYSTEM_WITH_TOOLS,  # noqa: F401  — kept for tier=TEXT code path
    EDIT_FORMAT_INSTRUCTIONS,  # noqa: F401  — used by _build_act_system
    NATIVE_SYSTEM,
    NATIVE_THINK_TOOL_TEACHING,
)
from augmentum.coder.state import CoderPhase
from augmentum.coder.termination import (
    NUDGE_BAILOUT,
    NUDGE_INSISTENCE,
    NUDGE_NO_PROGRESS,
    REASON_NUDGE_BAILOUT,
    REASON_SUBSTANTIVE_ACTIVE,
    TerminationContext,
    TerminationVerdict,
    evaluate_termination,
)

# Native-strategy preamble cutoff. Prose under this length that the
# gate classified as SUBSTANTIVE_ACTIVE (the 2-sentence-30-chars route)
# is treated as a preamble worth nudging on rather than a stop worth
# accepting — Qwen-3.6 routinely emits "I'll look at this. Let me
# check the file." style two-sentence preambles that trip the
# threshold. 200 chars matches the gate's own SUBSTANTIVE_MIN_LONG so
# anything above that is accepted as a real explanation without
# tools. Native-only — hybrid's plan/act flow has its own quality
# control and shouldn't be touched.
_NATIVE_PREAMBLE_OVERRIDE_MAX_CHARS = 200
from augmentum.coder.tools import READ_ONLY_TOOLS  # noqa: F401 — used by inspection detector
from augmentum.models.base import (
    InternalChatRequest,
    InternalStreamChunk,
    Message,
)
from augmentum.modes.analytical.tool_calling import ToolCallingTier

# ``select_tier`` is live-bound via ``_bind_handler_helpers`` so tests
# that monkeypatch ``handler.select_tier`` propagate here.
from augmentum.modes.coder.chat_egress import (  # noqa: F401 — emit_relay for future use
    emit,
    emit_relay,
)
from augmentum.modes.coder.intent import TIER_LIMITS, TurnIntentKind

log = structlog.get_logger(__name__)


def _iteration_thinking_kwargs(request: InternalChatRequest) -> dict:
    """Per-iteration thinking policy, shared by ALL act strategies.

    Default: OFF. Verified 2026-05-10 against Qwen 3.6 35B-A3B: with
    ``enable_thinking: true`` (the chat-template default), the asymmetric
    ``<think>\\n`` prefix forces the model to reason before any tool call.
    Disabling sends the prompt without the leading ``<think>`` tag — the
    model goes straight to normal assistant/tool-call output. For agentic
    work this is the reliability + latency trade we want.

    User override (2026-05-30): the coder composer's per-turn thinking
    toggle arrives via ``request.chat_template_kwargs["enable_thinking"]``
    and wins. Used for "let the model think before proposing a plan"
    workflows — toggle on for the plan turn, off before the implement
    turn. Trust the user; they understand the trade.

    Was native-only until 2026-07-02; canonical and hybrid previously fell
    through to the template default (thinking ON for most reasoning
    families), so the default and the toggle silently didn't apply there —
    same policy now holds everywhere (fix the class, not the strategy).
    """
    user_choice = (request.chat_template_kwargs or {}).get("enable_thinking")
    if user_choice is None:
        return {"enable_thinking": False}
    return {"enable_thinking": bool(user_choice)}


def _env_int(name: str, default: int) -> int:
    """Read an int env var with sane fallback; guards against bad input.

    Duplicated from handler.py so module-level constants below resolve at
    import time (handler.py's ``_env_int`` isn't available yet because
    handler imports this module for ``ActPhaseMixin``).
    """
    import os as _os
    raw = _os.environ.get(name, "")
    if not raw:
        return default
    try:
        v = int(raw)
        return v if v > 0 else default
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Termination / stagnation constants
# Exclusively referenced by the strategies below.
# ---------------------------------------------------------------------------

# Phase 2 / PR-2.3: threshold constants moved into
# augmentum/loops/breakers.py as the single source of truth. The
# names below alias the registry so existing phase_act references keep
# working unchanged; PR-2.4 retires the locals when the LoopRunner
# drives the act loop directly.
from augmentum.loops.breakers import (
    HYBRID_CONTINUATION_LOOKBACK as _HYBRID_CONTINUATION_LOOKBACK,
)
from augmentum.loops.breakers import (
    HYBRID_MIN_TURN_PROSE_CHARS as _HYBRID_MIN_TURN_PROSE_CHARS,
)
from augmentum.loops.breakers import (
    HYBRID_STAGNATION_REPEATS as _HYBRID_STAGNATION_REPEATS,
)
from augmentum.loops.breakers import (
    live_max_iters as _live_max_iters,
)
from augmentum.loops.breakers import (
    live_native_nudge_max,
)
from augmentum.loops.breakers import (
    live_threshold as _live_threshold,
)

# ``_CANONICAL_MAX_ITERS``, ``_HYBRID_OBSERVATION_EVERY``,
# ``_HYBRID_READ_FANOUT`` live in handler.py so tests can monkeypatch
# them at the canonical import site. Methods below read them via
# :func:`_cfg` (late lookup — fresh value each call) to respect
# monkeypatches applied at test runtime.


def _cfg(name: str):
    """Late-resolve a constant from handler.py (respects monkeypatches)."""
    from augmentum.modes.coder import handler as _h
    return getattr(_h, name)


def _tap_tool_result(ev, collector: list):
    """Append tool_result payloads to ``collector`` as events fly past.

    Passive inspection helper for the Phase 6 synthesis path. Every
    yield-site in the hybrid loop routes through this so the collector
    ends up with the full set of tool_result dicts the turn produced
    (real, fanout-dropped, and batch-duplicated). Returns the event
    unchanged so callers can ``yield _tap_tool_result(ev, c)``.
    """
    if ev.augmentum and ev.augmentum.get("status") == "tool_result":
        tr = ev.augmentum.get("tool_result")
        if tr:
            collector.append(tr)
    return ev


# First-person process narration that weak models emit between useful
# sentences during the act phase. The "DTLN preamble" filter in
# phase_plan.py catches this class of prose at the plan boundary;
# the act-phase equivalent strips it from every iteration's clean
# prose so users never see "Let me check if the Dockerfile exists"
# / "I don't seem to have received the dir_tree output yet" /
# "I'm stuck in a loop" / "I realize I need to try again" noise.
#
# Regex matches sentence-initial tells (after sentence boundary or
# line start) up to the next sentence-terminator. Conservatively
# bounded to ~200 chars per sentence so we never swallow a paragraph
# on a missing period.
#
# 2026-04-22: extended from the original 9-tell set after a real
# Pong-game transcript where the model narrated "I see the issue",
# "I'm stuck in a loop", "I realize I", "I'm having trouble", "I
# notice I", "I keep getting" as per-iteration prose. All pure
# process-talk — zero value to the user. Added "I see / notice /
# realize / can / keep / hit / try" and "I'm (stuck / having /
# getting / trying / not)" variants.
_ACT_MONOLOGUE_TELL_RE = re.compile(
    r"(?:^|(?<=[.!?\n]))\s*"
    r"(?:"
        r"Let\s+me\b"
        r"|Let\'s\b"
        r"|I\'ll\b"
        r"|I\s+(?:need|should|want|have|had)\s+to\b"
        # Plain "I should|need|can|could|must <verb>" without "to" —
        # catches "I should look", "I can just explain", "I must
        # first" which slip past the stricter "-to" variant.
        r"|I\s+(?:should|need|can|could|must|will|would)\s+(?:just\s+)?(?:look|check|get|run|try|see|start|kill|restart|explain|find|verify|test|first|also|now)\b"
        r"|I\s+(?:see|notice|realize|realise|noticed|realized|realised)\b"
        r"|I\s+(?:keep|kept)\s+(?:getting|hitting|seeing|running)\b"
        r"|I\s+can\'t\s+(?:see|read|access|find|tell)\b"
        r"|I\s+don\'t\s+(?:seem|see|have|appear|think|know)\b"
        r"|I\s+(?:noticed|realized|realised)\s+that\b"
        r"|I\'m\s+going\s+to\b"
        r"|I\'m\s+thinking\b"
        r"|I\'m\s+(?:stuck|having|getting|trying|hitting|not\s+getting)\b"
        r"|I\s+realize\s+I\b"
        r"|I\s+(?:need|want)\s+to\s+(?:try|check|look|see)\b"
        # "The user wants / is asking / needs / would like" + the
        # less common but equally leaky "The user is X-ing" pattern.
        r"|The\s+user\s+(?:wants|is\s+asking|is\s+saying|is\s+requesting|needs|would\s+like|asked)\b"
    r")"
    r"[^.!?\n]{0,200}[.!?\n]",
    re.IGNORECASE,
)


def _strip_act_monologue(text: str) -> str:
    """Strip first-person process narration from act-phase prose.

    Empty in → empty out; preserves substantive content between tells.
    A fully-monologue response reduces to ``""``, which keeps
    ``total_prose_chars`` accurate (the model produced no visible
    substance, so the fallback summary path should engage).

    Two-pass strategy:
      1. Per-sentence regex strip (existing behaviour) — catches clean
         single-sentence tells like "Let me check the file."
      2. Multi-tell block detector — when the pre-strip text starts
         with multiple monologue tells in close succession (≥3 in the
         first 300 chars), the model is doing chain-of-thought in
         prose. Per-sentence stripping misses this because the tells
         are interleaved with non-tell sentences ("They'd need to…",
         "The answer is: no, because…"). Nuke everything up to the
         first discourse marker ("Good question", "Yes,", "No,",
         "Here", "Actually,", etc.) — the point at which the model
         has finished narrating and begun addressing the user.
    """
    if not text:
        return text

    # Pass 1: existing per-sentence strip.
    stripped = _ACT_MONOLOGUE_TELL_RE.sub("", text)

    # Pass 2: multi-tell block detector. Conservatively scoped to the
    # specific pathology where the model reasons ABOUT the user ("The
    # user is asking / wants / needs...") rather than responding TO
    # them — observed 2026-04-22 in the networking-question transcript
    # where 500 chars of thinking leaked before "Good question — right
    # now...". Signal is a ``The user`` tell in the opening 200 chars,
    # plus ≥1 additional tell elsewhere. Single-sentence "Let me
    # check" / "I'm stuck" slabs WITHOUT the "The user" opener don't
    # qualify — pass 1 handles those, preserving any trailing
    # substantive claim ("The file compiled successfully.").
    head = text[:200]
    has_user_reference = bool(
        re.search(
            r"\bThe\s+user\s+(?:wants|is\s+asking|is\s+saying|is\s+requesting|needs|would\s+like|asked)\b",
            head,
            re.IGNORECASE,
        ),
    )
    total_tells = len(_ACT_MONOLOGUE_TELL_RE.findall(text[:500]))
    if has_user_reference and total_tells >= 2:
        marker_match = _DISCOURSE_MARKER_RE.search(text)
        if marker_match:
            # Resume at the marker. Use pass-1 output from that point
            # forward (which catches any further in-answer tells).
            resume_at = marker_match.start()
            stripped = _ACT_MONOLOGUE_TELL_RE.sub("", text[resume_at:])
        else:
            # Whole response is monologue with no user-facing turn.
            # Drop it entirely — the fallback summary will explain.
            stripped = ""

    # Collapse paragraph spacing that the regex opened up.
    stripped = re.sub(r"\n{3,}", "\n\n", stripped)
    return stripped.strip()


# Discourse markers that signal the start of the actual user-facing
# response after a block of process narration. Match at sentence /
# paragraph start. Kept short and common — a false positive here just
# means we don't strip a long preamble, which is the current behaviour.
# A false negative (the monologue has no marker) is handled by dropping
# the whole response.
_DISCOURSE_MARKER_RE = re.compile(
    r"(?:^|\n)\s*"
    r"(?:"
        r"Good\s+question\b"
        r"|Yes[,.!]"
        r"|No[,.!]"
        r"|Short\s+answer\b"
        r"|Here(?:'s|\s+is)\b"
        r"|Here\s+you\s+go\b"
        r"|Actually[,.!]"
        r"|In\s+short\b"
        r"|Well[,.!]"
        r"|So[,.!]"
        r"|Done[.!]"
        r"|Fixed[.!]"
        r"|Confirmed[.!]"
    r")",
    re.IGNORECASE,
)

_ACTIONABLE_TURN_KINDS = frozenset({
    TurnIntentKind.IMPLEMENT,
    TurnIntentKind.DEBUG,
    TurnIntentKind.OPERATE,
})

_FUTURE_ACTION_PROSE_RE = re.compile(
    r"(?i)(?:^|[.!?\n]\s*)"
    r"(?:"
        r"plan\s*:"
        r"|step\s+\d+\s*:"
        r"|the\s+next\s+step\b"
        r"|now\s+let\s+me\b"
        r"|let\s+me\b"
        r"|i(?:'ll| will)\s+"
        r"(?:kill|start|run|install|launch|try|switch|move|clone|read|inspect|verify|"
        r"use|open|expose|check|download|debug|build)\b"
    r")",
)

_IMPERATIVE_STEP_LINE_RE = re.compile(
    r"^(?:"
    r"kill|start|run|install|launch|try|switch|move|clone|read|inspect|verify|"
    r"use|open|expose|check|download|debug|build"
    r")\b",
    re.IGNORECASE,
)

_OPERATE_REMOTE_ACCESS_RE = re.compile(
    r"\b("
    r"expose|public(?:ly)?|tunnel|remote|share|browser|"
    r"access(?:\s+it|\s+directly)?|open\s+(?:it|the\s+app)|link|url"
    r")\b",
    re.IGNORECASE,
)

_OPERATE_PUBLIC_URL_RE = re.compile(
    r"https?://(?!localhost\b|127(?:\.\d{1,3}){3}\b|0\.0\.0\.0\b|\[::1\]|::1\b)[^\s\"')]+",
    re.IGNORECASE,
)

_OPERATE_LOCAL_SIGNAL_RE = re.compile(
    r"(?:"
    r"https?://(?:localhost|127(?:\.\d{1,3}){3}|0\.0\.0\.0|\[::1\]|::1)\b"
    r"|\blocalhost(?::\d+)?\b"
    r"|\b127(?:\.\d{1,3}){3}(?::\d+)?\b"
    r"|\bServer\s+HTTP:\s*\d{3}\b"
    r"|\bHTTP/\d\.\d\s+\d{3}\b"
    r"|\bport\s+\d{2,5}\b"
    r")",
    re.IGNORECASE,
)

_OPERATE_TUNNEL_ATTEMPT_RE = re.compile(
    r"\b(localtunnel|loca\.lt|ngrok|cloudflared|cloudflare\s+tunnel|tunnel)\b",
    re.IGNORECASE,
)

_OPERATE_PUBLIC_SUCCESS_RE = re.compile(
    r"(?:"
    r"\b(?:verified|confirmed|checked)\b.{0,80}\b(?:public|remote|tunnel)\b.{0,80}\b(?:reachable|responding|working|serving|live)\b"
    r"|"
    r"\b(?:public|remote|tunnel)\s+(?:url|link)\b.{0,80}\b(?:responds?|reachable|working|live|serving)\b"
    r"|"
    r"\bverified\b.{0,80}\b(?:https?://(?!localhost\b|127(?:\.\d{1,3}){3}\b|0\.0\.0\.0\b|\[::1\]|::1\b)[^\s\"')]+)\b.{0,80}\b(?:200|201|204|301|302|307|308)\b"
    r")",
    re.IGNORECASE | re.DOTALL,
)

_OPERATE_BLOCKER_RE = re.compile(
    r"\b("
    r"can't|cannot|unable|couldn't|won't|blocked|"
    r"failed|failing|failure|bad\s+gateway|unavailable|timeout|timed\s+out|"
    r"not\s+reachable|not\s+accessible|not\s+serving|"
    r"requires?\s+(?:an?\s+)?auth\s+token|auth\s+token|authentication\s+failed|"
    r"needs?\s+(?:credentials|an?\s+auth\s+token|a\s+token)|"
    r"interstitial|reminder\s+page|503\b|502\b|4018\b|ERR_[A-Z0-9_]+"
    r")\b",
    re.IGNORECASE,
)

_OPERATE_SUCCESS_CLAIM_RE = re.compile(
    r"\b("
    r"it's\s+live|it\s+is\s+live|running\s+and\s+exposed|"
    r"exposed\s+via\s+tunnel|you\s+can\s+access\s+it|"
    r"your\s+url\s+is|server\s+is\s+running\s+and\s+responding|"
    r"public\s+url|tunnel\s+url"
    r")\b",
    re.IGNORECASE,
)

_OPERATE_EVIDENCE_NUDGE_TEXT = (
    "<nudge>This is an operate/exposure task. A localhost check or "
    "printed tunnel URL is not enough to claim success. Verify the public "
    "URL itself and report that result, or explain the concrete blocker "
    "plainly (for example auth token, interstitial, timeout, or bad "
    "gateway).</nudge>"
)


def _looks_like_future_action_prose(text: str) -> bool:
    """True when the assistant narrated next steps instead of doing them.

    This catches two weak-model failure shapes observed in live coder runs:
    a plan echoed back in act mode (``Plan:``, ``Step 1:``) and plain
    imperative next-step lists such as ``Kill the tunnel`` / ``Try ngrok``.
    """
    if not text:
        return False
    normalized = text.strip()
    if not normalized:
        return False
    if _FUTURE_ACTION_PROSE_RE.search(normalized):
        return True

    body_lines = [
        line.strip(" -*\t")
        for line in normalized.splitlines()
        if line.strip()
    ]
    imperative_lines = [
        line for line in body_lines
        if _IMPERATIVE_STEP_LINE_RE.match(line)
    ]
    return len(imperative_lines) >= 2


def _collect_operate_evidence_text(
    messages: list[Message],
    tool_results: list[dict] | None,
) -> str:
    """Collect recent operational evidence from tool history and meta previews."""
    parts: list[str] = []
    for tool_result in tool_results or []:
        if not isinstance(tool_result, dict):
            continue
        snippet = str(
            tool_result.get("output_preview")
            or tool_result.get("error")
            or "",
        ).strip()
        if snippet:
            parts.append(snippet[:400])

    for message in messages[-20:]:
        content = str(getattr(message, "content", "") or "").strip()
        if not content:
            continue
        if message.role == "tool" or (
            message.role == "user" and "[Tool result:" in content
        ):
            parts.append(content[:1200])

    return "\n".join(parts)


def _needs_operate_completion_nudge(
    *,
    goal_text: str,
    candidate_text: str,
    messages: list[Message],
    tool_results: list[dict] | None,
) -> bool:
    """True when an operate/exposure turn is trying to stop on weak evidence.

    Narrowly scoped to remote/public-access goals. A printed tunnel URL or
    localhost probe is not enough; allow completion only when the assistant
    either explains the concrete blocker plainly or provides stronger public
    reachability evidence.
    """
    goal = (goal_text or "").strip()
    if not goal or not _OPERATE_REMOTE_ACCESS_RE.search(goal):
        return False

    candidate = (candidate_text or "").strip()
    if _OPERATE_BLOCKER_RE.search(candidate):
        return False

    evidence_text = _collect_operate_evidence_text(messages, tool_results)
    combined = "\n".join(part for part in (candidate, evidence_text) if part)
    if not combined:
        return False
    if _OPERATE_PUBLIC_SUCCESS_RE.search(combined):
        return False

    has_attempt_signal = bool(
        _OPERATE_PUBLIC_URL_RE.search(combined)
        or _OPERATE_LOCAL_SIGNAL_RE.search(combined)
        or _OPERATE_TUNNEL_ATTEMPT_RE.search(combined)
    )
    if not has_attempt_signal:
        return False

    return bool(
        _OPERATE_BLOCKER_RE.search(evidence_text)
        or _OPERATE_PUBLIC_URL_RE.search(combined)
        or _OPERATE_LOCAL_SIGNAL_RE.search(evidence_text)
        or _OPERATE_SUCCESS_CLAIM_RE.search(candidate)
    )


def _completion_nudge_for_turn(
    *,
    intent_kind: TurnIntentKind,
    goal_text: str,
    candidate_text: str,
    messages: list[Message],
    tool_results: list[dict] | None,
) -> str:
    """Return a completion-contract nudge for the current turn, if any.

    Starts with operate/exposure evidence gating, but centralizes the
    decision so other turn contracts can plug in without scattering new
    stop-path conditionals across the loop.
    """
    if (
        intent_kind == TurnIntentKind.OPERATE
        and _needs_operate_completion_nudge(
            goal_text=goal_text,
            candidate_text=candidate_text,
            messages=messages,
            tool_results=tool_results,
        )
    ):
        return _OPERATE_EVIDENCE_NUDGE_TEXT
    return ""


def _pending_contract_for_turn(
    *,
    intent_kind: TurnIntentKind,
    goal_text: str,
    candidate_text: str,
    messages: list[Message],
    tool_results: list[dict] | None,
) -> dict:
    """Return a persisted unresolved completion contract for this turn.

    This mirrors :func:`_completion_nudge_for_turn`, but returns a compact
    structured payload the handler can persist across turns.
    """
    if intent_kind != TurnIntentKind.OPERATE:
        return {}
    if not _needs_operate_completion_nudge(
        goal_text=goal_text,
        candidate_text=candidate_text,
        messages=messages,
        tool_results=tool_results,
    ):
        return {}

    evidence_text = _collect_operate_evidence_text(messages, tool_results)
    latest_signal = ""
    for source in (candidate_text, evidence_text):
        for line in str(source or "").splitlines():
            line = " ".join(line.split()).strip()
            if line:
                latest_signal = line[:180]
                break
        if latest_signal:
            break

    return {
        "kind": "operate_remote_access",
        "summary": "Remote/public access is not yet proven for this task.",
        "required_next": (
            "Verify the public URL itself, or explain the concrete blocker "
            "plainly instead of claiming success from local-only evidence."
        ),
        "latest_signal": latest_signal,
        "goal": " ".join((goal_text or "").split())[:220],
    }

# Phase 2 / PR-2.3: aliased from augmentum/loops/breakers.py registry.
from augmentum.loops.breakers import (
    SAME_VALIDATION_REPEAT_BREAK as _SAME_VALIDATION_REPEAT_BREAK,
)


def _find_repeat_offender(state) -> dict | None:
    """Return the recent_validation_errors entry that's repeated past
    the same-signature break threshold, or None. Highest repeat_count
    wins on ties so the bail message names the worst offender."""
    candidates = [
        e for e in (getattr(state, "recent_validation_errors", None) or [])
        if isinstance(e, dict)
        and int(e.get("repeat_count") or 0) >= _SAME_VALIDATION_REPEAT_BREAK
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda e: int(e.get("repeat_count") or 0))


def _format_repeat_break_message(offender: dict) -> str:
    """The user-facing bail copy. Names the tool, the repeat count, and
    the most likely fix — distinct from the generic 5-iteration message
    so operators can tell at a glance which breaker fired.

    Wording note: the failure mode here is the MODEL's per-response
    output token budget — when the JSON tool-call args exceed that
    budget the tail (often the ``path`` field for file_write) gets
    silently dropped at the wire and the call arrives malformed. This
    is independent of ``coder_file_write_max_tokens`` (Augmentum's
    optional pre-emptive cap, default 0 = uncapped). Don't conflate
    them in the copy — operators read "too large" and assume our cap
    needs raising when the cap is already off."""
    tool = str(offender.get("tool") or "?")
    reps = int(offender.get("repeat_count") or 0)
    return (
        f"\n\n[Stopped: `{tool}` failed with the same validation error "
        f"{reps}× in a row. The hint didn't help — the model is stuck. "
        "Most likely cause: the model's per-response output budget is "
        "smaller than the tool-call args it's trying to emit, so the "
        "JSON tail (e.g. `path` for file_write) gets truncated before "
        "it reaches us. Try `code_edit_batch` for targeted changes, "
        "split the write into smaller chunks, raise the backend's "
        "max-output tokens, or switch to a model with a larger output "
        "budget.]\n"
    )


def _identical_result_signature(tool_name: str, tool_input, output) -> str:
    """Stable hash of (tool, arguments, output) for one tool call.

    Used by the identical-call loop detector. Two calls collide ONLY when
    the tool, its arguments, AND its byte-for-byte output all match — so
    naturally time-varying output (timestamps, PIDs, changing file
    contents) won't false-trip, while a genuinely repeated read/shell
    will. Arguments are canonicalised with sorted JSON keys so key order
    doesn't matter; unserialisable inputs fall back to ``repr``.
    """
    import hashlib
    import json
    try:
        canon_input = json.dumps(tool_input, sort_keys=True, default=str)
    except (TypeError, ValueError):
        canon_input = repr(tool_input)
    h = hashlib.sha1()
    h.update((tool_name or "").encode("utf-8", "replace"))
    h.update(b"\x00")
    h.update(canon_input.encode("utf-8", "replace"))
    h.update(b"\x00")
    h.update((output or "").encode("utf-8", "replace"))
    return h.hexdigest()


def _bump_identical_streaks(
    streaks: dict[str, int], iter_sigs: set[str],
) -> int:
    """Advance per-signature consecutive-iteration counters in place.

    ``streaks`` maps a call signature → how many consecutive iterations
    it has appeared in. Signatures NOT seen this iteration are dropped
    (the streak must be consecutive); signatures seen this iteration are
    incremented. Returns the peak streak after the update (0 when no
    successful calls ran). Memory stays bounded by calls-per-iteration
    because non-recurring signatures are pruned every call.
    """
    for sig in list(streaks):
        if sig not in iter_sigs:
            del streaks[sig]
    for sig in iter_sigs:
        streaks[sig] = streaks.get(sig, 0) + 1
    return max(streaks.values(), default=0)


# Command-carousel detector: the same NORMALIZED shell command (output-
# shaping pipe tail + redirections stripped) re-run with no improvement
# in its meaningful signal (pytest counts, error sigs). The test/probe
# re-run class that duplicate_calls (shell-excluded), probe_no_signal
# (byte-identical only), and action_stagnation (same tool name) all miss.
# Motivated by three 2026-07-07 Qwen3.6-35B runs of 147-150 iterations.
# Carries the flaky-test flag too (same cmd, different result, no edit).
from augmentum.coder.command_carousel import (
    carousel_nudge_body,
    carousel_reorientation_body,
    flaky_test_body,
)
from augmentum.coder.command_carousel import (
    extract_signal as _extract_signal,
)

# Always-green-probe detector: the same shell probe re-run with byte-
# identical output despite edits landing in between — the model's
# verification loop carries no signal (print-script "tests"). Threshold
# probe_no_signal_nudge in loops/breakers.py.
from augmentum.coder.probe_signal import (
    probe_no_signal_nudge_body,
)

# Task-list staleness detector. When the model has set a task list via
# ``task_list`` but then runs N iterations without updating it, fire
# a one-shot nudge reminding it to re-render the list (mark the
# in_progress task done, promote the next pending). Catches the
# pattern where the model loses track of its own plan mid-execution.
# Only fires when the list is non-empty AND has at least one
# non-completed item — a stable all-completed list isn't stale, the
# loop's ``tasks_completed`` early termination handles that.
# Threshold lives in loops/breakers.py (TASK_STALE_NUDGE_AT); the
# check logic is the shared TaskSpineTracker (2026-07-06 — hybrid's
# inline copy factored out so native runs the same spine).
from augmentum.coder.task_spine import TaskSpineTracker

# Coarse turn-level progress backstop: N iterations with no measurable
# progress (changed-file set didn't grow AND no new test started
# passing) → nudge then break. The superset floor beneath the narrow
# breakers, making runaway turn length structurally impossible.
from augmentum.coder.turn_progress import (
    progress_stall_nudge_body,
)

# Same-file write-churn ladder (nudge → break). Hybrid had only the
# hard break; native had NOTHING — a 2026-07-06 live 9B run rewrote
# one file 20+ times with every guard blind to it (writes succeed,
# args differ, model never stops). Shared tracker; thresholds
# same_file_edit_nudge / same_file_edit_break in loops/breakers.py.
from augmentum.coder.write_churn import (
    WriteChurnTracker,
    churn_nudge_body,
    escalation_handoff_body,
)

# Coordination-churn detector. Weak models can get stuck repeatedly
# calling ``task_list`` / ``ask_user`` instead of taking the next
# concrete step. We nudge earlier here than the broader stagnation
# breakers because these tools are meta-coordination, not task
# progress.
from augmentum.loops.breakers import COORDINATION_ONLY_NUDGE_AT as _COORDINATION_ONLY_NUDGE_AT

# Failing-build-without-edit detector. Complements
# ``no_write_progress_break`` (which requires a mutating-tool attempt
# to fire). This one catches the pattern where the agent runs a
# failing shell_exec (build / test / install) repeatedly without
# editing anything between attempts — the classic "retry the same
# thing expecting a different result" trap. Opens the shell path that
# silent_success / no_write_progress leave uncovered: shell is being
# used, it's failing, but the agent isn't trying anything new.
#
# Nudge fires once per turn at this threshold; the agent gets told
# explicitly to change something (code, config, or env) before the
# next shell retry, or to explain what it thinks is wrong.
from augmentum.loops.breakers import FAILING_SHELL_NUDGE_AT as _FAILING_SHELL_NUDGE_AT

# Identical-call loop detector. A model that re-issues the SAME tool with
# the SAME arguments and gets byte-identical output is stuck even when each
# call "succeeds" — which is exactly why the other guards miss it:
# no_progress resets on any success, silent_success only fires on empty
# shell stdout, and the validation breaks only watch errors. Hash
# (tool, input, output) per successful call, count consecutive-iteration
# repeats, and nudge once when any signature reaches the threshold.
from augmentum.loops.breakers import (
    IDENTICAL_TOOL_RESULT_NUDGE_AT as _IDENTICAL_RESULT_NUDGE_AT,
)
from augmentum.loops.breakers import (
    INSPECTION_COLD_START_GRACE as _INSPECTION_COLD_START_GRACE,
)

# Tools whose presence-ONLY in an iteration flags it as "pure inspection".
# The inspection_only_streak detector increments when every call in an
# iter is in this set AND no writes happened. Threshold hit → nudge →
# break.
#
# 2026-04-22: ``shell_exec`` REMOVED. It is the agent's only path for
# actions with side effects (apt-get / gradle / cargo / docker build /
# ./server / pkill / rm). Counting it as inspection mis-classified
# legitimate build/run/deploy/install turns as "stuck probing" —
# observed on a Glowstone build task where ``gradle --version``,
# ``which java``, and ``apt-get install`` were all tagged inspection,
# tripping the break at iter 5 mid-build.
#
# ``shell_read`` stays — it's the explicit read-only shell surface
# (``git log``, ``cat``, ``ls``). A turn that only uses shell_read is
# genuinely inspecting.
#
# Safety nets if shell_exec-only turns misbehave:
#   - ``action_stagnation_break``: 20 iters of the same tool name
#   - ``silent_success_nudge``: 3 consecutive "(exit 0, no stdout)"
#   - ``no_write_progress_break``: mutating-tool attempts all failing
#   - ``max_iterations_reached``: 150-iter ceiling
from augmentum.loops.breakers import (
    INSPECTION_TOOLS as _INSPECTION_TOOLS,
)

# Write-without-progress circuit-breaker. Fires when the agent is
# attempting mutating tool calls (code_edit / code_multi_edit /
# file_write) but NONE are succeeding — i.e. every edit bounces off
# a stale search-block, idempotence guard, or validation error.
# Complement to inspection_loop_break (which fires when NO mutations
# are attempted at all) and same_file_edit_break (which fires on too
# many successful edits to one file). The three together cover the
# write-axis degenerate states:
#   - not writing anything               → inspection_loop_break
#   - writing everything to one file     → same_file_edit_break
#   - trying to write but nothing sticks → no_write_progress_break
from augmentum.loops.breakers import (
    MUTATING_TOOL_NAMES as _MUTATING_TOOL_NAMES,
)
from augmentum.loops.breakers import (
    NATIVE_SERIAL_TOOL_NAMES as _NATIVE_SERIAL_TOOL_NAMES,
)
from augmentum.loops.breakers import (
    PARALLEL_READ_TOOLS as _HYBRID_PARALLEL_READ_TOOLS,
)

# Silent-success fog detector. A shell_exec that returns "(exit 0,
# command succeeded with no stdout)" is honest but opaque. When 3+
# consecutive iterations are dominated by silent successes, the model
# has no grounding signal about what its commands are doing and tends
# to spiral (kill → restart → check → check → check). Threshold-based
# nudge fires once per turn, pushing the model toward a diagnostic
# (ps / curl / ls) before its next mutation.
from augmentum.loops.breakers import SILENT_SUCCESS_NUDGE_AT as _SILENT_SUCCESS_NUDGE_AT

# ---------------------------------------------------------------------------
# Termination Quality Gate — nudge messages (Phase 3.6)
#
# The gate (``augmentum.coder.termination``) returns a ``nudge_kind`` tag
# when it judges the model's stop premature. Each variant below is the
# message we inject into the conversation to recover the turn. Three
# distinct framings rather than one generic nudge because the *cause*
# of the bail differs:
#
# * INSISTENCE — user explicitly asked for completion. The framing
#   reminds the model of that contract specifically.
# * BAILOUT — short single-sentence excuse-shaped prose. The framing
#   names the pattern so the model can correct it explicitly.
# * NO_PROGRESS — empty / near-empty prose. The framing is the same
#   structured "what / why / next" as the pre-3.6 default — that nudge
#   already worked well for this case (see project memory note
#   "meaningful-prose stops don't over-nudge").
#
# All three end with the same hard rule: implement, articulate the
# block precisely, or say "complete" with evidence. No middle ground.
# ---------------------------------------------------------------------------

_NUDGE_MESSAGES: dict[str, str] = {
    NUDGE_INSISTENCE: (
        "<nudge>The user explicitly asked you to continue until the task "
        "is finished. You stopped without completing it.\n"
        "Either:\n"
        "- Make the next tool call to continue the work, OR\n"
        "- State PRECISELY what blocks you (a specific file, permission, "
        "or capability you genuinely lack — not 'I would need').\n"
        "A status report is not completion. Truncated tool output is not "
        "a blocker — re-call the tool with the offset/limit hint it gave "
        "you. The user wants the work finished, not a status report.</nudge>"
    ),
    # Qwen Code's nudge: literally "Please continue." (no rationale, no
    # structured block). The verbose 3-sentence reply we used pre-
    # 2026-05-31 was effectively teaching chatty models to write MORE
    # prose explaining themselves — exactly the bail loop we're trying
    # to escape. Short user-role nudge gives the model less room to
    # respond with another preamble. See
    # https://github.com/QwenLM/qwen-code/blob/main/packages/core/src/utils/nextSpeakerChecker.ts#L139
    NUDGE_BAILOUT: "Please continue.",
    NUDGE_NO_PROGRESS: (
        "<nudge>You stopped without narrating anything. The user CAN'T "
        "see the tool results you ran — they only see your prose. "
        "Respond now with:\n"
        "- What you tried this turn\n"
        "- What you learned from it\n"
        "- Next step: either a tool call to continue, or a sentence "
        "explaining why you can't.\n"
        "If the task is genuinely complete, list what you delivered and "
        "stop.</nudge>"
    ),
}


def _nudge_message_for(kind: str) -> str:
    """Look up the nudge text for a termination ``nudge_kind``.

    Falls back to the no-progress message when the gate hands us an
    unknown kind — defensive, since the gate's nudge_kind set is a
    closed enum but a future addition shouldn't break the loop.
    """
    return _NUDGE_MESSAGES.get(kind, _NUDGE_MESSAGES[NUDGE_NO_PROGRESS])


# ---------------------------------------------------------------------------
# Late-bind helpers from handler.py (same pattern as _legacy / phase_plan).
# ---------------------------------------------------------------------------

_HELPERS_BOUND = False


def _format_iteration_error(
    iteration: int,
    error_kind: str,
    error_status: int | None,
    error_message: str,
) -> str:
    """Build the user-visible ``[Agent error ...]`` chunk text.

    Includes status code + a truncated provider message so the user can
    act on the failure without grepping logs. Previously emitted only
    the bare ``[Agent error on iteration N]``, which left both transient
    (rate limit, 5xx) and permanent (4xx auth/validation) cases
    indistinguishable in the conversation.
    """
    if error_kind == "quota":
        status_label = (
            f"HTTP {error_status} — token quota exceeded" if error_status
            else "token quota exceeded"
        )
    else:
        status_label = f"HTTP {error_status}" if error_status else (
            "permanent" if error_kind == "permanent" else "transient"
        )
    detail = (error_message or "").strip()
    # Strip the prefix our retry classifier adds — the status code is
    # already in status_label, repeating it is just noise.
    if detail.startswith(f"Backend returned {error_status}:"):
        detail = detail[len(f"Backend returned {error_status}:"):].strip()
    # Truncate hard for the conversation surface. Full text is still in
    # logs (coder.stream_failed) for deeper investigation.
    if len(detail) > 240:
        detail = detail[:237] + "…"
    if detail:
        return f"\n\n[Agent error on iteration {iteration} — {status_label}: {detail}]\n"
    return f"\n\n[Agent error on iteration {iteration} — {status_label}]\n"


class _LiveProxy:
    __slots__ = ("_attr",)

    def __init__(self, attr: str) -> None:
        self._attr = attr

    def __call__(self, *args, **kwargs):  # noqa: ANN002, ANN003
        from augmentum.modes.coder import handler as _h
        return getattr(_h, self._attr)(*args, **kwargs)


_LIVE_NAMES = ("create_coder_tools", "select_tier")


def _bind_handler_helpers() -> None:
    global _HELPERS_BOUND
    if _HELPERS_BOUND:
        return
    from augmentum.modes.coder import handler as _h

    forwarded = (
        "_tool_to_schema", "_strip_cot_tokens", "_strip_tool_json",
        "_extract_tool_calls_from_text", "_act_system_for_tier",
        "_build_act_system",
        "_parse_plan_steps", "_preview_len", "_has_unclaimed_code_block",
        "_has_unclaimed_tool_markup", "_has_leaked_tool_markup",
        "_has_content_loop", "_is_transient_backend_error",
        "_short_error_reason", "_soft_failure_target",
        "_batch_signature", "_tool_fingerprint", "_env_int",
        "_intent_key", "_MUTATING_TOOLS", "_CREATION_VERB_RE",
        # Continuation detection + goal-split (parroting fix, 2026-04-22)
        "_is_continuation_request", "_extract_goal_split",
    )
    g = globals()
    for name in forwarded:
        if hasattr(_h, name):
            g[name] = getattr(_h, name)
    for name in _LIVE_NAMES:
        g[name] = _LiveProxy(name)
    _HELPERS_BOUND = True


# ---------------------------------------------------------------------------
# Mixin
# ---------------------------------------------------------------------------


class ActPhaseMixin:
    """Production act-phase strategies."""

    # ==================================================================
    # Canonical strategy — consensus agent loop
    # ==================================================================
    # Reproduction of the pattern Codex, Claude Code, opencode, qwen-code,
    # and cline all converge on:
    #
    #     while True:
    #         response = model(messages, tools)
    #         append(assistant)
    #         if no tool calls → break
    #         for tc in tool calls: execute; append result
    #
    # No earned-autonomy budget, no repeat detection, no planner. Iteration
    # cap is a pure fail-safe.
    # ==================================================================

    async def _act_canonical(
        self,
        request: InternalChatRequest,
        workspace_context: str,
    ) -> AsyncIterator[InternalStreamChunk]:
        """Consensus agent loop — one while, model-driven stop, cap as fail-safe.

        **FROZEN — comparison/rollback only.** ``native`` is the shipped
        production default (see ``_act_native``); ``canonical`` is reachable
        only via ``AUGMENTUM_CODER_STRATEGY=canonical`` / the strategy header.
        Do NOT port new loop guards (TQG, silent-success, stagnation, sticky
        reminder, verify/goal gates) into this method — that hand-sync across
        three parallel loops was the maintenance hazard the freeze removes.
        New guard work lands in ``_act_native`` only. Kept loadable so a
        rollback/A-B comparison still runs; expect its guards to lag native.
        """
        tools = create_coder_tools(  # noqa: F821 — bound by _bind_handler_helpers
            self._container_manager, self._workspace_id, self._state,
            executor=getattr(self, "_executor", None),
            tool_registry=self._tool_registry,
            question_callback=self._question_callback,
            profile_store=getattr(self, "_profile_store", None),
            service_store=getattr(self, "_service_store", None),
            user_id=getattr(self, "_user_id", ""),
            planning_mode=getattr(self, "_planning_mode", "default"),
            subagent_dispatcher=self._get_subagent_dispatcher(),
            db_conn=self._resolve_archive_conn(),
            jobs_store=getattr(self, "_jobs_store", None),
        )
        tool_map = {t.name: t for t in tools}
        tier = select_tier(self._backend, request.model)
        tool_schemas = (
            [_tool_to_schema(t) for t in tools]  # noqa: F821
            if tier == ToolCallingTier.NATIVE else None
        )

        yield self._meta_chunk(
            phase="executing", status="strategy",
            model=request.model, extra={"strategy": "canonical"},
        )

        # Priming tree (Sprint 1): intent-aware system prompt assembly.
        # Falls back to UNKNOWN if state.current_intent is None, which
        # _build_act_system handles as a safe write-capable superset.
        # state= captures per-branch token telemetry for the ledger.
        act_system = _build_act_system(  # noqa: F821 — bound by _bind_handler_helpers
            tier=tier, intent=self._state.current_intent,
            state=self._state,
        )
        if workspace_context:
            act_system += f"\n\n{workspace_context}"
        if self._state.plan:
            act_system += f"\n\n## Current Plan\n\n{self._state.plan}"
        # Plan-mode notice. No-op outside plan mode; concatenation is
        # cheap on empty.
        act_system += self._plan_mode_addendum()
        messages = self._build_messages(request, act_system)

        iteration = 0
        termination_reason = "model_stop"
        validation_error_streak = 0
        total_writes = 0
        turn_tool_results: list[dict] = []
        # Split the goal: ``latest_input`` is what the user just typed,
        # ``user_goal`` is the last substantive request (walking past
        # "continue please" / "keep going" continuations). Both get
        # surfaced in the sticky reminder so the model can distinguish
        # "you're working on X, and the user said 'continue'" from "the
        # user asked you to continue X" — a distinction that caused
        # parroting when collapsed to one value.
        latest_input, user_goal = _extract_goal_split(request.messages)  # noqa: F821 — bound by _bind_handler_helpers

        progress_without_action_nudged = False
        operate_completion_nudged = False
        # When the workspace has safeguards_enabled=False, lift the
        # canonical cap to the ungated runaway-protection ceiling so
        # strong models can run as long as they need.
        if getattr(self._state, "safeguards_enabled", True):
            canonical_max_iters = _cfg("_CANONICAL_MAX_ITERS")
        else:
            canonical_max_iters = _live_max_iters(ungated=True)
        while iteration < canonical_max_iters:
            iteration += 1

            # Cooperative pause-gate + steer-message inbox drain.
            # No-op when not running under the background-runs broker.
            # See ``_coop_iteration_check`` for the contract; drained
            # steers are appended as user messages so the next backend
            # call sees the redirect mid-turn.
            _steer = await self._coop_iteration_check()
            if _steer:
                for _s in _steer:
                    messages.append(Message(
                        role="user",
                        content=self._format_steer_content(_s),
                    ))
                yield self._meta_chunk(
                    phase="executing", status="steer_delivered",
                    model=request.model,
                    extra={"count": len(_steer), "iteration": iteration},
                )

            # Early termination: explicit finish_task signal (see
            # _act_hybrid for rationale). Checked at iteration top so
            # a mid-iteration finish_task call terminates on the very
            # next pass without burning another backend round-trip.
            if self._state.finish_requested:
                completion_nudge = ""
                pending_contract: dict = {}
                if (
                    not operate_completion_nudged
                ):
                    pending_contract = _pending_contract_for_turn(
                        intent_kind=self._turn_intent_for_turn.kind,
                        goal_text=user_goal or latest_input,
                        candidate_text=self._state.finish_summary,
                        messages=messages,
                        tool_results=turn_tool_results,
                    )
                    completion_nudge = _completion_nudge_for_turn(
                        intent_kind=self._turn_intent_for_turn.kind,
                        goal_text=user_goal or latest_input,
                        candidate_text=self._state.finish_summary,
                        messages=messages,
                        tool_results=turn_tool_results,
                    )
                if completion_nudge:
                    self._state.set_pending_objective_contract(pending_contract)
                    messages.append(Message(
                        role="user",
                        content=completion_nudge,
                    ))
                    self._state.finish_requested = False
                    self._state.finish_summary = ""
                    operate_completion_nudged = True
                    yield self._meta_chunk(
                        phase="executing",
                        status="operate_evidence_nudge",
                        model=request.model,
                        extra={"iteration": iteration},
                    )
                    continue
                self._state.clear_pending_objective_contract()
                termination_reason = "finish_task_called"
                yield self._meta_chunk(
                    phase="executing", status="finish_task_called",
                    model=request.model,
                    extra={"summary_chars": len(self._state.finish_summary)},
                )
                break

            # Plan.md refresh per iteration — the attention-anchor
            # artifact edit-able by the agent. Sticky reminder carries
            # the content at the tail so compaction preserves it.
            #
            # Workspace-kernel v2: when the flag is on, suppress the
            # plan-content injection. Plan lives at
            # /workspace/.augmentum/plan.md and the model reads it on
            # demand via the file_read tool — the system prompt's
            # <workspace_kernel> block tells it where to look. Skipping
            # the read is the actual win: we stop fetching + injecting
            # ~2 KB of plan content into every iteration.
            from augmentum.config import settings as _settings_for_kernel
            plan_md = (
                "" if _settings_for_kernel.coder_kernel_v2
                else await self._read_plan_md()
            )

            # Sticky reminder (see _act_hybrid for rationale). Inject
            # before compaction so the fresh reminder sits at the tail
            # that compaction preserves verbatim.
            self._inject_sticky_reminder(
                messages, goal=user_goal, iteration=iteration,
                max_iters=canonical_max_iters, writes=total_writes,
                plan_md=plan_md, latest_input=latest_input,
            )

            yield self._token_budget_chunk(
                messages,
                scope="canonical_iteration",
                model=request.model,
                iteration=iteration,
            )
            compacted, before, after = await self._compact_messages_with_synthesis(messages, request)
            if compacted:
                yield self._meta_chunk(
                    phase="executing", status="compaction",
                    model=request.model,
                    extra={
                        "iteration": iteration,
                        "tokens_before": before,
                        "tokens_after": after,
                    },
                )
                yield self._token_budget_chunk(
                    messages,
                    scope="canonical_iteration",
                    model=request.model,
                    iteration=iteration,
                    compacted=True,
                )

            # Live drain + shared thinking policy (default OFF, composer
            # toggle wins) — see the native loop for both rationales.
            _live_result: list = []
            async for ev in self._stream_and_parse_live(
                request, messages, tool_schemas, tool_map, tier, iteration,
                result_out=_live_result,
                chat_template_kwargs=_iteration_thinking_kwargs(request),
            ):
                yield ev
            full_content, tool_calls, error_kind, full_thinking, error_status, error_message = _live_result[0]
            if error_kind:
                termination_reason = "backend_error"
                # Transient errors (429 / 5xx / network) exhausted the
                # retry budget — the next attempt is still likely to work,
                # so emit ``recoverable_error`` so the UI can offer a
                # Try Again pill instead of dead-ending the turn. Permanent
                # errors (4xx) hard-stop — retrying just hits the same wall.
                ui_status = (
                    "recoverable_error"
                    if error_kind == "transient" else "error"
                )
                yield emit(
                    _format_iteration_error(iteration, error_kind, error_status, error_message),
                    phase="executing", status=ui_status,
                    model=request.model,
                    extra={
                        "error_kind": error_kind,
                        "iteration": iteration,
                        "retry_status_code": error_status,
                        "error_message": error_message,
                    },
                )
                break

            # Emit cleaned prose
            clean_text = _strip_tool_json(full_content)  # noqa: F821
            clean_text = _strip_cot_tokens(clean_text)  # noqa: F821
            clean_text = re.sub(r'```[\s\S]*?```', '', clean_text)
            clean_text = re.sub(r'\n{3,}', '\n\n', clean_text).strip()
            clean_text = _strip_act_monologue(clean_text)
            if clean_text:
                yield emit(
                    clean_text + "\n",
                    phase="executing", status="streaming",
                    model=request.model,
                )

            # Record assistant turn in history
            self._append_assistant_to_history(
                messages, full_content, tool_calls, tier,
                thinking=full_thinking,
            )

            # Consensus stop signal: model didn't call any tools
            if not tool_calls:
                if (
                    self._turn_intent_for_turn.kind in _ACTIONABLE_TURN_KINDS
                    and _looks_like_future_action_prose(clean_text)
                    and not progress_without_action_nudged
                ):
                    messages.append(Message(
                        role="user",
                        content=(
                            "<nudge>You described next steps, but you didn't "
                            "actually do them. For actionable tasks, don't stop "
                            "on a plan or progress note. Emit the next real tool "
                            "call now, or explain the blocker plainly if you "
                            "cannot proceed.</nudge>"
                        ),
                    ))
                    progress_without_action_nudged = True
                    yield self._meta_chunk(
                        phase="executing",
                        status="progress_without_action_nudge",
                        model=request.model,
                        extra={"iteration": iteration},
                    )
                    continue
                if (
                    not operate_completion_nudged
                ):
                    pending_contract = _pending_contract_for_turn(
                        intent_kind=self._turn_intent_for_turn.kind,
                        goal_text=user_goal or latest_input,
                        candidate_text=clean_text,
                        messages=messages,
                        tool_results=turn_tool_results,
                    )
                    completion_nudge = _completion_nudge_for_turn(
                        intent_kind=self._turn_intent_for_turn.kind,
                        goal_text=user_goal or latest_input,
                        candidate_text=clean_text,
                        messages=messages,
                        tool_results=turn_tool_results,
                    )
                else:
                    completion_nudge = ""
                if completion_nudge:
                    self._state.set_pending_objective_contract(pending_contract)
                    messages.append(Message(
                        role="user",
                        content=completion_nudge,
                    ))
                    operate_completion_nudged = True
                    yield self._meta_chunk(
                        phase="executing",
                        status="operate_evidence_nudge",
                        model=request.model,
                        extra={"iteration": iteration},
                    )
                    continue
                self._state.clear_pending_objective_contract()
                termination_reason = "model_stop"
                break

            # Execute tools sequentially (Claude-Code style)
            counters: dict[str, int] = {
                "writes": 0, "validation_errors": 0, "tool_calls": 0,
                "iteration": iteration,
            }
            for tc in tool_calls:
                async for ev in self._run_tool_tracked(
                    tc, tool_map, tier, messages, request.model, counters,
                ):
                    yield _tap_tool_result(ev, turn_tool_results)

            # Mid-turn persist for inspector live updates. Cheap when
            # task_list / mission haven't changed (hash-guarded inside
            # ``_persist_state_if_dirty``). Originally only wired in
            # _act_hybrid; added here 2026-05-31 so canonical also
            # reflects task_list mutations in the inspector without
            # waiting for end-of-turn. Gated on whether any call this
            # iteration was task_list so we don't pay the hash cost
            # on every iteration of every turn.
            if any(
                (tc.get("function") or {}).get("name") == "task_list"
                or tc.get("name") == "task_list"
                for tc in (tool_calls or [])
            ):
                try:
                    await self._persist_state_if_dirty()
                except Exception:
                    log.debug("coder_mid_turn_persist_failed", exc_info=True)

            total_writes += counters["writes"]
            if counters["tool_calls"] > 0:
                operate_completion_nudged = False

            # Clear stale blockers on any successful call (mirror hybrid).
            if (
                counters["tool_calls"] > 0
                and counters["validation_errors"] < counters["tool_calls"]
            ):
                self._state.clear_validation_errors()

            # Circuit breaker: if every tool call this iteration failed
            # input validation, count it against a streak. See the same
            # block in _act_hybrid for rationale (module constant).
            if (
                counters["tool_calls"] > 0
                and counters["validation_errors"] == counters["tool_calls"]
            ):
                validation_error_streak += 1
                if (
                    self._state.safeguards_enabled
                    and validation_error_streak >= _live_threshold("validation_error_streak")
                ):
                    termination_reason = "validation_error_streak"
                    yield emit(
                        (
                            f"\n\n[Stopped: {validation_error_streak} consecutive "
                            "iterations of malformed tool calls. Try a different "
                            "model or rephrase the task.]\n"
                        ),
                        phase="executing", status="validation_error_break",
                        model=request.model,
                        extra={"streak": validation_error_streak},
                    )
                    async for _rev in self._reflect_on_streak_break(
                        request, break_kind="validation_error_streak",
                        streak=validation_error_streak,
                    ):
                        yield _rev
                    break
            else:
                validation_error_streak = 0

            # Same-signature break: catches the common "file_write without
            # path, model loops 5×" pattern. Fires after 2 identical
            # consecutive failures of the same tool — far tighter than the
            # 5-iteration check above, and unambiguous about WHY we
            # stopped (the model can't escape an identical error).
            if self._state.safeguards_enabled:
                offender = _find_repeat_offender(self._state)
                if offender is not None:
                    termination_reason = "same_validation_error_repeat"
                    yield emit(
                        _format_repeat_break_message(offender),
                        phase="executing", status="validation_error_break",
                        model=request.model,
                        extra={
                            "tool": offender.get("tool"),
                            "repeat_count": int(offender.get("repeat_count") or 0),
                        },
                    )
                    async for _rev in self._reflect_on_streak_break(
                        request, break_kind="same_validation_error_repeat",
                        streak=int(offender.get("repeat_count") or 0),
                    ):
                        yield _rev
                    break

        else:
            # while-else: cap hit without natural termination
            termination_reason = "max_iterations_reached"
            yield self._meta_chunk(
                phase="executing", status="max_iterations_reached",
                model=request.model, extra={"reason": termination_reason},
            )

        # Persist this turn's trace for the next user turn's system
        # prompt. Mirrors _act_hybrid's write so the inspector's
        # "Prior turns" panel and the next turn's <prior_turns> block
        # are populated regardless of strategy. Gated on having any
        # tool exchanges — pure-prose turns leave no trace.
        if turn_tool_results or any(m.role == "tool" for m in messages):
            summary = self._build_turn_summary(
                messages=messages, user_goal=user_goal,
                termination_reason=termination_reason,
            )
            self._state.add_turn_summary(summary)
            # Durable archive — beyond the cap-10 FIFO. Phase 1 of
            # the LTM design (see project_coder_turn_archive memory).
            await self._archive_turn_summary(summary)

        # Publish the reviewable-turn bundle (no-op if no registry
        # wired, no snapshot attached, or zero diffs). Must happen
        # BEFORE the "complete" meta chunk so the frontend can look up
        # the bundle synchronously when it sees the turn end.
        await self._publish_turn_review(user_goal)

        self._state.phase = CoderPhase.WAITING
        yield self._meta_chunk(
            phase="waiting", status="complete",
            model=request.model,
            extra={
                "strategy": "canonical",
                "tool_calls_made": self._state.tool_calls_made,
                "termination_reason": termination_reason,
                "iterations_used": iteration,
                "review_turn_id": self._state.active_turn_id,
            },
        )

    # ==================================================================
    # Native strategy — Claude Code / Qwen Code / OpenCode parity
    # ==================================================================
    # Pure minimal loop for capable native-tool-calling models. Drops
    # nearly all Augmentum scaffolding that exists to compensate for
    # weaker models:
    #
    #   - No plan phase (caller skips it before dispatch).
    #   - No semantic-heavy to_act_context.
    #   - No Termination Quality Gate, no nudges (insistence/bailout/
    #     no-progress), no sticky reminder, no monologue stripping,
    #     no observation refresh, no stagnation detector, no streak
    #     breakers, no read-only / populated-repo / re-inspection
    #     refusals, no per-tool permission gate, no post-write lint /
    #     syntax check.
    #   - No workspace guide, sticky reminder, or controller/active
    #     power blocks.
    #
    # Contract: build messages = NATIVE_SYSTEM + bounded native context
    # prelude + user history. Call backend with native tools. If the
    # model emits tool_calls, run independent reads in parallel and
    # serialize stateful calls, append results, continue. If the model
    # emits no tool_calls, the turn ends — same signal Claude Code /
    # Qwen Code rely on.
    #
    # Iteration cap (AUGMENTUM_CODER_MAX_ITERS, default 150) is the
    # only failsafe; the model should hit a natural stop long before.
    # ==================================================================

    async def _act_native(
        self,
        request: InternalChatRequest,
        workspace_context: str,
    ) -> AsyncIterator[InternalStreamChunk]:
        """Minimal Claude-Code/Qwen-Code style loop. Native tool calling only."""
        import asyncio as _asyncio

        from augmentum.modes.coder.handler import (
            _execute_tool,
            _execute_tool_with_shell_stream,
        )
        from augmentum.tools.base import ToolResult

        tools = create_coder_tools(  # noqa: F821 — bound by _bind_handler_helpers
            self._container_manager, self._workspace_id, self._state,
            executor=getattr(self, "_executor", None),
            tool_registry=self._tool_registry,
            question_callback=self._question_callback,
            profile_store=getattr(self, "_profile_store", None),
            service_store=getattr(self, "_service_store", None),
            user_id=getattr(self, "_user_id", ""),
            planning_mode=getattr(self, "_planning_mode", "default"),
            subagent_dispatcher=self._get_subagent_dispatcher(),
            strict_edit_guard=False,
            db_conn=self._resolve_archive_conn(),
            jobs_store=getattr(self, "_jobs_store", None),
        )
        # Native CLIs terminate with a normal assistant response: the
        # model simply stops calling tools and writes final prose. The
        # finish_task pseudo-tool exists for Augmentum's forced-tool
        # hybrid paths; exposing it here teaches Qwen to finish via a
        # hidden tool argument instead of a visible assistant message.
        tools = [t for t in tools if t.name != "finish_task"]
        # Elective reasoning (Anthropic think-tool pattern): expose the
        # `think` tool ONLY when native per-turn thinking is OFF for this
        # turn — a model already reasoning before every tool call doesn't
        # need an explicit think tool (it would be redundant, and on an
        # always-on family it's just ignored). Behind coder_think_tool_enabled
        # (default off) so it can be A/B'd without changing default behavior.
        # `expose_think` is reused below to gate the matching prompt teaching.
        from augmentum.config import settings as _think_settings
        expose_think = bool(
            getattr(_think_settings, "coder_think_tool_enabled", False)
            and not _iteration_thinking_kwargs(request).get("enable_thinking")
        )
        if not expose_think:
            tools = [t for t in tools if t.name != "think"]
        # Model-initiated compaction (default off). Constructed here
        # rather than via ALL_CODER_TOOLS because only the native loop
        # consumes the signal flag (_compact_messages_with_synthesis) —
        # registering it globally would hand subagents / bug-finder /
        # builds a tool that acks a fold nothing performs.
        if bool(getattr(_think_settings, "coder_compact_tool_enabled", False)):
            from augmentum.coder.tools import CompactTool
            tools.append(CompactTool(
                container_manager=self._container_manager,
                workspace_id=self._workspace_id,
                state=self._state,
                profile_store=getattr(self, "_profile_store", None),
                service_store=getattr(self, "_service_store", None),
                user_id=getattr(self, "_user_id", ""),
                strict_edit_guard=False,
            ))
        tool_map = {t.name: t for t in tools}

        # Native strategy is opt-in. The user picking "Native" in the UI
        # IS the contract — they're telling us "trust me, this model can
        # do tools." Don't second-guess via select_tier. Hard-force
        # NATIVE so the message history is formatted consistently and
        # tools always go on the wire as ``tools=[...]``. If the backend
        # or model genuinely can't handle native function calling, the
        # worst case is one iteration where tool_calls comes back empty
        # and we exit cleanly — far better UX than a hard refusal that
        # second-guesses the user's explicit choice.
        tier = ToolCallingTier.NATIVE
        tool_schemas = [_tool_to_schema(t) for t in tools]  # noqa: F821

        yield self._meta_chunk(
            phase="executing", status="strategy",
            model=request.model, extra={"strategy": "native"},
        )

        # Build a stripped message list: NATIVE_SYSTEM + a bounded
        # native context prelude + (user-supplied system, if any) + the
        # user's conversation. No workspace guide, datetime block, or
        # power scaffolding. Mimics the "system prompt + chat" contract
        # of Claude Code / Qwen Code while retaining a few cheap
        # Augmentum grounding facts. The sticky <system-reminder> IS
        # rendered per iteration since 2026-07-06 (Claude Code has one
        # too) — it's the plan spine's visibility channel; see the
        # injection at the top of the loop below.
        user_messages = list(request.messages)
        sys_text = NATIVE_SYSTEM
        native_context = (workspace_context or "").strip()
        if native_context:
            sys_text += "\n\n" + native_context
        # Workspace-kernel v2 hint — native previously didn't render
        # this block because it builds its own sys_text instead of
        # going through CoderHandler._build_messages. Symptom (live
        # log 2026-05-28): a "continue"-after-cancel turn on Qwen3.6
        # ran six silent shell_exec calls trying to find credentials
        # because the model didn't know `/workspace/.augmentum/plan.md`
        # existed with the prior turn's context. Single-sourced from
        # ``WorkspaceKernel.hint_text`` (static method) so wording
        # stays in lockstep with the canonical/hybrid path.
        from augmentum.coder.workspace_kernel import WorkspaceKernel as _Kernel
        kernel_hint = _Kernel.hint_text()
        if kernel_hint:
            sys_text += "\n\n" + kernel_hint

        # Workspace facts block — compact objective + identity +
        # constraint/gotcha observations. Sourced from the handler's
        # turn-start cache (``self._cached_facts_block``, refreshed
        # by ``CoderHandler._refresh_kernel_facts``) so every
        # strategy renders the same content. Single source of truth.
        # Empty string is benign — appended only when non-empty.
        cached_facts = getattr(self, "_cached_facts_block", "") or ""
        if cached_facts:
            sys_text += "\n\n" + cached_facts

        # Repo-map block — same turn-start cache as the facts block, so
        # native gets the same structural grounding as canonical/hybrid
        # (which read it via _build_messages). Byte-stable by design.
        repo_map = getattr(self, "_repo_map_context_block", "") or ""
        if repo_map:
            sys_text += "\n\n" + repo_map

        # Working Agreements — the user's durable, model-agnostic "how this
        # maker wants to be worked with" principles (mig 273). Injected once
        # per turn so ANY local model inherits the accrued relationship, not
        # just one. No-op until the user has accrued any (ships empty). This
        # is the substrate for "the assistant accrues a relationship too,
        # not only the companion." See augmentum/coder/maker_agreements.py.
        agreements_block = await self._render_maker_agreements_block()
        if agreements_block:
            sys_text += "\n\n" + agreements_block

        # Plan-mode notice — frames the tool-list filter as deliberate.
        # No-op outside plan mode.
        _plan_notice = self._plan_mode_addendum()
        if _plan_notice:
            sys_text += _plan_notice

        # Elective-reasoning teaching — appended ONLY when the think tool is
        # actually exposed this turn (gated identically above), so a turn
        # without the tool never advertises it. Schema description alone
        # doesn't shift local-model habits (batch-read "0 adoption" lesson);
        # this prompt line is the adoption lever.
        if expose_think:
            sys_text += "\n\n" + NATIVE_THINK_TOOL_TEACHING
        # Same adoption lever for the compact tool — teaching appended
        # ONLY when the tool is exposed (gate mirrored above).
        if bool(getattr(_think_settings, "coder_compact_tool_enabled", False)):
            from augmentum.coder.prompts import NATIVE_COMPACT_TOOL_TEACHING
            sys_text += "\n\n" + NATIVE_COMPACT_TOOL_TEACHING

        if user_messages and user_messages[0].role == "system":
            existing = re.sub(
                r"<current_time>.*?</current_time>\n*",
                "",
                user_messages[0].content,
                flags=re.DOTALL,
            ).strip()
            if existing:
                sys_text += "\n\n" + existing
            user_messages = user_messages[1:]

        # Continuation-after-cancellation guard. Narrow trigger: only
        # fires when (a) the latest user message is a continuation
        # request ("continue", "keep going", "please") AND (b) the
        # most recent turn_summary is marked cancelled. The combo
        # caught a real failure 2026-05-28 — user said "continue"
        # after cancelling the credentials-setup turn, model resumed
        # against fragmentary context and looped on silent shells
        # trying to "find the credentials" that were only mentioned
        # in the cancelled turn's prose.
        #
        # The addendum tells the model to restate-then-act so the
        # user sees what's about to happen before any tool runs.
        # ``<prior_turns>`` already carries the INTERRUPTED stanza
        # (see CoderHandler._render_interruption_stanza) — this
        # directive points the model AT that stanza rather than
        # letting "continue" override the "don't silently resume"
        # warning baked into the prior_turns block intro.
        last_user_msg = ""
        for _m in reversed(user_messages):
            if getattr(_m, "role", "") == "user":
                last_user_msg = (getattr(_m, "content", "") or "").strip()
                break
        last_summary = (
            self._state.turn_summaries[-1]
            if self._state.turn_summaries else {}
        )
        if (
            _is_continuation_request(last_user_msg)  # noqa: F821 — bound helper
            and last_summary.get("cancelled")
        ):
            sys_text += (
                "\n\n<continuation_after_cancel>\n"
                "The user just asked you to continue, but the previous "
                "turn was cancelled mid-flight. Before you run any "
                "tool: state in one sentence what you understood the "
                "user's ORIGINAL ask to be — read the INTERRUPTED "
                "stanza in prior_turns to recover it. If that ask is "
                "unambiguous from the stanza, proceed normally. If "
                "you cannot reconstruct what was being worked on, "
                "ask ONE targeted clarifying question instead of "
                "guessing. Running shell commands against a half-"
                "remembered goal wastes the user's time and produces "
                "exactly the silent-success loops we keep seeing.\n"
                "</continuation_after_cancel>"
            )

        # Hydrated recency buffer: replace the client's collapsed prior
        # turns (getMessagesForLLM strips tool_calls/results/thinking) with
        # the last N turns' FULL in-format chains, so the model sees its
        # own recent tool-using behavior and keeps interleaving on
        # follow-ups instead of lapsing to prose-only answers. ``fresh`` is
        # this turn's new input; it's also stashed for the turn-end capture
        # so the two halves agree on the seam. No-op fallback when the
        # buffer is empty (first turn / fresh handler).
        fresh_input = self._fresh_turn_input(user_messages)
        user_messages = self._apply_recency_buffer(user_messages, fresh_input)
        self._pending_turn_input = fresh_input

        messages: list[Message] = [
            Message(role="system", content=sys_text),
            *user_messages,
        ]

        # Per-turn runtime carrier. Native deliberately omits the power
        # scaffolding (see the "stripped message list" contract at the
        # top of this method), so we cannot reuse
        # _build_runtime_carrier_message wholesale; that helper includes
        # those blocks for the canonical/hybrid path. The carrier itself
        # uses the same user-role framing as
        # ``CoderHandler._build_messages`` so per-turn-mutating content
        # lives at the tail rather than in ``sys_text`` — that move was
        # the whole point: prior_turns mutated every full turn and
        # invalidated the prefix cache for the entire system text below
        # it. Same ordering contract as the canonical carrier: append-
        # mostly prior_turns first, per-query auto-recall next,
        # <current_time> LAST (mutates by the minute, so it sits at the
        # extreme tail of the stable zone — recall/prior stanzas carry
        # ABSOLUTE timestamps the model can diff against it).
        from augmentum.modes.coder.chat_egress import (
            RUNTIME_CARRIER_HEADER as _CARRIER_HEADER,
        )
        from augmentum.utils.datetime_context import (
            get_datetime_context as _get_dt_context,
        )

        carrier_parts = [
            p
            for p in [
                self._render_prior_turns(),
                getattr(self, "_turn_dynamic_context_block", "") or "",
                _get_dt_context(),
            ]
            if p
        ]
        if carrier_parts:
            carrier = Message(
                role="user",
                content=(
                    f"{_CARRIER_HEADER}\n"
                    "Treat the following block as authoritative state "
                    "at the start of this turn. Do not answer it; use "
                    "it when relevant to the user message that "
                    f"follows.\n\n" + "\n\n".join(carrier_parts)
                ),
            )
            insert_at = self._last_user_index(messages)
            if insert_at is None:
                messages.append(carrier)
            else:
                messages.insert(insert_at, carrier)

        iteration = 0
        termination_reason = "model_stop"
        # Goal split is used only by _publish_turn_review at the end so
        # the review bundle gets a sensible label. The native loop itself
        # doesn't classify or branch on intent.
        _, user_goal = _extract_goal_split(request.messages)  # noqa: F821

        # Latest user message text — fed to the Termination Quality Gate
        # so it can classify demand (PASSIVE / ACTIVE / INSISTENT). Native
        # deliberately does NOT classify intent (kept as UNKNOWN below);
        # the gate biases UNKNOWN toward ACTION which is the safer default
        # for a coding agent.
        latest_input = ""
        for _msg in reversed(request.messages):
            if _msg.role == "user":
                latest_input = (_msg.content or "").strip()
                break

        if getattr(self._state, "safeguards_enabled", True):
            max_iters = _live_max_iters()
        else:
            max_iters = _live_max_iters(ungated=True)
        empty_stop_retries = 0
        max_empty_stop_retries = 2

        # Termination Quality Gate state — mirrors hybrid (phase_act.py
        # line ~1724). Pre-2026-05-27 native broke immediately on the
        # first prose-with-no-tools response, which caught Qwen-3.6 mid-
        # preamble ("Let me check that for you.", content_len=55,
        # tool_call_count=0) and terminated the turn before the model
        # ever got to call a tool. The gate distinguishes "model bailed
        # before doing work" from "model wrapped up after doing work"
        # so the loop nudges in the first case and accepts in the second.
        #
        # Native diverges from hybrid in ONE place: ``progress_iters``
        # tracks successful tool calls of any kind (reads OR writes),
        # not just writes. Native doesn't classify turn intent — a
        # read-only inspect turn ("show me the file structure") looks
        # the same as an implement turn at this layer. Treating any
        # tool use as progress lets the gate accept the legitimate
        # read+summary pattern (file_list → "Found index.html.") while
        # still nudging the read-zero-then-bail pattern that motivated
        # this fix. ``total_writes`` stays writes-only so the gate's
        # INSISTENT rule still trips when the user demanded completion
        # and the model only inspected.
        progress_iters: collections.deque[int] = collections.deque(
            maxlen=_HYBRID_CONTINUATION_LOOKBACK,
        )
        total_writes = 0
        # Pre-2026-05-31 native capped at 1 nudge — the gate accepted
        # the second prose-no-tools response via REASON_ALREADY_NUDGED.
        # Observed footgun: Qwen-3.6 emits a 78-char preamble, gets
        # nudged, emits another 97-char preamble, gate stops the turn.
        # Now tunable via ``coder_native_nudge_max`` (default 2). We
        # count nudges and only flip the gate's "already nudged" flag
        # once the cap is reached, so writes-progress can still reset
        # the streak mid-turn.
        native_nudge_streak = 0
        native_nudge_cap = live_native_nudge_max()
        continuation_nudged = False
        # Goal-judge re-entries this turn (MiMo-style stop gate; see
        # augmentum/coder/goal_judge.py). Independent cap from the
        # nudge streak — the judge fires on ACCEPTED stops only.
        goal_judge_streak = 0
        # Verification-gate re-entries this turn (Arbor principle #2; see
        # augmentum/coder/verify_command.py). The held-out mechanical
        # gate runs BEFORE the goal judge on accepted write-stops — a real
        # non-zero exit is authoritative, so the agent can't argue past it.
        verify_gate_streak = 0

        # Silent-success-fog detector — observed 2026-05-28 (real user
        # log): on a "continue"-after-cancel turn the model ran 6+
        # shell_exec calls in a row, every one returning "(exit 0,
        # command succeeded with no stdout)". The model had no
        # grounding signal so it kept trying — classic fog spiral.
        # Hybrid has this detector (line ~2836); native didn't.
        # Threshold reuses ``_SILENT_SUCCESS_NUDGE_AT`` (default 3,
        # env-tunable via ``AUGMENTUM_CODER_SILENT_SUCCESS_STREAK``).
        # One-shot per turn — model isn't nagged repeatedly.
        silent_success_streak = 0
        silent_success_nudge_fired = False

        # Identical-call loop detector state. Maps a (tool, args, output)
        # signature → consecutive-iteration count; one-shot nudge per turn.
        # See _identical_result_signature / _bump_identical_streaks.
        identical_result_streaks: dict[str, int] = {}
        identical_result_nudge_fired = False

        # Loop-health coordinator — ONE subsystem over the guard
        # ladders (loop_health.py). Owns tracker construction (single
        # threshold table), per-turn counters, and the at-most-one-
        # nudge-per-iteration arbitration; the per-detector rationale
        # comments live in each tracker's module docstring. Locals
        # alias the owned trackers so the observation sites below read
        # unchanged.
        from augmentum.coder.loop_health import LoopHealthCoordinator
        health = LoopHealthCoordinator.create(
            threshold=_live_threshold,
            tasks=self._state.tasks,
            tracked_read_tools=READ_ONLY_TOOLS,
        )
        task_spine = health.task_spine
        write_churn = health.write_churn
        probe_signal = health.probe_signal
        command_carousel = health.command_carousel
        progress_ledger = health.progress_ledger
        duplicate_calls = health.duplicate_calls
        code_intel_adoption = health.code_intel

        # Stagnation → buddy-model escalation. Tracks repeated-failure
        # signals across iterations within this turn. When tripped AND
        # the workspace has a heavyweight model configured, swap
        # request.model so the remaining iterations run on the buddy.
        # Otherwise fall through to the existing recoverable-error pill
        # — the user keeps control via the visible "Try Again" affordance.
        consecutive_validation_errors = 0
        consecutive_no_progress = 0
        escalated_buddy: str = ""

        while iteration < max_iters:
            iteration += 1

            # Cooperative pause-gate + steer-message inbox drain.
            # No-op outside the background-runs broker path.
            _steer = await self._coop_iteration_check()
            if _steer:
                for _s in _steer:
                    messages.append(Message(
                        role="user",
                        content=self._format_steer_content(_s),
                    ))
                yield self._meta_chunk(
                    phase="executing", status="steer_delivered",
                    model=request.model,
                    extra={"count": len(_steer), "iteration": iteration},
                )

            # True native agent loop: do not force a tool. Claude Code,
            # Qwen Code, OpenCode, and Codex-style runners give the
            # model the tool schemas, let it choose zero or more calls,
            # append tool results, and repeat until it stops. llama.cpp
            # also rejects ``tool_choice="required"`` for this Qwen
            # build with "Failed to initialize samplers", so omitting
            # tool_choice is both more faithful and more compatible.
            iter_tool_choice = None
            # Thinking-mode policy — default OFF, coder-composer toggle
            # wins. Shared across strategies; see _iteration_thinking_kwargs.
            iter_template_kwargs = _iteration_thinking_kwargs(request)

            # Sticky reminder — re-rendered each iteration (Claude
            # Code's system-reminder pattern; same helper as hybrid /
            # canonical). Native's original "stripped message list"
            # contract omitted it, which left the task_list plan spine
            # invisible: the model could write its plan but never saw
            # it again, so it never maintained one. Injected BEFORE
            # compaction so it lives in the verbatim tail; tail-
            # positioned appends leave the stable prefix above it
            # untouched (KV contract holds).
            from augmentum.config import settings as _sr_settings
            plan_md = (
                "" if _sr_settings.coder_kernel_v2
                else await self._read_plan_md()
            )
            self._inject_sticky_reminder(
                messages, goal=user_goal, iteration=iteration,
                max_iters=max_iters, writes=total_writes,
                plan_md=plan_md, latest_input=latest_input,
            )

            yield self._token_budget_chunk(
                messages,
                scope="native_iteration",
                model=request.model,
                iteration=iteration,
            )
            compacted, before, after = await self._compact_messages_with_synthesis(messages, request)
            if compacted:
                yield self._meta_chunk(
                    phase="executing",
                    status="compaction",
                    model=request.model,
                    extra={
                        "scope": "native_iteration",
                        "iteration": iteration,
                        "tokens_before": before,
                        "tokens_after": after,
                    },
                )
                yield self._token_budget_chunk(
                    messages,
                    scope="native_iteration",
                    model=request.model,
                    iteration=iteration,
                    compacted=True,
                )

            # Live drain: progress transitions + coalesced reasoning
            # deltas reach the UI WHILE the model generates (the buffered
            # progress_events list used to flush only after the call
            # returned — dead air for the whole generation window).
            _live_result: list = []
            async for ev in self._stream_and_parse_live(
                request, messages, tool_schemas, tool_map, tier, iteration,
                result_out=_live_result,
                tool_choice=iter_tool_choice,
                chat_template_kwargs=iter_template_kwargs,
            ):
                yield ev
            full_content, tool_calls, error_kind, full_thinking, error_status, error_message = _live_result[0]
            # Brief instrumentation for native bring-up. Tells us at a
            # glance whether the loop is exiting because the model
            # genuinely stopped or because something dropped tool_calls
            # in transit. Cheap (one log line per iteration).
            log.info(
                "coder.native_iter",
                iteration=iteration,
                content_len=len(full_content or ""),
                tool_call_count=len(tool_calls or []),
                tool_call_names=[
                    (tc.get("name") or "?") for tc in (tool_calls or [])
                ][:5],
                tool_choice=iter_tool_choice or "auto",
                error_kind=error_kind or "",
            )
            if error_kind:
                termination_reason = "backend_error"
                ui_status = (
                    "recoverable_error"
                    if error_kind == "transient" else "error"
                )
                yield emit(
                    _format_iteration_error(iteration, error_kind, error_status, error_message),
                    phase="executing", status=ui_status,
                    model=request.model,
                    extra={
                        "error_kind": error_kind,
                        "iteration": iteration,
                        "retry_status_code": error_status,
                        "error_message": error_message,
                    },
                )
                break

            # Stream the model's prose verbatim — no monologue strip,
            # no tool-JSON strip, no triple-backtick strip. Native mode
            # surfaces what the model said so the user gets the same
            # transparency Claude Code / Qwen Code provide.
            if full_content.strip():
                yield emit(
                    full_content + "\n",
                    phase="executing", status="streaming",
                    model=request.model,
                )

            self._append_assistant_to_history(
                messages, full_content, tool_calls, tier,
                thinking=full_thinking,
            )

            # Termination contract: zero tool_calls.
            #
            # Two paths:
            #   - Zero prose AND zero tools — the model said nothing at
            #     all. Use the bounded retry path (empty_stop_retries)
            #     with a generic "continue or answer visibly" nudge.
            #     The Termination Quality Gate would also catch this
            #     case (ProseKind.EMPTY → NUDGE_NO_PROGRESS), but the
            #     retry-count cap is the load-bearing safeguard against
            #     a model that just refuses to emit anything.
            #   - Prose WITH content but no tools — feed through the
            #     gate (same one hybrid uses at phase_act.py:2243). The
            #     gate decides accept-vs-nudge based on whether the
            #     prose is substantive, whether writes happened
            #     recently, and whether the user demanded completion.
            #     Pre-2026-05-27 this branch was an unconditional
            #     break, which caused Qwen-3.6 to terminate mid-preamble
            #     ("Let me check that for you.") before ever calling a
            #     tool. The gate fixes that.
            if not tool_calls:
                if not full_content.strip():
                    termination_reason = "empty_model_stop"
                    if empty_stop_retries >= max_empty_stop_retries:
                        yield emit(
                            "\n\n[Model stopped without a visible answer]\n",
                            phase="executing",
                            status="error",
                            model=request.model,
                        )
                        break
                    empty_stop_retries += 1
                    yield self._meta_chunk(
                        phase="executing",
                        status="empty_model_stop_retry",
                        model=request.model,
                        extra={
                            "iteration": iteration,
                            "retry": empty_stop_retries,
                            "max_retries": max_empty_stop_retries,
                        },
                    )
                    messages.append(Message(
                        role="user",
                        content=(
                            "You stopped without a visible answer and without "
                            "calling a tool. Continue the task now. If the "
                            "answer depends on current state, inspect it with "
                            "the available tools. Otherwise answer visibly."
                        ),
                    ))
                    continue

                # Leaked-tool-markup gate (2026-07-02, Qwythos-9B run
                # …0d0d4a6ebc): the model wrote Qwen-XML tool calls
                # INSIDE its thinking channel, where no parser is
                # listening, then reasoned "I've already gathered a lot
                # of information" (it hadn't — nothing executed) and
                # wrapped up. A stop candidate whose reasoning or prose
                # carries raw markup that never became a parsed call is
                # built on actions that never happened — nudge channel
                # discipline instead of accepting. Bounded by the shared
                # native nudge cap so a model that can't comply still
                # terminates.
                if (
                    (_has_leaked_tool_markup(full_thinking)  # noqa: F821
                     or _has_leaked_tool_markup(full_content))  # noqa: F821
                    and native_nudge_streak < native_nudge_cap
                ):
                    native_nudge_streak += 1
                    yield self._meta_chunk(
                        phase="executing",
                        status="leaked_tool_markup_nudge",
                        model=request.model,
                        extra={"iteration": iteration},
                    )
                    messages.append(Message(
                        role="user",
                        content=(
                            "<nudge>Your last response contained raw "
                            "tool-call markup (e.g. <tool_call> / "
                            "<function=...>) inside your reasoning or "
                            "prose. Those calls were NOT executed — "
                            "nothing runs while you are still thinking. "
                            "Finish your reasoning first, then emit the "
                            "calls as REAL function calls so they "
                            "actually run. Do not assume any result you "
                            "have not seen in a tool result "
                            "message.</nudge>"
                        ),
                    ))
                    continue

                had_recent_progress = any(
                    iteration - pi <= _HYBRID_CONTINUATION_LOOKBACK
                    for pi in progress_iters
                )
                verdict = evaluate_termination(TerminationContext(
                    user_text=latest_input,
                    # Native deliberately doesn't classify intent — pass
                    # UNKNOWN. The gate biases UNKNOWN toward ACTION via
                    # intent_is_action(), so a short prose with zero
                    # writes nudges (matching the user's "force a tool
                    # call at the beginning" intuition) while a
                    # substantive prose still accepts (the model
                    # legitimately answered without needing tools).
                    intent_kind=TurnIntentKind.UNKNOWN,
                    clean_prose=full_content,
                    total_writes=total_writes,
                    had_recent_progress=had_recent_progress,
                    continuation_nudged=continuation_nudged,
                ))

                # Native override: the gate's SUBSTANTIVE classifier
                # (>= 30 chars + >= 2 sentences) was tuned for hybrid's
                # completion summaries. Qwen-3.6 in native produces
                # 2-sentence preambles ("I'll look at this. Let me
                # check the file.") that pass that bar and short-circuit
                # the turn before any tool call. While the model has
                # done literally nothing (zero writes, zero recent
                # progress) AND we still have nudge budget, prefer
                # pushing for action over accepting a preamble. Once
                # the cap is hit we accept whatever prose came so we
                # never infinite-loop.
                _is_substantive_active_accept = (
                    verdict.accept_stop
                    and verdict.reason == REASON_SUBSTANTIVE_ACTIVE
                    and total_writes == 0
                    and not had_recent_progress
                    and native_nudge_streak < native_nudge_cap
                )
                if (
                    _is_substantive_active_accept
                    and len(full_content.strip()) < _NATIVE_PREAMBLE_OVERRIDE_MAX_CHARS
                ):
                    # Length override fires first (cheap). Covers the
                    # common Qwen-3.6 preamble case at zero latency
                    # cost.
                    verdict = TerminationVerdict(
                        accept_stop=False,
                        reason=REASON_NUDGE_BAILOUT,
                        nudge_kind=NUDGE_BAILOUT,
                    )
                elif _is_substantive_active_accept:
                    # Longer prose (>= 200 chars) — could be a real
                    # explanation, could be a chatty stall. Delegate
                    # to the next-speaker classifier (Qwen-Code-style
                    # second LLM call) when enabled. Best-effort; a
                    # None verdict falls back to accepting the stop.
                    from augmentum.config import settings as _ns_settings
                    if bool(getattr(_ns_settings, "coder_next_speaker_check_enabled", True)):
                        from augmentum.coder.next_speaker import (
                            check_next_speaker as _check_next_speaker,
                        )
                        ns_verdict = await _check_next_speaker(
                            self._backend,
                            source_request=request,
                            messages=messages,
                        )
                        if ns_verdict.next_speaker == "model":
                            yield self._meta_chunk(
                                phase="executing",
                                status="next_speaker_check",
                                model=request.model,
                                extra={
                                    "iteration": iteration,
                                    "strategy": "native",
                                    "next_speaker": "model",
                                    "reasoning": ns_verdict.reasoning,
                                },
                            )
                            verdict = TerminationVerdict(
                                accept_stop=False,
                                reason=REASON_NUDGE_BAILOUT,
                                nudge_kind=NUDGE_BAILOUT,
                            )

                if verdict.accept_stop:
                    # Plan-spine stop gate (task_spine.py): the model
                    # engaged the task list THIS TURN and is stopping
                    # with open items — one nudge to finish them or
                    # update the list to what actually happened. Runs
                    # before the verify/goal-judge gates because it's
                    # free (no I/O, no LLM call). One-shot: a repeat
                    # stop passes so this can never loop. The engaged-
                    # this-turn guard keeps a leftover list from a
                    # PRIOR turn from blocking an unrelated stop.
                    _pending_nudge = task_spine.stop_gate_nudge(
                        self._state.tasks,
                    )
                    if _pending_nudge:
                        messages.append(Message(
                            role="user",
                            content=f"<nudge>{_pending_nudge}</nudge>",
                        ))
                        yield self._meta_chunk(
                            phase="executing", status="task_stale_nudge",
                            model=request.model,
                            extra={
                                "strategy": "native",
                                "kind": "open_tasks_on_stop",
                            },
                        )
                        continue
                    # Held-out verification gate (Arbor principle #2;
                    # augmentum/coder/verify_command.py). The goal judge
                    # below reasons over the agent's OWN report — a
                    # confident summary can persuade it. This gate runs the
                    # project's verification command in the container FIRST;
                    # a real non-zero exit is ground truth the agent cannot
                    # argue past, so we reject the stop and re-enter with the
                    # actual failure output. Default OFF; fail-open (skip) on
                    # any no-signal path, loudly, inside the gate.
                    from augmentum.config import settings as _vg_settings
                    if (
                        total_writes > 0
                        and bool(getattr(_vg_settings, "coder_verify_command_gate_enabled", False))
                    ):
                        from augmentum.coder.verify_command import (
                            MAX_VERIFY_REENTRY,
                            run_verification_gate,
                        )
                        if verify_gate_streak < MAX_VERIFY_REENTRY:
                            vo = await run_verification_gate(
                                self._container_manager,
                                self._workspace_id,
                                command=str(
                                    getattr(_vg_settings, "coder_verify_command", "") or ""
                                ),
                            )
                            yield self._meta_chunk(
                                phase="executing",
                                status="verify_command",
                                model=request.model,
                                extra={
                                    "iteration": iteration,
                                    "strategy": "native",
                                    "result": vo.status,
                                    "exit_code": vo.exit_code,
                                    "command": vo.command[:120],
                                    "attempt": verify_gate_streak + 1,
                                },
                            )
                            if vo.status == "fail":
                                verify_gate_streak += 1
                                messages.append(Message(
                                    role="user",
                                    content=(
                                        "<verification>The project's "
                                        f"verification command `{vo.command}` "
                                        f"failed (exit {vo.exit_code}). This is "
                                        "ground truth, not an opinion — the "
                                        "request is not satisfied until it "
                                        "passes. Fix the underlying cause, then "
                                        f"finish.\n\n{vo.output}</verification>"
                                    ),
                                ))
                                continue

                    # Goal judge (MiMo-Code's stop-condition gate,
                    # adapted 2026-06-12): the TQG is heuristic and
                    # the model's own summary is optimistic by nature
                    # — before honoring a stop on a turn that DID
                    # WRITE, ask an independent judge whether the
                    # user's request is actually satisfied. Not
                    # satisfied → inject the judge's reason and go
                    # around (cap MAX_JUDGE_REENTRY). Fail-open on
                    # judge trouble, loudly, inside judge_goal_satisfied.
                    from augmentum.config import settings as _gj_settings
                    if (
                        total_writes > 0
                        and bool(getattr(_gj_settings, "coder_goal_judge_enabled", True))
                    ):
                        from augmentum.coder.goal_judge import (
                            MAX_JUDGE_REENTRY,
                            judge_goal_satisfied,
                        )
                        if goal_judge_streak < MAX_JUDGE_REENTRY:
                            jv = await judge_goal_satisfied(
                                self._backend,
                                source_request=request,
                                user_goal=latest_input,
                                final_response=full_content,
                                edited_paths=list(
                                    getattr(self, "_controller_edited_paths", []) or []
                                ),
                                total_writes=total_writes,
                            )
                            yield self._meta_chunk(
                                phase="executing",
                                status="goal_judge",
                                model=request.model,
                                extra={
                                    "iteration": iteration,
                                    "strategy": "native",
                                    "ok": jv.ok,
                                    "impossible": jv.impossible,
                                    "reason": jv.reason[:200],
                                    "attempt": goal_judge_streak + 1,
                                },
                            )
                            if jv.ok is False and not jv.impossible:
                                goal_judge_streak += 1
                                messages.append(Message(
                                    role="user",
                                    content=(
                                        "<judge>An independent completion "
                                        "check says the request is not yet "
                                        f"satisfied: {jv.reason} — address "
                                        "this concretely, then finish."
                                        "</judge>"
                                    ),
                                ))
                                continue
                    termination_reason = f"model_stop:{verdict.reason}"
                    break

                # Gate said nudge. Append the matching nudge body and
                # let the loop go around once more. The streak bounds
                # this to ``native_nudge_cap`` consecutive prose-no-tools
                # nudges before the gate sees ``continuation_nudged=True``
                # and accepts. Progress (any successful tool call) resets
                # the streak below.
                messages.append(Message(
                    role="user",
                    content=_nudge_message_for(verdict.nudge_kind),
                ))
                native_nudge_streak += 1
                continuation_nudged = native_nudge_streak >= native_nudge_cap
                yield self._meta_chunk(
                    phase="executing", status="continuation_nudge",
                    model=request.model,
                    extra={
                        "iteration":  iteration,
                        "strategy":   "native",
                        "nudge_streak": native_nudge_streak,
                        "nudge_cap":   native_nudge_cap,
                        "nudge_kind": verdict.nudge_kind,
                        "reason":     verdict.reason,
                        "explain":    verdict.explain(),
                    },
                )
                continue

            # Emit tool_call meta chunks first so the UI sees them in
            # request order. Then run independent read tools in parallel
            # and serialize everything that can mutate or depend on
            # workspace/runtime state. This keeps native mode fast for
            # file/search fanout without racing edits, shell commands,
            # services, browser sessions, port publishing, or test runs
            # against one another.
            normalized = [self._normalize_tool_call(tc) for tc in tool_calls]
            for tool_id, tool_name, tool_input in normalized:
                yield self._meta_chunk(
                    phase="executing", status="tool_call",
                    model=request.model,
                    extra={"tool_call": {
                        "id": tool_id, "tool": tool_name, "input": tool_input,
                    }},
                )

            parallel_reads = [
                item for item in normalized
                if item[1] in READ_ONLY_TOOLS
                and item[1] not in _NATIVE_SERIAL_TOOL_NAMES
            ]
            serial_calls = [
                item for item in normalized
                if item[1] not in READ_ONLY_TOOLS
                or item[1] in _NATIVE_SERIAL_TOOL_NAMES
            ]
            results_by_id = {}

            if parallel_reads:
                read_results = await _asyncio.gather(*(
                    _execute_tool(
                        tool_map=tool_map, tool_name=name, tool_input=inp,
                        workspace_id=self._workspace_id,
                    )
                    for (_id, name, inp) in parallel_reads
                ))
                for item, result in zip(parallel_reads, read_results, strict=True):
                    results_by_id[item[0]] = result

            for tool_id, tool_name, tool_input in serial_calls:
                # Shell tools stream stdout live; task_dispatch streams
                # subagent progress events; everything else falls through
                # to the standard one-shot executor. The streaming
                # wrappers each yield zero-or-more chunks and exactly
                # one ToolResult (the final value). Routing is name-
                # based — checked against the wrapper's frozenset
                # rather than threading a flag through the tool model.
                from augmentum.modes.coder.handler import (  # noqa: F401
                    _SUBAGENT_STREAMING_TOOLS as _SUB_STREAM,
                )
                from augmentum.modes.coder.handler import (
                    _execute_tool_with_subagent_stream,
                )
                if tool_name in _SUB_STREAM:
                    wrapper = _execute_tool_with_subagent_stream
                else:
                    wrapper = _execute_tool_with_shell_stream
                async for ev in wrapper(
                    tool_map=tool_map,
                    tool_name=tool_name,
                    tool_input=tool_input,
                    model=request.model,
                    workspace_id=self._workspace_id,
                ):
                    if isinstance(ev, ToolResult):
                        results_by_id[tool_id] = ev
                    else:
                        yield ev

            results = [results_by_id[tool_id] for tool_id, _, _ in normalized]
            self._state.tool_calls_made += len(results)

            # Mid-turn persist so the inspector panel reflects
            # task_list / mission mutations as they happen instead of
            # showing 0-of-0 until end-of-turn. ``_persist_state_if_dirty``
            # is hash-guarded — cheap when nothing the inspector cares
            # about has changed. Originally only wired in _act_hybrid;
            # added here 2026-05-31 so native (the default strategy)
            # gets the same live-inspector behaviour. See the analogous
            # block in _act_hybrid for the long-form rationale.
            tasks_mutated, task_stale_nudge = task_spine.observe(
                self._state.tasks,
                nudge_enabled=getattr(self._state, "safeguards_enabled", True),
            )
            if tasks_mutated:
                try:
                    await self._persist_state_if_dirty()
                except Exception:
                    log.debug("coder_mid_turn_persist_failed", exc_info=True)
            if task_stale_nudge:
                # Staleness nudge (task_spine.py). One-shot, re-armed
                # on any real task_list mutation. Arbitrated with the
                # rest of this iteration's nudges below.
                health.submit(
                    "task_stale_nudge", task_stale_nudge,
                    {"streak": task_spine.stale_streak},
                )

            # Tool-outcome accounting for the Termination Quality Gate
            # and the silent-success detector.
            #   - ``progress_iters``: any successful tool call (reads
            #     OR writes). Drives had_recent_progress so the gate
            #     accepts the "file_list → summary" wrap-up pattern.
            #   - ``total_writes``: mutating-tool successes only. The
            #     gate's INSISTENT rule trips on zero writes regardless
            #     of how many reads happened, which is the right shape
            #     for "user said implement, model only inspected".
            #   - ``iter_shell_calls`` / ``iter_shell_silent``: feed
            #     the silent-success-fog detector. An iteration where
            #     every shell_exec returned the "(exit 0, no stdout)"
            #     sentinel adds to ``silent_success_streak``; any non-
            #     silent shell_exec output resets it. Iterations with
            #     no shell_exec at all don't move the streak in either
            #     direction (the model may legitimately be reading
            #     files this turn).
            #
            # Clearing continuation_nudged on any progress mirrors
            # hybrid (line ~2487): a fresh round of useful tool work
            # re-arms the nudge so a *later* prose stop can be caught
            # again if it bails before any further progress.
            any_progress_this_iter = False
            writes_this_iter = 0
            iter_shell_calls = 0
            iter_shell_silent = 0
            iter_edited_paths: list[str] = []
            iter_probes: list[tuple[str, str, str]] = []
            iter_dup_actions: list[tuple[str, object]] = []
            for (_tool_id, _tool_name, _tool_input), _r in zip(
                normalized, results, strict=True,
            ):
                if _tool_name == "shell_exec":
                    iter_shell_calls += 1
                    # Substring match (not full equality) so a future
                    # tweak to the sentinel string in tools.py doesn't
                    # silently disable this detector.
                    if (
                        _r.success
                        and "(exit 0, command succeeded" in (_r.output or "")
                    ):
                        iter_shell_silent += 1
                    elif _r.success and (_r.output or "").strip():
                        # Non-empty successful shell output → candidate
                        # verification probe for the always-green check
                        # AND the command-carousel detector. Carries the
                        # tool_id so the carousel reorient can prune the
                        # redundant results in place.
                        iter_probes.append((
                            _tool_id,
                            str((_tool_input or {}).get("command") or ""),
                            _r.output or "",
                        ))
                if not _r.success:
                    continue
                any_progress_this_iter = True
                if _tool_name in _MUTATING_TOOL_NAMES:
                    writes_this_iter += 1
                    iter_edited_paths.extend(
                        p for p in self._mutation_paths(_tool_name, _tool_input)
                        if p
                    )
                # Windowed duplicate-read ladder (duplicate_calls.py) —
                # successful read-shaped calls only; the tracker itself
                # filters to READ_ONLY_TOOLS.
                _dup_action, _dup_rec = duplicate_calls.observe(
                    tool_id=_tool_id,
                    tool=_tool_name,
                    tool_input=_tool_input if isinstance(_tool_input, dict) else {},
                    output=_r.output or "",
                )
                if _dup_action:
                    iter_dup_actions.append((_dup_action, _dup_rec))
                code_intel_adoption.observe(
                    _tool_name,
                    _tool_input if isinstance(_tool_input, dict) else {},
                )
            if any_progress_this_iter:
                progress_iters.append(iteration)
                continuation_nudged = False
                native_nudge_streak = 0
            total_writes += writes_this_iter

            # Same-file write-churn ladder: nudge → escalate → break.
            # Runs on SUCCESSFUL mutations only — that's the blind
            # spot: a rewrite loop where every call succeeds trips no
            # other guard. The escalate rung is loop CONFIRMATION
            # (nudge fired, same path kept churning anyway) and hands
            # the turn to the heavyweight buddy with a handoff
            # briefing instead of waiting for the hard cap; the break
            # remains the backstop when no buddy is configured.
            churn_action, churn_path, churn_count = write_churn.observe(
                iter_edited_paths,
            )
            if (
                churn_action == "escalate"
                and getattr(self._state, "safeguards_enabled", True)
                and not escalated_buddy
            ):
                buddy = await self._get_workspace_buddy_model()
                if buddy and buddy != request.model:
                    previous_model = request.model
                    request.model = buddy
                    escalated_buddy = buddy
                    # Fresh ladder for the buddy — inheriting the
                    # looping model's count would hard-break it almost
                    # immediately even if it edits legitimately.
                    write_churn.reset_counts()
                    probe_signal.reset()
                    health.note_intervention("escalated_to_buddy")
                    messages.append(Message(
                        role="user",
                        content=escalation_handoff_body(
                            previous_model=previous_model,
                            reason="write_churn_loop",
                            detail=(
                                f"rewrote {churn_path} {churn_count} "
                                "times, ignored a corrective nudge"
                            ),
                        ),
                    ))
                    yield self._meta_chunk(
                        phase="executing", status="escalated_to_buddy",
                        model=buddy,
                        extra={
                            "reason": "write_churn_loop",
                            "path": churn_path,
                            "edit_count": churn_count,
                            "previous_model": previous_model,
                            "buddy": buddy,
                            "iteration": iteration,
                        },
                    )
                # No buddy configured → fall through silently; the
                # hard break below stays the backstop.
            if getattr(self._state, "safeguards_enabled", True):
                if churn_action == "break":
                    termination_reason = "same_file_edit_break"
                    health.note_intervention("same_file_edit_break")
                    yield emit(
                        (
                            f"\n\n[Stopped: edited {churn_path} "
                            f"{churn_count} times this turn without the "
                            "outcome changing. The agent is thrashing "
                            "on a single file. Re-read the file and "
                            "the failing output, or clarify the goal, "
                            "then try again.]\n"
                        ),
                        phase="executing", status="same_file_edit_break",
                        model=request.model,
                        extra={
                            "path": churn_path,
                            "edit_count": churn_count,
                            "strategy": "native",
                        },
                    )
                    break
                if churn_action == "nudge":
                    health.submit(
                        "same_file_edit_nudge",
                        churn_nudge_body(churn_path, churn_count),
                        {"path": churn_path, "edit_count": churn_count},
                    )

            # ─── Duplicate-read ladder: nudge → reorient → escalate ──
            # Reorientation is context REPAIR, not a break (2026-07-06
            # direction: re-orient the model without the damage but
            # maintaining the lesson): duplicate tool results are
            # stubbed in place (first kept as ground truth, pairing
            # intact) and a <reorientation> note preserves what the
            # repeated call established. Escalate reuses the write-
            # churn buddy handoff; with no buddy it degrades to the
            # reorientation note alone — this ladder never kills the
            # turn.
            if self._state.safeguards_enabled and iter_dup_actions:
                from augmentum.coder.duplicate_calls import (
                    duplicate_nudge_body,
                    prune_duplicate_results,
                    reorientation_body,
                )
                for _dup_action, _dup_rec in iter_dup_actions:
                    if _dup_action == "nudge":
                        health.submit(
                            "duplicate_call_nudge",
                            duplicate_nudge_body(_dup_rec),
                            {"tool": _dup_rec.tool, "count": _dup_rec.count},
                        )
                    elif _dup_action in {"reorient", "escalate"}:
                        health.note_intervention("loop_reorient")
                        _pruned = prune_duplicate_results(messages, _dup_rec)
                        messages.append(Message(
                            role="user",
                            content=reorientation_body(_dup_rec),
                        ))
                        yield self._meta_chunk(
                            phase="executing", status="loop_reorient",
                            model=request.model,
                            extra={
                                "tool": _dup_rec.tool,
                                "count": _dup_rec.count,
                                "pruned_results": _pruned,
                                "strategy": "native",
                            },
                        )
                        if _dup_action == "escalate" and not escalated_buddy:
                            buddy = await self._get_workspace_buddy_model()
                            if buddy and buddy != request.model:
                                previous_model = request.model
                                request.model = buddy
                                escalated_buddy = buddy
                                health.note_intervention("escalated_to_buddy")
                                messages.append(Message(
                                    role="user",
                                    content=escalation_handoff_body(
                                        previous_model=previous_model,
                                        reason="duplicate_call_loop",
                                        detail=(
                                            f"re-ran `{_dup_rec.tool}` with "
                                            f"identical input {_dup_rec.count} "
                                            "times, through a nudge and a "
                                            "context repair"
                                        ),
                                    ),
                                ))
                                yield self._meta_chunk(
                                    phase="executing", status="escalated_to_buddy",
                                    model=buddy,
                                    extra={
                                        "reason": "duplicate_call_loop",
                                        "tool": _dup_rec.tool,
                                        "count": _dup_rec.count,
                                        "previous_model": previous_model,
                                        "buddy": buddy,
                                        "iteration": iteration,
                                    },
                                )

            # ─── Code-intel adoption nudges (one-shot each) ─────────
            # Definition-shaped grep → find_symbol; single-read streak
            # → batch paths=[...] / file_outline. Purely advisory —
            # never reorients, escalates, or breaks.
            for _ci_kind, _ci_body in code_intel_adoption.end_iteration():
                if getattr(self._state, "safeguards_enabled", True):
                    health.submit(_ci_kind, _ci_body)

            # Silent-success-fog detection. Only iterations that
            # actually issued shell calls can advance the streak; an
            # iteration with no shell_exec at all is neutral. Reset
            # only on a non-silent shell_exec (the model now has real
            # output to reason about).
            if iter_shell_calls > 0:
                if iter_shell_silent == iter_shell_calls:
                    silent_success_streak += 1
                else:
                    silent_success_streak = 0

            if (
                self._state.safeguards_enabled
                and silent_success_streak >= _SILENT_SUCCESS_NUDGE_AT
                and not silent_success_nudge_fired
            ):
                health.submit(
                    "silent_success_nudge",
                    (
                        f"Your last {silent_success_streak} shell "
                        "commands all returned `(exit 0, no stdout)` — "
                        "successful, but opaque. Before the next "
                        "mutation, run a diagnostic command (e.g. "
                        "`ps aux | grep <name>`, `curl http://localhost:"
                        "<port>/`, `ls -la <path>`) and READ the output. "
                        "Don't guess at state — confirm it."
                    ),
                    {"streak": silent_success_streak},
                )
                silent_success_nudge_fired = True

            # ─── Always-green-probe detector ─────────────────────────
            # Probes observed BEFORE this iteration's mutations bump
            # the epoch: within one wave the edit/probe order is
            # unknowable, so a same-iteration edit never counts as
            # "landed between two probe runs" — one wave of detection
            # latency in exchange for zero false positives.
            if self._state.safeguards_enabled:
                for _probe_id, _probe_cmd, _probe_out in iter_probes:
                    if probe_signal.observe_probe(_probe_cmd, _probe_out) == "nudge":
                        health.submit(
                            "probe_no_signal_nudge",
                            probe_no_signal_nudge_body(
                                _probe_cmd, probe_signal.nudge_at,
                            ),
                            {
                                "command": _probe_cmd.strip()[:160],
                                "repeats": probe_signal.nudge_at,
                            },
                        )
                        break
            probe_signal.note_mutations(writes_this_iter)

            # ─── Command-carousel ladder: nudge → reorient → escalate ─
            # Same NORMALIZED shell command re-run with no improvement in
            # its meaningful signal (pytest counts / error sig). The
            # test/probe re-run class no other guard sees. Reorient prunes
            # the redundant shell RESULTS in place (reusing the read-
            # carousel surgery) so the context stops few-shot-pressuring
            # the next re-run; escalate hands to the buddy. Observed
            # before this iteration's mutations bump the epoch (same
            # one-wave-latency contract as the probe detector).
            if self._state.safeguards_enabled:
                for _cc_id, _cc_cmd, _cc_out in iter_probes:
                    _cc_action, _cc_rec = command_carousel.observe(
                        tool_id=_cc_id, command=_cc_cmd, output=_cc_out,
                    )
                    # Flaky-test flag (#5) — orthogonal one-shot; a test
                    # that flapped with no edit between runs can't be
                    # stabilized by re-running or rewriting.
                    if _cc_rec is not None and _cc_rec.just_flagged_flaky:
                        health.submit(
                            "flaky_test_nudge",
                            flaky_test_body(_cc_rec),
                            {"command": _cc_rec.command[:160]},
                        )
                    if _cc_action == "nudge":
                        health.submit(
                            "command_carousel_nudge",
                            carousel_nudge_body(_cc_rec),
                            {
                                "command": _cc_rec.command[:160],
                                "count": _cc_rec.count,
                            },
                        )
                    elif _cc_action in {"reorient", "escalate"}:
                        from augmentum.coder.duplicate_calls import (
                            prune_duplicate_results,
                        )
                        health.note_intervention("loop_reorient")
                        _cc_pruned = prune_duplicate_results(messages, _cc_rec)
                        messages.append(Message(
                            role="user",
                            content=carousel_reorientation_body(_cc_rec),
                        ))
                        yield self._meta_chunk(
                            phase="executing", status="loop_reorient",
                            model=request.model,
                            extra={
                                "command": _cc_rec.command[:160],
                                "count": _cc_rec.count,
                                "pruned_results": _cc_pruned,
                                "kind": "command_carousel",
                                "strategy": "native",
                            },
                        )
                        if _cc_action == "escalate" and not escalated_buddy:
                            buddy = await self._get_workspace_buddy_model()
                            if buddy and buddy != request.model:
                                previous_model = request.model
                                request.model = buddy
                                escalated_buddy = buddy
                                # #3: hand the buddy a CLEANED window (the
                                # prune above already ran) and a fresh
                                # ladder + re-baselined stall clock — but
                                # keep accumulated file/test state so a
                                # buddy that merely repeats trips promptly.
                                command_carousel.reset()
                                probe_signal.reset()
                                progress_ledger.reset_after_handoff(iteration)
                                health.note_intervention("escalated_to_buddy")
                                messages.append(Message(
                                    role="user",
                                    content=escalation_handoff_body(
                                        previous_model=previous_model,
                                        reason="command_carousel_loop",
                                        detail=(
                                            f"re-ran `{_cc_rec.command[:120]}` "
                                            f"{_cc_rec.count} times with no change "
                                            "in the result, through a nudge and a "
                                            "context repair"
                                        ),
                                    ),
                                ))
                                yield self._meta_chunk(
                                    phase="executing", status="escalated_to_buddy",
                                    model=buddy,
                                    extra={
                                        "reason": "command_carousel_loop",
                                        "command": _cc_rec.command[:160],
                                        "count": _cc_rec.count,
                                        "previous_model": previous_model,
                                        "buddy": buddy,
                                        "iteration": iteration,
                                    },
                                )
            command_carousel.note_mutations(writes_this_iter)

            # ─── Coarse turn-progress ceiling: nudge → break ─────────
            # The superset backstop beneath every narrow breaker: no new
            # file changed AND no additional test passing for N iters =
            # measurably standing still. Resets on any genuine step so a
            # long legitimate build never trips it. The break is the
            # floor that makes runaway turn length structurally
            # impossible even for a carousel shape not anticipated above.
            if self._state.safeguards_enabled:
                _iter_signals = [_extract_signal(o) for (_i, _c, o) in iter_probes]
                _prog_action = progress_ledger.note(
                    iteration, iter_edited_paths, _iter_signals,
                )
                if _prog_action == "break":
                    _stalled = iteration - progress_ledger.last_progress_iter
                    health.note_intervention("progress_stall_break")
                    termination_reason = (
                        "escalation_exhausted"
                        if escalated_buddy
                        else "progress_stall_break"
                    )
                    _stop_copy = (
                        f"\n\n[Stopped: no measurable progress for {_stalled} "
                        "iterations"
                        + (
                            " — even after escalating to the buddy model"
                            if escalated_buddy
                            else ""
                        )
                        + ". The changed-file set stopped growing and no new "
                        "test started passing. Re-read the specific failure "
                        "or clarify the goal, then try again.]\n"
                    )
                    yield emit(
                        _stop_copy,
                        phase="executing", status=termination_reason,
                        model=request.model,
                        extra={
                            "stalled_iters": _stalled,
                            "escalated": bool(escalated_buddy),
                            "strategy": "native",
                        },
                    )
                    break
                if _prog_action == "nudge":
                    _stalled = iteration - progress_ledger.last_progress_iter
                    health.submit(
                        "progress_stall_nudge",
                        progress_stall_nudge_body(
                            _stalled, progress_ledger.best_passed,
                        ),
                        {"stalled_iters": _stalled},
                    )

            # ─── Identical-call loop detector ────────────────────────
            # Re-issuing the SAME tool with the SAME args for byte-
            # identical output is a stuck loop even when every call
            # succeeds — which is exactly the case no_progress (resets
            # on success), silent_success (empty shell stdout only), and
            # the validation breaks (errors only) all miss. Hash each
            # successful (tool, input, output) this iteration, track
            # consecutive repeats, and nudge once when any signature
            # reaches the threshold. Gated on safeguards + one-shot.
            if self._state.safeguards_enabled:
                iter_result_sigs: dict[str, str] = {}
                for (_, _tn, _ti), _r in zip(normalized, results, strict=True):
                    if not _r.success:
                        continue
                    _sig = _identical_result_signature(_tn, _ti, _r.output)
                    iter_result_sigs[_sig] = _tn
                identical_peak = _bump_identical_streaks(
                    identical_result_streaks, set(iter_result_sigs),
                )
                # Loop CONFIRMATION: the one-shot nudge fired and the
                # identical-call streak kept climbing to 2× threshold
                # anyway — hand off to the buddy with a briefing.
                # (Streak tracking used to stop once the nudge fired,
                # which made post-nudge confirmation impossible.)
                if (
                    identical_result_nudge_fired
                    and identical_peak >= _IDENTICAL_RESULT_NUDGE_AT * 2
                    and not escalated_buddy
                ):
                    buddy = await self._get_workspace_buddy_model()
                    if buddy and buddy != request.model:
                        previous_model = request.model
                        request.model = buddy
                        escalated_buddy = buddy
                        _peak_sig = max(
                            identical_result_streaks,
                            key=identical_result_streaks.get,
                        )
                        _peak_tool = iter_result_sigs.get(_peak_sig, "a tool")
                        identical_result_streaks.clear()
                        identical_result_nudge_fired = False
                        probe_signal.reset()
                        health.note_intervention("escalated_to_buddy")
                        messages.append(Message(
                            role="user",
                            content=escalation_handoff_body(
                                previous_model=previous_model,
                                reason="identical_call_loop",
                                detail=(
                                    f"re-issued `{_peak_tool}` with "
                                    "identical arguments and results "
                                    f"{identical_peak}+ times, ignored "
                                    "a corrective nudge"
                                ),
                            ),
                        ))
                        yield self._meta_chunk(
                            phase="executing", status="escalated_to_buddy",
                            model=buddy,
                            extra={
                                "reason": "identical_call_loop",
                                "tool": _peak_tool,
                                "streak": identical_peak,
                                "previous_model": previous_model,
                                "buddy": buddy,
                                "iteration": iteration,
                            },
                        )
                elif (
                    identical_peak >= _IDENTICAL_RESULT_NUDGE_AT
                    and not identical_result_nudge_fired
                ):
                    _peak_sig = max(
                        identical_result_streaks,
                        key=identical_result_streaks.get,
                    )
                    _peak_tool = iter_result_sigs.get(_peak_sig, "a tool")
                    health.submit(
                        "identical_result_nudge",
                        (
                            f"You've called `{_peak_tool}` with the "
                            "same arguments and gotten the exact same result "
                            f"{identical_peak} times in a row. Re-running it "
                            "won't change the output. If that result was "
                            "truncated or paged, advance to the next part "
                            "(e.g. a higher `offset`) instead of re-reading "
                            "the same window. Otherwise act on what it "
                            "already told you, or take a different approach — "
                            "inspect something new, change the arguments, or "
                            "make an edit."
                        ),
                        {"tool": _peak_tool, "streak": identical_peak},
                    )
                    identical_result_nudge_fired = True

            # ─── Loop-health arbitration: at most ONE nudge lands ────
            # Every guard above submitted instead of appending. The
            # highest-priority nudge is injected; the rest are dropped
            # with telemetry (their bodies reference stale windows by
            # next iteration, so requeueing would mislead). Iterations
            # where a reorient/escalate/break fired inject no nudge at
            # all — the model already has a stronger corrective message.
            _lh_winner, _lh_suppressed = health.arbitrate()
            if _lh_winner is not None:
                messages.append(Message(
                    role="user",
                    content="<nudge>" + _lh_winner.body + "</nudge>",
                ))
                yield self._meta_chunk(
                    phase="executing", status=_lh_winner.kind,
                    model=request.model,
                    extra={**_lh_winner.extra, "strategy": "native"},
                )
            for _lh_s in _lh_suppressed:
                yield self._meta_chunk(
                    phase="executing", status="loop_health_suppressed",
                    model=request.model,
                    extra={"kind": _lh_s.kind, "strategy": "native"},
                )

            for (tool_id, tool_name, tool_input), tool_result in zip(
                normalized, results, strict=True,
            ):
                preview = (
                    tool_result.output if tool_result.success
                    else (tool_result.error or "")
                ) or ""
                yield self._meta_chunk(
                    phase="executing", status="tool_result",
                    model=request.model,
                    extra={"tool_result": {
                        "id": tool_id,
                        "tool": tool_name,
                        "success": tool_result.success,
                        "output_preview": preview[:_preview_len(  # noqa: F821
                            tool_name, success=tool_result.success,
                        )],
                    }},
                )
                # Feed the cross-iteration ledger that
                # _find_repeat_offender reads. Native used to skip this
                # entirely: the local `consecutive_validation_errors`
                # counter caught the streak for buddy escalation, but
                # the global state's `recent_validation_errors` ring
                # buffer stayed empty, so the same-signature breaker
                # below could never fire — even though canonical and
                # hybrid both rely on it. Surfaced 2026-05-29 by a live
                # transcript that looped 6× on "file_write without
                # 'path'" with no break.
                if getattr(tool_result, "validation_error", False):
                    self._state.record_validation_error(
                        tool_name=tool_name,
                        error=tool_result.error or "",
                    )
                elif not tool_result.success:
                    self._state.record_tool_failure(
                        tool_name=tool_name,
                        target=_soft_failure_target(  # noqa: F821
                            tool_name, tool_input,
                        ),
                        error=tool_result.error or "",
                    )
                self._append_tool_result_to_history(
                    messages, tool_id, tool_name, tool_result, tier,
                )

            # ─── Stagnation detector → buddy-model escalation ────────
            # Two trip signals — both bias toward "the model is stuck
            # in a way the same model won't get itself out of":
            #
            #   1. Two consecutive iterations where every tool call
            #      returned validation_error=True — the model is
            #      fumbling the tool schema (missing required field,
            #      wrong arg shape) and not learning from the hint.
            #      THIS is the failure class the file_write missing-
            #      path loop exemplified.
            #
            #   2. Three consecutive iterations with zero successful
            #      tool calls — model isn't making progress; whether
            #      that's bad tool selection, repeated errors, or
            #      thrashing on the wrong target, a stronger model
            #      usually breaks the streak.
            #
            # On trip, look up the per-workspace heavyweight model
            # (``bug_finder_verifier_model`` — the same slot the user
            # configured for Bug Finder verification, repurposed as
            # the escalation buddy). If set, swap ``request.model`` so
            # the remaining iterations of THIS TURN run on the buddy.
            # If not set, the loop continues unchanged — the existing
            # recoverable-error pill keeps the user in control.
            #
            # Once escalated, stay escalated for the rest of the turn.
            # Handing back mid-turn is a future refinement; today the
            # cost discipline is "buddy runs the rescue, next user turn
            # starts fresh on the main model."
            if not escalated_buddy:
                any_validation_error = any(
                    getattr(r, "validation_error", False) for r in results
                )
                any_success = any(r.success for r in results)
                if any_validation_error:
                    consecutive_validation_errors += 1
                else:
                    consecutive_validation_errors = 0
                if any_success:
                    consecutive_no_progress = 0
                else:
                    consecutive_no_progress += 1

                trip_reason = ""
                if consecutive_validation_errors >= 2:
                    trip_reason = "repeated_validation_error"
                elif consecutive_no_progress >= 3:
                    trip_reason = "no_progress"

                if trip_reason:
                    buddy = await self._get_workspace_buddy_model()
                    if buddy and buddy != request.model:
                        previous_model = request.model
                        request.model = buddy
                        escalated_buddy = buddy
                        # Buddy gets a clean slate: clearing the
                        # validation-error ledger prevents the breaker
                        # below from firing immediately on stale records
                        # from the previous model.
                        self._state.clear_validation_errors()
                        consecutive_validation_errors = 0
                        health.note_intervention("escalated_to_buddy")
                        # Handoff briefing (2026-07-06): the swap used
                        # to be silent, so the buddy continued the
                        # weaker model's approach by momentum. Same
                        # briefing the loop-class escalations use.
                        messages.append(Message(
                            role="user",
                            content=escalation_handoff_body(
                                previous_model=previous_model,
                                reason=trip_reason,
                                detail="",
                            ),
                        ))
                        yield self._meta_chunk(
                            phase="executing", status="escalated_to_buddy",
                            model=buddy,
                            extra={
                                "reason": trip_reason,
                                "previous_model": previous_model,
                                "buddy": buddy,
                                "iteration": iteration,
                            },
                        )

            # Same-signature break — sibling of the check in canonical and
            # hybrid. Native had only buddy escalation here, which is a
            # no-op when no buddy model is configured: the model could
            # then loop on an identical validation error until
            # max_iterations. 2026-05-29: bug surfaced in user transcript
            # — 6× "file_write without 'path'" in a row with no break.
            # Fires after the buddy attempt so an active swap gets a fair
            # chance, and skipped on the iteration where escalation just
            # happened (clear_validation_errors above resets the ledger).
            if self._state.safeguards_enabled:
                offender = _find_repeat_offender(self._state)
                if offender is not None:
                    termination_reason = "same_validation_error_repeat"
                    yield emit(
                        _format_repeat_break_message(offender),
                        phase="executing", status="validation_error_break",
                        model=request.model,
                        extra={
                            "tool": offender.get("tool"),
                            "repeat_count": int(offender.get("repeat_count") or 0),
                        },
                    )
                    break

            # finish_task is part of the native tool set too. When the
            # model chooses it, its user-facing answer lives in the
            # tool's summary argument, not in assistant prose. Honor it
            # immediately so native mode doesn't do one extra empty model
            # call and appear silent to the user.
            if self._state.finish_requested:
                termination_reason = "finish_task_called"
                yield self._meta_chunk(
                    phase="executing",
                    status="finish_task_called",
                    model=request.model,
                    extra={"summary_chars": len(self._state.finish_summary)},
                )
                if self._state.finish_summary:
                    yield emit(
                        self._state.finish_summary,
                        phase="executing",
                        status="streaming",
                        model=request.model,
                    )
                break
        else:
            termination_reason = "max_iterations_reached"
            yield self._meta_chunk(
                phase="executing", status="max_iterations_reached",
                model=request.model, extra={"reason": termination_reason},
            )

        # Persist this turn's trace for the next user turn's system
        # prompt. Mirrors _act_hybrid's write so the inspector's
        # "Prior turns" panel and the next turn's <prior_turns> block
        # are populated regardless of strategy. Gated on having any
        # tool exchanges — pure-prose turns leave no trace.
        #
        # Scope to THIS turn's chain, not the full ``messages`` — the
        # hydrated recency buffer seeded prior turns' tool messages into
        # ``messages``, and summarizing the whole list would re-attribute
        # their files/commands to this turn. ``_current_turn_chain``
        # isolates the fresh input + this turn's assistant/tool tail;
        # falls back to ``messages`` when the seam can't be resolved
        # (defensive — shouldn't happen in native).
        this_turn = self._current_turn_chain(messages) or messages
        if any(m.role == "tool" for m in this_turn):
            summary = self._build_turn_summary(
                messages=this_turn, user_goal=user_goal,
                termination_reason=termination_reason,
            )
            self._state.add_turn_summary(summary)
            await self._archive_turn_summary(summary)

        # Hydrated recency buffer: stash THIS turn's full in-format chain
        # so the NEXT turn seeds hydrated (tool_calls + results) instead of
        # the client's collapsed history — keeps the model interleaving on
        # follow-ups. Tool-using turns only (a prose turn in the window is
        # an anti-exemplar). Best-effort, never raises.
        try:
            self._capture_recency_turn(messages)
        except Exception:
            log.debug("coder.recency_capture_failed", exc_info=True)

        # Reviewable-turn bundle (no-op when registry/snapshot absent).
        await self._publish_turn_review(user_goal)

        self._state.phase = CoderPhase.WAITING
        yield self._meta_chunk(
            phase="waiting", status="complete",
            model=request.model,
            extra={
                "strategy": "native",
                "tool_calls_made": self._state.tool_calls_made,
                "termination_reason": termination_reason,
                "iterations_used": iteration,
                "review_turn_id": self._state.active_turn_id,
                # Unified guard telemetry — every nudge/repair/escalate/
                # break the coordinator saw this turn, incl. suppressed
                # ("suppressed:<kind>"). Empty dict = healthy turn.
                "loop_health": health.summary(),
            },
        )

    # ==================================================================
    # Hybrid strategy — canonical + four resilience innovations
    # ==================================================================
    # FROZEN (comparison/rollback only — native is the shipped default; see
    # the _act_hybrid docstring). Adds four mechanisms on top of the
    # canonical backbone to make weaker models behave:
    #
    #   #1 Observation refresh — inject fresh workspace snapshot every 8
    #      iterations so the model doesn't re-run dir_tree/ls from
    #      orientation loss.
    #   #2 Stagnation nudge — if the model emits the same tool batch
    #      twice in a row, append a <nudge> asking for a different angle.
    #   #3 Parallel read fan-out — batch up to 5 read-only tools per
    #      iteration. Mutations stay sequential. Overflow reads get a
    #      synthesized "try again in a smaller batch" tool_result.
    #   #4 Heuristic continuation judge — when the model emits no tool
    #      calls but also hasn't written recently, append a <nudge> once
    #      before terminating. Distinguishes "done" from "stopped early".
    #
    # On top of the innovations, five streak-break detectors catch
    # specific failure modes observed 2026-04-20:
    #   - validation_error_streak (malformed tool args)
    #   - test_failure_streak (test_run loops without a pass)
    #   - same_file_edit_break (thrashing on one path)
    #   - action_stagnation (parameter-thrashing on same tool)
    #   - inspection_loop_break (reads-only loop on a creation task)
    # ==================================================================

    async def _act_hybrid(
        self,
        request: InternalChatRequest,
        workspace_context: str,
    ) -> AsyncIterator[InternalStreamChunk]:
        """Rebuttal agent loop — consensus backbone + four innovations.

        **FROZEN — comparison/rollback only.** ``native`` is the shipped
        production default (see ``_act_native``); ``hybrid`` is reachable only
        via ``AUGMENTUM_CODER_STRATEGY=hybrid`` / the strategy header. Its
        innovations (observation refresh, stagnation nudge, parallel read
        fan-out, continuation judge) and streak-break detectors were the
        proving ground for the guards native now carries — but native is the
        single source of truth going forward. Do NOT hand-sync new guards
        here; new guard work lands in ``_act_native`` only. Kept loadable for
        rollback/A-B comparison; expect its guards to lag native.
        """
        tools = create_coder_tools(  # noqa: F821
            self._container_manager, self._workspace_id, self._state,
            executor=getattr(self, "_executor", None),
            tool_registry=self._tool_registry,
            question_callback=self._question_callback,
            profile_store=getattr(self, "_profile_store", None),
            service_store=getattr(self, "_service_store", None),
            user_id=getattr(self, "_user_id", ""),
            planning_mode=getattr(self, "_planning_mode", "default"),
            subagent_dispatcher=self._get_subagent_dispatcher(),
            db_conn=self._resolve_archive_conn(),
            jobs_store=getattr(self, "_jobs_store", None),
        )
        tool_map = {t.name: t for t in tools}
        tier = select_tier(self._backend, request.model)
        tool_schemas = (
            [_tool_to_schema(t) for t in tools]  # noqa: F821
            if tier == ToolCallingTier.NATIVE else None
        )

        yield self._meta_chunk(
            phase="executing", status="strategy",
            model=request.model, extra={"strategy": "hybrid"},
        )

        # Priming tree (Sprint 1): intent-aware system prompt assembly.
        # See ``_build_act_system`` in handler.py — branches by intent
        # for shortlist + exemplar; EDIT_FORMAT only for edit-capable.
        # state= captures per-branch token telemetry for the ledger.
        act_system = _build_act_system(  # noqa: F821 — bound by _bind_handler_helpers
            tier=tier, intent=self._state.current_intent,
            state=self._state,
        )
        if workspace_context:
            act_system += f"\n\n{workspace_context}"
        if self._state.plan:
            act_system += f"\n\n## Current Plan\n\n{self._state.plan}"
        # Plan-mode notice. No-op outside plan mode.
        act_system += self._plan_mode_addendum()
        messages = self._build_messages(request, act_system)

        iteration = 0
        termination_reason = "model_stop"
        last_refresh_iter = 0
        recent_sigs: collections.deque[tuple] = collections.deque(maxlen=4)
        stagnation_streak = 0
        validation_error_streak = 0
        write_iters: collections.deque[int] = collections.deque(
            maxlen=_HYBRID_CONTINUATION_LOOKBACK,
        )
        continuation_nudged = False
        operate_completion_nudged = False
        total_writes = 0

        #Tracking for the extra streak-break detectors.
        test_failure_streak = 0
        # Same-file write-churn ladder (shared with native; see
        # augmentum/coder/write_churn.py). Nudge rung is new 2026-07-06;
        # the hard break keeps its historical threshold.
        write_churn = WriteChurnTracker(
            nudge_at=_live_threshold("same_file_edit_nudge"),
            break_at=_live_threshold("same_file_edit_break"),
        )
        # Alias for the fallback-summary renderer at turn end.
        same_file_edits = write_churn.counts
        same_tool_streak = 0
        prev_iter_tool_name: str | None = None

        # Tool-result rollup for Phase 6 final-turn synthesis. Every
        # tool_result meta chunk (real, fanout-dropped, batch-duplicate)
        # gets its payload collected here so the synthesis prompt has
        # the full set of what the turn did.
        synth_tool_results: list[dict] = []
        inspection_only_streak = 0
        inspection_nudge_fired = False
        no_write_progress_streak = 0
        silent_success_streak = 0
        silent_success_nudge_fired = False
        failing_shell_streak = 0
        failing_shell_nudge_fired = False
        coordination_only_streak = 0
        coordination_nudge_fired = False
        # Task-list plan-spine tracker — staleness streak + one-shot
        # nudge bookkeeping, shared with native (see
        # augmentum/coder/task_spine.py).
        task_spine = TaskSpineTracker.start(self._state.tasks)
        total_prose_chars = 0

        # Split the goal: ``latest_input`` is what the user just typed
        # (may be a continuation like "keep going"), ``user_goal`` is
        # the last substantive request. Sticky reminder gets both so
        # the model distinguishes "working on X, user said continue"
        # from "user asked me to continue X." See handler.py's
        # _extract_goal_split for rationale.
        latest_input, user_goal = _extract_goal_split(request.messages)  # noqa: F821 — bound by _bind_handler_helpers

        # Pre-compute tightened inspection thresholds for creation-verb
        # goals. A task like "build the project" / "run the container"
        # should not wait 8 iterations to catch a pure-inspection loop —
        # the user's intent is "act, don't explore". For non-creation
        # goals (debug / explain / audit) the looser defaults still
        # apply so legitimate exploration isn't over-constrained.
        _goal_has_creation_verb = bool(
            user_goal and _CREATION_VERB_RE.search(user_goal)  # noqa: F821
        )
        if _goal_has_creation_verb:
            inspection_cold_start = 1
            inspection_nudge_at = 3
            inspection_break_delta = 2
        else:
            inspection_cold_start = _INSPECTION_COLD_START_GRACE
            inspection_nudge_at = _live_threshold("inspection_loop_nudge")
            inspection_break_delta = _live_threshold("inspection_loop_break")

        hybrid_observation_every = _cfg("_HYBRID_OBSERVATION_EVERY")
        hybrid_read_fanout = _cfg("_HYBRID_READ_FANOUT")

        # Tier-aware iteration cap — Phase 1.2. The tier is set in
        # handler._handle_stream_body via classify_tier; the env-var
        # _HYBRID_MAX_ITERS remains the ultimate safety ceiling. We
        # take min() so a tighter env-var override also wins, and so
        # an absent classification (defensive getattr) falls back to
        # current behavior.
        #
        # Per-workspace safeguards toggle: when the user has set
        # safeguards_enabled=False on this workspace, lift the cap to
        # _HYBRID_MAX_ITERS_UNGATED (500) and skip the tier ceiling so
        # strong models that legitimately run long aren't cut off. Soft
        # breakers below also short-circuit on the same flag.
        _safeguards_on = bool(getattr(self._state, "safeguards_enabled", True))
        _tier_classification = getattr(self, "_turn_tier_for_turn", None)
        if not _safeguards_on:
            effective_max_iters = _live_max_iters(ungated=True)
        elif _tier_classification is not None:
            _tier_max = TIER_LIMITS[_tier_classification.tier].max_iterations
            effective_max_iters = min(_live_max_iters(), _tier_max)
        else:
            effective_max_iters = _live_max_iters()

        while iteration < effective_max_iters:
            iteration += 1

            # Cooperative pause-gate + steer-message inbox drain.
            # See ``_coop_iteration_check`` — no-op outside the
            # background-runs broker path.
            _steer = await self._coop_iteration_check()
            if _steer:
                for _s in _steer:
                    messages.append(Message(
                        role="user",
                        content=self._format_steer_content(_s),
                    ))
                yield self._meta_chunk(
                    phase="executing", status="steer_delivered",
                    model=request.model,
                    extra={"count": len(_steer), "iteration": iteration},
                )

            # Early termination: explicit finish_task signal. Checked
            # before tasks_completed because it's more specific — a model
            # that called finish_task has declared the whole request
            # done, not just the tracked subtasks. Honoured from
            # iteration 1 onward (unlike tasks_completed which protects
            # resumed sessions) because finish_task on iteration 1 is
            # a deliberate act, not stale state.
            if self._state.finish_requested:
                if await self._maybe_request_pre_finish_review(
                    request=request,
                    messages=messages,
                    total_writes=total_writes,
                    latest_input=latest_input,
                    user_goal=user_goal,
                ):
                    for power_event in self._drain_pending_power_activation_events():
                        yield self._meta_chunk(
                            phase="executing",
                            status="power_activated",
                            model=request.model,
                            extra={"power_activation": power_event},
                        )
                    self._state.finish_requested = False
                    self._state.finish_summary = ""
                    continue
                completion_nudge = ""
                pending_contract: dict = {}
                if not operate_completion_nudged:
                    pending_contract = _pending_contract_for_turn(
                        intent_kind=self._turn_intent_for_turn.kind,
                        goal_text=user_goal or latest_input,
                        candidate_text=self._state.finish_summary,
                        messages=messages,
                        tool_results=synth_tool_results,
                    )
                    completion_nudge = _completion_nudge_for_turn(
                        intent_kind=self._turn_intent_for_turn.kind,
                        goal_text=user_goal or latest_input,
                        candidate_text=self._state.finish_summary,
                        messages=messages,
                        tool_results=synth_tool_results,
                    )
                if completion_nudge:
                    self._state.set_pending_objective_contract(pending_contract)
                    messages.append(Message(
                        role="user",
                        content=completion_nudge,
                    ))
                    self._state.finish_requested = False
                    self._state.finish_summary = ""
                    operate_completion_nudged = True
                    yield self._meta_chunk(
                        phase="executing",
                        status="operate_evidence_nudge",
                        model=request.model,
                        extra={"iteration": iteration},
                    )
                    continue
                self._state.clear_pending_objective_contract()
                termination_reason = "finish_task_called"
                yield self._meta_chunk(
                    phase="executing", status="finish_task_called",
                    model=request.model,
                    extra={"summary_chars": len(self._state.finish_summary)},
                )
                break

            # Early termination: if every seeded task is marked completed
            # AND we've already done at least one iteration AND tools have
            # actually fired this turn, stop.
            #
            # Three guards, each catches a distinct false-positive:
            # - ``iteration > 1`` protects resumed sessions that start
            #   with a preseeded all-completed task list from short-
            #   circuiting the act phase entirely.
            # - ``recent_sigs`` (Phase 3.6) — added when the termination
            #   gate started nudging into iter 2 on stop-with-prose. A
            #   nudged retry that didn't itself call any tool shouldn't
            #   trip "tasks_completed" on stale state. ``recent_sigs`` is
            #   populated only when the model emits a tool batch (see the
            #   stagnation-streak update below), so it's a clean
            #   "did-tool-work-happen-this-turn" signal.
            if (
                iteration > 1
                and self._state.tasks
                and recent_sigs
                and all(
                    t.get("status") == "completed" for t in self._state.tasks
                )
            ):
                if await self._maybe_request_pre_finish_review(
                    request=request,
                    messages=messages,
                    total_writes=total_writes,
                    latest_input=latest_input,
                    user_goal=user_goal,
                ):
                    for power_event in self._drain_pending_power_activation_events():
                        yield self._meta_chunk(
                            phase="executing",
                            status="power_activated",
                            model=request.model,
                            extra={"power_activation": power_event},
                        )
                    continue
                completion_nudge = ""
                pending_contract: dict = {}
                if not operate_completion_nudged:
                    pending_contract = _pending_contract_for_turn(
                        intent_kind=self._turn_intent_for_turn.kind,
                        goal_text=user_goal or latest_input,
                        candidate_text="",
                        messages=messages,
                        tool_results=synth_tool_results,
                    )
                    completion_nudge = _completion_nudge_for_turn(
                        intent_kind=self._turn_intent_for_turn.kind,
                        goal_text=user_goal or latest_input,
                        candidate_text="",
                        messages=messages,
                        tool_results=synth_tool_results,
                    )
                if completion_nudge:
                    self._state.set_pending_objective_contract(pending_contract)
                    messages.append(Message(
                        role="user",
                        content=completion_nudge,
                    ))
                    operate_completion_nudged = True
                    yield self._meta_chunk(
                        phase="executing",
                        status="operate_evidence_nudge",
                        model=request.model,
                        extra={"iteration": iteration},
                    )
                    continue
                if not operate_completion_nudged:
                    self._state.clear_pending_objective_contract()
                    termination_reason = "tasks_completed"
                    yield self._meta_chunk(
                        phase="executing", status="tasks_completed",
                        model=request.model,
                        extra={"task_count": len(self._state.tasks)},
                    )
                    break

            # Plan.md refresh per iteration (see _act_canonical for the
            # kernel-v2 rationale; same gate applies here).
            from augmentum.config import settings as _settings_for_kernel
            plan_md = (
                "" if _settings_for_kernel.coder_kernel_v2
                else await self._read_plan_md()
            )

            # Sticky reminder: re-rendered each turn so compaction can't
            # eat it (Claude Code pattern). Must happen BEFORE compaction
            # so the reminder lives at the tail where compaction
            # preserves it verbatim. latest_input is included so
            # continuation turns show the model BOTH the substantive
            # goal AND the "keep going" / "monitor X" signal that
            # triggered this turn.
            self._inject_sticky_reminder(
                messages, goal=user_goal, iteration=iteration,
                max_iters=effective_max_iters, writes=total_writes,
                plan_md=plan_md, latest_input=latest_input,
            )

            yield self._token_budget_chunk(
                messages,
                scope="hybrid_iteration",
                model=request.model,
                iteration=iteration,
            )
            compacted, before, after = await self._compact_messages_with_synthesis(messages, request)
            if compacted:
                yield self._meta_chunk(
                    phase="executing", status="compaction",
                    model=request.model,
                    extra={
                        "iteration": iteration,
                        "tokens_before": before,
                        "tokens_after": after,
                    },
                )
                yield self._token_budget_chunk(
                    messages,
                    scope="hybrid_iteration",
                    model=request.model,
                    iteration=iteration,
                    compacted=True,
                )

            # --- Innovation #1: canonical-observation refresh ---
            if (iteration - last_refresh_iter) >= hybrid_observation_every:
                obs = await self._canonical_observation()
                if obs:
                    messages.append(Message(
                        role="user",
                        content=f"<observation iteration=\"{iteration}\">\n{obs}\n</observation>",
                    ))
                    yield self._meta_chunk(
                        phase="executing", status="observation_refresh",
                        model=request.model, extra={"iteration": iteration},
                    )
                last_refresh_iter = iteration

            # Live drain + shared thinking policy (default OFF, composer
            # toggle wins) — see the native loop for both rationales.
            _live_result: list = []
            async for ev in self._stream_and_parse_live(
                request, messages, tool_schemas, tool_map, tier, iteration,
                result_out=_live_result,
                chat_template_kwargs=_iteration_thinking_kwargs(request),
            ):
                yield ev
            full_content, tool_calls, error_kind, full_thinking, error_status, error_message = _live_result[0]
            if error_kind:
                termination_reason = "backend_error"
                ui_status = (
                    "recoverable_error"
                    if error_kind == "transient" else "error"
                )
                yield emit(
                    _format_iteration_error(iteration, error_kind, error_status, error_message),
                    phase="executing", status=ui_status,
                    model=request.model,
                    extra={
                        "error_kind": error_kind,
                        "iteration": iteration,
                        "retry_status_code": error_status,
                        "error_message": error_message,
                    },
                )
                break

            clean_text = _strip_tool_json(full_content)  # noqa: F821
            clean_text = _strip_cot_tokens(clean_text)  # noqa: F821
            clean_text = re.sub(r'```[\s\S]*?```', '', clean_text)
            clean_text = re.sub(r'\n{3,}', '\n\n', clean_text).strip()
            # Preserve a pre-monologue-strip copy for loop detection.
            # A response that's 3× "Let me run diagnostics carefully"
            # is degenerate regardless of whether the sentences are
            # monologue tells; the content-loop detector below needs
            # to see the raw repetition to catch it.
            clean_text_for_loop = clean_text
            clean_text = _strip_act_monologue(clean_text)
            if clean_text:
                yield emit(
                    clean_text + "\n",
                    phase="executing", status="streaming",
                    model=request.model,
                )
                total_prose_chars += len(clean_text)

            self._append_assistant_to_history(
                messages, full_content, tool_calls, tier,
                thinking=full_thinking,
            )

            # --- Innovation #4: heuristic continuation judge ---
            if not tool_calls:
                had_recent_progress = any(
                    iteration - wi <= _HYBRID_CONTINUATION_LOOKBACK
                    for wi in write_iters
                )

                # Gating checks: a "substantive answer" on its own isn't
                # enough to accept the stop — the model may have emitted
                # an action-shaped code block (expected a shell_exec) or
                # gotten stuck in a degenerate content loop. Both must
                # trigger a targeted nudge rather than a normal break.
                has_unclaimed_bash = _has_unclaimed_code_block(full_content)  # noqa: F821
                has_unclaimed_tool_markup = _has_unclaimed_tool_markup(full_content)  # noqa: F821
                has_content_loop = _has_content_loop(clean_text_for_loop)  # noqa: F821

                if has_unclaimed_bash:
                    messages.append(Message(
                        role="user",
                        content=(
                            "<nudge>You showed a bash/shell code block in "
                            "prose but didn't actually call shell_exec. "
                            "If you want to run that command, emit a "
                            "shell_exec tool call. If you just wanted to "
                            "describe it, say so explicitly — don't leave "
                            "the user to guess whether it ran.</nudge>"
                        ),
                    ))
                    yield self._meta_chunk(
                        phase="executing", status="unclaimed_code_block_nudge",
                        model=request.model,
                        extra={"unclaimed_code": True, "iteration": iteration},
                    )
                    continue

                if has_unclaimed_tool_markup:
                    messages.append(Message(
                        role="user",
                        content=(
                            "<nudge>You wrote a tool-like line in prose "
                            "(for example `<tool_call>shell_exec: ...`) but "
                            "didn't emit a real tool call. That does NOT run "
                            "anything. Emit the actual tool call now, or explain "
                            "the blocker plainly instead of pseudo-calling it.</nudge>"
                        ),
                    ))
                    yield self._meta_chunk(
                        phase="executing", status="stagnation_nudge",
                        model=request.model,
                        extra={
                            "unclaimed_tool_markup": True,
                            "iteration": iteration,
                            "kind": "unclaimed_tool_markup",
                        },
                    )
                    continue

                if has_content_loop:
                    messages.append(Message(
                        role="user",
                        content=(
                            "<nudge>Your last response repeated itself — "
                            "same phrase multiple times in a row. That's "
                            "usually a sign you got stuck. Take a fresh "
                            "angle: pick the next concrete step, call a "
                            "tool, or ask the user to clarify.</nudge>"
                        ),
                    ))
                    yield self._meta_chunk(
                        phase="executing", status="content_loop_nudge",
                        model=request.model,
                        extra={"content_loop": True, "iteration": iteration},
                    )
                    continue

                if (
                    self._turn_intent_for_turn.kind in _ACTIONABLE_TURN_KINDS
                    and _looks_like_future_action_prose(clean_text)
                ):
                    messages.append(Message(
                        role="user",
                        content=(
                            "<nudge>You described next steps, but you didn't "
                            "actually do them. For actionable tasks, don't stop "
                            "on a plan or progress note. Emit the next real tool "
                            "call now, or explain the blocker plainly if you "
                            "cannot proceed.</nudge>"
                        ),
                    ))
                    yield self._meta_chunk(
                        phase="executing",
                        status="progress_without_action_nudge",
                        model=request.model,
                        extra={"iteration": iteration},
                    )
                    continue

                if self._response_contradicts_populated_repo(full_content):
                    repo_descriptor = getattr(
                        self,
                        "_populated_repo_descriptor_for_turn",
                        lambda: "an existing repo",
                    )()
                    messages.append(Message(
                        role="user",
                        content=(
                            "<nudge>You just treated the workspace as empty or "
                            "asked for a repository URL, but this turn already "
                            f"has {repo_descriptor}. That conclusion is false. Do NOT "
                            "ask what to build from scratch. Re-anchor on the "
                            "current repository, inspect one or two high-signal "
                            "files or subdirectories (README, package manifest, "
                            "src/, tests/, docs/), then identify a small real "
                            "improvement that fits the codebase.</nudge>"
                        ),
                    ))
                    yield self._meta_chunk(
                        phase="executing",
                        status="populated_repo_nudge",
                        model=request.model,
                        extra={"iteration": iteration},
                    )
                    continue

                completion_nudge = ""
                pending_contract: dict = {}
                if not operate_completion_nudged:
                    pending_contract = _pending_contract_for_turn(
                        intent_kind=self._turn_intent_for_turn.kind,
                        goal_text=user_goal or latest_input,
                        candidate_text=clean_text,
                        messages=messages,
                        tool_results=synth_tool_results,
                    )
                    completion_nudge = _completion_nudge_for_turn(
                        intent_kind=self._turn_intent_for_turn.kind,
                        goal_text=user_goal or latest_input,
                        candidate_text=clean_text,
                        messages=messages,
                        tool_results=synth_tool_results,
                    )
                if completion_nudge:
                    self._state.set_pending_objective_contract(pending_contract)
                    messages.append(Message(
                        role="user",
                        content=completion_nudge,
                    ))
                    operate_completion_nudged = True
                    yield self._meta_chunk(
                        phase="executing",
                        status="operate_evidence_nudge",
                        model=request.model,
                        extra={"iteration": iteration},
                    )
                    continue

                # Phase 3.6 — Termination Quality Gate.
                #
                # Pre-3.6 the decision was a single ``len(prose) >= 40``
                # check that conflated three distinct signals: did the
                # model produce an answer, did it do work this turn, did
                # the user demand completion. The gate splits those into
                # independent primitives (see ``coder/termination.py``)
                # and returns a structured verdict.
                #
                # Headline failure that motivated the gate (observed
                # 2026-05-10): the model emitted "I read the file but
                # the middle was elided." (47 chars, 1 sentence) under
                # an INSISTENT user request. The 40-char floor accepted
                # it as substantive; the gate now classifies it as
                # BAILOUT and nudges.
                verdict = evaluate_termination(TerminationContext(
                    user_text=latest_input,
                    intent_kind=self._turn_intent_for_turn.kind,
                    clean_prose=clean_text,
                    total_writes=total_writes,
                    had_recent_progress=had_recent_progress,
                    continuation_nudged=continuation_nudged,
                ))

                if verdict.accept_stop:
                    if await self._maybe_request_pre_finish_review(
                        request=request,
                        messages=messages,
                        total_writes=total_writes,
                        latest_input=latest_input,
                        user_goal=user_goal,
                    ):
                        for power_event in self._drain_pending_power_activation_events():
                            yield self._meta_chunk(
                                phase="executing",
                                status="power_activated",
                                model=request.model,
                                extra={"power_activation": power_event},
                            )
                        continuation_nudged = False
                        continue
                    self._state.clear_pending_objective_contract()
                    # Termination reason now anchors on the gate's
                    # verdict so session traces explain *why* we
                    # accepted the stop, not just that we did.
                    termination_reason = f"model_stop:{verdict.reason}"
                    break

                # Gate said nudge. Pick the framing that matches the
                # bail mode (insistence / bailout / no-progress) so the
                # model gets feedback specific to what it did wrong,
                # not a generic "try again" message.
                messages.append(Message(
                    role="user",
                    content=_nudge_message_for(verdict.nudge_kind),
                ))
                continuation_nudged = True
                yield self._meta_chunk(
                    phase="executing", status="continuation_nudge",
                    model=request.model,
                    extra={
                        "iteration":   iteration,
                        "nudge_kind":  verdict.nudge_kind,
                        "reason":      verdict.reason,
                        # ``explain`` is for human log readers; safe to
                        # surface alongside the structured tags.
                        "explain":     verdict.explain(),
                    },
                )
                continue

            # --- Innovation #2: soft stagnation nudge ---
            sig = _batch_signature(tool_calls)  # noqa: F821
            if recent_sigs and sig == recent_sigs[-1]:
                stagnation_streak += 1
            else:
                stagnation_streak = 0
            recent_sigs.append(sig)

            if stagnation_streak >= _HYBRID_STAGNATION_REPEATS:
                messages.append(Message(
                    role="user",
                    content=(
                        "<nudge>You've repeated the same tool batch twice. "
                        "Try a different angle: check your assumptions, use a "
                        "different tool, or ask the user for clarification.</nudge>"
                    ),
                ))
                yield self._meta_chunk(
                    phase="executing", status="stagnation_nudge",
                    model=request.model,
                    extra={"iteration": iteration, "streak": stagnation_streak},
                )
                stagnation_streak = 0
                recent_sigs.clear()
                continue

            #Track same-tool-name streak for action stagnation.
            # Fires when N consecutive iterations use the SAME tool (with
            # different args) — misses the stagnation_streak detector
            # above which requires identical batches.
            iter_tool_name = (
                tool_calls[0].get("name", "")
                if len(tool_calls) == 1 else None
            )
            if iter_tool_name and iter_tool_name == prev_iter_tool_name:
                # Mutating tools and shell_exec count; pure-read tools
                # are exempt — legitimate debug cycles loop on reads.
                if iter_tool_name in _MUTATING_TOOLS or iter_tool_name == "shell_exec":  # noqa: F821
                    same_tool_streak += 1
                else:
                    same_tool_streak = 0
            else:
                same_tool_streak = 0
            prev_iter_tool_name = iter_tool_name

            # --- Innovation #3: parallel read fan-out ---
            reads: list[dict] = []
            serial_calls: list[dict] = []
            for tc in tool_calls:
                if tc.get("name", "") in _HYBRID_PARALLEL_READ_TOOLS:
                    reads.append(tc)
                else:
                    serial_calls.append(tc)

            # In-batch dedup: if the model emits two file_reads with
            # identical args in the same batch, run one and synthesize a
            # ``batch_duplicate`` tool_result pointing at the canonical
            # call's id. Prevents wasted container round-trips and
            # surfaces the dedup to the UI.
            dedup_canonical: dict[tuple, str] = {}
            deduped_reads: list[dict] = []
            dup_reads: list[tuple[dict, str]] = []  # (dup_call, canonical_id)
            for tc in reads:
                key = (
                    tc.get("name", ""),
                    json.dumps(
                        tc.get("input") or tc.get("function", {}).get("arguments", {}),
                        sort_keys=True, default=str,
                    ),
                )
                if key in dedup_canonical:
                    dup_reads.append((tc, dedup_canonical[key]))
                else:
                    canonical_id = tc.get("id") or str(uuid.uuid4())
                    dedup_canonical[key] = canonical_id
                    deduped_reads.append(tc)
            reads = deduped_reads

            # Cap parallel reads so a hallucinated fan-out doesn't DOS
            # the container.
            dropped_reads: list[dict] = []
            if len(reads) > hybrid_read_fanout:
                dropped_reads = reads[hybrid_read_fanout:]
                reads = reads[:hybrid_read_fanout]

            counters: dict[str, int] = {
                "writes": 0, "validation_errors": 0, "tool_calls": 0,
                "iteration": iteration,
                #Extra counters set by _run_tool_tracked:
                "test_passed": 0, "test_failed": 0,
                "edited_paths": [],
                "inspection_tools_only": 0,
                # Silent-success shell counter — incremented when a
                # shell_exec returns exit 0 with no stdout. Used to
                # detect the "fog" state where the model can't reason
                # about what actually happened because every command
                # returns (exit 0, nothing). See _SILENT_SUCCESS_NUDGE_AT.
                "shell_exec_calls": 0, "shell_exec_silent": 0,
                # Failed shell count — feeds _FAILING_SHELL_NUDGE_AT.
                "shell_exec_failed": 0,
            }
            if reads:
                async for ev in self._run_tools_parallel(
                    reads, tool_map, tier, messages, request.model, counters,
                ):
                    yield _tap_tool_result(ev, synth_tool_results)

            # Synthesize ``batch_duplicate`` tool_results for reads the
            # dedup collapsed. Point each at the canonical call's id so
            # the UI can link them, and keep the tool_use schema valid
            # (one tool_result per committed tool_call).
            if dup_reads:
                from augmentum.tools.base import ToolResult as _TR
                for dup_tc, canonical_id in dup_reads:
                    tool_id, tool_name, _dup_input = self._normalize_tool_call(dup_tc)
                    dup_msg = (
                        f"Duplicate: this tool call has identical arguments "
                        f"to an earlier call in the same batch "
                        f"(id={canonical_id}). The result is already in "
                        f"your context — scroll back to the canonical "
                        f"tool_result."
                    )
                    synthetic = _TR(
                        success=False,
                        error=dup_msg,
                        validation_error=False,
                    )
                    yield _tap_tool_result(self._meta_chunk(
                        phase="executing", status="tool_result",
                        model=request.model,
                        extra={"tool_result": {
                            "id": tool_id, "tool": tool_name,
                            "success": False,
                            "output_preview": dup_msg[:_preview_len(tool_name)],  # noqa: F821
                            "batch_duplicate": True,
                            "canonical_tool_call_id": canonical_id,
                        }},
                    ), synth_tool_results)
                    self._append_tool_result_to_history(
                        messages, tool_id, tool_name, synthetic, tier,
                    )
                    counters["tool_calls"] = counters.get("tool_calls", 0) + 1

            # Synthesize tool_results for fanout-dropped calls so the
            # conversation schema stays valid and the model gets a clear
            # signal (not silence) telling it to retry in a smaller batch.
            if dropped_reads:
                from augmentum.tools.base import ToolResult as _TR
                for tc in dropped_reads:
                    tool_id, tool_name, _ = self._normalize_tool_call(tc)
                    drop_msg = (
                        f"Skipped: this tool call was part of a batch that "
                        f"exceeded the parallel-read cap ({hybrid_read_fanout}). "
                        "Only the first N calls ran. Re-emit this call on the "
                        "next iteration (on its own or in a smaller batch)."
                    )
                    synthetic = _TR(
                        success=False,
                        error=drop_msg,
                        validation_error=False,
                    )
                    yield _tap_tool_result(self._meta_chunk(
                        phase="executing", status="tool_result",
                        model=request.model,
                        extra={"tool_result": {
                            "id": tool_id, "tool": tool_name,
                            "success": False,
                            "output_preview": drop_msg[:_preview_len(tool_name)],  # noqa: F821
                            "fanout_dropped": True,
                        }},
                    ), synth_tool_results)
                    self._append_tool_result_to_history(
                        messages, tool_id, tool_name, synthetic, tier,
                    )
                    counters["tool_calls"] = counters.get("tool_calls", 0) + 1

            for tc in serial_calls:
                async for ev in self._run_tool_tracked(
                    tc, tool_map, tier, messages, request.model, counters,
                ):
                    yield _tap_tool_result(ev, synth_tool_results)

            if counters["tool_calls"] > 0:
                operate_completion_nudged = False

            if counters["writes"] > 0:
                write_iters.append(iteration)
                total_writes += counters["writes"]
                continuation_nudged = False

            if (
                counters["tool_calls"] > 0
                and counters["validation_errors"] < counters["tool_calls"]
            ):
                self._state.clear_validation_errors()

            # Circuit breaker: validation-error streak.
            if (
                counters["tool_calls"] > 0
                and counters["validation_errors"] == counters["tool_calls"]
            ):
                validation_error_streak += 1
                if (
                    self._state.safeguards_enabled
                    and validation_error_streak >= _live_threshold("validation_error_streak")
                ):
                    termination_reason = "validation_error_streak"
                    yield emit(
                        (
                            f"\n\n[Stopped: {validation_error_streak} consecutive "
                            "iterations of malformed tool calls. The model is "
                            "emitting tools with missing required arguments and "
                            "isn't recovering from the error hints. Try a "
                            "different model or rephrase the task.]\n"
                        ),
                        phase="executing", status="validation_error_break",
                        model=request.model,
                        extra={"streak": validation_error_streak},
                    )
                    async for _rev in self._reflect_on_streak_break(
                        request, break_kind="validation_error_streak",
                        streak=validation_error_streak,
                    ):
                        yield _rev
                    break
            else:
                validation_error_streak = 0

            # Same-signature break (sibling of the 5-iteration check
            # above). See _act_canonical for the rationale — the most
            # common failure mode is "model loops on the IDENTICAL bad
            # call", not "model wanders across 5 different bad calls".
            # Catch it earlier so the user isn't watching 5 broken
            # file_write attempts back-to-back.
            if self._state.safeguards_enabled:
                offender = _find_repeat_offender(self._state)
                if offender is not None:
                    termination_reason = "same_validation_error_repeat"
                    yield emit(
                        _format_repeat_break_message(offender),
                        phase="executing", status="validation_error_break",
                        model=request.model,
                        extra={
                            "tool": offender.get("tool"),
                            "repeat_count": int(offender.get("repeat_count") or 0),
                        },
                    )
                    async for _rev in self._reflect_on_streak_break(
                        request, break_kind="same_validation_error_repeat",
                        streak=int(offender.get("repeat_count") or 0),
                    ):
                        yield _rev
                    break

            #test_run failure-streak detector (2026-04-20). Break
            # when test_run has failed repeatedly without ever passing.
            # A single pass resets the streak.
            if counters.get("test_failed", 0) > 0 and counters.get("test_passed", 0) == 0:
                test_failure_streak += 1
            elif counters.get("test_passed", 0) > 0:
                test_failure_streak = 0

            if (
                self._state.safeguards_enabled
                and test_failure_streak >= _live_threshold("test_failure_streak")
            ):
                termination_reason = "test_failure_streak"
                yield emit(
                    (
                        f"\n\n[Stopped: {test_failure_streak} consecutive "
                        "iterations with failing tests and no passes. The "
                        "agent appears unable to make the tests green. "
                        "Explain what's blocking, try a different approach, "
                        "or ask the user for guidance.]\n"
                    ),
                    phase="executing", status="test_failure_streak_break",
                    model=request.model,
                    extra={"streak": test_failure_streak},
                )
                async for _rev in self._reflect_on_streak_break(
                    request, break_kind="test_failure_streak",
                    streak=test_failure_streak,
                ):
                    yield _rev
                break

            #Single-file edit ladder: early prescriptive nudge, then
            # the historical hard cap. Shared tracker with native
            # (write_churn.py).
            churn_action, thrash_path, churn_count = write_churn.observe(
                counters.get("edited_paths", []),
            )
            if churn_action == "break" and self._state.safeguards_enabled:
                termination_reason = "same_file_edit_break"
                yield emit(
                    (
                        f"\n\n[Stopped: edited {thrash_path} "
                        f"{churn_count} times this "
                        "turn without making progress. The agent is "
                        "thrashing on a single file. Stop and re-read "
                        "the file fully, or ask the user for clarity "
                        "on what the goal is.]\n"
                    ),
                    phase="executing", status="same_file_edit_break",
                    model=request.model,
                    extra={
                        "path": thrash_path,
                        "edit_count": churn_count,
                    },
                )
                async for _rev in self._reflect_on_streak_break(
                    request, break_kind="same_file_edit_break",
                    streak=churn_count,
                    extra_context=f"thrashed file: {thrash_path}",
                ):
                    yield _rev
                break
            if churn_action == "nudge" and self._state.safeguards_enabled:
                messages.append(Message(
                    role="user",
                    content=(
                        "<nudge>"
                        + churn_nudge_body(thrash_path, churn_count)
                        + "</nudge>"
                    ),
                ))
                yield self._meta_chunk(
                    phase="executing", status="same_file_edit_nudge",
                    model=request.model,
                    extra={
                        "path": thrash_path,
                        "edit_count": churn_count,
                    },
                )

            #Action-stagnation break (qwen-code port). Fires
            # AFTER same_file_edit_break so the more specific message
            # wins on overlapping patterns.
            if (
                self._state.safeguards_enabled
                and same_tool_streak >= _live_threshold("action_stagnation_break")
            ):
                termination_reason = "action_stagnation"
                yield emit(
                    (
                        f"\n\n[Stopped: {same_tool_streak} consecutive "
                        f"iterations all calling `{prev_iter_tool_name}` "
                        "with different arguments. The model is "
                        "parameter-thrashing on a single tool without "
                        "making progress. Explain what you're trying "
                        "to accomplish so a different approach can be "
                        "tried, or ask the user for guidance.]\n"
                    ),
                    phase="executing", status="action_stagnation_break",
                    model=request.model,
                    extra={
                        "streak": same_tool_streak,
                        "tool": prev_iter_tool_name,
                    },
                )
                async for _rev in self._reflect_on_streak_break(
                    request, break_kind="action_stagnation",
                    streak=same_tool_streak,
                    extra_context=f"stagnated tool: {prev_iter_tool_name}",
                ):
                    yield _rev
                break

            #Inspection-only streak detector. Fires when a task
            # with a creation verb loops on inspection tools without
            # writing anything. Grace period skips the first N iterations
            # (legitimate orientation exploration).
            iter_is_inspection_only = (
                counters.get("tool_calls", 0) > 0
                and counters.get("writes", 0) == 0
                and all(
                    tc.get("name", "") in _INSPECTION_TOOLS
                    for tc in tool_calls
                )
            )
            past_cold_start = iteration > inspection_cold_start
            if (
                iter_is_inspection_only
                and past_cold_start
                and _goal_has_creation_verb
            ):
                inspection_only_streak += 1
            elif counters.get("writes", 0) > 0:
                inspection_only_streak = 0
                inspection_nudge_fired = False

            if (
                self._state.safeguards_enabled
                and _goal_has_creation_verb
                and inspection_only_streak >= inspection_nudge_at
                and not inspection_nudge_fired
            ):
                messages.append(Message(
                    role="user",
                    content=(
                        "<nudge>The task asked you to CREATE something, but "
                        "you've only been inspecting. Write a first version "
                        "now — it doesn't have to be perfect, you can "
                        "iterate. If the goal is ambiguous, ask for "
                        "clarification.</nudge>"
                    ),
                ))
                inspection_nudge_fired = True
                yield self._meta_chunk(
                    phase="executing", status="inspection_loop_nudge",
                    model=request.model,
                    extra={"streak": inspection_only_streak},
                )
            elif (
                self._state.safeguards_enabled
                and _goal_has_creation_verb
                and inspection_nudge_fired
                and inspection_only_streak
                  >= inspection_nudge_at + inspection_break_delta
            ):
                termination_reason = "inspection_loop_break"
                yield emit(
                    (
                        f"\n\n[Stopped: {inspection_only_streak} "
                        "iterations of inspection tools with zero "
                        "creation attempts on a task that asked for "
                        "creation. The model appears stuck probing "
                        "instead of writing. Rephrase the task more "
                        "explicitly (\"create file X with Y\"), split "
                        "it into smaller pieces, or try a different "
                        "model.]\n"
                    ),
                    phase="executing", status="inspection_loop_break",
                    model=request.model,
                    extra={"streak": inspection_only_streak},
                )
                async for _rev in self._reflect_on_streak_break(
                    request, break_kind="inspection_loop_break",
                    streak=inspection_only_streak,
                ):
                    yield _rev
                break

            # Silent-success streak detector. Fires a one-shot nudge
            # when the model has had N iterations where every shell
            # call returned "(exit 0, no stdout)". The nudge steers
            # toward a diagnostic check (ps / curl / ls) before the
            # next mutation — observed 2026-04-22 a model got stuck
            # killing and restarting a server for 15 iters because
            # every kill / nohup / check returned silent success and
            # it had no grounding signal about actual state.
            iter_all_shell_silent = (
                counters.get("shell_exec_calls", 0) > 0
                and counters.get("shell_exec_silent", 0) ==
                    counters.get("shell_exec_calls", 0)
            )
            if iter_all_shell_silent:
                silent_success_streak += 1
            else:
                # A single non-silent shell iter resets — the model
                # has output to reason about now.
                silent_success_streak = 0

            if (
                self._state.safeguards_enabled
                and silent_success_streak >= _SILENT_SUCCESS_NUDGE_AT
                and not silent_success_nudge_fired
            ):
                messages.append(Message(
                    role="user",
                    content=(
                        "<nudge>Your last "
                        f"{silent_success_streak} shell commands all "
                        "returned `(exit 0, no stdout)` — successful, "
                        "but opaque. Before the next mutation, run a "
                        "diagnostic command (e.g. `ps aux | grep <name>`, "
                        "`curl http://localhost:<port>/`, `ls -la "
                        "<path>`) and READ the output. Don't guess at "
                        "state — confirm it.</nudge>"
                    ),
                ))
                silent_success_nudge_fired = True
                yield self._meta_chunk(
                    phase="executing", status="silent_success_nudge",
                    model=request.model,
                    extra={"streak": silent_success_streak},
                )

            # Failing-shell-without-edit detector. When shell_exec fails
            # in an iter AND no file write happened the same iter, the
            # agent is retrying without changing anything — the classic
            # "do the same thing, expect different result" trap. Streak
            # resets on a successful shell OR any write landing.
            iter_failing_shell_no_edit = (
                counters.get("shell_exec_failed", 0) > 0
                and counters.get("writes", 0) == 0
            )
            if iter_failing_shell_no_edit:
                failing_shell_streak += 1
            elif counters.get("writes", 0) > 0 or (
                counters.get("shell_exec_calls", 0) > 0
                and counters.get("shell_exec_failed", 0) == 0
            ):
                failing_shell_streak = 0

            if (
                self._state.safeguards_enabled
                and failing_shell_streak >= _FAILING_SHELL_NUDGE_AT
                and not failing_shell_nudge_fired
            ):
                messages.append(Message(
                    role="user",
                    content=(
                        "<nudge>You've run failing shell_exec "
                        f"{failing_shell_streak} iterations in a row "
                        "without editing any files between. Retrying "
                        "the same command won't produce a different "
                        "result — something needs to change. Options: "
                        "(1) read the relevant file and fix the code, "
                        "(2) check the error's root cause (missing "
                        "dep? wrong path? permission?), (3) explain "
                        "what you think is wrong and ask the user. "
                        "Don't just retry.</nudge>"
                    ),
                ))
                failing_shell_nudge_fired = True
                yield self._meta_chunk(
                    phase="executing", status="failing_shell_nudge",
                    model=request.model,
                    extra={"streak": failing_shell_streak},
                )

            # Write-without-progress detector. Counts iterations where
            # at least one mutating tool was CALLED but ZERO writes
            # stuck. Catches the degenerate state where the agent is
            # working (not just reading) but nothing it produces is
            # landing — failed code_edit searches, idempotence no-ops,
            # file_write permission errors, etc.
            mutating_attempted_this_iter = any(
                tc.get("name", "") in _MUTATING_TOOL_NAMES
                for tc in tool_calls
            )
            if mutating_attempted_this_iter and counters.get("writes", 0) == 0:
                no_write_progress_streak += 1
            elif counters.get("writes", 0) > 0:
                no_write_progress_streak = 0

            # Task-list staleness detector — shared TaskSpineTracker
            # (augmentum/coder/task_spine.py). One-shot nudge when the
            # list has open work but hasn't changed for
            # TASK_STALE_NUDGE_AT iterations; re-armed on any real
            # mutation. The all-completed case is handled by the
            # loop's ``tasks_completed`` early termination at
            # iteration top.
            tasks_mutated, task_stale_nudge = task_spine.observe(
                self._state.tasks,
                nudge_enabled=self._state.safeguards_enabled,
            )

            # Mid-turn persist so the inspector panel reflects task_list /
            # mission mutations without waiting for turn end. Cheap when
            # nothing changed (hash-and-skip inside the helper).
            if tasks_mutated:
                try:
                    await self._persist_state_if_dirty()
                except Exception:
                    log.debug("coder_mid_turn_persist_failed", exc_info=True)

            if task_stale_nudge:
                messages.append(Message(
                    role="user",
                    content=f"<nudge>{task_stale_nudge}</nudge>",
                ))
                yield self._meta_chunk(
                    phase="executing", status="task_stale_nudge",
                    model=request.model,
                    extra={"streak": task_spine.stale_streak},
                )

            coordination_tools = {"task_list", "ask_user"}
            iter_coordination_only = bool(tool_calls) and all(
                tc.get("name", "") in coordination_tools for tc in tool_calls
            )
            if iter_coordination_only and counters.get("writes", 0) == 0:
                coordination_only_streak += 1
            else:
                coordination_only_streak = 0
                coordination_nudge_fired = False

            if (
                self._state.safeguards_enabled
                and coordination_only_streak >= _COORDINATION_ONLY_NUDGE_AT
                and not coordination_nudge_fired
            ):
                messages.append(Message(
                    role="user",
                    content=(
                        "<nudge>You've spent "
                        f"{coordination_only_streak} iterations only updating task "
                        "coordination (task_list / ask_user) without taking a concrete "
                        "debug, test, read, or edit action. Stop coordinating and do the "
                        "next smallest real step. If you truly need the user, ask one "
                        "targeted blocker question instead of another task-list refresh.</nudge>"
                    ),
                ))
                coordination_nudge_fired = True
                yield self._meta_chunk(
                    phase="executing",
                    status="stagnation_nudge",
                    model=request.model,
                    extra={"streak": coordination_only_streak, "kind": "coordination_only"},
                )

            if (
                self._state.safeguards_enabled
                and no_write_progress_streak >= _live_threshold("no_write_progress_break")
            ):
                termination_reason = "no_write_progress"
                yield emit(
                    (
                        f"\n\n[Stopped: {no_write_progress_streak} "
                        "iterations of attempted edits with zero "
                        "successful writes — every code_edit / "
                        "file_write bounced (stale search-block, "
                        "idempotence no-op, or validation error). "
                        "The agent is calling tools but nothing is "
                        "sticking. Re-read the target file(s) to "
                        "confirm their current state, or ask the "
                        "user what outcome is expected.]\n"
                    ),
                    phase="executing", status="no_write_progress_break",
                    model=request.model,
                    extra={"streak": no_write_progress_streak},
                )
                async for _rev in self._reflect_on_streak_break(
                    request, break_kind="no_write_progress",
                    streak=no_write_progress_streak,
                ):
                    yield _rev
                break

        else:
            termination_reason = "max_iterations_reached"
            yield self._meta_chunk(
                phase="executing", status="max_iterations_reached",
                model=request.model, extra={"reason": termination_reason},
            )

        # finish_task short-circuit: the model wrote the user-facing
        # answer as the tool's ``summary`` argument — emit that
        # verbatim and skip synthesis / fallback entirely. Running
        # either would either stack a redundant summary or generate a
        # weaker reconstruction of what the model already said cleanly.
        if termination_reason == "finish_task_called" and self._state.finish_summary:
            yield emit(
                self._state.finish_summary,
                phase="executing", status="streaming",
                model=request.model,
            )
            total_prose_chars += len(self._state.finish_summary)

        # Final-turn summary when the model narrated nothing. Tool
        # results are invisible to the user in our UI (only tool names
        # render), so a silent stop leaves them with an empty "Done"
        # and no explanation. Two paths are available:
        #
        #   1. Phase 6 LLM synthesis (opt-in via
        #      AUGMENTUM_CODER_SYNTHESIZE_HYBRID=1). Calls
        #      ``_synthesize_response`` with the full tool-result
        #      rollup to produce a natural-language summary. Higher
        #      quality, costs one extra backend round-trip, can fail.
        #   2. Deterministic fallback (always-on safety net). Renders
        #      a terse report from counters + state. Fast, never
        #      fails, but reads as mechanical.
        #
        # If synthesis is enabled AND succeeds, skip the deterministic
        # fallback (they'd stack visibly in chat). Synthesis failure
        # falls through to the fallback so the user always sees
        # something when the model did real work.
        needs_summary = (
            total_prose_chars < _HYBRID_MIN_TURN_PROSE_CHARS
            and (total_writes > 0 or self._state.tool_calls_made > 0)
        )
        if needs_summary:
            import os as _os
            synth_enabled = (
                _os.environ.get("AUGMENTUM_CODER_SYNTHESIZE_HYBRID", "").lower()
                in ("1", "true", "yes", "on")
            )
            synthesis_succeeded = False
            if synth_enabled and synth_tool_results:
                try:
                    emitted_any = False
                    async for chunk in self._synthesize_response(
                        request, user_goal or "", synth_tool_results,
                    ):
                        if chunk.content_delta:
                            emitted_any = True
                        yield chunk
                    synthesis_succeeded = emitted_any
                except Exception as exc:
                    log.warning(
                        "hybrid_synthesis_failed",
                        error=str(exc),
                        termination_reason=termination_reason,
                    )

            if not synthesis_succeeded:
                fallback = self._render_fallback_summary(
                    iteration=iteration,
                    total_writes=total_writes,
                    termination_reason=termination_reason,
                    same_file_edits=same_file_edits,
                    messages=messages,
                    user_goal=user_goal,
                    tool_results=synth_tool_results,
                )
                yield emit(
                    fallback,
                    phase="executing", status="fallback_summary",
                    model=request.model,
                    extra={
                        "total_prose_chars": total_prose_chars,
                        "termination_reason": termination_reason,
                    },
                )

        #Persist this turn's trace for the next user turn's
        # system prompt. See turn_summaries doc in CoderState.
        if synth_tool_results or any(m.role == "tool" for m in messages):
            summary = self._build_turn_summary(
                messages=messages, user_goal=user_goal,
                termination_reason=termination_reason,
            )
            self._state.add_turn_summary(summary)
            await self._archive_turn_summary(summary)

        # Publish the reviewable-turn bundle — see _act_canonical for
        # the contract. No-op on turns without mutations or without a
        # wired registry.
        await self._publish_turn_review(user_goal)

        self._state.phase = CoderPhase.WAITING
        yield self._meta_chunk(
            phase="waiting", status="complete",
            model=request.model,
            extra={
                "strategy": "hybrid",
                "tool_calls_made": self._state.tool_calls_made,
                "termination_reason": termination_reason,
                "review_turn_id": self._state.active_turn_id,
                "iterations_used": iteration,
                "observations_injected": iteration // hybrid_observation_every,
            },
        )
