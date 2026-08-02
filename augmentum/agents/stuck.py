"""Five-pattern semantic stuck detector.

Port of the patterns OpenHands shipped after issue #5480 (agents that
looped without the user being able to recover). Compares actions
semantically (tool name + content + thought) ignoring IDs, timestamps,
and metrics — two parallel reads of the same file are equivalent
regardless of their call IDs.

Sleep-polling exception (lesson from OpenHands #5355): a subagent
legitimately polling a long-running job with ``sleep`` / ``wait`` looks
like a same-action loop to a naive detector. We exclude actions whose
tool name matches a sleep-primitive whitelist from the identical-pair
and ping-pong patterns. Monologue and context-window-error patterns are
unaffected.
"""

from __future__ import annotations

import hashlib
from collections import deque
from dataclasses import dataclass, field
from enum import Enum


class StuckPattern(str, Enum):
    IDENTICAL_ACTION_OBSERVATION = "identical_action_observation"
    IDENTICAL_ACTION_ERROR = "identical_action_error"
    AGENT_MONOLOGUE = "agent_monologue"
    PING_PONG = "ping_pong"
    REPEATED_CONTEXT_OVERFLOW = "repeated_context_overflow"


# Tool names that legitimately repeat in a polling loop. Matched as a
# prefix on the lowercased tool name so "shell_exec(sleep 5)" registers
# only if the action *itself* is the wait — we still detect a stuck
# read-the-same-file loop regardless of any sleeps interleaved.
_SLEEP_PRIMITIVES = frozenset({"sleep", "wait"})

# Substring matched against observation/error text — present on the
# llama-server context-overflow path, the OpenAI 400-too-many-tokens
# path, and Anthropic's overflow message. Cheap, no false positives in
# practice.
_CTX_OVERFLOW_MARKERS = (
    "context length",
    "context_length_exceeded",
    "maximum context length",
    "prompt is too long",
    "too many tokens",
)


# Thresholds — direct port of OpenHands' field-tested values. Do not
# tune these on a hunch; if a workload trips false positives, add to
# the polling-primitive list or report a bug.
_T_IDENTICAL_ACTION_OBS = 4
_T_IDENTICAL_ACTION_ERR = 3
_T_MONOLOGUE = 3
_T_PING_PONG = 6


def _semantic_hash(tool: str, content: str, thought: str) -> str:
    """Stable fingerprint for a subagent action.

    Excludes everything caller-volatile (IDs, retry counts, timestamps).
    Truncates content/thought because tiny formatting drift (a trailing
    newline, an updated timestamp string in a log line) shouldn't break
    semantic equality.
    """
    payload = f"{tool.strip().lower()}|{content.strip()[:512]}|{thought.strip()[:256]}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _is_sleep_primitive(tool: str) -> bool:
    name = (tool or "").strip().lower()
    if not name:
        return False
    # Match bare tool name or tool-arg form like "shell_exec sleep ..."
    head = name.split("(", 1)[0].strip()
    head = head.split(" ", 1)[0]
    return head in _SLEEP_PRIMITIVES


def _contains_overflow_marker(text: str) -> bool:
    lowered = (text or "").lower()
    return any(marker in lowered for marker in _CTX_OVERFLOW_MARKERS)


@dataclass
class Turn:
    """One subagent turn. ``observation`` is the tool result (or empty for
    monologue turns), ``error`` is non-empty on tool failure."""

    tool: str
    content: str  # tool input / agent text content
    thought: str  # the reasoning/explanation that preceded the action
    observation: str = ""
    error: str = ""


@dataclass
class StuckResult:
    stuck: bool
    pattern: StuckPattern | None = None
    detail: str = ""


@dataclass
class StuckDetector:
    """Stateful detector. ``record(turn)`` once per subagent turn; call
    ``check()`` to test whether the loop should bail."""

    history: deque[Turn] = field(default_factory=lambda: deque(maxlen=64))

    def record(self, turn: Turn) -> None:
        self.history.append(turn)

    def check(self) -> StuckResult:
        # Order: cheapest-and-most-decisive checks first.
        ctx = self._check_context_overflow()
        if ctx.stuck:
            return ctx
        ident_obs = self._check_identical_action_observation()
        if ident_obs.stuck:
            return ident_obs
        ident_err = self._check_identical_action_error()
        if ident_err.stuck:
            return ident_err
        ping = self._check_ping_pong()
        if ping.stuck:
            return ping
        mono = self._check_monologue()
        if mono.stuck:
            return mono
        return StuckResult(stuck=False)

    def _check_context_overflow(self) -> StuckResult:
        if not self.history:
            return StuckResult(stuck=False)
        last = self.history[-1]
        if _contains_overflow_marker(last.error) or _contains_overflow_marker(last.observation):
            return StuckResult(
                stuck=True,
                pattern=StuckPattern.REPEATED_CONTEXT_OVERFLOW,
                detail="context window exceeded; subagent cannot recover in-band",
            )
        return StuckResult(stuck=False)

    def _check_identical_action_observation(self) -> StuckResult:
        if len(self.history) < _T_IDENTICAL_ACTION_OBS:
            return StuckResult(stuck=False)
        tail = list(self.history)[-_T_IDENTICAL_ACTION_OBS:]
        if any(_is_sleep_primitive(t.tool) for t in tail):
            return StuckResult(stuck=False)
        if any(t.error for t in tail):
            return StuckResult(stuck=False)
        sig = _semantic_hash(tail[0].tool, tail[0].content, tail[0].thought)
        obs_sig = (tail[0].observation or "").strip()[:512]
        for t in tail[1:]:
            if _semantic_hash(t.tool, t.content, t.thought) != sig:
                return StuckResult(stuck=False)
            if (t.observation or "").strip()[:512] != obs_sig:
                return StuckResult(stuck=False)
        return StuckResult(
            stuck=True,
            pattern=StuckPattern.IDENTICAL_ACTION_OBSERVATION,
            detail=f"same action+observation repeated {_T_IDENTICAL_ACTION_OBS}x",
        )

    def _check_identical_action_error(self) -> StuckResult:
        if len(self.history) < _T_IDENTICAL_ACTION_ERR:
            return StuckResult(stuck=False)
        tail = list(self.history)[-_T_IDENTICAL_ACTION_ERR:]
        if any(_is_sleep_primitive(t.tool) for t in tail):
            return StuckResult(stuck=False)
        if not all(t.error for t in tail):
            return StuckResult(stuck=False)
        sig = _semantic_hash(tail[0].tool, tail[0].content, tail[0].thought)
        err_sig = tail[0].error.strip()[:256]
        for t in tail[1:]:
            if _semantic_hash(t.tool, t.content, t.thought) != sig:
                return StuckResult(stuck=False)
            if t.error.strip()[:256] != err_sig:
                return StuckResult(stuck=False)
        return StuckResult(
            stuck=True,
            pattern=StuckPattern.IDENTICAL_ACTION_ERROR,
            detail=f"same action+error repeated {_T_IDENTICAL_ACTION_ERR}x",
        )

    def _check_monologue(self) -> StuckResult:
        if len(self.history) < _T_MONOLOGUE:
            return StuckResult(stuck=False)
        tail = list(self.history)[-_T_MONOLOGUE:]
        for t in tail:
            if t.tool or t.observation or t.error:
                return StuckResult(stuck=False)
        return StuckResult(
            stuck=True,
            pattern=StuckPattern.AGENT_MONOLOGUE,
            detail=f"{_T_MONOLOGUE} consecutive agent turns with no environment input",
        )

    def _check_ping_pong(self) -> StuckResult:
        window = _T_PING_PONG * 2
        if len(self.history) < window:
            return StuckResult(stuck=False)
        tail = list(self.history)[-window:]
        if any(_is_sleep_primitive(t.tool) for t in tail):
            return StuckResult(stuck=False)
        sig_a = _semantic_hash(tail[0].tool, tail[0].content, tail[0].thought)
        sig_b = _semantic_hash(tail[1].tool, tail[1].content, tail[1].thought)
        if sig_a == sig_b:
            return StuckResult(stuck=False)
        for i, t in enumerate(tail):
            expected = sig_a if i % 2 == 0 else sig_b
            if _semantic_hash(t.tool, t.content, t.thought) != expected:
                return StuckResult(stuck=False)
        return StuckResult(
            stuck=True,
            pattern=StuckPattern.PING_PONG,
            detail=f"alternating between two actions for {_T_PING_PONG} cycles",
        )
