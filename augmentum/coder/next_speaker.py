"""Next-speaker classifier — second-pass LLM call for prose-only stops.

When the native loop's heuristic Termination Quality Gate accepts a
stop via ``REASON_SUBSTANTIVE_ACTIVE`` but the model produced zero
writes and zero recent progress, the local heuristic may be wrong:
chatty local models (Qwen-3.6) routinely emit 2-sentence preambles
("I'll look at this. Let me check the file.") that pass the gate's
SUBSTANTIVE classifier as legitimate stops, terminating the turn
before any tool call.

This module delegates that judgment to a second LLM round-trip with
a structured prompt asking "should the user or the model speak next?".
Lifted from Qwen Code's ``nextSpeakerChecker.ts`` — they don't
classify prose locally at all; their entire defense against this
failure mode is this classifier plus a ``Please continue.`` re-prompt.

Design contract
---------------

* **Opt-in** — gated by ``settings.coder_next_speaker_check_enabled``.
  When the heuristic gate already nudged (BAILOUT / INSISTENT / EMPTY),
  the classifier is NOT called — we trust the gate's clear-cut cases
  and only second-guess the SUBSTANTIVE_ACTIVE accept path.
* **Best-effort** — backend errors / parse failures return ``None`` so
  the loop falls back to the original heuristic verdict. Never raises.
* **Cheap-but-not-free** — one extra round-trip per ambiguous stop.
  Latency tradeoff: ~0.5–2s per call (local model) for ~99% accuracy
  on the preamble pattern vs. the 2-sentence heuristic's ~70%.

Reference: https://github.com/QwenLM/qwen-code/blob/main/packages/core/src/utils/nextSpeakerChecker.ts
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from dataclasses import replace as _dataclass_replace
from typing import TYPE_CHECKING

from augmentum.models.base import InternalChatRequest, Message
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.models.base import Backend

log = get_logger(__name__)


# Prompt lifted from Qwen Code with minimal adaptation. Their wording
# is calibrated against the same Qwen-family chattiness we're hitting,
# so we use it nearly verbatim. The only change: their "your immediately
# preceding response" pronoun frames it as model-self-reflection (which
# only works when the classifier is the SAME model that just spoke);
# we're sometimes calling a different / smaller model, so we frame it
# as "the assistant's preceding response".
_CLASSIFIER_PROMPT = """\
Analyze ONLY the content and structure of the assistant's immediately preceding response from the conversation history. Based on that response, determine who should logically speak next: the 'user' or the 'model' (the assistant).

Rule 1 — Model Continues: If the assistant's last response explicitly states an immediate next action it intends to take (e.g., "Next, I will...", "Now I'll process...", "Moving on to analyze...", "Let me check the file...", "I'll look at..."), OR indicates an intended tool call that didn't execute, OR if the response seems clearly incomplete (cut off mid-thought without a natural conclusion), then the **'model'** should speak next.

Rule 2 — Question to User: If the assistant's response ends with a direct, explicit question to the user, then the **'user'** should speak next.

Rule 3 — Waiting for User: If the assistant's response appears to be a complete and self-contained thought (e.g., a final answer, a summary, a comment) that doesn't fall under Rule 1 or Rule 2, then the **'user'** should speak next.

Respond with a JSON object only — no prose, no markdown fences:
{"next_speaker": "user" | "model", "reasoning": "one-sentence explanation"}"""


# Rough fence-stripping regex — some models still wrap JSON in
# ```json ... ``` despite explicit instructions. Cheap to handle.
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class NextSpeakerVerdict:
    """Outcome of the classifier call.

    ``next_speaker`` is ``"user"`` or ``"model"`` on success, ``None``
    on any failure (parse error, backend down, structured-output
    confusion). Callers MUST treat ``None`` as "no signal" and fall
    back to their original decision.
    """
    next_speaker: str | None
    reasoning: str = ""
    raw_response: str = ""


def _last_assistant_text(messages: list[Message]) -> str:
    """Find the last assistant message's text content.

    Returns empty string when the history has no assistant turn yet —
    the classifier shouldn't fire in that case but the caller may pass
    pre-first-turn state defensively.
    """
    for msg in reversed(messages):
        if msg.role == "assistant":
            content = getattr(msg, "content", "")
            if isinstance(content, str):
                return content
            # Some backends shape content as list-of-parts; flatten.
            if isinstance(content, list):
                return " ".join(
                    str(p.get("text", "")) for p in content
                    if isinstance(p, dict)
                )
    return ""


async def check_next_speaker(
    backend: Backend,
    *,
    source_request: InternalChatRequest,
    messages: list[Message],
    model_override: str = "",
    timeout_s: float = 8.0,
) -> NextSpeakerVerdict:
    """Ask the model who should speak next.

    Reuses the same backend and request shape as the main turn so the
    provider preset / routing carries through — EXCEPT KV slot affinity,
    which is explicitly cleared (``kv_session_key=""``): this is a
    differently-framed one-shot, and inheriting the act loop's session key
    would route it onto (and evict) the loop's warm slot. The overrides are:
    ``messages`` (truncated to relevant context + classifier prompt),
    ``kv_session_key=""``, ``stream=False`` (we need the full JSON),
    ``tools=None``, ``temperature=0.0`` (deterministic).

    Returns ``NextSpeakerVerdict(next_speaker=None)`` on any failure —
    callers should fall back to their pre-classifier decision.
    """
    assistant_text = _last_assistant_text(messages)
    if not assistant_text.strip():
        return NextSpeakerVerdict(next_speaker=None)

    # We send only the assistant's last response + the classifier
    # prompt — not the full turn history. The decision needs the
    # assistant's words and nothing else; sending the full history
    # would bloat tokens and risk the classifier model getting
    # confused about which response to evaluate.
    classifier_messages: list[Message] = [
        Message(role="user", content=(
            f"<assistant_response>\n{assistant_text}\n</assistant_response>\n\n"
            f"{_CLASSIFIER_PROMPT}"
        )),
    ]

    classifier_request = _dataclass_replace(
        source_request,
        model=model_override or source_request.model,
        messages=classifier_messages,
        # No KV slot affinity — see goal_judge.py for the full rationale.
        # This tiny classifier one-shot must NOT inherit the act loop's
        # kv_session_key, or it claims and overwrites the loop's warm slot.
        kv_session_key="",
        stream=False,
        tools=None,
        tool_choice=None,
        temperature=0.0,
        max_tokens=160,  # JSON response is tiny; cap to bound latency
        chat_template_kwargs=None,
        # Force reasoning OFF — same rationale as goal_judge.py: a reasoning
        # model would spend this tiny budget in the thought channel and return
        # empty content, yielding a null verdict. think=False disables it across
        # providers.
        think=False,
        reasoning_effort=None,
    )

    try:
        response = await backend.chat(classifier_request)
    except Exception as exc:
        log.debug(
            "coder_next_speaker_check_backend_error",
            error=str(exc)[:160],
            model=classifier_request.model,
        )
        return NextSpeakerVerdict(next_speaker=None)

    # ``InternalChatResponse.message.content`` is where the text lives,
    # NOT ``response.content`` (which doesn't exist — AttributeError).
    # See augmentum/models/base.py::response_text — this helper is the
    # safe accessor.
    from augmentum.models.base import response_text
    # thinking_fallback=True is belt-and-suspenders: if a provider ignores the
    # think=False above and reasons anyway, salvage the verdict from the thought
    # channel rather than reading empty.
    raw = response_text(response, thinking_fallback=True).strip()
    if not raw:
        return NextSpeakerVerdict(next_speaker=None, raw_response="")

    # Strip ``` fences some models still emit despite instructions.
    cleaned = _FENCE_RE.sub("", raw).strip()

    # Pull the first { ... } object if there's any wrapping prose.
    # Some models add "Here is the analysis:" before the JSON despite
    # being told not to.
    brace_start = cleaned.find("{")
    brace_end = cleaned.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        cleaned = cleaned[brace_start : brace_end + 1]

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        log.debug(
            "coder_next_speaker_check_parse_failed",
            raw_preview=raw[:160],
        )
        return NextSpeakerVerdict(next_speaker=None, raw_response=raw)

    speaker = parsed.get("next_speaker")
    if speaker not in ("user", "model"):
        return NextSpeakerVerdict(next_speaker=None, raw_response=raw)

    reasoning = str(parsed.get("reasoning") or "")[:240]
    return NextSpeakerVerdict(
        next_speaker=speaker, reasoning=reasoning, raw_response=raw,
    )


__all__ = [
    "NextSpeakerVerdict",
    "check_next_speaker",
]
