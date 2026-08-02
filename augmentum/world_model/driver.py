"""AgentWorldDriver — Qwen-AgentWorld as a callable environment simulator.

Wraps the Slot B backend (``app.state.secondary_slot``) with the
AgentWorld interaction contract, matched to how the model was trained
and how AgentWorldBench evaluates it (QwenLM/Qwen-AgentWorld eval code):

- one system prompt per domain (vendored templates from
  ``prompts/{domain}/system_prompt.txt``; upstream marks them "templates
  for reference" — callers with a benchmark sample's own ``system_str``
  should pass ``system_override``)
- the episode is serialized into ONE user message: agent actions as-is,
  each environment observation prefixed with the
  ``**Environment Observation:**`` marker. That is the one-turn-per-
  trajectory shape the model was trained on; multi-turn chat framing is
  off-distribution (kept available via ``serialized=False``).
- the model thinks in ``<think>`` and wraps its prediction in
  ``<predicted_observation>`` tags — an anti-reward-hacking measure so
  self-praise outside the tags never reaches the judge. We extract the
  tag body (last block wins, robust to unclosed think tags) and strip
  the response marker.

Sampling follows the model card (temperature 0.6) plus the Qwen3.5
thinking-family guidance (top_p 0.95, top_k 20 — greedy causes
repetition on this family).

The driver NEVER auto-loads a model. Loading AgentWorld into Slot B is
the user's action in the model manager; when the slot is empty or holds
something else, ``simulate`` raises :class:`WorldModelUnavailable` with
a message that says exactly that.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from augmentum.models.base import InternalChatRequest, Message
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.models.secondary_slot import SecondarySlot

log = get_logger(__name__)

WORLD_DOMAINS = ("android", "mcp", "os", "search", "swe", "terminal", "web")

# AgentWorld's trained output contract (AgentWorldBench task_configs.py):
# identical across all seven domains.
RESPONSE_TAG = "predicted_observation"
RESPONSE_MARKER = "**Environment Observation:**"

_PROMPTS_DIR = Path(__file__).parent / "prompts"

# Substring match against the loaded model id — how we recognize that
# Slot B actually holds a world model rather than an ordinary chat
# model. A wrong model would happily "simulate" an environment from its
# imagination, which is worse than an error.
_WORLD_MODEL_MARKERS = ("agentworld", "webworld")


class WorldModelUnavailable(RuntimeError):
    """Slot B is not serving a world model right now."""


@dataclass
class WorldStep:
    """One simulated environment transition."""

    observation: str
    thinking: str | None
    model: str
    domain: str
    latency_ms: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Whether the observation came out of a <predicted_observation> tag
    # (on-contract) or fell back to the whole cleaned response. A run of
    # tag_found=False means the model is drifting off its format —
    # surface it, don't hide it.
    tag_found: bool = True
    # The unextracted response content, so extraction never silently
    # loses anything a caller might need for debugging.
    raw_content: str = field(default="", repr=False)


def load_domain_prompt(domain: str) -> str:
    """Read the vendored system prompt for ``domain``. Raises on unknown."""
    if domain not in WORLD_DOMAINS:
        raise ValueError(
            f"unknown world domain {domain!r} — expected one of {WORLD_DOMAINS}"
        )
    return (_PROMPTS_DIR / f"{domain}.txt").read_text(encoding="utf-8")


def serialize_episode(history: list[dict[str, str]], initial_state: str = "") -> str:
    """Render an episode as ONE user message, the way AgentWorld was trained.

    Agent actions (``user`` turns) pass through as-is; environment
    observations (``assistant`` turns) get the ``**Environment
    Observation:**`` marker if they don't already carry it. A detailed
    ``initial_state`` (upstream stresses this drives simulation quality)
    leads the episode as the first observation block.
    """
    parts: list[str] = []
    if initial_state.strip():
        parts.append(f"{RESPONSE_MARKER}\n{initial_state.strip()}")
    for m in history:
        content = str(m.get("content") or "").strip()
        if not content:
            continue
        if m.get("role") == "assistant" and not content.startswith(RESPONSE_MARKER):
            content = f"{RESPONSE_MARKER}\n{content}"
        parts.append(content)
    return "\n\n".join(parts)


def _strip_think_blocks(text: str, response_tag: str = RESPONSE_TAG) -> str:
    """Remove <think> blocks, robust to malformed tags.

    Ported from AgentWorldBench's ``_remove_thinking_tags``: handles
    matched pairs, multiple blocks, orphaned closers (drop the prefix),
    and orphaned openers (drop to the next ``<response_tag>`` if present,
    else to end-of-string — so an unclosed think can't eat the answer).
    """
    if not text:
        return text
    tags = [(m.start(), "open") for m in re.finditer(r"<think>", text, re.IGNORECASE)]
    tags += [(m.start(), "close") for m in re.finditer(r"</think>", text, re.IGNORECASE)]
    if not tags:
        return text
    tags.sort(key=lambda t: t[0])

    close_len = len("</think>")
    ranges: list[tuple[int, int]] = []
    used: set[int] = set()
    for i, (pos, kind) in enumerate(tags):
        if i in used or kind != "open":
            continue
        for j in range(i + 1, len(tags)):
            if j in used:
                continue
            j_pos, j_kind = tags[j]
            if j_kind == "close":
                ranges.append((pos, j_pos + close_len))
                used.update((i, j))
                break
    for i, (pos, kind) in enumerate(tags):
        if i in used:
            continue
        if kind == "close":
            ranges.append((0, pos + close_len))
        else:  # orphaned opener
            end = len(text)
            m = re.search(rf"<{re.escape(response_tag)}>", text[pos:], re.IGNORECASE)
            if m:
                end = pos + m.start()
            ranges.append((pos, end))
        used.add(i)

    ranges.sort()
    merged: list[tuple[int, int]] = []
    for start, end in ranges:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    out: list[str] = []
    prev = 0
    for start, end in merged:
        if start > prev:
            out.append(text[prev:start])
        prev = end
    out.append(text[prev:])
    return "".join(out).strip()


def _strip_marker(text: str) -> str:
    text = text.strip()
    if text.startswith(RESPONSE_MARKER):
        return text[len(RESPONSE_MARKER):].strip()
    # Upstream's clean_response_marker removes stray occurrences too.
    return text.replace(RESPONSE_MARKER, "").strip()


def extract_observation(raw: str) -> tuple[str, bool]:
    """Pull the prediction out of ``<predicted_observation>`` tags.

    Returns ``(observation, tag_found)``. Mirrors AgentWorldBench's
    ``parse_model_output``: strip think blocks first (so fake tags
    inside reasoning don't win), take the LAST tag block, tolerate a
    missing closer, fall back to the whole cleaned text when the model
    skipped the tags entirely.
    """
    cleaned = _strip_think_blocks(raw or "")
    starts = list(re.finditer(rf"<{re.escape(RESPONSE_TAG)}>", cleaned, re.IGNORECASE))
    if not starts:
        return _strip_marker(cleaned), False
    start = starts[-1].end()
    close = re.search(rf"</{re.escape(RESPONSE_TAG)}>", cleaned[start:], re.IGNORECASE)
    body = cleaned[start : start + close.start()] if close else cleaned[start:]
    return _strip_marker(body), True


class AgentWorldDriver:
    """Stateless facade over Slot B for world-model inference.

    Stateless by design: the environment state lives in the caller's
    trajectory (the message history), exactly as AgentWorld was trained.
    That keeps episodes replayable and lets many concurrent episodes
    share one resident model.
    """

    def __init__(self, secondary_slot: SecondarySlot | None) -> None:
        self._slot = secondary_slot

    # -- readiness -------------------------------------------------------

    def status(self) -> dict[str, Any]:
        """Report whether a world model is currently servable.

        "Servable" includes the pinned-but-not-resident state: after a
        container restart Slot B keeps its pin (``engine_secondary_model``)
        and lazy-loads on first use, so a simulate call works — the first
        one just pays the model-load latency. ``resident`` distinguishes
        the two for callers that care.
        """
        if self._slot is None:
            return {
                "available": False,
                "reason": "Slot B (engine_secondary_enabled) is disabled",
            }
        st = self._slot.status()
        resident_id = str(st.get("model_id") or "")
        loaded = bool(st.get("loaded"))
        pinned_id = ""
        if not loaded:
            from augmentum.config import settings

            pinned_id = str(getattr(settings, "engine_secondary_model", "") or "")
        model_id = resident_id or pinned_id
        is_world = self._looks_like_world_model(model_id)
        return {
            "available": bool(model_id) and is_world,
            "loaded": loaded,
            "resident": loaded,
            "model_id": model_id,
            "is_world_model": is_world,
            "domains": list(WORLD_DOMAINS),
            "reason": (
                "" if model_id and is_world
                else "no model loaded in Slot B — load Qwen-AgentWorld via the "
                     "model manager's '2nd slot' button" if not model_id
                else f"Slot B holds {model_id!r}, which does not look like a "
                     "world model (expected an AgentWorld/WebWorld build)"
            ),
        }

    @staticmethod
    def _looks_like_world_model(model_id: str) -> bool:
        low = model_id.lower()
        return any(marker in low for marker in _WORLD_MODEL_MARKERS)

    def _require_backend(self):
        if self._slot is None:
            raise WorldModelUnavailable(
                "Slot B is disabled (engine_secondary_enabled=False) — the "
                "world-model driver needs it to host AgentWorld."
            )
        st = self.status()
        if not st["available"]:
            raise WorldModelUnavailable(st["reason"])
        backend = self._slot.backend
        if backend is None:
            raise WorldModelUnavailable("Slot B backend is not constructed")
        return backend, st["model_id"]

    # -- simulation ------------------------------------------------------

    async def simulate(
        self,
        domain: str,
        history: list[dict[str, str]],
        *,
        initial_state: str = "",
        system_override: str = "",
        max_tokens: int = 8192,
        temperature: float = 0.6,
        serialized: bool = True,
    ) -> WorldStep:
        """Predict the next environment observation for ``history``.

        ``history`` is the episode so far as ``{role, content}`` dicts —
        agent actions as ``user`` turns, prior observations as
        ``assistant`` turns, ending on the action to simulate. By
        default it is serialized into one user message (the trained
        format); ``serialized=False`` keeps raw multi-turn chat framing
        for experiments. ``initial_state`` should describe the starting
        environment in detail — upstream found it drives sim quality.
        The domain system prompt is prepended here; don't include one
        in ``history``.
        """
        backend, model_id = self._require_backend()
        if not history:
            raise ValueError("history must contain at least the action to simulate")
        if history[-1].get("role") != "user":
            raise ValueError(
                "history must end with a user-role agent action "
                f"(got role={history[-1].get('role')!r})"
            )
        for m in history:
            if (m.get("role") or "") not in ("user", "assistant"):
                raise ValueError(
                    f"history roles must be user/assistant, got {m.get('role')!r} — "
                    "the domain system prompt is supplied by the driver"
                )

        system_text = system_override or load_domain_prompt(domain)
        messages = [Message(role="system", content=system_text)]
        if serialized:
            messages.append(
                Message(role="user", content=serialize_episode(history, initial_state))
            )
        else:
            if initial_state.strip():
                # Environment speaks first: lead with the starting state
                # as an observation turn.
                messages.append(
                    Message(
                        role="assistant",
                        content=f"{RESPONSE_MARKER}\n{initial_state.strip()}",
                    )
                )
            for m in history:
                messages.append(
                    Message(role=m["role"], content=str(m.get("content") or ""))
                )

        request = InternalChatRequest(
            model=model_id,
            messages=messages,
            stream=False,
            temperature=temperature,
            top_p=0.95,
            top_k=20,
            max_tokens=max_tokens,
            think=True,
        )

        t0 = time.monotonic()
        response = await backend.chat(request)
        latency_ms = int((time.monotonic() - t0) * 1000)

        raw_content = (response.message.content or "").strip()
        observation, tag_found = extract_observation(raw_content)
        thinking = response.message.thinking or None
        usage = response.usage
        step = WorldStep(
            observation=observation,
            thinking=thinking,
            model=response.model or model_id,
            domain=domain,
            latency_ms=latency_ms,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            tag_found=tag_found,
            raw_content=raw_content,
        )
        log.info(
            "world_model_step",
            domain=domain,
            turns=len(history),
            latency_ms=latency_ms,
            prompt_tokens=step.prompt_tokens,
            completion_tokens=step.completion_tokens,
            tag_found=tag_found,
            empty=not observation,
        )
        if not tag_found and raw_content:
            log.warning(
                "world_model_off_format",
                domain=domain,
                detail="no <predicted_observation> tag in response — "
                "used cleaned full text as the observation",
            )
        if not observation:
            # An empty observation is a real failure for a simulator —
            # surface it rather than letting an episode silently continue
            # on a blank environment state.
            raise WorldModelUnavailable(
                "world model returned an empty observation "
                f"(domain={domain}, turns={len(history)}, tag_found={tag_found})"
            )
        return step
