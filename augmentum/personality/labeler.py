"""Post-response facet labeler.

After a turn completes, ask a small/cheap LLM to label the response with
which facets from the vocabulary were active and with what intensity. The
labels feed PersonalityStore.record_activations, which writes the audit
log AND updates pairwise cooccurrence in one transaction.

Design choices:
  - Decoupled from any specific backend. The caller supplies an
    `llm_call(messages) -> str` async callable; this module owns the
    prompt + parsing. Backend resolution and model selection live in
    the runtime (per CLAUDE.md internal-LLM-call pattern). This module
    NEVER passes `model=""` because it never makes the call directly.
  - Graceful degradation. Any failure — network, parse error, schema
    violation, empty response — returns an empty list. Losing a turn's
    labels is preferable to crashing the response path.
  - Robust JSON extraction. Models often wrap JSON in markdown fences
    or commentary; we extract the first valid `{...}` block.
"""
from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable

from augmentum.personality.vocabulary import SEED_FACETS
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


LLMCall = Callable[[list[dict[str, str]]], Awaitable[str]]


def _vocabulary_listing() -> str:
    """Render the seed vocabulary as a categorized list for the prompt.

    Built from SEED_FACETS at import time — if vocabulary changes at
    runtime (via PersonalityStore.seed_vocabulary on user-added facets),
    this listing won't reflect that until restart. That's intentional:
    the labeler is calibrated against the canonical seed vocabulary;
    user-added facets need their own labeler-tuning before going live.
    """
    by_category: dict[str, list[str]] = {}
    for facet in SEED_FACETS:
        by_category.setdefault(facet.category.value, []).append(facet.name)
    lines = []
    for category in sorted(by_category):
        lines.append(f"- {category}: {', '.join(sorted(by_category[category]))}")
    return "\n".join(lines)


_VOCABULARY_LISTING = _vocabulary_listing()


def build_labeler_messages(
    response_text: str,
    recent_context: str,
    *,
    companion_name: str = "the companion",
    max_facets: int = 5,
) -> list[dict[str, str]]:
    """Build the system + user message list for the labeler LLM call.

    Pure function — no I/O. Used directly in tests; production runtime
    wraps this in a real backend call.
    """
    system = (
        f"You are labeling a response from {companion_name} with which "
        "personality facets were active. Personality facets are short "
        "named dispositions from a fixed vocabulary.\n"
        "\n"
        f"VOCABULARY (use ONLY these names, lowercase):\n{_VOCABULARY_LISTING}\n"
        "\n"
        f"INSTRUCTIONS: Identify 1 to {max_facets} facets most active in "
        "the response below. For each, give an intensity from 0.0 to 1.0 "
        "representing how strongly the facet shaped the response. Only "
        "use facets from the vocabulary above — do NOT invent names.\n"
        "\n"
        "OUTPUT STRICT JSON only, no commentary:\n"
        '{"facets": [{"name": "warm", "intensity": 0.7}, '
        '{"name": "patient", "intensity": 0.4}]}\n'
        "\n"
        "If no facets are clearly active (very short, neutral response), "
        'output: {"facets": []}'
    )
    user = (
        f"RECENT CONTEXT:\n{recent_context}\n"
        f"\n"
        f"RESPONSE TO LABEL:\n{response_text}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# Regex extracts the first balanced {...} block. Handles markdown fences,
# commentary, multi-line JSON. Not bulletproof against deeply-nested
# objects (none expected here — output schema is shallow).
_JSON_BLOCK = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)


def parse_labeler_response(text: str) -> list[tuple[str, float]]:
    """Parse the labeler model's JSON output. Returns list of
    (facet_name, intensity) tuples. Empty list on any failure.

    Intensity is clamped to [0.0, 1.0]. Non-numeric or out-of-range
    values are coerced; the labeler is allowed to be sloppy, the
    parser is not.

    Tries `json.loads` on the cleaned input first (handles nested
    objects correctly) and only falls back to the regex extractor
    when the input is wrapped in markdown fences or commentary.
    """
    if not text or not text.strip():
        return []

    payload = None
    # Direct parse first — robust to nested objects (the regex isn't).
    try:
        payload = json.loads(text.strip())
    except json.JSONDecodeError:
        # Fall back: extract the first {...} block from wrapped output.
        match = _JSON_BLOCK.search(text)
        if match is None:
            log.debug(
                "personality.labeler_parse_no_json",
                text_preview=text[:100],
            )
            return []
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            log.debug(
                "personality.labeler_parse_invalid_json",
                text_preview=match.group(0)[:100],
            )
            return []

    raw = payload.get("facets") if isinstance(payload, dict) else None
    if not isinstance(raw, list):
        log.debug("personality.labeler_parse_missing_facets", payload_type=type(payload).__name__)
        return []

    result: list[tuple[str, float]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name:
            continue
        intensity = item.get("intensity", 1.0)
        try:
            intensity_float = float(intensity)
        except (TypeError, ValueError):
            intensity_float = 1.0
        # Clamp to [0, 1]
        intensity_float = max(0.0, min(1.0, intensity_float))
        result.append((name.lower().strip(), intensity_float))
    return result


async def label_response(
    response_text: str,
    recent_context: str,
    *,
    llm_call: LLMCall,
    companion_name: str = "the companion",
    max_facets: int = 5,
) -> list[tuple[str, float]]:
    """High-level entry: build messages, call LLM, parse.

    Graceful degradation — returns empty list on any failure (network,
    parse, schema). Caller is expected to be the runtime, which owns
    backend + model resolution and passes `llm_call` as a closure.

    Empty response_text returns empty list without making the call.
    """
    if not response_text or not response_text.strip():
        return []
    messages = build_labeler_messages(
        response_text,
        recent_context,
        companion_name=companion_name,
        max_facets=max_facets,
    )
    try:
        raw = await llm_call(messages)
    except Exception:
        log.warning("personality.labeler_call_failed", exc_info=True)
        return []
    return parse_labeler_response(raw)
