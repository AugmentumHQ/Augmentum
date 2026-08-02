"""Mode-aware inference hints — optimize sampling and generation per mode.

Each mode has different needs. A creative writing session benefits from
high temperature and repeat penalty. An analytical pipeline needs
deterministic, concise output. A coding assistant needs low temperature.

These hints are applied as DEFAULTS — they never override the user's
explicit settings. If the user set temperature=0.5, we respect it.
"""
from __future__ import annotations

from augmentum.models.base import InternalChatRequest


# Mode → default inference parameters.
# Only applied when the user hasn't set the value explicitly.
#
# ``reasoning_effort`` is gated by the OpenAI-family check at adapter
# level — adding it here is harmless for non-OAI providers (they never
# see the field) but gives GPT-5.x / o-series / Grok the right effort
# tier per mode without forcing the user to configure it.
#   coder/agentic → "high"      (multi-step problem-solving)
#   narrative     → "low"       (creative writing wants flow, not CoT)
#   passthrough   → "low"       (chat: snappy, not deliberative)
#   analytical    → "low"       (per-phase, short, fast pipeline)
#
# NOTE: ``minimal`` was the original default for narrative/passthrough,
# but the codex-proxy bridge (and some other OpenAI-compat re-routers)
# accept only ``low | medium | high | xhigh`` — ``minimal`` 400s
# with ``Invalid enum value``. ``low`` is the safe universal default.
# Users who want the truly minimal tier on OpenAI-native targets can
# still pick it explicitly from the composer dropdown; the adapter
# layer demotes ``minimal`` → ``low`` automatically for profiles that
# don't support it (see ``supports_reasoning_effort_minimal``).
_MODE_HINTS: dict[str, dict] = {
    "narrative": {
        "temperature": 0.85,
        "top_p": 0.95,
        "max_tokens": 4096,
        "reasoning_effort": "low",
        # raw_options applied to local backends (Ollama, llama.cpp)
        "_raw": {
            "repeat_penalty": 1.15,
            "dry_multiplier": 0.8,
        },
    },
    "analytical": {
        "temperature": 0.3,
        "top_p": 0.9,
        "max_tokens": 512,  # per UARF phase — keeps pipeline fast
        "reasoning_effort": "low",
        "_raw": {
            "repeat_penalty": 1.0,
        },
    },
    "agentic": {
        "temperature": 0.4,
        "top_p": 0.9,
        "max_tokens": 1024,  # per plan step
        "reasoning_effort": "high",
        "_raw": {},
    },
    "coder": {
        "temperature": 0.15,
        "top_p": 0.95,
        "max_tokens": 8192,
        "reasoning_effort": "high",
        "_raw": {},
    },
    "passthrough": {
        # Don't override anything — user's settings are king
        # 32768, not 8192 (2026-07-08): same class as the coder-side
        # 2026-05-31 fix — the flat 8192 "runaway cap" was routinely hit
        # by reasoning models on big single-file deliveries (a 46KB
        # inline HTML game is ~13K tokens BEFORE the thinking budget),
        # producing 0-177 char truncated answers. 32K matches
        # coder_local_max_tokens_cap's "comfortably above any
        # single-file write" rationale; still caps true runaways.
        "max_tokens": 32768,
        "reasoning_effort": "low",
        "_raw": {},
    },
}


def apply_mode_hints(request: InternalChatRequest, mode: str) -> None:
    """Apply mode-aware default inference parameters to a request.

    Only sets values that the user hasn't explicitly provided.
    Modifies the request in-place.
    """
    hints = _MODE_HINTS.get(mode)
    if not hints:
        return

    # Temperature — only if user didn't set it
    if request.temperature is None and "temperature" in hints:
        request.temperature = hints["temperature"]

    # Top-p — only if user didn't set it
    if request.top_p is None and "top_p" in hints:
        request.top_p = hints["top_p"]

    # Max tokens — only if user didn't set it AND mode has a cap
    if request.max_tokens is None and "max_tokens" in hints:
        request.max_tokens = hints["max_tokens"]

    # Reasoning effort — only if user didn't set it AND mode has a default.
    # Field is on InternalChatRequest; openai_compat decides whether to
    # actually transmit it based on the provider's OpenAI-family flag,
    # so unconditional setting here is safe for non-OAI providers.
    if (
        getattr(request, "reasoning_effort", None) is None
        and "reasoning_effort" in hints
    ):
        request.reasoning_effort = hints["reasoning_effort"]

    # Raw options (local backend-specific params)
    raw_hints = hints.get("_raw", {})
    if raw_hints:
        if request.raw_options is None:
            request.raw_options = {}
        for key, value in raw_hints.items():
            if key not in request.raw_options:
                request.raw_options[key] = value
