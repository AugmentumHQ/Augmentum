"""Capability gate for the bug-finder pipeline.

The detector / verifier / fixer prompts in ``prompts.py`` and
``verifier.py`` are written for capable instruction-following models —
they require disciplined JSON-block emission, multi-step disproof
reasoning, and tool-call accuracy. Smaller / older models produce
malformed output the parsers reject silently, leading to "zero
findings" runs that look successful but were structurally broken.

This module gates run creation on a known-good floor and surfaces a
structured refusal the UI and callable surfaces can act on. Users can
opt in to running below the floor (e.g. for benchmarking adaptation
work) via ``force_below_minimum=True`` — the orchestrator records a
note on the report so downstream consumers can interpret zero-finding
runs in context.

The floor list reflects the family + version landscape captured in
``project_q2_2026_model_family_audit`` memory plus subsequent
adjustments. Adding new families: extend ``_CAPABLE_PATTERNS`` and
keep entries lowercased.
"""

from __future__ import annotations

import re

# Substring patterns matched against the lowercased model id. A model
# is capable when ANY pattern matches.
_CAPABLE_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Anthropic Claude 4.x family — opus / sonnet / haiku, any minor.
    re.compile(r"claude-(opus|sonnet|haiku)-4(\b|[-.])"),
    # OpenAI GPT-5.x + o1 / o3 / codex reasoning lineup.
    re.compile(r"^gpt-5(\b|[-.])"),
    re.compile(r"^o[13](\b|[-.])"),
    re.compile(r"^codex(-|\b)"),
    # Qwen 3.5+ (qwen3.5 / qwen3.6 / qwen3-coder / qwen3-omni). Older
    # qwen2.x and qwen1.x reject; qwen3.0 and qwen3.1 reject (prompts
    # target the 3.5+ JSON discipline).
    re.compile(r"qwen3[._-]?(5|6|7|8|9|coder|omni)"),
    re.compile(r"qwen3\.5"),
    # DeepSeek V3.x and R1.x reasoning lineup.
    re.compile(r"deepseek[-_.]?(v3|r1|v4)"),
    # Kimi K2.x+.
    re.compile(r"kimi[-_.]?k2"),
    # GLM-4.x family — Z.ai's reasoning-strong releases.
    re.compile(r"glm[-_.]?4"),
    # MiniMax M2.x.
    re.compile(r"minimax[-_.]?m2"),
    # Mistral Magistral (the reasoning variant — NOT vanilla Mistral 7B).
    re.compile(r"magistral"),
    # EXAONE 4.x.
    re.compile(r"exaone[-_.]?4"),
    # Gemma 3 / Gemma 4 (channel-token reasoning).
    re.compile(r"gemma[-_.]?[34]"),
    # GPT-OSS (channel-token reasoning).
    re.compile(r"gpt[-_.]?oss"),
    # Hunyuan Hy3 reasoning.
    re.compile(r"hunyuan[-_.]?hy3"),
    # Nemotron Nano 3 / Cascade 2 reasoning.
    re.compile(r"nemotron[-_.]?(nano[-_.]?3|cascade[-_.]?2)"),
    # MiMo V2.5+.
    re.compile(r"mimo[-_.]?v?2[._-]?5"),
)


# Sentinel returned to surfaces so they can render a stable error code
# rather than match the message string.
PRIMARY_MODEL_BELOW_MINIMUM = "primary_model_below_minimum"


def is_capable(model_id: str) -> bool:
    """Return True when the model meets the prompt-compatibility floor.

    Matching is permissive on prefixes (provider tags like
    ``model@provider`` or ``model@fabric:peer`` are stripped) and on
    case. Custom local model names that don't match any known family
    are treated as below the floor — users who know their model is
    capable can pass ``force_below_minimum=True``.
    """
    if not model_id:
        return False
    bare = model_id.split("@", 1)[0].strip().lower()
    if not bare:
        return False
    for pat in _CAPABLE_PATTERNS:
        if pat.search(bare):
            return True
    return False


def capability_floor_label() -> str:
    """One-line human description of the floor for UI / error messages."""
    return (
        "Claude 4.x, GPT-5.x, o1/o3/codex, Qwen 3.5+, DeepSeek V3.x/R1.x, "
        "Kimi K2.x, GLM-4.x, MiniMax M2.x, Magistral, EXAONE 4.x, Gemma 3/4, "
        "GPT-OSS, Hunyuan Hy3, Nemotron Nano 3 / Cascade 2, MiMo V2.5+"
    )


def below_floor_note(model_id: str) -> str:
    """Note appended to the run report when ``force_below_minimum=True``.

    Captures the override for downstream interpretation — the model
    isn't on the known-good list, so a zero-findings result may reflect
    prompt incompatibility rather than a clean codebase.
    """
    return (
        f"primary_model '{model_id}' is below the bug-finder capability "
        "floor; run was started with force_below_minimum=True. Detector / "
        "verifier / fixer prompts target capable instruction-followers "
        f"({capability_floor_label()}); smaller models may produce "
        "malformed output the parsers silently reject. Interpret zero-"
        "finding outcomes with care."
    )
