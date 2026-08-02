"""Class-wide thinking-control policy: detect capability, then apply the setting.

Every provider adapter differs in WIRE FORMAT (Claude thinking block with
``budget_tokens`` / Gemini ``thinkingConfig`` / OpenAI-compat ``think`` +
``enable_thinking``), but the DECISION is identical and belongs in one place:

  1. Does this model support thinking control at all? (detection)
  2. Given the user's setting, should thinking be ON for this request?
  3. If it should be OFF, must we send an EXPLICIT disable — i.e. is this a
     provider whose DEFAULT is thinking-ON (Gemini 2.5+/3.x), where merely
     omitting a directive leaves the model reasoning anyway?

Point 3 is the subtle one that bit the coder's goal-judge on Gemini: the judge
set ``think=False`` but the adapter only *added* a thinking directive when think
was truthy, so Gemini kept thinking, exhausted the tiny token budget in the
thought channel, and returned empty content.

Adapters call :func:`resolve_thinking` and translate the returned
:class:`ThinkingDecision` into their own wire format. Detection lives here so a
new model family is added ONCE, not re-derived per connector.

Gemini is wired through this today; Claude and the OpenAI-compat families are
already correct on the off-path (Claude is off-by-default; openai_compat sends
explicit fields) and can adopt this incrementally.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Family id -> matcher. Order matters: more specific ids first. Kept lenient on
# VERSION so new point-releases and *-lite / *-latest aliases are covered
# without a per-name edit (the whole reason the strict per-family Gemini
# regexes missed gemini-3.1-flash-lite).
_FAMILY_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("gemini_3", re.compile(r"gemini-?3", re.IGNORECASE)),
    ("gemini_25", re.compile(r"gemini-?2\.5", re.IGNORECASE)),
    ("deepseek_r1", re.compile(r"deepseek.*(?:r1|reasoner)", re.IGNORECASE)),
    ("deepseek_hybrid", re.compile(r"deepseek.*(?:v3\.2|v4)", re.IGNORECASE)),
    ("claude", re.compile(r"claude", re.IGNORECASE)),
    ("glm", re.compile(r"glm-?4", re.IGNORECASE)),
    ("qwen3", re.compile(r"qwen-?3", re.IGNORECASE)),
]

# Families whose provider DEFAULT is thinking-ON: an off setting must be sent as
# an EXPLICIT disable, because omitting the directive leaves the model reasoning.
_DEFAULT_ON: frozenset[str] = frozenset({"gemini_25", "gemini_3", "deepseek_r1"})

# Families that ALWAYS reason and cannot be turned off — the setting is ignored
# for enablement (e.g. DeepSeek-R1 and its distills are reasoning-locked).
_ALWAYS_ON: frozenset[str] = frozenset({"deepseek_r1"})

_VALID_EFFORTS: frozenset[str] = frozenset({"min", "low", "medium", "high", "max"})


@dataclass(frozen=True, slots=True)
class ThinkingDecision:
    """Provider-agnostic thinking decision. Adapters translate to wire format."""

    capable: bool          # model supports thinking control
    enabled: bool          # thinking should be ON for this request
    effort: str            # normalized effort when enabled ("min".."max")
    must_disable: bool     # capable + not enabled + provider default is ON
    family: str            # detected family id ("" when unknown)


def detect_thinking_family(model: str) -> str:
    """Return the thinking family id for ``model``, or "" if none matches."""
    m = model or ""
    for family, pattern in _FAMILY_PATTERNS:
        if pattern.search(m):
            return family
    return ""


def is_thinking_capable(model: str) -> bool:
    """Whether ``model`` supports thinking control (any known family)."""
    return bool(detect_thinking_family(model))


def resolve_thinking(
    model: str,
    *,
    think: bool,
    effort: str = "medium",
    template_thinking: bool | None = None,
) -> ThinkingDecision:
    """Resolve the thinking decision for a request.

    ``think`` is the user's setting; ``effort`` the desired depth.

    ``template_thinking`` is GROUND TRUTH from a local model's GGUF chat_template
    (does it reference the thinking kwarg?), when the caller can read it. It
    overrides name/arch detection — the only reliable signal for custom SFT
    models whose name is meaningless and whose template may diverge from the base
    architecture's defaults. Leave it None for cloud models (no file) or when the
    template can't be inspected, and detection falls back to name/family.

    The result tells the adapter whether the model is capable, whether thinking
    should be on, and — crucially — whether an explicit disable must be sent when
    off (providers whose default is thinking-ON).
    """
    family = detect_thinking_family(model)
    # Ground truth (local chat_template) wins over name/arch when provided.
    capable = bool(template_thinking) if template_thinking is not None else bool(family)
    always_on = family in _ALWAYS_ON
    enabled = capable and (bool(think) or always_on)
    must_disable = capable and not enabled and family in _DEFAULT_ON
    eff = (effort or "medium").lower()
    if eff not in _VALID_EFFORTS:
        eff = "medium"
    return ThinkingDecision(
        capable=capable,
        enabled=enabled,
        effort=eff,
        must_disable=must_disable,
        family=family,
    )
