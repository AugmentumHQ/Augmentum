"""Tests for the Q2-2026 new-model-family audit (May 31, 2026).

Pins the wiring added for trending open-weight releases:

- **Moonshot Kimi K2.6 / K2.6-Thinking** (Apr 2026). Symmetric
  ``<think>`` parser; toggle via ``thinking`` (bool) kwarg — NOT
  ``enable_thinking``. K2.6-Thinking variant locked thinking-on.
- **Xiaomi MiMo-V2.5 / V2.5-Pro** (Apr-May 2026). Symmetric
  ``<think>`` parser; standard ``enable_thinking`` kwarg. New GGUF
  arch ``mimo2``.
- **Mistral Ministral-3-Reasoning-2512** (3B/8B/14B, Dec 2025-Mar
  2026). Reuses Magistral ``[THINK]`` bracket-token convention via
  the ``ministral3`` family key.
- **NVIDIA Nemotron-Cascade 2** (Mar 2026). Symmetric ``<think>``
  with ChatML base. Toggle is a new empty-block prefix pattern;
  parser keyed under ``nemotron_cascade`` to distinguish from
  Nemotron-3 Nano's ``enable_thinking`` kwarg.

These tests don't verify wire behaviour end-to-end (that requires
a loaded GGUF) — they pin the wiring contracts so a future refactor
can't silently regress family detection / parser routing / template
override for these models.
"""

from __future__ import annotations

import pytest

# ----------------------------------------------------------------------
# Parser routing — _FAMILY_PARSERS
# ----------------------------------------------------------------------

@pytest.mark.parametrize("family_key, expected_parsers", [
    # Kimi K2.x — symmetric <think>
    ("kimi",    ("think",)),
    ("kimi2",   ("think",)),
    ("kimi_k2", ("think",)),
    # MiMo-V2.5 — symmetric <think>
    ("mimo",    ("think",)),
    ("mimo2",   ("think",)),
    # Nemotron-Cascade 2 — symmetric <think>, distinct from nemotron-nano
    ("nemotron_cascade",  ("think",)),
    ("nemotron_cascade2", ("think",)),
    # Ministral-3-Reasoning — reuses [THINK] bracket-tokens, family-key
    # is the bridge to _MAGISTRAL_FAMILIES (below).
    ("ministral3", ("think",)),
])
def test_family_parser_registered(family_key, expected_parsers):
    """Every new family must appear in ``_FAMILY_PARSERS`` so the
    streaming buffer routes to the right extractor instead of the
    unknown-family fallback that runs every parser (noisy + slow)."""
    from augmentum.utils.thinking import _FAMILY_PARSERS
    assert family_key in _FAMILY_PARSERS, (
        f"Family '{family_key}' missing from _FAMILY_PARSERS — "
        f"reasoning extraction would fall through to the all-parsers "
        f"fallback and emit noise."
    )
    assert _FAMILY_PARSERS[family_key] == expected_parsers


def test_ministral3_uses_magistral_brackets():
    """Ministral-3-Reasoning emits ``[THINK]`` / ``[/THINK]`` SPECIAL
    TOKENS, not literal ``<think>`` tags. The streaming buffer picks
    the right delimiters based on ``_MAGISTRAL_FAMILIES`` membership —
    Ministral-3 must be IN that set or extraction silently no-ops."""
    from augmentum.utils.thinking import _MAGISTRAL_FAMILIES
    assert "ministral3" in _MAGISTRAL_FAMILIES, (
        "Ministral-3-Reasoning would emit [THINK] but the buffer would "
        "look for <think> and miss every reasoning block."
    )


def test_kimi_and_mimo_use_xml_brackets_not_magistral():
    """Kimi K2.6 and MiMo-V2.5 use the SAME family-key as the standard
    <think> parser. Putting them in _MAGISTRAL_FAMILIES would make the
    buffer look for [THINK] in a <think>-emitting stream and lose
    every reasoning block."""
    from augmentum.utils.thinking import _MAGISTRAL_FAMILIES
    for key in ("kimi", "kimi2", "kimi_k2", "mimo", "mimo2",
                "nemotron_cascade", "nemotron_cascade2"):
        assert key not in _MAGISTRAL_FAMILIES, (
            f"Family '{key}' uses standard <think> tokens; including "
            f"it in _MAGISTRAL_FAMILIES would route extraction at the "
            f"wrong delimiters and silently lose all reasoning."
        )


# ----------------------------------------------------------------------
# Template thinking override — _template_thinking_override
# ----------------------------------------------------------------------

@pytest.mark.parametrize("model_name, think_value, expected", [
    # Kimi K2.6 — bare bool drives the toggle; the adapter rewrites
    # the kwarg name to "thinking" for the Kimi family at the
    # llama_cpp.py wire layer.
    ("Kimi-K2.6",            True,  True),
    ("Kimi-K2.6",            False, False),
    ("kimi-k2.6-thinking",   False, True),   # locked-on suffix wins
    ("Kimi-K2.6-Thinking",   True,  True),
    # MiMo-V2.5 — standard enable_thinking kwarg path.
    ("MiMo-V2.5",            True,  True),
    ("MiMo-V2.5-Pro",        False, False),
    ("XiaomiMiMo/MiMo-V2.5", True,  True),
])
def test_template_thinking_override_forwards_for_new_families(
    model_name, think_value, expected,
):
    """The UI toggle's per-turn ``request.think`` must reach the
    chat-template via this function for the new families. Returning
    None here would silently drop the toggle."""
    from augmentum.models.openai_compat import _template_thinking_override
    assert _template_thinking_override(model_name, think_value) is expected


def test_template_thinking_override_skips_unrelated_models():
    """Forwarding ``enable_thinking`` to a non-hybrid template would
    add an unknown kwarg that strict templates reject. Confirm the
    new family branches don't false-positive on close-by names."""
    from augmentum.models.openai_compat import _template_thinking_override
    # Llama 3.3 isn't a thinking model — no kwarg.
    assert _template_thinking_override("llama-3.3-70b", True) is None
    # Plain Mistral isn't reasoning either; Magistral / Ministral-3-
    # Reasoning use their own [THINK] path. ``mistral-7b`` should
    # NOT receive enable_thinking.
    assert _template_thinking_override("mistral-7b-instruct", True) is None
