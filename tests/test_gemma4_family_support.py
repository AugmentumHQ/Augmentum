"""Tests pinning Gemma 4 family support across the inference stack.

Gemma 4 ships in four variants (April 2026):
- ``gemma-4-E2B-it`` (Edge, 5B total)
- ``gemma-4-E4B-it`` (Edge, 8B total)
- ``gemma-4-26B-A4B-it`` (sparse MoE, ~4B active)
- ``gemma-4-31B-it`` (dense)

These tests pin the cross-cutting support each variant needs:
  1. Catalog entry exists so the curated installer surfaces them.
  2. ``_model_family_key`` reduces every variant to ``"gemma4"``.
  3. The thinking parser dispatcher routes the family to
     ``(gemma4_channel, think)`` — both parsers run, model can emit
     either format.
  4. The MoE 26B variant doesn't need per-model expert wiring;
     ``llama_server_manager.py`` auto-handles via the GGUF profile.

If a future Gemma 5 ships with the same channel format, mirror this
file for that family — the parser, family detection, and catalog are
all keyed on the family name and stay decoupled from variant counts.
"""

from __future__ import annotations

import pytest

# ----------------------------------------------------------------------
# Catalog coverage
# ----------------------------------------------------------------------

GEMMA4_VARIANTS = (
    "gemma-4-e2b-it",
    "gemma-4-e4b-it",
    "gemma-4-26b-a4b-it",
    "gemma-4-31b-it",
)


@pytest.mark.parametrize("variant", GEMMA4_VARIANTS)
def test_catalog_has_entry(variant):
    """Every Gemma 4 variant must have a curated catalog entry so the
    Models modal's installer can surface it without the user pasting
    a custom HF repo URL."""
    from augmentum.models.model_catalog import _CATALOG as MODEL_CATALOG
    assert variant in MODEL_CATALOG, (
        f"{variant} missing from model_catalog.MODEL_CATALOG; users "
        f"can't one-click install it via the curated picker."
    )


@pytest.mark.parametrize("variant", GEMMA4_VARIANTS)
def test_catalog_default_quant_resolves(variant):
    """The default_quant key on each entry must point at a real
    ``quants`` map entry — broken indirection lets a 'Download' click
    silently no-op."""
    from augmentum.models.model_catalog import _CATALOG as MODEL_CATALOG
    entry = MODEL_CATALOG[variant]
    default = entry["default_quant"]
    assert default in entry["quants"], (
        f"{variant}.default_quant={default!r} not in quants keys "
        f"{sorted(entry['quants'])}; UI 'Download Recommended' would "
        f"fail silently."
    )


def test_26b_is_moe_variant():
    """26B-A4B uses Unsloth Dynamic Q4 builds only — make sure the
    catalog's q4_k_m alias maps to the UD file so users picking the
    'Q4_K_M' quant in the picker get a real download, not a 404."""
    from augmentum.models.model_catalog import _CATALOG as MODEL_CATALOG
    entry = MODEL_CATALOG["gemma-4-26b-a4b-it"]
    assert "UD" in entry["quants"]["q4_k_m"], (
        "q4_k_m should alias to the UD-Q4_K_M file — Unsloth ships no "
        "plain Q4_K_M for this build."
    )


# ----------------------------------------------------------------------
# Family detection — every variant must reduce to ``gemma4``
# ----------------------------------------------------------------------

@pytest.mark.parametrize("raw_name, expected", [
    ("Gemma-4-31B-it", "gemma4"),
    ("Gemma-4-31B-It", "gemma4"),
    ("gemma-4-26B-A4B-it", "gemma4"),
    ("Gemma_4_26B_A4B_it", "gemma4"),
    ("gemma-4-E4B-it", "gemma4"),
    ("gemma-4-E2B-it", "gemma4"),
    # Variants with quant suffixes — name detection should ignore them.
    ("gemma-4-26B-A4B-it-UD-Q4_K_XL", "gemma4"),
    ("gemma-4-31B-it.Q8_0", "gemma4"),
])
def test_family_key_normalises_to_gemma4(raw_name, expected):
    """Every Gemma 4 variant must collapse to a single family key so
    downstream code (parser routing, capability flags, fabric peer
    descriptors) doesn't need per-variant branches."""
    from augmentum.models.llama_server_manager import _model_family_key
    assert _model_family_key(raw_name) == expected


def test_family_key_distinguishes_from_gemma3():
    """The family-reduction must not collapse Gemma 4 onto Gemma 3 or
    vice-versa — the two use DIFFERENT asymmetric/symmetric channel
    formats, so the parser dispatcher MUST see distinct keys."""
    from augmentum.models.llama_server_manager import _model_family_key
    assert _model_family_key("gemma3-12b-it") == "gemma3"
    assert _model_family_key("gemma-4-31B-it") == "gemma4"
    assert _model_family_key("gemma3") != _model_family_key("gemma-4")


# ----------------------------------------------------------------------
# Thinking parser routing
# ----------------------------------------------------------------------

def test_thinking_dispatcher_routes_gemma4_to_correct_parsers():
    """Gemma 4 uses asymmetric channel markers (``<|channel>thought\\n…
    <channel|>``) — NOT a slash-variant of Gemma 3 / GPT-OSS. The
    family-key MUST route to ``gemma4_channel`` (the asymmetric
    parser); ``think`` is also run as a fallback because some Gemma 4
    finetunes emit standard <think> blocks."""
    from augmentum.utils.thinking import _FAMILY_PARSERS
    assert "gemma4" in _FAMILY_PARSERS, (
        "Gemma 4 missing from thinking._FAMILY_PARSERS — extraction "
        "would silently no-op for every Gemma 4 turn."
    )
    parsers = _FAMILY_PARSERS["gemma4"]
    assert "gemma4_channel" in parsers, (
        "gemma4 must route to gemma4_channel (asymmetric); without it "
        "reasoning content leaks into the visible response."
    )


def test_gemma4_not_routed_to_gemma3_parsers():
    """The Gemma 3 / GPT-OSS family uses SYMMETRIC <|channel|>...<|end|>
    markers. Routing Gemma 4 through the Gemma 3 parser would mis-
    delimit its asymmetric stream entirely. Pin the distinction."""
    from augmentum.utils.thinking import _FAMILY_PARSERS
    assert _FAMILY_PARSERS["gemma4"] != _FAMILY_PARSERS["gemma3"]
    assert "gemma_channel" not in _FAMILY_PARSERS["gemma4"], (
        "gemma4 must NOT use the symmetric gemma_channel parser — it'd "
        "misread the asymmetric stream and emit malformed output."
    )


# ----------------------------------------------------------------------
# Thinking-toggle wiring — enable_thinking flows end-to-end
# ----------------------------------------------------------------------

def test_template_thinking_override_forwards_for_gemma4():
    """Upstream Gemma 4 model card: ``enable_thinking`` abstracts the
    ``<|think|>`` system-prompt control token via the jinja template,
    so the UI's per-turn thinking flag must reach the chat-template
    kwarg. Without this wiring, the toggle would be UI-only — flipping
    it would not actually change model behaviour."""
    from augmentum.models.openai_compat import _template_thinking_override
    assert _template_thinking_override("gemma-4-31B-it", True) is True
    assert _template_thinking_override("gemma-4-31B-it", False) is False
    assert _template_thinking_override("gemma-4-26B-A4B-it", True) is True
    assert _template_thinking_override("Gemma-4-E4B-it", False) is False
    # Non-thinking families still return None so we don't add an
    # unrecognised top-level kwarg to their payloads.
    assert _template_thinking_override("llama-3.3-70b", True) is None
    assert _template_thinking_override("mistral-large", True) is None


def test_template_thinking_override_distinguishes_gemma3_from_gemma4():
    """Gemma 3 doesn't expose ``enable_thinking`` (its symmetric
    channel format is emitted unconditionally). Forwarding the kwarg
    to a Gemma 3 chat template would add an unknown kwarg that the
    template's strict-keys handling may reject."""
    from augmentum.models.openai_compat import _template_thinking_override
    assert _template_thinking_override("gemma-3-12b-it", True) is None
    assert _template_thinking_override("gemma3", False) is None
