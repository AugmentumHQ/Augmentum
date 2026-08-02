"""Reliability guard for the Apr–Jun 2026 open-weight flagships.

Augmentum serves arbitrary GGUF models via the bundled llama-server, so a new
family is "supported" only if (a) its reasoning markers are parsed (no
chain-of-thought leaking into content / TTS) and (b) it gets the right sampling.
This pins both for the models a self-hoster is most likely to pull in mid-2026:
GLM-5.x, MiniMax-M3, DeepSeek-V4, Qwen3.6.

Added 2026-06-25 after a cross-model readiness audit. The headline regression
this prevents: MiniMax-M3 silently inheriting M2's temp=0.6 (its card wants 1.0).
"""
from __future__ import annotations

import pytest

from augmentum.models.sampling_profiles import recommended_for
from augmentum.utils.thinking import (
    _STARTS_THINKING_FAMILIES,
    _resolve_active_parsers,
    detect_reasoning_family,
)

# (label, model_name, arch, expected_family)
_FAMILY_CASES = [
    ("glm52-name",  "zai-org/GLM-5.2",            None,            "glm52"),
    ("glm52-arch",  None,                          "glm52",        "glm52"),
    ("glm5-name",   "GLM-5.1-Air",                 None,            "glm5"),
    ("m3-name",     "MiniMaxAI/MiniMax-M3",        None,            "minimaxm3"),
    ("m3-arch",     None,                          "minimax_m3_vl", "minimax_m3_vl"),
    ("dsv4-name",   "deepseek-ai/DeepSeek-V4-Flash", None,         "deepseek4"),
    ("dsv4-arch",   None,                          "deepseek4",    "deepseek4"),
    ("qwen36-name", "Qwen/Qwen3.6-35B-A3B",        None,            "qwen35"),
]


@pytest.mark.parametrize("label,model,arch,expected", _FAMILY_CASES)
def test_family_resolves(label, model, arch, expected):
    assert detect_reasoning_family(model, arch) == expected, label


@pytest.mark.parametrize("label,model,arch,expected", _FAMILY_CASES)
def test_new_flagships_are_asymmetric_closers(label, model, arch, expected):
    """All four 2026 flagships put the opening <think> in the prompt prefix, so
    the parser must start INSIDE a think block — else CoT leaks as content."""
    assert expected in _STARTS_THINKING_FAMILIES, label


@pytest.mark.parametrize("label,model,arch,expected", _FAMILY_CASES)
def test_think_parser_selected(label, model, arch, expected):
    assert "think" in _resolve_active_parsers(expected), label


def test_minimax_m3_does_not_inherit_m2_sampling():
    """The regression that motivated this file: M3's card wants temp 1.0 / top_k
    40; the M2.x family default is temp 0.6. M3 must get its own profile."""
    m3 = recommended_for("MiniMax-M3")
    m2 = recommended_for("MiniMax-M2")
    assert m3.temperature == 1.0 and m3.top_k == 40
    assert m2.temperature == 0.6           # family default, unchanged
    assert m3.temperature != m2.temperature


def test_glm5_sampling_has_min_p():
    s = recommended_for("GLM-5.2")
    assert s.temperature == 1.0 and s.top_p == 0.95 and s.min_p == 0.01


def test_qwen36_variants_distinguished():
    moe = recommended_for("Qwen3.6-35B-A3B")
    dense = recommended_for("Qwen3.6-27B")
    assert moe.temperature == 1.0 and moe.top_k == 20
    assert dense.temperature == 1.0 and dense.top_k == 20
    # MoE vs dense diverge on presence_penalty per the cards
    assert moe.presence_penalty != dense.presence_penalty
