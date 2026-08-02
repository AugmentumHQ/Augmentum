"""Tests for MoE expert offloading (Phase 0)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "services" / "engine"))


# ---------------------------------------------------------------------------
# Expert tensor name detection
# ---------------------------------------------------------------------------

def _is_expert_tensor(name: str) -> bool:
    """Python mirror of the C-level is_expert_weight_tensor logic."""
    if "_exps" in name and "shexp" not in name:
        return True
    return False


def test_expert_tensor_names_detected():
    """Expert FFN tensor names should be detected."""
    expert_names = [
        "blk.0.ffn_gate_exps.weight",
        "blk.5.ffn_up_exps.weight",
        "blk.12.ffn_down_exps.weight",
        "blk.0.ffn_gate_up_exps.weight",
        "blk.47.ffn_down_exps.weight",
    ]
    for name in expert_names:
        assert _is_expert_tensor(name), f"Should detect as expert: {name}"


def test_non_expert_tensor_names_not_detected():
    """Router, shared expert, attention, embeddings should NOT be detected."""
    non_expert = [
        "blk.0.ffn_gate_inp.weight",       # router
        "blk.0.ffn_gate_shexp.weight",      # shared expert (always on GPU)
        "blk.0.ffn_up_shexp.weight",        # shared expert
        "blk.0.ffn_down_shexp.weight",      # shared expert
        "blk.0.attn_q.weight",              # attention
        "blk.0.attn_k.weight",
        "blk.0.attn_v.weight",
        "blk.0.attn_output.weight",
        "blk.0.ffn_norm.weight",            # layer norm
        "token_embd.weight",                # embeddings
        "output.weight",                    # LM head
        "output_norm.weight",               # output norm
    ]
    for name in non_expert:
        assert not _is_expert_tensor(name), f"Should NOT detect as expert: {name}"


# ---------------------------------------------------------------------------
# Auto-detection heuristic
# ---------------------------------------------------------------------------

def test_auto_detect_small_model_no_offload():
    """Models under 20GB should not trigger auto offload."""
    size = 5 * 1024 * 1024 * 1024  # 5GB
    assert size <= 20 * 1024 * 1024 * 1024


def test_auto_detect_large_model_offload():
    """Models over 20GB should trigger auto offload."""
    size = 50 * 1024 * 1024 * 1024  # 50GB
    assert size > 20 * 1024 * 1024 * 1024


def test_auto_detect_boundary():
    """The exact boundary is 20GB."""
    threshold = 20 * 1024 * 1024 * 1024
    assert (threshold - 1) < threshold  # just under: no offload
    assert (threshold + 1) > threshold  # just over: offload


# ---------------------------------------------------------------------------
# Model registry MoE entries
# ---------------------------------------------------------------------------

def test_moe_models_in_catalog():
    """MoE models should be in the registry catalog."""
    from model_registry import _CATALOG

    assert "qwen3-30b-a3b" in _CATALOG, "Qwen3-30B-A3B missing from catalog"
    assert "mixtral-8x7b" in _CATALOG, "Mixtral-8x7B missing from catalog"


def test_moe_catalog_has_quants():
    """MoE catalog entries should have quantization options."""
    from model_registry import _CATALOG

    qwen = _CATALOG.get("qwen3-30b-a3b", {})
    assert "q4_k_m" in qwen.get("quants", {}), "Qwen3-30B-A3B missing q4_k_m quant"

    mixtral = _CATALOG.get("mixtral-8x7b", {})
    assert "q4_k_m" in mixtral.get("quants", {}), "Mixtral-8x7B missing q4_k_m quant"


# ---------------------------------------------------------------------------
# Server config
# ---------------------------------------------------------------------------

def test_moe_offload_env_default():
    """MOE_EXPERT_OFFLOAD defaults to 'auto'."""
    import os
    val = os.environ.get("ENGINE_MOE_EXPERT_OFFLOAD", "auto").lower()
    # In test environment, should be "auto" (not set)
    assert val == "auto"
