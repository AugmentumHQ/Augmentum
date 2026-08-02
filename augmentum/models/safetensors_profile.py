"""Capability profile for a downloaded safetensors model repo.

The vLLM parity backbone (spec 2026-07-22 §F1). llama.cpp models get their
profile from GGUF headers (``model_profile_cache``) or the engine ``/props``
endpoint; a safetensors model served by vLLM has neither, so we derive the same
facts from the repo's ``config.json`` (+ ``tokenizer_config.json`` for the chat
template, ``generation_config.json``). Every downstream surface — library tags,
reasoning-family detection, launch-param defaults, vision gating — reads this
instead of reaching for GGUF metadata when the model is vLLM-served.

Pure filesystem read, no model load. Returns a plain dict (JSON-friendly for the
model-manager UI). Missing/garbled config yields a minimal profile, never raises.
"""

from __future__ import annotations

import json
import os
from typing import Any

from augmentum.utils.logging import get_logger
from augmentum.utils.thinking import detect_reasoning_family

log = get_logger(__name__)

# torch_dtype string -> bytes per parameter, for a param-count estimate from
# on-disk weight bytes.
_DTYPE_BYTES = {
    "float32": 4, "float": 4,
    "bfloat16": 2, "float16": 2, "half": 2,
    "float8_e4m3fn": 1, "float8_e5m2": 1,
    "int8": 1, "uint8": 1,
}


def _read_json(path: str) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _int(v: Any) -> int:
    try:
        # Some hybrid-arch configs publish list-typed values (per-layer).
        if isinstance(v, list):
            v = v[0] if v else 0
        return int(v)
    except (TypeError, ValueError):
        return 0


def _is_vision(config: dict, arch: str, model_type: str) -> bool:
    """Heuristic vision/VLM detection from config shape + naming."""
    if config.get("vision_config") or config.get("vision_tower") or config.get("mm_projector_type"):
        return True
    blob = f"{arch} {model_type}".lower()
    return any(tag in blob for tag in ("vl", "vision", "-vl", "vlm", "image"))


def _has_tool_template(tokenizer_config: dict) -> bool:
    """Best-effort: does the chat template reference tools/function-calling?"""
    tmpl = tokenizer_config.get("chat_template")
    if isinstance(tmpl, list):  # some models ship named templates
        tmpl = " ".join(str(t.get("template", "")) for t in tmpl if isinstance(t, dict))
    if not isinstance(tmpl, str):
        return False
    low = tmpl.lower()
    return "tool" in low or "function" in low


def safetensors_profile(repo_dir: str) -> dict[str, Any]:
    """Build a capability profile dict for a safetensors repo directory."""
    config = _read_json(os.path.join(repo_dir, "config.json"))
    tok_config = _read_json(os.path.join(repo_dir, "tokenizer_config.json"))

    name = os.path.basename(repo_dir.rstrip("/"))
    archs = config.get("architectures") or []
    architecture = str(archs[0]) if archs else ""
    model_type = str(config.get("model_type") or "")

    # Multimodal/VLM configs nest the language-model params under ``text_config``
    # (Gemma-4 unified, Qwen-VL, Llava, …). Read LM facts from there when present,
    # while vision detection still uses the top-level config.
    lm = config.get("text_config") if isinstance(config.get("text_config"), dict) else config

    # MoE detection (Qwen3-MoE etc. — num_local_experts / num_experts).
    expert_count = _int(lm.get("num_local_experts") or lm.get("num_experts")
                        or config.get("num_local_experts") or config.get("num_experts"))
    expert_used = _int(lm.get("num_experts_per_tok") or config.get("num_experts_per_tok"))

    # Param-count estimate from on-disk weight bytes / dtype width.
    size_bytes = 0
    try:
        for n in os.listdir(repo_dir):
            if n.lower().endswith(".safetensors"):
                size_bytes += os.path.getsize(os.path.join(repo_dir, n))
    except OSError:
        pass
    dtype = str(config.get("torch_dtype") or config.get("dtype") or "bfloat16").lower()
    per_param = _DTYPE_BYTES.get(dtype, 2)
    params_est = int(size_bytes / per_param) if size_bytes and per_param else 0

    # Reasoning family: prefer model_type as the arch hint, fall back to name.
    reasoning_family = detect_reasoning_family(name, arch=model_type or architecture) or ""

    vision = _is_vision(config, architecture, model_type)
    # auto_map present => custom modeling code => needs trust_remote_code (like
    # nanbeige). Surfaced so launch params/UX can flag it (security opt-in).
    needs_remote_code = bool(config.get("auto_map"))

    return {
        "model_name": name,
        "architecture": architecture,
        "model_type": model_type,
        "context_length": _int(lm.get("max_position_embeddings") or config.get("max_position_embeddings")),
        "n_layers": _int(lm.get("num_hidden_layers")),
        "n_heads": _int(lm.get("num_attention_heads")),
        "n_heads_kv": _int(lm.get("num_key_value_heads")),
        "n_embed": _int(lm.get("hidden_size")),
        "n_vocab": _int(lm.get("vocab_size") or config.get("vocab_size")),
        "is_moe": expert_count > 0,
        "expert_count": expert_count,
        "expert_used_count": expert_used,
        "dtype": dtype,
        "size_bytes": size_bytes,
        "params_est": params_est,
        "reasoning_family": reasoning_family,
        "vision": vision,
        "tools": _has_tool_template(tok_config),
        "needs_remote_code": needs_remote_code,
        "backend": "vllm",
    }
