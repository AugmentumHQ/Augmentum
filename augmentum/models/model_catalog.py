"""Model catalog - maps friendly names to Hugging Face GGUF downloads.

Enables Ollama-style pull syntax such as "qwen3.6-27b" or
"qwen3.6-27b:ud_q4_k_xl" and resolves it to a Hugging Face repo + filename.
"""
from __future__ import annotations

# Format:
#   "name" -> {
#       "repo": "org/repo",
#       "quants": {"q4_k_m": "file.gguf", ...},
#       "default_quant": "q4_k_m",
#   }
#
# The first entries are intentionally ordered as the curated recommendations
# shown in the Model Manager UI. These curated picks are kept Unsloth-first
# so the recommendations feel consistent in quant quality and naming. Older
# aliases remain available below for backwards compatibility and manual pulls.
_CATALOG: dict[str, dict] = {
    "qwen3.6-27b": {
        "repo": "unsloth/Qwen3.6-27B-GGUF",
        "quants": {
            "q4_k_m": "Qwen3.6-27B-Q4_K_M.gguf",
            "ud_q4_k_xl": "Qwen3.6-27B-UD-Q4_K_XL.gguf",
            "q8_0": "Qwen3.6-27B-Q8_0.gguf",
        },
        "default_quant": "ud_q4_k_xl",
    },
    "qwen3.6-35b-a3b": {
        "repo": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "quants": {
            "q4_k_m": "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            "ud_q4_k_m": "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
            "ud_q4_k_xl": "Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf",
        },
        "default_quant": "ud_q4_k_xl",
    },
    "gemma-4-e4b-it": {
        "repo": "unsloth/gemma-4-E4B-it-GGUF",
        "quants": {
            "q4_k_m": "gemma-4-E4B-it-Q4_K_M.gguf",
            "ud_q4_k_xl": "gemma-4-E4B-it-UD-Q4_K_XL.gguf",
            "q8_0": "gemma-4-E4B-it-Q8_0.gguf",
        },
        "default_quant": "ud_q4_k_xl",
    },
    "magistral-small-2507": {
        "repo": "unsloth/Magistral-Small-2507-GGUF",
        "quants": {
            "q4_k_m": "Magistral-Small-2507-Q4_K_M.gguf",
            "ud_q4_k_xl": "Magistral-Small-2507-UD-Q4_K_XL.gguf",
            "q8_0": "Magistral-Small-2507-Q8_0.gguf",
        },
        "default_quant": "ud_q4_k_xl",
    },
    "phi-4": {
        "repo": "unsloth/phi-4-GGUF",
        "quants": {
            "q4_k_m": "phi-4-Q4_K_M.gguf",
            "q8_0": "phi-4-Q8_0.gguf",
        },
        "default_quant": "q4_k_m",
    },
    "glm-4.7-flash": {
        "repo": "unsloth/GLM-4.7-Flash-GGUF",
        "quants": {
            "q4_k_m": "GLM-4.7-Flash-Q4_K_M.gguf",
            "ud_q4_k_xl": "GLM-4.7-Flash-UD-Q4_K_XL.gguf",
            "q8_0": "GLM-4.7-Flash-Q8_0.gguf",
        },
        "default_quant": "ud_q4_k_xl",
    },
    "gemma-4-31b-it": {
        "repo": "unsloth/gemma-4-31B-it-GGUF",
        "quants": {
            "q4_k_m": "gemma-4-31B-it-Q4_K_M.gguf",
            "ud_q4_k_xl": "gemma-4-31B-it-UD-Q4_K_XL.gguf",
            "q8_0": "gemma-4-31B-it-Q8_0.gguf",
        },
        "default_quant": "ud_q4_k_xl",
    },
    # Gemma 4 26B sparse-MoE (A4B = ~4B active params). Sits between
    # E4B (8B) and 31B dense; better quality/VRAM ratio than 31B on
    # consumer GPUs thanks to the MoE expert-offload path
    # (--n-cpu-moe is auto-tuned by llama_server_manager based on the
    # GGUF profile, no per-model wiring needed). Unsloth only ships
    # UD (Unsloth Dynamic) Q4 variants for this build — no plain
    # Q4_K_M — so we map ``q4_k_m`` to the UD file too for the curated
    # picker; users wanting non-UD quant pick a different model.
    "gemma-4-26b-a4b-it": {
        "repo": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "quants": {
            "q4_k_m": "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
            "ud_q4_k_m": "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
            "ud_q4_k_xl": "gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf",
            "q8_0": "gemma-4-26B-A4B-it-Q8_0.gguf",
        },
        "default_quant": "ud_q4_k_xl",
    },
    "phi-4-mini": {
        "repo": "unsloth/Phi-4-mini-instruct-GGUF",
        "quants": {
            "q4_k_m": "Phi-4-mini-instruct-Q4_K_M.gguf",
            "q8_0": "Phi-4-mini-instruct-Q8_0.gguf",
        },
        "default_quant": "q4_k_m",
    },
    "gemma-4-e2b-it": {
        "repo": "unsloth/gemma-4-E2B-it-GGUF",
        "quants": {
            "q4_k_m": "gemma-4-E2B-it-Q4_K_M.gguf",
            "ud_q4_k_xl": "gemma-4-E2B-it-UD-Q4_K_XL.gguf",
            "q8_0": "gemma-4-E2B-it-Q8_0.gguf",
        },
        "default_quant": "ud_q4_k_xl",
    },
    "qwen2.5-coder-14b": {
        "repo": "unsloth/Qwen2.5-Coder-14B-Instruct-128K-GGUF",
        "quants": {
            "q4_k_m": "Qwen2.5-Coder-14B-Instruct-Q4_K_M.gguf",
            "q8_0": "Qwen2.5-Coder-14B-Instruct-Q8_0.gguf",
        },
        "default_quant": "q4_k_m",
    },
    "deepseek-r1-14b": {
        "repo": "unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF",
        "quants": {
            "q4_k_m": "DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf",
            "q8_0": "DeepSeek-R1-Distill-Qwen-14B-Q8_0.gguf",
        },
        "default_quant": "q4_k_m",
    },
    "llama-3.3-70b": {
        "repo": "unsloth/Llama-3.3-70B-Instruct-GGUF",
        "quants": {
            "q4_k_m": "Llama-3.3-70B-Instruct-Q4_K_M.gguf",
            "ud_q4_k_xl": "Llama-3.3-70B-Instruct-UD-Q4_K_XL.gguf",
            "q8_0": "Llama-3.3-70B-Instruct-Q8_0.gguf",
        },
        "default_quant": "ud_q4_k_xl",
    },
    "qwen3.5-7b": {
        "repo": "bartowski/Qwen_Qwen3.5-7B-GGUF",
        "quants": {
            "q4_k_m": "Qwen3.5-7B-Q4_K_M.gguf",
            "q8_0": "Qwen3.5-7B-Q8_0.gguf",
        },
        "default_quant": "q4_k_m",
    },
    "qwen3.5-4b": {
        "repo": "bartowski/Qwen_Qwen3.5-4B-GGUF",
        "quants": {
            "q4_k_m": "Qwen3.5-4B-Q4_K_M.gguf",
            "q8_0": "Qwen3.5-4B-Q8_0.gguf",
        },
        "default_quant": "q4_k_m",
    },
    "gemma-3-12b": {
        "repo": "bartowski/google_gemma-3-12b-it-GGUF",
        "quants": {
            "q4_k_m": "gemma-3-12b-it-Q4_K_M.gguf",
            "q8_0": "gemma-3-12b-it-Q8_0.gguf",
        },
        "default_quant": "q4_k_m",
    },
    "gemma-3-4b": {
        "repo": "bartowski/gemma-3-4b-it-GGUF",
        "quants": {
            "q4_k_m": "gemma-3-4b-it-Q4_K_M.gguf",
            "q8_0": "gemma-3-4b-it-Q8_0.gguf",
        },
        "default_quant": "q4_k_m",
    },
    "llama-4-scout": {
        "repo": "bartowski/meta-llama_Llama-4-Scout-17B-16E-Instruct-old-GGUF",
        "quants": {
            "q4_k_m": "meta-llama_Llama-4-Scout-17B-16E-Instruct-old-Q4_K_M.gguf",
        },
        "default_quant": "q4_k_m",
    },
    "cogito-14b": {
        "repo": "bartowski/deepcogito_cogito-v1-preview-qwen-14B-GGUF",
        "quants": {
            "q4_k_m": "deepcogito_cogito-v1-preview-qwen-14B-Q4_K_M.gguf",
        },
        "default_quant": "q4_k_m",
    },
}


def resolve_model_name(name: str) -> tuple[str, str] | None:
    """Resolve a friendly model name to (hf_repo, filename).

    Accepts:
      "qwen3.6-27b"         -> default quant
      "qwen3.6-27b:q8_0"    -> specific quant
      "org/repo:file.gguf"  -> direct Hugging Face reference

    Returns (repo, filename) or None if not in catalog.
    """
    if "/" in name and ":" in name:
        repo, filename = name.split(":", 1)
        return (repo, filename)

    if ":" in name:
        model_name, quant = name.split(":", 1)
    else:
        model_name = name
        quant = ""

    model_name = model_name.lower().strip()
    entry = _CATALOG.get(model_name)

    if not entry:
        for key, val in _CATALOG.items():
            if model_name in key or key in model_name:
                entry = val
                break

    if not entry:
        return None

    if not quant:
        quant = entry.get("default_quant", "q4_k_m")

    filename = entry["quants"].get(quant)
    if not filename:
        filename = next(iter(entry["quants"].values()), None)

    if not filename:
        return None

    return (entry["repo"], filename)


def list_catalog() -> list[dict]:
    """List all models in the catalog for UI display."""
    result = []
    for name, entry in _CATALOG.items():
        result.append({
            "name": name,
            "repo": entry["repo"],
            "quants": list(entry["quants"].keys()),
            "default_quant": entry.get("default_quant", "q4_k_m"),
        })
    return result
