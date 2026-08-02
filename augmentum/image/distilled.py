"""Detection and auto-configuration for distilled/accelerated image models.

Provides recommended steps and CFG scale for known model families, sourced
from official HuggingFace model cards and documentation.
"""

from __future__ import annotations

DISTILLED_PATTERNS: dict[str, dict] = {
    "z-image-turbo": {
        "steps": 9,
        "cfg_scale": 0.0,
        "patterns": ["z-image-turbo", "z_image_turbo", "zimage-turbo"],
    },
    "turbo": {
        "steps": 4,
        "cfg_scale": 2.0,
        "patterns": ["turbo"],
    },
    "lightning": {
        "steps": 4,
        "cfg_scale": 2.0,
        "patterns": ["lightning"],
    },
    "lcm": {
        "steps": 8,
        "cfg_scale": 1.5,
        "patterns": ["lcm"],
    },
    "hyper": {
        "steps": 4,
        "cfg_scale": 1.5,
        "patterns": ["hyper-sd", "hyper_sd", "hypersd"],
    },
}


# ---------------------------------------------------------------------------
# Model-specific defaults (from official HuggingFace cards / docs)
# Checked BEFORE distilled patterns so exact model names win over generic
# "turbo" substring matching.  Longer patterns checked first within each
# group to avoid prefix collisions (e.g. "qwen-image-edit" before
# "qwen-image").
# ---------------------------------------------------------------------------

_MODEL_DEFAULTS: list[tuple[list[str], dict]] = [
    # --- Qwen Image family ---
    # Source: https://huggingface.co/Qwen/Qwen-Image-Edit-2511
    (["qwen-image-edit"], {"steps": 40, "cfg_scale": 4.0}),
    # Source: https://huggingface.co/Qwen/Qwen-Image-Layered
    (["qwen-image-layered"], {"steps": 50, "cfg_scale": 4.0}),
    # Source: https://huggingface.co/Qwen/Qwen-Image-2512
    (["qwen-image-2512"], {"steps": 50, "cfg_scale": 4.0}),
    # Source: https://huggingface.co/Qwen/Qwen-Image
    (["qwen-image"], {"steps": 50, "cfg_scale": 4.0}),

    # --- FLUX family ---
    # Source: https://huggingface.co/black-forest-labs/FLUX.1-schnell
    (["flux.1-schnell", "flux1-schnell"], {"steps": 4, "cfg_scale": 0.0}),
    # Source: https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev
    (["flux.1-fill", "flux1-fill"], {"steps": 50, "cfg_scale": 30.0}),
    # Source: https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev
    (["flux.1-kontext", "flux1-kontext", "kontext"], {"steps": 28, "cfg_scale": 2.5}),
    # Source: https://huggingface.co/black-forest-labs/FLUX.2-klein-4B, FLUX.2-klein-9B
    (["flux.2-klein", "flux2-klein"], {"steps": 4, "cfg_scale": 1.0}),
    # Source: https://huggingface.co/black-forest-labs/FLUX.1-dev
    (["flux.1-dev", "flux1-dev"], {"steps": 50, "cfg_scale": 3.5}),
    # Source: https://huggingface.co/black-forest-labs/FLUX.2-dev
    (["flux.2-dev", "flux2-dev"], {"steps": 28, "cfg_scale": 4.0}),

    # --- Z-Image family ---
    # Source: https://huggingface.co/Tongyi-MAI/Z-Image-Turbo
    (["z-image-turbo", "z_image_turbo", "zimage-turbo"], {"steps": 9, "cfg_scale": 0.0}),
    # Source: https://huggingface.co/Tongyi-MAI/Z-Image
    (["z-image", "z_image", "zimage"], {"steps": 50, "cfg_scale": 4.0}),
    # Source: https://huggingface.co/SeeSee21/Z-Anime  (full fine-tune of Z-Image Base)
    (["z-anime", "z_anime", "zanime"], {"steps": 30, "cfg_scale": 4.0}),

    # --- Stable Diffusion 3.5 family ---
    # Source: https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo
    # (matched by DISTILLED_PATTERNS "turbo" → 4 steps, but CFG should be 0.0 not 2.0)
    (["stable-diffusion-3.5-large-turbo", "sd-3.5-large-turbo", "sd3.5-large-turbo"],
     {"steps": 4, "cfg_scale": 0.0}),
    # Source: https://huggingface.co/stabilityai/stable-diffusion-3.5-medium
    (["stable-diffusion-3.5-medium", "sd-3.5-medium", "sd3.5-medium"],
     {"steps": 40, "cfg_scale": 4.5}),

    # --- Sana family ---
    # Source: https://huggingface.co/Efficient-Large-Model/Sana_600M_1024px_diffusers
    (["sana-600m", "sana_600m", "sana-sprint"], {"steps": 20, "cfg_scale": 4.5}),

    # --- PixArt-Sigma ---
    # Source: https://huggingface.co/PixArt-alpha/PixArt-Sigma-XL-2-1024-MS (pipeline defaults)
    (["pixart-sigma", "pixart_sigma"], {"steps": 20, "cfg_scale": 4.5}),

    # --- Lumina / NetaYume ---
    # Source: https://huggingface.co/duongve/NetaYume-Lumina-Image-2.0-Diffusers-v40
    (["netayume", "lumina-image"], {"steps": 30, "cfg_scale": 4.0}),

    # --- SD 1.5 fine-tunes ---
    # Source: https://huggingface.co/Lykon/dreamshaper-8
    (["dreamshaper-8", "dreamshaper8"], {"steps": 25, "cfg_scale": 7.0}),
    # Source: https://huggingface.co/emilianJR/epiCRealism
    (["epicrealism"], {"steps": 25, "cfg_scale": 5.5}),
    # Source: https://huggingface.co/stablediffusionapi/realistic-vision-v51
    (["realistic-vision", "realistic_vision"], {"steps": 25, "cfg_scale": 5.5}),

    # --- SDXL family ---
    # Source: https://huggingface.co/cagliostrolab/animagine-xl-4.0
    (["animagine"], {"steps": 28, "cfg_scale": 5.0}),
    # Source: https://huggingface.co/playgroundai/playground-v2.5-1024px-aesthetic
    (["playground-v2.5", "playground_v2.5"], {"steps": 50, "cfg_scale": 3.0}),
    # Source: https://huggingface.co/Lykon/dreamshaper-xl-v2-turbo (matched by "turbo" too)
    (["dreamshaper-xl"], {"steps": 6, "cfg_scale": 2.0}),
    # Source: https://huggingface.co/RunDiffusion/Juggernaut-XL-v9
    (["juggernaut"], {"steps": 30, "cfg_scale": 6.5}),
    # Source: https://huggingface.co/nroggendorff/epicrealismxl
    (["epicrealismxl"], {"steps": 25, "cfg_scale": 6.0}),
    # Source: https://huggingface.co/SG161222/RealVisXL_V4.0
    (["realvisxl", "realvis"], {"steps": 30, "cfg_scale": 6.0}),
]


def detect_distilled_type(model_name: str) -> str | None:
    """Detect if a model is a distilled/accelerated variant.

    Performs case-insensitive check of *model_name* (which may be a file
    path or HuggingFace repo-style name) against known distilled-model
    patterns.

    Returns the distilled type key (``"turbo"``, ``"lightning"``,
    ``"lcm"``, ``"hyper"``) or ``None`` if the model is not recognised
    as distilled.
    """
    name_lower = model_name.lower()
    for dist_type, info in DISTILLED_PATTERNS.items():
        for pattern in info["patterns"]:
            if pattern in name_lower:
                return dist_type
    return None


def detect_model_defaults(model_name: str) -> dict:
    """Return recommended defaults for a specific model.

    Checks the model name against known model families.  Returns a dict
    with ``"steps"`` and ``"cfg_scale"`` keys, or an empty dict if the
    model is not recognized.
    """
    name_lower = (
        model_name.lower()
        .replace("/", "-")
        .replace("\\", "-")
        .replace("_", "-")
    )
    for patterns, defaults in _MODEL_DEFAULTS:
        for pattern in patterns:
            if pattern in name_lower:
                return defaults
    return {}


# Keep legacy helpers for backwards compatibility with existing callers.
def detect_qwen_defaults(model_name: str) -> dict:
    """Return Qwen-specific generation defaults if the model is a Qwen Image model."""
    return detect_model_defaults(model_name) if "qwen" in model_name.lower() else {}


def detect_zimage_defaults(model_name: str) -> dict:
    """Return Z-Image-specific generation defaults."""
    return detect_model_defaults(model_name) if "z-image" in model_name.lower().replace("_", "-") else {}


def get_distilled_defaults(distilled_type: str | None) -> dict:
    """Return recommended generation defaults for a distilled type.

    Returns a dict with ``"steps"`` and ``"cfg_scale"`` keys for known
    types, or an empty dict for ``None`` / unknown types.
    """
    if distilled_type is None:
        return {}
    info = DISTILLED_PATTERNS.get(distilled_type)
    if info is None:
        return {}
    return {"steps": info["steps"], "cfg_scale": info["cfg_scale"]}


def apply_distilled_defaults(
    model_name: str,
    steps: int | None,
    cfg_scale: float | None,
) -> tuple[int, float]:
    """Return ``(steps, cfg_scale)`` with model-aware defaults.

    * First checks the model-specific defaults table (exact model names).
    * Then checks distilled-type patterns (turbo, lightning, etc.).
    * User-provided values (non-``None``) always override auto-defaults.
    * For unrecognized models, falls back to 20 steps / 7.0 CFG.
    """
    # Model-specific defaults take priority over generic distilled patterns
    defaults = detect_model_defaults(model_name)

    # Fall back to generic distilled-type detection (turbo, lightning, etc.)
    if not defaults:
        distilled_type = detect_distilled_type(model_name)
        defaults = get_distilled_defaults(distilled_type)

    if defaults:
        effective_steps = steps if steps is not None else defaults["steps"]
        effective_cfg = cfg_scale if cfg_scale is not None else defaults["cfg_scale"]
    else:
        effective_steps = steps if steps is not None else 20
        effective_cfg = cfg_scale if cfg_scale is not None else 7.0

    return effective_steps, effective_cfg


def get_recommended_defaults(model_name: str) -> dict:
    """Return the recommended defaults for a model (for UI display).

    Returns ``{"steps": N, "cfg_scale": X}`` or ``{}`` if the model
    has no specific recommendations (generic 20/7.0 applies).
    """
    defaults = detect_model_defaults(model_name)
    if defaults:
        return defaults
    distilled_type = detect_distilled_type(model_name)
    return get_distilled_defaults(distilled_type)
