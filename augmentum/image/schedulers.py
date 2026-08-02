"""Scheduler/sampler mapping for image generation pipelines.

Maps user-friendly sampler names to diffusers scheduler classes with
appropriate configuration kwargs. Diffusers imports are lazy — this module
can be imported safely without diffusers installed (e.g. in tests).
"""

from __future__ import annotations

# Each entry maps a canonical sampler name to (scheduler_class_name, kwargs).
# The class name is resolved lazily inside ``apply_sampler`` so that
# ``diffusers`` is only imported when actually swapping a scheduler.
SAMPLER_MAP: dict[str, tuple[str, dict]] = {
    "euler": ("EulerDiscreteScheduler", {}),
    "euler_a": ("EulerAncestralDiscreteScheduler", {}),
    "dpm++_2m": ("DPMSolverMultistepScheduler", {}),
    "dpm++_2m_karras": ("DPMSolverMultistepScheduler", {"use_karras_sigmas": True}),
    "dpm++_2m_sde": ("DPMSolverMultistepScheduler", {"algorithm_type": "sde-dpmsolver++"}),
    "dpm++_2m_sde_karras": (
        "DPMSolverMultistepScheduler",
        {"algorithm_type": "sde-dpmsolver++", "use_karras_sigmas": True},
    ),
    "ddim": ("DDIMScheduler", {}),
    "lms": ("LMSDiscreteScheduler", {}),
    "unipc": ("UniPCMultistepScheduler", {}),
    "lcm": ("LCMScheduler", {}),
    "heun": ("HeunDiscreteScheduler", {}),
    "deis": ("DEISMultistepScheduler", {}),
    "pndm": ("PNDMScheduler", {}),
}

# Common aliases that map to canonical names above.
SAMPLER_ALIASES: dict[str, str] = {
    "euler_ancestral": "euler_a",
    "ancestral_euler": "euler_a",
    "dpmpp_2m": "dpm++_2m",
    "k_dpm_2m": "dpm++_2m",
    "dpmpp_2m_karras": "dpm++_2m_karras",
    "k_dpm_2m_karras": "dpm++_2m_karras",
    "dpmpp_2m_sde": "dpm++_2m_sde",
    "dpmpp_2m_sde_karras": "dpm++_2m_sde_karras",
    "k_dpm_2m_sde": "dpm++_2m_sde",
    "k_dpm_2m_sde_karras": "dpm++_2m_sde_karras",
    "uni_pc": "unipc",
}

# Human-readable display names, categories, and descriptions for UI.
_SAMPLER_META: dict[str, dict] = {
    "euler": {
        "display": "Euler",
        "category": "recommended",
        "desc": "Fast and reliable. Great starting point for most images.",
    },
    "euler_a": {
        "display": "Euler Ancestral",
        "category": "recommended",
        "desc": "More creative variation between steps. Good for artistic styles.",
    },
    "dpm++_2m_karras": {
        "display": "DPM++ 2M Karras",
        "category": "recommended",
        "desc": "Sharp details with smooth gradients. Best all-rounder for quality.",
    },
    "dpm++_2m": {
        "display": "DPM++ 2M",
        "category": "quality",
        "desc": "Clean, detailed results. Slightly less contrast than Karras variant.",
    },
    "dpm++_2m_sde_karras": {
        "display": "DPM++ 2M SDE Karras",
        "category": "quality",
        "desc": "Adds controlled randomness for more natural-looking textures.",
    },
    "dpm++_2m_sde": {
        "display": "DPM++ 2M SDE",
        "category": "quality",
        "desc": "Natural textures with stochastic sampling. Good for landscapes.",
    },
    "ddim": {
        "display": "DDIM",
        "category": "specialized",
        "desc": "Deterministic and consistent. Useful for img2img and inpainting.",
    },
    "unipc": {
        "display": "UniPC",
        "category": "specialized",
        "desc": "Very fast convergence — good results in fewer steps (10-15).",
    },
    "heun": {
        "display": "Heun",
        "category": "specialized",
        "desc": "Higher quality per step but takes ~2x longer. Worth it for low step counts.",
    },
    "lcm": {
        "display": "LCM",
        "category": "fast",
        "desc": "Designed for speed models (LCM/Turbo). 4-8 steps, low CFG.",
    },
    "lms": {
        "display": "LMS",
        "category": "specialized",
        "desc": "Classic sampler. Stable but can produce less detail than DPM++.",
    },
    "deis": {
        "display": "DEIS",
        "category": "specialized",
        "desc": "Fast solver with good quality at low step counts.",
    },
    "pndm": {
        "display": "PNDM",
        "category": "specialized",
        "desc": "Legacy sampler. Included for compatibility with older workflows.",
    },
}

# Backwards-compat helper
_DISPLAY_NAMES: dict[str, str] = {k: v["display"] for k, v in _SAMPLER_META.items()}


def _resolve_alias(name: str) -> str:
    """Resolve a sampler name through the alias table (case-insensitive)."""
    lower = name.strip().lower()
    return SAMPLER_ALIASES.get(lower, lower)


def get_available_samplers() -> list[dict]:
    """Return a list of available samplers with metadata.

    Each entry is a dict with ``name``, ``display_name``, ``aliases``,
    ``category``, and ``description``.
    """
    # Order: recommended first, then quality, fast, specialized
    _CAT_ORDER = {"recommended": 0, "quality": 1, "fast": 2, "specialized": 3}
    result: list[dict] = []
    for canonical in SAMPLER_MAP:
        meta = _SAMPLER_META.get(canonical, {})
        aliases = [alias for alias, target in SAMPLER_ALIASES.items() if target == canonical]
        result.append({
            "name": canonical,
            "display_name": meta.get("display", canonical),
            "aliases": aliases,
            "category": meta.get("category", "specialized"),
            "description": meta.get("desc", ""),
        })
    result.sort(key=lambda s: (_CAT_ORDER.get(s["category"], 9), s["display_name"]))
    return result


def apply_sampler(pipe, sampler_name: str) -> None:
    """Swap the scheduler on *pipe* to match *sampler_name*.

    Imports ``diffusers`` lazily so the module can be loaded without the
    library installed.  Raises :class:`ValueError` for unknown sampler names.

    EDM-based models (Playground v2.5, etc.) use a different sigma space.
    Swapping to a non-EDM scheduler produces static/noise. We detect EDM
    pipelines and skip the swap if no EDM-compatible scheduler is available.
    """
    canonical = _resolve_alias(sampler_name)

    if canonical not in SAMPLER_MAP:
        available = ", ".join(sorted(SAMPLER_MAP.keys()))
        raise ValueError(
            f"Unknown sampler '{sampler_name}'. Available: {available}"
        )

    class_name, kwargs = SAMPLER_MAP[canonical]

    # Lazy import — only pull in diffusers when actually applying
    import diffusers  # noqa: E402

    current_scheduler = type(pipe.scheduler).__name__

    # --- Flow Matching models (Flux, SD3.5, Sana, Z-Image, Lumina) ---
    # These use FlowMatchEulerDiscreteScheduler with a completely different
    # denoising objective (rectified flow). Standard schedulers produce
    # garbage. We must keep the native scheduler.
    is_flow = "FlowMatch" in current_scheduler
    if is_flow:
        # FlowMatchEulerDiscreteScheduler is the ONLY valid scheduler.
        # Silently ignore sampler swap requests.
        return

    # --- EDM models (Playground v2.5) ---
    # Use EDM sigma schedule. Standard schedulers produce static/noise.
    is_edm = "EDM" in current_scheduler or getattr(pipe.scheduler.config, "sigma_min", None) is not None
    if is_edm:
        edm_class = getattr(diffusers, "EDMEulerScheduler", None)
        edm_dpm = getattr(diffusers, "EDMDPMSolverMultistepScheduler", None)

        if canonical in ("euler", "euler_a") and edm_class:
            pipe.scheduler = edm_class.from_config(pipe.scheduler.config)
            return
        if canonical.startswith("dpm") and edm_dpm:
            pipe.scheduler = edm_dpm.from_config(pipe.scheduler.config)
            return
        # For other samplers, keep current EDM scheduler
        return

    # --- Standard models (SD1.5, SDXL, PixArt-Sigma) ---
    scheduler_cls = getattr(diffusers, class_name)
    pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config, **kwargs)
