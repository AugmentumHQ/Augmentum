"""Model profile system — per-model optimal settings with user override persistence.

Each image model has a profile defining:
- Recommended generation defaults (steps, CFG, resolution, sampler)
- Feature compatibility (which enhancers work with this pipeline type)
- Auto-enable list (features turned on by default for this model)

User overrides are stored per-model in the settings store. When a model
loads, the profile applies defaults, then user overrides win. Users can
reset to defaults at any time.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Feature compatibility by pipeline type
# ---------------------------------------------------------------------------

# Which features work with which pipeline architectures.
# True = works, False = incompatible (will be greyed out in UI).
PIPELINE_FEATURES: dict[str, dict[str, bool]] = {
    "sd15": {
        "freeu": True,
        "tome": True,
        "ip_adapter": True,
        "lora": True,
        "hires_fix": True,
        "negative_prompt": True,
        "sampler_swap": True,
        "cfg_rescale": True,
        "prompt_condense": True,
        "clip_skip": True,
    },
    "sdxl": {
        "freeu": True,
        "tome": True,
        "ip_adapter": True,
        "lora": True,
        "hires_fix": True,
        "negative_prompt": True,
        "sampler_swap": True,  # except EDM models (handled per-model)
        "cfg_rescale": True,
        "prompt_condense": True,
        "clip_skip": True,
    },
    "flux": {
        "freeu": False,       # no UNet (transformer-based)
        "tome": False,        # no UNet
        "ip_adapter": False,  # no IP-Adapter weights for Flux yet
        "lora": True,
        "hires_fix": True,
        "negative_prompt": False,  # flow matching, no negative
        "sampler_swap": False,     # FlowMatchEulerDiscreteScheduler only
        "cfg_rescale": False,
        "prompt_condense": False,  # T5 handles long prompts natively
        "clip_skip": False,
    },
}

# Default features to auto-enable per pipeline type
PIPELINE_AUTO_ENABLE: dict[str, list[str]] = {
    "sd15": ["freeu", "negative_prompt"],
    "sdxl": ["freeu", "negative_prompt"],
    "flux": [],
}

# Default generation parameters per pipeline type (fallback if no model-specific entry)
PIPELINE_DEFAULTS: dict[str, dict[str, Any]] = {
    "sd15": {"steps": 25, "cfg_scale": 7.0, "width": 512, "height": 512, "sampler": "dpm++_2m_karras"},
    "sdxl": {"steps": 30, "cfg_scale": 7.0, "width": 1024, "height": 1024, "sampler": "dpm++_2m_karras"},
    "flux": {"steps": 4, "cfg_scale": 0.0, "width": 1024, "height": 1024, "sampler": ""},
}


# ---------------------------------------------------------------------------
# Per-model overrides (from research + model cards)
# ---------------------------------------------------------------------------

# Keys match against model path/name (case-insensitive substring).
# First match wins, so more specific patterns go first.
MODEL_SPECIFIC: list[dict] = [
    # --- SD 1.5 ---
    {"match": ["bk-sdm-small", "bk_sdm"],
     "defaults": {"steps": 25, "cfg_scale": 7.0},
     "auto_enable": ["freeu", "negative_prompt"]},

    {"match": ["dreamshaper-8", "dreamshaper8"],
     "defaults": {"steps": 25, "cfg_scale": 7.0, "sampler": "dpm++_2m_karras"},
     "auto_enable": ["freeu", "negative_prompt"]},

    {"match": ["epicrealismxl"],  # must be before "epicrealism" (substring match)
     "defaults": {"steps": 25, "cfg_scale": 6.0, "sampler": "euler"},
     "pipeline": "sdxl",
     "auto_enable": ["freeu", "negative_prompt"]},

    {"match": ["epicrealism"],
     "defaults": {"steps": 25, "cfg_scale": 5.5, "sampler": "euler_a"},
     "auto_enable": ["freeu", "negative_prompt"],
     "note": "Lower CFG avoids over-saturation"},

    {"match": ["realistic-vision", "realistic_vision"],
     "defaults": {"steps": 25, "cfg_scale": 5.5, "sampler": "dpm++_2m_sde_karras"},
     "auto_enable": ["freeu", "negative_prompt"]},

    # --- SDXL ---
    {"match": ["dreamshaper-xl"],
     "defaults": {"steps": 6, "cfg_scale": 2.0, "sampler": "dpm++_2m_sde_karras"},
     "pipeline": "sdxl",
     "auto_enable": ["negative_prompt"],
     "note": "Turbo distilled — low steps required"},

    {"match": ["animagine"],
     "defaults": {"steps": 28, "cfg_scale": 5.0, "sampler": "euler"},
     "pipeline": "sdxl",
     "auto_enable": ["freeu", "negative_prompt"],
     "note": "Danbooru-style tag prompts recommended"},

    {"match": ["playground-v2.5", "playground_v2.5"],
     "defaults": {"steps": 50, "cfg_scale": 3.0, "sampler": "euler"},
     "pipeline": "sdxl",
     "auto_enable": ["negative_prompt"],
     "features_override": {"sampler_swap": False},
     "note": "EDM sigma model — uses EDMEulerScheduler only"},

    {"match": ["juggernaut"],
     "defaults": {"steps": 30, "cfg_scale": 6.5, "sampler": "dpm++_2m_karras"},
     "pipeline": "sdxl",
     "auto_enable": ["freeu", "negative_prompt"]},

    {"match": ["realvisxl", "realvis"],
     "defaults": {"steps": 30, "cfg_scale": 6.0, "sampler": "dpm++_2m_sde_karras"},
     "pipeline": "sdxl",
     "auto_enable": ["freeu", "negative_prompt"]},

    {"match": ["stable-diffusion-xl-base"],
     "defaults": {"steps": 30, "cfg_scale": 7.0, "sampler": "dpm++_2m_karras"},
     "pipeline": "sdxl",
     "auto_enable": ["freeu", "negative_prompt"]},

    # --- SD 3.5 (flow matching) ---
    {"match": ["stable-diffusion-3.5-large-turbo", "sd3.5-large-turbo"],
     "defaults": {"steps": 4, "cfg_scale": 0.0},
     "pipeline": "flux",  # uses flow matching like Flux
     "note": "Guidance-distilled — CFG must be 0"},

    {"match": ["stable-diffusion-3.5-medium", "sd3.5-medium"],
     "defaults": {"steps": 40, "cfg_scale": 4.5},
     "pipeline": "flux"},

    # --- Flux ---
    {"match": ["flux.1-schnell", "flux1-schnell"],
     "defaults": {"steps": 4, "cfg_scale": 0.0},
     "pipeline": "flux",
     "note": "Guidance-free distilled — 1-4 steps, no CFG"},

    {"match": ["flux.1-fill", "flux1-fill"],
     "defaults": {"steps": 50, "cfg_scale": 30.0},
     "pipeline": "flux",
     "note": "Inpainting model — very high guidance scale"},

    {"match": ["flux.1-kontext", "flux1-kontext", "kontext"],
     "defaults": {"steps": 28, "cfg_scale": 2.5},
     "pipeline": "flux",
     "note": "Image editing / style transfer model"},

    {"match": ["flux.2-klein", "flux2-klein"],
     "defaults": {"steps": 4, "cfg_scale": 1.0},
     "pipeline": "flux",
     "note": "Timestep-distilled — 4 steps"},

    {"match": ["flux.2-dev", "flux2-dev"],
     "defaults": {"steps": 28, "cfg_scale": 4.0},
     "pipeline": "flux"},

    {"match": ["flux.1-dev", "flux1-dev"],
     "defaults": {"steps": 50, "cfg_scale": 3.5},
     "pipeline": "flux"},

    # --- Sana ---
    {"match": ["sana-600m", "sana_600m", "sana-sprint"],
     "defaults": {"steps": 20, "cfg_scale": 4.5},
     "pipeline": "flux"},

    # --- PixArt-Sigma (standard scheduler despite DiT architecture) ---
    {"match": ["pixart-sigma", "pixart_sigma"],
     "defaults": {"steps": 20, "cfg_scale": 4.5},
     "pipeline": "sdxl",  # uses standard schedulers, NOT flow matching
     "features_override": {"clip_skip": False}},

    # --- Z-Image ---
    {"match": ["z-image-turbo", "z_image_turbo", "zimage-turbo"],
     "defaults": {"steps": 9, "cfg_scale": 0.0},
     "pipeline": "flux",
     "note": "Guidance-free distilled"},

    {"match": ["z-image", "z_image", "zimage"],
     "defaults": {"steps": 50, "cfg_scale": 4.0},
     "pipeline": "flux"},

    # --- Z-Anime (full fine-tune of Z-Image Base) ---
    {"match": ["z-anime", "z_anime", "zanime"],
     "defaults": {"steps": 30, "cfg_scale": 4.0, "sampler": "euler_a"},
     "pipeline": "flux",
     "note": "Anime fine-tune — 28-50 steps, CFG 3.0-5.0, euler_ancestral"},

    # --- Lumina ---
    {"match": ["netayume", "lumina-image"],
     "defaults": {"steps": 50, "cfg_scale": 4.0},
     "pipeline": "flux"},
]


# ---------------------------------------------------------------------------
# Profile Resolution
# ---------------------------------------------------------------------------

@dataclass
class ModelProfile:
    """Resolved profile for a specific model."""
    model_name: str
    pipeline_type: str
    defaults: dict[str, Any] = field(default_factory=dict)
    features: dict[str, bool] = field(default_factory=dict)
    auto_enable: list[str] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "model": self.model_name,
            "pipeline": self.pipeline_type,
            "defaults": self.defaults,
            "features": self.features,
            "autoEnable": self.auto_enable,
            "note": self.note,
        }


def resolve_profile(model_name: str, pipeline_type: str = "") -> ModelProfile:
    """Resolve the optimal profile for a model.

    Checks model-specific overrides first, then falls back to pipeline
    type defaults. Returns a complete ModelProfile with all fields populated.
    """
    name_lower = model_name.lower()

    # Detect pipeline type if not provided
    if not pipeline_type:
        pipeline_type = "sd15"  # conservative default

    # Start with pipeline-level defaults
    pipe_defaults = dict(PIPELINE_DEFAULTS.get(pipeline_type, PIPELINE_DEFAULTS["sd15"]))
    pipe_features = dict(PIPELINE_FEATURES.get(pipeline_type, PIPELINE_FEATURES["sd15"]))
    auto_enable = list(PIPELINE_AUTO_ENABLE.get(pipeline_type, []))
    note = ""

    # Check for model-specific override
    for entry in MODEL_SPECIFIC:
        if any(pat in name_lower for pat in entry["match"]):
            # Override pipeline type if specified
            if "pipeline" in entry:
                pipeline_type = entry["pipeline"]
                pipe_features = dict(PIPELINE_FEATURES.get(pipeline_type, pipe_features))
                pipe_defaults.update(PIPELINE_DEFAULTS.get(pipeline_type, {}))
                auto_enable = list(PIPELINE_AUTO_ENABLE.get(pipeline_type, []))

            # Apply model-specific defaults
            if "defaults" in entry:
                pipe_defaults.update(entry["defaults"])

            # Apply feature overrides
            if "features_override" in entry:
                pipe_features.update(entry["features_override"])

            # Auto-enable overrides
            if "auto_enable" in entry:
                auto_enable = entry["auto_enable"]

            if "note" in entry:
                note = entry["note"]

            break

    return ModelProfile(
        model_name=model_name,
        pipeline_type=pipeline_type,
        defaults=pipe_defaults,
        features=pipe_features,
        auto_enable=auto_enable,
        note=note,
    )


# ---------------------------------------------------------------------------
# User Override Persistence
# ---------------------------------------------------------------------------

def _profile_key(model_name: str) -> str:
    """Generate a settings store key for a model's user overrides."""
    # Normalize: strip paths, lowercase, replace special chars
    short = model_name.rsplit("/", 1)[-1].lower()
    short = short.replace(" ", "-").replace(".", "-")
    return f"ui.modelProfile.{short}"


async def load_user_overrides(model_name: str, settings_store) -> dict:
    """Load user overrides for a model from the settings store."""
    if not settings_store:
        return {}
    key = _profile_key(model_name)
    raw = await settings_store.get(key)
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {}


async def save_user_overrides(model_name: str, overrides: dict, settings_store) -> None:
    """Save user overrides for a model to the settings store."""
    if not settings_store:
        return
    key = _profile_key(model_name)
    await settings_store.set(key, json.dumps(overrides))


async def clear_user_overrides(model_name: str, settings_store) -> None:
    """Clear user overrides for a model (reset to defaults)."""
    if not settings_store:
        return
    key = _profile_key(model_name)
    await settings_store.delete(key)


async def get_effective_profile(
    model_name: str,
    pipeline_type: str,
    settings_store=None,
) -> dict:
    """Get the effective profile with user overrides applied.

    Returns a dict ready for the frontend:
    {
        "model": "...",
        "pipeline": "sd15",
        "defaults": {steps, cfg_scale, width, height, sampler},
        "features": {freeu: true, tome: true, ...},
        "autoEnable": ["freeu", "negative_prompt"],
        "userOverrides": {cfg_scale: 5.0},  # what the user changed
        "note": "..."
    }
    """
    profile = resolve_profile(model_name, pipeline_type)
    result = profile.to_dict()

    # Load and apply user overrides
    overrides = await load_user_overrides(model_name, settings_store)
    result["userOverrides"] = overrides

    # Merge overrides into defaults (overrides win)
    if overrides:
        for key, value in overrides.items():
            if key in result["defaults"]:
                result["defaults"][key] = value
            elif key.startswith("feature_"):
                # Feature toggles: "feature_freeu" -> features.freeu
                feat = key.replace("feature_", "")
                if feat in result["features"]:
                    # Only override if the feature is compatible
                    if result["features"][feat] or not value:
                        result["autoEnable"] = [
                            f for f in result["autoEnable"] if f != feat
                        ]
                        if value:
                            result["autoEnable"].append(feat)

    return result
