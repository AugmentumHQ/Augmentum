"""Default generation parameters per pipeline type."""

from __future__ import annotations

PIPELINE_NEGATIVE_DEFAULTS: dict[str, str] = {
    "sd15": "worst quality, low quality, blurry, jpeg artifacts, watermark, text",
    "sdxl": "worst quality, low quality, blurry, jpeg artifacts, watermark, text",
    "flux": "",  # FLUX doesn't use negative prompts (no CFG)
}


def resolve_negative_prompt(
    negative_prompt: str,
    pipeline_type: str,
    config_default: str = "",
) -> str:
    """Resolve the effective negative prompt.

    Priority: explicit *negative_prompt* > *config_default* > pipeline default.

    Returns the negative prompt string to use.
    """
    if negative_prompt:
        return negative_prompt
    if config_default:
        return config_default
    return PIPELINE_NEGATIVE_DEFAULTS.get(pipeline_type, "")
