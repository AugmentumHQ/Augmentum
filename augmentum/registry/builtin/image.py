"""Image-generation settings — full user-tunable surface migrated
into the declarative substrate.
"""

from __future__ import annotations

from augmentum.registry.registry import SettingsRegistry
from augmentum.registry.settings import Setting


def register(r: SettingsRegistry) -> None:
    # ---- Performance ----
    r.register(
        Setting(
            key="image_torch_compile",
            kind="enum",
            default="auto",
            label="torch.compile mode",
            description=(
                "Use PyTorch 2.x torch.compile to JIT image-model forward "
                "passes. 'auto' enables on Ampere+ with gcc. 'on' forces. "
                "'off' disables (slowest, safest, no first-call warmup)."
            ),
            section="image.performance",
            enum_values=("auto", "on", "off"),
            max_length=10,
            tags=("image", "advanced"),
            advanced=True,
            restart_required=True,
            trust_tier="admin_only",
            companion_surfaceable=False,
        )
    )

    r.register(
        Setting(
            key="image_prompt_condense_model",
            kind="str",
            default="",
            label="Prompt-condense model",
            description=(
                "Model used to condense overlong prompts to fit the image "
                "model's CLIP/T5 token limit. Empty = default backend model."
            ),
            section="image.performance",
            max_length=256,
            tags=("image", "advanced"),
            advanced=True,
            trust_tier="admin_only",
        )
    )

    # ---- Quality knobs ----
    r.register(
        Setting(
            key="image_freeu_enabled",
            kind="bool",
            default=True,
            label="FreeU",
            description=(
                "FreeU: rebalance UNet skip connections for crisper detail. "
                "SD1.5 / SDXL only; no speed cost."
            ),
            section="image.quality",
            tags=("image", "advanced"),
            advanced=True,
            trust_tier="admin_only",
        )
    )

    r.register(
        Setting(
            key="image_tome_enabled",
            kind="bool",
            default=False,
            label="Token Merging (ToMe)",
            description=(
                "Merge similar tokens for 20-40% speedup. SD1.5 / SDXL only. "
                "Trades a small quality hit for throughput."
            ),
            section="image.quality",
            tags=("image", "advanced"),
            advanced=True,
            trust_tier="admin_only",
        )
    )

    r.register(
        Setting(
            key="image_tome_ratio",
            kind="float",
            default=0.5,
            label="ToMe merge ratio",
            description=(
                "Fraction of tokens merged when ToMe is enabled. Higher = "
                "faster but lower quality. 0.1-0.9."
            ),
            section="image.quality",
            min_value=0.1,
            max_value=0.9,
            tags=("image", "advanced"),
            advanced=True,
            trust_tier="admin_only",
        )
    )

    r.register(
        Setting(
            key="image_cfg_rescale",
            kind="float",
            default=0.0,
            label="CFG rescale",
            description=(
                "Rescale classifier-free guidance to prevent overexposure at "
                "high CFG values. 0 = off, 0.7 = strong rescale."
            ),
            section="image.quality",
            min_value=0.0,
            max_value=1.0,
            tags=("image", "advanced"),
            advanced=True,
            trust_tier="admin_only",
        )
    )

    r.register(
        Setting(
            key="image_hires_fix",
            kind="bool",
            default=False,
            label="Hires fix",
            description=(
                "Two-pass generation: generate at base resolution, upscale, "
                "then img2img refine. Better detail at the cost of ~2x time."
            ),
            section="image.quality",
            tags=("image", "advanced"),
            advanced=True,
            trust_tier="admin_only",
        )
    )

    r.register(
        Setting(
            key="image_hires_scale",
            kind="float",
            default=1.5,
            label="Hires fix scale",
            description=(
                "Upscale factor for the hires fix pass. 1.5 = +50% pixels; "
                "2.0 = 4x pixels (expensive)."
            ),
            section="image.quality",
            min_value=1.0,
            max_value=4.0,
            tags=("image", "advanced"),
            advanced=True,
            trust_tier="admin_only",
        )
    )

    r.register(
        Setting(
            key="image_hires_denoise",
            kind="float",
            default=0.5,
            label="Hires fix denoise",
            description=(
                "img2img denoise strength on the hires-fix refine pass. "
                "Higher = more divergence from the base image, more refinement."
            ),
            section="image.quality",
            min_value=0.0,
            max_value=1.0,
            tags=("image", "advanced"),
            advanced=True,
            trust_tier="admin_only",
        )
    )

    r.register(
        Setting(
            key="image_ip_adapter_enabled",
            kind="bool",
            default=True,
            label="IP-Adapter",
            description=(
                "Enable image-reference conditioning via IP-Adapter. Lets "
                "you inject a style or subject from a reference image."
            ),
            section="image.quality",
            tags=("image",),
            trust_tier="admin_only",
        )
    )

    r.register(
        Setting(
            key="image_ip_adapter_scale",
            kind="float",
            default=0.55,
            label="IP-Adapter strength",
            description=(
                "How strongly the reference image influences generation. "
                "0.55 is the recommended default; tune per use case."
            ),
            section="image.quality",
            min_value=0.0,
            max_value=1.0,
            tags=("image",),
            trust_tier="admin_only",
        )
    )

    # ---- Security / boundaries ----
    r.register(
        Setting(
            key="image_allow_pickle_formats",
            kind="bool",
            default=False,
            label="Allow pickle imports",
            description=(
                "Opt-in: accept .bin / .pt / .pth / .ckpt model uploads. These "
                "are pickle files and can execute arbitrary code on load — "
                "only enable for trusted sources."
            ),
            section="image.security",
            tags=("image", "security"),
            trust_tier="admin_only",
            companion_surfaceable=False,
        )
    )

    r.register(
        Setting(
            key="image_upload_max_size_gb",
            kind="int",
            default=20,
            label="Max upload size (GB)",
            description=(
                "Reject custom model imports larger than this. Prevents "
                "accidental terabyte uploads from filling disk."
            ),
            section="image.security",
            min_value=1,
            max_value=500,
            tags=("image", "security"),
            trust_tier="admin_only",
        )
    )

    r.register(
        Setting(
            key="image_imports_dir",
            kind="str",
            default="",
            label="Server-side imports directory",
            description=(
                "Allowlisted server-side path prefix for offline imports. "
                "Empty = path imports disabled. Set to a specific dir to "
                "allow filesystem-based model registration."
            ),
            section="image.security",
            max_length=512,
            tags=("image", "security"),
            trust_tier="admin_only",
            companion_surfaceable=False,
        )
    )
