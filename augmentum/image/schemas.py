"""Pydantic request/response models for the image generation API."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator


class AspectRatio(str, Enum):
    PORTRAIT = "portrait"
    LANDSCAPE = "landscape"
    SQUARE = "square"


class JobType(str, Enum):
    TXT2IMG = "txt2img"
    IMG2IMG = "img2img"
    INPAINT = "inpaint"


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelineType(str, Enum):
    SD15 = "sd15"
    SDXL = "sdxl"
    FLUX = "flux"


# --- Request models ---


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10000)
    negative_prompt: str = ""
    model: str = ""
    preset: str = ""
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    cfg_scale: float | None = None
    seed: int = -1
    loras: list[LoraWeight] | None = None
    aspect: AspectRatio = AspectRatio.SQUARE
    sampler: str | None = None
    scheduler: str | None = None
    condense_model: str = ""
    enhance_prompt: bool = True    # Whether to run LLM prompt enhancement
    condense_prompt: bool = True   # Whether to condense prompt if over token limit
    # Per-generation quality optimizations (None = fall through to config default)
    guidance_rescale: float | None = None
    hires_fix: bool | None = None
    hires_scale: float | None = None
    hires_denoise: float | None = None
    # CLIP skip — skip last N layers of text encoder (SD1.5/SDXL, not FLUX)
    clip_skip: int | None = None
    # IP-Adapter reference image(s) for visual consistency.
    # Single string OR list of strings. Each can be base64, file path, or /api/image/<id>.
    # Multiple images blend identity features (useful for group scenes).
    ip_adapter_image: str | list[str] = ""
    ip_adapter_scale: float = Field(default=0.55, ge=0.0, le=1.0)
    # Provenance, not silos: the architect-dispatched client path
    # (image.generate channel -> generateFromArchitect) marks its
    # generations 'companion'. Anything else is coerced to '' at the
    # route so a client can't invent origins.
    origin: str = ""


class LoraWeight(BaseModel):
    name: str
    weight: float = Field(default=1.0, ge=0.0, le=2.0)


# Fix forward reference for GenerateRequest.loras
GenerateRequest.model_rebuild()


class Img2ImgRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10000)
    negative_prompt: str = ""
    model: str = ""
    source_image: str = Field(..., description="Base64-encoded source image or image_id")
    strength: float = Field(default=0.75, ge=0.0, le=1.0)
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    cfg_scale: float | None = None
    seed: int = -1
    sampler: str | None = None
    preset: str = ""
    condense_model: str = ""
    enhance_prompt: bool = True    # Whether to run LLM prompt enhancement
    condense_prompt: bool = True   # Whether to condense prompt if over token limit


class InpaintRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10000)
    negative_prompt: str = ""
    model: str = ""
    source_image: str = Field(..., description="Base64-encoded source image or image_id")
    mask_image: str = Field(..., description="Base64-encoded mask (white = repaint)")
    strength: float = Field(default=1.0, ge=0.0, le=1.0)
    mask_blur: int = Field(default=4, ge=0, le=20)
    inpaint_mode: str = Field(default="default", pattern="^(default|improve|modify)$")
    width: int | None = None
    height: int | None = None
    steps: int | None = None
    cfg_scale: float | None = None
    seed: int = -1
    sampler: str | None = None
    preset: str = ""
    condense_model: str = ""
    enhance_prompt: bool = True    # Whether to run LLM prompt enhancement
    condense_prompt: bool = True   # Whether to condense prompt if over token limit
    inpaint_full_res: bool = False
    inpaint_padding: int = Field(default=32, ge=0, le=256)


class ModelPullRequest(BaseModel):
    source: str = Field(..., min_length=1)
    name: str = ""
    pipeline_type: PipelineType | None = None
    allow_patterns: list[str] | None = None
    variant: str = ""  # "fp16", "fp32", "bf16" — overrides auto-detection
    asset_type: str = ""  # "lora" routes download to loras/ subdirectory
    trigger_words: list[str] = []  # saved as companion JSON for LoRAs
    base_model: str = ""  # "sd15", "sdxl", "flux" — for LoRA compatibility checking


class ModelDeleteRequest(BaseModel):
    name: str


# --- Response models ---


class GenerateResponse(BaseModel):
    image_id: str
    job_id: str
    status: JobStatus
    url: str = ""
    seed: int = -1
    prompt: str = ""
    negative_prompt: str = ""
    width: int = 0
    height: int = 0
    steps: int = 0
    model: str = ""


class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    position: int = 0
    result: GenerateResponse | None = None
    error: str = ""


class ModelCapabilities(BaseModel):
    """What generation modes a model natively supports."""

    txt2img: str = "yes"       # "yes" | "fallback" | "no"
    img2img: str = "yes"       # "yes" | "fallback" | "no"
    inpaint: str = "fallback"  # "yes" | "fallback" | "no"


class ModelInfo(BaseModel):
    name: str
    pipeline_type: PipelineType
    path: str = ""
    size_bytes: int = 0
    source: str = ""
    is_loaded: bool = False
    distilled_type: str = ""
    capabilities: ModelCapabilities = ModelCapabilities()
    recommended_steps: int | None = None
    recommended_cfg: float | None = None
    # Phase 8 — only populated for source=="peer" entries (fabric-hosted
    # image models). The UI uses these to decorate the dropdown item;
    # local + cloud models carry "" for both.
    peer_icon: str = ""
    peer_hostname: str = ""


class HardwareInfo(BaseModel):
    device: str
    device_name: str = ""
    vram_total_mb: int = 0
    vram_free_mb: int = 0
    tier: str = ""
    recommended_pipeline: str = ""
    recommended_model: str = ""


class HistoryEntry(BaseModel):
    image_id: str
    prompt: str
    negative_prompt: str = ""
    model: str = ""
    seed: int = -1
    width: int = 0
    height: int = 0
    steps: int = 0
    cfg_scale: float = 0.0
    preset: str = ""
    loras: list[dict] = []
    created_at: str = ""
    url: str = ""
    job_type: str = "txt2img"
    strength: float = 1.0
    source_image_id: str = ""
    is_private: bool = False
    is_background: bool = False
    # Provenance: 'companion' when she generated it, '' = user.
    origin: str = ""


class HistoryPage(BaseModel):
    total: int
    entries: list[HistoryEntry]


class BatchDeleteRequest(BaseModel):
    image_ids: list[str]


class LoraInfo(BaseModel):
    name: str
    path: str = ""
    trigger_words: list[str] = []
    size_bytes: int = 0
    base_model: str = ""  # "sd15", "sdxl", "flux", "" (unknown)


class SamplerInfo(BaseModel):
    name: str
    display_name: str
    aliases: list[str] = []


class CatalogModelInfo(BaseModel):
    repo_id: str
    name: str
    description: str = ""
    pipeline_type: str = "sd15"
    size_gb: float = 0.0
    min_vram_mb: int = 0
    min_tier: str = "cpu"
    cpu_friendly: bool = False
    speed_note: str = ""
    compatible: bool = True
    installed: bool = False
    installed_name: str = ""  # Filesystem name for delete operations
    allow_patterns: list[str] | None = None
    capabilities: ModelCapabilities = ModelCapabilities()
    precision_variants: list[dict] | None = None  # [{"variant":"fp16","size_gb":1.1,"label":"..."}]


# --- Custom Import (file/folder/path) ---

class GgufFamily(BaseModel):
    """A known GGUF model family — feeds the family-preset dropdown in the upload UI.

    Derived from RECOMMENDED_MODELS entries that declare gguf_base_repo so the
    catalog is the single source of truth for available presets.
    """
    family: str                   # Display label, e.g. "Z-Image / Z-Anime"
    base_repo: str                # HF repo with pipeline scaffolding
    pipeline_class: str           # diffusers class, e.g. "ZImagePipeline"
    transformer_class: str        # diffusers class, e.g. "ZImageTransformer2DModel"
    pipeline_type: str = "flux"   # bucket for VRAM/feature gating
    name_patterns: list[str] = [] # filename substrings used by inspect to auto-suggest this family


class InspectResponse(BaseModel):
    """Result of POST /api/image/models/inspect — describes a candidate import."""
    kind: str                              # gguf | safetensors-single | diffusers-zip | diffusers-folder
    suggested_name: str                    # Derived from filename, sanitised
    suggested_family: str = ""             # Empty if not a GGUF or unknown family
    suggested_pipeline_type: str = ""      # Empty if not derivable pre-install
    size_bytes: int = 0
    warnings: list[str] = []               # Non-fatal notes (e.g. "GGUF header unreadable, falling back to filename match")


class ImportRequest(BaseModel):
    """Body for POST /api/image/models/import (form-encoded alongside the file upload).

    Either ``path`` (server-side import) or an uploaded file must be supplied.
    For GGUFs, EITHER ``gguf_family`` (preset key from /api/image/gguf-families)
    OR all three raw fields must be supplied. The family preset wins if both are set.
    """
    name: str                              # Target dir name under image_models/
    kind: str                              # gguf | safetensors-single | diffusers-zip
    path: str = ""                         # Optional: server-side path under image_imports_dir
    gguf_family: str = ""                  # Preset key (e.g. "Z-Image / Z-Anime")
    gguf_base_repo: str = ""               # Raw override (used if no family set)
    gguf_pipeline_class: str = ""
    gguf_transformer_class: str = ""
    prefetch_base_components: bool = True  # Pull text encoder/VAE/scheduler from base_repo after install


# --- OpenAI Images API models ---


class OpenAIImageRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    model: str = ""
    n: int = Field(default=1, ge=1, le=4)
    size: str = "1024x1024"
    quality: str = "standard"
    style: str = "vivid"
    response_format: str = "url"
    user: str = ""

    # Augmentum extensions — SD-specific params that power users and
    # Open WebUI's IMAGES_OPENAI_API_PARAMS can inject
    negative_prompt: str = ""
    steps: int | None = None
    cfg_scale: float | None = None
    seed: int = -1
    sampler: str | None = None
    scheduler: str | None = None

    # Cloud provider routing — set to use a configured cloud provider instead of local GPU
    provider: str = ""  # Provider ID (e.g. "openai", "stability", "together", "fal", "bfl")

    # LLM prompt enhancement — rewrite the prompt to be more effective for the target model
    enhance_prompt: bool = False

    # Accept unknown fields (gpt-image-1 fields like background,
    # output_format, moderation, or anything from IMAGES_OPENAI_API_PARAMS)
    model_config = {"extra": "allow"}

    @field_validator("size")
    @classmethod
    def validate_size(cls, v: str) -> str:
        if v.lower() == "auto":
            return v
        try:
            w_str, h_str = v.lower().split("x")
            w, h = int(w_str), int(h_str)
        except (ValueError, AttributeError):
            raise ValueError(f"Invalid size format '{v}', expected WxH (e.g. 1024x1024)") from None
        if not (256 <= w <= 2048 and 256 <= h <= 2048):
            raise ValueError(f"Width and height must be between 256 and 2048, got {w}x{h}")
        if w % 8 != 0 or h % 8 != 0:
            raise ValueError(f"Width and height must be multiples of 8, got {w}x{h}")
        return v

    @field_validator("quality")
    @classmethod
    def validate_quality(cls, v: str) -> str:
        # "standard"/"hd" = DALL-E 3; "low"/"medium"/"high"/"auto" = gpt-image-1
        allowed = ("standard", "hd", "low", "medium", "high", "auto")
        if v not in allowed:
            raise ValueError(f"quality must be one of {allowed}, got '{v}'")
        return v

    @field_validator("style")
    @classmethod
    def validate_style(cls, v: str) -> str:
        if v not in ("vivid", "natural"):
            raise ValueError(f"style must be 'vivid' or 'natural', got '{v}'")
        return v

    @field_validator("response_format")
    @classmethod
    def validate_response_format(cls, v: str) -> str:
        if v not in ("url", "b64_json"):
            raise ValueError(f"response_format must be 'url' or 'b64_json', got '{v}'")
        return v


class OpenAIImageData(BaseModel):
    url: str | None = None
    b64_json: str | None = None
    revised_prompt: str | None = None


class OpenAIImageResponse(BaseModel):
    created: int
    data: list[OpenAIImageData]
