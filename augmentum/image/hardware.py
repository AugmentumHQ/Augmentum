"""GPU/VRAM detection and hardware tier classification."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


class ModelTier(str, Enum):
    HIGH = "high"       # 12GB+ VRAM — FLUX or large SDXL
    MEDIUM = "medium"   # 6-8GB VRAM — SDXL with Lightning LoRA
    LOW = "low"         # 4GB VRAM — SD 1.5 with LCM-LoRA
    CPU = "cpu"         # No GPU — SD 1.5 with OpenVINO


@dataclass
class HardwareProfile:
    device: str                     # "cuda", "mps" (Apple Silicon), "cpu"
    device_name: str = ""           # e.g. "NVIDIA GeForce RTX 3080"
    vram_total_mb: int = 0
    vram_free_mb: int = 0
    tier: ModelTier = ModelTier.CPU
    recommended_pipeline: str = ""  # "sd15", "sdxl", "flux"
    recommended_model: str = ""     # Default HF repo ID


def detect_hardware(vram_limit: int | None = None) -> HardwareProfile:
    """Detect available hardware and classify into a ModelTier.

    Args:
        vram_limit: Optional VRAM limit override in MB.  When set, the tier
            is computed from this value instead of the detected VRAM.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            # Apple Silicon Metal (MPS) is a real GPU path; check it before
            # falling through to nvidia-smi / CPU. MPS doesn't expose a
            # VRAM API like CUDA — Apple's unified memory means a chunk
            # of system RAM is addressable as VRAM. We estimate ~60% of
            # system RAM as the effective ceiling, which matches what
            # practical workloads see on M-series chips.
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                try:
                    from augmentum.resource import hostmem

                    # Unified memory: this IS the one pool, so the 60%
                    # estimate must come off whatever ceiling actually
                    # applies to us rather than the machine's total.
                    sys_ram_mb = hostmem.total_mib()
                    vram_estimate_mb = int(sys_ram_mb * 0.6)
                except Exception:
                    vram_estimate_mb = 8192  # safe conservative fallback
                effective_vram = vram_limit if vram_limit is not None else vram_estimate_mb
                tier, pipeline, model = _classify_tier(effective_vram)
                log.info(
                    "hardware_detected_mps",
                    device="mps",
                    vram_estimate_mb=vram_estimate_mb,
                    tier=tier.value,
                    note="Apple Silicon Metal — VRAM is a unified-memory estimate, not a hard limit",
                )
                return HardwareProfile(
                    device="mps",
                    device_name="Apple Silicon (Metal)",
                    vram_total_mb=vram_estimate_mb,
                    vram_free_mb=vram_estimate_mb,
                    tier=tier,
                    recommended_pipeline=pipeline,
                    recommended_model=model,
                )
            raise RuntimeError("CUDA not available")

        props = torch.cuda.get_device_properties(0)
        device_name = props.name
        # PyTorch ≥2.10 renamed total_mem → total_memory
        total_bytes = getattr(props, "total_memory", None) or getattr(props, "total_mem", 0)
        vram_total_mb = total_bytes // (1024 * 1024)
        vram_free_mb = (
            total_bytes - torch.cuda.memory_allocated(0)
        ) // (1024 * 1024)

        effective_vram = vram_limit if vram_limit is not None else vram_total_mb

        tier, pipeline, model = _classify_tier(effective_vram)

        profile = HardwareProfile(
            device="cuda",
            device_name=device_name,
            vram_total_mb=vram_total_mb,
            vram_free_mb=vram_free_mb,
            tier=tier,
            recommended_pipeline=pipeline,
            recommended_model=model,
        )
        log.info(
            "hardware_detected",
            device=profile.device,
            device_name=device_name,
            vram_total_mb=vram_total_mb,
            tier=tier.value,
        )
        return profile

    except Exception as exc:
        reason = str(exc) or type(exc).__name__

        # Fallback: try nvidia-smi when torch CUDA fails (driver mismatch, etc.)
        try:
            import subprocess
            out = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if out.returncode == 0 and out.stdout.strip():
                parts = [p.strip() for p in out.stdout.strip().split("\n")[0].split(",")]
                if len(parts) >= 3:
                    device_name = parts[0]
                    vram_total_mb = int(parts[1])
                    vram_free_mb = int(parts[2])
                    effective_vram = vram_limit if vram_limit is not None else vram_total_mb
                    tier, pipeline, model = _classify_tier(effective_vram)
                    log.info("hardware_detected_via_smi",
                             device_name=device_name,
                             vram_total_mb=vram_total_mb,
                             tier=tier.value,
                             note="torch.cuda failed, used nvidia-smi fallback")
                    return HardwareProfile(
                        device="cuda",
                        device_name=device_name,
                        vram_total_mb=vram_total_mb,
                        vram_free_mb=vram_free_mb,
                        tier=tier,
                        recommended_pipeline=pipeline,
                        recommended_model=model,
                    )
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass

        if "No module named" in reason:
            log.warning(
                "hardware_detection_missing_dependency",
                error=reason,
                hint="Install image dependencies: pip install .[image]",
            )
        else:
            log.warning("hardware_detection_fallback", device="cpu", reason=reason)

        # If the operator clearly intended GPU mode (env signal from
        # install scripts / docker compose), surface a separate, more
        # pointed warning so they don't silently get CPU-only image gen
        # and blame the project for being slow. Most common cause:
        # Colima/Docker Desktop on macOS — GPU passthrough isn't
        # supported under either, and the operator may not know.
        if os.environ.get("AUGMENTUM_VARIANT", "").lower() == "gpu":
            log.warning(
                "gpu_requested_but_unavailable",
                hint=(
                    "AUGMENTUM_VARIANT=gpu is set but no CUDA or Apple Silicon "
                    "Metal GPU was detected. On macOS/Colima this is expected — "
                    "Docker GPU passthrough isn't available there. Image "
                    "generation will run on CPU (~10x slower). Set "
                    "AUGMENTUM_VARIANT=cpu in .env to acknowledge and silence "
                    "this warning."
                ),
            )
        return HardwareProfile(
            device="cpu",
            device_name="CPU (GPU detection failed)",
            tier=ModelTier.CPU,
            recommended_pipeline="sd15",
            recommended_model="Lykon/dreamshaper-8",
        )


# Minimum VRAM (MB) needed to load each pipeline type (fp16)
PIPELINE_VRAM_REQUIREMENTS: dict[str, int] = {
    "sd15": 2_000,
    "sdxl": 5_500,
    "flux": 10_000,
}


def refresh_vram_free() -> int:
    """Return current free VRAM in MB (live probe, not cached).

    Returns 0 if CUDA is unavailable.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return 0
        free_bytes, _total = torch.cuda.mem_get_info(0)
        return free_bytes // (1024 * 1024)
    except Exception:
        return 0


def get_system_ram_free_mb() -> int:
    """Return current free system RAM in MB.

    Container-aware via ``hostmem`` (cgroup limit minus our working set),
    falling back to ``/proc/meminfo`` on Linux and ``os.sysconf`` as a last
    resort.  Returns 0 on failure.
    """
    try:
        from augmentum.resource import hostmem

        return hostmem.available_mib()
    except Exception:
        pass

    # Linux fallback — read /proc/meminfo
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024  # kB → MB
    except (OSError, ValueError):
        # Non-Linux platform or malformed /proc/meminfo line — return 0
        # and let the caller treat absent RAM info as "unknown".
        pass

    return 0


def estimate_model_ram_mb(model_path: str) -> int:
    """Estimate RAM required to load a model from its files on disk.

    Returns the total size of weight files (safetensors/bin/gguf) in MB,
    which is a reasonable lower bound for RAM needed during loading.
    Returns 0 if the path doesn't exist or has no weight files.
    """
    import os

    weight_extensions = {".safetensors", ".bin", ".gguf", ".pt", ".pth"}
    total_bytes = 0

    if os.path.isfile(model_path):
        total_bytes = os.path.getsize(model_path)
    elif os.path.isdir(model_path):
        for dirpath, _dirs, files in os.walk(model_path):
            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext in weight_extensions:
                    total_bytes += os.path.getsize(os.path.join(dirpath, f))

    return total_bytes // (1024 * 1024)


def pre_load_safety_check(
    model_path: str,
    pipeline_type: str,
    hw: HardwareProfile,
) -> str | None:
    """Run all safety checks before loading a model.

    Returns an error message string if the load should be blocked,
    or None if it's safe to proceed.

    Checks (in order):
    1. VRAM requirement vs live free VRAM (not stale startup snapshot)
    2. System RAM vs estimated model size on disk
    """
    # 1. Live VRAM check (refresh, don't trust startup snapshot)
    if hw.device == "cuda":
        required_vram = PIPELINE_VRAM_REQUIREMENTS.get(pipeline_type, 0)
        if required_vram:
            live_free = refresh_vram_free()
            if live_free and live_free < required_vram:
                return (
                    f"{pipeline_type.upper()} needs ~{required_vram // 1000}GB VRAM, "
                    f"but only {live_free / 1000:.1f}GB currently free. "
                    f"Close other GPU applications or choose a smaller model."
                )

    # 2. System RAM check — model must fit in RAM during loading
    model_size_mb = estimate_model_ram_mb(model_path)
    if model_size_mb > 0:
        sys_ram_free = get_system_ram_free_mb()
        # Need ~1.5x model size for loading overhead (deserialization buffers)
        required_ram = int(model_size_mb * 1.5)
        if sys_ram_free and sys_ram_free < required_ram:
            return (
                f"Model is ~{model_size_mb / 1000:.1f}GB but only "
                f"{sys_ram_free / 1000:.1f}GB system RAM free. "
                f"Loading would likely cause an out-of-memory crash. "
                f"Free up RAM or use a smaller/quantized model."
            )

    return None


def check_vram_for_pipeline(pipeline_type: str, hw: HardwareProfile) -> str | None:
    """Return an error message if VRAM is insufficient, or None if OK.

    Note: This uses the startup-time VRAM snapshot.  For a live check
    before loading, use ``pre_load_safety_check()`` instead.
    """
    if hw.device == "cpu":
        if pipeline_type != "sd15":
            return f"{pipeline_type.upper()} requires a GPU. Only SD 1.5 works on CPU."
        return None
    required = PIPELINE_VRAM_REQUIREMENTS.get(pipeline_type, 0)
    if required and hw.vram_free_mb < required:
        return (
            f"{pipeline_type.upper()} needs ~{required // 1000}GB VRAM, "
            f"but only {hw.vram_free_mb / 1000:.1f}GB free."
        )
    return None


def _classify_tier(vram_mb: int) -> tuple[ModelTier, str, str]:
    """Classify VRAM amount into tier + recommendations."""
    if vram_mb >= 12_000:
        return (
            ModelTier.HIGH,
            "flux",
            "black-forest-labs/FLUX.1-schnell",
        )
    if vram_mb >= 6_000:
        return (
            ModelTier.MEDIUM,
            "sdxl",
            "stabilityai/stable-diffusion-xl-base-1.0",
        )
    if vram_mb >= 4_000:
        return (
            ModelTier.LOW,
            "sd15",
            "Lykon/dreamshaper-8",
        )
    return (
        ModelTier.CPU,
        "sd15",
        "Lykon/dreamshaper-8",
    )


@dataclass
class CatalogModel:
    """A recommended model available for one-click download."""

    repo_id: str               # HuggingFace repo ID or CivitAI URL
    name: str                  # Short display name
    description: str           # One-line description
    pipeline_type: str         # "sd15", "sdxl", "flux"
    size_gb: float             # Approximate download size in GB
    min_vram_mb: int           # Minimum VRAM in MB (0 = CPU-friendly)
    min_tier: ModelTier        # Lowest tier that can run this model
    cpu_friendly: bool = False # Explicitly recommended for CPU inference
    speed_note: str = ""       # e.g. "~1s/image", "~8s/image on CPU"
    allow_patterns: list[str] | None = None  # Filter for multi-version repos
    # Capability flags — what generation modes this model natively supports
    # Values: "yes" (native), "fallback" (latent-space hack, lower quality), "no" (unsupported)
    cap_txt2img: str = "yes"
    cap_img2img: str = "yes"
    cap_inpaint: str = "fallback"
    # Precision variants: list of {"variant": "fp16"|"fp32"|"bf16", "size_gb": float}
    # For transformers repos with multiple precision options. Empty = auto-detect.
    precision_variants: list[dict] | None = None
    # GGUF-specific: base repo for pipeline components + class names
    gguf_base_repo: str = ""          # e.g. "Qwen/Qwen-Image-2512"
    gguf_pipeline_class: str = ""     # e.g. "QwenImagePipeline"
    gguf_transformer_class: str = ""  # e.g. "QwenImageTransformer2DModel"


RECOMMENDED_MODELS: list[CatalogModel] = [
    # =====================================================================
    # Sorted by peak VRAM usage (lowest → highest) based on community
    # benchmarks.  Within the same VRAM tier, lighter/faster models first.
    # =====================================================================

    # --- CPU / ~2.5 GB VRAM ---
    CatalogModel(
        repo_id="nota-ai/bk-sdm-small",
        name="BK-SDM Small",
        description="Compressed SD 1.5 variant. 2x faster than standard SD 1.5 on CPU with minimal quality loss.",
        pipeline_type="sd15",
        size_gb=1.7,
        min_vram_mb=0,
        min_tier=ModelTier.CPU,
        cpu_friendly=True,
        speed_note="~4s on CPU, <1s on GPU",
        precision_variants=[
            {"variant": "fp16", "size_gb": 0.9, "label": "FP16 — Half precision (recommended)"},
            {"variant": "fp32", "size_gb": 1.7, "label": "FP32 — Full precision"},
        ],
    ),
    # --- CPU / ~3.5 GB VRAM ---
    CatalogModel(
        repo_id="runwayml/stable-diffusion-v1-5",
        name="Stable Diffusion 1.5",
        description="Classic SD model. Works everywhere including CPU. Great ecosystem of LoRAs and fine-tunes.",
        pipeline_type="sd15",
        size_gb=4.3,
        min_vram_mb=0,
        min_tier=ModelTier.CPU,
        cpu_friendly=True,
        speed_note="~8s on CPU, ~1s on GPU",
        precision_variants=[
            {"variant": "fp16", "size_gb": 2.1, "label": "FP16 — Half precision (recommended)"},
            {"variant": "fp32", "size_gb": 4.3, "label": "FP32 — Full precision"},
        ],
    ),
    # --- ~3.5-4 GB VRAM (SD 1.5 fine-tunes, FP16) ---
    CatalogModel(
        repo_id="Lykon/dreamshaper-8",
        name="DreamShaper 8",
        description="Top-rated SD 1.5 fine-tune. Versatile, artistic, great for fantasy and portraits.",
        pipeline_type="sd15",
        size_gb=2.1,
        min_vram_mb=4_000,
        min_tier=ModelTier.LOW,
        speed_note="~1s on GPU",
        precision_variants=[
            {"variant": "fp16", "size_gb": 1.1, "label": "FP16 — Half precision (recommended)"},
            {"variant": "fp32", "size_gb": 2.1, "label": "FP32 — Full precision"},
        ],
    ),
    CatalogModel(
        repo_id="emilianJR/epiCRealism",
        name="epiCRealism",
        description="Photorealistic portraits and faces with exceptional detail at low VRAM. Best SD1.5 model for character headshots and avatar generation.",
        pipeline_type="sd15",
        size_gb=2.1,
        min_vram_mb=4_000,
        min_tier=ModelTier.LOW,
        speed_note="~1s on GPU",
        precision_variants=[
            {"variant": "fp16", "size_gb": 1.1, "label": "FP16 — Half precision (recommended)"},
            {"variant": "fp32", "size_gb": 2.1, "label": "FP32 — Full precision"},
        ],
    ),
    # --- ~5-6 GB VRAM ---
    CatalogModel(
        repo_id="stablediffusionapi/realistic-vision-v51",
        name="Realistic Vision 5.1",
        description="Photorealistic SD 1.5 fine-tune. Best realism at low VRAM. FP32 only — uses ~5-6 GB VRAM.",
        pipeline_type="sd15",
        size_gb=2.1,
        min_vram_mb=4_000,
        min_tier=ModelTier.LOW,
        speed_note="~1s on GPU",
        precision_variants=[
            {"variant": "fp32", "size_gb": 2.1, "label": "FP32 — Full precision (only available format)"},
        ],
    ),
    CatalogModel(
        repo_id="unsloth/FLUX.2-klein-4B-GGUF",
        name="FLUX.2 Klein 4B (GGUF Q4)",
        description="Fast native img2img — 4-step distilled 4B model. Apache 2.0 license. Q4_K_M quantized (~2.4 GB) — fits in 6 GB VRAM.",
        pipeline_type="flux",
        size_gb=2.4,
        min_vram_mb=6_000,
        min_tier=ModelTier.MEDIUM,
        cap_img2img="yes",
        cap_inpaint="no",
        speed_note="<1s on 8GB+ GPU (4 steps)",
        allow_patterns=["*Q4_K_M*"],
        gguf_base_repo="black-forest-labs/FLUX.2-klein-4B",
        gguf_pipeline_class="Flux2KleinPipeline",
        gguf_transformer_class="Flux2Transformer2DModel",
    ),
    # --- ~6-7 GB VRAM ---
    CatalogModel(
        repo_id="Efficient-Large-Model/Sana_600M_1024px_diffusers",
        name="Sana 0.6B",
        description="Ultra-fast 0.6B DiT with 32x compressed latents. Artistic/stylized strength, 1024px native. Gemma2-2B-IT text encoder (the encoder itself is ~5 GB — needs ~8 GB VRAM without offload). NVIDIA NSCL license.",
        pipeline_type="flux",
        size_gb=16.5,
        min_vram_mb=8_000,
        min_tier=ModelTier.MEDIUM,
        speed_note="~1-2s on 8GB GPU (20 steps)",
        cap_img2img="fallback",
        cap_inpaint="fallback",
        precision_variants=[
            {"variant": "fp16", "size_gb": 8.3, "label": "FP16 — Half precision (recommended)"},
            {"variant": "fp32", "size_gb": 16.5, "label": "FP32 — Full precision"},
        ],
    ),
    CatalogModel(
        repo_id="unsloth/Z-Image-Turbo-GGUF",
        name="Z-Image Turbo (GGUF Q4)",
        description="6B param S3-DiT turbo model from Tongyi. 9-step generation with excellent quality. Qwen3 text encoder. Q4_K_M quantized.",
        pipeline_type="flux",
        size_gb=5.5,
        min_vram_mb=6_000,
        min_tier=ModelTier.MEDIUM,
        cap_inpaint="yes",
        speed_note="~3s on 12GB+ GPU (9 steps)",
        allow_patterns=["*Q4_K_M*"],
        gguf_base_repo="Tongyi-MAI/Z-Image-Turbo",
        gguf_pipeline_class="ZImagePipeline",
        gguf_transformer_class="ZImageTransformer2DModel",
    ),
    CatalogModel(
        repo_id="unsloth/Z-Image-GGUF",
        name="Z-Image (GGUF Q4)",
        description="6B param S3-DiT model from Tongyi. Full quality version (30 steps). Qwen3 text encoder. Q4_K_M quantized.",
        pipeline_type="flux",
        size_gb=5.5,
        min_vram_mb=6_000,
        min_tier=ModelTier.MEDIUM,
        cap_inpaint="yes",
        speed_note="~8s on 12GB+ GPU (30 steps)",
        allow_patterns=["*Q4_K_M*"],
        gguf_base_repo="Tongyi-MAI/Z-Image",
        gguf_pipeline_class="ZImagePipeline",
        gguf_transformer_class="ZImageTransformer2DModel",
    ),
    CatalogModel(
        repo_id="SeeSee21/Z-Anime",
        name="Z-Anime (GGUF)",
        description="Anime fine-tune of Tongyi Z-Image. 6B S3-DiT with Qwen3 text encoder, full fine-tune (not LoRA merge). Best for anime/illustration. Pick a quant from the dropdown.",
        pipeline_type="flux",
        size_gb=4.5,
        min_vram_mb=6_000,
        min_tier=ModelTier.MEDIUM,
        cap_inpaint="yes",
        speed_note="~8s on 12GB+ GPU (30 steps)",
        allow_patterns=["gguf/*q4_k_s*"],
        gguf_base_repo="Tongyi-MAI/Z-Image",
        gguf_pipeline_class="ZImagePipeline",
        gguf_transformer_class="ZImageTransformer2DModel",
    ),
    # --- ~7-8 GB VRAM (SDXL family) ---
    CatalogModel(
        repo_id="Lykon/dreamshaper-xl-v2-turbo",
        name="DreamShaper XL Turbo",
        description="Fast SDXL variant with turbo distillation. Only 4-8 steps needed. ~7 GB VRAM with VAE tiling.",
        pipeline_type="sdxl",
        size_gb=6.9,
        min_vram_mb=6_000,
        min_tier=ModelTier.MEDIUM,
        speed_note="~1s on 8GB GPU (4 steps)",
        precision_variants=[
            {"variant": "fp16", "size_gb": 3.5, "label": "FP16 — Half precision (recommended)"},
            {"variant": "fp32", "size_gb": 6.9, "label": "FP32 — Full precision"},
        ],
    ),
    CatalogModel(
        repo_id="cagliostrolab/animagine-xl-4.0",
        name="Animagine XL 4.0",
        description="Top anime SDXL model. 8.4M image dataset, booru tag prompting, excellent anatomy. Use quality tags like 'masterpiece, high score'. Recommended: 28 steps, CFG 5, Euler a.",
        pipeline_type="sdxl",
        size_gb=6.5,
        min_vram_mb=6_000,
        min_tier=ModelTier.MEDIUM,
        speed_note="~3s on 8GB GPU (28 steps)",
        precision_variants=[
            {"variant": "fp16", "size_gb": 6.5, "label": "FP16 — Native precision (ships as FP16)"},
        ],
    ),
    CatalogModel(
        repo_id="playgroundai/playground-v2.5-1024px-aesthetic",
        name="Playground v2.5",
        description="Top aesthetic quality among SDXL models. MJHQ-30K FID 4.48 (vs SDXL 9.55 — ~2.1x lower). Stunning color and contrast.",
        pipeline_type="sdxl",
        size_gb=6.5,
        min_vram_mb=6_000,
        min_tier=ModelTier.MEDIUM,
        speed_note="~5s on 8GB GPU (50 steps)",
        precision_variants=[
            {"variant": "fp16", "size_gb": 3.3, "label": "FP16 — Half precision (recommended)"},
            {"variant": "fp32", "size_gb": 6.5, "label": "FP32 — Full precision"},
        ],
    ),
    CatalogModel(
        repo_id="RunDiffusion/Juggernaut-XL-v9",
        name="Juggernaut XL v9",
        description="Top-rated photorealistic SDXL fine-tune. Excels at portraits, skin detail, cinematic lighting, and character art.",
        pipeline_type="sdxl",
        size_gb=6.5,
        min_vram_mb=6_000,
        min_tier=ModelTier.MEDIUM,
        speed_note="~3s on 8GB GPU",
        precision_variants=[
            {"variant": "fp16", "size_gb": 3.3, "label": "FP16 — Half precision (recommended)"},
            {"variant": "fp32", "size_gb": 6.5, "label": "FP32 — Full precision"},
        ],
    ),
    CatalogModel(
        repo_id="nroggendorff/epicrealismxl",
        name="epiCRealism XL",
        description="Best overall photorealism for SDXL. Exceptional portrait photography, headshots, fashion, and avatar creation.",
        pipeline_type="sdxl",
        size_gb=6.5,
        min_vram_mb=6_000,
        min_tier=ModelTier.MEDIUM,
        speed_note="~3s on 8GB GPU",
        precision_variants=[
            {"variant": "fp16", "size_gb": 3.3, "label": "FP16 — Half precision (recommended)"},
            {"variant": "fp32", "size_gb": 6.5, "label": "FP32 — Full precision"},
        ],
    ),
    CatalogModel(
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        name="SDXL 1.0",
        description="Official SDXL base model. Excellent prompt understanding and detail at 1024x1024.",
        pipeline_type="sdxl",
        size_gb=6.9,
        min_vram_mb=6_000,
        min_tier=ModelTier.MEDIUM,
        speed_note="~3s on 8GB GPU",
        precision_variants=[
            {"variant": "fp16", "size_gb": 3.5, "label": "FP16 — Half precision (recommended)"},
            {"variant": "fp32", "size_gb": 6.9, "label": "FP32 — Full precision"},
        ],
    ),
    CatalogModel(
        repo_id="SG161222/RealVisXL_V4.0",
        name="RealVisXL V4",
        description="Best photorealistic SDXL fine-tune. Stunning realism and natural lighting.",
        pipeline_type="sdxl",
        size_gb=6.9,
        min_vram_mb=6_000,
        min_tier=ModelTier.MEDIUM,
        speed_note="~3s on 8GB GPU",
        precision_variants=[
            {"variant": "fp16", "size_gb": 3.5, "label": "FP16 — Half precision (recommended)"},
            {"variant": "fp32", "size_gb": 6.9, "label": "FP32 — Full precision"},
        ],
    ),
    # --- ~8 GB VRAM ---
    CatalogModel(
        repo_id="stabilityai/stable-diffusion-3.5-medium",
        name="SD 3.5 Medium",
        description="Modern MMDiT architecture with 3 text encoders. Great text rendering and complex prompts. T5 encoder can be dropped to save VRAM. Stability Community License — gated (needs HF token + license accept).",
        pipeline_type="flux",
        size_gb=14.4,
        min_vram_mb=6_000,
        min_tier=ModelTier.MEDIUM,
        cap_inpaint="yes",
        speed_note="~5s on 12GB GPU (40 steps)",
        precision_variants=[
            {"variant": "fp16", "size_gb": 9.7, "label": "FP16 — Half precision (recommended)"},
            {"variant": "bf16", "size_gb": 14.4, "label": "BF16 — Full precision"},
        ],
    ),
    CatalogModel(
        repo_id="PixArt-alpha/PixArt-Sigma-XL-2-1024-MS",
        name="PixArt-Sigma",
        description="Ultra-lightweight 0.6B DiT with T5 text encoder. 4K capable, fast inference, Apache 2.0 license. T5 can be loaded in 8-bit.",
        pipeline_type="flux",
        size_gb=10.7,
        min_vram_mb=6_000,
        min_tier=ModelTier.MEDIUM,
        cap_img2img="no",
        cap_inpaint="no",
        speed_note="~2s on 8GB GPU (20 steps)",
        precision_variants=[
            {"variant": "fp16", "size_gb": 10.7, "label": "FP16 — Native precision (ships as FP16)"},
        ],
    ),
    # --- ~10-12 GB VRAM ---
    CatalogModel(
        repo_id="unsloth/FLUX.2-klein-9B-GGUF",
        name="FLUX.2 Klein 9B (GGUF Q4)",
        description="Higher quality Klein model. Native img2img, 4-step distilled. Apache 2.0 license. Q4_K_M quantized (~5.5 GB). ~10 GB VRAM.",
        pipeline_type="flux",
        size_gb=5.5,
        min_vram_mb=10_000,
        min_tier=ModelTier.MEDIUM,
        cap_img2img="yes",
        cap_inpaint="no",
        speed_note="~2s on 12GB+ GPU (4 steps)",
        allow_patterns=["*Q4_K_M*"],
        gguf_base_repo="black-forest-labs/FLUX.2-klein-9B",
        gguf_pipeline_class="Flux2KleinPipeline",
        gguf_transformer_class="Flux2Transformer2DModel",
    ),
    CatalogModel(
        repo_id="unsloth/FLUX.1-Kontext-dev-GGUF",
        name="FLUX Kontext (GGUF Q4)",
        description="12B image editing model — best with img2img mode. Upload an image and describe changes. Q4_K_M ~6.5 GB. ~10 GB VRAM. Base repo is gated (needs HF token + license accept).",
        pipeline_type="flux",
        size_gb=6.5,
        min_vram_mb=10_000,
        min_tier=ModelTier.MEDIUM,
        cap_img2img="yes",
        cap_inpaint="fallback",
        speed_note="~8s on 12GB+ GPU",
        allow_patterns=["*Q4_K_M*"],
        gguf_base_repo="black-forest-labs/FLUX.1-Kontext-dev",
        gguf_pipeline_class="FluxPipeline",
        gguf_transformer_class="FluxTransformer2DModel",
    ),
    CatalogModel(
        repo_id="unsloth/FLUX.2-dev-GGUF",
        name="FLUX.2 Dev (GGUF Q4)",
        description="32B-param flagship FLUX generation. Improved quality and coherence over FLUX.1. Q4_K_M ~18.6 GB. ~16-20 GB VRAM. Non-commercial license — base repo is gated (needs HF token + license accept).",
        pipeline_type="flux",
        size_gb=18.6,
        min_vram_mb=16_000,
        min_tier=ModelTier.HIGH,
        speed_note="~12s on 24GB GPU (28 steps)",
        allow_patterns=["*Q4_K_M*"],
        gguf_base_repo="black-forest-labs/FLUX.2-dev",
        gguf_pipeline_class="FluxPipeline",
        gguf_transformer_class="FluxTransformer2DModel",
    ),
    CatalogModel(
        repo_id="gpustack/FLUX.1-Fill-dev-GGUF",
        name="FLUX Fill (GGUF Q4)",
        description="Purpose-built inpainting model. Flawless blending, generates only inside mask. Q4_0 quantized. ~10-12 GB VRAM.",
        pipeline_type="flux",
        size_gb=10.2,
        min_vram_mb=10_000,
        min_tier=ModelTier.MEDIUM,
        cap_txt2img="no",
        cap_img2img="no",
        cap_inpaint="yes",
        speed_note="~15s on 16GB+ GPU (50 steps)",
        # Pin the exact file: "*Q4_0*" also matches the repo's separate
        # "pure-Q4_0" quant, so the pull grabbed BOTH ggufs (~20GB, two
        # ambiguous transformers in one dir) instead of the single ~10GB
        # file the catalog advertises.
        allow_patterns=["FLUX.1-Fill-dev-Q4_0.gguf"],
        gguf_base_repo="black-forest-labs/FLUX.1-Fill-dev",
        gguf_pipeline_class="FluxFillPipeline",
        gguf_transformer_class="FluxTransformer2DModel",
    ),
    # --- ~13-16 GB VRAM ---
    CatalogModel(
        repo_id="black-forest-labs/FLUX.2-klein-4B",
        name="FLUX.2 Klein 4B",
        description="Fastest native img2img model. Sub-second inference, 4-step distilled. Apache 2.0 license. Qwen3 text encoder. ~13 GB VRAM.",
        pipeline_type="flux",
        size_gb=23.7,
        min_vram_mb=12_000,
        min_tier=ModelTier.HIGH,
        cap_img2img="yes",
        cap_inpaint="no",
        speed_note="<1s on 12GB+ GPU (4 steps)",
        precision_variants=[
            {"variant": "bf16", "size_gb": 23.7, "label": "BF16 — Native precision (only available format)"},
        ],
    ),
    CatalogModel(
        repo_id="duongve/NetaYume-Lumina-Image-2.0-Diffusers-v40",
        name="NetaYume Lumina v4",
        description="Lumina Image 2.0 fine-tune (v4 latest). High-quality anime/artistic generation with Gemma 2 text encoder. ~14-16 GB VRAM.",
        pipeline_type="flux",
        size_gb=9.9,
        min_vram_mb=12_000,
        min_tier=ModelTier.HIGH,
        speed_note="~3s on 12GB+ GPU",
        precision_variants=[
            {"variant": "bf16", "size_gb": 9.9, "label": "BF16 — Native precision (only available format)"},
        ],
    ),
    # --- ~18 GB VRAM ---
    CatalogModel(
        repo_id="stabilityai/stable-diffusion-3.5-large-turbo",
        name="SD 3.5 Large Turbo",
        description="Turbo-distilled SD3 — only 4 steps needed, no CFG. 8B MMDiT, superb quality. ~18 GB VRAM. Stability Community License — gated (needs HF token + license accept).",
        pipeline_type="flux",
        size_gb=16.5,
        min_vram_mb=16_000,
        min_tier=ModelTier.HIGH,
        cap_inpaint="yes",
        speed_note="~2s on 16GB GPU (4 steps)",
        precision_variants=[
            {"variant": "fp16", "size_gb": 11.1, "label": "FP16 — Half precision (recommended)"},
            {"variant": "bf16", "size_gb": 16.5, "label": "BF16 — Full precision"},
        ],
    ),
    # --- ~22-34 GB VRAM ---
    CatalogModel(
        repo_id="black-forest-labs/FLUX.1-schnell",
        name="FLUX.1 Schnell",
        description="Fastest FLUX.1 model. 4-step generation with excellent quality. Apache 2.0 license, but the HF repo is gated (needs login + contact-sharing acceptance). ~22 GB VRAM.",
        pipeline_type="flux",
        size_gb=23.8,
        min_vram_mb=20_000,
        min_tier=ModelTier.HIGH,
        speed_note="~1.2s on 24GB GPU (4 steps)",
        precision_variants=[
            {"variant": "bf16", "size_gb": 23.8, "label": "BF16 — Native precision (only available format)"},
        ],
    ),
    CatalogModel(
        repo_id="black-forest-labs/FLUX.1-dev",
        name="FLUX.1 Dev",
        description="Highest quality FLUX.1 model. Superior text rendering and coherence. 20-50 steps. ~24-34 GB VRAM peak. Non-commercial license — gated (needs HF token + license accept).",
        pipeline_type="flux",
        size_gb=23.8,
        min_vram_mb=24_000,
        min_tier=ModelTier.HIGH,
        speed_note="~10s on 24GB+ GPU (20 steps)",
        precision_variants=[
            {"variant": "bf16", "size_gb": 23.8, "label": "BF16 — Native precision (only available format)"},
        ],
    ),
]


@dataclass
class CatalogLora:
    """A recommended LoRA adapter available for one-click download."""
    civitai_id: str            # CivitAI model ID (numeric string)
    name: str                  # Short display name
    description: str           # One-line description
    base_model: str            # "sd15", "sdxl", "flux"
    size_mb: float             # Approximate download size in MB
    category: str              # "style", "quality", "realistic", "anime", "concept"
    trigger_words: list[str] = field(default_factory=list)


RECOMMENDED_LORAS: list[CatalogLora] = [
    # =====================================================================
    # SD 1.5 LoRAs — largest ecosystem, most mature
    # Sizes are small (7-20 MB) due to the smaller base model.
    # =====================================================================
    CatalogLora(
        civitai_id="82098",
        name="Add More Details",
        description="Increases or decreases image detail via positive/negative weight. The most popular detail LoRA.",
        base_model="sd15", size_mb=9, category="quality",
    ),
    CatalogLora(
        civitai_id="58390",
        name="Detail Tweaker",
        description="Enhances fine surface detail and textures. Complementary to Add More Details.",
        base_model="sd15", size_mb=10, category="quality",
    ),
    CatalogLora(
        civitai_id="13941",
        name="epi_noiseoffset",
        description="Fixes SD1.5 midtone bias. Enables true blacks, bright whites, and dramatic contrast.",
        base_model="sd15", size_mb=19, category="quality",
    ),
    CatalogLora(
        civitai_id="16014",
        name="Anime Lineart",
        description="Clean lineart and manga illustration with improved anatomy. 20K+ five-star reviews.",
        base_model="sd15", size_mb=18, category="anime",
    ),
    CatalogLora(
        civitai_id="81291",
        name="flat2",
        description="Flat illustration at positive weight, fine detail at negative weight. Dual-purpose slider.",
        base_model="sd15", size_mb=7, category="style",
    ),
    CatalogLora(
        civitai_id="16055",
        name="Colorwater",
        description="Watercolor painting style with translucent washes and soft edges.",
        base_model="sd15", size_mb=18, category="style",
        trigger_words=["colorwater"],
    ),

    # =====================================================================
    # SDXL LoRAs — utility LoRAs ~25 MB, style LoRAs 185-800 MB
    # =====================================================================
    CatalogLora(
        civitai_id="122359",
        name="Detail Tweaker XL",
        description="Bidirectional detail control for SDXL. 40K+ five-star reviews. Most essential SDXL utility.",
        base_model="sdxl", size_mb=25, category="quality",
    ),
    CatalogLora(
        civitai_id="134338",
        name="FaeTastic Details",
        description="Detail, saturation, and visual richness with a magical aesthetic. 24 iterative versions.",
        base_model="sdxl", size_mb=435, category="style",
    ),
    CatalogLora(
        civitai_id="402462",
        name="Detail Slider",
        description="Enhances background and scene detail while preserving subject integrity. Always-on safe.",
        base_model="sdxl", size_mb=25, category="quality",
    ),
    CatalogLora(
        civitai_id="484723",
        name="Watercolor Style XL",
        description="Traditional watercolor aesthetics with bleed and wash effects for SDXL.",
        base_model="sdxl", size_mb=185, category="style",
        trigger_words=["watercolor"],
    ),
    CatalogLora(
        civitai_id="475047",
        name="Aesthetic Enhancer",
        description="General composition, color harmony, and visual appeal boost. Cross-compatible SD/XL/Pony.",
        base_model="sdxl", size_mb=25, category="quality",
        trigger_words=["EnhanceImage"],
    ),
    CatalogLora(
        civitai_id="195519",
        name="LCM-LoRA Weights",
        description="Reduces inference steps from 25-50 down to 4-8. Official Latent Consistency distillation.",
        base_model="sdxl", size_mb=394, category="quality",
    ),

    # =====================================================================
    # Flux LoRAs — larger files (~600 MB) due to bigger architecture
    # =====================================================================
    CatalogLora(
        civitai_id="821668",
        name="Style Enhancing",
        description="Trained on curated top CivitAI images. Brighter, more colorful, more detailed. 4.98/5 rating.",
        base_model="flux", size_mb=600, category="style",
    ),
    CatalogLora(
        civitai_id="659540",
        name="Add Details Flux",
        description="Adds fine detail and surface texture. The Flux equivalent of the legendary SD1.5 detail LoRA.",
        base_model="flux", size_mb=600, category="quality",
    ),
    CatalogLora(
        civitai_id="703451",
        name="Illustration Flux",
        description="Illustration, comic, cartoon, manga, and drawing styles. 5 sub-styles in one LoRA.",
        base_model="flux", size_mb=600, category="style",
    ),
    CatalogLora(
        civitai_id="730373",
        name="Hyper Realism",
        description="Pushes Flux toward photographic hyper-realism. By well-known creator aidma.",
        base_model="flux", size_mb=600, category="realistic",
    ),
    CatalogLora(
        civitai_id="890914",
        name="Detail + Photorealism",
        description="Incremental detail/realism slider with 0.0-2.0 weight range. Universal quality dial.",
        base_model="flux", size_mb=600, category="quality",
    ),
    CatalogLora(
        civitai_id="840424",
        name="Watercolor Flux",
        description="Traditional watercolor aesthetics tuned for Flux. Fills the style gap vs photorealistic default.",
        base_model="flux", size_mb=600, category="style",
    ),
]


def get_lora_catalog(installed_bases: list[str] | None = None) -> list[dict]:
    """Return LoRA catalog sorted by compatibility with installed base models.

    LoRAs matching installed bases appear first, followed by others.
    """
    def sort_key(lora: CatalogLora) -> tuple[int, str, str]:
        compatible = 0 if installed_bases and lora.base_model in installed_bases else 1
        return (compatible, lora.base_model, lora.name)

    sorted_loras = sorted(RECOMMENDED_LORAS, key=sort_key)
    return [
        {
            "civitai_id": l.civitai_id,
            "name": l.name,
            "description": l.description,
            "base_model": l.base_model,
            "size_mb": l.size_mb,
            "category": l.category,
            "trigger_words": l.trigger_words,
            "compatible": bool(installed_bases and l.base_model in installed_bases),
        }
        for l in sorted_loras
    ]


def get_catalog_for_tier(tier: ModelTier) -> list[CatalogModel]:
    """Return catalog models annotated with compatibility for the given tier.

    All models are returned (so users can see what's available with
    better hardware), sorted with compatible models first.
    """
    tier_order = {ModelTier.CPU: 0, ModelTier.LOW: 1, ModelTier.MEDIUM: 2, ModelTier.HIGH: 3}
    user_rank = tier_order.get(tier, 0)

    def sort_key(m: CatalogModel) -> tuple[int, int, str]:
        model_rank = tier_order.get(m.min_tier, 0)
        compatible = 0 if model_rank <= user_rank else 1
        return (compatible, model_rank, m.name)

    return sorted(RECOMMENDED_MODELS, key=sort_key)


def get_gguf_families() -> list[dict]:
    """Return deduplicated GGUF family presets derived from the catalog.

    Each unique ``(base_repo, pipeline_class, transformer_class)`` triple in
    ``RECOMMENDED_MODELS`` becomes one family entry — that triple is what the
    GGUF loader needs to instantiate the right diffusers pipeline. Custom
    imports use these to auto-fill the three metadata fields without the user
    typing them.

    ``name_patterns`` aggregates filename hints from every catalog entry that
    targets the same base_repo, so a file like ``z-anime-base-q4_k_s.gguf``
    can match the Z-Image-Base family even though the catalog entry for it
    has the SeeSee21/Z-Anime repo_id.
    """
    by_key: dict[tuple[str, str, str], dict] = {}
    for cm in RECOMMENDED_MODELS:
        if not cm.gguf_base_repo:
            continue
        key = (cm.gguf_base_repo, cm.gguf_pipeline_class, cm.gguf_transformer_class)
        # Pull filename hints from both the repo_id leaf and the catalog name.
        repo_leaf = cm.repo_id.split("/")[-1].lower()
        base_leaf = cm.gguf_base_repo.split("/")[-1].lower()
        candidate_patterns = {
            repo_leaf, repo_leaf.replace("-gguf", "").replace("_gguf", ""),
            base_leaf, base_leaf.replace("-", "_"),
        }
        entry = by_key.setdefault(key, {
            "family": base_leaf,
            "base_repo": cm.gguf_base_repo,
            "pipeline_class": cm.gguf_pipeline_class,
            "transformer_class": cm.gguf_transformer_class,
            "pipeline_type": cm.pipeline_type,
            "name_patterns": [],
        })
        for pat in candidate_patterns:
            pat = pat.strip()
            if pat and pat not in entry["name_patterns"]:
                entry["name_patterns"].append(pat)
    return list(by_key.values())
