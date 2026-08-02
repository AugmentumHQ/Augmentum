"""Abstract ImagePipeline and concrete implementations for SD1.5, SDXL, and FLUX."""

from __future__ import annotations

import asyncio
import os
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from augmentum.image.schemas import PipelineType
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Approximate VRAM requirements per pipeline type (MB).
# Used by auto CPU-offload to decide whether the model fits in GPU memory.
_VRAM_REQUIREMENTS: dict[str, int] = {
    "sd15": 4_000,
    "sdxl": 6_000,
    "flux": 12_000,
}


@dataclass
class GenerationResult:
    """Result of a single image generation."""

    image_id: str = ""
    file_path: str = ""
    seed: int = -1
    width: int = 0
    height: int = 0


class ImagePipeline(ABC):
    """Abstract base for image generation pipelines."""

    @property
    @abstractmethod
    def pipeline_type(self) -> PipelineType: ...

    @property
    @abstractmethod
    def is_loaded(self) -> bool: ...

    @property
    @abstractmethod
    def model_name(self) -> str: ...

    @abstractmethod
    async def load(self, model_path: str, device: str = "cuda", dtype: str = "fp16") -> None:
        """Load a model from disk into GPU/CPU memory."""
        ...

    @abstractmethod
    async def unload(self) -> None:
        """Unload the model and free memory."""
        ...

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        cfg_scale: float = 7.0,
        seed: int = -1,
        sampler: str = "",
        output_dir: str = "/data/image_output",
        ip_adapter_image=None,
        ip_adapter_scale: float = 0.55,
    ) -> GenerationResult:
        """Generate an image and save to output_dir. Returns GenerationResult."""
        ...

    @abstractmethod
    async def img2img(
        self,
        prompt: str,
        image,  # PIL.Image.Image
        negative_prompt: str = "",
        strength: float = 0.75,
        steps: int = 20,
        cfg_scale: float = 7.0,
        seed: int = -1,
        sampler: str = "",
        output_dir: str = "/data/image_output",
    ) -> GenerationResult:
        """Run img2img on a source image. Returns GenerationResult."""
        ...

    @abstractmethod
    async def inpaint(
        self,
        prompt: str,
        image,  # PIL.Image.Image
        mask,   # PIL.Image.Image (white = repaint)
        negative_prompt: str = "",
        strength: float = 1.0,
        steps: int = 20,
        cfg_scale: float = 7.0,
        seed: int = -1,
        sampler: str = "",
        output_dir: str = "/data/image_output",
    ) -> GenerationResult:
        """Run inpainting on a source image with a mask. Returns GenerationResult."""
        ...

    @abstractmethod
    async def load_lora(self, lora_path: str, weight: float = 1.0) -> None:
        """Load a LoRA adapter into the current pipeline."""
        ...

    @abstractmethod
    async def unload_loras(self) -> None:
        """Unload all LoRA adapters."""
        ...


def _get_torch_dtype(dtype_str: str):
    """Convert string dtype to torch dtype, with auto-detection.

    When dtype_str is ``"auto"`` and CUDA is available, prefers bfloat16 on
    hardware that supports it (Ampere+).  This avoids NaN / black-image
    issues that fp16 can cause on FLUX with Ampere+ GPUs while matching
    fp16 speed.
    """
    import torch

    if dtype_str == "auto":
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
            return torch.float16
        return torch.float32
    return {
        "fp16": torch.float16,
        "fp32": torch.float32,
        "bf16": torch.bfloat16,
    }.get(dtype_str, torch.float32)


def _apply_pipeline_optimizations(
    pipe,
    device: str,
    cpu_offload: str = "auto",
    pipeline_key: str = "sd15",
):
    """Apply performance optimizations and handle device placement.

    This replaces the old ``pipe.to(device)`` call.  Depending on
    *cpu_offload* and available VRAM it either moves the whole model to the
    device or enables model-level CPU offloading so that individual
    sub-models are moved to GPU only when needed.

    Optimizations applied:
    * VAE tiling + slicing (prevents OOM at high res)
    * ``channels_last`` memory format on CUDA (faster convolutions)
    * Fused QKV projections (~10 % speed-up where available)
    * TF32 math on Ampere+ GPUs
    * Attention slicing as last-resort on very low VRAM (<4 GB)

    Returns the (possibly mutated) *pipe* so callers can write
    ``pipe = _apply_pipeline_optimizations(pipe, ...)``.
    """
    import torch

    # --- VAE tiling + slicing (always safe, reduces peak VRAM) -----------
    if hasattr(pipe, "vae"):
        if hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()
        if hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()

    # --- Device placement / CPU offload ----------------------------------
    use_offload = False
    very_low_vram = False

    if device == "cuda":
        if cpu_offload == "always":
            use_offload = True
        elif cpu_offload == "auto":
            # Check available VRAM against requirement
            try:
                free_mb = (
                    torch.cuda.mem_get_info()[0] // (1024 * 1024)
                )
                required_mb = _VRAM_REQUIREMENTS.get(pipeline_key, 4_000)
                if free_mb < required_mb * 1.3:
                    use_offload = True
                    log.info(
                        "cpu_offload_auto_enabled",
                        free_mb=free_mb,
                        required_mb=required_mb,
                        pipeline=pipeline_key,
                    )
                if free_mb < 4_000:
                    very_low_vram = True
            except Exception as exc:
                # If VRAM detection fails, fall back to normal .to(device)
                log.debug("vram_detection_failed", error=str(exc))
        # cpu_offload == "never" -> use_offload stays False

    if use_offload and device == "cuda":
        # Try group offloading first (block-level CPU/GPU swap with CUDA
        # stream overlap) for tighter VRAM control.  Falls back to
        # model-level offload if group offloading isn't available or fails.
        # Note: group offloading is NOT compatible with sequential CPU offload.
        group_offloaded = False
        if hasattr(pipe, "transformer"):
            try:
                from diffusers.hooks import apply_group_offloading
                # Use ondemand offloading — moves individual blocks to GPU
                # only during their forward pass, then immediately back to CPU.
                # CUDA stream overlap hides most of the transfer latency.
                apply_group_offloading(
                    pipe.transformer,
                    offload_type="leaf_level",
                    onload_device=torch.device("cuda"),
                    offload_device=torch.device("cpu"),
                    use_stream=torch.cuda.is_available(),
                )
                group_offloaded = True
                log.info("group_offload_enabled", component="transformer", type="leaf_level")
            except (ImportError, Exception) as exc:
                log.debug("group_offload_failed", error=str(exc))

        if group_offloaded:
            # VAE is small (~160MB) — move to GPU permanently.
            vae = getattr(pipe, "vae", None)
            if vae is not None and hasattr(vae, "to"):
                try:
                    vae.to(device)
                except (AttributeError, RuntimeError) as exc:
                    log.debug("vae_to_device_failed", device=device, error=str(exc))

            # Text encoders can be huge (e.g. Qwen2.5-VL-7B = ~14GB in fp16).
            # Apply group offloading so they only move to GPU during encoding,
            # then immediately return to CPU — avoids pinning them in VRAM
            # alongside the transformer during denoising.
            for name in ("text_encoder", "text_encoder_2", "text_encoder_3"):
                comp = getattr(pipe, name, None)
                if comp is None or not hasattr(comp, "parameters"):
                    continue
                try:
                    from diffusers.hooks import apply_group_offloading
                    apply_group_offloading(
                        comp,
                        offload_type="leaf_level",
                        onload_device=torch.device("cuda"),
                        offload_device=torch.device("cpu"),
                        use_stream=torch.cuda.is_available(),
                    )
                    log.info("group_offload_enabled", component=name, type="leaf_level")
                except Exception:
                    # Fallback: move to GPU (some encoders don't support hooks)
                    try:
                        comp.to(device)
                    except (AttributeError, RuntimeError) as exc:
                        log.debug(
                            "text_encoder_to_device_fallback_failed",
                            component=name,
                            error=str(exc),
                        )
        elif hasattr(pipe, "enable_model_cpu_offload"):
            pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)

    # --- xformers memory-efficient attention --------------------------------
    # xformers replaces standard attention with a fused CUDA kernel that
    # reduces VRAM by ~20-40% and is faster.  Mutually exclusive with
    # attention slicing (xformers is strictly better when available).
    # Transformer-based pipelines that pass custom kwargs to their attention
    # processor (``image_rotary_emb``, ``freqs_cis``, joint attention masks,
    # etc.) break under ``enable_xformers_memory_efficient_attention()``
    # because ``XFormersAttnProcessor.__call__`` ends with ``*args, **kwargs``
    # and silently swallows them — the rotary positional state never makes it
    # into the attention, and shapes blow up downstream.
    #
    # Confirmed by inspecting installed diffusers' processor signatures:
    #   FluxAttnProcessor.__call__:    (..., image_rotary_emb)
    #   XFormersAttnProcessor.__call__: (..., *args, **kwargs)
    # so the FluxAttention forward call site (``processor(self, ...,
    # image_rotary_emb=...)``) routes ``image_rotary_emb`` into ``**kwargs``
    # where it is ignored. Same story for SD3 (joint_attention_kwargs),
    # Lumina, QwenImage, and ZImage.
    #
    # Concrete failure for ZImagePipeline:
    #   "cross_attention_kwargs ['freqs_cis'] are not expected by
    #   XFormersAttnProcessor and will be ignored"
    #   → ``RuntimeError: The expanded size of the tensor (128) must match
    #     the existing size (60) at non-singleton dimension 1``.
    #
    # These pipelines fall back to torch's native SDPA, which honors the
    # extra kwargs and is also fast on Ampere+. Substring match because
    # diffusers ships variants (FluxFillPipeline, Flux2KleinPipeline, etc.)
    # that all need the same treatment.
    _XFORMERS_INCOMPATIBLE = (
        "Lumina",
        "QwenImage",
        "ZImage",
        "Flux",              # FluxPipeline, FluxFillPipeline, Flux2KleinPipeline, Flux2Pipeline
        "StableDiffusion3",  # SD3.5 family — joint_attention_kwargs
    )
    xformers_enabled = False
    pipe_cls_name = type(pipe).__name__
    if device == "cuda" and hasattr(pipe, "enable_xformers_memory_efficient_attention"):
        if any(tag in pipe_cls_name for tag in _XFORMERS_INCOMPATIBLE):
            log.info("xformers_skipped", pipeline=pipe_cls_name, reason="incompatible architecture")
        else:
            try:
                pipe.enable_xformers_memory_efficient_attention()
                xformers_enabled = True
                log.info("xformers_enabled", pipeline=pipeline_key)
            except Exception as exc:
                log.debug("xformers_unavailable", error=str(exc))

    # --- Attention slicing for very low VRAM (<4 GB) ---------------------
    # Only used as fallback when xformers is not available.
    if not xformers_enabled and very_low_vram and hasattr(pipe, "enable_attention_slicing"):
        pipe.enable_attention_slicing("auto")
        log.info("attention_slicing_enabled", pipeline=pipeline_key)
    # Architectures where xformers is incompatible (Lumina, QwenImage) still
    # benefit from slicing — recovers most of the memory/perf loss from the
    # xformers skip without dropping custom cross_attention_kwargs.
    elif (
        not xformers_enabled
        and device == "cuda"
        and any(tag in pipe_cls_name for tag in _XFORMERS_INCOMPATIBLE)
        and hasattr(pipe, "enable_attention_slicing")
    ):
        pipe.enable_attention_slicing("auto")
        log.info("attention_slicing_enabled", pipeline=pipeline_key, reason="xformers_incompatible")

    # --- channels_last memory format (CUDA only) -------------------------
    # Skip when offloading: accelerate hooks leave parameters as meta/CPU
    # placeholders between forwards, so Module.to() walks them and fails
    # with "Cannot copy out of meta tensor". The layout change would also
    # be undone by the streaming hooks anyway.
    if device == "cuda" and not use_offload:
        if hasattr(pipe, "unet"):
            pipe.unet.to(memory_format=torch.channels_last)
        if hasattr(pipe, "transformer"):
            pipe.transformer.to(memory_format=torch.channels_last)

    # --- Fuse QKV projections (~10% speedup) -----------------------------
    # Same reason as channels_last: QKV fusion rewrites parameter tensors
    # and is unsafe on meta/offloaded modules.
    if not use_offload and hasattr(pipe, "fuse_qkv_projections"):
        try:
            pipe.fuse_qkv_projections()
        except Exception as exc:
            log.debug("qkv_fusion_skipped", error=str(exc))

    # --- TF32 on Ampere+ (compute capability >= 8.0) ---------------------
    if torch.cuda.is_available():
        try:
            major, _minor = torch.cuda.get_device_capability()
            if major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                log.info("tf32_enabled", compute_capability=f"{major}.{_minor}")
        except RuntimeError as exc:
            # CUDA device-capability query can fail on driver issues —
            # skip TF32 opt-in rather than abort pipeline setup.
            log.debug("tf32_capability_probe_failed", error=str(exc))

    # --- Default scheduler: DPM++ 2M Karras for UNet models ---------------
    # Only set when the user hasn't explicitly selected a sampler.
    # Transformer models (FLUX/SD3) keep their flow-matching schedulers.
    try:
        from augmentum.config import settings as _cfg
    except Exception:
        _cfg = None
    if _cfg is not None and hasattr(pipe, "unet") and not hasattr(pipe, "transformer"):
        try:
            from diffusers import DPMSolverMultistepScheduler

            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config,
                algorithm_type="dpmsolver++",
                solver_order=2,
                use_karras_sigmas=True,
            )
            log.info("default_scheduler_set", scheduler="DPM++ 2M Karras", pipeline=pipeline_key)
        except Exception:
            log.debug("default_scheduler_failed", exc_info=True)

    # --- FreeU (rebalance UNet skip connections) ---------------------------
    if _cfg is not None and _cfg.image_freeu_enabled:
        _apply_freeu(pipe, pipeline_key)

    # --- torch.compile (JIT compilation for ~30% speedup) -----------------
    compile_mode = getattr(_cfg, "image_torch_compile", "off") if _cfg else "off"
    if compile_mode == "auto" or compile_mode == "on" or compile_mode is True:
        _apply_torch_compile(pipe, device, force=(compile_mode == "on"))

    # --- Token Merging (ToMe) for 20-40% speedup -------------------------
    if _cfg is not None and _cfg.image_tome_enabled:
        _apply_tome(pipe, _cfg.image_tome_ratio)

    return pipe


def _apply_freeu(pipe, pipeline_key: str) -> None:
    """Apply FreeU to rebalance UNet skip connections. SD1.5/SDXL only."""
    if not hasattr(pipe, "enable_freeu"):
        log.info("freeu_skipped", pipeline=pipeline_key, reason="no UNet (transformer model)")
        return
    try:
        if pipeline_key == "sdxl":
            pipe.enable_freeu(s1=0.6, s2=0.4, b1=1.1, b2=1.2)
        else:
            # SD1.5 and other UNet models
            pipe.enable_freeu(s1=0.9, s2=0.2, b1=1.2, b2=1.4)
        log.info("freeu_enabled", pipeline=pipeline_key)
    except Exception as exc:
        log.warning("freeu_failed", error=str(exc))


def _apply_torch_compile(pipe, device: str, *, force: bool = False) -> None:
    """JIT-compile the denoising model for faster inference.

    In ``"auto"`` mode (force=False), silently skips if requirements aren't
    met (no CUDA, pre-Ampere, no gcc).  In ``"on"`` mode (force=True), logs
    warnings instead of silently skipping.
    """
    import torch

    if device != "cuda":
        if force:
            log.warning("torch_compile_skipped", reason="not CUDA device")
        return
    try:
        major, minor = torch.cuda.get_device_capability()
        if major < 8:
            if force:
                log.warning(
                    "torch_compile_skipped",
                    reason=f"compute capability {major}.{minor} < 8.0 (Ampere+ required)",
                )
            return
    except Exception:
        if force:
            log.warning("torch_compile_skipped", reason="could not detect compute capability")
        return

    # torch.compile with reduce-overhead uses Triton, which requires a C
    # compiler at first inference time.  Check upfront so we don't crash
    # mid-generation with a confusing "Failed to find C compiler" error.
    import shutil

    if not shutil.which("gcc") and not shutil.which("cc"):
        if force:
            log.warning(
                "torch_compile_skipped",
                reason="no C compiler found (gcc/cc) — required by Triton JIT. "
                       "Install gcc in the container: apt-get install -y gcc",
            )
        return

    target = getattr(pipe, "unet", None) or getattr(pipe, "transformer", None)
    if target is None:
        log.info("torch_compile_skipped", reason="no unet or transformer found")
        return

    try:
        compiled = torch.compile(target, mode="reduce-overhead")
        if hasattr(pipe, "unet"):
            pipe.unet = compiled
        else:
            pipe.transformer = compiled
        log.info("torch_compile_enabled", target=type(target).__name__)
    except Exception as exc:
        log.warning("torch_compile_failed", error=str(exc))


def _apply_tome(pipe, ratio: float) -> None:
    """Apply Token Merging (tomesd) for speedup. UNet-only."""
    if not hasattr(pipe, "unet"):
        log.info("tome_skipped", reason="no UNet (transformer model)")
        return
    try:
        import tomesd

        tomesd.apply_patch(pipe, ratio=ratio)
        log.info("tome_enabled", ratio=ratio)
    except ImportError:
        log.warning("tome_skipped", reason="tomesd not installed (pip install tomesd)")
    except Exception as exc:
        log.warning("tome_failed", error=str(exc))


def _resolve_seed(seed: int) -> int:
    """Resolve seed: -1 means random."""
    if seed == -1:
        import random
        return random.randint(0, 2**32 - 1)
    return seed


def _ensure_output_dir(output_dir: str) -> None:
    Path(output_dir).mkdir(parents=True, exist_ok=True)


def _get_cpu_offload_setting() -> str:
    """Read the cpu_offload setting from config, defaulting to 'auto'."""
    try:
        from augmentum.config import settings
        return settings.image_cpu_offload
    except Exception:
        return "auto"


def _cuda_oom_cleanup() -> None:
    """Best-effort cleanup after a CUDA OOM error."""
    from augmentum.image.vram import flush_cuda_cache
    flush_cuda_cache()


def _detect_local_variant(model_path: str) -> str | None:
    """Detect which weight variant exists on disk by scanning subdirectories.

    Returns 'fp16', 'bf16', etc. if variant-tagged files are found, else None.
    """
    for root, _dirs, files in os.walk(model_path):
        for f in files:
            if not f.endswith(".safetensors"):
                continue
            for v in ("fp16", "bf16", "fp8"):
                if f".{v}." in f:
                    return v
    return None


def _load_pretrained_with_fallback(pipeline_cls, model_path, torch_dtype, dtype_str, **extra_kwargs):
    """Load a diffusers pipeline, trying multiple format/variant combinations.

    Downloads may contain only fp16-variant files (e.g. diffusion_pytorch_model.fp16.safetensors)
    with no non-variant copies. Detects the actual variant on disk and uses it.
    """
    # Detect what's actually on disk rather than assuming
    disk_variant = _detect_local_variant(model_path)

    attempts = []
    if disk_variant:
        # Variant files found on disk — try those first
        attempts.append(dict(variant=disk_variant))
    # Try requested dtype variant if different from disk
    requested_variant = "fp16" if dtype_str == "fp16" else None
    if requested_variant and requested_variant != disk_variant:
        attempts.append(dict(variant=requested_variant))
    # No variant (full precision)
    attempts.append(dict())

    last_exc = None
    for kwargs in attempts:
        try:
            return pipeline_cls.from_pretrained(
                model_path, torch_dtype=torch_dtype, **kwargs, **extra_kwargs,
            )
        except Exception as exc:
            last_exc = exc
            continue
    raise last_exc


def _latent_img2img_fallback(pipe, prompt, image, strength, steps, cfg_scale, generator, device):
    """Generic img2img via latent-space noise injection.

    Works with any DiffusionPipeline that has a VAE and scheduler,
    even if no dedicated img2img variant exists (e.g. Lumina2).

    Follows the canonical pattern from FluxImg2ImgPipeline / SD3Img2ImgPipeline
    and validated by ComfyUI's universal KSampler approach:
    1. Encode source image to latents (with shift_factor + scaling_factor)
    2. Truncate the timestep schedule based on strength
    3. Noise latents via scheduler.scale_noise() (or manual flow-matching interp)
    4. Run the pipeline from the truncated schedule onward
    """
    import inspect

    import torch
    from diffusers.image_processor import VaeImageProcessor

    vae = pipe.vae
    scheduler = pipe.scheduler

    # --- Step 1: Encode image to latents ---
    scale_factor = getattr(pipe, "vae_scale_factor", 8)
    processor = VaeImageProcessor(vae_scale_factor=scale_factor)
    img_tensor = processor.preprocess(image).to(device=vae.device, dtype=vae.dtype)

    try:
        with torch.no_grad():
            enc_out = vae.encode(img_tensor)
            if hasattr(enc_out, "latent_dist"):
                image_latents = enc_out.latent_dist.sample(generator=generator)
            elif hasattr(enc_out, "latents"):
                image_latents = enc_out.latents
            else:
                image_latents = enc_out[0] if isinstance(enc_out, (tuple, list)) else enc_out

            # Apply VAE normalization (shift + scale), matching FluxImg2Img / SD3Img2Img
            # Some models have these set to None explicitly in config — guard against it
            shift = getattr(vae.config, "shift_factor", None)
            scaling = getattr(vae.config, "scaling_factor", None)
            if shift is not None and scaling is not None:
                image_latents = (image_latents - shift) * scaling
            elif scaling is not None:
                image_latents = image_latents * scaling
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if "out of memory" in str(exc).lower():
            torch.cuda.empty_cache()
            raise RuntimeError("GPU out of memory during VAE encoding. Try a smaller image or lower resolution.") from exc
        raise

    # --- Step 2: Truncate timestep schedule based on strength ---
    num_inference_steps = max(1, steps)
    scheduler.set_timesteps(num_inference_steps, device=device)

    # Canonical truncation from FluxImg2ImgPipeline.get_timesteps():
    #   init_timestep = min(num_inference_steps * strength, num_inference_steps)
    #   t_start = max(num_inference_steps - init_timestep, 0)
    init_timestep = min(int(num_inference_steps * strength), num_inference_steps)
    t_start = max(num_inference_steps - init_timestep, 0)
    timesteps = scheduler.timesteps[t_start * scheduler.order :]

    if len(timesteps) == 0:
        # strength ≈ 0 means no change — return original image as-is
        from PIL import Image
        return image if isinstance(image, Image.Image) else image

    # --- Step 3: Noise latents using scheduler ---
    # Use torch.randn instead of randn_like to ensure compatibility across
    # all tensor types (quantized, offloaded, etc.).  Generate on the same
    # device as the generator and move to match latents if needed.
    gen_device = generator.device if generator is not None else image_latents.device
    noise = torch.randn(
        image_latents.shape,
        dtype=image_latents.dtype,
        device=gen_device,
        generator=generator,
    )
    if noise.device != image_latents.device:
        noise = noise.to(image_latents.device)
    latent_timestep = timesteps[:1]  # first timestep in truncated schedule

    if hasattr(scheduler, "scale_noise"):
        # Flow-matching schedulers (FlowMatchEulerDiscrete) — the canonical method
        noisy_latents = scheduler.scale_noise(image_latents, latent_timestep, noise)
    elif hasattr(scheduler, "add_noise"):
        # DDPM-style schedulers (DDIM, Euler, DPM, etc.)
        noisy_latents = scheduler.add_noise(image_latents, noise, latent_timestep)
    else:
        # Manual flow-matching interpolation as last resort
        sigma = latent_timestep.float() / getattr(
            scheduler.config, "num_train_timesteps", 1000
        )
        noisy_latents = sigma * noise + (1.0 - sigma) * image_latents

    # --- Step 4: Run pipeline from truncated schedule ---
    pipe_params = inspect.signature(pipe.__call__).parameters
    call_kwargs = dict(
        prompt=prompt,
        guidance_scale=cfg_scale,
        generator=generator,
        latents=noisy_latents,
        output_type="pil",
    )

    # Pass the truncated schedule so the pipeline denoises only the tail.
    # Prefer sigmas (modern flow-matching), fall back to timesteps.
    if "sigmas" in pipe_params:
        # Convert truncated timesteps to sigmas for the pipeline
        sigmas = scheduler.sigmas.cpu()
        call_kwargs["sigmas"] = sigmas[t_start:]
    elif "timesteps" in pipe_params:
        call_kwargs["timesteps"] = timesteps.cpu()
    else:
        call_kwargs["num_inference_steps"] = len(timesteps)

    try:
        result = pipe(**call_kwargs)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if "out of memory" in str(exc).lower():
            torch.cuda.empty_cache()
            raise RuntimeError("GPU out of memory during image generation. Try fewer steps or lower resolution.") from exc
        raise
    return result.images[0]


