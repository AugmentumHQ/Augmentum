"""Unified image pipeline — single class handles all diffusers model types.

Replaces the per-architecture SD15Pipeline/SDXLPipeline/FluxPipeline with
one UnifiedPipeline that auto-detects the model type at load time and uses
universal generation, img2img, and inpaint logic.

Design inspired by ComfyUI's approach: load the model generically, detect
its architecture, and use a single sampling path.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import uuid
from pathlib import Path

from augmentum.image.pipeline import (
    GenerationResult,
    ImagePipeline,
    _apply_pipeline_optimizations,
    _cuda_oom_cleanup,
    _ensure_output_dir,
    _get_cpu_offload_setting,
    _get_torch_dtype,
    _latent_img2img_fallback,
    _load_pretrained_with_fallback,
    _resolve_seed,
)
from augmentum.image.schemas import PipelineType
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Map loaded pipeline class names → PipelineType for auto-detection.
# Covers all mainstream diffusers pipeline classes.
_CLASS_TO_TYPE: dict[str, PipelineType] = {
    # SD 1.x
    "StableDiffusionPipeline": PipelineType.SD15,
    "StableDiffusionImg2ImgPipeline": PipelineType.SD15,
    "StableDiffusionInpaintPipeline": PipelineType.SD15,
    # SDXL
    "StableDiffusionXLPipeline": PipelineType.SDXL,
    "StableDiffusionXLImg2ImgPipeline": PipelineType.SDXL,
    "StableDiffusionXLInpaintPipeline": PipelineType.SDXL,
    # FLUX
    "FluxPipeline": PipelineType.FLUX,
    "FluxImg2ImgPipeline": PipelineType.FLUX,
    "FluxInpaintPipeline": PipelineType.FLUX,
    # SD3
    "StableDiffusion3Pipeline": PipelineType.FLUX,
    "StableDiffusion3Img2ImgPipeline": PipelineType.FLUX,
    "StableDiffusion3InpaintPipeline": PipelineType.FLUX,
    # Lumina
    "Lumina2Pipeline": PipelineType.FLUX,
    "LuminaPipeline": PipelineType.FLUX,
    # PixArt
    "PixArtAlphaPipeline": PipelineType.FLUX,
    "PixArtSigmaPipeline": PipelineType.FLUX,
    # HunyuanDiT
    "HunyuanDiTPipeline": PipelineType.FLUX,
    # Kandinsky 3
    "Kandinsky3Pipeline": PipelineType.FLUX,
    # AuraFlow
    "AuraFlowPipeline": PipelineType.FLUX,
    # Wuerstchen / Stable Cascade
    "WuerstchenCombinedPipeline": PipelineType.FLUX,
    "StableCascadeCombinedPipeline": PipelineType.FLUX,
    # Qwen Image
    "QwenImagePipeline": PipelineType.FLUX,
    "QwenImageImg2ImgPipeline": PipelineType.FLUX,
    "QwenImageInpaintPipeline": PipelineType.FLUX,
    "QwenImageEditPipeline": PipelineType.FLUX,
    "QwenImageEditPlusPipeline": PipelineType.FLUX,
    "QwenImageEditInpaintPipeline": PipelineType.FLUX,
    # Sana (Linear DiT, 32x DC-AE compression)
    "SanaPipeline": PipelineType.FLUX,
    "SanaPAGPipeline": PipelineType.FLUX,
    # Z-Image (S3-DiT architecture, flow matching)
    "ZImagePipeline": PipelineType.FLUX,
    "ZImageImg2ImgPipeline": PipelineType.FLUX,
    "ZImageInpaintPipeline": PipelineType.FLUX,
    "ZImageOmniPipeline": PipelineType.FLUX,
    "ZImageControlNetPipeline": PipelineType.FLUX,
    "ZImageControlNetInpaintPipeline": PipelineType.FLUX,
}

# Default generation parameters per detected pipeline type.
_DEFAULTS: dict[PipelineType, dict] = {
    PipelineType.SD15: {"width": 512, "height": 512, "steps": 20, "cfg_scale": 7.0},
    PipelineType.SDXL: {"width": 1024, "height": 1024, "steps": 20, "cfg_scale": 7.0},
    PipelineType.FLUX: {"width": 1024, "height": 1024, "steps": 4, "cfg_scale": 0.0},
}

# Qwen pipelines use ``true_cfg_scale`` instead of ``guidance_scale``.
_QWEN_PIPELINE_PREFIXES = ("QwenImage",)

# Default overrides for Qwen pipelines (much higher step counts than FLUX).
_QWEN_DEFAULTS: dict = {"steps": 50, "cfg_scale": 4.0}
_QWEN_EDIT_DEFAULTS: dict = {"steps": 40, "cfg_scale": 4.0}


def _is_qwen_pipeline(pipe) -> bool:
    """Check if a pipeline is a Qwen Image pipeline."""
    return type(pipe).__name__.startswith(_QWEN_PIPELINE_PREFIXES)


def _is_qwen_edit_pipeline(pipe) -> bool:
    """Check if a pipeline is a Qwen Image *editing* pipeline."""
    name = type(pipe).__name__
    return name.startswith(_QWEN_PIPELINE_PREFIXES) and "Edit" in name


def _clip_skip_safe_for_pipe(pipe) -> bool:
    """Return False when the pipeline's clip_skip path will crash.

    Diffusers' SD1.5-family pipelines reach for
    ``self.text_encoder.text_model.final_layer_norm`` inside the
    clip_skip branch. transformers ≥ 5.0 removed that wrapper, so the
    access raises AttributeError before the first denoising step.
    SDXL/Flux pipelines take a different code path and are unaffected,
    so they always return True here.
    """
    te = getattr(pipe, "text_encoder", None)
    if te is None:
        return True
    return hasattr(te, "text_model")

# VRAM key for _apply_pipeline_optimizations
_TYPE_TO_KEY: dict[PipelineType, str] = {
    PipelineType.SD15: "sd15",
    PipelineType.SDXL: "sdxl",
    PipelineType.FLUX: "flux",
}


def _detect_type_from_pipe(pipe) -> PipelineType:
    """Detect PipelineType from a loaded diffusers pipeline object."""
    class_name = type(pipe).__name__

    # Direct class name match
    if class_name in _CLASS_TO_TYPE:
        return _CLASS_TO_TYPE[class_name]

    # Heuristic: anything with "XL" in the class name is SDXL
    if "XL" in class_name.upper():
        return PipelineType.SDXL

    # Heuristic: transformer-based models (FLUX, Lumina, SD3, etc.) → FLUX bucket
    if hasattr(pipe, "transformer"):
        return PipelineType.FLUX

    # Heuristic: UNet with cross_attention_dim >= 2048 → SDXL
    if hasattr(pipe, "unet"):
        try:
            cross_dim = getattr(pipe.unet.config, "cross_attention_dim", 0)
            if cross_dim >= 2048:
                return PipelineType.SDXL
        except AttributeError:
            pass

    # Default to SD15 (most compatible)
    return PipelineType.SD15


def _is_edit_pipeline(pipe) -> bool:
    """Check if a pipeline is natively an edit/img2img pipeline.

    Edit pipelines (like QwenImageEditPipeline) require a source ``image``
    parameter and cannot do txt2img.  They should be called directly for
    img2img rather than going through variant conversion.
    """
    class_name = type(pipe).__name__
    # Explicit class name patterns
    if "Edit" in class_name or "Img2Img" in class_name:
        return True
    # Fallback: check if __call__ has a required `image` parameter
    try:
        sig = inspect.signature(pipe.__call__)
        param = sig.parameters.get("image")
        if param and param.default is inspect.Parameter.empty:
            return True
    except (ValueError, TypeError):
        pass
    return False


def _try_get_variant_pipe(pipe, variant: str):
    """Try to get a specialized pipeline variant (img2img/inpaint) from the
    loaded txt2img pipeline.

    Args:
        pipe: The loaded diffusers pipeline.
        variant: One of "img2img" or "inpaint".

    Returns the variant pipeline, or None if unavailable.
    """
    class_name = type(pipe).__name__

    # Edit/img2img pipelines are already the right variant — don't convert
    if _is_edit_pipeline(pipe):
        if variant == "img2img":
            return pipe  # use directly
        # For inpaint on an edit pipeline, try edit-inpaint variant below

    # Build a list of candidate classes to try
    candidates = []

    if variant == "img2img":
        # Try architecture-specific img2img classes
        if "QwenImage" in class_name and "Edit" not in class_name:
            try:
                from diffusers import QwenImageImg2ImgPipeline
                candidates.append(QwenImageImg2ImgPipeline)
            except ImportError:
                pass
        elif "ZImage" in class_name:
            try:
                from diffusers import ZImageImg2ImgPipeline
                candidates.append(ZImageImg2ImgPipeline)
            except ImportError:
                pass
        elif "Flux" in class_name:
            try:
                from diffusers import FluxImg2ImgPipeline
                candidates.append(FluxImg2ImgPipeline)
            except ImportError:
                pass
        elif "XL" in class_name or "SDXL" in class_name:
            try:
                from diffusers import StableDiffusionXLImg2ImgPipeline
                candidates.append(StableDiffusionXLImg2ImgPipeline)
            except ImportError:
                pass
        elif "StableDiffusion" in class_name:
            try:
                from diffusers import StableDiffusionImg2ImgPipeline
                candidates.append(StableDiffusionImg2ImgPipeline)
            except ImportError:
                pass

        # Generic fallback
        try:
            from diffusers import AutoPipelineForImage2Image
            candidates.append(AutoPipelineForImage2Image)
        except ImportError:
            pass

    elif variant == "inpaint":
        # Try edit-inpaint variant for edit pipelines
        if "QwenImageEdit" in class_name or "QwenImageEditPlus" in class_name:
            try:
                from diffusers import QwenImageEditInpaintPipeline
                candidates.append(QwenImageEditInpaintPipeline)
            except ImportError:
                pass
        elif "QwenImage" in class_name:
            try:
                from diffusers import QwenImageInpaintPipeline
                candidates.append(QwenImageInpaintPipeline)
            except ImportError:
                pass
        elif "ZImage" in class_name:
            try:
                from diffusers import ZImageInpaintPipeline
                candidates.append(ZImageInpaintPipeline)
            except ImportError:
                pass
        elif "Flux" in class_name:
            try:
                from diffusers import FluxInpaintPipeline
                candidates.append(FluxInpaintPipeline)
            except ImportError:
                pass
        elif "XL" in class_name or "SDXL" in class_name:
            try:
                from diffusers import StableDiffusionXLInpaintPipeline
                candidates.append(StableDiffusionXLInpaintPipeline)
            except ImportError:
                pass
        elif "StableDiffusion" in class_name:
            try:
                from diffusers import StableDiffusionInpaintPipeline
                candidates.append(StableDiffusionInpaintPipeline)
            except ImportError:
                pass

        # Generic fallback
        try:
            from diffusers import AutoPipelineForInpainting
            candidates.append(AutoPipelineForInpainting)
        except ImportError:
            pass

    for cls in candidates:
        try:
            return cls.from_pipe(pipe)
        except Exception as exc:
            log.debug("inpaint_from_pipe_failed", cls=cls.__name__, error=str(exc))
            continue
    return None


def _list_gguf_files(model_dir: Path) -> list[Path]:
    """Return GGUF files directly under *model_dir* or one level deeper.

    Walks the dir + non-hidden subdirs so layouts like ``<name>/file.gguf``
    AND ``<name>/gguf/file.gguf`` (catalog downloads that preserve repo
    subfolders like SeeSee21/Z-Anime's ``gguf/`` prefix) both resolve.
    Skips dotfile dirs (``.cache``) to avoid picking up HF resume partials.
    """
    out: list[Path] = []
    if not model_dir.is_dir():
        return out
    for f in model_dir.iterdir():
        if f.is_file() and f.suffix.lower() == ".gguf":
            out.append(f)
    for sub in model_dir.iterdir():
        if not sub.is_dir() or sub.name.startswith("."):
            continue
        for f in sub.iterdir():
            if f.is_file() and f.suffix.lower() == ".gguf":
                out.append(f)
    return out


def _is_gguf_model(model_path: str) -> bool:
    """Check if a model path points to a GGUF model (directory containing .gguf files)."""
    p = Path(model_path)
    if p.is_file() and p.suffix.lower() == ".gguf":
        return True
    if p.is_dir():
        return bool(_list_gguf_files(p))
    # HuggingFace repo ID with GGUF in the name
    if "gguf" in model_path.lower() and "/" in model_path:
        return True
    return False


def _find_gguf_file(model_path: str) -> str:
    """Find the best GGUF file from a path or directory.

    Prefers Q4_K_M, then Q5_K_M, then Q8_0, then any .gguf.
    """
    p = Path(model_path)
    if p.is_file():
        return str(p)

    if p.is_dir():
        gguf_files = sorted(_list_gguf_files(p))
        if not gguf_files:
            raise FileNotFoundError(f"No .gguf files found in {model_path}")

        # Prefer common quantization levels
        for preference in ("Q4_K_M", "Q5_K_M", "Q8_0", "Q4_K_S", "Q6_K"):
            for f in gguf_files:
                if preference in f.name:
                    return str(f)
        return str(gguf_files[0])

    # HuggingFace repo — return the path as-is, let from_single_file handle it
    return model_path


def _load_gguf_meta(model_path: str) -> dict | None:
    """Read ``gguf_meta.json`` from a model directory if it exists."""
    import json

    meta_path = os.path.join(model_path, "gguf_meta.json")
    if not os.path.exists(meta_path):
        p = Path(model_path)
        if p.is_file():
            meta_path = os.path.join(p.parent, "gguf_meta.json")
        if not os.path.exists(meta_path):
            return None
    try:
        with open(meta_path) as f:
            return json.load(f)
    except Exception:
        return None


def _infer_gguf_meta_from_path(model_path: str) -> dict | None:
    """Try to match a model path against known GGUF catalog entries."""
    from augmentum.image.hardware import RECOMMENDED_MODELS

    dir_name = Path(model_path).name.lower()
    for cm in RECOMMENDED_MODELS:
        if not cm.gguf_base_repo:
            continue
        # Match by repo_id-based directory name (e.g. "unsloth--Qwen-Image-2512-GGUF")
        safe_name = cm.repo_id.replace("/", "--").lower()
        if safe_name == dir_name or cm.repo_id.split("/")[-1].lower() == dir_name:
            meta = {
                "gguf_base_repo": cm.gguf_base_repo,
                "gguf_pipeline_class": cm.gguf_pipeline_class,
                "gguf_transformer_class": cm.gguf_transformer_class,
            }
            # Save it for next time
            meta_path = os.path.join(model_path, "gguf_meta.json")
            try:
                with open(meta_path, "w") as f:
                    json.dump(meta, f)
                log.info("gguf_meta_inferred_and_saved", path=meta_path, repo=cm.repo_id)
            except OSError as exc:
                log.warning(
                    "gguf_meta_save_failed",
                    path=meta_path,
                    error=str(exc),
                )
            return meta
    return None


def _setup_gguf_cuda_kernels() -> None:
    """Enable CUDA dequantization kernels for GGUF if appropriate.

    Sets the ``DIFFUSERS_GGUF_CUDA_KERNELS`` environment variable based on
    config and hardware capability.  Must be called before any GGUF loading.

    - ``auto`` (default): enable when GPU compute capability >= 7.0
    - ``on``: always enable (user takes responsibility)
    - ``off``: never enable
    """
    import os

    # Already set by user env — respect it
    if os.environ.get("DIFFUSERS_GGUF_CUDA_KERNELS"):
        return

    try:
        from augmentum.config import settings
        mode = settings.image_gguf_cuda_kernels
    except Exception:
        mode = "auto"

    if mode == "off":
        return

    if mode == "on":
        os.environ["DIFFUSERS_GGUF_CUDA_KERNELS"] = "true"
        log.info("gguf_cuda_kernels_enabled", reason="config=on")
        return

    # auto: check GPU capability
    try:
        import torch
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            if major >= 7:
                os.environ["DIFFUSERS_GGUF_CUDA_KERNELS"] = "true"
                log.info(
                    "gguf_cuda_kernels_enabled",
                    reason="auto",
                    compute_capability=f"{major}.{minor}",
                )
            else:
                log.info(
                    "gguf_cuda_kernels_skipped",
                    reason=f"compute capability {major}.{minor} < 7.0",
                )
    except (ImportError, RuntimeError) as exc:
        # ImportError: torch missing (CPU-only build). RuntimeError:
        # CUDA driver issue / device query failed. Either way the kernel
        # opt-in stays off — debug log so a torch oddity is findable.
        log.debug("gguf_cuda_kernels_probe_failed", error=str(exc))


def _load_gguf_pipeline(model_path: str, torch_dtype, device: str):
    """Load a GGUF-quantized diffusion model.

    GGUF models require a two-step load:
    1. Load the transformer from the GGUF file using TransformerModel.from_single_file()
    2. Load the full pipeline from the base repo using Pipeline.from_pretrained(transformer=...)

    The model directory must contain a ``gguf_meta.json`` with:
    - ``gguf_base_repo``: HuggingFace repo for pipeline components (e.g. "Qwen/Qwen-Image-2512")
    - ``gguf_pipeline_class``: Pipeline class name (e.g. "QwenImagePipeline")
    - ``gguf_transformer_class``: Transformer class name (e.g. "QwenImageTransformer2DModel")

    Falls back to FluxPipeline/FluxTransformer2DModel if no metadata is found.

    Optimizations applied:
    - CUDA dequant kernels (auto-enabled on compute cap >= 7.0, ~10% speedup)
    - Parallel transformer + text encoder quantization config preparation
    - Group offloading for tight VRAM (block-level CPU/GPU swap)
    """

    import torch

    try:
        from diffusers import GGUFQuantizationConfig
    except ImportError:
        raise ImportError(
            "GGUF support requires diffusers >= 0.32.0 and the gguf package. "
            "Run: pip install --upgrade diffusers gguf"
        )

    # Enable CUDA dequant kernels before any GGUF loading
    _setup_gguf_cuda_kernels()

    gguf_path = _find_gguf_file(model_path)
    log.info("gguf_loading", path=gguf_path)

    # Read GGUF metadata for loading instructions
    meta = _load_gguf_meta(model_path)

    if not meta:
        # Try to infer from directory name by matching against catalog
        meta = _infer_gguf_meta_from_path(model_path)

    if meta and meta.get("gguf_base_repo"):
        base_repo = meta["gguf_base_repo"]
        pipeline_cls_name = meta.get("gguf_pipeline_class", "FluxPipeline")
        transformer_cls_name = meta.get("gguf_transformer_class", "FluxTransformer2DModel")
    else:
        # Fallback: assume FLUX-compatible
        base_repo = ""
        pipeline_cls_name = "FluxPipeline"
        transformer_cls_name = "FluxTransformer2DModel"
        log.warning(
            "gguf_no_metadata",
            path=model_path,
            hint="No gguf_meta.json found. Falling back to FluxPipeline. "
                 "Re-download from the catalog to get proper metadata.",
        )

    import diffusers

    # Resolve classes dynamically
    transformer_cls = getattr(diffusers, transformer_cls_name, None)
    pipeline_cls = getattr(diffusers, pipeline_cls_name, None)

    if not transformer_cls:
        raise ImportError(
            f"Transformer class '{transformer_cls_name}' not found in diffusers. "
            f"You may need diffusers >= 0.36.0 for this model."
        )
    if not pipeline_cls:
        raise ImportError(
            f"Pipeline class '{pipeline_cls_name}' not found in diffusers. "
            f"You may need diffusers >= 0.36.0 for this model."
        )

    # Choose compute dtype: prefer bfloat16 if supported, else float16
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        compute_dtype = torch.bfloat16
    elif torch.cuda.is_available():
        compute_dtype = torch.float16
        log.info("gguf_compute_dtype_fp16", hint="GPU does not support bfloat16, using float16")
    else:
        compute_dtype = torch.float32

    quantization_config = GGUFQuantizationConfig(compute_dtype=compute_dtype)

    # Step 1: Load transformer from GGUF
    log.info("gguf_loading_transformer", cls=transformer_cls_name, base_repo=base_repo, path=gguf_path)
    sf_kwargs = {
        "quantization_config": quantization_config,
        "torch_dtype": torch_dtype,
    }
    if base_repo:
        sf_kwargs["config"] = base_repo
        sf_kwargs["subfolder"] = "transformer"
    transformer = transformer_cls.from_single_file(gguf_path, **sf_kwargs)

    # Step 2: Load full pipeline from base repo with our quantized transformer.
    # This downloads the text encoder, VAE, scheduler, tokenizer, etc. on first
    # use — can be 10-20 GB depending on the model.  Subsequent loads use cache.
    if base_repo:
        log.info(
            "gguf_loading_pipeline",
            cls=pipeline_cls_name,
            base_repo=base_repo,
            hint="Downloading pipeline components from HF (first load may take a while)",
        )

        # Try to quantize the text encoder with bitsandbytes 4-bit (NF4).
        # This reduces the Qwen2.5-VL-7B text encoder from ~15GB to ~4GB,
        # making the model usable on 24GB cards.
        #
        # Some pipelines (e.g. QwenImageEditPlusPipeline) silently ignore
        # ``text_encoder_quantization_config``.  For those, we load the
        # text encoder separately with quantization and inject it.
        extra_kwargs: dict = {}
        text_encoder_override = None
        try:
            from transformers import AutoModel, BitsAndBytesConfig

            text_enc_quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
            )

            # Check if the pipeline class accepts text_encoder_quantization_config
            import inspect
            pipe_init_params = set(inspect.signature(pipeline_cls.from_pretrained).parameters)
            if "text_encoder_quantization_config" in pipe_init_params:
                extra_kwargs["text_encoder_quantization_config"] = text_enc_quant
                log.info("gguf_text_encoder_4bit", hint="Quantizing text encoder to 4-bit NF4 (pipeline kwarg)")
            else:
                # Load text encoder separately with quantization
                log.info("gguf_text_encoder_4bit_separate", hint="Loading text encoder separately with 4-bit NF4")
                try:
                    text_encoder_override = AutoModel.from_pretrained(
                        base_repo,
                        subfolder="text_encoder",
                        quantization_config=text_enc_quant,
                        torch_dtype=torch_dtype,
                    )
                except Exception as te_exc:
                    log.warning("gguf_text_encoder_4bit_failed", error=str(te_exc),
                                hint="Falling back to full-precision text encoder")
        except ImportError:
            log.info(
                "gguf_text_encoder_full",
                hint="bitsandbytes not installed — loading text encoder in full precision. "
                     "Install bitsandbytes to reduce VRAM usage by ~11GB.",
            )

        if text_encoder_override is not None:
            extra_kwargs["text_encoder"] = text_encoder_override

        def _load_pipeline(**overrides):
            merged = {**extra_kwargs, **overrides}
            return pipeline_cls.from_pretrained(
                base_repo,
                transformer=transformer,
                torch_dtype=torch_dtype,
                **merged,
            )

        try:
            pipe = _load_pipeline()
        except (OSError, ConnectionError, Exception) as exc:
            # Network failure (DNS, timeout, etc.) — retry with cached files only.
            if "local_files_only" not in str(extra_kwargs):
                log.warning(
                    "gguf_network_fallback",
                    error=str(exc),
                    hint="HuggingFace unreachable — retrying with cached files",
                )
                pipe = _load_pipeline(local_files_only=True)
            else:
                raise
    else:
        # No base repo — try loading pipeline with just the transformer
        # This is a best-effort fallback
        log.warning("gguf_no_base_repo", hint="Loading pipeline without base repo components")
        pipe = pipeline_cls(transformer=transformer)

    return pipe


def _find_single_safetensors(model_path: str) -> str | None:
    """Check if a model directory contains a single safetensors checkpoint.

    CivitAI downloads typically produce a directory with just one .safetensors
    file and no model_index.json (diffusers layout).  These need to be loaded
    via from_single_file() instead of from_pretrained().

    Returns the path to the safetensors file, or None if not a single-file layout.
    """
    if not os.path.isdir(model_path):
        return None
    # If model_index.json exists, this is a diffusers layout
    if os.path.exists(os.path.join(model_path, "model_index.json")):
        return None
    # Find safetensors files (root level only — subdirs indicate diffusers components)
    st_files = [
        f for f in os.listdir(model_path)
        if f.endswith(".safetensors") and os.path.isfile(os.path.join(model_path, f))
    ]
    if len(st_files) == 1:
        return os.path.join(model_path, st_files[0])
    return None


def _load_pretrained_pipeline(model_path: str, torch_dtype, dtype: str):
    """Load a standard diffusers model via from_pretrained with fallbacks.

    Handles both diffusers-format directories (model_index.json + components)
    and single-file safetensors checkpoints (common from CivitAI downloads).
    """
    from diffusers import DiffusionPipeline

    # If the path isn't a local directory and contains "--", restore it to a
    # valid HuggingFace repo ID (org/model).  Local downloads use "--" as a
    # directory-safe separator (e.g. "runwayml--stable-diffusion-v1-5"), but
    # from_pretrained rejects that as a repo_id.
    if not os.path.isdir(model_path) and "--" in model_path:
        model_path = model_path.replace("--", "/", 1)

    # Check for single-file checkpoint (CivitAI downloads)
    single_file = _find_single_safetensors(model_path)
    if single_file:
        return _load_single_file_pipeline(single_file, torch_dtype)

    pipe = None
    try:
        pipe = _load_pretrained_with_fallback(
            DiffusionPipeline, model_path, torch_dtype, dtype,
        )
    except Exception:
        # If DiffusionPipeline fails (no model_index.json, etc.),
        # try architecture-specific loaders in order
        for loader_name in (
            "StableDiffusionPipeline",
            "StableDiffusionXLPipeline",
        ):
            try:
                import diffusers
                loader_cls = getattr(diffusers, loader_name)
                extra = {}
                if "XL" not in loader_name:
                    extra = {"safety_checker": None, "requires_safety_checker": False}
                pipe = _load_pretrained_with_fallback(
                    loader_cls, model_path, torch_dtype, dtype, **extra,
                )
                break
            except Exception as exc:
                log.debug("pretrained_loader_failed", loader=loader_name, error=str(exc))
                continue
        if pipe is None:
            raise
    return pipe


def _load_single_file_pipeline(safetensors_path: str, torch_dtype):
    """Load a single-file safetensors checkpoint via from_single_file().

    Tries SDXL first (since most CivitAI checkpoints are SDXL), then SD1.5.
    Uses diffusers' built-in architecture detection where available.
    """
    log.info("loading_single_file_checkpoint", path=safetensors_path)

    # Try DiffusionPipeline.from_single_file first (diffusers >= 0.25)
    # — it auto-detects the architecture from the state dict
    try:
        from diffusers import DiffusionPipeline

        pipe = DiffusionPipeline.from_single_file(
            safetensors_path,
            torch_dtype=torch_dtype,
            safety_checker=None,
        )
        log.info("single_file_loaded", loader="DiffusionPipeline", pipe=type(pipe).__name__)
        return pipe
    except Exception as exc:
        log.debug("single_file_auto_detect_failed", error=str(exc))

    # Fallback: try architecture-specific loaders
    import diffusers

    for loader_name in ("StableDiffusionXLPipeline", "StableDiffusionPipeline"):
        try:
            loader_cls = getattr(diffusers, loader_name)
            extra = {}
            if "XL" not in loader_name:
                extra = {"safety_checker": None, "requires_safety_checker": False}
            pipe = loader_cls.from_single_file(
                safetensors_path,
                torch_dtype=torch_dtype,
                **extra,
            )
            log.info("single_file_loaded", loader=loader_name, pipe=type(pipe).__name__)
            return pipe
        except Exception as exc:
            log.debug("single_file_loader_failed", loader=loader_name, error=str(exc))
            continue

    raise RuntimeError(
        f"Could not load single-file checkpoint: {safetensors_path}. "
        "The model may require a diffusers-format conversion."
    )


async def _run_on_thread(fn):
    """Run a blocking function on a thread, converting CUDA OOM to friendly messages."""
    try:
        return await asyncio.to_thread(fn)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            _cuda_oom_cleanup()
            raise RuntimeError(
                "Out of GPU memory. Try a smaller model or close other GPU applications."
            ) from exc
        raise


def _has_ip_adapter_reference(value) -> bool:
    """Return True when a generation request has a real IP-Adapter reference."""
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, tuple)):
        return any(_has_ip_adapter_reference(item) for item in value)
    return True


class UnifiedPipeline(ImagePipeline):
    """Single pipeline that handles any diffusers model type.

    Auto-detects SD1.5, SDXL, FLUX, Lumina2, SD3, etc. at load time.
    Uses DiffusionPipeline.from_pretrained() for universal model loading,
    then dispatches to the appropriate generation strategy.
    """

    # Max number of prompt embeddings to cache.  Keyed by (prompt, neg_prompt)
    # tuple.  Cleared when the pipeline is reloaded or model changes.
    _PROMPT_CACHE_SIZE = 8

    def __init__(self) -> None:
        self._pipe = None
        self._model_path = ""
        self._device = "cpu"
        self._detected_type: PipelineType = PipelineType.SD15
        # Cached after load to avoid per-call introspection
        self._pipe_params: set[str] = set()
        self._is_qwen: bool = False
        self._is_edit: bool = False
        # Cached variant pipelines (created on first use, reused thereafter)
        self._variant_cache: dict[str, object | None] = {}
        # Prompt embedding cache — avoids re-encoding identical prompts
        self._prompt_cache: dict[tuple[str, str], dict] = {}
        self._ip_adapter_loaded = False

    @property
    def pipeline_type(self) -> PipelineType:
        return self._detected_type

    @property
    def is_loaded(self) -> bool:
        return self._pipe is not None

    @property
    def model_name(self) -> str:
        return self._model_path

    @property
    def diffusers_pipe(self):
        """Access the underlying diffusers pipeline (for token limit detection, etc.)."""
        return self._pipe

    def _cache_pipe_metadata(self) -> None:
        """Cache pipeline introspection results after load.

        Avoids calling inspect.signature() on every generate/img2img/inpaint call.
        """
        if not self._pipe:
            return
        self._pipe_params = set(inspect.signature(self._pipe.__call__).parameters)
        self._is_qwen = _is_qwen_pipeline(self._pipe)
        self._is_edit = _is_edit_pipeline(self._pipe)
        self._variant_cache.clear()
        self._prompt_cache.clear()

    def _get_cached_variant(self, variant: str):
        """Get or create a variant pipeline (img2img/inpaint), caching the result."""
        if variant in self._variant_cache:
            return self._variant_cache[variant]
        result = _try_get_variant_pipe(self._pipe, variant)
        self._variant_cache[variant] = result
        return result

    async def load(self, model_path: str, device: str = "cuda", dtype: str = "fp16") -> None:
        def _load():
            torch_dtype = _get_torch_dtype(dtype)

            # Check if this is a GGUF single-file model
            if model_path.lower().endswith(".gguf") or _is_gguf_model(model_path):
                pipe = _load_gguf_pipeline(model_path, torch_dtype, device)
            else:
                pipe = _load_pretrained_pipeline(model_path, torch_dtype, dtype)

            # Disable safety checker if present (SD1.5 models)
            if hasattr(pipe, "safety_checker") and pipe.safety_checker is not None:
                pipe.safety_checker = None
                if hasattr(pipe, "requires_safety_checker"):
                    pipe.requires_safety_checker = False

            pipe.set_progress_bar_config(disable=True)

            # Detect type from the loaded pipeline object
            detected = _detect_type_from_pipe(pipe)
            pipeline_key = _TYPE_TO_KEY.get(detected, "sd15")

            cpu_offload = _get_cpu_offload_setting()
            pipe = _apply_pipeline_optimizations(
                pipe, device, cpu_offload=cpu_offload, pipeline_key=pipeline_key,
            )
            return pipe, detected

        log.info("pipeline_loading", type="unified", model=model_path, device=device)
        self._pipe, self._detected_type = await _run_on_thread(_load)
        self._model_path = model_path
        self._device = device
        self._cache_pipe_metadata()
        log.info(
            "pipeline_loaded",
            type="unified",
            detected=self._detected_type.value,
            pipe_class=type(self._pipe).__name__,
            model=model_path,
        )

    async def unload(self) -> None:
        if self._pipe is not None:
            # Move pipeline ref out of self — do NOT keep a local reference
            # that would prevent GC from collecting inside release_pipeline.
            pipe_ref = self._pipe
            self._pipe = None
            self._variant_cache.clear()
            self._prompt_cache.clear()
            self._pipe_params = set()
            self._ip_adapter_loaded = False
            model_label = self._model_path or "unified"

            from augmentum.image.vram import release_pipeline

            try:
                # release_pipeline will del the pipe and gc.collect().
                # We must not hold any other reference to it.
                await asyncio.to_thread(
                    release_pipeline, pipe_ref, label=model_label,
                )
            except Exception:
                log.debug("cuda_cleanup_failed", pipeline="unified", exc_info=True)

            log.info("pipeline_unloaded", type="unified")

    def _encode_prompt_cached(
        self,
        pipe,
        pipe_params: set[str],
        prompt: str,
        negative_prompt: str,
    ) -> dict | None:
        """Encode prompt to embeddings, returning cached result if available.

        Returns a dict of ``prompt_embeds`` (and optionally
        ``negative_prompt_embeds``, ``pooled_prompt_embeds``, etc.) that can
        be spread into the pipeline call kwargs in place of the text prompt.

        Returns ``None`` if the pipeline doesn't expose ``encode_prompt`` or
        if encoding fails — caller should fall back to passing the raw text.
        """
        if "prompt_embeds" not in pipe_params:
            return None

        cache_key = (prompt, negative_prompt)
        if cache_key in self._prompt_cache:
            return self._prompt_cache[cache_key]

        # Try to call the pipeline's encode_prompt method
        encode_fn = getattr(pipe, "encode_prompt", None)
        if encode_fn is None:
            return None

        try:
            import inspect

            import torch

            # Introspect encode_prompt return names from its signature
            # to correctly map tuple positions across architectures.
            # Lumina2: (prompt_embeds, prompt_attention_mask, neg_embeds, neg_mask)
            # SDXL:    (prompt_embeds, neg_embeds, pooled, neg_pooled)
            _return_names: list[str] | None = None
            try:
                _sig = inspect.signature(encode_fn)
                _ann = _sig.return_annotation
                # diffusers annotates as tuple[Tensor, Tensor, ...]
                if hasattr(_ann, "__args__"):
                    # Can't get names from tuple[T,T,...], fall through
                    pass
                # Try to read the source for return variable names
                _src = inspect.getsource(encode_fn)
                # Look for the final "return x, y, z" line
                for _line in reversed(_src.splitlines()):
                    _stripped = _line.strip()
                    if _stripped.startswith("return ") and "," in _stripped:
                        _names = [n.strip() for n in _stripped[7:].split(",")]
                        if len(_names) >= 2 and "prompt_embeds" in _names[0]:
                            _return_names = _names
                        break
            except (TypeError, OSError):
                pass

            with torch.inference_mode():
                # Try calling with both prompts if the method supports it
                _enc_params = set(inspect.signature(encode_fn).parameters)
                if "negative_prompt" in _enc_params and negative_prompt:
                    enc_result = encode_fn(
                        prompt=prompt, negative_prompt=negative_prompt,
                    )
                else:
                    enc_result = encode_fn(prompt=prompt)

                embeds: dict = {}
                if isinstance(enc_result, dict):
                    embeds = enc_result
                elif isinstance(enc_result, tuple):
                    if _return_names and len(_return_names) == len(enc_result):
                        # Map by introspected return names — accurate for any arch
                        for name, val in zip(_return_names, enc_result):
                            if val is not None:
                                embeds[name] = val
                    else:
                        # Fallback: assume SDXL-style positional layout
                        # (prompt_embeds, negative_prompt_embeds, pooled, neg_pooled)
                        if len(enc_result) >= 1 and enc_result[0] is not None:
                            embeds["prompt_embeds"] = enc_result[0]
                        if len(enc_result) >= 2 and enc_result[1] is not None:
                            embeds["negative_prompt_embeds"] = enc_result[1]
                        if len(enc_result) >= 3 and enc_result[2] is not None:
                            embeds["pooled_prompt_embeds"] = enc_result[2]
                        if len(enc_result) >= 4 and enc_result[3] is not None:
                            embeds["negative_pooled_prompt_embeds"] = enc_result[3]
                else:
                    return None

                # Must have at least prompt_embeds
                if "prompt_embeds" not in embeds:
                    return None

                # If we didn't get negative embeds and they're needed, encode separately
                if negative_prompt and "negative_prompt_embeds" not in embeds:
                    try:
                        neg_result = encode_fn(prompt=negative_prompt)
                        if isinstance(neg_result, tuple) and len(neg_result) >= 1:
                            embeds["negative_prompt_embeds"] = neg_result[0]
                            if _return_names:
                                # Use introspected names for neg encoding too
                                for name, val in zip(_return_names, neg_result):
                                    if val is not None and "negative" in name:
                                        embeds[name] = val
                            elif len(neg_result) >= 3 and neg_result[2] is not None:
                                embeds["negative_pooled_prompt_embeds"] = neg_result[2]
                    except Exception:
                        # Diffusers' encode_prompt signatures vary widely;
                        # if the negative-side call shape doesn't match,
                        # we keep the positive embeds and skip neg.
                        log.debug("encode_negative_prompt_failed", exc_info=True)

                # Filter to only keys the pipeline actually accepts
                embeds = {k: v for k, v in embeds.items() if k in pipe_params}

                # Safety: if pipeline requires attention masks with embeds
                # but we don't have them, skip caching to avoid runtime error
                if "prompt_attention_mask" in pipe_params and "prompt_embeds" in embeds:
                    if "prompt_attention_mask" not in embeds:
                        log.debug("prompt_cache_skip", reason="missing prompt_attention_mask")
                        return None

                if "prompt_embeds" in embeds:
                    # Evict oldest if cache full
                    if len(self._prompt_cache) >= self._PROMPT_CACHE_SIZE:
                        oldest = next(iter(self._prompt_cache))
                        del self._prompt_cache[oldest]
                    self._prompt_cache[cache_key] = embeds
                    log.debug("prompt_cache_hit", prompt_len=len(prompt), cached=len(self._prompt_cache))
                    return embeds
        except Exception:
            log.debug("prompt_encode_failed", exc_info=True)

        return None

    def _build_call_kwargs(
        self,
        pipe,
        pipe_params: set[str],
        *,
        prompt: str,
        negative_prompt: str,
        cfg_scale: float,
        steps: int,
        generator,
        **extra,
    ) -> dict:
        """Build the kwargs dict for a pipeline __call__, handling Qwen vs standard."""
        # Pop augmentum-specific keys from extra before spreading into call_kwargs
        # so they don't leak into the diffusers pipeline call.
        guidance_rescale = extra.pop("guidance_rescale", None)
        clip_skip = extra.pop("clip_skip", None)

        call_kwargs: dict = {
            "num_inference_steps": steps,
            "generator": generator,
            **extra,
        }

        # Try to use cached prompt embeddings (skips text encoder entirely)
        cached_embeds = self._encode_prompt_cached(pipe, pipe_params, prompt, negative_prompt)
        if cached_embeds:
            call_kwargs.update(cached_embeds)
        else:
            call_kwargs["prompt"] = prompt

        is_qwen = _is_qwen_pipeline(pipe)
        if is_qwen and "true_cfg_scale" in pipe_params:
            call_kwargs["true_cfg_scale"] = cfg_scale
            if "negative_prompt" not in call_kwargs and "negative_prompt_embeds" not in call_kwargs:
                call_kwargs["negative_prompt"] = negative_prompt or " "
        else:
            call_kwargs["guidance_scale"] = cfg_scale
            if "negative_prompt" in pipe_params and negative_prompt and "negative_prompt_embeds" not in call_kwargs:
                call_kwargs["negative_prompt"] = negative_prompt

        # CFG rescale — prevents overexposure at high CFG (primarily SDXL)
        if guidance_rescale and guidance_rescale > 0.0 and "guidance_rescale" in pipe_params:
            call_kwargs["guidance_rescale"] = guidance_rescale

        # CLIP skip — skip last N text encoder layers (SD1.5/SDXL).
        # transformers ≥ 5.0 dropped the `text_model` wrapper on
        # CLIPTextModel — `final_layer_norm` now lives directly on the
        # top-level module. Diffusers ≤ 0.38 still calls
        # `self.text_encoder.text_model.final_layer_norm(...)` inside the
        # clip_skip branch of every SD1.5-family pipeline (txt2img, img2img,
        # inpaint, upscale, PAG, ControlNet), so passing clip_skip to those
        # pipelines raises AttributeError before the first denoising step.
        # SDXL/Flux/etc. take a different code path and aren't affected.
        # Until either diffusers updates or we encode clip_skip ourselves,
        # gate the param on the attribute actually existing.
        if clip_skip and clip_skip > 0 and "clip_skip" in pipe_params:
            if _clip_skip_safe_for_pipe(pipe):
                call_kwargs["clip_skip"] = clip_skip
            else:
                log.warning(
                    "clip_skip_dropped_incompatible",
                    pipeline=type(pipe).__name__,
                    reason=("diffusers expects text_encoder.text_model.final_layer_norm "
                            "but transformers ≥ 5.0 flattened that attribute"),
                    requested=clip_skip,
                )

        return call_kwargs

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
        *,
        guidance_rescale: float = 0.0,
        hires_fix: bool = False,
        hires_scale: float = 1.5,
        hires_denoise: float = 0.5,
        clip_skip: int | None = None,
        ip_adapter_image=None,
        ip_adapter_scale: float = 0.55,
        step_callback=None,
    ) -> GenerationResult:
        if not self._pipe:
            raise RuntimeError("Pipeline not loaded")

        actual_seed = _resolve_seed(seed)

        if sampler:
            from augmentum.image.schedulers import apply_sampler
            apply_sampler(self._pipe, sampler)

        active_ip_adapter_image = (
            ip_adapter_image if _has_ip_adapter_reference(ip_adapter_image) else None
        )
        if active_ip_adapter_image is None and self._ip_adapter_loaded:
            log.info(
                "ip_adapter_unloading_for_plain_generation",
                pipeline=type(self._pipe).__name__,
            )
            await self.unload_ip_adapter()

        pipe_params = self._pipe_params

        def _generate():
            import torch

            if self._is_edit:
                raise RuntimeError(
                    f"{type(self._pipe).__name__} is an image editing model that requires "
                    f"a source image. Use img2img instead of txt2img."
                )

            generator = torch.Generator(device=self._device).manual_seed(actual_seed)
            call_kwargs = self._build_call_kwargs(
                self._pipe, pipe_params,
                prompt=prompt, negative_prompt=negative_prompt,
                cfg_scale=cfg_scale, steps=steps, generator=generator,
                width=width, height=height,
                guidance_rescale=guidance_rescale,
                clip_skip=clip_skip,
            )

            # Inject IP-Adapter reference image if provided and supported
            if active_ip_adapter_image is not None:
                if "ip_adapter_image" in pipe_params:
                    call_kwargs["ip_adapter_image"] = active_ip_adapter_image
                else:
                    log.info("ip_adapter_skipped", reason="pipeline does not accept ip_adapter_image")

            # Wire the diffusers per-step callback when the caller
            # provided one. Diffusers' modern API expects
            # ``callback_on_step_end(pipe, step_index, timestep, kwargs) → kwargs``.
            # We adapt our simpler ``step_callback(done, total)`` to it
            # so the queue-side instrumentation doesn't need to know
            # about diffusers internals. Gated on pipe_params because
            # GGUF / single-file pipelines built before the modern
            # callback API may not accept the kwarg.
            _step_total = call_kwargs.get("num_inference_steps", steps)
            if step_callback is not None and "callback_on_step_end" in pipe_params:
                def _diffusers_step_cb(_pipe, step_index, _ts, cb_kwargs):
                    try:
                        # step_index is 0-based; report done = index + 1
                        # because the step has just finished its denoise pass.
                        step_callback(step_index + 1, _step_total)
                    except Exception as exc:
                        # Progress reporting is best-effort — a bad
                        # callback must never break the diffusion loop.
                        log.debug("step_callback_error", error=str(exc))
                    return cb_kwargs
                call_kwargs["callback_on_step_end"] = _diffusers_step_cb

            with torch.inference_mode():
                result = self._pipe(**call_kwargs)
            return result.images[0]

        image = await _run_on_thread(_generate)

        # --- Hires fix: upscale + img2img refine ---
        if hires_fix and hires_scale > 1.0:
            image = await self._hires_fix(
                image, prompt, negative_prompt, cfg_scale, steps,
                actual_seed, sampler, hires_scale, hires_denoise,
                guidance_rescale=guidance_rescale,
            )

        _ensure_output_dir(output_dir)
        image_id = uuid.uuid4().hex[:16]
        file_path = os.path.join(output_dir, f"{image_id}.png")
        await asyncio.to_thread(image.save, file_path)

        final_w, final_h = image.size
        log.info(
            "image_generated",
            pipeline=type(self._pipe).__name__,
            seed=actual_seed,
            size=f"{final_w}x{final_h}",
            hires_fix=hires_fix,
        )
        return GenerationResult(
            image_id=image_id,
            file_path=file_path,
            seed=actual_seed,
            width=final_w,
            height=final_h,
        )

    async def _hires_fix(
        self,
        image,
        prompt: str,
        negative_prompt: str,
        cfg_scale: float,
        steps: int,
        seed: int,
        sampler: str,
        scale: float,
        denoise: float,
        *,
        guidance_rescale: float = 0.0,
    ):
        """Upscale with Lanczos then refine with img2img at low denoise."""
        from PIL import Image as PILImage

        new_w = int(image.width * scale)
        new_h = int(image.height * scale)
        # Round to nearest 8 for diffusion model compatibility
        new_w = (new_w // 8) * 8
        new_h = (new_h // 8) * 8

        log.info("hires_fix_upscaling", from_size=f"{image.width}x{image.height}", to_size=f"{new_w}x{new_h}")
        upscaled = image.resize((new_w, new_h), PILImage.LANCZOS)

        # Run img2img at the upscaled resolution
        result = await self.img2img(
            prompt=prompt,
            image=upscaled,
            negative_prompt=negative_prompt,
            strength=denoise,
            steps=steps,
            cfg_scale=cfg_scale,
            seed=seed,
            sampler=sampler,
            output_dir="/tmp",  # temp — caller saves the final image
        )

        # Load the refined image from disk and clean up temp file
        refined = PILImage.open(result.file_path).copy()
        try:
            os.remove(result.file_path)
        except OSError:
            pass
        return refined

    async def img2img(
        self,
        prompt: str,
        image,
        negative_prompt: str = "",
        strength: float = 0.75,
        steps: int = 20,
        cfg_scale: float = 7.0,
        seed: int = -1,
        sampler: str = "",
        output_dir: str = "/data/image_output",
        *,
        step_callback=None,
    ) -> GenerationResult:
        if not self._pipe:
            raise RuntimeError("Pipeline not loaded")

        actual_seed = _resolve_seed(seed)
        if sampler:
            from augmentum.image.schedulers import apply_sampler
            apply_sampler(self._pipe, sampler)

        def _run():
            import torch
            generator = torch.Generator(device=self._device).manual_seed(actual_seed)

            i2i_pipe = self._get_cached_variant("img2img")
            if i2i_pipe is not None:
                try:
                    pipe_params = set(inspect.signature(i2i_pipe.__call__).parameters)
                    extra: dict = {"image": image}
                    if "strength" in pipe_params:
                        extra["strength"] = strength
                    if "width" in pipe_params and hasattr(image, "width"):
                        extra["width"] = image.width
                    if "height" in pipe_params and hasattr(image, "height"):
                        extra["height"] = image.height
                    call_kwargs = self._build_call_kwargs(
                        i2i_pipe, pipe_params,
                        prompt=prompt, negative_prompt=negative_prompt,
                        cfg_scale=cfg_scale, steps=steps, generator=generator,
                        **extra,
                    )
                    # Per-step progress — see generate() for the
                    # adapter rationale; same shape for img2img.
                    if step_callback is not None and "callback_on_step_end" in pipe_params:
                        _step_total = call_kwargs.get("num_inference_steps", steps)
                        def _diffusers_step_cb(_pipe, step_index, _ts, cb_kwargs):
                            try:
                                step_callback(step_index + 1, _step_total)
                            except Exception as exc:
                                log.debug("step_callback_error", error=str(exc))
                            return cb_kwargs
                        call_kwargs["callback_on_step_end"] = _diffusers_step_cb
                    with torch.inference_mode():
                        result = i2i_pipe(**call_kwargs)
                    return result.images[0]
                except Exception:
                    log.debug("img2img_variant_failed", exc_info=True)
                    generator = torch.Generator(device=self._device).manual_seed(actual_seed)

            log.info("img2img_using_latent_fallback", pipeline=type(self._pipe).__name__)
            with torch.inference_mode():
                return _latent_img2img_fallback(
                    self._pipe, prompt, image, strength, steps,
                    cfg_scale, generator, self._device,
                )

        out_image = await _run_on_thread(_run)

        _ensure_output_dir(output_dir)
        image_id = uuid.uuid4().hex[:16]
        file_path = os.path.join(output_dir, f"{image_id}.png")
        await asyncio.to_thread(out_image.save, file_path)

        log.info("img2img_generated", pipeline=type(self._pipe).__name__, seed=actual_seed)
        return GenerationResult(
            image_id=image_id, file_path=file_path, seed=actual_seed,
            width=out_image.width, height=out_image.height,
        )

    async def inpaint(
        self,
        prompt: str,
        image,
        mask,
        negative_prompt: str = "",
        strength: float = 1.0,
        steps: int = 20,
        cfg_scale: float = 7.0,
        seed: int = -1,
        sampler: str = "",
        output_dir: str = "/data/image_output",
        *,
        step_callback=None,
    ) -> GenerationResult:
        if not self._pipe:
            raise RuntimeError("Pipeline not loaded")

        actual_seed = _resolve_seed(seed)
        if sampler:
            from augmentum.image.schedulers import apply_sampler
            apply_sampler(self._pipe, sampler)

        def _run():
            import torch
            from PIL import Image

            generator = torch.Generator(device=self._device).manual_seed(actual_seed)

            inpaint_pipe = self._get_cached_variant("inpaint")
            if inpaint_pipe is not None:
                try:
                    pipe_params = set(inspect.signature(inpaint_pipe.__call__).parameters)
                    call_kwargs = self._build_call_kwargs(
                        inpaint_pipe, pipe_params,
                        prompt=prompt, negative_prompt=negative_prompt,
                        cfg_scale=cfg_scale, steps=steps, generator=generator,
                        image=image, mask_image=mask, strength=strength,
                    )
                    # Per-step progress — same adapter as generate().
                    if step_callback is not None and "callback_on_step_end" in pipe_params:
                        _step_total = call_kwargs.get("num_inference_steps", steps)
                        def _diffusers_step_cb(_pipe, step_index, _ts, cb_kwargs):
                            try:
                                step_callback(step_index + 1, _step_total)
                            except Exception as exc:
                                log.debug("step_callback_error", error=str(exc))
                            return cb_kwargs
                        call_kwargs["callback_on_step_end"] = _diffusers_step_cb
                    with torch.inference_mode():
                        result = inpaint_pipe(**call_kwargs)
                    return result.images[0]
                except Exception:
                    log.debug("inpaint_variant_failed", exc_info=True)
                    generator = torch.Generator(device=self._device).manual_seed(actual_seed)

            log.info("inpaint_using_composite_fallback", pipeline=type(self._pipe).__name__)
            with torch.inference_mode():
                generated = _latent_img2img_fallback(
                    self._pipe, prompt, image, strength, steps,
                    cfg_scale, generator, self._device,
                )
            return Image.composite(generated, image, mask.convert("L"))

        out_image = await _run_on_thread(_run)

        _ensure_output_dir(output_dir)
        image_id = uuid.uuid4().hex[:16]
        file_path = os.path.join(output_dir, f"{image_id}.png")
        await asyncio.to_thread(out_image.save, file_path)

        log.info("inpaint_generated", pipeline=type(self._pipe).__name__, seed=actual_seed)
        return GenerationResult(
            image_id=image_id, file_path=file_path, seed=actual_seed,
            width=out_image.width, height=out_image.height,
        )

    async def load_lora(self, lora_path: str, weight: float = 1.0) -> None:
        if not self._pipe:
            raise RuntimeError("Pipeline not loaded")

        def _load_lora():
            if not hasattr(self._pipe, "load_lora_weights"):
                raise RuntimeError(
                    f"{type(self._pipe).__name__} does not support LoRA. "
                    "LoRA is supported on SD1.5, SDXL, and FLUX models."
                )
            self._pipe.load_lora_weights(lora_path)
            self._pipe.fuse_lora(lora_scale=weight)

        await asyncio.to_thread(_load_lora)
        # Invalidate variant cache since LoRA changes model behavior
        self._variant_cache.clear()
        log.info("lora_loaded", pipeline=type(self._pipe).__name__, path=lora_path, weight=weight)

    async def unload_loras(self) -> None:
        if not self._pipe:
            return

        def _unload():
            self._pipe.unfuse_lora()
            self._pipe.unload_lora_weights()

        await asyncio.to_thread(_unload)
        self._variant_cache.clear()
        log.info("loras_unloaded", pipeline=type(self._pipe).__name__)

    # ------------------------------------------------------------------
    # IP-Adapter
    # ------------------------------------------------------------------

    _IP_ADAPTER_WEIGHTS: dict = {
        # SD1.5: Plus variant — patch-level detail transfer, auto-loads ViT-H encoder
        PipelineType.SD15: ("models", "ip-adapter-plus_sd15.safetensors"),
        # SDXL: standard adapter matching the auto-downloaded ViT-bigG encoder.
        # ViT-H variants (ip-adapter_sdxl_vit-h) cause shape mismatch because
        # diffusers auto-downloads ViT-bigG (dim=1280) not ViT-H (dim=1024).
        PipelineType.SDXL: ("sdxl_models", "ip-adapter_sdxl.safetensors"),
        # Flux & SD3.5: no IP-Adapter support in mainline diffusers.
        # FLUX.1-Redux is a separate prior pipeline (different architecture).
    }

    async def load_ip_adapter(
        self,
        repo_id: str = "h94/IP-Adapter",
        scale: float = 0.55,
    ) -> None:
        """Load IP-Adapter weights for the current pipeline.

        Raises RuntimeError if the pipeline doesn't support IP-Adapter.
        """
        if not self._pipe:
            raise RuntimeError("Pipeline not loaded")

        if self._ip_adapter_loaded:
            # Already loaded — just update scale
            if hasattr(self._pipe, "set_ip_adapter_scale"):
                self._pipe.set_ip_adapter_scale(scale)
            return

        pt = self._detected_type
        weight_info = self._IP_ADAPTER_WEIGHTS.get(pt)
        if not weight_info:
            raise RuntimeError(
                f"IP-Adapter is not supported on {pt.value} pipelines. "
                "Use SD1.5 or SDXL for IP-Adapter reference images."
            )

        subfolder, weight_name = weight_info

        def _load():
            self._pipe.load_ip_adapter(
                repo_id, subfolder=subfolder, weight_name=weight_name,
            )
            self._pipe.set_ip_adapter_scale(scale)

        await asyncio.to_thread(_load)
        self._ip_adapter_loaded = True
        self._variant_cache.clear()
        # Re-cache pipe params since IP-Adapter adds new accepted params
        self._cache_pipe_metadata()
        log.info(
            "ip_adapter_loaded",
            pipeline=type(self._pipe).__name__,
            weight=weight_name,
            scale=scale,
        )

    async def unload_ip_adapter(self) -> None:
        """Unload IP-Adapter weights and free memory."""
        if not self._ip_adapter_loaded or not self._pipe:
            return
        if not hasattr(self._pipe, "unload_ip_adapter"):
            self._ip_adapter_loaded = False
            return

        def _unload():
            self._pipe.unload_ip_adapter()

        await asyncio.to_thread(_unload)
        self._ip_adapter_loaded = False
        self._variant_cache.clear()
        self._cache_pipe_metadata()
        log.info("ip_adapter_unloaded")
