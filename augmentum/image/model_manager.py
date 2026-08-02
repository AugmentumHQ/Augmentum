"""Model manager — download, import, and manage image generation models."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from collections.abc import AsyncIterator
from pathlib import Path

from augmentum.image.schemas import PipelineType
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# Eager-import the huggingface_hub submodules our async paths use. The
# library has a circular dep between hf_api and utils._xet that only
# manifests when those submodules are imported lazily from a background
# coroutine (the jobs-queue dispatch path does exactly that) — surfacing as
# "cannot import name 'XetConnectionInfo' from partially initialized module
# 'huggingface_hub.utils._xet'" and failing every image pull. Forcing the
# full import chain at module load resolves the order once, on the main
# thread, so subsequent in-handler imports hit the warmed cache.
try:
    import huggingface_hub  # noqa: F401
    from huggingface_hub import HfApi as _HfApi  # noqa: F401
    from huggingface_hub import hf_hub_download as _hf_hub_download  # noqa: F401
    from huggingface_hub.utils import _xet as _hf_xet_module  # noqa: F401
except ImportError:
    # Falls back to the in-function lazy import which surfaces the original
    # "huggingface_hub not installed" error to the user.
    pass


class _ByteProgressSink:
    """Thread-safe holder for the current file's downloaded byte count.

    Written by the huggingface_hub tqdm hook (which runs on the
    ``asyncio.to_thread`` download worker) and read by the async generator
    that yields progress events. Monotonic ``int`` writes are atomic under
    the GIL, so no lock is needed — we only ever take the max.
    """

    __slots__ = ("current",)

    def __init__(self) -> None:
        self.current = 0

    def report(self, n: int) -> None:
        if n and n > self.current:
            self.current = n


def _make_hf_progress_tqdm(sink: _ByteProgressSink):
    """Build an hf-tqdm subclass that streams byte counts into *sink*.

    huggingface_hub drives a progress bar for every download backend (Xet,
    hf_transfer, and the plain-python fallback) through the ``tqdm_class``
    it's handed — but in a non-TTY/server context it instantiates that class
    with ``disable=True``, so ``tqdm.update()`` short-circuits and ``.n``
    never advances. We force ``disable=False`` so the counter advances, then
    suppress the visual rendering (``display`` no-op) so nothing is printed
    to the server logs. Without this, single-file HF pulls show 0% for the
    whole download and then lurch to 100% (the disk-size monitor can't see
    Xet's burst-reconstruct staging). Must subclass huggingface_hub's own
    tqdm wrapper — it accepts the ``name=`` kwarg hf passes (e.g.
    ``huggingface_hub.xet_get``) that plain ``tqdm`` rejects.
    """
    from huggingface_hub.utils import tqdm as _hf_tqdm

    class _ProgressTqdm(_hf_tqdm):  # type: ignore[misc, valid-type]
        def __init__(self, *args, **kwargs):
            kwargs["disable"] = False
            super().__init__(*args, **kwargs)
            self.disable = False
            # Only byte-unit bars carry download progress; hf may also spin
            # up count bars ("Fetching N files") we must ignore.
            self._track = getattr(self, "unit", None) == "B"

        def update(self, n=1):
            r = super().update(n)
            if self._track:
                sink.report(self.n)
            return r

        def display(self, *args, **kwargs):
            # Suppress terminal rendering — we only want the counter.
            return None

    return _ProgressTqdm


def detect_model_capabilities(
    model_path: str,
    model_name: str,
    pipeline_type: PipelineType,
) -> dict[str, str]:
    """Detect txt2img / img2img / inpaint capability for a local model.

    Returns dict with keys ``txt2img``, ``img2img``, ``inpaint``, each
    valued ``"yes"`` (native support), ``"fallback"`` (latent-space
    workaround with lower quality), or ``"no"`` (not possible).

    Detection order:
    1. Known model name patterns (most reliable for catalog models).
    2. UNet ``in_channels == 9`` → native SD/SDXL inpaint checkpoint.
    3. Pipeline class name from model_index.json.
    4. Default by pipeline type.
    """
    caps = {"txt2img": "yes", "img2img": "yes", "inpaint": "fallback"}
    name_lower = model_name.lower()

    # --- 1. Known model name patterns ---
    # Edit-only models (cannot do txt2img)
    if any(k in name_lower for k in (
        "qwen-image-edit", "qwen_image_edit",
    )):
        return {"txt2img": "no", "img2img": "yes", "inpaint": "yes"}

    # FLUX Fill — dedicated inpaint model
    if any(k in name_lower for k in ("flux-fill", "flux.1-fill", "flux_fill")):
        return {"txt2img": "no", "img2img": "no", "inpaint": "yes"}

    # FLUX.2 Klein — native img2img, no inpaint
    if "klein" in name_lower and ("flux" in name_lower or "flux.2" in name_lower or "flux2" in name_lower):
        return {"txt2img": "yes", "img2img": "yes", "inpaint": "no"}

    # FLUX Kontext — editing model with inpaint support
    if "kontext" in name_lower:
        return {"txt2img": "yes", "img2img": "yes", "inpaint": "yes"}

    # Qwen-Image (non-edit) — has native img2img and inpaint pipelines
    if any(k in name_lower for k in ("qwen-image", "qwen_image")) and "edit" not in name_lower and "layered" not in name_lower:
        return {"txt2img": "yes", "img2img": "yes", "inpaint": "yes"}

    # Z-Image and Z-Anime (fine-tune) — native img2img + inpaint via ZImagePipeline variants
    if any(k in name_lower for k in (
        "z-image", "z_image", "zimage",
        "z-anime", "z_anime", "zanime",
    )):
        return {"txt2img": "yes", "img2img": "yes", "inpaint": "yes"}

    # SD / SDXL inpainting checkpoints (by name convention)
    if "inpaint" in name_lower:
        return {"txt2img": "yes", "img2img": "yes", "inpaint": "yes"}

    # --- 2. UNet in_channels detection (SD/SDXL) ---
    unet_config_path = os.path.join(model_path, "unet", "config.json")
    if os.path.exists(unet_config_path):
        try:
            with open(unet_config_path) as f:
                unet_cfg = json.load(f)
            if unet_cfg.get("in_channels", 4) == 9:
                # 9-channel UNet = native inpaint model
                return {"txt2img": "yes", "img2img": "yes", "inpaint": "yes"}
        except (OSError, json.JSONDecodeError) as exc:
            log.debug("unet_config_read_failed", path=unet_config_path, error=str(exc))

    # --- 3. Pipeline class name from model_index.json ---
    config_path = os.path.join(model_path, "model_index.json")
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                config = json.load(f)
            class_name = config.get("_class_name", "")

            # Inpaint pipeline classes
            if "Inpaint" in class_name:
                return {"txt2img": "yes", "img2img": "yes", "inpaint": "yes"}

            # Edit pipeline classes (cannot do txt2img)
            if "Edit" in class_name:
                return {"txt2img": "no", "img2img": "yes", "inpaint": "yes"}

            # Img2Img-only pipelines
            if "Img2Img" in class_name:
                return {"txt2img": "no", "img2img": "yes", "inpaint": "fallback"}

            # ControlNet pipelines — structural guidance, not inpaint
            if "ControlNet" in class_name or "Canny" in class_name or "Depth" in class_name:
                return {"txt2img": "no", "img2img": "yes", "inpaint": "no"}

            # Redux — variation only
            if "Redux" in class_name:
                return {"txt2img": "no", "img2img": "yes", "inpaint": "no"}

            # Fill pipelines (FLUX Fill)
            if "Fill" in class_name:
                return {"txt2img": "no", "img2img": "no", "inpaint": "yes"}

        except (OSError, json.JSONDecodeError) as exc:
            log.debug("model_index_read_failed", path=config_path, error=str(exc))

    # --- 4. GGUF models — check gguf_meta.json or name patterns ---
    gguf_meta_path = os.path.join(model_path, "gguf_meta.json")
    if os.path.exists(gguf_meta_path):
        try:
            with open(gguf_meta_path) as f:
                meta = json.load(f)
            pipe_class = meta.get("gguf_pipeline_class", "")
            if "Edit" in pipe_class:
                return {"txt2img": "no", "img2img": "yes", "inpaint": "yes"}
            if "Fill" in pipe_class:
                return {"txt2img": "no", "img2img": "no", "inpaint": "yes"}
        except (OSError, json.JSONDecodeError) as exc:
            log.debug("gguf_meta_read_failed", path=gguf_meta_path, error=str(exc))

    # --- 5. Default by pipeline type ---
    # SD15/SDXL: img2img via variant, inpaint via latent fallback
    # FLUX/transformer: img2img via variant, inpaint via latent fallback
    return caps


def _detect_pipeline_type(model_path: str) -> PipelineType:
    """Detect the pipeline type from a model's config files.

    This is a best-effort pre-detection used for VRAM checks, default
    negative prompts, and skip-if-loaded optimization.  The actual
    architecture is authoritatively detected at load time by
    ``UnifiedPipeline._detect_type_from_pipe()``.
    """
    config_path = os.path.join(model_path, "model_index.json")
    if os.path.exists(config_path):
        try:
            with open(config_path) as f:
                config = json.load(f)
            class_name = config.get("_class_name", "")

            # Transformer-based architectures → FLUX bucket
            if any(k in class_name for k in (
                "Flux", "Lumina", "StableDiffusion3",
                "PixArt", "HunyuanDiT", "Kandinsky3",
                "Wuerstchen", "AuraFlow", "CogView",
            )):
                return PipelineType.FLUX

            if "StableDiffusionXL" in class_name:
                return PipelineType.SDXL

            if "StableDiffusion" in class_name:
                return PipelineType.SD15
        except Exception:
            log.debug("model_index_parse_failed", path=config_path, exc_info=True)

    # Check for transformer directory (FLUX, SD3, Lumina, etc.)
    transformer_config = os.path.join(model_path, "transformer", "config.json")
    if os.path.exists(transformer_config):
        return PipelineType.FLUX

    # Check for SDXL indicators in unet config
    unet_config = os.path.join(model_path, "unet", "config.json")
    if os.path.exists(unet_config):
        try:
            with open(unet_config) as f:
                unet = json.load(f)
            # SDXL unet has cross_attention_dim of 2048
            if unet.get("cross_attention_dim", 0) >= 2048:
                return PipelineType.SDXL
        except Exception:
            log.debug("unet_config_parse_failed", path=unet_config, exc_info=True)

    # GGUF single-file models — typically transformer-based (Qwen, FLUX,
    # Z-Image, Z-Anime, etc.). Two signals:
    #   1. ``gguf_meta.json`` sidecar (written by _save_gguf_meta_if_needed
    #      / _write_gguf_meta for any catalog GGUF or custom import) is the
    #      strongest signal — if present, we already know this is a GGUF
    #      with a transformer-based pipeline class.
    #   2. ``.gguf`` file at the top level OR one level deep (e.g.
    #      ``<model>/gguf/foo.gguf`` — SeeSee21/Z-Anime, FLUX-Fill, etc.).
    #      Earlier this only walked the top level, which mis-classified
    #      Z-Anime as SD15 because its GGUF lives under ``gguf/``.
    if isinstance(model_path, str) and model_path:
        if model_path.lower().endswith(".gguf"):
            return PipelineType.FLUX
        if os.path.isdir(model_path):
            if os.path.exists(os.path.join(model_path, "gguf_meta.json")):
                return PipelineType.FLUX
            try:
                top_entries = os.listdir(model_path)
            except OSError:
                top_entries = []
            for entry in top_entries:
                full = os.path.join(model_path, entry)
                if os.path.isfile(full) and entry.endswith(".gguf"):
                    return PipelineType.FLUX
                if os.path.isdir(full) and not entry.startswith("."):
                    try:
                        if any(f.endswith(".gguf") for f in os.listdir(full)):
                            return PipelineType.FLUX
                    except OSError:
                        continue

    return PipelineType.SD15


def _get_dir_size(path: str) -> int:
    """Get total size of a directory in bytes."""
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            total += os.path.getsize(fp)
    return total


class ModelManager:
    """Manages image model downloads and local storage.

    Two roots are scanned for installed models:

    * ``model_dir`` — the user-writable directory (default
      ``{data_dir}/image_models``). Downloads and deletions go here.
    * ``system_dir`` — an optional read-only directory available for
      models bundled into the Docker image at build time (e.g.
      ``/opt/augmentum/image_models``). Currently unused — nothing is
      pre-baked; the hook remains for any future bundle. Paths under
      this dir survive the ``/data`` volume mount because they're
      outside it.

    Models in both dirs are merged in :meth:`list_local_models`; user
    copies shadow system copies on name collision so a user re-download
    can override a baked default. :meth:`delete_model` only touches the
    user dir — system models are immutable.
    """

    def __init__(self, model_dir: str, system_dir: str | None = None) -> None:
        self._model_dir = model_dir
        self._system_dir = system_dir if system_dir and os.path.isdir(system_dir) else None
        Path(model_dir).mkdir(parents=True, exist_ok=True)

    @property
    def model_dir(self) -> str:
        return self._model_dir

    @property
    def system_dir(self) -> str | None:
        return self._system_dir

    def _scan_dir(self, root: str, *, read_only: bool) -> list[dict]:
        """Return a list of model dicts found under *root*."""
        out: list[dict] = []
        if not os.path.isdir(root):
            return out
        for entry in os.scandir(root):
            if not entry.is_dir() or not self._has_model_files(entry.path):
                continue
            from augmentum.image.distilled import detect_distilled_type

            pipeline_type = _detect_pipeline_type(entry.path)
            distilled = detect_distilled_type(entry.name)
            caps = detect_model_capabilities(entry.path, entry.name, pipeline_type)
            out.append({
                "name": entry.name,
                "path": entry.path,
                "pipeline_type": pipeline_type,
                "size_bytes": _get_dir_size(entry.path),
                "source": "system" if read_only else "local",
                "read_only": read_only,
                "distilled_type": distilled or "",
                "capabilities": caps,
            })
        return out

    def list_local_models(self) -> list[dict]:
        """List all locally available models (user + image-baked system)."""
        # Scan system first so user models can override system ones via name collision.
        models = self._scan_dir(self._system_dir, read_only=True) if self._system_dir else []
        models.extend(self._scan_dir(self._model_dir, read_only=False))

        # Dedupe by name — keep the *last* occurrence so user-installed shadows system.
        by_name: dict[str, dict] = {}
        for m in models:
            by_name[m["name"]] = m
        return list(by_name.values())

    @staticmethod
    def _has_model_files(path: str) -> bool:
        """Check if a directory contains any model weight files (no size check).

        Used for listing — more lenient than ``_is_valid_model_dir`` which
        enforces a minimum file size to reject partial downloads.
        """
        if not os.path.isdir(path):
            return False
        if os.path.exists(os.path.join(path, "model_index.json")):
            return True
        for f in os.listdir(path):
            if os.path.isfile(os.path.join(path, f)):
                ext = os.path.splitext(f)[1].lower()
                if ext in ModelManager._WEIGHT_EXTENSIONS:
                    return True
        # Check one level deep (diffusers component layout)
        for sub in os.listdir(path):
            subpath = os.path.join(path, sub)
            if not os.path.isdir(subpath) or sub.startswith("."):
                continue
            for f in os.listdir(subpath):
                if os.path.isfile(os.path.join(subpath, f)):
                    ext = os.path.splitext(f)[1].lower()
                    if ext in ModelManager._WEIGHT_EXTENSIONS:
                        return True
        return False

    _WEIGHT_EXTENSIONS: set[str] = {".safetensors", ".gguf", ".bin", ".pt"}

    @staticmethod
    def _is_valid_model_dir(path: str) -> bool:
        """Check if a directory contains actual model files.

        Weight files must be at least 1 MB to avoid treating partial
        downloads as valid models.  Recognises safetensors, GGUF,
        bin/pt weight files, and standard diffusers model_index.json.
        """
        if not os.path.isdir(path):
            return False
        model_index = os.path.join(path, "model_index.json")
        if os.path.exists(model_index):
            return True
        min_size = 1024 * 1024  # 1 MB
        for f in os.listdir(path):
            fp = os.path.join(path, f)
            if not os.path.isfile(fp):
                continue
            ext = os.path.splitext(f)[1].lower()
            if ext in ModelManager._WEIGHT_EXTENSIONS:
                try:
                    if os.path.getsize(fp) >= min_size:
                        return True
                except OSError:
                    pass
        # Also check subdirectories one level deep (diffusers component layout)
        for sub in os.listdir(path):
            subpath = os.path.join(path, sub)
            if not os.path.isdir(subpath) or sub.startswith("."):
                continue
            for f in os.listdir(subpath):
                fp = os.path.join(subpath, f)
                if not os.path.isfile(fp):
                    continue
                ext = os.path.splitext(f)[1].lower()
                if ext in ModelManager._WEIGHT_EXTENSIONS:
                    try:
                        if os.path.getsize(fp) >= min_size:
                            return True
                    except OSError:
                        pass
        return False

    # hf_hub_download stores partial downloads as
    # ``.cache/huggingface/download/<hash>.incomplete`` and resumes from
    # there on the next call. Augmentum used to ``shutil.rmtree`` the whole
    # dest dir whenever it looked invalid, which took the cache (and
    # multi-GB of progress) with it. This threshold ignores empty stubs
    # that hf creates as soon as a download starts.
    _RESUMABLE_PARTIAL_MIN_BYTES: int = 1024 * 1024  # 1 MB

    @classmethod
    def _has_resumable_partial(cls, dest_path: str) -> bool:
        """Detect a meaningful HuggingFace ``.incomplete`` partial in *dest_path*.

        Returns True if ``.cache/huggingface/download`` contains at least one
        ``*.incomplete`` file larger than :attr:`_RESUMABLE_PARTIAL_MIN_BYTES`.
        Callers should skip ``rmtree`` when this is True so the next
        ``hf_hub_download`` call resumes rather than re-fetches.
        """
        cache_dir = os.path.join(dest_path, ".cache", "huggingface", "download")
        if not os.path.isdir(cache_dir):
            return False
        for dirpath, _dirs, files in os.walk(cache_dir):
            for f in files:
                if not f.endswith(".incomplete"):
                    continue
                try:
                    if os.path.getsize(os.path.join(dirpath, f)) >= cls._RESUMABLE_PARTIAL_MIN_BYTES:
                        return True
                except OSError:
                    continue
        return False

    @staticmethod
    def _clean_invalid_keep_cache(dest_path: str) -> None:
        """Remove top-level garbage from *dest_path* but preserve ``.cache/``.

        The catalog/URL pull path used to ``shutil.rmtree`` the whole dest
        dir as cleanup for "looks invalid". That nuked any ``.cache/`` state
        too — including Xet/LFS partials below the
        :attr:`_RESUMABLE_PARTIAL_MIN_BYTES` threshold that
        ``_has_resumable_partial`` can't see but ``hf_hub_download`` would
        otherwise know how to resume from. By scoping the cleanup to
        non-cache entries, we let hf own its own state and only nuke what
        we know is ours-and-stale.
        """
        if not os.path.isdir(dest_path):
            return
        for item in os.listdir(dest_path):
            if item == ".cache":
                continue
            full = os.path.join(dest_path, item)
            try:
                if os.path.isdir(full) and not os.path.islink(full):
                    shutil.rmtree(full, ignore_errors=True)
                else:
                    os.remove(full)
            except OSError:
                log.debug("clean_invalid_entry_failed", path=full, exc_info=True)

    def get_model_path(self, name: str) -> str | None:
        """Return the local path for a valid model, or None if not found.

        Checks the user model_dir first, then the read-only system_dir
        (image-baked defaults). Each lookup also tries the HF-slug form
        (``foo/bar`` → ``foo--bar``) so callers can pass repo IDs.
        """
        for safe in (name, name.replace("/", "--")):
            path = self._safe_model_path(safe)
            if path and os.path.exists(path) and self._is_valid_model_dir(path):
                return path
            sys_path = self._safe_system_path(safe)
            if sys_path and os.path.exists(sys_path) and self._is_valid_model_dir(sys_path):
                return sys_path
        return None

    def _safe_model_path(self, name: str) -> str | None:
        """Resolve a model name to a path, blocking traversal outside model_dir."""
        candidate = os.path.normpath(os.path.join(self._model_dir, name))
        # Ensure the resolved path is actually inside the model directory
        if not candidate.startswith(os.path.normpath(self._model_dir) + os.sep):
            log.warning("path_traversal_blocked", name=name, resolved=candidate)
            return None
        return candidate

    def _safe_system_path(self, name: str) -> str | None:
        """Resolve a model name to a path inside the read-only system_dir."""
        if not self._system_dir:
            return None
        candidate = os.path.normpath(os.path.join(self._system_dir, name))
        if not candidate.startswith(os.path.normpath(self._system_dir) + os.sep):
            log.warning("path_traversal_blocked", name=name, resolved=candidate)
            return None
        return candidate

    @staticmethod
    def _resolve_hf_token(token: str | None) -> str | None:
        """Resolve HuggingFace token: explicit > env var > cached login."""
        if token:
            return token
        # Check HF_TOKEN / HUGGING_FACE_HUB_TOKEN env vars
        env_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if env_token:
            return env_token
        # Fall back to huggingface-cli cached token
        try:
            from huggingface_hub import HfFolder
            return HfFolder.get_token()
        except Exception:
            return None

    @staticmethod
    def _select_variant(siblings) -> str | None:
        """Pick the best weight variant available in the repo.

        Preference order: fp16 > bf16 > fp8 > (no variant / full precision).
        Returns None if no variant-specific files are found (download all weights).
        """
        variant_scores = {"fp16": 3, "bf16": 2, "fp8": 1}
        found: dict[str, int] = {}
        for s in siblings:
            name = s.rfilename
            if not name.endswith(".safetensors"):
                continue
            for variant, score in variant_scores.items():
                if f".{variant}." in name or name.endswith(f".{variant}.safetensors"):
                    found[variant] = found.get(variant, 0) + score
        if not found:
            return None
        return max(found, key=lambda v: found[v])

    # File patterns to always skip — not needed for PyTorch/diffusers inference
    _IGNORE_PATTERNS: list[str] = [
        # Redundant weight formats
        "*.ckpt",              # old checkpoint format (redundant with safetensors)
        "*.bin",               # pytorch_model.bin (redundant with safetensors)
        "*.pt",                # raw pytorch checkpoints
        "*.pth",               # pytorch training checkpoints
        # Alternative runtime formats
        "*.onnx",              # ONNX runtime
        "*.onnx_data",         # ONNX external data
        "*.msgpack",           # Flax/JAX format
        "openvino_model.*",    # Intel OpenVINO
        # Training artifacts
        "optimizer*",          # optimizer states
        "training_args*",      # training config
        "logs/*",              # training logs
        # Non-model files
        "*.md",                # readmes
        "*.png",               # example images
        "*.jpg",               # example images
        "*.jpeg",              # example images
        "*.webp",              # example images
        "*.gif",               # example images
        "*.py",                # conversion/utility scripts
        "*.xml",               # OpenVINO IR configs
        ".gitattributes",
    ]

    # File patterns to always include regardless of variant filtering
    _ALWAYS_INCLUDE: list[str] = [
        "*.json",              # config files (model_index.json, config.json, etc.)
        "*.txt",               # tokenizer vocab, merges
        "*.model",             # sentencepiece models
        "tokenizer*",          # tokenizer files
        "scheduler/*",         # scheduler config
        "feature_extractor/*", # feature extractor config
    ]

    async def pull_from_huggingface(
        self,
        repo_id: str,
        name: str = "",
        token: str | None = None,
        allow_patterns: list[str] | None = None,
        variant: str = "",
    ) -> AsyncIterator[dict]:
        """Download a model from HuggingFace Hub with streaming progress.

        Only downloads files needed for inference: config files, tokenizer
        files, and a single set of safetensors weights (preferring fp16
        variant when available).  Training artifacts, optimizer states,
        and redundant weight formats are skipped.

        If *allow_patterns* is provided (e.g. ``["Unet/v4/*", "Vae/*",
        "Text_Encoder/*", "*.json"]``), only files matching at least one
        pattern are downloaded.  This is useful for multi-version repos
        where the caller knows which subset of files they need.

        If *variant* is provided (e.g. ``"fp16"`` or ``"fp32"``), it
        overrides the automatic precision selection.  ``"fp32"`` forces
        full-precision weights.
        """
        try:
            from huggingface_hub import HfApi
        except ImportError:
            yield {
                "status": "error",
                "error": "huggingface_hub is not installed. Run: pip install huggingface-hub",
            }
            return

        # Resolve token: explicit > env var > cached login
        resolved_token = self._resolve_hf_token(token)

        model_name = name or repo_id.replace("/", "--")
        dest_path = os.path.join(self._model_dir, model_name)

        # Only treat as "exists" if the directory contains actual model files
        if os.path.exists(dest_path) and self._is_valid_model_dir(dest_path):
            yield {"status": "exists", "name": model_name, "path": dest_path}
            return

        # Clean up empty/invalid directories from prior failed downloads —
        # but ALWAYS preserve ``.cache/`` so hf_hub_download owns its own
        # resume state. The previous "rmtree the whole dest" approach
        # destroyed multi-GB partials whenever the visible .incomplete
        # file dipped below our 1 MB heuristic (e.g. Xet-backed transfers
        # that stage data outside the legacy .incomplete file).
        resuming = False
        if os.path.exists(dest_path) and not self._is_valid_model_dir(dest_path):
            if self._has_resumable_partial(dest_path):
                log.info("resuming_partial_download", path=dest_path)
                resuming = True
            else:
                log.info("cleaning_invalid_model_dir", path=dest_path, mode="keep_cache")
                await asyncio.to_thread(self._clean_invalid_keep_cache, dest_path)

        # Fetch file listing so we can filter before downloading.
        # files_metadata=True is required to populate LFS file sizes — without
        # it, weight-file `size` is None and the progress bar stays at 0/0.
        api = HfApi(token=resolved_token)
        try:
            repo_info = await asyncio.to_thread(
                lambda: api.repo_info(repo_id, files_metadata=True),
            )
            siblings = repo_info.siblings or []
        except Exception as exc:
            log.debug("hf_repo_info_failed", repo_id=repo_id, exc_info=True)
            yield {"status": "error", "error": f"Failed to access repository: {exc}"}
            return

        # Filter to only inference-required files
        filtered = self._filter_inference_files(
            siblings, allow_patterns=allow_patterns, variant_override=variant,
        )
        total_size = sum(s.size or 0 for s in filtered)
        file_count = len(filtered)

        original_count = len(siblings)
        original_size = sum(s.size or 0 for s in siblings)
        if original_count != file_count:
            log.info(
                "hf_download_filtered",
                name=model_name,
                original_files=original_count,
                filtered_files=file_count,
                original_size_mb=round(original_size / 1024 / 1024),
                filtered_size_mb=round(total_size / 1024 / 1024),
                skipped_files=original_count - file_count,
            )

        log.info(
            "hf_download_starting",
            name=model_name,
            source=repo_id,
            file_count=file_count,
            total_size_bytes=total_size,
            total_size_gb=round(total_size / (1024**3), 2) if total_size else 0,
            has_token=bool(resolved_token),
            resumed=resuming,
        )

        yield {
            "status": "downloading",
            "name": model_name,
            "source": repo_id,
            "total_size": total_size,
            "file_count": file_count,
            "_dest_path": dest_path,
            "resumed": resuming,
            # The download loop below yields real intra-file byte progress
            # via the hf tqdm hook, so the caller's disk-size monitor (a
            # workaround that can't see Xet's burst staging) is redundant
            # and should stay dormant.
            "native_progress": True,
        }

        # Download file-by-file for progress reporting
        from huggingface_hub import hf_hub_download  # noqa: E402

        Path(dest_path).mkdir(parents=True, exist_ok=True)
        downloaded_size = 0
        downloaded_files = 0

        try:
            for sibling in filtered:
                fname = sibling.rfilename
                fsize = sibling.size or 0

                # Per-file sink + tqdm hook so we can stream intra-file byte
                # progress while hf_hub_download blocks on the worker thread.
                sink = _ByteProgressSink()
                tqdm_cls = _make_hf_progress_tqdm(sink)

                def _dl_file(filename=fname, tq=tqdm_cls):
                    return hf_hub_download(
                        repo_id,
                        filename=filename,
                        local_dir=dest_path,
                        token=resolved_token,
                        tqdm_class=tq,
                    )

                dl_fut = asyncio.ensure_future(asyncio.to_thread(_dl_file))
                # Poll the live byte counter until the file finishes. This is
                # what keeps the progress bar moving during a single large
                # weight file (e.g. a 2GB single-file checkpoint) instead of
                # sitting at 0% until the whole file lands.
                while not dl_fut.done():
                    await asyncio.sleep(0.5)
                    cur = min(sink.current, fsize) if fsize else sink.current
                    seen = downloaded_size + cur
                    percent = round(seen / total_size * 100, 1) if total_size > 0 else 0
                    yield {
                        "status": "progress",
                        "downloaded": seen,
                        "total": total_size,
                        "percent": percent,
                        "file": fname,
                        "files_done": downloaded_files,
                        "files_total": file_count,
                    }
                await dl_fut  # propagate any download exception to the except below

                downloaded_size += fsize
                downloaded_files += 1

                percent = round(downloaded_size / total_size * 100, 1) if total_size > 0 else 0
                yield {
                    "status": "progress",
                    "downloaded": downloaded_size,
                    "total": total_size,
                    "percent": percent,
                    "file": fname,
                    "files_done": downloaded_files,
                    "files_total": file_count,
                }
        except Exception as exc:
            # Preserve ``.cache/`` so the next attempt can resume — even on
            # error. Earlier this branch did a blanket ``shutil.rmtree``,
            # which destroyed multi-GB partials whenever a transient
            # exception interrupted the pull (network blip, circular
            # import in huggingface_hub, etc.). Now we only clear top-level
            # garbage so hf owns its own resume state across attempts.
            if os.path.exists(dest_path):
                try:
                    await asyncio.to_thread(self._clean_invalid_keep_cache, dest_path)
                except OSError:
                    log.debug("partial_download_cleanup_failed", path=dest_path, exc_info=True)
            log.info("preserving_cache_on_error", path=dest_path)
            yield {"status": "error", "error": str(exc)}
            return

        pipeline_type = _detect_pipeline_type(dest_path)
        size_bytes = _get_dir_size(dest_path)

        yield {
            "status": "complete",
            "name": model_name,
            "path": dest_path,
            "pipeline_type": pipeline_type.value,
            "size_bytes": size_bytes,
        }
        log.info("model_downloaded", name=model_name, source=repo_id, size_bytes=size_bytes)

    # Subdirectories that only contain non-PyTorch formats (skip entirely)
    _SKIP_SUBDIRS: set[str] = {"vae_decoder", "vae_encoder"}

    def _filter_inference_files(
        self, siblings, *, allow_patterns: list[str] | None = None,
        variant_override: str = "",
    ) -> list:
        """Filter HuggingFace repo files to only those needed for inference.

        Strategy:
        1. If *allow_patterns* given, only keep files matching at least one
        2. Always skip training artifacts, ONNX, Flax, OpenVINO formats
        3. Always include config/tokenizer files (.json, .txt, .model)
        4. For safetensors: pick the best variant (fp16 > bf16 > fp8 > full)
           and skip all other variants to avoid downloading redundant weights
        5. Skip root-level single-file checkpoints when component subdirs exist
        6. Skip ONNX-only subdirectories (vae_decoder, vae_encoder)

        If *variant_override* is provided (e.g. ``"fp16"`` or ``"fp32"``),
        it overrides the automatic variant selection.  ``"fp32"`` forces
        full-precision weights (no variant tag in filenames).
        """
        import fnmatch

        # If caller specified explicit patterns, apply them first — but union
        # with _ALWAYS_INCLUDE so a narrow weight-only filter (e.g.
        # ["*Q4_K_M*"] for a GGUF catalog entry) still keeps config.json,
        # tokenizer, scheduler, and feature_extractor files when the repo
        # ships them. Without the union, those are dropped here and the
        # model dir ends up containing only the weight blob — leading to
        # JSONDecodeError / "missing config.json" at inference time.
        if allow_patterns:
            effective_patterns = list(allow_patterns) + list(self._ALWAYS_INCLUDE)
            siblings = [
                s for s in siblings
                if any(
                    fnmatch.fnmatch(s.rfilename, pat)
                    or fnmatch.fnmatch(s.rfilename.split("/")[-1], pat)
                    for pat in effective_patterns
                )
            ]

        # Detect standard diffusers layout: model_index.json + component subdirs.
        # In this case, root-level .safetensors are single-file checkpoints
        # (redundant with the component files) and should be skipped.
        has_model_index = any(s.rfilename == "model_index.json" for s in siblings)
        has_component_layout = has_model_index and any(
            "/" in s.rfilename and s.rfilename.endswith(".safetensors")
            for s in siblings
        )

        # Determine which variant to download
        force_full_precision = False
        if variant_override and variant_override != "fp32":
            # Explicit user choice (fp16, bf16, etc.) — use it directly
            variant = variant_override
            log.info("hf_variant_override", variant=variant)
        elif variant_override == "fp32":
            # User wants full precision — skip all variant-tagged files
            variant = None
            force_full_precision = True
            log.info("hf_variant_override", variant="fp32 (full precision)")
        else:
            # Auto-detect best variant
            variant = self._select_variant(siblings)
        if variant:
            log.info("hf_variant_selected", variant=variant)

        filtered = []
        for s in siblings:
            name = s.rfilename

            # Skip files in ONNX-only subdirectories
            top_dir = name.split("/")[0] if "/" in name else ""
            if top_dir in self._SKIP_SUBDIRS:
                continue

            # Always skip ignored patterns
            if any(fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(name.split("/")[-1], pat)
                   for pat in self._IGNORE_PATTERNS):
                continue

            # Always include config/tokenizer files
            if any(fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(name.split("/")[-1], pat)
                   for pat in self._ALWAYS_INCLUDE):
                filtered.append(s)
                continue

            # For safetensors: apply variant + layout filtering
            if name.endswith(".safetensors"):
                # Skip root-level single-file checkpoints when components exist
                if has_component_layout and "/" not in name:
                    continue

                # Force full precision: skip any variant-tagged files
                if force_full_precision:
                    has_any_variant = any(
                        f".{v}." in name for v in ("fp16", "bf16", "fp8")
                    )
                    if has_any_variant:
                        continue  # Skip variant files, keep only base files

                if variant:
                    # Include files matching our variant OR files with no variant tag
                    has_any_variant = any(
                        f".{v}." in name for v in ("fp16", "bf16", "fp8")
                    )
                    if has_any_variant and f".{variant}." not in name:
                        continue  # Wrong variant, skip
                    if not has_any_variant:
                        # No variant tag — only include if no variant-specific
                        # files exist for this component (fallback)
                        subdir = "/".join(name.split("/")[:-1]) if "/" in name else ""
                        has_variant_in_subdir = any(
                            s2.rfilename.endswith(".safetensors")
                            and f".{variant}." in s2.rfilename
                            and (("/".join(s2.rfilename.split("/")[:-1]) if "/" in s2.rfilename else "") == subdir)
                            for s2 in siblings
                        )
                        if has_variant_in_subdir:
                            continue  # Variant-specific files exist, skip the full-precision one

                filtered.append(s)
                continue

            # Include everything else that wasn't explicitly ignored
            filtered.append(s)

        return filtered

    async def pull_from_civitai(
        self, url_or_id: str, name: str = "", api_key: str | None = None,
    ) -> AsyncIterator[dict]:
        """Download a model from CivitAI with streaming progress."""
        try:
            import httpx
        except ImportError:
            yield {
                "status": "error",
                "error": "httpx is not installed. Run: pip install httpx",
            }
            return

        # Extract model version ID from URL or use directly
        model_version_id = url_or_id
        direct_download_url = ""
        if "civitai.com" in url_or_id:
            from urllib.parse import parse_qs, urlparse

            parsed = urlparse(url_or_id)
            path_parts = [p for p in parsed.path.rstrip("/").split("/") if p]
            qs = parse_qs(parsed.query)

            # If this is already a direct download URL (/api/download/models/...),
            # use it as-is to preserve file-specific query params (fp, size, format)
            if "/api/download/models/" in parsed.path:
                direct_download_url = url_or_id
                for i, part in enumerate(path_parts):
                    if part == "models" and i + 1 < len(path_parts):
                        model_version_id = path_parts[i + 1]
                        break
            elif "modelVersionId" in qs:
                # Prefer explicit modelVersionId query param
                model_version_id = qs["modelVersionId"][0]
            else:
                # Extract from path: /models/<id>
                for i, part in enumerate(path_parts):
                    if part == "models" and i + 1 < len(path_parts):
                        model_version_id = path_parts[i + 1]
                        break

            # Ensure the extracted ID is numeric to avoid injection
            if not model_version_id.isdigit():
                yield {"status": "error", "error": f"Invalid model version ID: {model_version_id}"}
                return

        yield {"status": "downloading", "name": name or model_version_id, "source": "civitai"}

        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        # Use the direct download URL if provided (preserves file-specific
        # variant selection from the detect endpoint), otherwise build generic
        download_url = direct_download_url or f"https://civitai.com/api/download/models/{model_version_id}"
        model_name = name or f"civitai-{model_version_id}"
        dest_dir = os.path.join(self._model_dir, model_name)

        # Clean up partial downloads from prior interrupted attempts
        if os.path.exists(dest_dir) and not self._is_valid_model_dir(dest_dir):
            log.info("cleaning_invalid_model_dir", path=dest_dir)
            await asyncio.to_thread(shutil.rmtree, dest_dir)

        if os.path.exists(dest_dir) and self._is_valid_model_dir(dest_dir):
            yield {"status": "exists", "name": model_name, "path": dest_dir}
            return

        Path(dest_dir).mkdir(parents=True, exist_ok=True)

        try:
            async with httpx.AsyncClient(follow_redirects=True, timeout=600.0) as client, \
                    client.stream("GET", download_url, headers=headers) as resp:
                    resp.raise_for_status()

                    # Determine filename from content-disposition or default
                    cd = resp.headers.get("content-disposition", "")
                    if "filename=" in cd:
                        filename = cd.split("filename=")[1].strip('"').strip("'")
                    else:
                        filename = "model.safetensors"

                    file_path = os.path.join(dest_dir, filename)
                    total = int(resp.headers.get("content-length", 0))
                    downloaded = 0

                    with open(file_path, "wb") as f:
                        async for chunk in resp.aiter_bytes(chunk_size=1024 * 1024):
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total > 0:
                                yield {
                                    "status": "progress",
                                    "downloaded": downloaded,
                                    "total": total,
                                    "percent": round(downloaded / total * 100, 1),
                                }

        except Exception as exc:
            # Clean up partial download directory
            if os.path.exists(dest_dir):
                try:
                    await asyncio.to_thread(shutil.rmtree, dest_dir)
                except OSError:
                    log.debug("partial_download_cleanup_failed", path=dest_dir, exc_info=True)
            yield {"status": "error", "error": str(exc)}
            return

        pipeline_type = _detect_pipeline_type(dest_dir)
        size_bytes = _get_dir_size(dest_dir)

        yield {
            "status": "complete",
            "name": model_name,
            "path": dest_dir,
            "pipeline_type": pipeline_type.value,
            "size_bytes": size_bytes,
        }
        log.info("model_downloaded_civitai", name=model_name, size_bytes=size_bytes)

    async def import_local(self, source_path: str, name: str = "") -> dict:
        """Import a local .safetensors file or model directory."""
        source = Path(source_path)
        if not source.exists():
            raise FileNotFoundError(f"Source not found: {source_path}")

        model_name = name or source.stem
        dest_path = os.path.join(self._model_dir, model_name)

        if source.is_file() and source.suffix == ".safetensors":
            Path(dest_path).mkdir(parents=True, exist_ok=True)
            dest_file = os.path.join(dest_path, source.name)
            await asyncio.to_thread(shutil.copy2, str(source), dest_file)
        elif source.is_dir():
            await asyncio.to_thread(shutil.copytree, str(source), dest_path, dirs_exist_ok=True)
        else:
            raise ValueError(f"Unsupported source: {source_path}")

        pipeline_type = _detect_pipeline_type(dest_path)
        size_bytes = _get_dir_size(dest_path)

        log.info("model_imported", name=model_name, source=source_path)
        return {
            "name": model_name,
            "path": dest_path,
            "pipeline_type": pipeline_type.value,
            "size_bytes": size_bytes,
        }

    async def delete_model(self, name: str) -> bool:
        """Delete a locally stored model."""
        path = self._safe_model_path(name)
        if not path or not os.path.exists(path):
            # Try HF-style name
            safe_name = name.replace("/", "--")
            path = self._safe_model_path(safe_name)
            if not path or not os.path.exists(path):
                return False

        await asyncio.to_thread(shutil.rmtree, path)
        log.info("model_deleted", name=name)
        return True
