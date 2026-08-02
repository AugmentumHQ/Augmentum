"""Singleton pipeline manager — loads, swaps, and unloads image pipelines.

Only one pipeline is active at a time to conserve VRAM.
"""

from __future__ import annotations

import asyncio

from augmentum.image.pipeline import ImagePipeline
from augmentum.image.pipeline_v2 import UnifiedPipeline
from augmentum.image.schemas import PipelineType
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# All pipeline types use UnifiedPipeline — it auto-detects the model
# architecture at load time via DiffusionPipeline.from_pretrained().
_PIPELINE_CLASSES: dict[PipelineType, type[ImagePipeline]] = {
    PipelineType.SD15: UnifiedPipeline,
    PipelineType.SDXL: UnifiedPipeline,
    PipelineType.FLUX: UnifiedPipeline,
}


class PipelineRegistry:
    """Manages the currently active image pipeline (only one at a time)."""

    def __init__(self) -> None:
        self._current: ImagePipeline | None = None
        self._current_model: str = ""
        self._current_type: PipelineType | None = None
        self._lock = asyncio.Lock()

    @property
    def current(self) -> ImagePipeline | None:
        return self._current

    @property
    def current_model(self) -> str:
        return self._current_model

    @property
    def is_loaded(self) -> bool:
        return self._current is not None and self._current.is_loaded

    async def load(
        self,
        model_path: str,
        pipeline_type: PipelineType,
        device: str = "cuda",
        dtype: str = "fp16",
    ) -> ImagePipeline:
        """Load a model. Swaps out any currently loaded pipeline first."""
        async with self._lock:
            # Skip if same model is already loaded
            if (
                self._current is not None
                and self._current.is_loaded
                and self._current_model == model_path
                and self._current_type == pipeline_type
            ):
                log.info("pipeline_already_loaded", model=model_path)
                return self._current

            # Unload current pipeline (release_pipeline handles all VRAM cleanup)
            if self._current is not None:
                await self._unload_current()

            # Create and load new pipeline
            pipeline_cls = _PIPELINE_CLASSES.get(pipeline_type)
            if not pipeline_cls:
                raise ValueError(f"Unknown pipeline type: {pipeline_type}")

            pipeline = pipeline_cls()
            try:
                await pipeline.load(model_path, device=device, dtype=dtype)
            except Exception:
                # Ensure clean state on failed load
                self._current = None
                self._current_model = ""
                self._current_type = None
                from augmentum.image.vram import flush_cuda_cache
                flush_cuda_cache()
                raise

            self._current = pipeline
            self._current_model = model_path
            # Use the pipeline's auto-detected type (may differ from
            # the pre-detection hint passed by the caller).
            self._current_type = pipeline.pipeline_type

            return pipeline

    async def unload(self) -> None:
        """Unload the current pipeline and free all memory."""
        async with self._lock:
            if self._current is not None:
                await self._unload_current()

    async def reload_current(self) -> ImagePipeline | None:
        """Unload and re-load the current model so load-time settings take effect."""
        async with self._lock:
            if self._current is None or not self._current_model:
                return None
            model = self._current_model
            ptype = self._current_type or PipelineType.SD15
            device = getattr(self._current, "_device", "cuda")
            try:
                from augmentum.config import settings
                dtype = settings.image_precision
            except Exception:
                dtype = "fp16"
            log.info("pipeline_reloading", model=model)
            await self._unload_current()

            # Re-load under the same lock to prevent concurrent access
            pipeline_cls = _PIPELINE_CLASSES.get(ptype)
            if not pipeline_cls:
                raise ValueError(f"Unknown pipeline type: {ptype}")
            pipeline = pipeline_cls()
            try:
                await pipeline.load(model, device=device, dtype=dtype)
            except Exception:
                self._current = None
                self._current_model = ""
                self._current_type = None
                from augmentum.image.vram import flush_cuda_cache
                flush_cuda_cache()
                raise
            self._current = pipeline
            self._current_model = model
            self._current_type = pipeline.pipeline_type
            return pipeline

    async def _unload_current(self) -> None:
        """Internal: unload without acquiring lock."""
        if self._current is None:
            return

        model = self._current_model
        log.info("pipeline_swapping_out", model=model)

        await self._current.unload()
        self._current = None
        self._current_model = ""
        self._current_type = None

        # Final sweep — the pipeline's own unload() already called
        # release_pipeline(), but run one more flush in case the
        # pipeline object itself held residual CUDA allocations.
        from augmentum.image.vram import flush_cuda_cache
        flush_cuda_cache()

        log.info("pipeline_swap_complete", previous_model=model)

    async def generate_for_fabric(
        self,
        gen_request,
        *,
        persistence=None,
    ) -> tuple[bytes, dict]:
        """Slim image-generation helper for cross-peer dispatch.

        Used by ``/api/fabric/image/generate`` to run a generation
        without going through the regular user-facing flow's queue,
        preset application, library write, or cache check. The
        initiating peer has already done all of those on its side
        before dispatching; this receiver only contributes pixel-level
        compute.

        Steps:
          1. Resolve the requested model name to (path, pipeline_type)
             via the persistence layer (image_models table).
          2. Ensure ``self.load(path, pipeline_type)`` — same loader
             the regular path uses, so model swap + VRAM accounting
             stay consistent.
          3. Call ``self._current.generate(prompt, ...)`` with the
             exact params from the request.
          4. Read the generated file bytes off disk.
          5. Return ``(bytes, metadata_dict)``.

        The result file lives in /data/image_output (or whatever the
        pipeline writes to) — we don't delete it. Operator can manage
        cleanup via the existing image library tooling. We return
        bytes so the sender can write them to its OWN library with a
        local image_id; the peer's local image_id stays local.

        ``persistence`` is the ImagePersistence that owns image_models
        — usually ``app.state.image_persistence``. Passed in rather
        than looked up so this method stays independent of app.state
        (cleaner for testing and avoids a cyclic import).

        Raises:
          ValueError: model not in image_models table on this peer.
          Other exceptions from pipeline.load / generate propagate so
            the fabric route layer can wrap them as a 500 + log.
        """
        from augmentum.config import settings as _settings

        model_name = (gen_request.model or "").strip()
        if not model_name:
            raise ValueError("model required")

        # 1. Resolve via persistence layer.
        if persistence is None:
            raise ValueError(
                "persistence (ImageStore) is required to resolve model name → path"
            )
        model_info = await persistence.get_model(model_name)
        if model_info is None:
            raise ValueError(
                f"model {model_name!r} not in image_models table on this peer"
            )

        # 2. Ensure loaded. ``load`` no-ops when the same model is
        # already active, so the warm-path is cheap.
        device = _settings.image_device
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = _settings.image_precision or "fp16"
        await self.load(
            model_path=model_info.path,
            pipeline_type=model_info.pipeline_type,
            device=device,
            dtype=dtype,
        )
        if self._current is None or not self._current.is_loaded:
            raise RuntimeError(
                f"pipeline failed to load model {model_name!r}"
            )

        # 3. Run generation. Use the request's params directly — the
        # sender already applied preset + computed final values.
        width = getattr(gen_request, "width", 512) or 512
        height = getattr(gen_request, "height", 512) or 512
        steps = getattr(gen_request, "steps", 20) or 20
        cfg_scale = getattr(gen_request, "cfg_scale", 7.0) or 7.0
        seed = getattr(gen_request, "seed", -1)
        sampler = getattr(gen_request, "sampler", "") or getattr(gen_request, "scheduler", "") or ""

        result = await self._current.generate(
            prompt=gen_request.prompt,
            negative_prompt=getattr(gen_request, "negative_prompt", "") or "",
            width=int(width),
            height=int(height),
            steps=int(steps),
            cfg_scale=float(cfg_scale),
            seed=int(seed) if seed is not None else -1,
            sampler=sampler,
        )

        # 4. Read bytes.
        import asyncio as _asyncio
        from pathlib import Path
        file_path = Path(result.file_path)
        if not file_path.is_file():
            raise RuntimeError(
                f"pipeline returned non-existent file_path: {result.file_path!r}"
            )
        image_bytes = await _asyncio.to_thread(file_path.read_bytes)

        # 5. Build metadata. Mirror the regular GenerateResponse shape
        # so the sender's parser can pick out the fields it wants.
        metadata = {
            "seed": result.seed,
            "width": result.width,
            "height": result.height,
            "steps": int(steps),
            "cfg_scale": float(cfg_scale),
            "model": model_name,
            "sampler": sampler,
            "peer_image_id": result.image_id,
        }
        return image_bytes, metadata
