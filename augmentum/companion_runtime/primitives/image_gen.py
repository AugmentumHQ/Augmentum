"""Image-gen primitive — wraps the diffusion pipeline registry.

``call(prompt=..., neg_prompt=..., mode="txt2img"|"img2img"|"inpaint",
**kwargs)`` returns a generated image (bytes or PIL.Image). Pipeline
selection (flux / SD3 / SDXL) is owned by the pipeline registry on
``app.state.image_pipeline_registry``.
"""

from __future__ import annotations

from typing import Any

from augmentum.companion_runtime.primitives.base import (
    PrimitiveBase,
    PrimitiveContext,
    PrimitiveResult,
)
from augmentum.companion_runtime.primitives.registry import PrimitiveRegistry
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


class ImageGenPrimitive(PrimitiveBase):
    name = "image_gen"
    description = "Generate an image from a prompt via the loaded diffusion pipeline."

    async def call(self, ctx: PrimitiveContext, **kwargs: Any) -> PrimitiveResult:
        prompt = kwargs.get("prompt", "").strip()
        if not prompt:
            return PrimitiveResult(ok=False, error="image_gen: empty prompt")

        app_state = getattr(ctx.runtime, "_app_state", None)
        registry = getattr(app_state, "image_pipeline_registry", None) if app_state else None
        if registry is None:
            return PrimitiveResult(
                ok=False,
                error="image_gen: no image_pipeline_registry on app.state",
            )

        pipeline = getattr(registry, "current", None) or getattr(registry, "active", None)
        if pipeline is None:
            return PrimitiveResult(
                ok=False,
                error="image_gen: no active pipeline loaded",
            )

        mode = kwargs.get("mode", "txt2img")
        try:
            if mode == "img2img":
                img = await pipeline.img2img(prompt=prompt, **{
                    k: v for k, v in kwargs.items() if k not in ("prompt", "mode")
                })
            elif mode == "inpaint":
                img = await pipeline.inpaint(prompt=prompt, **{
                    k: v for k, v in kwargs.items() if k not in ("prompt", "mode")
                })
            else:
                img = await pipeline.generate(prompt=prompt, **{
                    k: v for k, v in kwargs.items() if k not in ("prompt", "mode")
                })
        except Exception as exc:
            log.exception("image_gen_failed", error=str(exc), mode=mode)
            return PrimitiveResult(ok=False, error=f"image_gen_failed: {exc!s}")

        return PrimitiveResult(ok=True, payload=img, metadata={"mode": mode})


PrimitiveRegistry.register(ImageGenPrimitive)
