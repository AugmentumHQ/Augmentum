"""Image generation tool — allows LLMs to generate images via the tool framework."""

from __future__ import annotations

from typing import TYPE_CHECKING

from augmentum.tools.base import Tool, ToolCategory, ToolResult
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.image.queue import GenerationQueue

log = get_logger(__name__)

# Base description — model-style hints are appended dynamically.
_BASE_DESCRIPTION = (
    "Generate an image from a text prompt. Supports style presets "
    "(fantasy_rpg, anime, scifi, horror, realism) and aspect ratios "
    "(portrait, landscape, square)."
)


class ImageGenerationTool(Tool):
    """Generate images from text prompts using Stable Diffusion or FLUX."""

    @property
    def name(self) -> str:
        return "image_generation"

    @property
    def description(self) -> str:
        # Dynamically append model-style guidance so the LLM writes the
        # right kind of prompt (natural language for FLUX, tags for SD1.5).
        model = self._active_image_model()
        if not model:
            return _BASE_DESCRIPTION

        from augmentum.image.prompt_condenser import detect_image_model_style

        info = detect_image_model_style(model)
        hint = info.get("style_hint", "")
        if not hint:
            return _BASE_DESCRIPTION

        parts = [_BASE_DESCRIPTION, f"\nTarget model: {model}. {hint}"]
        if not info.get("supports_negative"):
            parts.append("Do NOT provide a negative_prompt for this model.")
        return " ".join(parts)

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.IMAGE

    @property
    def error_hints(self) -> dict[str, str]:
        return {
            "Queue error": "The image generation queue is full or not started. Try again in a moment.",
            "timed out": "Image generation took too long. Try a simpler prompt or smaller dimensions.",
            "Generation failed": "The image model failed. Try rephrasing the prompt or using a different style.",
        }

    @property
    def requires_services(self) -> list[str]:
        return ["image_pipeline"]

    @property
    def produces(self) -> list[str]:
        return ["image_url"]

    def health_check(self) -> bool:
        """Check if the image generation queue is running."""
        return self._queue is not None and getattr(self._queue, '_running', False)

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Detailed text description of the image to generate",
                },
                "negative_prompt": {
                    "type": "string",
                    "description": "Things to exclude from the image",
                    "default": "",
                },
                "style": {
                    "type": "string",
                    "description": "Genre preset: fantasy_rpg, anime, scifi, horror, realism",
                    "default": "",
                },
                "aspect": {
                    "type": "string",
                    "description": "Aspect ratio: portrait, landscape, or square",
                    "default": "square",
                },
            },
            "required": ["prompt"],
        }

    @property
    def timeout(self) -> float:
        return 600.0

    @property
    def cacheable(self) -> bool:
        return False

    def __init__(self, queue: GenerationQueue, preset_manager=None, app_state=None) -> None:
        self._queue = queue
        self._preset_manager = preset_manager
        self._app_state = app_state

    def _active_image_model(self) -> str:
        """Return the currently selected image model name."""
        if not self._app_state:
            return ""
        ui = getattr(self._app_state, "image_active_settings", None) or {}
        model = ui.get("model", "")
        if model:
            return model
        from augmentum.config import settings
        return settings.image_default_model

    def _default_agentic_model(self) -> str:
        """The model an agentic generation would use absent an override."""
        from augmentum.config import settings
        if settings.agentic_image_model:
            return settings.agentic_image_model
        ui = getattr(self._app_state, "image_active_settings", None) or {}
        return ui.get("model", "") or settings.image_default_model

    async def select_model_for(
        self, *, need_photoreal: bool = False, need_diagram: bool = False,
    ) -> str:
        """Pick an installed model matching the requested capability.

        Returns the best model name, or ``""`` when no installed model
        satisfies the need (the caller should then skip generation rather
        than render a believable-photo request with a stylised checkpoint —
        the "anime tire diagram" failure mode). When no capability is
        requested, returns the default agentic model unchanged.
        """
        from augmentum.image.prompt_condenser import derive_image_capabilities

        default = self._default_agentic_model()
        if not (need_photoreal or need_diagram):
            return default

        def _satisfies(name: str) -> bool:
            caps = derive_image_capabilities(name)
            if need_photoreal and not caps["photoreal"]:
                return False
            return not (need_diagram and not caps["diagram"])

        # Prefer the user's default when it already fits the job.
        if default and _satisfies(default):
            return default

        persistence = getattr(self._app_state, "image_persistence", None)
        if not persistence:
            return ""
        try:
            models = await persistence.list_models()
        except Exception as exc:  # pragma: no cover - persistence wobble
            log.warning("select_model_list_failed", error=str(exc))
            return ""
        for info in models or []:
            name = getattr(info, "name", "") or ""
            if name and _satisfies(name):
                return name
        return ""

    def validate_input(self, **kwargs) -> bool:
        prompt = kwargs.get("prompt", "")
        return isinstance(prompt, str) and len(prompt.strip()) > 0

    async def execute(
        self,
        *,
        prompt: str,
        negative_prompt: str = "",
        style: str = "",
        aspect: str = "square",
        **kwargs,
    ) -> ToolResult:
        """Generate an image and return the URL."""
        from augmentum.config import settings

        # Read the user's current image panel settings (pushed by the UI).
        # These override static config defaults so the tool uses the same
        # model/steps/cfg/sampler the user configured in the image panel.
        # MUST resolve PER-USER: the panel persists per-user, so reading the
        # process-global mirror ignored every authenticated user's selection and
        # fell back to the install default (the "last-installed, not selected"
        # bug). See augmentum/image/active_settings.py.
        from augmentum.image.active_settings import resolve_active_settings
        from augmentum.image.queue import GenerationJob
        _tool_user_id = Tool.extract_user_id(kwargs)
        ui = await resolve_active_settings(self._app_state, _tool_user_id)

        # Apply preset — UI preset wins, then tool param, then config default
        preset_name = style or ui.get("preset", "") or settings.image_default_preset
        final_prompt = prompt
        final_negative = negative_prompt or ui.get("negative_prompt", "") or settings.image_default_negative_prompt
        if preset_name and self._preset_manager:
            preset = self._preset_manager.get(preset_name)
            if preset:
                final_prompt, final_negative = preset.apply(prompt, final_negative)

        # Dimensions: aspect ratio overrides UI panel, UI panel overrides config
        base_w = ui.get("width") or settings.image_default_width
        base_h = ui.get("height") or settings.image_default_height
        if aspect == "portrait":
            width, height = base_w, int(base_h * 1.5)
        elif aspect == "landscape":
            width, height = int(base_w * 1.5), base_h
        else:
            width, height = base_w, base_h

        # Model priority: agentic override > UI panel > config default
        # When called from agentic flows (task_id present in kwargs) and
        # the user has configured agentic_image_model, that wins — even
        # over the image panel's last-used model.  This lets users set a
        # lightweight model for storybook/tutorial illustrations while
        # keeping a high-quality model for interactive generation.
        # An explicit model kwarg wins over everything — agentic flows use it to
        # request a capability-matched model (e.g. a photoreal checkpoint for a
        # how-to photo) instead of inheriting the user's stylised default.
        explicit_model = (kwargs.get("model") or "").strip()
        is_agentic = bool(kwargs.get("task_id"))
        if explicit_model:
            model = explicit_model
        elif is_agentic and settings.agentic_image_model:
            model = settings.agentic_image_model
        elif ui.get("model"):
            model = ui["model"]
        else:
            model = settings.image_default_model

        # Apply distilled-model-aware defaults (FLUX, etc.)
        from augmentum.image.distilled import apply_distilled_defaults

        raw_steps = ui.get("steps") or settings.image_default_steps
        raw_cfg = ui.get("cfg_scale") or settings.image_default_cfg
        steps, cfg_scale = apply_distilled_defaults(model, raw_steps, raw_cfg)
        sampler = ui.get("sampler", "")

        # Resolve negative prompt defaults (FLUX gets empty, SD gets quality tags)
        from augmentum.image.defaults import resolve_negative_prompt
        from augmentum.image.prompt_condenser import detect_image_model_style

        model_info = detect_image_model_style(model)
        hw = getattr(self._app_state, "image_hardware", None) if self._app_state else None
        pipeline_type = hw.recommended_pipeline if hw else model_info.get("family", "sd15")
        final_negative = resolve_negative_prompt(
            final_negative, pipeline_type, settings.image_default_negative_prompt,
        )

        # Resolve condense model so auto-condensation works in the queue worker
        condense_model = settings.image_prompt_condense_model

        # _tool_user_id resolved above (drives both the per-user panel settings
        # read and the generation row's ownership).
        # Prefer an explicit session_id kwarg (handlers pass it directly for
        # artifact tools) and fall back to whatever landed in ``_context``.
        # Without this, ``GenerationJob.session_id`` defaulted to empty and
        # the persistence write at proxy/server.py:914 blew up on the
        # FOREIGN KEY constraint for every image produced inside
        # ``create_ebook`` / other artifact-driven flows.
        _ctx = kwargs.get("_context") or {}
        _tool_session_id = kwargs.get("session_id") or (
            _ctx.get("session_id", "") if isinstance(_ctx, dict) else ""
        )
        _ip_ref = kwargs.get("ip_adapter_image", "")
        _ip_scale = kwargs.get("ip_adapter_scale", 0.55)

        # Provenance: the companion's loop stamps _context.origin =
        # 'companion'; the row lands in the same gallery as user
        # generations, filterable by the chip.
        _origin = (
            _ctx.get("origin", "") if isinstance(_ctx, dict) else ""
        )

        job = GenerationJob(
            prompt=final_prompt,
            negative_prompt=final_negative,
            model=model,
            preset=preset_name,
            width=width,
            height=height,
            steps=steps,
            cfg_scale=cfg_scale,
            sampler=sampler,
            condense_model=condense_model,
            ip_adapter_image=_ip_ref,
            ip_adapter_scale=_ip_scale,
            user_id=_tool_user_id,
            session_id=_tool_session_id,
            origin=_origin,
        )

        try:
            job = await self._queue.submit(job)
            result = await self._queue.wait_for_result(job, timeout=600.0)
        except RuntimeError as exc:
            return ToolResult(success=False, error=f"Queue error: {exc}")
        except TimeoutError:
            return ToolResult(success=False, error="Image generation timed out")
        except Exception as exc:
            return ToolResult(success=False, error=f"Generation failed: {exc}")

        image_id = result["image_id"]
        url = f"/api/image/{image_id}"

        warnings: list[str] = []
        if not result.get("vfs_registered", True):
            # The image is on disk and in the DB but never made it into the
            # file index — it will be invisible in the file browser until
            # the next startup repair sweep. Surface this so the user
            # understands why the file panel is missing the image.
            warnings.append(
                "Image saved but failed to register in the file browser index; "
                "it will still be accessible via the gallery URL."
            )

        from augmentum.tools.base import format_output_with_warnings
        base_output = (
            f"Image generated successfully (url: {url}) and is now visible "
            "in the gallery. "
            "Do NOT call image_generation again unless the user explicitly "
            "asks for another image. Describe the image you created to the "
            "user. If the user wants it kept somewhere (e.g. added to the "
            "open note), pass that url to the appropriate tool."
        )
        return ToolResult(
            success=True,
            output=format_output_with_warnings(base_output, warnings),
            metadata={
                "image_id": image_id,
                "url": url,
                "seed": result.get("seed", -1),
                "width": width,
                "height": height,
                "prompt": final_prompt,
            },
            warnings=warnings,
        )
