"""ATP vision + OCR tools — let text-only harness models "see" images.

Closes the loop with the ATP browser tools: ``browser_screenshot``
returns an artifact URL; ``vision_describe`` / ``ocr_extract`` take
that same artifact (or any fetchable image URL) and return text.

Input resolution is deliberately restricted for the multi-tenant
boundary: an artifact id / ``/api/artifacts/<id>/download`` URL
(ownership-checked against the authenticated user) or an http(s) URL.
Raw server filesystem paths are NOT accepted.

Both tools are ATP-only (``SurfaceExposure`` off everywhere else) and
health-gated: ``vision_describe`` on the vision router having a live
provider, ``ocr_extract`` on the docling sidecar answering.
"""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any

from augmentum.tools.base import SurfaceExposure, Tool, ToolCategory, ToolResult
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_ARTIFACT_URL_RE = re.compile(r"/api/artifacts/([A-Za-z0-9_-]+)/download")
_ARTIFACT_ID_RE = re.compile(r"^[A-Za-z0-9_-]{6,64}$")
_MAX_IMAGE_BYTES = 20_000_000


class _AtpImageToolBase(Tool):
    """Shared image-input resolution for the ATP vision/OCR tools."""

    def __init__(self, app_state) -> None:
        self._app_state = app_state

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.IMAGE

    @property
    def surfaces(self) -> SurfaceExposure:
        return SurfaceExposure(chat=False, coder=False, flow=False)

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def timeout(self) -> float:
        return 120.0

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "image": {
                    "type": "string",
                    "description": (
                        "Artifact id, /api/artifacts/<id>/download URL "
                        "(e.g. from browser_screenshot), or an http(s) "
                        "image URL"
                    ),
                },
            },
            "required": ["image"],
        }

    async def _load_image(self, kwargs: dict) -> tuple[bytes, str] | ToolResult:
        """Resolve the ``image`` argument to bytes.

        Returns ``(bytes, source_label)`` or a failed ToolResult.
        """
        ref = str(kwargs.get("image") or "").strip()
        if not ref:
            return ToolResult(success=False, validation_error=True,
                              error="'image' is required")
        m = _ARTIFACT_URL_RE.search(ref)
        artifact_id = m.group(1) if m else (
            ref if _ARTIFACT_ID_RE.match(ref) and "://" not in ref else ""
        )
        if artifact_id:
            store = getattr(self._app_state, "artifact_store", None)
            if store is None:
                return ToolResult(success=False, error="artifact store is unavailable")
            user_id = self.extract_user_id(kwargs)
            info = await store.get(artifact_id, user_id=user_id)
            if info is None:
                return ToolResult(
                    success=False,
                    error=f"artifact {artifact_id!r} not found (or not yours)",
                )
            path = store.get_file_path(info["path"])
            if path is None:
                return ToolResult(success=False,
                                  error=f"artifact {artifact_id!r} file is missing on disk")
            data = path.read_bytes()
            if len(data) > _MAX_IMAGE_BYTES:
                return ToolResult(success=False,
                                  error=f"image too large ({len(data)} bytes; cap {_MAX_IMAGE_BYTES})")
            return data, f"artifact {artifact_id}"
        if ref.startswith(("http://", "https://")):
            client = getattr(self._app_state, "http_client", None)
            if client is None:
                return ToolResult(success=False, error="http client is unavailable")
            try:
                resp = await client.get(ref, follow_redirects=True, timeout=30.0)
                resp.raise_for_status()
            except Exception as exc:
                return ToolResult(success=False, error=f"image fetch failed: {exc}")
            data = resp.content
            if len(data) > _MAX_IMAGE_BYTES:
                return ToolResult(success=False,
                                  error=f"image too large ({len(data)} bytes; cap {_MAX_IMAGE_BYTES})")
            return data, ref
        return ToolResult(
            success=False, validation_error=True,
            error=(
                "unrecognized image reference — pass an artifact id, an "
                "/api/artifacts/<id>/download URL, or an http(s) URL "
                "(raw filesystem paths are not accepted)"
            ),
        )


class VisionDescribeTool(_AtpImageToolBase):
    @property
    def name(self) -> str:
        return "vision_describe"

    @property
    def description(self) -> str:
        return (
            "Describe an image with the local vision model. Accepts an "
            "artifact id/URL (e.g. from browser_screenshot) or an http(s) "
            "image URL. Optional 'prompt' steers the description (e.g. "
            "'read the error message in this screenshot')."
        )

    @property
    def input_schema(self) -> dict:
        schema = super().input_schema
        schema["properties"]["prompt"] = {
            "type": "string",
            "description": "What to look at / answer about the image",
        }
        return schema

    def _router(self):
        return getattr(self._app_state, "vision_router", None)

    def health_check(self) -> bool:
        router = self._router()
        return router is not None and router.has_any_provider

    async def health_check_async(self) -> bool:
        router = self._router()
        if router is None:
            return False
        return await router.is_available()

    async def execute(self, **kwargs) -> ToolResult:
        router = self._router()
        if router is None or not await router.is_available():
            return ToolResult(success=False,
                              error="no vision provider is available right now")
        loaded = await self._load_image(kwargs)
        if isinstance(loaded, ToolResult):
            return loaded
        data, source = loaded
        prompt = str(kwargs.get("prompt") or "").strip() or (
            "Describe this image in detail, including any visible text."
        )
        from augmentum.vision.router import Workload

        text = await router.caption(
            data, prompt=prompt, max_tokens=1024, timeout_s=90.0,
            workload=Workload.QUALITY,
        )
        if not text:
            return ToolResult(success=False,
                              error="vision provider returned no description")
        return ToolResult(success=True, output=text,
                          metadata={"source": source, "bytes": len(data)})


class OcrExtractTool(_AtpImageToolBase):
    @property
    def name(self) -> str:
        return "ocr_extract"

    @property
    def description(self) -> str:
        return (
            "Extract text from an image via OCR (docling), returned in "
            "reading order with normalized bounding boxes. Accepts an "
            "artifact id/URL (e.g. from browser_screenshot) or an "
            "http(s) image URL."
        )

    def _enabled(self) -> bool:
        from augmentum.config import settings
        from augmentum.ocr import ocr_enabled

        return ocr_enabled(settings)

    def health_check(self) -> bool:
        return self._enabled()

    async def health_check_async(self) -> bool:
        if not self._enabled():
            return False
        from augmentum.config import settings
        from augmentum.ocr import get_docling_client

        try:
            return await get_docling_client(settings).health()
        except Exception:
            return False

    async def execute(self, **kwargs) -> ToolResult:
        if not self._enabled():
            return ToolResult(success=False,
                              error="OCR is disabled (ocr_enabled setting is off)")
        loaded = await self._load_image(kwargs)
        if isinstance(loaded, ToolResult):
            return loaded
        data, source = loaded
        from augmentum.ocr import extract_page_script

        try:
            lines: list[dict[str, Any]] = await extract_page_script(
                SimpleNamespace(state=self._app_state), data, assemble=False,
            )
        except Exception as exc:
            return ToolResult(success=False, error=f"OCR failed: {exc}")
        if not lines:
            return ToolResult(success=True, output="(no text detected)",
                              metadata={"source": source, "lines": []})
        text = "\n".join(str(ln.get("text") or "") for ln in lines)
        return ToolResult(
            success=True,
            output=text,
            metadata={"source": source, "lines": lines},
        )


ATP_VISION_TOOL_CLASSES = (VisionDescribeTool, OcrExtractTool)
