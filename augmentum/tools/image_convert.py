"""Image format conversion tool — Pillow-based PNG/JPG/WEBP interop.

Canonical home of the Pillow conversion logic. ``artifact_routes._convert_image``
delegates to :meth:`ImageConvertTool.convert_bytes` so a single implementation
serves both the HTTP route (Artifact Studio button) and LLM/voice function
calls. See ``docs/superpowers/specs/2026-06-01-unified-primitive-layer-design.md``.
"""

from __future__ import annotations

from io import BytesIO
from typing import TYPE_CHECKING

from augmentum.tools.base import (
    SurfaceExposure,
    Tool,
    ToolCategory,
    ToolResult,
    make_artifact_card,
)
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.tools.artifact_storage import ArtifactStore

log = get_logger(__name__)

_SUPPORTED_TARGETS = {"png", "jpg", "jpeg", "webp"}


class ImageConvertTool(Tool):
    """Convert an image artifact between PNG / JPG / WEBP."""

    def __init__(self, artifact_store: ArtifactStore) -> None:
        self._store = artifact_store

    @property
    def name(self) -> str:
        return "convert_image"

    @property
    def description(self) -> str:
        return (
            "Convert an image artifact to a different format (PNG, JPG, or WEBP). "
            "Returns a new sibling artifact; the original is preserved."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.IMAGE

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "artifact_id": {
                    "type": "string",
                    "description": "Source image artifact id",
                },
                "target_format": {
                    "type": "string",
                    "enum": ["png", "jpg", "jpeg", "webp"],
                    "description": "Target image format",
                },
            },
            "required": ["artifact_id", "target_format"],
        }

    @property
    def surfaces(self) -> SurfaceExposure:
        return SurfaceExposure(
            chat=True,
            voice="core",
            coder=True,
            artifact_studio=True,
            file_context_menu=("image/*",),
            http_route="/api/tools/convert_image",
            voice_capability_line="convert an image between PNG, JPG, and WEBP (convert_image)",
        )

    # ------------------------------------------------------------------
    # Pure conversion — no artifact lookup. Used by the HTTP route's
    # ``_convert_image`` helper so the existing UI keeps working.
    # ------------------------------------------------------------------

    @staticmethod
    async def convert_bytes(src_path: str, target: str) -> tuple[bytes, str]:
        """Pillow-based conversion. Returns ``(bytes, new_extension)``.

        ``target`` accepts ``png``, ``jpg``, ``jpeg``, or ``webp``.
        JPEG output flattens transparency onto white to avoid the
        black-on-transparency artifact Pillow produces by default.
        """
        from PIL import Image

        target = target.lower()
        if target == "jpeg":
            target = "jpg"
        if target not in {"png", "jpg", "webp"}:
            raise ValueError(f"Unsupported image target format: {target}")

        pil_fmt = {"png": "PNG", "jpg": "JPEG", "webp": "WEBP"}[target]

        img = Image.open(str(src_path))
        if target == "jpg" and img.mode in ("RGBA", "LA", "P"):
            bg = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            bg.paste(img, mask=img.split()[-1] if img.mode == "RGBA" else None)
            img = bg
        elif img.mode == "P":
            img = img.convert("RGBA")

        buf = BytesIO()
        save_kwargs: dict = {"format": pil_fmt}
        if pil_fmt == "JPEG":
            save_kwargs["quality"] = 92
            save_kwargs["optimize"] = True
        elif pil_fmt == "WEBP":
            save_kwargs["quality"] = 90
        img.save(buf, **save_kwargs)
        return buf.getvalue(), target

    # ------------------------------------------------------------------
    # LLM-facing entry point — full artifact-id flow.
    # ------------------------------------------------------------------

    async def execute(self, **kwargs) -> ToolResult:
        artifact_id = (kwargs.get("artifact_id") or "").strip()
        target = (kwargs.get("target_format") or "").strip().lower()
        user_id = self.extract_user_id(kwargs)

        if not artifact_id:
            return ToolResult(success=False, error="artifact_id is required", validation_error=True)
        if target not in _SUPPORTED_TARGETS:
            return ToolResult(
                success=False,
                error=f"target_format must be one of {sorted(_SUPPORTED_TARGETS)}",
                validation_error=True,
            )

        info = await self._store.get(artifact_id, user_id=user_id)
        if not info:
            return ToolResult(success=False, error="Artifact not found")
        src_path = self._store.get_file_path(info.get("path", ""))
        if not src_path or not src_path.is_file():
            return ToolResult(success=False, error="Artifact file missing")

        try:
            data, new_ext = await self.convert_bytes(str(src_path), target)
        except Exception as exc:  # noqa: BLE001 — surface to LLM
            log.warning("convert_image_failed", artifact_id=artifact_id, error=str(exc))
            return ToolResult(success=False, error=f"Conversion failed: {exc}")

        base = info.get("display_name") or info.get("filename") or "image"
        stem = base.rsplit(".", 1)[0] if "." in base else base
        try:
            saved = await self._store.save(
                data=data,
                filename=f"{stem}.{new_ext}",
                fmt=new_ext,
                session_id=info.get("session_id") or "",
                task_id=info.get("task_id") or "",
                display_name=f"{stem} ({new_ext.upper()})",
                metadata={"converted_from": artifact_id, "target": new_ext},
                user_id=user_id,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("convert_image_save_failed", artifact_id=artifact_id, error=str(exc))
            return ToolResult(success=False, error=f"Save failed: {exc}")

        card = make_artifact_card(
            saved,
            kind="image",
            title=f"{stem}.{new_ext}",
            subtitle=f"Converted to {new_ext.upper()}",
            summary=f"Converted from {info.get('format', '?').upper()} to {new_ext.upper()}",
        )
        return ToolResult(
            success=True,
            output=f"Converted to {new_ext.upper()} ({len(data)} bytes). New artifact: {saved['id']}",
            metadata={
                "artifact_id": saved["id"],
                "filename": saved["filename"],
                "format": new_ext,
                "size_bytes": len(data),
            },
            card=card,
        )
