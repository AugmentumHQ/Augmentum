"""Background removal tool — rembg ``isnet-general-use`` for image artifacts.

Wraps :func:`augmentum.image.postprocess.remove_background` so the chat LLM
and voice surface can invoke background removal end-to-end. The existing
``POST /api/artifacts/{id}/remove-bg`` route keeps its in-place semantics
(replaces the source artifact); this tool produces a NEW sibling artifact
because LLM-driven actions should be non-destructive by default.
"""

from __future__ import annotations

from pathlib import Path
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


class BackgroundRemoveTool(Tool):
    """Remove the background from an image artifact via rembg."""

    def __init__(self, artifact_store: ArtifactStore) -> None:
        self._store = artifact_store

    @property
    def name(self) -> str:
        return "remove_background"

    @property
    def description(self) -> str:
        return (
            "Remove the background from an image artifact, producing a PNG with "
            "transparency. Saves the result as a new sibling artifact; the original "
            "is preserved."
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
            },
            "required": ["artifact_id"],
        }

    @property
    def surfaces(self) -> SurfaceExposure:
        return SurfaceExposure(
            chat=True,
            voice="costly",
            coder=True,
            artifact_studio=True,
            file_context_menu=("image/*",),
            voice_capability_line="remove the background from an image (remove_background)",
        )

    @property
    def requires_services(self) -> list[str]:
        return ["image_pipeline"]

    def health_check(self) -> bool:
        try:
            import rembg  # noqa: F401
        except ImportError:
            return False
        return True

    async def execute(self, **kwargs) -> ToolResult:
        artifact_id = (kwargs.get("artifact_id") or "").strip()
        user_id = self.extract_user_id(kwargs)

        if not artifact_id:
            return ToolResult(success=False, error="artifact_id is required", validation_error=True)

        info = await self._store.get(artifact_id, user_id=user_id)
        if not info:
            return ToolResult(success=False, error="Artifact not found")
        src_path = self._store.get_file_path(info.get("path", ""))
        if not src_path or not src_path.is_file():
            return ToolResult(success=False, error="Artifact file missing")

        try:
            from augmentum.image.postprocess import remove_background
        except ImportError as exc:
            return ToolResult(success=False, error=f"rembg unavailable: {exc}")

        try:
            _id, new_path, w, h = await remove_background(str(src_path))
        except Exception as exc:  # noqa: BLE001
            log.warning("remove_background_failed", artifact_id=artifact_id, error=str(exc))
            return ToolResult(success=False, error=f"Background removal failed: {exc}")

        try:
            data = Path(new_path).read_bytes()
        finally:
            try:
                Path(new_path).unlink(missing_ok=True)
            except Exception:  # noqa: BLE001
                log.debug("remove_background_cleanup_failed", path=str(new_path))

        base = info.get("display_name") or info.get("filename") or "image"
        stem = base.rsplit(".", 1)[0] if "." in base else base
        try:
            saved = await self._store.save(
                data=data,
                filename=f"{stem}-nobg.png",
                fmt="png",
                session_id=info.get("session_id") or "",
                task_id=info.get("task_id") or "",
                display_name=f"{stem} (no background)",
                metadata={"derived_from": artifact_id, "op": "remove_background"},
                user_id=user_id,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("remove_background_save_failed", artifact_id=artifact_id, error=str(exc))
            return ToolResult(success=False, error=f"Save failed: {exc}")

        card = make_artifact_card(
            saved,
            kind="image",
            title=f"{stem}-nobg.png",
            subtitle="Background removed",
            summary=f"Removed background, {w}×{h}, {len(data)} bytes",
        )
        return ToolResult(
            success=True,
            output=f"Background removed ({w}×{h}, {len(data)} bytes). New artifact: {saved['id']}",
            metadata={
                "artifact_id": saved["id"],
                "filename": saved["filename"],
                "width": w,
                "height": h,
                "size_bytes": len(data),
            },
            card=card,
        )
