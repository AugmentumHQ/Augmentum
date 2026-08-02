"""Document conversion tool — re-renders a structured-document source
artifact to PDF or DOCX.

Canonical home of the document re-render logic. ``artifact_routes._convert_document``
delegates to :meth:`DocumentConvertTool.render_bytes` so a single
implementation serves both the HTTP route (Artifact Studio button) and
LLM/voice function calls. See
``docs/superpowers/specs/2026-06-01-unified-primitive-layer-design.md``.

This tool only works on artifacts that carry a ``source_json`` (structured
content produced by :class:`augmentum.tools.artifact_document.DocumentTool`).
Arbitrary PDFs without a source cannot be edited via this path — that's a
Phase 3 concern (pypdf-based page ops).
"""

from __future__ import annotations

import asyncio
import json
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

_SUPPORTED_TARGETS = {"pdf", "docx"}


class DocumentConvertTool(Tool):
    """Re-render a structured document artifact to PDF or DOCX."""

    def __init__(self, artifact_store: ArtifactStore) -> None:
        self._store = artifact_store

    @property
    def name(self) -> str:
        return "convert_document"

    @property
    def description(self) -> str:
        return (
            "Convert a structured document artifact to PDF or DOCX by re-rendering "
            "from its source. Returns a new sibling artifact; the original is "
            "preserved. Only works on documents created via the document tool — "
            "arbitrary PDFs without a structured source cannot be converted here."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.ARTIFACT

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "artifact_id": {
                    "type": "string",
                    "description": "Source document artifact id",
                },
                "target_format": {
                    "type": "string",
                    "enum": ["pdf", "docx"],
                    "description": "Target document format",
                },
            },
            "required": ["artifact_id", "target_format"],
        }

    @property
    def surfaces(self) -> SurfaceExposure:
        return SurfaceExposure(
            chat=True,
            voice="interactive",
            coder=True,
            artifact_studio=True,
            file_context_menu=(".pdf", ".docx", ".md", ".html"),
            http_route="/api/tools/convert_document",
            voice_capability_line="convert a document between PDF and Word (convert_document)",
        )

    # ------------------------------------------------------------------
    # Pure re-render — no artifact lookup. Used by the HTTP route's
    # ``_convert_document`` helper so the existing UI keeps working.
    # ------------------------------------------------------------------

    @staticmethod
    async def render_bytes(info: dict, target: str) -> tuple[bytes, str]:
        """Re-render structured document source to PDF or DOCX.

        ``info`` is the artifact metadata dict (as returned by
        :meth:`ArtifactStore.get`). Must carry ``source_json``.
        Returns ``(bytes, new_extension)``.
        """
        src_json = info.get("source_json")
        if not src_json:
            raise ValueError("Document has no source to re-render")
        src = json.loads(src_json) if isinstance(src_json, str) else src_json

        title = src.get("title") or info.get("display_name") or "Document"
        author = src.get("author") or ""
        sections = src.get("sections") or []
        theme_field = src.get("theme")
        theme = (
            theme_field.get("preset")
            if isinstance(theme_field, dict)
            else (theme_field or "")
        )

        target = target.lower()
        if target == "pdf":
            from augmentum.tools.artifact_document import _render_pdf

            data = await asyncio.to_thread(_render_pdf, title, author, sections, theme or "")
            return data, "pdf"
        if target == "docx":
            from augmentum.tools.artifact_document import _render_docx

            data = await asyncio.to_thread(_render_docx, title, author, sections, theme or "")
            return data, "docx"
        raise ValueError(f"Unsupported document target: {target}")

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

        try:
            data, new_ext = await self.render_bytes(info, target)
        except ValueError as exc:
            return ToolResult(success=False, error=str(exc), validation_error=True)
        except Exception as exc:  # noqa: BLE001
            log.warning("convert_document_failed", artifact_id=artifact_id, error=str(exc))
            return ToolResult(success=False, error=f"Conversion failed: {exc}")

        base = info.get("display_name") or info.get("filename") or "document"
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
            log.warning("convert_document_save_failed", artifact_id=artifact_id, error=str(exc))
            return ToolResult(success=False, error=f"Save failed: {exc}")

        card = make_artifact_card(
            saved,
            kind="artifact",
            title=f"{stem}.{new_ext}",
            subtitle=f"Converted to {new_ext.upper()}",
            summary=f"Re-rendered document to {new_ext.upper()} ({len(data)} bytes)",
        )
        return ToolResult(
            success=True,
            output=f"Converted document to {new_ext.upper()} ({len(data)} bytes). New artifact: {saved['id']}",
            metadata={
                "artifact_id": saved["id"],
                "filename": saved["filename"],
                "format": new_ext,
                "size_bytes": len(data),
            },
            card=card,
        )
