"""Lightweight export tools — markdown, CSV, and code file artifacts.

These tools save text content as downloadable files through the artifact
store.  No rendering libraries needed — just raw text with the right
extension and content type.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from augmentum.tools.base import Tool, ToolCategory, ToolResult
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.tools.artifact_storage import ArtifactStore
    from augmentum.tools.base import SurfaceExposure

log = get_logger(__name__)


class MarkdownExportTool(Tool):
    """Export markdown content as a downloadable .md file."""

    def __init__(self, artifact_store: ArtifactStore) -> None:
        self._store = artifact_store

    @property
    def name(self) -> str:
        return "export_markdown"

    @property
    def description(self) -> str:
        return (
            "Save markdown content as a downloadable .md file. "
            "Write the FULL content in the content field — complete, polished prose. "
            "Do not use placeholders or summaries. Returns a download link."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.ARTIFACT

    @property
    def error_hints(self) -> dict[str, str]:
        return {
            "No content": "The content field was empty. Write the full document text in the content parameter.",
        }

    @property
    def produces(self) -> list[str]:
        return ["artifact_url"]

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Filename (without extension)",
                },
                "content": {
                    "type": "string",
                    "description": "Markdown content to export",
                },
            },
            "required": ["title", "content"],
        }

    async def execute(
        self, *, title: str = "document", content: str = "",
        task_id: str = "", session_id: str = "", **kwargs,
    ) -> ToolResult:
        if not content:
            return ToolResult(success=False, error="No content provided")

        safe_title = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_")[:60]
        filename = f"{safe_title}.md"

        try:
            info = await self._store.save(
                data=content.encode("utf-8"),
                filename=filename,
                fmt="md",
                task_id=task_id,
                session_id=session_id,
                display_name=f"{title}.md",
                metadata={"page_type": "markdown", "char_count": len(content)},
                user_id=Tool.extract_user_id(kwargs),
            )
            return ToolResult(
                success=True,
                output=(
                    f"Markdown exported: **{info['display_name']}** "
                    f"({info['size_bytes']:,} bytes)\n"
                    f"Download: {info['download_url']}"
                ),
                metadata=info,
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Export failed: {e}")


class CsvExportTool(Tool):
    """Export tabular data as a downloadable .csv file."""

    def __init__(self, artifact_store: ArtifactStore) -> None:
        self._store = artifact_store

    @property
    def name(self) -> str:
        return "export_csv"

    @property
    def description(self) -> str:
        return (
            "Save tabular data as a downloadable .csv file. "
            "Lighter than create_spreadsheet — use when the user needs "
            "data they can import into other tools. Returns a download link."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.ARTIFACT

    @property
    def surfaces(self) -> SurfaceExposure:
        from augmentum.tools.base import SurfaceExposure

        return SurfaceExposure(
            chat=True,
            coder=True,
            artifact_studio=True,
            file_context_menu=(".csv", ".json", ".xlsx"),
        )

    @property
    def error_hints(self) -> dict[str, str]:
        return {
            "No data": "Both headers and rows must be provided. Headers is a list of column names, rows is a list of lists or dicts.",
        }

    @property
    def produces(self) -> list[str]:
        return ["artifact_url"]

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Filename (without extension)",
                },
                "headers": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Column header names",
                },
                "rows": {
                    "type": "array",
                    "description": "Data rows (each row is an array of values)",
                    "items": {
                        "type": "array",
                        "items": {},
                    },
                },
            },
            "required": ["title", "headers", "rows"],
        }

    async def execute(
        self, *, title: str = "data", headers: list | None = None,
        rows: list | None = None,
        task_id: str = "", session_id: str = "", **kwargs,
    ) -> ToolResult:
        from augmentum.tools.artifact_normalize import normalize_list, normalize_str

        headers = [normalize_str(h) for h in normalize_list(headers)]
        rows = normalize_list(rows)

        if not headers:
            return ToolResult(success=False, error="No headers provided")

        # Build CSV content
        import csv
        import io

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(headers)
        for row in rows:
            if isinstance(row, list):
                writer.writerow(row)
            elif isinstance(row, dict):
                writer.writerow([row.get(h, "") for h in headers])
        csv_text = buf.getvalue()

        safe_title = re.sub(r"[^\w\s-]", "", title).strip().replace(" ", "_")[:60]
        filename = f"{safe_title}.csv"

        try:
            info = await self._store.save(
                data=csv_text.encode("utf-8"),
                filename=filename,
                fmt="csv",
                task_id=task_id,
                session_id=session_id,
                display_name=f"{title}.csv",
                user_id=Tool.extract_user_id(kwargs),
                metadata={
                    "page_type": "csv",
                    "column_count": len(headers),
                    "row_count": len(rows),
                },
            )
            return ToolResult(
                success=True,
                output=(
                    f"CSV exported: **{info['display_name']}** "
                    f"({len(rows)} rows, {len(headers)} columns)\n"
                    f"Download: {info['download_url']}"
                ),
                metadata=info,
            )
        except Exception as e:
            return ToolResult(success=False, error=f"CSV export failed: {e}")


class CodeExportTool(Tool):
    """Export code as a downloadable file with the correct extension."""

    def __init__(self, artifact_store: ArtifactStore) -> None:
        self._store = artifact_store

    @property
    def name(self) -> str:
        return "export_code"

    @property
    def description(self) -> str:
        return (
            "Save code as a downloadable file with the appropriate extension "
            "(.py, .js, .ts, .html, .css, .sh, .sql, .json, .yaml, etc.). "
            "Write complete, working code — not snippets or pseudocode. "
            "Returns a download link."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.ARTIFACT

    @property
    def error_hints(self) -> dict[str, str]:
        return {
            "No content": "The content field was empty. Write the full code in the content parameter.",
        }

    @property
    def produces(self) -> list[str]:
        return ["artifact_url"]

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "filename": {
                    "type": "string",
                    "description": "Full filename with extension (e.g. 'app.py', 'index.html')",
                },
                "content": {
                    "type": "string",
                    "description": "The complete file content",
                },
                "language": {
                    "type": "string",
                    "description": "Programming language (for metadata)",
                    "default": "",
                },
            },
            "required": ["filename", "content"],
        }

    async def execute(
        self, *, filename: str = "code.txt", content: str = "",
        language: str = "",
        task_id: str = "", session_id: str = "", **kwargs,
    ) -> ToolResult:
        if not content:
            return ToolResult(success=False, error="No content provided")

        # Sanitize filename but preserve the extension
        safe = re.sub(r"[^\w.\-]", "_", filename)[:80]
        ext = safe.rsplit(".", 1)[-1] if "." in safe else "txt"

        try:
            info = await self._store.save(
                data=content.encode("utf-8"),
                filename=safe,
                fmt=ext,
                task_id=task_id,
                session_id=session_id,
                display_name=filename,
                metadata={
                    "page_type": "code",
                    "language": language or ext,
                    "line_count": content.count("\n") + 1,
                },
                user_id=Tool.extract_user_id(kwargs),
            )
            return ToolResult(
                success=True,
                output=(
                    f"Code file saved: **{info['display_name']}** "
                    f"({info['size_bytes']:,} bytes, "
                    f"{content.count(chr(10)) + 1} lines)\n"
                    f"Download: {info['download_url']}"
                ),
                metadata=info,
            )
        except Exception as e:
            return ToolResult(success=False, error=f"Code export failed: {e}")
