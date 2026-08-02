"""File operations tool — read, write, list files in a sandboxed directory."""

from __future__ import annotations

import asyncio
from pathlib import Path

from augmentum.tools.base import Tool, ToolCategory, ToolResult
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


class FileOpsTool(Tool):
    """Read, write, and list files in the working directory."""

    @property
    def name(self) -> str:
        return "file_ops"

    @property
    def description(self) -> str:
        return "Read, write, and list files in the working directory"

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.FILE

    @property
    def error_hints(self) -> dict[str, str]:
        return {
            "not found": "The file doesn't exist at that path. Use operation='list' first to see available files.",
            "Permission denied": "Cannot access that file — it's outside the allowed directory.",
            "path traversal": "Paths must be relative to the working directory. Don't use '..' or absolute paths.",
        }

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["read", "write", "list", "exists"],
                    "description": "File operation to perform",
                },
                "path": {
                    "type": "string",
                    "description": "File path relative to the working directory",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write (only for 'write' operation)",
                    "default": "",
                },
            },
            "required": ["operation", "path"],
        }

    @property
    def cacheable(self) -> bool:
        return False

    def __init__(self, base_dir: str = "/data/workdir") -> None:
        self._base_dir = Path(base_dir).resolve()

    def _resolve_safe_path(self, path: str) -> Path:
        """Resolve a user-supplied path and verify it stays under base_dir.

        Raises ValueError if the resolved path escapes the sandbox.
        """
        # Resolve relative to base_dir, then ensure it is a descendant.
        target = (self._base_dir / path).resolve()
        # Use os.path prefix check (works on both POSIX and Windows).
        try:
            target.relative_to(self._base_dir)
        except ValueError:
            raise ValueError(
                f"Path traversal blocked: '{path}' resolves outside the working directory"
            ) from None
        return target

    def validate_input(self, **kwargs) -> bool:
        op = kwargs.get("operation", "")
        path = kwargs.get("path", "")
        return op in ("read", "write", "list", "exists") and isinstance(path, str) and len(path) > 0

    async def execute(
        self,
        *,
        operation: str,
        path: str,
        content: str = "",
    ) -> ToolResult:
        """Perform a file operation within the sandboxed working directory."""
        if operation not in ("read", "write", "list", "exists"):
            return ToolResult(
                success=False,
                error=f"Unknown operation: {operation} (expected read, write, list, or exists)",
            )

        try:
            safe_path = self._resolve_safe_path(path)
        except ValueError as exc:
            return ToolResult(success=False, error=str(exc))

        if operation == "read":
            return await self._read(safe_path)
        if operation == "write":
            return await self._write(safe_path, content)
        if operation == "list":
            return await self._list(safe_path)
        # operation == "exists"
        return await self._exists(safe_path)

    async def _read(self, path: Path) -> ToolResult:
        """Read and return the contents of a file."""
        try:
            text = await asyncio.to_thread(path.read_text, encoding="utf-8")
        except FileNotFoundError:
            return ToolResult(success=False, error=f"File not found: {path.name}")
        except Exception as exc:
            return ToolResult(success=False, error=f"Read error: {exc}")

        return ToolResult(
            success=True,
            output=text,
            metadata={"path": str(path), "size": len(text)},
        )

    async def _write(self, path: Path, content: str) -> ToolResult:
        """Write content to a file, creating parent directories as needed."""
        try:
            await asyncio.to_thread(path.parent.mkdir, parents=True, exist_ok=True)
            await asyncio.to_thread(path.write_text, content, encoding="utf-8")
        except Exception as exc:
            return ToolResult(success=False, error=f"Write error: {exc}")

        return ToolResult(
            success=True,
            output=f"Wrote {len(content)} characters to {path.name}",
            metadata={"path": str(path), "size": len(content)},
        )

    async def _list(self, path: Path) -> ToolResult:
        """List files and directories at the given path."""
        try:
            if not path.is_dir():
                return ToolResult(success=False, error=f"Not a directory: {path.name}")
            entries = sorted(path.iterdir())
        except Exception as exc:
            return ToolResult(success=False, error=f"List error: {exc}")

        lines = []
        for entry in entries:
            kind = "dir" if entry.is_dir() else "file"
            lines.append(f"[{kind}] {entry.name}")

        output = "\n".join(lines) if lines else "(empty directory)"

        return ToolResult(
            success=True,
            output=output,
            metadata={"path": str(path), "count": len(entries)},
        )

    async def _exists(self, path: Path) -> ToolResult:
        """Check whether a file or directory exists."""
        exists = path.exists()
        kind = ""
        if exists:
            kind = "directory" if path.is_dir() else "file"

        return ToolResult(
            success=True,
            output=f"{'Exists' if exists else 'Does not exist'}"
            + (f" ({kind})" if kind else ""),
            metadata={"path": str(path), "exists": exists, "type": kind},
        )
