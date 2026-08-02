"""search_files tool — search the user's files across all sources."""

from __future__ import annotations

from augmentum.tools.base import Tool, ToolCategory, ToolResult
from augmentum.utils.logging import get_logger
from augmentum.vfs.context import build_file_context, detect_tier

log = get_logger(__name__)


class SearchFilesTool(Tool):
    """Search the user's files across all sources."""

    @property
    def name(self) -> str:
        return "search_files"

    @property
    def description(self) -> str:
        return (
            "Search the user's files across all sources — documents, images, "
            "artifacts, knowledge packs, and more. Returns file metadata, "
            "descriptions, and tags. Use when the user asks about their files, "
            "wants to find something they created, or references a document."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.SEARCH

    @property
    def timeout(self) -> float:
        return 10.0

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query — can be a filename, description, tag, or natural language",
                },
                "type": {
                    "type": "string",
                    "enum": ["all", "documents", "images", "artifacts", "audio", "knowledge"],
                    "description": "Filter by file type (default: all)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max results to return (default: 5)",
                },
            },
            "required": ["query"],
        }

    @property
    def error_hints(self) -> dict[str, str]:
        return {
            "No files found": "Try broader search terms or a different file type filter.",
        }

    async def execute(self, **kwargs) -> ToolResult:
        query = kwargs.get("query", "")
        file_type = kwargs.get("type", "all")
        limit = min(kwargs.get("limit", 5), 10)
        context = kwargs.get("_context", {})
        user_id = context.get("user_id", "")

        if not query:
            return ToolResult(success=False, output="Please provide a search query.")

        if not user_id:
            return ToolResult(success=False, output="Authentication required to search files.")

        file_index = context.get("file_index")
        if not file_index:
            return ToolResult(success=False, output="File index not available.")

        # Map type filter to source name used in the index
        source_map = {
            "documents": "documents",
            "images": "images",
            "artifacts": "artifacts",
            "audio": "voices",
            "knowledge": "knowledge",
        }
        source = source_map.get(file_type)  # None means all sources

        results = await file_index.search(
            query, user_id=user_id, source=source, limit=limit,
        )

        if not results:
            return ToolResult(success=True, output=f"No files found matching '{query}'.")

        ctx_length = context.get("context_length", 4096)
        tier = detect_tier(ctx_length)
        output = build_file_context(results, tier)

        # Indexed file content can carry injected instructions (a malicious
        # uploaded doc) — same threat class as documents/rag, which is wrapped.
        from augmentum.security.untrusted import wrap_untrusted
        return ToolResult(success=True, output=wrap_untrusted("files/search", output))
