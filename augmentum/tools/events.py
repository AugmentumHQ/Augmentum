"""Unified tool execution event schema.

One vocabulary for tool state across every mode so the UI renders any tool
with one component. Three events, emitted inside the NDJSON `augmentum`
metadata dict on a chat stream:

  tool_start    — before execution. Includes tool name, display label, and
                  an extracted context string (the query / URL / code / path
                  most relevant to the user).
  tool_progress — optional mid-flight update for long-running tools.
                  `percent` (0–100) or a free-form `message`.
  tool_complete — execution finished. success/failure + output preview +
                  duration_ms. Tools that return rich metadata (image_id,
                  project, youtube payload) pass those through in `extra`.

Legacy `tool_call` (post-execution only) is still emitted alongside
tool_complete for backward compatibility until the UI fully migrates.
"""
from __future__ import annotations

import time
from typing import Any

TOOL_LABELS: dict[str, str] = {
    "web_search": "Searching the web",
    "web_fetch": "Reading page",
    "web": "Searching the web",
    "calculator": "Calculating",
    "datetime": "Checking the time",
    "unit_converter": "Converting units",
    "python_exec": "Running code",
    "python_executor": "Running code",
    "file_search": "Searching files",
    "search_files": "Searching files",
    "read_file": "Reading file",
    "youtube": "Looking up video",
    "youtube_search": "Searching YouTube",
    "image_generation": "Generating image",
    "image_search": "Searching images",
    "app_builder": "Building application",
    "create_document": "Creating document",
    "create_presentation": "Creating slides",
    "create_spreadsheet": "Creating spreadsheet",
    "create_chart": "Creating chart",
    "create_ebook": "Creating ebook",
    "memory_recall": "Recalling memory",
    "flow_tool": "Running flow",
    "document_parse": "Parsing document",
    "export_document": "Exporting",
    "json_tool": "Parsing JSON",
    "hash_tool": "Hashing",
    "math_verify": "Verifying math",
}


_CONTEXT_FIELDS = (
    "query", "url", "prompt", "description", "code", "path",
    "video_id", "expression", "question", "filename", "from_unit",
)


def label_for(tool_name: str) -> str:
    """User-facing label for a tool — falls back to title-cased name."""
    if tool_name in TOOL_LABELS:
        return TOOL_LABELS[tool_name]
    return tool_name.replace("_", " ").title()


def extract_context(tool_name: str, args: dict[str, Any] | None) -> str:
    """Best human-readable string from tool args — shown as subtitle in UI.

    Search tools expose the query; fetchers expose the URL; code tools
    expose a trimmed code preview. Kept under 120 chars.
    """
    if not args:
        return ""
    for key in _CONTEXT_FIELDS:
        val = args.get(key)
        if val:
            s = str(val).strip()
            if not s:
                continue
            # Collapse whitespace for readable single-line display
            s = " ".join(s.split())
            return s if len(s) <= 120 else s[:117] + "…"
    return ""


def make_tool_start(
    tc_id: str, tool_name: str, args: dict[str, Any] | None,
    *, phase: str = "",
) -> dict[str, Any]:
    """tool_start event payload."""
    payload: dict[str, Any] = {
        "id": tc_id,
        "tool": tool_name,
        "label": label_for(tool_name),
        "started_at": time.time(),
    }
    ctx = extract_context(tool_name, args)
    if ctx:
        payload["context"] = ctx
    if phase:
        payload["phase"] = phase
    return payload


def make_tool_progress(
    tc_id: str, *, percent: float | None = None, message: str = "",
) -> dict[str, Any]:
    """tool_progress event payload."""
    payload: dict[str, Any] = {"id": tc_id}
    if percent is not None:
        payload["percent"] = max(0.0, min(100.0, float(percent)))
    if message:
        payload["message"] = message
    return payload


def make_tool_complete(
    tc_id: str, tool_name: str,
    *,
    success: bool,
    output_preview: str = "",
    error: str = "",
    duration_ms: int = 0,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """tool_complete event payload."""
    payload: dict[str, Any] = {
        "id": tc_id,
        "tool": tool_name,
        "success": success,
        "duration_ms": duration_ms,
    }
    if output_preview:
        payload["output_preview"] = output_preview[:240]
    if error:
        payload["error"] = error[:240]
    if extra:
        payload.update(extra)
    return payload
