"""Dual-harness agent execution — ReWOO for small models, ReAct for large.

Adapts execution strategy based on model capability:

ReWOO (Reasoning Without Observation):
  - Model outputs ONE plan with ALL tool calls as a JSON array
  - We execute them sequentially, no LLM calls during execution
  - Deterministic verification after all tools complete
  - If verification fails: ONE correction pass with error context
  - Total LLM calls: 1-2 (plan + optional fix)
  - Best for: small models (< 14B), fast/cheap models, simple tasks

ReAct (Reason + Act):
  - Model outputs ONE tool call per turn
  - We execute, feed result back, model decides next action
  - LLM sees each result before deciding next step
  - Total LLM calls: N (one per tool call)
  - Best for: large models (14B+), cloud APIs, complex/exploratory tasks

The router selects harness based on model name, tracks success rates,
and adapts over time.

Architecture mirrors the ApplicationBuilderTool pipeline:
  - PassResult for structured step outcomes
  - Error-preserving context (Manus pattern)
  - Regression detection (rollback if fix makes things worse)
  - Deterministic validation before LLM correction
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, AsyncIterator

from augmentum.coder.prompts import ACT_SYSTEM, EDIT_FORMAT_INSTRUCTIONS
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.coder.containers import ContainerManager
    from augmentum.coder.state import CoderState
    from augmentum.models.base import InternalStreamChunk, ModelBackend
    from augmentum.tools.base import Tool

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Router — select harness based on model capability
# ---------------------------------------------------------------------------

# Models where ReAct (multi-turn native tool calling) is proven reliable.
# Most models work BETTER with ReWOO (single JSON array) for this use case.
# Only add models here after testing them with the harness at 80%+ success.
_REACT_CAPABLE: set[str] = {
    # Currently empty — ReWOO is default and works across all tested models.
    # The adaptive router will switch to react if rewoo fails consistently.
}

# Per-model success tracking (persisted in memory, not DB — resets on restart)
_model_stats: dict[str, dict] = {}


def select_harness(model_name: str) -> str:
    """Select 'rewoo' or 'react' for a model.

    Checks learned stats first, then known model families, defaults to react.
    ReAct is the only strategy that re-prompts the model after each tool
    result, which is required for the >2-step chain execution users expect.
    """
    model_lower = model_name.lower()

    # Check learned stats (3+ attempts needed)
    if model_lower in _model_stats:
        stats = _model_stats[model_lower]
        if stats["attempts"] >= 3:
            return stats["harness"]

    # Check known models
    for family in _REACT_CAPABLE:
        if family in model_lower:
            return "react"

    # Default: ReAct — the observe→reprompt loop is needed for multi-step
    # tasks. The adaptive router will demote to rewoo if react fails
    # consistently for a given model.
    return "react"


def record_result(model_name: str, harness: str, success: bool) -> None:
    """Track success rate for adaptive routing."""
    model_lower = model_name.lower()
    if model_lower not in _model_stats:
        _model_stats[model_lower] = {
            "harness": harness, "success_rate": 0.5, "attempts": 0,
        }
    stats = _model_stats[model_lower]
    stats["attempts"] += 1
    # Exponential moving average
    alpha = 0.3
    stats["success_rate"] = alpha * (1.0 if success else 0.0) + (1 - alpha) * stats["success_rate"]
    # Switch harness if failing consistently
    if stats["attempts"] >= 5 and stats["success_rate"] < 0.25:
        stats["harness"] = "react" if harness == "rewoo" else "rewoo"
        stats["attempts"] = 0
        log.info("harness_switched", model=model_name,
                 from_harness=harness, to_harness=stats["harness"])


# ---------------------------------------------------------------------------
# ReWOO Prompt
# ---------------------------------------------------------------------------

REWOO_PLAN_SYSTEM = """\
You are a coding agent. Read the user's task and output a JSON array of tool calls.

Output ONLY a valid JSON array. No explanation. No markdown. No text before or after.

Tools (use exact names and input fields):
- dir_tree: {"path": "/workspace", "depth": 3}  — directory hierarchy
- file_read: {"path": "/workspace/file.py"}  — read file (REQUIRED before code_edit)
- file_write: {"path": "/workspace/file.py", "content": "full content"}  — create/overwrite file
- file_list: {"path": "/workspace"}  — flat directory listing
- code_edit: {"path": "/workspace/file.py", "search": "old text", "replace": "new text"}  — edit existing file
- code_grep: {"pattern": "search_term", "path": "/workspace"}  — regex search
- find_files: {"pattern": "*.py", "path": "/workspace"}  — find files by pattern
- code_search: {"query": "authentication logic", "limit": 5}  — semantic search
- shell_exec: {"command": "python3 script.py"}  — run any command
- shell_read: {"command": "cat file.txt"}  — read-only command
- git: {"action": "status"} / {"action": "diff"} / {"action": "commit", "message": "msg"}  — git operations
- test_run: {"command": "pytest -x"} or {}  — run tests (auto-detects framework)
- env_info: {}  — show installed runtimes, packages, disk usage
- doc_search: {"query": "python asyncio", "language": "python"}  — search documentation
- doc_fetch: {"url": "https://docs.python.org/3/..."}  — read a documentation page

Rules:
- Use python3 (not python) for all commands
- file_read before code_edit (always required)
- ALWAYS include a final verification step (test_run or shell_exec)
- file_write for new files, code_edit for existing files
- Include ALL steps in one JSON array — do not stop early
- Use dir_tree to explore project structure before editing

Example — "Create a hello world script and run it":
[{"tool": "file_write", "input": {"path": "/workspace/hello.py", "content": "print('Hello, World!')\\n"}}, {"tool": "shell_exec", "input": {"command": "python3 /workspace/hello.py"}}]

Example — "Fix the bug in auth.py":
[{"tool": "dir_tree", "input": {"path": "/workspace", "depth": 2}}, {"tool": "file_read", "input": {"path": "/workspace/auth.py"}}, {"tool": "code_edit", "input": {"path": "/workspace/auth.py", "search": "old code", "replace": "fixed code"}}, {"tool": "test_run", "input": {}}]

Now output the JSON array for the user's task:\
"""

REWOO_FIX_SYSTEM = """\
The previous plan had errors. Fix them by outputting a NEW JSON array of tool calls.

Previous errors:
{errors}

Previous plan (do NOT repeat the same mistakes):
{previous_plan}

Output ONLY a corrected JSON array of tool calls:\
"""


# ---------------------------------------------------------------------------
# ReWOO Executor
# ---------------------------------------------------------------------------

@dataclass
class ToolCallSpec:
    """A planned tool call from ReWOO output."""
    tool_name: str
    tool_input: dict
    index: int


def parse_rewoo_plan(text: str, known_tools: set[str]) -> list[ToolCallSpec]:
    """Parse a ReWOO JSON array of tool calls.

    Handles models that wrap in markdown code fences or add extra text.
    """
    text = text.strip()
    # Strip code fences
    if "```" in text:
        text = re.sub(r"```[\w]*\n?", "", text).replace("```", "").strip()

    # Find the JSON array
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end <= start:
        # Try individual JSON objects
        calls = []
        for match in re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}', text):
            try:
                obj = json.loads(match.group())
                name = obj.get("tool", "")
                inp = obj.get("input", {})
                if name in known_tools:
                    calls.append(ToolCallSpec(tool_name=name, tool_input=inp, index=len(calls)))
            except json.JSONDecodeError:
                continue
        return calls

    try:
        arr = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        # Array parse failed (truncated or malformed JSON)
        # Fall back to extracting individual tool objects
        calls = []
        for match in re.finditer(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}', text):
            try:
                obj = json.loads(match.group())
                name = obj.get("tool", "")
                inp = obj.get("input", {})
                if name in known_tools:
                    calls.append(ToolCallSpec(tool_name=name, tool_input=inp, index=len(calls)))
            except json.JSONDecodeError:
                continue
        return calls

    if not isinstance(arr, list):
        return []

    calls = []
    for i, item in enumerate(arr):
        if not isinstance(item, dict):
            continue
        name = item.get("tool", "")
        inp = item.get("input", {})
        if isinstance(inp, str):
            try:
                inp = json.loads(inp)
            except json.JSONDecodeError:
                inp = {}
        if name in known_tools:
            calls.append(ToolCallSpec(tool_name=name, tool_input=inp, index=i))

    return calls
