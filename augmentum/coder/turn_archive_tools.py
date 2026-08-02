"""Recall tools — agent-facing surface for the durable turn archive.

Exposes two tools to the coder agent:

* ``recall(query, k=3)`` — semantic search over past turns in this
  workspace. Returns top-k hits with turn_index, user_goal, outcome,
  summary, and event_time so the agent can decide whether the past
  work is relevant.
* ``recall_expand(turn_index, before=2, after=2)`` — fetch turns
  adjacent to a known turn_index for context. Use after recall
  identifies a relevant moment.

Both tools are gated on the durable archive being available —
``db_conn=None`` callers get a graceful "archive not wired" error,
not a crash. Settings-tunable via ``coder_archive_recall_tool_exposed``
(when added to settings registry).

Framing — every result is rendered with an explicit "HISTORICAL —
NOT current task" header so the model doesn't treat returned file
state as authoritative. The framing matches the design spec
(see project_coder_turn_archive memory).
"""
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from augmentum.coder.tools import _CoderTool, _truncate
from augmentum.tools.base import ToolCategory, ToolResult
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    import aiosqlite

log = get_logger(__name__)


def _format_ago(event_time: int) -> str:
    """Human-readable 'N ago' string for a unix timestamp."""
    if not event_time:
        return ""
    delta = max(0, int(time.time() - event_time))
    if delta < 60:
        return f"{delta}s ago"
    if delta < 3600:
        return f"{delta // 60}m ago"
    if delta < 86_400:
        return f"{delta // 3600}h ago"
    return f"{delta // 86_400}d ago"


def _render_hit_line(hit, *, verbose: bool = False) -> str:
    """One-line summary of a recall hit for the tool output."""
    ago = _format_ago(hit.event_time)
    parts = [f"Turn {hit.turn_index}"]
    if ago:
        parts.append(ago)
    parts.append(f"Outcome: {hit.outcome or '—'}")
    head = " — ".join(parts)
    goal = (hit.user_goal or "").strip()
    if goal:
        goal_preview = goal[:160] + ("…" if len(goal) > 160 else "")
    else:
        goal_preview = "(no goal recorded)"
    out = f"{head}\n  Goal: {goal_preview}"
    if verbose and hit.summary:
        s = hit.summary.strip().replace("\n", " ")
        if len(s) > 200:
            s = s[:200] + "…"
        out += f"\n  Summary: {s}"
    if hit.files_edited_count or hit.files_read_count:
        out += (
            f"\n  Files: read={hit.files_read_count} "
            f"edited={hit.files_edited_count}"
        )
    return out


def _render_entry(entry, *, include_files: bool = False) -> str:
    """Render a TurnArchiveEntry (full row) for recall_expand output."""
    ago = _format_ago(entry.event_time)
    head = (
        f"Turn {entry.turn_index} — {ago or '(time unknown)'} — "
        f"Outcome: {entry.outcome or '—'}"
    )
    parts = [head, f"  Goal: {(entry.user_goal or '').strip()[:200]}"]
    if entry.summary:
        s = entry.summary.strip().replace("\n", " ")
        parts.append(f"  Summary: {s[:240]}{'…' if len(s) > 240 else ''}")
    if entry.verdict_reason:
        parts.append(f"  Verdict: {entry.verdict_reason}")
    if entry.blockers:
        parts.append(f"  Blocker: {entry.blockers[:160]}")
    if include_files and entry.files_edited:
        # Render up to 5 file edits compactly.
        names = []
        for item in (entry.files_edited or [])[:5]:
            if isinstance(item, dict):
                p = item.get("path") or ""
                if p:
                    names.append(p)
            elif isinstance(item, str):
                names.append(item)
        if names:
            parts.append(f"  Edited: {', '.join(names)}")
    return "\n".join(parts)


_HISTORICAL_HEADER = (
    "HISTORICAL CONTEXT — work already completed in this workspace.\n"
    "These results are READ-ONLY references. File contents shown are\n"
    "STALE — re-read with file_read if you need current state.\n"
    "Past outcomes are point-in-time — re-verify with tests/probes if\n"
    "current status matters.\n"
)


class _ArchiveTool(_CoderTool):
    """Common base for recall tools — exposes ``self._db_conn``.

    The conn is set via the ``db_conn`` kwarg threaded through
    ``create_coder_tools``. When ``None`` the tools gracefully refuse
    with a structured error rather than crashing — the agent can
    still continue without recall.
    """

    def __init__(self, *, db_conn=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._db_conn = db_conn

    def _unavailable_result(self) -> ToolResult:
        return ToolResult(
            success=False,
            error=(
                "Turn-archive recall is not wired in this session. "
                "Continue without recalled context — workspace_facts "
                "and the in-prompt prior_turns block still apply."
            ),
        )


class RecallTool(_ArchiveTool):
    """Semantic search over the workspace's archived turns."""

    @property
    def name(self) -> str:
        return "recall"

    @property
    def description(self) -> str:
        return (
            "Search past turns from this workspace's archive by topic. "
            "Returns top-k turn summaries with timestamps + outcomes.\n\n"
            "Returned turns are HISTORICAL (already completed or "
            "abandoned). Use when:\n"
            "  - The user asks 'have I done this before?' / 'how did I "
            "fix this last time?'\n"
            "  - Current task overlaps with prior work\n"
            "  - You want to avoid re-discovering something\n\n"
            "Do NOT treat returned file edits as pending — re-read "
            "files for current state. Do NOT cite returned outcomes "
            "as if they're current. Use recall_expand with a "
            "turn_index from results to fetch adjacent turns when a "
            "match looks relevant but you need more context."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.SEARCH

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Free-text description of what you're looking "
                        "for in past turns. E.g., 'fix auth bug', "
                        "'add timezone handling', 'last build failure'."
                    ),
                },
                "k": {
                    "type": "integer",
                    "description": (
                        "Number of top matches to return (1-10, "
                        "default 3). Higher k surfaces more candidates "
                        "but adds noise."
                    ),
                    "default": 3,
                },
            },
            "required": ["query"],
        }

    async def execute(
        self, *, query: str = "", k: int = 3, **_kwargs,
    ) -> ToolResult:
        query = (query or "").strip()
        if not query:
            return ToolResult(
                success=False,
                error="query is required",
                validation_error=True,
            )
        if self._db_conn is None:
            return self._unavailable_result()
        if not self._workspace_id:
            return self._unavailable_result()

        try:
            from augmentum.coder.turn_archive_embed import search_similar
            hits = await search_similar(
                self._db_conn,
                user_id=self._user_id,
                workspace_id=self._workspace_id,
                query=query,
                k=max(1, min(int(k or 3), 10)),
            )
        except Exception as exc:
            log.debug(
                "coder.recall_search_failed", error=str(exc)[:160],
            )
            return ToolResult(
                success=False,
                error=f"recall failed: {str(exc)[:200]}",
            )

        if not hits:
            return ToolResult(
                success=True,
                output=(
                    f"No archived turns matched '{query}' in this "
                    "workspace. The archive may be small (first few "
                    "turns) or the query may not match past work."
                ),
                metadata={"query": query, "hits": 0},
            )

        lines = [_HISTORICAL_HEADER, f"Query: '{query}'", ""]
        for hit in hits:
            lines.append(_render_hit_line(hit))
            lines.append("")

        return ToolResult(
            success=True,
            output=_truncate("\n".join(lines)),
            metadata={
                "query": query,
                "hits": len(hits),
                "top_turn_index": hits[0].turn_index if hits else 0,
                "top_distance": hits[0].distance if hits else 0.0,
                "framing": "historical_reference_only",
            },
        )


class RecallExpandTool(_ArchiveTool):
    """Fetch turns adjacent to a known turn_index for context."""

    @property
    def name(self) -> str:
        return "recall_expand"

    @property
    def description(self) -> str:
        return (
            "Fetch past turns around a specific turn_index (typically "
            "one returned by `recall`). Returns the matched turn + "
            "adjacent turns so you can see what happened just before "
            "and after.\n\n"
            "Same historical framing applies as `recall` — the entries "
            "are READ-ONLY references. Re-read files via file_read if "
            "you need current contents."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.SEARCH

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "turn_index": {
                    "type": "integer",
                    "description": (
                        "Turn index to center the window on (from a "
                        "prior recall result)."
                    ),
                },
                "before": {
                    "type": "integer",
                    "description": "Turns to include before (0-10, default 2).",
                    "default": 2,
                },
                "after": {
                    "type": "integer",
                    "description": "Turns to include after (0-10, default 2).",
                    "default": 2,
                },
            },
            "required": ["turn_index"],
        }

    async def execute(
        self,
        *,
        turn_index: int = 0,
        before: int = 2,
        after: int = 2,
        **_kwargs,
    ) -> ToolResult:
        try:
            center = int(turn_index)
        except Exception:
            return ToolResult(
                success=False,
                error="turn_index must be an integer",
                validation_error=True,
            )
        if center < 1:
            return ToolResult(
                success=False,
                error="turn_index must be >= 1",
                validation_error=True,
            )
        if self._db_conn is None:
            return self._unavailable_result()
        if not self._workspace_id:
            return self._unavailable_result()

        try:
            from augmentum.coder.turn_archive import get_turn_window
            entries = await get_turn_window(
                self._db_conn,
                user_id=self._user_id,
                workspace_id=self._workspace_id,
                center_turn_index=center,
                before=max(0, min(int(before or 0), 10)),
                after=max(0, min(int(after or 0), 10)),
            )
        except Exception as exc:
            log.debug(
                "coder.recall_expand_failed", error=str(exc)[:160],
            )
            return ToolResult(
                success=False,
                error=f"recall_expand failed: {str(exc)[:200]}",
            )

        if not entries:
            return ToolResult(
                success=True,
                output=(
                    f"No archived turns found in window around "
                    f"turn_index={center}. The turn may be older than "
                    "the workspace's row-cap horizon."
                ),
                metadata={"turn_index": center, "entries": 0},
            )

        lines = [
            _HISTORICAL_HEADER,
            f"Window around turn {center} (showing turns "
            f"{entries[0].turn_index}–{entries[-1].turn_index}):",
            "",
        ]
        for entry in entries:
            lines.append(_render_entry(entry, include_files=True))
            lines.append("")

        return ToolResult(
            success=True,
            output=_truncate("\n".join(lines)),
            metadata={
                "turn_index": center,
                "entries": len(entries),
                "first": entries[0].turn_index,
                "last": entries[-1].turn_index,
                "framing": "historical_reference_only",
            },
        )


__all__ = ["RecallTool", "RecallExpandTool"]
