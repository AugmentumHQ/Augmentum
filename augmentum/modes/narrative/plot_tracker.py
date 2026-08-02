"""Plot tracker — tracks active narrative arcs and plot threads."""

from __future__ import annotations

import re
from dataclasses import dataclass

from augmentum.state.narrative_state import PlotStatus, PlotThread, _new_id
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Patterns that suggest plot progression
_PLOT_PROGRESSION_PATTERNS = [
    re.compile(r"(?:quest|mission|task|objective|goal)", re.IGNORECASE),
    re.compile(r"(?:discover(?:ed|s|ing)?|reveal(?:ed|s|ing)?|learn(?:ed|s|ing)?)", re.IGNORECASE),
    re.compile(r"(?:accomplish(?:ed)?|complet(?:ed|es|ing)|finish(?:ed|es|ing)|succeed(?:ed)?)", re.IGNORECASE),
    re.compile(r"(?:fail(?:ed|s|ing)?|defeat(?:ed)?|los(?:t|e|ing))", re.IGNORECASE),
    re.compile(r"(?:betray(?:ed|al)?|secret|conspiracy|plot|scheme)", re.IGNORECASE),
]

# Patterns that suggest a plot resolution
_RESOLUTION_PATTERNS = [
    re.compile(r"(?:finally|at\s+last|in\s+the\s+end)", re.IGNORECASE),
    re.compile(r"(?:resolved?|concluded?|settled?|ended?)", re.IGNORECASE),
    re.compile(r"(?:victori(?:ous|y)|triumph(?:ed|ant)?|prevail(?:ed)?)", re.IGNORECASE),
    re.compile(r"(?:peace\s+was|order\s+was|balance\s+was)\s+(?:restored|achieved)", re.IGNORECASE),
]


@dataclass
class PlotUpdate:
    """Extracted plot-relevant information from a message."""

    new_threads: list[str] | None = None  # New plot thread descriptions
    progressed_threads: list[str] | None = None  # IDs of progressed threads
    resolved_threads: list[str] | None = None  # IDs of resolved threads
    plot_signals: list[str] | None = None  # Detected plot keywords


class PlotTracker:
    """Tracks narrative plot threads across messages."""

    def __init__(self) -> None:
        self._threads: dict[str, PlotThread] = {}

    @property
    def threads(self) -> dict[str, PlotThread]:
        return dict(self._threads)

    @property
    def active_threads(self) -> list[PlotThread]:
        return [t for t in self._threads.values() if t.status == PlotStatus.ACTIVE]

    def extract_plot_signals(self, text: str) -> list[str]:
        """Extract plot-relevant signals from text (heuristic)."""
        signals = []
        for pattern in _PLOT_PROGRESSION_PATTERNS:
            if pattern.search(text):
                signals.append(pattern.pattern)
        return signals

    def detect_resolutions(self, text: str) -> bool:
        """Check if text suggests a plot resolution."""
        return any(p.search(text) for p in _RESOLUTION_PATTERNS)

    def add_thread(
        self,
        session_id: str,
        title: str,
        description: str = "",
        message_index: int = 0,
        branch_id: str = "main",
    ) -> PlotThread:
        """Add a new plot thread."""
        thread = PlotThread(
            id=_new_id(),
            session_id=session_id,
            title=title,
            description=description,
            status=PlotStatus.ACTIVE,
            established_at=message_index,
            branch_id=branch_id,
        )
        self._threads[thread.id] = thread
        log.info("plot_thread_added", title=title, id=thread.id)
        return thread

    def progress_thread(self, thread_id: str, update: str, message_index: int) -> None:
        """Record progression on a plot thread."""
        thread = self._threads.get(thread_id)
        if not thread:
            return

        # Store progression in state
        progressions = thread.state.get("progressions", [])
        progressions.append({
            "message_index": message_index,
            "update": update,
        })
        thread.state["progressions"] = progressions
        log.debug("plot_progressed", title=thread.title, message_index=message_index)

    def resolve_thread(self, thread_id: str, message_index: int) -> None:
        """Mark a plot thread as resolved."""
        thread = self._threads.get(thread_id)
        if not thread:
            return

        thread.status = PlotStatus.RESOLVED
        thread.resolved_at = message_index
        log.info("plot_resolved", title=thread.title, message_index=message_index)

    def pause_thread(self, thread_id: str) -> None:
        """Pause a plot thread."""
        thread = self._threads.get(thread_id)
        if thread:
            thread.status = PlotStatus.PAUSED
            log.debug("plot_paused", title=thread.title)

    def resume_thread(self, thread_id: str) -> None:
        """Resume a paused plot thread."""
        thread = self._threads.get(thread_id)
        if thread and thread.status == PlotStatus.PAUSED:
            thread.status = PlotStatus.ACTIVE
            log.debug("plot_resumed", title=thread.title)

    def rollback_to(self, message_index: int, branch_id: str = "main") -> None:
        """Roll back plot state to a specific message index."""
        to_remove = []
        for tid, thread in self._threads.items():
            if thread.branch_id == branch_id:
                if thread.established_at > message_index:
                    to_remove.append(tid)
                elif thread.resolved_at and thread.resolved_at > message_index:
                    thread.status = PlotStatus.ACTIVE
                    thread.resolved_at = None
                    # Trim progressions
                    progressions = thread.state.get("progressions", [])
                    thread.state["progressions"] = [
                        p for p in progressions if p["message_index"] <= message_index
                    ]

        for tid in to_remove:
            del self._threads[tid]

        if to_remove:
            log.info("plots_rolled_back", removed=len(to_remove), to_index=message_index)

    def set_threads(self, threads: list[PlotThread]) -> None:
        """Set threads directly (used when loading from DB)."""
        self._threads = {t.id: t for t in threads}

    def get_context_summary(self, max_length: int = 500) -> str:
        """Generate a summary of active plots for context injection."""
        active = self.active_threads
        if not active:
            return ""

        parts = ["Active plot threads:"]
        remaining = max_length - len(parts[0])

        for thread in active:
            line = f"- {thread.title}"
            if thread.description:
                line += f": {thread.description[:100]}"
            if len(line) > remaining:
                break
            parts.append(line)
            remaining -= len(line)

        return "\n".join(parts)
