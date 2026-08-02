"""Analytical mode state data models — tracks UARF pipeline state."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class AnalyticalPhase(str, Enum):
    """Phases of the UARF analytical pipeline."""

    ASSESS = "assess"
    IDENTIFY = "identify"
    GATHER = "gather"      # merged IDENTIFY+RELEVANT for moderate queries
    RELEVANT = "relevant"
    APPLY = "apply"
    VERIFY = "verify"
    CONCLUDE = "conclude"
    RESPOND = "respond"    # merged APPLY+CONCLUDE for simple queries


@dataclass
class PhaseResult:
    """Result of a single analytical phase."""

    phase: AnalyticalPhase
    output: str
    confidence: float = 0.0
    needs_backtrack: bool = False
    backtrack_reason: str = ""
    tokens_used: int = 0


@dataclass
class AnalyticalResult:
    """Final result of the full UARF pipeline."""

    conclusion: str
    phase_results: dict[str, PhaseResult]
    complexity: str = "moderate"
    total_tokens: int = 0
    backtrack_count: int = 0


@dataclass
class AnalyticalState:
    """Complete in-memory state for an analytical session."""

    query: str = ""
    complexity: str = "moderate"
    current_phase: AnalyticalPhase = AnalyticalPhase.ASSESS
    phase_results: dict[str, PhaseResult] = field(default_factory=dict)
    backtrack_count: int = 0
    max_backtracks: int = 3
    facts_identified: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    sub_tasks: list[str] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)

    # Auto-search state
    needs_search: bool = False
    search_queries: list[str] = field(default_factory=list)
    search_context: str = ""  # formatted results block for prompt injection

    # Search retry tracking
    search_retry_count: int = 0
    search_result_count: int = 0
    search_needed_by_verify: bool = False

    # Conversation history context
    conversation_context: str = ""  # formatted prior turns for prompt injection

    # Automated verification results (tool-based checks on APPLY output)
    auto_verify_summary: str = ""  # formatted results for VERIFY prompt injection


@dataclass
class ToolCallRecord:
    """Record of a tool invocation during an analytical phase."""

    phase: str
    tool_name: str
    input_data: dict = field(default_factory=dict)
    output: str = ""
    success: bool = False
    # Structured presentation envelope from ToolResult.card (forwarded
    # to the frontend so the UI can render a typed card with preview /
    # edit / download instead of plain markdown).
    card: dict | None = None
