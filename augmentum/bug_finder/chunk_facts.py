"""Pre-computed deterministic facts for one detector chunk.

The detector spends a non-trivial fraction of its iteration budget
*deciding* to call wiring tools. Each tool call costs a round-trip
(token-priced envelope + response) before the LLM gets to actually
reason about the chunk. For the most common wiring lookups — "are
there decorators on this function?", "what has this codebase seen
at this file before?" — the answers are deterministic and cheap.

This module pre-computes those facts in milliseconds and renders
them as a compact block injected directly into the detector's user
message. The LLM receives the wiring context as a fact-list, not as
a tool-call requirement. Tokens that would have been burned on
``tool_use`` + ``tool_result`` envelopes go into reasoning instead.

The pre-compute is best-effort: any failure is swallowed and an
empty block is returned. The detector's existing tool-call path
remains available as a fallback — the LLM can still call
``decorators_on`` / ``who_calls`` / etc. if the pre-computed facts
don't cover what it needs.

Design constraints:
* **Compact** — no fact block should exceed ~600 chars. The detector
  user prompt is already large with the chunk source.
* **Non-prescriptive** — facts are surfaced as observations, never as
  "you should think this is a bug" framing. Detector still has to
  reason.
* **Cheap** — total pre-compute time < 50ms per chunk. We're trading
  LLM round-trips for local AST work, which is a 1000:1 cost ratio
  in our favor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from augmentum.bug_finder.wiring import DecoratorInfo, decorators_on
from augmentum.bug_finder.workspace_substrate import (
    WorkspacePattern,
    load_workspace_patterns,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChunkFacts:
    """Deterministic facts gathered for one chunk.

    Empty fields mean "no signal" — the detector should not infer
    absence from the empty case (the gathering may have failed or
    the workspace may simply not have substrate yet).
    """

    decorators: tuple[DecoratorInfo, ...] = ()
    prior_patterns: tuple[WorkspacePattern, ...] = field(default_factory=tuple)

    @property
    def is_empty(self) -> bool:
        return not self.decorators and not self.prior_patterns


# ---------------------------------------------------------------------------
# Compute
# ---------------------------------------------------------------------------


def _filter_patterns_for_file(
    patterns: list[WorkspacePattern], *, file: str, limit: int = 5,
) -> tuple[WorkspacePattern, ...]:
    """Return up to ``limit`` workspace patterns whose ``file_pattern``
    overlaps with the chunk's file. We use substring match (not glob)
    because most pattern.json rows have concrete file paths and
    overlap is the natural semantic for "this file has had this
    pattern before".
    """
    file_norm = file.strip().lower()
    if not file_norm:
        return ()
    matched: list[WorkspacePattern] = []
    for p in patterns:
        if not p.file_pattern:
            continue
        if p.file_pattern.lower() in file_norm or file_norm in p.file_pattern.lower():
            matched.append(p)
    # Most-recurrent first — those are the strongest signal
    matched.sort(key=lambda p: (p.hit_count, p.last_seen_at), reverse=True)
    return tuple(matched[:limit])


def compute_chunk_facts(
    *,
    workspace_root: Path | None,
    file: str,
    line_start: int,
) -> ChunkFacts:
    """Gather deterministic facts about a detector chunk.

    Returns ``ChunkFacts()`` (all empty) when ``workspace_root`` is
    ``None`` or any underlying lookup raises — pre-compute is a
    best-effort enhancement, never a hard dependency.

    Args:
        workspace_root: Repository root the deterministic substrate
            scans. ``None`` skips pre-compute entirely.
        file: Workspace-relative path to the chunk's source file.
        line_start: First line of the chunk; we pass this to
            ``decorators_on`` to resolve the enclosing function.
    """
    if workspace_root is None or not file:
        return ChunkFacts()

    decorators: tuple[DecoratorInfo, ...] = ()
    prior_patterns: tuple[WorkspacePattern, ...] = ()

    try:
        deco_list = decorators_on(
            workspace_root, file=file, line=int(line_start or 1),
        )
        decorators = tuple(deco_list or ())
    except Exception as exc:  # noqa: BLE001 — best-effort
        log.debug(
            "bug_finder_chunk_facts_decorators_failed",
            file=file, line=line_start, error=str(exc),
        )

    try:
        all_patterns = load_workspace_patterns(workspace_root)
        prior_patterns = _filter_patterns_for_file(
            all_patterns, file=file,
        )
    except Exception as exc:  # noqa: BLE001
        log.debug(
            "bug_finder_chunk_facts_patterns_failed",
            file=file, error=str(exc),
        )

    return ChunkFacts(
        decorators=decorators,
        prior_patterns=prior_patterns,
    )


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def render_chunk_facts(facts: ChunkFacts) -> str:
    """Compose the pre-computed-facts block injected into the
    detector's user message.

    Empty input → empty string so existing prompt formatting is
    unchanged on first-contact workspaces / chunks with no facts.

    Layout: a short header naming the block as pre-computed (so the
    detector understands it didn't have to ask), followed by zero
    or more fact sections. Each section ends with a one-liner cue
    pointing the LLM at the relevant FP-killer if applicable.
    """
    if facts.is_empty:
        return ""

    lines: list[str] = ["", "## Pre-computed facts (deterministic — no tool call needed)"]

    if facts.decorators:
        lines.append("")
        lines.append(
            "**Decorator chain** on the enclosing function "
            "(outermost first):",
        )
        for deco in facts.decorators:
            args_part = (
                f"({', '.join(deco.args_repr)})"
                if deco.args_repr else ""
            )
            lines.append(f"  - `@{deco.name}{args_part}`  ({deco.file}:{deco.line})")
        lines.append(
            "  _If any of these is an auth/validation decorator "
            "(@require_auth, @rate_limit, etc.), the handler may "
            "already be guarded — factor that in before claiming a "
            "missing check._",
        )

    if facts.prior_patterns:
        lines.append("")
        lines.append(
            "**Prior patterns observed in this file** "
            "(workspace pattern memory):",
        )
        for p in facts.prior_patterns:
            note_part = (
                f" — {p.sample_claim[:80]}" if p.sample_claim else ""
            )
            status = "unresolved" if p.fix_count == 0 else f"{p.fix_count} fixes"
            lines.append(
                f"  - `{p.signature}` ×{p.hit_count} hits "
                f"({status}; severity {p.severity}){note_part}",
            )
        lines.append(
            "  _Treat these as hotspot priors, not confirmed claims. "
            "If your candidate finding matches an unresolved entry, "
            "the precision prior is high; if it doesn't, the file "
            "still has known issues._",
        )

    return "\n".join(lines) + "\n"
