"""Bug_finder-specific intelligence — pattern memory + call graph as
agent-callable query helpers.

The lead and the detector both benefit from queries the existing
substrate already answers:

* ``who_calls(name)`` — call graph reverse lookup (qualified callers).
* ``callees_of(name)`` — what does this function reach?
* ``pattern_recurrence(signature, file)`` — how often has this bug
  class shown up in this workspace's history?
* ``pattern_unresolved(workspace_id)`` — patterns that the fixer has
  never closed (likely still present).

These are zero-LLM-cost lookups against persisted state. The lead
should reach for them BEFORE asking the investigator to grep — the
intelligence module returns ground truth, the investigator returns
a model's best guess.

The functions are stateless and take their backing stores as
arguments so callers (orchestrator wires) pass the live
``CallGraph`` / ``PatternStore`` instances they already constructed.
This module is the contract; persistence stays in the existing
``call_graph.py`` / ``patterns.py`` modules.
"""

from __future__ import annotations

from dataclasses import dataclass

from augmentum.bug_finder.call_graph import CallGraph
from augmentum.bug_finder.patterns import Pattern

# ---------------------------------------------------------------------------
# Call graph queries
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CallerInfo:
    """Where one caller invokes a target. Same shape as ``CallSite``
    but exposed as a stable agent-facing dataclass."""

    caller: str             # qualified caller (e.g. "module.Class.method")
    file: str
    line: int


def who_calls(
    graph: CallGraph, target: str, *, limit: int = 20,
) -> list[CallerInfo]:
    """Return every caller of ``target`` with file/line.

    Bare-name matching — pass ``"eval"`` and you get every
    ``eval(...)`` invocation in the workspace. The graph's caller
    deduplication ensures one entry per (caller, file, line).
    """
    seen: set[tuple[str, str, int]] = set()
    out: list[CallerInfo] = []
    for site in graph.call_sites_for_target(target):
        key = (site.caller, site.file, site.line)
        if key in seen:
            continue
        seen.add(key)
        out.append(CallerInfo(
            caller=site.caller, file=site.file, line=site.line,
        ))
        if len(out) >= limit:
            break
    return out


def callees_of(graph: CallGraph, qualified_caller: str) -> list[str]:
    """What does ``qualified_caller`` call? Returns bare target names."""
    return graph.callees_of(qualified_caller)


def is_reachable_from(
    graph: CallGraph,
    *,
    source: str,
    sink: str,
    max_depth: int = 4,
) -> bool:
    """Is ``sink`` reachable from ``source`` within ``max_depth`` hops?

    BFS over the graph's forward edges. ``source`` is the qualified
    caller (``module.func``); ``sink`` is matched against the bare
    name of every callee in the chain — so ``eval`` returns True if
    any transitive callee is ``eval(...)``.

    Bare callees are resolved back to qualified nodes via name suffix
    match so the BFS can continue past the first hop. Multiple
    matching nodes are all enqueued (a bare-name match might resolve
    to several qualified definitions — we explore all of them).
    """
    sink_bare = sink.rsplit(".", 1)[-1]
    if source.rsplit(".", 1)[-1] == sink_bare:
        return True
    # Precompute bare-name → qualified-nodes index for resolution.
    by_bare: dict[str, list[str]] = {}
    for node in graph.nodes:
        by_bare.setdefault(node.rsplit(".", 1)[-1], []).append(node)
    visited: set[str] = {source}
    frontier: list[str] = [source]
    depth = 0
    while frontier and depth < max_depth:
        next_frontier: list[str] = []
        for node in frontier:
            for callee in graph.callees_of(node):
                callee_bare = callee.rsplit(".", 1)[-1]
                if callee_bare == sink_bare:
                    return True
                for resolved in by_bare.get(callee_bare, ()):
                    if resolved in visited:
                        continue
                    visited.add(resolved)
                    next_frontier.append(resolved)
        frontier = next_frontier
        depth += 1
    return False


# ---------------------------------------------------------------------------
# Pattern memory queries
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PatternRecurrence:
    """Aggregate view of one signature's history for one workspace.

    Fields:
      * ``hit_count``      — runs that flagged this pattern
      * ``fix_count``      — runs where the fixer closed it
      * ``unresolved_count`` — hits without a fix (likely still present)
      * ``last_severity``  — most recent severity observed
      * ``familiar``       — ``True`` when ``hit_count >= 2`` (the
                              same bug class has appeared at least
                              twice; deserves elevated attention)
    """

    signature: str
    hit_count: int
    fix_count: int
    unresolved_count: int
    last_severity: str
    familiar: bool


def pattern_recurrence(
    patterns: list[Pattern],
    *,
    signature: str,
    file: str = "",
) -> PatternRecurrence | None:
    """Roll the per-row ``Pattern`` history into one summary.

    Filters by ``signature`` and (optionally) ``file``. Returns
    ``None`` when no matching pattern rows exist — caller treats
    that as "first time we've seen this; no prior signal".
    """
    sig_norm = signature.strip().lower()
    file_norm = file.strip().lower()
    matched: list[Pattern] = []
    for p in patterns:
        if p.claim_signature.lower() != sig_norm:
            continue
        if file_norm and file_norm not in p.file.lower():
            continue
        matched.append(p)
    if not matched:
        return None
    total_hits = sum(p.hit_count for p in matched)
    total_fixes = sum(p.fix_count for p in matched)
    unresolved = sum(
        max(0, p.hit_count - p.fix_count) for p in matched
    )
    last_severity = max(
        matched,
        key=lambda p: p.last_seen_at,
    ).last_severity
    return PatternRecurrence(
        signature=signature,
        hit_count=total_hits,
        fix_count=total_fixes,
        unresolved_count=unresolved,
        last_severity=last_severity,
        familiar=total_hits >= 2,
    )


def unresolved_patterns(
    patterns: list[Pattern],
    *,
    min_hit_count: int = 1,
    limit: int = 20,
) -> list[Pattern]:
    """Patterns the fixer has never closed in this workspace.

    Sorted by recency. The lead consumes this as priors — an
    unresolved auth-bypass pattern from last week is much more
    interesting than a fresh first-look at random code.
    """
    candidates = [
        p for p in patterns
        if p.is_unresolved and p.hit_count >= min_hit_count
    ]
    candidates.sort(key=lambda p: p.last_seen_at, reverse=True)
    return candidates[:limit]


def signature_familiarity_score(
    patterns: list[Pattern], *, signature: str,
) -> float:
    """Score 0.0-1.0 for how 'familiar' this signature is to the
    workspace's history. 0 = never seen; 1 = seen many times.

    The detector uses this as a precision prior — familiar
    signatures get a lower bar for promotion to ``confirmed`` since
    the verifier has historically agreed with them.
    """
    sig_norm = signature.strip().lower()
    total = 0
    for p in patterns:
        if p.claim_signature.lower() == sig_norm:
            total += p.hit_count
    if total == 0:
        return 0.0
    # Diminishing-returns curve: 1 hit → 0.5, 4 hits → 0.8, 10+ hits → 1.0
    return min(1.0, 0.5 + 0.1 * (total - 1))
