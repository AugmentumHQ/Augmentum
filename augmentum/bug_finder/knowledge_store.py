"""Per-workspace structural map of the code under audit.

Bug-finder used to drop straight from workspace prep into planning. The
planner then had to (re)survey the codebase from scratch every run —
which for a large repo wastes a ton of token budget on dir_tree /
file_list / find_files that the model has already done before.

Comprehension solves it once: walk the workspace, identify subsystems
+ pillars + risk surfaces, persist the resulting map. Subsequent runs
load the brief from this store and inject it as prompt context to
every downstream subagent (planner, detector, verifier). The map
survives compaction, restart, and individual run failure — knowledge
is permanent until the user forgets it or the code drifts enough to
warrant a refresh.

The store is intentionally schema-light: ``brief`` is a markdown blob
the comprehender writes; the JSON columns hold structured shadows of
the same content for callers that want to query specific axes
(routes, pillars, etc.) without re-parsing the brief. Both are
written together as a single atomic upsert.

User-scoped per the auth pattern (same shape as PatternStore).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data shape
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Subsystem:
    """One identified subsystem within the workspace."""

    name: str                   # "narrative", "coder", "bug_finder", ...
    purpose: str                # one-sentence summary
    paths: tuple[str, ...] = ()  # source directories / glob patterns
    size_files: int = 0         # # .py / .js / .ts files under paths
    pillars: tuple[str, ...] = ()  # invariant names this subsystem owns


@dataclass(frozen=True)
class Pillar:
    """One load-bearing invariant in the codebase."""

    name: str
    statement: str              # human-readable claim
    evidence: tuple[str, ...] = ()  # "file.py:line" anchors


@dataclass(frozen=True)
class RiskSurface:
    """One untrusted-input boundary worth special attention."""

    name: str                   # "http_routes", "websocket_handlers", ...
    entry_points: tuple[str, ...] = ()    # "file:function" anchors
    trust_boundary: str = ""             # "user-supplied" / "third-party" / "fs"
    downstream_sinks: tuple[str, ...] = ()  # what risky things this can reach


@dataclass(frozen=True)
class EntryPoint:
    """Catalog of every callable entry into the system."""

    kind: str                   # "http" | "websocket" | "job" | "cli" | "mcp" | ...
    path: str                   # e.g. "POST /api/users", "media_sync_job"
    handler: str                # "file:function"


@dataclass(frozen=True)
class CodebaseKnowledge:
    """One row from ``bug_finder_codebase_knowledge``. Read-only."""

    workspace_id: str
    user_id: str
    brief: str
    subsystems: tuple[Subsystem, ...]
    pillars: tuple[Pillar, ...]
    risk_surfaces: tuple[RiskSurface, ...]
    entry_points: tuple[EntryPoint, ...]
    last_updated: int            # unix seconds; 0 = never / in-flight
    last_commit_sha: str
    refresh_count: int
    tokens_in: int
    tokens_out: int
    wallclock_seconds: float

    @property
    def is_populated(self) -> bool:
        """True when the comprehender has produced meaningful content.

        Either a markdown ``brief`` blob OR any structured data
        (subsystems / pillars / risk_surfaces / entry_points) counts.
        ``last_updated`` must be non-zero — a zero timestamp marks
        an in-flight or never-run comprehension.
        """
        if self.last_updated == 0:
            return False
        return (
            bool(self.brief.strip())
            or bool(self.subsystems)
            or bool(self.pillars)
            or bool(self.risk_surfaces)
            or bool(self.entry_points)
        )

    @property
    def age_seconds(self) -> int:
        if self.last_updated == 0:
            return 0
        return max(0, int(time.time()) - self.last_updated)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def _dump_list(items: tuple) -> str:
    return json.dumps([_as_dict(i) for i in items], ensure_ascii=False)


def _as_dict(item: Any) -> dict[str, Any]:
    """Render a dataclass instance as a JSON-safe dict.

    Tuples become lists (JSON has no tuple type). Frozen dataclasses
    expose ``__dataclass_fields__``; everything else round-trips via
    ``vars()`` which is enough for the simple shapes here.
    """
    if hasattr(item, "__dataclass_fields__"):
        out: dict[str, Any] = {}
        for f in item.__dataclass_fields__:
            v = getattr(item, f)
            out[f] = list(v) if isinstance(v, tuple) else v
        return out
    return dict(item) if isinstance(item, dict) else {"value": str(item)}


def _load_subsystems(raw: str) -> tuple[Subsystem, ...]:
    try:
        items = json.loads(raw or "[]")
    except (ValueError, TypeError):
        return ()
    return tuple(
        Subsystem(
            name=str(i.get("name") or ""),
            purpose=str(i.get("purpose") or ""),
            paths=tuple(str(p) for p in (i.get("paths") or [])),
            size_files=int(i.get("size_files") or 0),
            pillars=tuple(str(p) for p in (i.get("pillars") or [])),
        )
        for i in items if isinstance(i, dict)
    )


def _load_pillars(raw: str) -> tuple[Pillar, ...]:
    try:
        items = json.loads(raw or "[]")
    except (ValueError, TypeError):
        return ()
    return tuple(
        Pillar(
            name=str(i.get("name") or ""),
            statement=str(i.get("statement") or ""),
            evidence=tuple(str(e) for e in (i.get("evidence") or [])),
        )
        for i in items if isinstance(i, dict)
    )


def _load_risk_surfaces(raw: str) -> tuple[RiskSurface, ...]:
    try:
        items = json.loads(raw or "[]")
    except (ValueError, TypeError):
        return ()
    return tuple(
        RiskSurface(
            name=str(i.get("name") or ""),
            entry_points=tuple(str(e) for e in (i.get("entry_points") or [])),
            trust_boundary=str(i.get("trust_boundary") or ""),
            downstream_sinks=tuple(
                str(s) for s in (i.get("downstream_sinks") or [])
            ),
        )
        for i in items if isinstance(i, dict)
    )


def _load_entry_points(raw: str) -> tuple[EntryPoint, ...]:
    try:
        items = json.loads(raw or "[]")
    except (ValueError, TypeError):
        return ()
    return tuple(
        EntryPoint(
            kind=str(i.get("kind") or ""),
            path=str(i.get("path") or ""),
            handler=str(i.get("handler") or ""),
        )
        for i in items if isinstance(i, dict)
    )


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


_EMPTY_KNOWLEDGE_TEMPLATE = CodebaseKnowledge(
    workspace_id="",
    user_id="",
    brief="",
    subsystems=(),
    pillars=(),
    risk_surfaces=(),
    entry_points=(),
    last_updated=0,
    last_commit_sha="",
    refresh_count=0,
    tokens_in=0,
    tokens_out=0,
    wallclock_seconds=0.0,
)


class KnowledgeStore:
    """Read/write access to ``bug_finder_codebase_knowledge``.

    Mirrors PatternStore: aiosqlite connection in, async methods out.
    All operations are user-scoped — the caller MUST pass the
    authenticated user_id; the table's composite primary key
    (user_id, workspace_id) prevents accidental cross-user reads.
    """

    def __init__(self, conn: Any) -> None:
        self._conn = conn

    async def get(
        self, *, user_id: str = "", workspace_id: str,
    ) -> CodebaseKnowledge:
        """Return the current knowledge for one workspace.

        Returns an empty (``is_populated=False``) CodebaseKnowledge
        when no row exists yet — callers branch on
        ``knowledge.is_populated`` to decide whether to run the
        comprehender.
        """
        async with self._conn.execute(
            """
            SELECT brief, subsystems_json, pillars_json,
                   risk_surfaces_json, entry_points_json,
                   last_updated, last_commit_sha, refresh_count,
                   tokens_in, tokens_out, wallclock_seconds
            FROM bug_finder_codebase_knowledge
            WHERE user_id = ? AND workspace_id = ?
            """,
            (user_id, workspace_id),
        ) as cur:
            row = await cur.fetchone()
        if row is None:
            return CodebaseKnowledge(
                workspace_id=workspace_id, user_id=user_id,
                brief="", subsystems=(), pillars=(),
                risk_surfaces=(), entry_points=(),
                last_updated=0, last_commit_sha="", refresh_count=0,
                tokens_in=0, tokens_out=0, wallclock_seconds=0.0,
            )
        return CodebaseKnowledge(
            workspace_id=workspace_id,
            user_id=user_id,
            brief=row[0] or "",
            subsystems=_load_subsystems(row[1] or "[]"),
            pillars=_load_pillars(row[2] or "[]"),
            risk_surfaces=_load_risk_surfaces(row[3] or "[]"),
            entry_points=_load_entry_points(row[4] or "[]"),
            last_updated=int(row[5] or 0),
            last_commit_sha=str(row[6] or ""),
            refresh_count=int(row[7] or 0),
            tokens_in=int(row[8] or 0),
            tokens_out=int(row[9] or 0),
            wallclock_seconds=float(row[10] or 0.0),
        )

    async def upsert(
        self,
        *,
        user_id: str = "",
        workspace_id: str,
        brief: str,
        subsystems: tuple[Subsystem, ...] = (),
        pillars: tuple[Pillar, ...] = (),
        risk_surfaces: tuple[RiskSurface, ...] = (),
        entry_points: tuple[EntryPoint, ...] = (),
        commit_sha: str = "",
        tokens_in: int = 0,
        tokens_out: int = 0,
        wallclock_seconds: float = 0.0,
    ) -> None:
        """Write a freshly-comprehended map for one workspace.

        Bumps ``refresh_count`` and stamps ``last_updated`` to now.
        Idempotent — the primary key (user_id, workspace_id) makes
        repeat calls overwrite cleanly.
        """
        now = int(time.time())
        # Capture pre-existing refresh_count so we can bump it.
        async with self._conn.execute(
            """
            SELECT refresh_count FROM bug_finder_codebase_knowledge
            WHERE user_id = ? AND workspace_id = ?
            """,
            (user_id, workspace_id),
        ) as cur:
            prior = await cur.fetchone()
        next_refresh = (int(prior[0]) + 1) if prior else 1

        await self._conn.execute(
            """
            INSERT INTO bug_finder_codebase_knowledge (
                workspace_id, user_id, brief,
                subsystems_json, pillars_json,
                risk_surfaces_json, entry_points_json,
                last_updated, last_commit_sha, refresh_count,
                tokens_in, tokens_out, wallclock_seconds
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id, workspace_id) DO UPDATE SET
                brief             = excluded.brief,
                subsystems_json   = excluded.subsystems_json,
                pillars_json      = excluded.pillars_json,
                risk_surfaces_json = excluded.risk_surfaces_json,
                entry_points_json = excluded.entry_points_json,
                last_updated      = excluded.last_updated,
                last_commit_sha   = excluded.last_commit_sha,
                refresh_count     = excluded.refresh_count,
                tokens_in         = excluded.tokens_in,
                tokens_out        = excluded.tokens_out,
                wallclock_seconds = excluded.wallclock_seconds
            """,
            (
                workspace_id, user_id, brief,
                _dump_list(subsystems),
                _dump_list(pillars),
                _dump_list(risk_surfaces),
                _dump_list(entry_points),
                now, commit_sha, next_refresh,
                tokens_in, tokens_out, wallclock_seconds,
            ),
        )
        await self._conn.commit()
        log.info(
            "bug_finder_knowledge_persisted",
            user_id=user_id, workspace_id=workspace_id,
            subsystems=len(subsystems), pillars=len(pillars),
            refresh_count=next_refresh,
        )

    async def forget(
        self, *, user_id: str = "", workspace_id: str,
    ) -> None:
        """Delete the knowledge row, forcing re-comprehension on next run."""
        await self._conn.execute(
            """
            DELETE FROM bug_finder_codebase_knowledge
            WHERE user_id = ? AND workspace_id = ?
            """,
            (user_id, workspace_id),
        )
        await self._conn.commit()


# ---------------------------------------------------------------------------
# Prompt-friendly renderer
# ---------------------------------------------------------------------------


def render_knowledge_brief(knowledge: CodebaseKnowledge) -> str:
    """Render the comprehender's output as a system-prompt prefix block.

    Used by every downstream subagent (planner, detector, verifier,
    fixer) so they share the same understanding of the codebase. The
    ``brief`` markdown takes precedence — when populated, it's the
    authoritative narrative the comprehender already shaped. The
    structured tables underneath are appended only when ``brief``
    is empty (early-life store, or comprehender output a structured
    map without a brief).

    Returns ``""`` for an unpopulated store — callers can concat
    without an ``if``.
    """
    if not knowledge.is_populated:
        return ""

    if knowledge.brief.strip():
        return (
            "## Codebase knowledge (comprehender output — "
            f"age: {_humanize_age(knowledge.age_seconds)}, "
            f"refresh #{knowledge.refresh_count})\n\n"
            f"{knowledge.brief.strip()}"
        )

    # Fallback rendering from the structured shadows when no brief blob
    # exists. Compact tabular shape so token cost stays bounded.
    parts: list[str] = ["## Codebase knowledge (structural map)\n"]
    if knowledge.subsystems:
        parts.append("### Subsystems")
        for s in knowledge.subsystems[:20]:
            parts.append(f"- **{s.name}** — {s.purpose}")
            if s.paths:
                parts.append(f"  Paths: {', '.join(s.paths[:5])}")
    if knowledge.pillars:
        parts.append("\n### Pillars (load-bearing invariants)")
        for p in knowledge.pillars[:20]:
            parts.append(f"- **{p.name}**: {p.statement}")
    if knowledge.risk_surfaces:
        parts.append("\n### Risk surfaces (untrusted-input boundaries)")
        for r in knowledge.risk_surfaces[:15]:
            parts.append(f"- **{r.name}** ({r.trust_boundary})")
    return "\n".join(parts)


def _humanize_age(seconds: int) -> str:
    if seconds < 3600:
        return f"{max(1, seconds // 60)}m"
    if seconds < 86400:
        return f"{seconds // 3600}h"
    return f"{seconds // 86400}d"
