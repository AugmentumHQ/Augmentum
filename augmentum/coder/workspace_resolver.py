"""Workspace resolver — map a companion "go build X" request onto one of the
user's EXISTING coder workspaces, or surface a pickable set when ambiguous.

Never auto-picks silently (CLAUDE.md rule #2): a confident single winner is
returned so the caller can ANNOUNCE which workspace it chose; anything
ambiguous returns candidates for a tap-or-say choice (the companion-candidates
dock). New-workspace CREATION is deliberately NOT done here — the caller routes
that to the Coder create UI so the user still picks the template/repo (auto-
picking a template would be the same auto-select regression one level down).

Scoring is intentionally cheap: workspace NAME + repo-slug token overlap with
the request, with recency as a tiebreak only. It does not exec into containers
to list files (per-candidate `docker exec` would blow the interactive budget);
a file-content signal can be layered on later behind the same interface.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Words that carry no workspace-identifying signal in a build request — the
# verbs and glue of "build me a dark-mode toggle in my ui project".
_STOPWORDS = frozenset({
    "build", "make", "create", "add", "implement", "fix", "update", "change",
    "the", "a", "an", "in", "on", "to", "for", "my", "me", "please", "some",
    "new", "into", "with", "and", "of", "app", "page", "feature", "bug", "code",
    "project", "workspace", "repo", "this", "that", "it", "can", "you", "could",
    "would", "set", "up", "setup", "write", "refactor", "test", "go", "let",
    "lets", "want", "need", "just", "then", "over", "there",
})

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Confidence gates. A single owned workspace that matches at all — or a clear
# margin over the runner-up — is "confident"; otherwise offer the picks.
_CONFIDENT_MIN = 0.34
_CONFIDENT_MARGIN = 0.20
_OFFER_CAP = 3


def _tokens(text: str) -> set[str]:
    return {
        t for t in _TOKEN_RE.findall((text or "").lower())
        if t not in _STOPWORDS and len(t) > 1
    }


def _repo_slug(git_url: str) -> str:
    """Last path segment of a git url, minus .git — 'user/foo.git' → 'foo'."""
    tail = (git_url or "").rstrip("/").rsplit("/", 1)[-1]
    return tail[:-4] if tail.endswith(".git") else tail


@dataclass
class WorkspaceCandidate:
    workspace_id: str
    title: str
    subtitle: str = ""
    score: float = 0.0
    is_new: bool = False

    def to_payload(self) -> dict[str, Any]:
        # ``kind``/``content_kind`` = coder_workspace so companion-candidates.js
        # renders the workspace card branch (no cover). ``workspace_id`` is the
        # accept id the offer parks under (see coder.delegate + the router's
        # generic offered-candidate resolution).
        return {
            "kind": "coder_workspace",
            "content_kind": "coder_workspace",
            "workspace_id": self.workspace_id,
            "title": self.title,
            "subtitle": self.subtitle,
            "is_new": self.is_new,
        }


@dataclass
class ResolveResult:
    # "confident" — dispatch to ``top`` (announce it). "offer" — present
    # ``candidates``. "none" — the user has no workspaces; caller offers a
    # new-only card.
    decision: str
    top: WorkspaceCandidate | None = None
    candidates: list[WorkspaceCandidate] = field(default_factory=list)


def _conn(app_state: Any) -> Any:
    return getattr(
        getattr(
            getattr(app_state, "state_manager", None),
            "backend", None,
        ),
        "conn", None,
    )


async def _owned_workspaces(app_state: Any, user_id: str) -> list[Any]:
    """The user's own workspaces (manager list ∩ project_checkouts ownership).

    Mirrors ``coder_routes.list_workspaces`` so the resolver can only ever
    return workspaces the user owns — no cross-tenant leak.
    """
    mgr = getattr(app_state, "container_manager", None)
    if mgr is None:
        return []
    try:
        all_ws = await mgr.list_workspaces()
    except Exception:
        log.warning("workspace_resolver_list_failed", exc_info=True)
        return []
    conn = _conn(app_state)
    if conn is None:
        return []
    try:
        cursor = await conn.execute(
            "SELECT id FROM project_checkouts WHERE user_id = ?", (user_id,),
        )
        owned = {row[0] for row in await cursor.fetchall()}
    except Exception:
        log.warning("workspace_resolver_owned_query_failed", exc_info=True)
        return []
    # Regular coding workspaces only — a bug_finder workspace isn't a build
    # target for a "go build X" delegation.
    return [
        w for w in all_ws
        if w.id in owned and (getattr(w, "kind", "regular") or "regular") == "regular"
    ]


async def resolve_workspace(
    app_state: Any, *, user_id: str, request_text: str,
) -> ResolveResult:
    """Score the user's workspaces against a build request.

    Returns a ``ResolveResult`` — never raises, never auto-creates.
    """
    workspaces = await _owned_workspaces(app_state, user_id)
    if not workspaces:
        return ResolveResult(decision="none")

    req = _tokens(request_text)
    now = time.time()

    scored: list[WorkspaceCandidate] = []
    for w in workspaces:
        name = getattr(w, "name", "") or w.id
        name_toks = _tokens(name) | _tokens(_repo_slug(getattr(w, "git_url", "") or ""))
        if req and name_toks:
            # Overlap coefficient (Szymkiewicz–Simpson): matched tokens over
            # the SMALLER set. A request naming most of a workspace's name
            # ("dark mode to the ui" → "augmentum-ui") scores high without
            # being penalised for the request's extra words, and a bare
            # "the api" fully identifies the lone "billing-api" workspace.
            score = len(req & name_toks) / min(len(req), len(name_toks))
        else:
            score = 0.0
        last_active = float(getattr(w, "last_active", 0.0) or 0.0)
        subtitle = _recency_label(now - last_active) if last_active else ""
        scored.append(WorkspaceCandidate(
            workspace_id=w.id, title=name, subtitle=subtitle,
            score=round(score, 4),
        ))

    # Name score first, recency as the tiebreak (most-recent wins a score tie).
    scored.sort(
        key=lambda c: (c.score, _recency_key(c.subtitle)), reverse=True,
    )
    top = scored[0]
    second_score = scored[1].score if len(scored) > 1 else 0.0

    confident = top.score >= _CONFIDENT_MIN and (
        len(scored) == 1 or (top.score - second_score) >= _CONFIDENT_MARGIN
    )
    if confident:
        return ResolveResult(decision="confident", top=top, candidates=[top])

    return ResolveResult(decision="offer", top=top, candidates=scored[:_OFFER_CAP])


# ── recency helpers (label for the card, coarse key for tiebreaks) ──────────

def _recency_label(age_s: float) -> str:
    if age_s < 0:
        return ""
    if age_s < 3600:
        return "active in the last hour"
    if age_s < 86400:
        h = int(age_s // 3600)
        return f"last active {h}h ago"
    d = int(age_s // 86400)
    return f"last active {d}d ago"


# The subtitle already encodes recency; derive a coarse sort key back out of it
# so the tiebreak needs no second field on the candidate. Fresher → larger.
def _recency_key(subtitle: str) -> int:
    if "last hour" in subtitle:
        return 3
    if "h ago" in subtitle:
        return 2
    if "d ago" in subtitle:
        return 1
    return 0
