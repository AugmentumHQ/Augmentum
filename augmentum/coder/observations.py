"""Observation ledger — durable facts the agent learns while working.

Observations are Layer 5 of the context-kernel state model. Distinct
from identity (Layer 1) by *ownership* and *lifetime*: identity is
auto-detected project facts + user-curated assertions; observations
are things the AGENT discovered during work and wants to remember.

The persistent ledger lives at ``/workspace/.augmentum/observations.jsonl``:

* **Append-only** so concurrent writes (rare but possible — background
  jobs + the active turn) can never corrupt earlier entries.
* **JSONL** so each entry is a self-contained line; partial reads
  recover everything before a corrupted entry; categorical queries are
  cheap (filter on a single field).
* **Cross-session** — survives container restart, repo clones, and
  conversation `/clear`. The second session in a repo starts with
  the first session's learnings instead of re-discovering them.

Categories (closed enum, chosen so a 100-line ledger still partitions
cleanly): ``build``, ``test``, ``deploy``, ``api``, ``data``, ``env``,
``constraint``, ``gotcha``, ``style``, ``other``. Categories matter
because the render path (Step 4 of the kernel migration) filters by
category to surface "constraints" with higher priority than "other"
when prompt budget is tight.

This module exposes the primitives. The model interacts via the
``observe`` tool (in runtime_tools.py) which delegates here. The
kernel exposes wrapper methods so callers can stay decoupled from
the on-disk format.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.coder.containers import ContainerManager

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Schema — categories closed, confidences closed, fact + source free-text.
# Closed sets keep query paths predictable; free-text fields stay flexible.
# ---------------------------------------------------------------------------


CATEGORIES: frozenset[str] = frozenset({
    "build",       # how to build / what the build command is / build artifacts
    "test",        # test runner, common failing tests, fixture conventions
    "deploy",      # deploy target, how to ship, infra particulars
    "api",         # endpoints, schemas, auth shapes
    "data",        # database location, schema notes, migration setup
    "env",         # environment variables, secrets locations, paths
    "constraint",  # user-imposed limits ("node 18 locked")
    "gotcha",      # surprises, "this dependency conflicts with that"
    "style",       # code style decisions, naming conventions
    "other",       # catch-all
})


CONFIDENCES: frozenset[str] = frozenset({
    "tentative",       # model inferred but didn't verify
    "confirmed",       # model verified via a tool result
    "user_asserted",   # user said this is true; treat as canonical
})


# Cap on ledger size before we warn. Past this the model's "query
# recent N" path still works fine, but the file gets unwieldy for
# human inspection. Soft cap; no auto-trim — let the user prune.
_LEDGER_SOFT_CAP = 500


@dataclass(slots=True)
class Observation:
    """One durable fact about the workspace.

    Schema is intentionally narrow — every field has a clear purpose.
    Adding fields requires a version bump (and a migration for the
    on-disk ledger). Free-form metadata goes in ``fact``.
    """
    ts: float                 # epoch seconds when recorded
    category: str             # one of CATEGORIES
    fact: str                 # the durable claim
    source: str               # provenance — "observe tool turn 9", etc.
    confidence: str = "confirmed"

    def to_jsonl_line(self) -> str:
        """Serialize as a single JSON line (no trailing newline)."""
        return json.dumps(asdict(self), default=str, separators=(",", ":"))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Observation | None:
        """Parse from a dict. Returns None on malformed input.

        Tolerant: unknown fields are ignored; required fields default
        to safe values. Categories / confidences are validated against
        the closed sets — invalid values rewrite to ``"other"`` /
        ``"confirmed"`` so a typo doesn't lose the observation.
        """
        try:
            category = str(data.get("category", "other"))
            if category not in CATEGORIES:
                category = "other"
            confidence = str(data.get("confidence", "confirmed"))
            if confidence not in CONFIDENCES:
                confidence = "confirmed"
            return cls(
                ts=float(data.get("ts", 0.0)),
                category=category,
                fact=str(data.get("fact", "")),
                source=str(data.get("source", "")),
                confidence=confidence,
            )
        except (TypeError, ValueError):
            return None


# ---------------------------------------------------------------------------
# Parse / serialize a whole ledger
# ---------------------------------------------------------------------------


def parse_jsonl(text: str) -> list[Observation]:
    """Parse JSONL text into observations.

    Malformed lines are skipped with a debug log — one bad line
    doesn't take out the whole ledger. Empty input returns ``[]``.
    """
    if not text:
        return []
    observations: list[Observation] = []
    for line_num, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError:
            log.debug("observations.malformed_line", line_num=line_num)
            continue
        if not isinstance(data, dict):
            continue
        obs = Observation.from_dict(data)
        if obs is not None:
            observations.append(obs)
    return observations


def serialize_observations(observations: list[Observation]) -> str:
    """Render a full ledger as JSONL text. Always ends with newline."""
    if not observations:
        return ""
    return "\n".join(o.to_jsonl_line() for o in observations) + "\n"


# ---------------------------------------------------------------------------
# Deduplication — exact-fact-match within the same category is treated
# as a refresh (replace older with newer, keep latest confidence).
#
# This is the minimal-viable conflict handling for v1. A future
# iteration could add subject-extraction-based conflict detection
# ("pytest is the test runner" vs "unittest is the test runner") but
# that needs an NLP-shaped heuristic that's worth its own design pass.
# ---------------------------------------------------------------------------


def _dedup_key(obs: Observation) -> tuple[str, str]:
    """The key by which two observations are considered "the same fact"."""
    return (obs.category, obs.fact.strip().lower())


def merge_observation(
    existing: list[Observation],
    new_obs: Observation,
) -> list[Observation]:
    """Return a new list with ``new_obs`` appended, deduplicating by
    (category, fact). If a prior observation has the same key, it's
    removed and the new one is appended at the end — preserving the
    "latest write wins" semantic AND the chronological ordering of
    the ledger as a whole.
    """
    key = _dedup_key(new_obs)
    filtered = [o for o in existing if _dedup_key(o) != key]
    filtered.append(new_obs)
    return filtered


# ---------------------------------------------------------------------------
# Query — most filtering happens at render time, but a few common
# shapes are useful enough to wrap.
# ---------------------------------------------------------------------------


def query_observations(
    observations: list[Observation],
    *,
    categories: list[str] | None = None,
    min_confidence: str | None = None,
    limit: int = 50,
) -> list[Observation]:
    """Filter + return most-recent-N observations.

    ``categories``: restrict to entries in this set (case-sensitive).
    ``min_confidence``: ``"confirmed"`` excludes ``"tentative"``;
    ``"user_asserted"`` excludes both. None = no filter.
    ``limit``: cap on returned count, applied AFTER filtering, by ts
    descending (most recent first).
    """
    filtered = observations
    if categories:
        cat_set = set(categories)
        filtered = [o for o in filtered if o.category in cat_set]
    if min_confidence is not None:
        order = ["tentative", "confirmed", "user_asserted"]
        try:
            min_idx = order.index(min_confidence)
        except ValueError:
            min_idx = 0
        filtered = [
            o for o in filtered
            if order.index(o.confidence) >= min_idx
            if o.confidence in order
        ]
    # Sort by ts descending so callers get most-recent first by default.
    filtered = sorted(filtered, key=lambda o: o.ts, reverse=True)
    return filtered[:limit]


# ---------------------------------------------------------------------------
# Container IO — read whole ledger; rewrite whole ledger on append.
#
# We don't have a file_append primitive on ContainerManager. Read-
# modify-write is fine at observation rates (a few per turn at most)
# and avoids a Python subprocess. The soft cap warning helps catch
# ledgers that get pathologically large before that assumption
# breaks down.
# ---------------------------------------------------------------------------


async def read_ledger(
    cm: ContainerManager,
    workspace_id: str,
    *,
    path: str = "/workspace/.augmentum/observations.jsonl",
) -> list[Observation]:
    """Read the full ledger from disk. ``[]`` on miss/error."""
    try:
        raw = await cm.file_read(workspace_id, path)
    except Exception:
        return []
    return parse_jsonl(raw or "")


async def append_observation(
    cm: ContainerManager,
    workspace_id: str,
    observation: Observation,
    *,
    path: str = "/workspace/.augmentum/observations.jsonl",
) -> bool:
    """Append + persist with dedup-by-(category, fact).

    Returns True on success, False on any failure. Best-effort — the
    caller can react to False (e.g. surface to the model "your
    observation didn't persist") or ignore it. The ledger is one of
    several layers of memory; a single missed write doesn't break the
    agent.
    """
    if cm is None or not workspace_id:
        return False
    try:
        existing = await read_ledger(cm, workspace_id, path=path)
        merged = merge_observation(existing, observation)
        if len(merged) > _LEDGER_SOFT_CAP:
            log.warning(
                "observations.soft_cap_exceeded",
                workspace_id=workspace_id,
                count=len(merged),
                cap=_LEDGER_SOFT_CAP,
            )
        # Make sure the parent directory exists. mkdir -p is idempotent
        # so this is safe to do on every append.
        try:
            await cm._run_command(
                workspace_id,
                ["bash", "-c", "mkdir -p /workspace/.augmentum"],
                timeout=3.0,
            )
        except Exception:
            log.debug("observations.mkdir_kernel_root_failed", workspace_id=workspace_id, exc_info=True)
        await cm.file_write(workspace_id, path, serialize_observations(merged))
        return True
    except Exception:
        log.debug(
            "observations.append_failed",
            workspace_id=workspace_id,
            category=observation.category,
            exc_info=True,
        )
        return False


# ---------------------------------------------------------------------------
# Rendering — compact + budget-aware. Used by future layer-render path
# (Step 4 of the kernel migration) when surfacing observations inline.
# Lives here rather than in a strategy module so the format stays
# consistent across native / hybrid / canonical.
# ---------------------------------------------------------------------------


def render_for_prompt(
    observations: list[Observation],
    *,
    budget_chars: int = 600,
    priority_categories: tuple[str, ...] = ("constraint", "gotcha"),
) -> str:
    """Format observations for inclusion in a system prompt.

    Renders most-recent constraints + gotchas first (those are
    load-bearing for action decisions), then any other recent
    observations until the character budget is exhausted. Empty
    observations or zero budget → empty string.

    The output is human-readable so the user can inspect the rendered
    block in chat dev tools, AND parseable by eye if needed.
    """
    if not observations or budget_chars <= 0:
        return ""

    # Priority queue: priority categories first, then others, both by
    # ts desc. Within a category, latest wins.
    priority = query_observations(
        observations, categories=list(priority_categories),
    )
    rest = [o for o in observations if o.category not in priority_categories]
    rest = sorted(rest, key=lambda o: o.ts, reverse=True)

    lines: list[str] = ["<observations>"]
    used = len("<observations>\n</observations>")
    for obs in priority + rest:
        # 1-line render: [category] fact — keep it tight so the
        # budget covers more facts.
        line = f"  [{obs.category}] {obs.fact}"
        # +1 for the newline we'll join with.
        if used + len(line) + 1 > budget_chars:
            break
        lines.append(line)
        used += len(line) + 1
    lines.append("</observations>")
    return "\n".join(lines)
