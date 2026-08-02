"""File-mirrored read/write surface for the coder inspector panel.

The inspector exposes two editable artifacts the model also reads each
turn: the objective pin (``/workspace/.augmentum/objective.md``) and the
observation ledger (``/workspace/.augmentum/observations.jsonl``).

Both files are written by other code paths too — the workspace kernel
auto-seeds the objective from the first substantive user message
([[project_coder_objective_pin]]), the ``observe`` tool appends to the
ledger ([[project_coder_observation_ledger]]). So inspector edits and
model writes both target the same files.

Read-modify-write on JSONL needs serialization or a concurrent observe
append can truncate a delete. We hold an asyncio.Lock per workspace —
keyed in a module-level dict — so PATCH/DELETE from the inspector and
the model's append path never interleave. The lock is best-effort
(single-process; the broker runs in-proc) which matches the rest of
the coder mode's concurrency model.

The objective file mirror is simpler: a single PUT replaces the whole
file, with an ``if_mtime_unchanged`` optimistic-concurrency check so a
model write during the user's edit surfaces as a 409 instead of a
silent overwrite.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import TYPE_CHECKING

from augmentum.coder.observations import (
    CATEGORIES,
    CONFIDENCES,
    Observation,
    _LEDGER_SOFT_CAP,
    append_observation,
    parse_jsonl,
    read_ledger,
    serialize_observations,
)
from augmentum.coder.workspace_kernel import KERNEL_ROOT, OBJECTIVE_MD
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.coder.containers import ContainerManager

log = get_logger(__name__)


# Per-workspace lock registry. Single-process: the run broker and all
# request handlers live in the same proxy process, so a dict guarded by
# a module-level lock is sufficient. We never evict — workspace count
# stays in the dozens even for power users.
_WORKSPACE_LOCKS: dict[str, asyncio.Lock] = {}
_REGISTRY_LOCK = asyncio.Lock()


# Limits — duplicated from the spec, validated in the route as well.
OBJECTIVE_MAX_BYTES = 2048
OBSERVATION_FACT_MAX_BYTES = 1024


class ObjectiveConflictError(Exception):
    """Raised when ``if_mtime_unchanged`` doesn't match current mtime."""

    def __init__(self, current_mtime: float, current_content: str) -> None:
        self.current_mtime = current_mtime
        self.current_content = current_content
        super().__init__("objective file modified since last read")


class ObservationNotFoundError(Exception):
    """Raised when an idx-based observation lookup misses."""


async def _get_workspace_lock(workspace_id: str) -> asyncio.Lock:
    """Return or create the per-workspace asyncio.Lock."""
    async with _REGISTRY_LOCK:
        lock = _WORKSPACE_LOCKS.get(workspace_id)
        if lock is None:
            lock = asyncio.Lock()
            _WORKSPACE_LOCKS[workspace_id] = lock
        return lock


# ---------------------------------------------------------------------------
# Objective
# ---------------------------------------------------------------------------


async def _stat_objective(
    cm: "ContainerManager", workspace_id: str,
) -> float:
    """Best-effort mtime read. Returns 0.0 on miss/error."""
    try:
        out = await cm._run_command(
            workspace_id,
            ["bash", "-c", f"stat -c %Y {OBJECTIVE_MD} 2>/dev/null || echo 0"],
            timeout=3.0,
        )
    except Exception:
        return 0.0
    text = (out or "").strip() if isinstance(out, str) else ""
    if not text:
        # _run_command sometimes returns a tuple/dict — defensive parse
        try:
            text = str(out).strip()
        except Exception:
            return 0.0
    try:
        return float(text.splitlines()[-1])
    except (ValueError, IndexError):
        return 0.0


async def read_objective(
    cm: "ContainerManager", workspace_id: str,
) -> dict:
    """Read ``objective.md`` + current mtime.

    Returns ``{"content": str, "mtime": float, "seeded": bool}``.
    ``seeded`` is True when the file exists; we don't differentiate
    auto-seeded vs user-written content because there's no marker in
    the file format to detect that reliably.

    Existence is checked explicitly (``test -f``) before reading so a
    missing file produces an empty string instead of leaking the
    ``cat: ... No such file or directory`` stderr text into the
    inspector panel.
    """
    content = ""
    mtime = 0.0
    try:
        # Single command: check + cat. The `printf` trailing newline
        # disambiguates "empty file" from "missing file" — the stat
        # call below confirms the latter via mtime=0.
        raw = await cm._run_command(
            workspace_id,
            ["bash", "-c", f"if [ -f {OBJECTIVE_MD} ]; then cat {OBJECTIVE_MD}; fi"],
            timeout=5.0,
        )
        if isinstance(raw, str):
            content = raw
    except Exception as exc:
        log.debug(
            "coder_inspector_objective_read_failed",
            workspace_id=workspace_id, error=str(exc)[:160],
        )
        content = ""
    if content:
        mtime = await _stat_objective(cm, workspace_id)
    return {
        "content": content,
        "mtime": mtime,
        "seeded": bool(content),
    }


async def write_objective(
    cm: "ContainerManager",
    workspace_id: str,
    *,
    content: str,
    if_mtime_unchanged: float | None = None,
) -> dict:
    """Replace ``objective.md`` with ``content``.

    When ``if_mtime_unchanged`` is provided and the current mtime differs,
    raises :class:`ObjectiveConflictError` carrying the live content.
    Caller maps that to a 409.

    Returns ``{"content": str, "mtime": float}`` after the write.
    """
    body = (content or "").strip()
    if not body:
        raise ValueError("objective content cannot be empty")
    encoded = body.encode("utf-8")
    if len(encoded) > OBJECTIVE_MAX_BYTES:
        raise ValueError(
            f"objective exceeds {OBJECTIVE_MAX_BYTES} bytes "
            f"({len(encoded)} bytes provided)",
        )
    if "\x00" in body:
        raise ValueError("objective must not contain null bytes")

    lock = await _get_workspace_lock(workspace_id)
    async with lock:
        if if_mtime_unchanged is not None:
            current_mtime = await _stat_objective(cm, workspace_id)
            # Tolerate small clock drift — the model writes via the same
            # container so mtimes match exactly when nothing changed.
            if current_mtime > 0 and abs(current_mtime - if_mtime_unchanged) > 0.5:
                current = await cm.file_read(workspace_id, OBJECTIVE_MD)
                raise ObjectiveConflictError(current_mtime, current or "")

        # Ensure the kernel dir exists. mkdir -p is idempotent. Best-
        # effort: file_write below may still succeed if the dir already
        # exists; log so a chronic mkdir failure surfaces.
        try:
            await cm._run_command(
                workspace_id,
                ["bash", "-c", f"mkdir -p {KERNEL_ROOT}"],
                timeout=3.0,
            )
        except Exception as exc:
            log.debug(
                "coder_inspector_mkdir_failed",
                workspace_id=workspace_id, error=str(exc)[:160],
            )

        # Always trail with a newline to keep the file POSIX-friendly.
        payload = body if body.endswith("\n") else body + "\n"
        await cm.file_write(workspace_id, OBJECTIVE_MD, payload)

    new_mtime = await _stat_objective(cm, workspace_id)
    log.info(
        "coder_inspector_objective_written",
        workspace_id=workspace_id,
        bytes=len(encoded),
        mtime=new_mtime,
    )
    return {"content": body, "mtime": new_mtime}


# ---------------------------------------------------------------------------
# Observations
# ---------------------------------------------------------------------------


def _obs_to_dict(obs: Observation, idx: int) -> dict:
    return {
        "idx": idx,
        "ts": obs.ts,
        "category": obs.category,
        "fact": obs.fact,
        "source": obs.source,
        "confidence": obs.confidence,
    }


async def list_observations(
    cm: "ContainerManager",
    workspace_id: str,
    *,
    categories: list[str] | None = None,
    limit: int = 200,
    offset: int = 0,
) -> dict:
    """Return paginated + optionally category-filtered observations.

    Most recent first (by ``ts`` descending). ``idx`` in the output is
    the position in the on-disk file (zero-based) so PATCH/DELETE callers
    address rows stably even when the visible list is sorted.
    """
    ledger = await read_ledger(cm, workspace_id)
    # idx is the file-order position; we sort for display but preserve
    # the original idx so edits/deletes stay row-addressable.
    indexed = list(enumerate(ledger))
    if categories:
        cat_set = {c.strip() for c in categories if c.strip()}
        if cat_set:
            indexed = [(i, o) for i, o in indexed if o.category in cat_set]
    indexed.sort(key=lambda pair: pair[1].ts, reverse=True)
    total = len(indexed)
    page = indexed[offset : offset + max(0, min(limit, 500))]
    return {
        "total": total,
        "items": [_obs_to_dict(o, idx=i) for i, o in page],
        "soft_cap": _LEDGER_SOFT_CAP,
        "near_cap": total > int(_LEDGER_SOFT_CAP * 0.8),
    }


def _validate_observation_input(
    *,
    category: str,
    fact: str,
    confidence: str,
) -> tuple[str, str, str]:
    """Normalize + validate user-supplied observation fields."""
    cat = (category or "").strip()
    if cat not in CATEGORIES:
        cat = "other"
    fact_stripped = (fact or "").strip()
    if not fact_stripped:
        raise ValueError("observation fact cannot be empty")
    if len(fact_stripped.encode("utf-8")) > OBSERVATION_FACT_MAX_BYTES:
        raise ValueError(
            f"observation fact exceeds {OBSERVATION_FACT_MAX_BYTES} bytes",
        )
    conf = (confidence or "").strip()
    if conf not in CONFIDENCES:
        conf = "user_asserted"
    return cat, fact_stripped, conf


async def create_observation(
    cm: "ContainerManager",
    workspace_id: str,
    *,
    user_id: str,
    category: str,
    fact: str,
    confidence: str = "user_asserted",
) -> dict:
    """Append a user-edited observation to the ledger.

    Source is stamped ``user-edit:{user_id}`` so the kernel + future
    render paths can treat user-asserted entries with higher priority
    than tool-observed ones.
    """
    cat, fact_clean, conf = _validate_observation_input(
        category=category, fact=fact, confidence=confidence,
    )
    obs = Observation(
        ts=time.time(),
        category=cat,
        fact=fact_clean,
        source=f"user-edit:{user_id or 'unknown'}",
        confidence=conf,
    )
    lock = await _get_workspace_lock(workspace_id)
    async with lock:
        ok = await append_observation(cm, workspace_id, obs)
    if not ok:
        raise RuntimeError("failed to persist observation")
    # Read back to discover the file-order idx (append_observation merges
    # dedups, so the new row may have replaced an older same-fact row).
    ledger = await read_ledger(cm, workspace_id)
    for i, existing in enumerate(ledger):
        if (
            existing.category == obs.category
            and existing.fact == obs.fact
            and abs(existing.ts - obs.ts) < 1.0
        ):
            return _obs_to_dict(existing, idx=i)
    # Couldn't pin it (race with another writer); return the input shape.
    return _obs_to_dict(obs, idx=len(ledger) - 1 if ledger else 0)


async def update_observation(
    cm: "ContainerManager",
    workspace_id: str,
    idx: int,
    *,
    user_id: str,
    category: str | None = None,
    fact: str | None = None,
    confidence: str | None = None,
) -> dict:
    """Rewrite the ``idx``-th observation row in-place.

    PATCH semantics — unspecified fields stay as-is. Source is rewritten
    to ``user-edit:{user_id}`` because the user has now taken ownership
    of this row.
    """
    lock = await _get_workspace_lock(workspace_id)
    async with lock:
        ledger = await read_ledger(cm, workspace_id)
        if idx < 0 or idx >= len(ledger):
            raise ObservationNotFoundError(f"idx {idx} out of range")
        existing = ledger[idx]
        new_category = category if category is not None else existing.category
        new_fact = fact if fact is not None else existing.fact
        new_confidence = (
            confidence if confidence is not None else existing.confidence
        )
        cat_clean, fact_clean, conf_clean = _validate_observation_input(
            category=new_category, fact=new_fact, confidence=new_confidence,
        )
        ledger[idx] = Observation(
            ts=existing.ts,
            category=cat_clean,
            fact=fact_clean,
            source=f"user-edit:{user_id or 'unknown'}",
            confidence=conf_clean,
        )
        await cm.file_write(
            workspace_id,
            "/workspace/.augmentum/observations.jsonl",
            serialize_observations(ledger),
        )
    log.info(
        "coder_inspector_observation_updated",
        workspace_id=workspace_id,
        idx=idx,
        category=cat_clean,
    )
    return _obs_to_dict(ledger[idx], idx=idx)


async def delete_observation(
    cm: "ContainerManager",
    workspace_id: str,
    idx: int,
) -> bool:
    """Remove the ``idx``-th observation row.

    Returns True on delete, raises :class:`ObservationNotFoundError`
    when ``idx`` is out of range (after relocking + re-reading).
    """
    lock = await _get_workspace_lock(workspace_id)
    async with lock:
        ledger = await read_ledger(cm, workspace_id)
        if idx < 0 or idx >= len(ledger):
            raise ObservationNotFoundError(f"idx {idx} out of range")
        removed = ledger.pop(idx)
        await cm.file_write(
            workspace_id,
            "/workspace/.augmentum/observations.jsonl",
            serialize_observations(ledger),
        )
    log.info(
        "coder_inspector_observation_deleted",
        workspace_id=workspace_id,
        idx=idx,
        category=removed.category,
    )
    return True


# ---------------------------------------------------------------------------
# Inspector state aggregator (read-only)
# ---------------------------------------------------------------------------


async def aggregate_session_costs(
    conn,
    *,
    user_id: str,
    workspace_id: str,
    session_id: str = "",
) -> dict:
    """Sum cost columns from ``coder_turn_runs`` for the current session.

    When ``session_id`` is empty, sums across the whole workspace
    (used as a fallback for sessions that never created a turn run).
    """
    # `project_id` post-migration 200 (was `workspace_id`). The Python
    # kwarg name stays workspace_id to limit caller churn — same
    # value, since checkout id == project id for legacy backfilled rows.
    where_clauses = ["user_id = ?", "project_id = ?"]
    params: list = [user_id, workspace_id]
    if session_id:
        where_clauses.append("session_id = ?")
        params.append(session_id)
    sql = (
        "SELECT "
        " COALESCE(SUM(input_cost_usd), 0.0), "
        " COALESCE(SUM(output_cost_usd), 0.0), "
        " COUNT(*) "
        "FROM coder_turn_runs "
        f"WHERE {' AND '.join(where_clauses)}"
    )
    try:
        cursor = await conn.execute(sql, params)
        row = await cursor.fetchone()
    except Exception:
        return {"input_usd": 0.0, "output_usd": 0.0, "turn_count": 0, "by_model": []}

    input_usd = float(row[0] or 0.0)
    output_usd = float(row[1] or 0.0)
    turn_count = int(row[2] or 0)

    # Per-model breakdown so the panel can show "qwen3 (local) + claude-opus".
    try:
        cursor = await conn.execute(
            "SELECT COALESCE(cost_model_id, model) AS model_id, "
            " COALESCE(SUM(input_cost_usd), 0.0), "
            " COALESCE(SUM(output_cost_usd), 0.0), "
            " COUNT(*) "
            "FROM coder_turn_runs "
            f"WHERE {' AND '.join(where_clauses)} "
            "GROUP BY model_id "
            "ORDER BY (SUM(input_cost_usd) + SUM(output_cost_usd)) DESC, model_id",
            params,
        )
        rows = await cursor.fetchall()
    except Exception:
        rows = []
    by_model = [
        {
            "model": (r[0] or "")[:120],
            "input_usd": float(r[1] or 0.0),
            "output_usd": float(r[2] or 0.0),
            "turns": int(r[3] or 0),
        }
        for r in rows
    ]

    return {
        "input_usd": input_usd,
        "output_usd": output_usd,
        "turn_count": turn_count,
        "by_model": by_model,
    }
