"""Training trace capture — records complete tool chains from live requests.

Gated by ``training_capture_enabled`` (default off). When active, appends
one JSONL line per completed chat turn to ``data/training_traces/{date}.jsonl``
containing the full message chain including tool calls and results.

The traces are raw material — the base model's response and thinking, the
real tool calls and real results. The personality layer (think blocks and
spoken response) is written over this skeleton by the curator or by
multi-provider fan-out.

Hook points: the ``finally`` blocks in streaming.py (3 streaming paths)
and the non-streaming paths in openai_routes.py / ollama_routes.py.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from augmentum.prompts.primer import tag_for
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.models.base import InternalChatRequest

log = get_logger(__name__)

# Mode/surface → training tag resolution is the canonical map in
# augmentum/prompts/primer.py (SURFACE_TAGS / tag_for). This module used to
# carry a partial DUPLICATE that silently relabeled every unknown mode
# (builder/game/stream/system/knowledge/voice/phone/xr/cast) as ":C" — a
# dataset-poisoning bug. The raw ``mode`` string is still stored on every
# trace, so the tag stays reversible if the canonical map ever changes.

# Synthesis prompt fragments injected by the tool loop — strip these
# from training traces since they're harness internals.
_SYNTH_MARKERS = (
    "Synthesize the tool",
    "synthesize the above",
    "Synthesize the results",
    "Now respond to the user",
    "Based on the tool results",
    "Use the information above",
    "Use the tool results above",
    "Incorporate the tool results",
    "Do NOT repeat the raw tool output",
)


def _is_synthesis_message(msg_content: str) -> bool:
    """Return True if the message is a tool-loop synthesis prompt."""
    if not msg_content:
        return False
    head = msg_content[:120]
    return any(marker.lower() in head.lower() for marker in _SYNTH_MARKERS)


def _serialize_message(msg: object) -> dict | None:
    """Convert a Message dataclass into a clean dict for the trace."""
    role = getattr(msg, "role", None)
    content = getattr(msg, "content", None)
    if role is None:
        return None

    # Skip synthesis prompts injected by the tool loop
    if role == "user" and content and _is_synthesis_message(content):
        return None

    entry: dict = {"role": role, "content": content or ""}

    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        cleaned = []
        for tc in tool_calls:
            if isinstance(tc, dict):
                func = tc.get("function", {})
                name = func.get("name", "")
                args_raw = func.get("arguments", "{}")
                try:
                    args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                except (json.JSONDecodeError, TypeError):
                    args = args_raw
                cleaned.append({
                    "id": tc.get("id", ""),
                    "name": name,
                    "arguments": args,
                })
        if cleaned:
            entry["tool_calls"] = cleaned

    thinking = getattr(msg, "thinking", None)
    if thinking:
        entry["thinking"] = thinking

    tool_call_id = getattr(msg, "tool_call_id", None)
    if tool_call_id:
        entry["tool_call_id"] = tool_call_id
        # Add the tool name for readability in traces
        entry["name"] = _tool_name_for_call_id(tool_call_id, msg)

    return entry


def _tool_name_for_call_id(tool_call_id: str, msg: object) -> str:
    """Try to extract the tool name from a tool-result message's content."""
    content = getattr(msg, "content", "") or ""
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict) and "name" in parsed:
            return parsed["name"]
    except (json.JSONDecodeError, TypeError):
        pass
    return ""


def _extract_tools_used(chain: list[dict]) -> list[str]:
    """Extract unique tool names from the chain in call order."""
    seen: set[str] = set()
    tools: list[str] = []
    for entry in chain:
        for tc in entry.get("tool_calls", []):
            name = tc.get("name", "")
            if name and name not in seen:
                seen.add(name)
                tools.append(name)
    return tools


def capture_training_trace(
    request: InternalChatRequest,
    final_response: str,
    session_id: str,
    user_id: str,
    mode: str,
) -> None:
    """Capture a complete message chain as a training trace.

    Called from the streaming/non-streaming finally blocks after the tool
    loop has mutated ``request.messages`` with the full chain.

    Writes one JSONL line to ``data/training_traces/{date}.jsonl``.
    """
    from augmentum.config import settings

    enabled = settings.training_capture_enabled
    capture_user = settings.training_capture_user_id

    # The settings singleton may not reflect DB values (set via setup script
    # before the server process started). Check once and cache on the singleton.
    if not enabled and not getattr(settings, "_training_capture_db_checked", False):
        object.__setattr__(settings, "_training_capture_db_checked", True)
        try:
            import os
            import sqlite3

            from augmentum.config import settings as _s
            # capture_training_trace is a SYNC function invoked from the
            # streaming/non-streaming finally blocks — i.e. inside a running
            # event loop. aiosqlite + run_until_complete is illegal there
            # (RuntimeError: cannot be called from a running event loop), so
            # this one-shot, cached read uses the stdlib sync driver. It runs
            # at most once per process (guarded by _training_capture_db_checked)
            # against a tiny key-value table, so the brief sync read is fine.
            db_path = getattr(_s, "_db_path", "") or "/data/augmentum.db"
            db_vals: dict[str, str] = {}
            if os.path.exists(db_path):
                conn = sqlite3.connect(db_path)
                try:
                    rows = conn.execute(
                        "SELECT key, value FROM app_settings WHERE key LIKE 'training_capture%'"
                    ).fetchall()
                finally:
                    conn.close()
                db_vals = {r[0]: r[1] for r in rows}
            if db_vals.get("training_capture_enabled", "").lower() == "true":
                object.__setattr__(settings, "training_capture_enabled", True)
                enabled = True
            uid = db_vals.get("training_capture_user_id", "")
            if uid:
                object.__setattr__(settings, "training_capture_user_id", uid)
                capture_user = uid
            log.info("training_capture_loaded_from_db", enabled=enabled, user_id=capture_user)
        except Exception:
            log.warning("training_capture_db_check_failed", exc_info=True)

    if not enabled:
        return

    if capture_user and user_id != capture_user:
        return

    # Skip trivially short responses
    if len(final_response.strip()) < settings.training_capture_min_content:
        return

    try:
        _write_trace(request, final_response, session_id, user_id, mode)
    except Exception:
        log.warning("training_capture_failed", exc_info=True)


def _write_trace(
    request: InternalChatRequest,
    final_response: str,
    session_id: str,
    user_id: str,
    mode: str,
) -> None:
    """Build and persist the trace."""
    from augmentum.config import settings

    tag = tag_for(mode)

    # Build the cleaned message chain
    chain: list[dict] = []
    system_prompt = ""

    for msg in request.messages:
        role = getattr(msg, "role", None)

        # Capture system prompt separately
        if role == "system":
            content = getattr(msg, "content", "") or ""
            if content and not system_prompt:
                system_prompt = content
            continue

        entry = _serialize_message(msg)
        if entry is not None:
            chain.append(entry)

    if not chain:
        return

    tools_used = _extract_tools_used(chain)
    tool_call_count = sum(
        len(e.get("tool_calls", [])) for e in chain
    )

    # Count chain depth (number of tool loop iterations)
    chain_depth = sum(
        1 for e in chain
        if e["role"] == "assistant" and e.get("tool_calls")
    )

    now = datetime.now(UTC)
    trace_id = f"tr_{int(now.timestamp())}_{hashlib.sha256(f'{session_id}{now.isoformat()}'.encode()).hexdigest()[:8]}"

    trace = {
        "trace_id": trace_id,
        "timestamp": now.isoformat(),
        "session_id": session_id,
        "user_id": user_id,
        "mode": mode,
        "tag": tag,
        "model": request.model or "",
        "chain": chain,
        "system_prompt_hash": hashlib.sha256(
            system_prompt.encode()
        ).hexdigest()[:16] if system_prompt else "",
        "tools_used": tools_used,
        "tool_call_count": tool_call_count,
        "chain_depth": chain_depth,
        "final_response": final_response,
        "error": False,
    }

    # Write to date-partitioned JSONL
    trace_dir = Path(settings.training_capture_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)

    trace_file = trace_dir / f"{now.strftime('%Y-%m-%d')}.jsonl"
    with open(trace_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(trace, ensure_ascii=False, default=str) + "\n")

    log.info(
        "training_trace_captured",
        trace_id=trace_id,
        tag=tag,
        tools=tools_used,
        chain_depth=chain_depth,
    )


# --- Harness harvest staging ----------------------------------------------
# Harness turns (OpenCode / Claude Code / …) DO NOT mutate live memory. Instead
# each turn's extracted harvest CANDIDATES — and, later, the companion's
# observations of them — are STAGED here as observation-only records you can
# review: "what's worth harvesting" + the trends, before a deliberate pass
# filters the good ones into the baseline. Folds into the training-capture area
# (same dir, same harvest surface) so there's ONE place the baseline grows from.
# Nothing here is recalled or injected anywhere — review-and-promote only.


def _harness_harvest_dir() -> Path:
    from augmentum.config import settings
    return Path(settings.training_capture_dir) / "harness_harvest"


def capture_harness_observation(
    *,
    user_id: str,
    session_id: str = "",
    harness: str = "",
    model: str = "",
    source_message: str = "",
    candidates: list[dict] | None = None,
    observations: list[dict] | None = None,
) -> str | None:
    """Stage a harness harvest-candidate record. Observation-only — never
    written to live memory. Returns the obs_id, or None if disabled/empty/failed.
    Gated by ``harness_capture_enabled`` (which now means "stage", not "write")."""
    from augmentum.config import settings

    if not getattr(settings, "harness_capture_enabled", True):
        return None
    candidates = candidates or []
    observations = observations or []
    if not user_id or (not candidates and not observations):
        return None
    try:
        return _write_harness_observation(
            user_id=user_id, session_id=session_id, harness=harness, model=model,
            source_message=source_message, candidates=candidates,
            observations=observations,
        )
    except Exception:
        log.warning("harness_observation_capture_failed", exc_info=True)
        return None


def _write_harness_observation(
    *, user_id: str, session_id: str, harness: str, model: str,
    source_message: str, candidates: list[dict], observations: list[dict],
) -> str:
    now = datetime.now(UTC)
    obs_id = (
        f"hh_{int(now.timestamp())}_"
        + hashlib.sha256(f"{user_id}{session_id}{now.isoformat()}".encode()).hexdigest()[:8]
    )
    record = {
        "obs_id": obs_id,
        "kind": "harness_harvest_candidate",
        "timestamp": now.isoformat(),
        "user_id": user_id,
        "session_id": session_id,
        "harness": harness,
        "model": model,
        "source_message": source_message[:2000],
        "candidates": candidates,
        "observations": observations,
        "harvested": False,
        "harvested_at": None,
    }
    d = _harness_harvest_dir()
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{now.strftime('%Y-%m-%d')}.jsonl"
    with open(f, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    log.info(
        "harness_observation_staged", obs_id=obs_id, harness=harness,
        candidates=len(candidates), observations=len(observations),
    )
    return obs_id


# --- Harvest decisions ledger ---------------------------------------------
# The staging JSONL is append-only, so a candidate's promote/dismiss VERDICT is
# recorded in a sidecar ledger rather than by rewriting the record. A candidate
# is keyed by ``{obs_id}:{idx}``; the last decision wins. ``promote`` carries the
# baseline memory id it created so the trail is auditable.


def _harvest_ledger_path() -> Path:
    return _harness_harvest_dir() / "decisions.jsonl"


def _load_harvest_ledger() -> dict[str, dict]:
    p = _harvest_ledger_path()
    if not p.exists():
        return {}
    ledger: dict[str, dict] = {}
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except Exception:
                continue
            key = f"{e.get('obs_id')}:{e.get('idx')}"
            ledger[key] = e  # last decision wins
    except Exception:
        log.warning("harvest_ledger_read_failed", exc_info=True)
    return ledger


def record_harvest_decision(
    *, obs_id: str, idx: int, action: str, user_id: str, baseline_id: str = "",
) -> None:
    """Append a promote/dismiss verdict for a staged candidate. Best-effort."""
    now = datetime.now(UTC)
    entry = {
        "obs_id": obs_id, "idx": idx, "action": action, "user_id": user_id,
        "baseline_id": baseline_id, "ts": now.isoformat(),
    }
    p = _harvest_ledger_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        log.warning("harvest_ledger_write_failed", exc_info=True)


def _annotate_record(rec: dict, ledger: dict[str, dict]) -> dict:
    """Attach each candidate's decision status + compute record-level harvested."""
    obs_id = rec.get("obs_id")
    pending = 0
    for i, c in enumerate(rec.get("candidates", [])):
        e = ledger.get(f"{obs_id}:{i}")
        c["status"] = e.get("action") if e else "pending"
        c["baseline_id"] = e.get("baseline_id", "") if e else ""
        if c["status"] == "pending":
            pending += 1
    rec["pending_count"] = pending
    rec["harvested"] = (pending == 0 and bool(rec.get("candidates")))
    return rec


def get_harness_record(user_id: str, obs_id: str) -> dict | None:
    """Fetch a single staged record by obs_id (scoped to the user). Used by the
    promote/dismiss endpoints to read the candidate text/kind."""
    d = _harness_harvest_dir()
    if not d.exists():
        return None
    for f in sorted(d.glob("*.jsonl"), reverse=True):
        # decisions.jsonl holds promote/dismiss ledger rows that ALSO carry the
        # obs_id but have no candidates — skip it (as read_harness_harvest does)
        # or we'd return a decision row instead of the staging record whenever a
        # prior decision for this obs_id exists (breaks a 2nd promote/dismiss).
        if f.name == "decisions.jsonl":
            continue
        try:
            lines = f.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        for line in lines:
            line = line.strip()
            if not line or obs_id not in line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if rec.get("obs_id") == obs_id and (not user_id or rec.get("user_id") == user_id):
                return rec
    return None


def read_harness_harvest(
    *, user_id: str = "", limit: int = 100, include_harvested: bool = False,
) -> list[dict]:
    """Read staged harvest candidates, most recent first, annotated with each
    candidate's promote/dismiss status. Best-effort; [] if nothing staged."""
    d = _harness_harvest_dir()
    if not d.exists():
        return []
    ledger = _load_harvest_ledger()
    out: list[dict] = []
    for f in sorted(d.glob("*.jsonl"), reverse=True):  # newest day first
        if f.name == "decisions.jsonl":
            continue
        try:
            lines = f.read_text(encoding="utf-8").splitlines()
        except Exception:
            continue
        for line in reversed(lines):  # newest record first within the day
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if user_id and rec.get("user_id") != user_id:
                continue
            rec = _annotate_record(rec, ledger)
            if rec["harvested"] and not include_harvested:
                continue
            out.append(rec)
            if len(out) >= limit:
                return out
    return out


def harness_harvest_trends(*, user_id: str = "") -> dict:
    """Aggregate staged candidates into simple trends for review — what's piling
    up by kind/harness, how much is durable, how much is still pending vs
    promoted vs dismissed."""
    recs = read_harness_harvest(user_id=user_id, limit=5000, include_harvested=True)
    by_kind: dict[str, int] = {}
    by_harness: dict[str, int] = {}
    durable = total_candidates = 0
    pending = promoted = dismissed = 0
    for r in recs:
        h = r.get("harness") or "?"
        by_harness[h] = by_harness.get(h, 0) + 1
        for c in r.get("candidates", []):
            total_candidates += 1
            k = c.get("kind") or "fact"
            by_kind[k] = by_kind.get(k, 0) + 1
            if c.get("durable"):
                durable += 1
            status = c.get("status", "pending")
            if status == "promote":
                promoted += 1
            elif status == "dismiss":
                dismissed += 1
            else:
                pending += 1
    return {
        "records": len(recs),
        "candidates": total_candidates,
        "pending": pending,
        "promoted": promoted,
        "dismissed": dismissed,
        "durable_candidates": durable,
        "by_kind": by_kind,
        "by_harness": by_harness,
    }
