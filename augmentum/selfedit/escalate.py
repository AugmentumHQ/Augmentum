"""Model-tier escalation for self-edit — local does the groundwork, a frontier
model closes the hard ones, the context is carried up so nothing is repeated.

A grow-with-the-user system must be able to make real frontend/code changes, and
a small local model often can't land the harder ones alone (it explores, confirms,
then runs out of budget or nerve). The answer isn't an easier task — it's an
ESCALATION LADDER:

* the SAME sovereign harness throughout (``engine="native"`` — Augmentum's own
  agentic loop); only the MODEL changes rung to rung;
* a cheap LOCAL model does the groundwork (reads, searches, confirmations);
* on failure we climb to a stronger model — up to a FRONTIER model (e.g. a
  DeepSeek), reached through the very same native loop via the registry — with the
  weaker tier's findings CARRIED FORWARD as managed context, so the stronger model
  doesn't repeat the legwork (no redundant, expensive calls);
* EVERY rung is archived (``run_self_edit`` records each attempt permanently) —
  that's the accountability: you can see what each model tried and why it failed.

We stop at the first rung that lands a *gated* edit (one that actually changed
files AND passed verification). The frontier rung is cost-gated (``frontier=True``
rungs only run when ``allow_frontier``), so the expensive model is opt-in.

This module pulls in ``engine_select`` (→ ``coder.external``), so — like
``edit_driver`` — it is NOT exported from ``augmentum.selfedit.__init__``; import
it where you wire the loop.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

from augmentum.coder.external import run_store
from augmentum.selfedit.engine_select import DEFAULT_ENGINE, build_selfedit_driver
from augmentum.selfedit.live import emit_progress
from augmentum.selfedit.orchestrator import EditDriver, SelfEditOutcome, run_self_edit
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# A self-edit "succeeds" (stop climbing) when it rests at gated — an edit that
# changed files and passed verification. Everything else (rejected no-op,
# rejected-on-verify-fail, failed) means the rung couldn't do it → escalate.
_STOP_STATUS = "gated"
_HANDOFF_CHARS = 2600


@dataclass
class RungSpec:
    """One requested rung of the ladder (cheapest first)."""
    model: str = ""             # model name; "" = the engine's role default
    label: str = ""             # display label (defaults to model/engine)
    engine: str = DEFAULT_ENGINE  # "native" (default) | "claude_code" | "codex"
    frontier: bool = False      # cost-gated — only runs when allow_frontier


@dataclass
class LadderRung:
    label: str
    driver: EditDriver
    frontier: bool = False


async def summarize_run_for_handoff(
    conn: Any, *, run_id: str, user_id: str, label: str, status: str, lesson: str,
    diff: str = "", max_chars: int = _HANDOFF_CHARS,
) -> str:
    """Condense a failed rung's run into a brief the next (stronger) rung reads, so
    it builds on the groundwork instead of re-deriving it. Carries the agent's own
    narration (persisted ``message`` events), the tool/file trail, and — crucially
    — the prior rung's ACTUAL committed ``diff`` so the stronger model repairs real
    code, not just prose.

    The notes came from a rung that FAILED, in a worktree the next rung hasn't
    seen, so the guidance is trust-but-verify (not "edit blind on these notes")."""
    notes: list[str] = []
    tools: list[str] = []
    files: list[str] = []
    with contextlib.suppress(Exception):
        run = await run_store.get_run(conn, run_id=run_id, user_id=user_id)
        for e in (run or {}).get("events", []):
            kind = e.get("kind", "")
            if kind == "message" and (e.get("text") or "").strip():
                notes.append(e["text"].strip())
            elif e.get("tool"):
                tools.append(e["tool"])
                if e.get("path"):
                    files.append(e["path"])
    head = (f"PRIOR ATTEMPT (model: {label}) did NOT finish the job "
            f"({status}: {lesson.strip()}).")
    guidance = ("Use its findings below to save time — but you're in a FRESH "
                "checkout and these came from a model that did NOT succeed, so "
                "VERIFY anything you're about to depend on (re-check the file/symbol "
                "before you edit) rather than trusting the notes blindly.")
    parts = [head, guidance]
    if notes:
        # keep the most recent reasoning (richest — it's where it got to)
        kept, total = [], 0
        for n in reversed(notes):
            if total + len(n) > max_chars:
                break
            kept.append(n)
            total += len(n)
        parts.append("Its notes:\n" + "\n".join(f"- {n}" for n in reversed(kept)))
    if tools:
        parts.append("Tools it ran: " + ", ".join(tools[:40]))
    if files:
        uniq = list(dict.fromkeys(files))[:20]
        parts.append("Files it touched/inspected: " + ", ".join(uniq))
    brief = "\n\n".join(parts)[:max_chars]
    if diff.strip():
        # Append the real diff AFTER the cap so it's never truncated away — this is
        # the cheap path: repair/improve an actual patch instead of starting over.
        brief += ("\n\nThe prior attempt's ACTUAL changes (they did NOT pass "
                  "verification — reason above). Reproduce the correct parts, fix "
                  "what failed, and DISCARD anything inappropriate (e.g. stray "
                  "helper files it created):\n```diff\n" + diff.strip() + "\n```")
    return brief


async def build_ladder(
    conn: Any, registry: Any, specs: list[RungSpec], *,
    allow_frontier: bool = False, oauth_token: str = "", api_key: str = "",
    cwd: str = "/workspace",
) -> list[LadderRung]:
    """Build the runnable ladder from requested rung specs (cheapest first).

    A frontier rung is skipped unless ``allow_frontier`` (so the expensive model
    is opt-in). A rung whose driver can't be built here (model/engine unavailable)
    is dropped with a log line, never a crash — the ladder degrades gracefully."""
    rungs: list[LadderRung] = []
    for s in specs:
        if s.frontier and not allow_frontier:
            log.info("selfedit_ladder_frontier_skipped", model=s.model, label=s.label)
            continue
        driver = await build_selfedit_driver(
            conn=conn, engine=s.engine, model=s.model, registry=registry,
            oauth_token=oauth_token, api_key=api_key, cwd=cwd, native_role="utility",
        )
        if driver is None:
            log.info("selfedit_ladder_rung_unavailable", engine=s.engine, model=s.model)
            continue
        rungs.append(LadderRung(label=(s.label or s.model or s.engine),
                                driver=driver, frontier=s.frontier))
    return rungs


async def run_self_edit_escalating(
    *, repo_dir: str, objective: str, user_id: str, conn: Any,
    rungs: list[LadderRung], start_index: int = 0, **run_kwargs: Any,
) -> SelfEditOutcome | None:
    """Climb the ladder until a rung lands a gated edit, carrying each failed
    rung's findings forward to the next (stronger) one. Returns the winning
    outcome, or the LAST attempt if none succeed (every rung is archived either
    way). Returns None only if the ladder is empty.

    ``start_index`` is the verified-skill-graph routing hint (read-only): a
    confidently failure-prone region skips the doomed cheap rung(s) and starts
    higher, instead of always-cheap-first. Clamped so at least the top rung always
    runs — it reorders which model tries first, never changes what gets promoted."""
    prior = ""
    last: SelfEditOutcome | None = None
    total = len(rungs)
    start_index = max(0, min(start_index, total - 1)) if total else 0
    for i, rung in enumerate(rungs):
        if i < start_index:
            log.info("selfedit_escalation_rung_skipped", rung=rung.label, index=i,
                     reason="skill_graph_failure_prone_region")
            emit_progress({
                "kind": "rung", "state": "skipped", "index": i, "total": total,
                "model": rung.label,
                "text": (f"Skipping rung {i + 1}/{total} ({rung.label}) — the skill "
                         "graph marks this region failure-prone; starting higher."),
            })
            continue
        emit_progress({
            "kind": "rung", "state": "start", "index": i, "total": total,
            "model": rung.label, "frontier": rung.frontier,
            "carried": bool(prior),
            "text": (f"Rung {i + 1}/{total}: {rung.label}"
                     + (" (building on the prior rung's findings)" if prior else "")),
        })
        outcome = await run_self_edit(
            repo_dir=repo_dir, objective=objective, user_id=user_id, conn=conn,
            driver=rung.driver, prior_context=prior, **run_kwargs,
        )
        last = outcome
        log.info("selfedit_escalation_rung", rung=rung.label, tier=i,
                 status=outcome.status, frontier=rung.frontier,
                 files=len(outcome.files_changed))
        emit_progress({
            "kind": "rung", "state": "done", "index": i, "total": total,
            "model": rung.label, "frontier": rung.frontier,
            "status": outcome.status,
            "tier": (outcome.verdict.tier if outcome.verdict else ""),
            "files": len(outcome.files_changed),
        })
        if outcome.status == _STOP_STATUS:
            log.info("selfedit_escalation_landed", rung=rung.label, tier=i,
                     verdict=(outcome.verdict.tier if outcome.verdict else ""))
            emit_progress({"kind": "rung", "state": "landed", "index": i,
                           "model": rung.label,
                           "text": f"{rung.label} landed a verified-eligible edit"})
            return outcome
        # carry this rung's groundwork (notes + the ACTUAL diff) up to the next,
        # stronger rung. Replace rather than accumulate: the latest rung's brief is
        # the richest and carries its own diff (older rungs already informed it).
        run_id = outcome.edit.run_id if outcome.edit else ""
        if i + 1 < len(rungs):
            prior = await summarize_run_for_handoff(
                conn, run_id=run_id, user_id=user_id, label=rung.label,
                status=outcome.status, lesson=outcome.lesson,
                diff=getattr(outcome, "diff", "") or "",
            )
    return last
