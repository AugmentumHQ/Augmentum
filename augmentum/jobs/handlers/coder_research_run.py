"""``coder_research_run`` job handler — autonomous improvement loop
(autoresearch-style) against a coder workspace.

"Point the agent at a workspace with intentions + a measurable objective
and walk away": the loop is propose → implement (one headless coder turn
through the EXACT same stack an interactive send uses) → measure (run the
user's objective command in the container) → keep if the score improved,
revert if not → repeat, budget permitting. Progress is physically visible
in the workspace itself:

  - ``RESEARCH_LOG.md`` at the workspace root grows one entry per
    experiment (git-ignored so reverts never eat it),
  - every KEPT experiment is a git checkpoint (``research exp N: …``)
    in the normal git panel,
  - each experiment turn is a normal broker run — watchable live via
    the standard ``/api/coder/runs/{id}/stream`` reattach path,
  - a ``coder.run.complete`` / ``coder.run.failed`` notification lands
    at the end with the baseline→best summary.

Payload shape (validated loudly at the top):

    {
        "workspace_id":      "ws_...",       // existing coder workspace
        "model":             "qwen3.6-35b",  // chosen by the USER at
                                             // queue time (never auto)
        "objective_command": "python bench.py",  // prints the score:
                                             // last "SCORE: <num>" line,
                                             // else last bare-number line
        "direction":         "minimize",     // or "maximize" — REQUIRED,
                                             // never assumed
        "intentions":        "...",          // optional; falls back to
                                             // /workspace/OBJECTIVES.md
        "experiments":       5,              // loop budget
        "objective_timeout": 300.0,          // seconds per measurement
        "turn_max_seconds":  1200.0,         // ceiling per agent turn
        "min_delta":         0.0,            // required improvement margin
        "coder_strategy":    ""              // optional override
    }

Design notes / invariants:

- The measurement is AUTHORITATIVE and ours: the agent may run tests or
  the objective itself while experimenting, but keep/revert is decided
  only by the handler's own ``run_command`` of the objective, after the
  candidate is committed. A confident agent claiming "improved!" changes
  nothing.
- Keep/revert rides the existing checkpoint machinery:
  ``git_checkpoint`` (add -A + commit) for candidates,
  ``git_revert`` (read-tree --reset, non-destructive) back to the last
  good sha on regression/parse-failure/timeout. History is preserved —
  the git log IS the lab notebook's raw form.
- Each experiment turn gets a STANDALONE prompt (intentions + objective
  + running experiment history), not the workspace conversation — the
  history table is the loop's memory, and it prevents the agent from
  re-proposing what already failed.
- Serialization: same contract as ``coder_background_run`` — the busy
  poll is politeness; ``WorkspaceBusyError`` from the broker is the
  real guarantee, handled per-experiment (bounded retries, then a
  graceful early finish — never a whole-mission requeue that would
  forget loop state).
"""

from __future__ import annotations

import asyncio
import contextlib
import re
import time
from typing import Any

from augmentum.coder.run_broker import WorkspaceBusyError
from augmentum.jobs.context import JobCancelled, JobContext
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

JOB_TYPE = "coder_research_run"

_OBJECTIVES_FILE = "/workspace/OBJECTIVES.md"
_LOG_FILE = "/workspace/RESEARCH_LOG.md"

# Budgets — payload may override within these rails.
_EXPERIMENTS_DEFAULT = 5
_EXPERIMENTS_MAX = 50
_OBJECTIVE_TIMEOUT_DEFAULT_S = 300.0
_OBJECTIVE_TIMEOUT_MAX_S = 1800.0
_TURN_MAX_DEFAULT_S = 1200.0
_TURN_MAX_CEILING_S = 3600.0

# Busy-workspace handling per experiment (an interactive turn can grab
# the workspace between our turns — that's the user's right of way).
_BUSY_RETRIES = 3
_BUSY_WAIT_S = 30.0

# Last "SCORE: <num>" wins; fallback is the last line that IS a number.
_SCORE_RE = re.compile(r"SCORE:\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")
_BARE_NUM_RE = re.compile(r"^-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")


class _EarlyFinish(Exception):
    """Internal control flow — the mission ends gracefully with partial
    results (e.g. the workspace stayed interactively busy)."""


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested in tests/test_coder_research_run.py)
# ---------------------------------------------------------------------------


def parse_score(output: str) -> float | None:
    """Extract the objective score from command output.

    Prefers the LAST ``SCORE: <num>`` occurrence (explicit contract);
    otherwise the last output line that is a bare number. None when
    neither exists — callers must treat that as a failed experiment,
    never as zero.
    """
    matches = _SCORE_RE.findall(output or "")
    if matches:
        try:
            return float(matches[-1])
        except ValueError:  # pragma: no cover — regex guarantees float
            return None
    for line in reversed((output or "").strip().splitlines()):
        line = line.strip()
        if _BARE_NUM_RE.fullmatch(line):
            try:
                return float(line)
            except ValueError:
                return None
    return None


def is_improvement(
    candidate: float, best: float, direction: str, min_delta: float = 0.0,
) -> bool:
    """Strict improvement in the user's chosen direction."""
    if direction == "minimize":
        return candidate < best - min_delta
    return candidate > best + min_delta


def extract_experiment_summary(final_text: str) -> str:
    """The agent is asked to end with ``EXPERIMENT: <one line>``; fall
    back to the first non-empty line of its answer."""
    for line in (final_text or "").splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("EXPERIMENT:"):
            return stripped[len("EXPERIMENT:"):].strip() or "(unnamed change)"
    for line in (final_text or "").splitlines():
        if line.strip():
            return line.strip()[:200]
    return "(no summary)"


def render_history(experiments: list[dict[str, Any]]) -> str:
    """Compact experiment table fed back into every prompt so the agent
    never re-proposes a failed idea. All entries are included — each is
    a one-line record we authored, so there is nothing to truncate."""
    if not experiments:
        return "(none yet — this is the first experiment)"
    lines = []
    for e in experiments:
        score = f"{e['score']:g}" if e.get("score") is not None else "unmeasurable"
        lines.append(
            f"{e['index']}. [{e['verdict']}] {e['summary']} — score: {score}"
        )
    return "\n".join(lines)


def build_experiment_prompt(
    *,
    index: int,
    total: int,
    intentions: str,
    objective_command: str,
    direction: str,
    baseline: float,
    best: float,
    experiments: list[dict[str, Any]],
) -> str:
    goal_word = "LOWER" if direction == "minimize" else "HIGHER"
    return (
        f"You are running experiment {index}/{total} of an autonomous "
        "improvement loop on this workspace.\n\n"
        "## Mission intentions (why this workspace exists, what better means)\n"
        f"{intentions}\n\n"
        "## Objective\n"
        f"The measurable objective is the command below — {goal_word} is "
        "better. After your change, the harness runs it itself and KEEPS "
        "your change only if the score strictly improves; otherwise every "
        "file change is reverted.\n"
        f"```\n{objective_command}\n```\n"
        f"Baseline score: {baseline:g} · Best so far: {best:g}\n\n"
        "## Experiment history (do NOT repeat failed ideas)\n"
        f"{render_history(experiments)}\n\n"
        "## Your task\n"
        "Propose and implement exactly ONE focused change that you expect "
        "to improve the objective. Rules:\n"
        "- One idea per experiment. Small and measurable beats sweeping.\n"
        "- You MAY run the objective command or tests yourself to sanity-"
        "check, but the harness's own measurement is authoritative.\n"
        "- Never modify RESEARCH_LOG.md or OBJECTIVES.md — those belong "
        "to the harness.\n"
        "- Never use git yourself (no commit/reset/checkout) — the "
        "harness owns version control for this loop.\n"
        "- End your reply with a single line:\n"
        "  EXPERIMENT: <one-line description of the change you made>"
    )


# ---------------------------------------------------------------------------
# Handler
# ---------------------------------------------------------------------------


async def _notify(
    app: Any,
    *,
    user_id: str,
    ok: bool,
    title: str,
    body: str,
    payload: dict[str, Any],
    dedupe_key: str,
) -> None:
    """Terminal notification on the pre-provisioned coder.run.* channels.
    Best-effort — a notify failure must never fail a finished mission."""
    try:
        sm = getattr(app.state, "state_manager", None)
        conn = getattr(getattr(sm, "backend", None), "conn", None) if sm else None
        if conn is None:
            return
        hub = getattr(app.state, "notification_hub", None)
        if hub is None:
            from augmentum.notifications.hub import NotificationHub
            hub = NotificationHub()
            app.state.notification_hub = hub
        from augmentum.notifications.hub import publish_and_dispatch
        from augmentum.notifications.store import NotificationAction
        await publish_and_dispatch(
            conn,
            hub=hub,
            user_id=user_id,
            channel_id="coder.run.complete" if ok else "coder.run.failed",
            source=JOB_TYPE,
            title=title,
            body=body,
            payload=payload,
            dedupe_key=dedupe_key,
            actions=[
                NotificationAction(
                    id="open", label="Open workspace", style="primary",
                ),
            ] if payload.get("workspace_id") else None,
        )
    except Exception:
        log.warning("coder_research_run_notify_failed", exc_info=True)


def make_coder_research_run_handler(app):
    """Factory bound to runtime app services (mirrors the background-run
    handler; lookups against app.state happen at job-run time)."""

    async def handler(ctx: JobContext) -> dict | None:
        payload = ctx.payload or {}
        workspace_id = str(payload.get("workspace_id") or "").strip()
        model = str(payload.get("model") or "").strip()
        objective_command = str(payload.get("objective_command") or "").strip()
        direction = str(payload.get("direction") or "").strip().lower()
        intentions = str(payload.get("intentions") or "").strip()
        coder_strategy = str(payload.get("coder_strategy") or "").strip()
        if not workspace_id or not model or not objective_command:
            raise ValueError(
                "coder_research_run payload requires workspace_id, model, "
                "objective_command",
            )
        if direction not in ("minimize", "maximize"):
            # The user states what better means — never assumed.
            raise ValueError("direction must be 'minimize' or 'maximize'")
        user_id = ctx.user_id
        if not user_id:
            raise ValueError("coder_research_run requires a user-scoped job")

        def _num(key: str, default: float, lo: float, hi: float) -> float:
            try:
                v = float(payload.get(key) or default)
            except (TypeError, ValueError):
                v = default
            return max(lo, min(hi, v))

        n_experiments = int(_num("experiments", _EXPERIMENTS_DEFAULT, 1, _EXPERIMENTS_MAX))
        objective_timeout = _num(
            "objective_timeout", _OBJECTIVE_TIMEOUT_DEFAULT_S, 5.0,
            _OBJECTIVE_TIMEOUT_MAX_S,
        )
        turn_max_s = _num(
            "turn_max_seconds", _TURN_MAX_DEFAULT_S, 60.0, _TURN_MAX_CEILING_S,
        )
        min_delta = max(0.0, _num("min_delta", 0.0, 0.0, float("inf")))

        state = app.state
        containers = getattr(state, "container_manager", None)
        if containers is None:
            raise RuntimeError("container manager unavailable")
        broker = getattr(state, "coder_run_broker", None)

        # ── Intentions: payload wins, OBJECTIVES.md is the fallback ────
        if not intentions:
            try:
                intentions = (await containers.file_read(
                    workspace_id, _OBJECTIVES_FILE,
                )).strip()
            except Exception:
                intentions = ""
        if not intentions:
            raise ValueError(
                "no intentions given — pass 'intentions' in the request or "
                f"create {_OBJECTIVES_FILE} in the workspace",
            )

        # ── Workspace setup ────────────────────────────────────────────
        # Keep the lab notebook out of git so keep/revert never eats it.
        try:
            gitignore = await containers.file_read(
                workspace_id, "/workspace/.gitignore",
            )
        except Exception:
            gitignore = ""
        if "RESEARCH_LOG.md" not in gitignore:
            await containers.file_write(
                workspace_id, "/workspace/.gitignore",
                (gitignore.rstrip("\n") + "\nRESEARCH_LOG.md\n").lstrip("\n"),
            )

        # Commit any dirty pre-mission state so reverts have a floor.
        await containers.git_checkpoint(workspace_id, "research: baseline")
        last_good = (await containers.run_command(
            workspace_id,
            ["bash", "-c",
             "cd /workspace && git rev-parse --verify --short HEAD 2>/dev/null || true"],
            timeout=10.0,
        )).strip()
        if not re.fullmatch(r"[0-9a-f]{4,40}", last_good):
            raise RuntimeError(
                "workspace has no git history — cannot run a keep/revert "
                "loop (open the workspace once so it initializes, or commit "
                "something first)",
            )

        async def _measure() -> tuple[float | None, str]:
            try:
                out = await containers.run_command(
                    workspace_id, ["bash", "-lc", objective_command],
                    timeout=objective_timeout,
                )
            except Exception as exc:
                return None, f"objective command failed: {exc}"
            score = parse_score(out)
            if score is None:
                tail = "\n".join((out or "").strip().splitlines()[-5:])
                return None, f"no score in output (tail):\n{tail}"
            return score, ""

        log_parts: list[str] = []

        async def _append_log(text: str) -> None:
            """Append to RESEARCH_LOG.md — the physically-visible notebook."""
            log_parts.append(text)
            try:
                existing = ""
                with contextlib.suppress(Exception):
                    existing = await containers.file_read(workspace_id, _LOG_FILE)
                await containers.file_write(
                    workspace_id, _LOG_FILE,
                    (existing.rstrip("\n") + "\n\n" if existing.strip() else "")
                    + text.strip() + "\n",
                )
            except Exception:
                log.warning(
                    "coder_research_log_write_failed",
                    workspace_id=workspace_id, exc_info=True,
                )

        # ── Baseline measurement (loud failure — no baseline, no loop) ─
        await ctx.update_progress(0.0, stage="measuring baseline")
        baseline, err = await _measure()
        if baseline is None:
            raise RuntimeError(
                f"baseline measurement failed — {err}\nThe objective "
                "command must print the score as 'SCORE: <number>' or as "
                "the last numeric line.",
            )
        best = baseline
        started = time.time()
        stamp = time.strftime("%Y-%m-%d %H:%M", time.localtime(started))
        await _append_log(
            f"# Research mission — {stamp}\n\n"
            f"**Intentions:**\n{intentions}\n\n"
            f"**Objective:** `{objective_command}` ({direction})\n"
            f"**Baseline:** {baseline:g} · **Budget:** {n_experiments} experiments",
        )

        # ── The loop ───────────────────────────────────────────────────
        experiments: list[dict[str, Any]] = []
        kept = 0
        run_id = ""

        def _cancel_broker_run(reason: str) -> None:
            if broker is not None and run_id:
                with contextlib.suppress(Exception):
                    broker.cancel(run_id, reason=reason)

        async def _revert() -> None:
            ok = await containers.git_revert(workspace_id, last_good)
            if not ok:
                # A failed revert means the workspace no longer matches the
                # loop's bookkeeping — continuing would attribute stale
                # changes to the next experiment. Stop loudly.
                raise RuntimeError(
                    f"git revert to {last_good} failed — stopping the loop "
                    "so unmeasured changes don't accumulate",
                )

        async def _drive_turn(prompt: str) -> str:
            """One headless coder turn through the canonical stack.
            Returns the agent's final text. Raises WorkspaceBusyError /
            TimeoutError for the caller's per-experiment handling."""
            nonlocal run_id
            run_id = ""
            registry = getattr(state, "provider_registry", None)
            if registry is None:
                raise RuntimeError("provider registry unavailable")
            backend, resolved_model = await registry.resolve_backend_with_fabric(
                model, user_id=user_id,
            )
            if backend is None:
                raise RuntimeError(f"model unavailable: {model}")

            from augmentum.models.base import InternalChatRequest, Message
            from augmentum.proxy.session import derive_kv_session_key
            request = InternalChatRequest(
                model=resolved_model or model,
                messages=[Message(role="user", content=prompt)],
                stream=True,
                # Same warm-slot affinity as interactive turns on this
                # workspace (background runs bypass the chat ingress where
                # kv keys are normally derived).
                kv_session_key=derive_kv_session_key(user_id, workspace_id),
                kv_mode="coder",
            )
            from augmentum.classifier.router import Mode
            from augmentum.proxy.handler_factory import (
                get_handler_for_mode,
                get_session_id_from_request,
            )
            session_id = get_session_id_from_request(request)
            mode_handler = get_handler_for_mode(
                Mode.CODER,
                backend,
                session_id,
                state,
                workspace_id=workspace_id,
                user_id=user_id,
                coder_strategy=coder_strategy,
            )
            if type(mode_handler).__name__ != "CoderHandler":
                raise RuntimeError(
                    "coder handler unavailable (factory fell back to "
                    f"{type(mode_handler).__name__})",
                )
            parts: list[str] = []
            async with asyncio.timeout(turn_max_s):
                async for chunk in mode_handler.handle_stream(request):
                    if chunk.content_delta:
                        parts.append(chunk.content_delta)
                    aug = chunk.augmentum or {}
                    if not run_id and aug.get("run_id"):
                        run_id = str(aug["run_id"])
            return "".join(parts).strip()

        try:
            for i in range(1, n_experiments + 1):
                await ctx.check_cancel()
                await ctx.update_progress(
                    (i - 1) / n_experiments,
                    stage=f"experiment {i}/{n_experiments} · best {best:g}",
                )
                prompt = build_experiment_prompt(
                    index=i, total=n_experiments, intentions=intentions,
                    objective_command=objective_command, direction=direction,
                    baseline=baseline, best=best, experiments=experiments,
                )

                # Drive the turn; the user's interactive sessions win ties.
                final_text = ""
                busy_tries = 0
                while True:
                    try:
                        final_text = await _drive_turn(prompt)
                        break
                    except WorkspaceBusyError:
                        busy_tries += 1
                        if busy_tries > _BUSY_RETRIES:
                            await _append_log(
                                f"## Mission ended early at experiment {i}\n"
                                "Workspace stayed busy (interactive use has "
                                "right of way).",
                            )
                            raise _EarlyFinish from None
                        await ctx.update_progress(
                            (i - 1) / n_experiments,
                            stage=f"experiment {i}: waiting for workspace",
                        )
                        await asyncio.sleep(_BUSY_WAIT_S)
                    except TimeoutError:
                        _cancel_broker_run("research_turn_timeout")
                        await _revert()
                        experiments.append({
                            "index": i, "verdict": "TIMED OUT",
                            "summary": f"agent turn exceeded {turn_max_s:g}s",
                            "score": None,
                        })
                        await _append_log(
                            f"## Experiment {i} — TIMED OUT ✗\n"
                            f"Agent turn exceeded {turn_max_s:g}s; reverted "
                            f"to {last_good}.",
                        )
                        final_text = ""
                        break
                if not final_text and experiments and experiments[-1]["index"] == i:
                    continue  # timed out — already logged, next experiment

                summary = extract_experiment_summary(final_text)

                # Commit the candidate so keep is a no-op and revert is exact.
                candidate_sha = await containers.git_checkpoint(
                    workspace_id, f"research exp {i}: {summary[:120]}",
                )
                if candidate_sha is None:
                    experiments.append({
                        "index": i, "verdict": "NO-OP",
                        "summary": summary, "score": None,
                    })
                    await _append_log(
                        f"## Experiment {i} — NO-OP —\n{summary}\n"
                        "Agent made no file changes; nothing to measure.",
                    )
                    continue

                await ctx.update_progress(
                    (i - 0.5) / n_experiments,
                    stage=f"experiment {i}: measuring",
                )
                score, err = await _measure()
                if score is None:
                    await _revert()
                    experiments.append({
                        "index": i, "verdict": "BROKE OBJECTIVE",
                        "summary": summary, "score": None,
                    })
                    await _append_log(
                        f"## Experiment {i} — BROKE OBJECTIVE ✗ "
                        f"(reverted {candidate_sha}→{last_good})\n"
                        f"{summary}\n{err}",
                    )
                    continue

                if is_improvement(score, best, direction, min_delta):
                    prev_best = best
                    best = score
                    last_good = candidate_sha
                    kept += 1
                    experiments.append({
                        "index": i, "verdict": "KEPT",
                        "summary": summary, "score": score,
                    })
                    await _append_log(
                        f"## Experiment {i} — KEPT ✓ ({prev_best:g} → {score:g})\n"
                        f"{summary}\nCommit: {candidate_sha}",
                    )
                else:
                    await _revert()
                    experiments.append({
                        "index": i, "verdict": "REVERTED",
                        "summary": summary, "score": score,
                    })
                    await _append_log(
                        f"## Experiment {i} — REVERTED ✗ "
                        f"(scored {score:g}, best {best:g})\n{summary}",
                    )
        except _EarlyFinish:
            pass
        except JobCancelled:
            # User cancel: stop the in-flight run and leave the workspace
            # at the last KEPT state — uncommitted partial work from the
            # interrupted experiment is unmeasured by contract.
            _cancel_broker_run("user_cancel")
            with contextlib.suppress(Exception):
                await _revert()
            await _append_log(
                "## Mission cancelled by user\n"
                f"Workspace left at last kept state ({last_good}).",
            )
            raise
        except Exception as exc:
            _cancel_broker_run("research_job_error")
            await _notify(
                app, user_id=user_id, ok=False,
                title="Research mission failed",
                body=str(exc)[:240],
                payload={
                    "kind": "coder_research", "workspace_id": workspace_id,
                    "job_id": ctx.job_id,
                },
                dedupe_key=f"coder-research-{ctx.job_id}",
            )
            raise

        # ── Wrap up ────────────────────────────────────────────────────
        elapsed_s = int(time.time() - started)
        ran = len(experiments)
        if direction == "minimize" and baseline:
            gain = (baseline - best) / abs(baseline) * 100.0
        elif baseline:
            gain = (best - baseline) / abs(baseline) * 100.0
        else:
            gain = 0.0
        summary_text = (
            f"## Mission summary\n"
            f"Experiments: {ran} · Kept: {kept} · "
            f"Baseline: {baseline:g} → Best: {best:g} "
            f"({gain:+.1f}%) · {elapsed_s}s"
        )
        await _append_log(summary_text)

        # One summary message into the workspace conversation (not one per
        # experiment — the log + git history carry the detail).
        try:
            sm = getattr(state, "state_manager", None)
            conn = getattr(getattr(sm, "backend", None), "conn", None) if sm else None
            if conn is not None:
                from augmentum.state.coder_persistence import CoderPersistence
                persistence = CoderPersistence(conn)
                fresh = await persistence.load_conversation(
                    workspace_id, user_id=user_id,
                ) or []
                now_ms = int(time.time() * 1000)
                fresh.append({
                    "id": f"msg_{now_ms}_research",
                    "role": "assistant",
                    "content": (
                        f"Research mission finished: {kept}/{ran} experiments "
                        f"kept, objective {baseline:g} → {best:g} "
                        f"({gain:+.1f}%). Full notebook in RESEARCH_LOG.md; "
                        "kept changes are git checkpoints."
                    ),
                })
                await persistence.save_conversation(
                    workspace_id, fresh, user_id=user_id,
                )
        except Exception:
            log.warning(
                "coder_research_conversation_save_failed",
                workspace_id=workspace_id, exc_info=True,
            )

        await ctx.update_progress(1.0, stage="complete")
        await _notify(
            app, user_id=user_id, ok=True,
            title=f"Research mission done: {kept}/{ran} kept",
            body=(
                f"Objective {baseline:g} → {best:g} ({gain:+.1f}%). "
                "See RESEARCH_LOG.md in the workspace."
            ),
            payload={
                "kind": "coder_research", "workspace_id": workspace_id,
                "run_id": run_id, "job_id": ctx.job_id,
            },
            dedupe_key=f"coder-research-{ctx.job_id}",
        )
        return {
            "experiments": ran,
            "kept": kept,
            "baseline": baseline,
            "best": best,
            "gain_pct": round(gain, 2),
            "elapsed_s": elapsed_s,
            "log_path": _LOG_FILE,
        }

    return handler
