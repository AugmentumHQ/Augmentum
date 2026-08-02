"""``coder_background_run`` job handler — queued, headless coder missions.

"Hand the agent a task and walk away": the user queues a prompt against a
workspace; this handler drives a full coder turn through the EXACT same
stack an interactive send uses — handler_factory → CoderHandler →
run broker → turn ledger — with the job runner standing in as the
"client" that drains the stream. Everything downstream is therefore
already wired: permission policy + audit, the verify gate, turn
archive/embeddings, training capture, and the reattachable run stream
(the UI can watch a background run live via the normal
``/api/coder/runs/{id}/stream`` reconnect path).

On completion the user gets a notification on the pre-provisioned
``coder.run.complete`` / ``coder.run.failed`` channels (catalog.py) with a
payload deep-link ``{workspace_id, run_id, review_turn_id}``, and the
prompt + final answer are appended to the workspace's persisted
conversation so the transcript survives reload (tool cards live in the
run record, reachable from the run-details drawer).

Payload shape (validated loudly at the top):

    {
        "workspace_id":   "ws_...",          // existing coder workspace
        "prompt":         "fix the ...",     // the mission
        "model":          "qwen3.6-35b",     // chosen by the USER at queue
                                             // time (never auto-selected)
        "coder_strategy": ""                 // optional override
    }

Design notes / invariants:

- Permissions: the run executes under the workspace's normal permission
  policy. ``planning_mode=auto`` allows everything; policy verdicts
  allow/deny apply with the standard audit rows; an ``ask`` verdict goes
  to the modal registry, whose 60s timeout resolves to DENY when nobody
  is watching. Headless missions on prompt-heavy workspaces will see
  denials, not hangs — the enqueue route surfaces this in its response
  so the UI can warn.
- Serialization: one run per workspace. If the workspace has an active
  run (interactive or a prior queued job), we wait briefly, then requeue
  via JobRetryable so the single-worker job loop isn't starved.
- Idempotency (restart contract): max_attempts=2 at enqueue — a
  container restart mid-mission re-queues once; the agent re-enters
  with the workspace's current state and the same objective, the same
  convergence contract interactive interrupted-turn continuation uses.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any

from augmentum.coder.run_broker import WorkspaceBusyError
from augmentum.jobs.context import JobCancelled, JobContext, JobRetryable
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

JOB_TYPE = "coder_background_run"

# How long to wait for a busy workspace to free up before handing the
# slot back to the queue (the job runner is single-worker — a coder job
# parked here blocks transcriptions/downloads behind it).
_BUSY_WAIT_MAX_S = 120
_BUSY_POLL_S = 10

# Progress-heartbeat cadence. The store throttles DB writes itself; this
# just bounds how often we even ask.
_PROGRESS_EVERY_CHUNKS = 25

# Overall wallclock ceiling for one mission. The coder loop has its own
# iteration ceilings, but a wedged backend stream (provider hang, dead
# fabric peer) would otherwise hold the SINGLE-WORKER job queue forever.
# Payload may override via ``max_seconds`` within [_MISSION_MIN_S, 6h].
_MISSION_MAX_S = 2 * 3600
_MISSION_MIN_S = 60
_MISSION_MAX_CEILING_S = 6 * 3600


def _preview(text: str, n: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1] + "…"


def _verdict_title(prompt: str, tier: str) -> str:
    """Notification title that leads with the verdict when it isn't a clean
    pass — the notification must not launder a self-report into 'done'."""
    p = _preview(prompt, 56)
    if tier == "failed":
        return f"Coder run needs a look: {p}"
    if tier == "human_required":
        return f"Coder run needs your call: {p}"
    return f"Coder mission done: {p}"


def _verdict_body(final: str, elapsed_s: int, verdict: Any) -> str:
    """One honest line about verification, then the answer preview."""
    tier = getattr(verdict, "tier", "unchecked")
    reason = getattr(verdict, "reason", "") or ""
    if tier == "probable":
        lead = "A second model reviewed the diff and judges it correct (not mechanically proven). "
    elif tier == "failed":
        lead = f"An independent check flagged a problem: {reason} "
    elif tier == "human_required":
        lead = f"Needs your decision: {reason} "
    else:  # unchecked
        lead = "Reported complete — not independently verified. "
    return _preview(lead + (final or f"Finished in {elapsed_s}s."), 240)


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
    """Publish the terminal notification. Best-effort — a notify failure
    must never turn a completed mission into a failed job."""
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
            # One-tap click-through. The server-side action handler (see
            # _handle_open_action) just acknowledges; the actual
            # navigation happens client-side via the synchronous
            # ``augmentum:notification-action`` DOM event coder.js
            # listens for (same gesture-context pattern connect uses).
            actions=[
                NotificationAction(
                    id="open", label="Open workspace", style="primary",
                ),
            ] if payload.get("workspace_id") else None,
        )
    except Exception:
        log.warning("coder_background_run_notify_failed", exc_info=True)


async def _emit_run_perception(
    app: Any,
    *,
    user_id: str,
    ok: bool,
    envelope: dict[str, Any],
) -> None:
    """Bridge a finished background coder run into the companion's world.

    Two coarse signals (never the fine turn stream): publish
    ``agent.run.completed`` on the PresenceBus so presence/perception sees it,
    and enqueue a coder-specific initiative so she can surface it in her own
    voice (and, later, open the brief). Best-effort — a perception failure
    must never fail a completed mission. The generic ``job_finished``
    initiative is suppressed for this job type (see bridges._QUIET_JOB_TYPES)
    so this richer signal is the only one.
    """
    try:
        runtime = getattr(app.state, "companion_runtime", None)
        bus = getattr(runtime, "bus", None)
        companion_id = getattr(runtime, "companion_id", "") or "becca"
        payload = {"ok": ok, **envelope}
        if bus is not None:
            with contextlib.suppress(Exception):
                await bus.publish_topic(
                    "agent.run.completed",
                    payload,
                    source_companion_id=companion_id,
                    propagation="factual_only",
                    owner_user_id=user_id,
                )
        sm = getattr(app.state, "state_manager", None)
        conn = getattr(getattr(sm, "backend", None), "conn", None) if sm else None
        if conn is not None:
            from augmentum.companion_runtime.bridges import (
                enqueue_external_initiative,
            )
            await enqueue_external_initiative(
                conn,
                companion_id=companion_id,
                target_user_id=user_id,
                kind="coder_run_completed",
                payload=payload,
                # A failed mission is more worth surfacing than a clean one.
                score=0.72 if not ok else 0.66,
            )
    except Exception:
        log.warning("coder_background_run_perception_failed", exc_info=True)


async def _handle_open_action(notification, action_id, request):  # noqa: ANN001
    """Action callback for the notification's "Open workspace" button.

    Navigation is client-side (the DOM event fires before this POST);
    the server's whole job here is to acknowledge + let the route layer
    persist the click. Registered for ``coder.run.*`` at import time.
    """
    return {"ok": True, "client_nav": "coder_workspace"}


def _register_action_handler() -> None:
    try:
        from augmentum.notifications.actions import register_action_handler
        register_action_handler("coder.run.*", _handle_open_action)
    except Exception:  # pragma: no cover — notifications optional in tests
        log.debug("coder_run_action_handler_registration_skipped", exc_info=True)


_register_action_handler()


def make_coder_background_run_handler(app):
    """Factory bound to runtime app services (mirrors make_bug_finder_run_handler)."""

    async def handler(ctx: JobContext) -> dict | None:
        payload = ctx.payload or {}
        workspace_id = str(payload.get("workspace_id") or "").strip()
        prompt = str(payload.get("prompt") or "").strip()
        model = str(payload.get("model") or "").strip()
        coder_strategy = str(payload.get("coder_strategy") or "").strip()
        if not workspace_id or not prompt or not model:
            raise ValueError(
                "coder_background_run payload requires workspace_id, prompt, model",
            )
        user_id = ctx.user_id
        if not user_id:
            raise ValueError("coder_background_run requires a user-scoped job")

        state = app.state

        # ── Serialize per workspace ────────────────────────────────────
        # An interactive run (or an earlier queued mission) owns the
        # workspace: two agents mutating the same files is corruption,
        # not concurrency. Wait briefly, then requeue.
        #
        # NOTE this poll is a POLITENESS gate only (don't hog the
        # single-worker queue behind a busy workspace) — it is NOT the
        # exclusivity guarantee. Many awaits pass between this check
        # and the eventual broker.start_run inside handle_stream; the
        # atomic guarantee is start_run's own exclusive_workspace check
        # under the broker lock, whose WorkspaceBusyError we catch
        # below and convert to the same JobRetryable requeue.
        broker = getattr(state, "coder_run_broker", None)
        if broker is not None:
            waited = 0.0
            while (
                broker.get_active_for_workspace(
                    user_id=user_id, workspace_id=workspace_id,
                )
                is not None
            ):
                if waited >= _BUSY_WAIT_MAX_S:
                    raise JobRetryable(
                        f"workspace {workspace_id} still busy after "
                        f"{_BUSY_WAIT_MAX_S}s — requeued",
                    )
                await ctx.update_progress(0.0, stage="waiting for workspace")
                await ctx.check_cancel()
                await asyncio.sleep(_BUSY_POLL_S)
                waited += _BUSY_POLL_S

        # ── Resolve the model with the user's provider visibility ──────
        registry = getattr(state, "provider_registry", None)
        if registry is None:
            raise RuntimeError("provider registry unavailable")
        backend, resolved_model = await registry.resolve_backend_with_fabric(
            model, user_id=user_id,
        )
        if backend is None:
            raise RuntimeError(f"model unavailable: {model}")

        # ── Conversation context (mirror of the UI's getMessagesForLLM) ─
        sm = getattr(state, "state_manager", None)
        conn = getattr(getattr(sm, "backend", None), "conn", None) if sm else None
        if conn is None:
            raise RuntimeError("state backend unavailable")
        from augmentum.state.coder_persistence import CoderPersistence
        persistence = CoderPersistence(conn)
        history = await persistence.load_conversation(
            workspace_id, user_id=user_id,
        ) or []

        from augmentum.models.base import InternalChatRequest, Message
        from augmentum.proxy.session import derive_kv_session_key
        messages = [
            Message(role=m["role"], content=str(m.get("content") or ""))
            for m in history
            if m.get("role") in ("user", "assistant")
            and str(m.get("content") or "").strip()
        ]
        messages.append(Message(role="user", content=prompt))
        request = InternalChatRequest(
            model=resolved_model or model, messages=messages, stream=True,
            # Managed KV affinity. Background runs bypass the chat ingress
            # (openai_routes → resolve_session_keys), which is where a coder
            # request normally gets its kv_session_key/kv_mode derived from
            # the workspace id. Without these, headless runs go
            # kv_tier=unmanaged — no slot affinity, no prefix tracking, no
            # cross-iteration reuse (full re-prefill every iteration). Mirror
            # resolve_session_keys' coder branch so a background turn shares
            # the same warm slot as interactive turns on this workspace.
            # Verified 2026-07-03 (headless runs logged kv_tier=unmanaged).
            kv_session_key=derive_kv_session_key(user_id, workspace_id),
            kv_mode="coder",
        )

        # ── Build the coder handler through the canonical factory ──────
        # Same session derivation as the HTTP entry (content fingerprint)
        # so a background turn shares CoderState continuity with the
        # interactive session it extends.
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
        # The factory falls back to PassthroughHandler when coder init
        # fails — a passthrough "mission" would burn tokens answering
        # without tools. Fail loudly instead.
        if type(mode_handler).__name__ != "CoderHandler":
            raise RuntimeError(
                "coder handler unavailable (factory fell back to "
                f"{type(mode_handler).__name__})",
            )

        # ── Drive the run; the job IS the client ───────────────────────
        run_id = ""
        review_turn_id = ""
        parts: list[str] = []
        tool_calls = 0
        stage = "starting"
        started = time.time()
        chunk_i = 0
        # Wallclock ceiling — see _MISSION_MAX_S. asyncio.timeout also
        # trips on a stream that hangs BETWEEN chunks (provider wedge),
        # which a per-chunk elapsed check couldn't.
        try:
            max_s = float(payload.get("max_seconds") or _MISSION_MAX_S)
        except (TypeError, ValueError):
            max_s = float(_MISSION_MAX_S)
        max_s = max(_MISSION_MIN_S, min(_MISSION_MAX_CEILING_S, max_s))

        def _cancel_broker_run(reason: str) -> None:
            # Cleanup path, not save/load — suppress is safe.
            if broker is not None and run_id:
                with contextlib.suppress(Exception):
                    broker.cancel(run_id, reason=reason)

        try:
            async with asyncio.timeout(max_s):
                async for chunk in mode_handler.handle_stream(request):
                    chunk_i += 1
                    if chunk.content_delta:
                        parts.append(chunk.content_delta)
                    aug = chunk.augmentum or {}
                    if not run_id and aug.get("run_id"):
                        run_id = str(aug["run_id"])
                    if aug.get("status") == "tool_call":
                        tool_calls += 1
                        tc = aug.get("tool_call") or {}
                        stage = f"tool: {tc.get('tool') or tc.get('name') or '?'}"
                    elif aug.get("phase"):
                        stage = str(aug["phase"])
                    if aug.get("status") == "complete" and aug.get("review_turn_id"):
                        review_turn_id = str(aug["review_turn_id"])
                    if chunk_i % _PROGRESS_EVERY_CHUNKS == 0:
                        # No meaningful % for an open-ended agent turn —
                        # the moving stage label is the liveness signal.
                        await ctx.update_progress(0.5, stage=stage)
                        await ctx.check_cancel()
        except JobCancelled:
            # User cancel is an outcome the user CHOSE — stop the
            # detached broker run, but never send a "mission failed"
            # notification for it. The jobs surface shows 'cancelled'.
            _cancel_broker_run("user_cancel")
            raise
        except TimeoutError:
            # Wedged stream / runaway mission: kill the detached run so
            # it can't keep mutating the workspace after we walk away.
            _cancel_broker_run("background_timeout")
            timeout_err = RuntimeError(
                f"mission exceeded the {max_s}s wallclock ceiling — "
                "run cancelled (partial work is in the workspace; see the "
                "run record for what happened)",
            )
            await _notify(
                app,
                user_id=user_id,
                ok=False,
                title=f"Coder mission timed out: {_preview(prompt, 60)}",
                body=_preview(str(timeout_err), 240),
                payload={
                    "kind": "coder_run",
                    "workspace_id": workspace_id,
                    "run_id": run_id,
                    "job_id": ctx.job_id,
                },
                dedupe_key=f"coder-bg-{ctx.job_id}",
            )
            await _emit_run_perception(
                app, user_id=user_id, ok=False,
                envelope={
                    "kind": "coder_run", "workspace_id": workspace_id,
                    "run_id": run_id, "job_id": ctx.job_id,
                    "prompt": _preview(prompt, 120),
                    "failure": "timeout",
                },
            )
            raise timeout_err from None
        except WorkspaceBusyError as exc:
            # Lost the start_run exclusivity race (an interactive turn
            # landed between our busy-poll and the broker registration).
            # Not a failure — hand the slot back to the queue and let
            # the retry re-enter the busy-wait poll above. No broker
            # cancel needed: our run was rejected before registration.
            raise JobRetryable(
                f"workspace {workspace_id} grabbed by run "
                f"{exc.holder_run_id} — requeued",
            ) from exc
        except Exception as exc:
            # Terminal failure — notify BEFORE re-raising so the user
            # hears about it even though nobody was watching. The broker
            # task dies with its own error; cancel is belt-and-suspenders
            # for the case where the failure was OURS (drain-side).
            _cancel_broker_run("background_job_error")
            await _notify(
                app,
                user_id=user_id,
                ok=False,
                title=f"Coder mission failed: {_preview(prompt, 60)}",
                body=_preview(str(exc), 240),
                payload={
                    "kind": "coder_run",
                    "workspace_id": workspace_id,
                    "run_id": run_id,
                    "job_id": ctx.job_id,
                },
                dedupe_key=f"coder-bg-{ctx.job_id}",
            )
            await _emit_run_perception(
                app, user_id=user_id, ok=False,
                envelope={
                    "kind": "coder_run", "workspace_id": workspace_id,
                    "run_id": run_id, "job_id": ctx.job_id,
                    "prompt": _preview(prompt, 120),
                    "failure": "error",
                },
            )
            raise

        final = "".join(parts).strip()
        elapsed_s = int(time.time() - started)

        # ── Independent cross-model verification (before we tell the user) ──
        # A DIFFERENT model (the pinned heavyweight) reviews the actual diff
        # against the original ask and returns an honest tiered verdict. This
        # is the automated first verifier so the human is the SECOND verifier
        # (exception-gate), not the depleted first-pass checker. Best-effort:
        # a verifier failure degrades to 'unchecked', never fails the mission.
        try:
            from augmentum.coder.run_verifier import save_verdict, verify_coder_run
            verdict = await verify_coder_run(
                state, user_id=user_id, review_turn_id=review_turn_id,
                prompt=prompt, answer=final, driver_model=resolved_model or model,
                run_id=run_id, workspace_id=workspace_id,
            )
            # Persist so a brief opened cold (stale notification deep-link) can
            # still show it — the envelope only covers the live open path.
            await save_verdict(
                conn, run_id=run_id, user_id=user_id,
                workspace_id=workspace_id, verdict=verdict,
            )
        except Exception:
            from augmentum.coder.run_verifier import TIER_UNCHECKED, RunVerdict
            log.warning("coder_background_run_verify_failed", exc_info=True)
            verdict = RunVerdict(tier=TIER_UNCHECKED, reason="verification errored")
        verification = verdict.to_envelope()

        # ── Persist the exchange into the workspace conversation ───────
        # Interactively the CLIENT saves the conversation; headless has
        # no client, so the transcript would silently drop this turn on
        # next load. Re-read fresh history first — an interactive client
        # may have saved while we ran. Tool cards stay in the run record
        # (run-details drawer); the transcript gets the prompt + answer.
        try:
            fresh = await persistence.load_conversation(
                workspace_id, user_id=user_id,
            ) or []
            now_ms = int(time.time() * 1000)
            fresh.append(
                {"id": f"msg_{now_ms}_bg1", "role": "user", "content": prompt},
            )
            if final:
                fresh.append({
                    "id": f"msg_{now_ms}_bg2",
                    "role": "assistant",
                    "content": final,
                })
            await persistence.save_conversation(
                workspace_id, fresh, user_id=user_id,
            )
        except Exception:
            log.warning(
                "coder_background_run_conversation_save_failed",
                workspace_id=workspace_id, exc_info=True,
            )

        await ctx.update_progress(1.0, stage="complete")
        # Honest completion title: lead with the verdict when it's not a clean
        # pass, so the notification itself doesn't launder a self-report.
        title = _verdict_title(prompt, verdict.tier)
        body = _verdict_body(final, elapsed_s, verdict)
        await _notify(
            app,
            user_id=user_id,
            ok=True,
            title=title,
            body=body,
            payload={
                "kind": "coder_run",
                "workspace_id": workspace_id,
                "run_id": run_id,
                "review_turn_id": review_turn_id,
                "job_id": ctx.job_id,
                "verification": verification,
            },
            dedupe_key=f"coder-bg-{ctx.job_id}",
        )
        await _emit_run_perception(
            app,
            user_id=user_id,
            ok=True,
            envelope={
                "kind": "coder_run",
                "workspace_id": workspace_id,
                "run_id": run_id,
                "review_turn_id": review_turn_id,
                "job_id": ctx.job_id,
                "prompt": _preview(prompt, 120),
                "answer_preview": _preview(final, 200),
                "tool_calls": tool_calls,
                "elapsed_s": elapsed_s,
                "verification": verification,
            },
        )
        return {
            "run_id": run_id,
            "review_turn_id": review_turn_id,
            "tool_calls": tool_calls,
            "elapsed_s": elapsed_s,
            "answer_chars": len(final),
            "verification": verification,
        }

    return handler
