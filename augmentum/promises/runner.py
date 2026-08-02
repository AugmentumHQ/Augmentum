"""MissionRunner — the domain-neutral driver for a list of Promises.

The runner owns control flow only:
  - pick the next runnable promise (DFS so children run before parents)
  - call the caller's ``act_fn`` to attempt it
  - on ``ATTEMPT_COMPLETE``, invoke the registered verifier
  - advance / retry / reject based on verification
  - give the caller a chance to ``replan_fn`` the remaining tail after
    each promise resolves, using freshly observed evidence
  - cascade rejection from children to parents

It knows nothing about tools, containers, TTS, or LLMs. Callers plug in
``act_fn`` (how to attempt), ``verify_fns`` (how to check), and
optionally ``replan_fn`` (how to re-draft the remaining promises given
what the last attempt observed).
"""
from __future__ import annotations

import time
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator

from augmentum.promises.models import (
    ActEvent,
    ActEventKind,
    Promise,
    PromiseContext,
    PromiseStatus,
    RunnerEvent,
    RunnerEventKind,
    VerificationKind,
)
from augmentum.promises.verify import VerifyFn
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

ActFn = Callable[[Promise, PromiseContext], AsyncIterator[ActEvent]]

# ReplanFn is called after a top-level promise resolves (fulfilled or
# rejected). Args: (mission, resolved_promise). Return either None to
# keep the existing tail, or a list of Promise objects to REPLACE every
# still-pending top-level promise that comes after ``resolved_promise``.
# Returning an empty list finishes the mission early (useful when the
# planner decides remaining steps aren't needed after earlier discoveries).
ReplanFn = Callable[[list[Promise], Promise], Awaitable[list[Promise] | None]]


class MissionRunner:
    """Drive a mission to completion by alternating act and verify.

    Parameters
    ----------
    max_promise_attempts:
        Hard cap on the total number of promise attempts across the
        mission. Prevents runaway loops from e.g. a verifier that
        always rejects combined with a max_attempts that never saturates.
        Defaults to 100, which comfortably covers 15-step missions with
        retries and one level of decomposition.
    """

    def __init__(self, *, max_promise_attempts: int = 100) -> None:
        self._cap = max_promise_attempts

    async def run(
        self,
        mission: list[Promise],
        act_fn: ActFn,
        verify_fns: dict[VerificationKind, VerifyFn],
        *,
        replan_fn: ReplanFn | None = None,
    ) -> AsyncIterator[RunnerEvent]:
        yield RunnerEvent(
            kind=RunnerEventKind.MISSION_STARTED,
            payload={"promises": len(mission)},
        )

        processed = 0
        while True:
            promise, ancestors = self._next_runnable(mission)
            if promise is None:
                break
            if processed >= self._cap:
                log.warning(
                    "mission_runner_cap_exceeded",
                    cap=self._cap,
                )
                yield RunnerEvent(
                    kind=RunnerEventKind.MISSION_FAILED,
                    payload={"reason": f"attempt cap exceeded ({self._cap})"},
                )
                return
            processed += 1

            promise.status = PromiseStatus.IN_PROGRESS
            yield RunnerEvent(
                kind=RunnerEventKind.PROMISE_STARTED, promise=promise,
            )

            ctx = PromiseContext(
                current=promise, mission=mission, ancestors=ancestors,
            )

            terminal_status: PromiseStatus | None = None
            async for event in self._drive_attempt(
                promise, ctx, act_fn, verify_fns,
            ):
                yield event
                if event.kind == RunnerEventKind.PROMISE_FULFILLED:
                    terminal_status = PromiseStatus.FULFILLED
                elif event.kind == RunnerEventKind.PROMISE_REJECTED:
                    terminal_status = PromiseStatus.REJECTED

            # Invite the caller to re-draft the remaining tail using the
            # evidence just observed. Only fires on terminal promise
            # resolution and only for top-level promises (decomposition
            # children live under a parent that's re-considered when its
            # subtree completes).
            if (
                replan_fn is not None
                and terminal_status is not None
                and not ancestors
            ):
                try:
                    replacement = await replan_fn(mission, promise)
                except Exception as exc:  # noqa: BLE001 — replan is best-effort
                    log.warning(
                        "mission_replan_failed",
                        promise_id=promise.id,
                        error=str(exc),
                    )
                    replacement = None
                if replacement is not None:
                    replaced = self._replace_tail(mission, promise, replacement)
                    if replaced:
                        yield RunnerEvent(
                            kind=RunnerEventKind.MISSION_REPLANNED,
                            promise=promise,
                            payload={
                                "after_id": promise.id,
                                "new_tail": [r.description for r in replacement],
                            },
                        )

        # Terminal cascade: parents with rejected children are rejected too.
        self._cascade_rejections(mission)

        failures = [
            p for p in _flatten(mission) if p.status == PromiseStatus.REJECTED
        ]
        if failures:
            yield RunnerEvent(
                kind=RunnerEventKind.MISSION_FAILED,
                payload={
                    "rejected": len(failures),
                    "first_reason": failures[0].evidence or "unknown",
                },
            )
        else:
            yield RunnerEvent(
                kind=RunnerEventKind.MISSION_COMPLETED,
                payload={"promises": len(list(_flatten(mission)))},
            )

    # ------------------------------------------------------------------
    # Per-attempt driver
    # ------------------------------------------------------------------

    async def _drive_attempt(
        self,
        promise: Promise,
        ctx: PromiseContext,
        act_fn: ActFn,
        verify_fns: dict[VerificationKind, VerifyFn],
    ) -> AsyncIterator[RunnerEvent]:
        try:
            async for act_event in act_fn(promise, ctx):
                if act_event.kind == ActEventKind.PROGRESS:
                    yield RunnerEvent(
                        kind=RunnerEventKind.PROMISE_PROGRESS,
                        promise=promise,
                        payload=act_event.payload,
                    )
                    continue

                if act_event.kind == ActEventKind.NEEDS_DECOMPOSITION:
                    children = act_event.payload or []
                    if not isinstance(children, list) or not children:
                        promise.status = PromiseStatus.REJECTED
                        promise.evidence = (
                            "decomposition emitted empty or invalid children"
                        )
                        yield RunnerEvent(
                            kind=RunnerEventKind.PROMISE_REJECTED,
                            promise=promise,
                            payload={"reason": promise.evidence},
                        )
                        return
                    for child in children:
                        if isinstance(child, Promise):
                            child.parent_id = promise.id
                    promise.children = [
                        c for c in children if isinstance(c, Promise)
                    ]
                    promise.status = PromiseStatus.PENDING  # waits on children
                    yield RunnerEvent(
                        kind=RunnerEventKind.PROMISE_DECOMPOSED,
                        promise=promise,
                        payload={"children": len(promise.children)},
                    )
                    return

                if act_event.kind == ActEventKind.CANNOT_FULFILL:
                    reason = (
                        str(act_event.payload) if act_event.payload
                        else "act_fn gave up"
                    )
                    promise.status = PromiseStatus.REJECTED
                    promise.evidence = reason
                    yield RunnerEvent(
                        kind=RunnerEventKind.PROMISE_REJECTED,
                        promise=promise,
                        payload={"reason": reason},
                    )
                    return

                if act_event.kind == ActEventKind.ATTEMPT_COMPLETE:
                    yield RunnerEvent(
                        kind=RunnerEventKind.PROMISE_VERIFYING,
                        promise=promise,
                    )
                    verifier = verify_fns.get(promise.verify.kind)
                    if verifier is None:
                        reason = (
                            f"no verifier registered for "
                            f"{promise.verify.kind.value}"
                        )
                        promise.status = PromiseStatus.REJECTED
                        promise.evidence = reason
                        yield RunnerEvent(
                            kind=RunnerEventKind.PROMISE_REJECTED,
                            promise=promise,
                            payload={"reason": reason},
                        )
                        return

                    result = await verifier(promise, act_event.evidence)
                    if result.passed:
                        promise.status = PromiseStatus.FULFILLED
                        promise.evidence = (
                            (act_event.evidence or "").strip() or result.reason
                        )
                        promise.fulfilled_at = time.time()
                        yield RunnerEvent(
                            kind=RunnerEventKind.PROMISE_FULFILLED,
                            promise=promise,
                            payload={"reason": result.reason},
                        )
                        return

                    promise.attempts += 1
                    if promise.attempts >= promise.max_attempts:
                        promise.status = PromiseStatus.REJECTED
                        promise.evidence = result.reason
                        yield RunnerEvent(
                            kind=RunnerEventKind.PROMISE_REJECTED,
                            promise=promise,
                            payload={"reason": result.reason},
                        )
                        return

                    # Retry: revert to pending; next loop iteration will
                    # re-run act_fn with promise.attempts now > 0 so the
                    # act layer can adapt.
                    promise.status = PromiseStatus.PENDING
                    promise.evidence = result.reason  # transient
                    yield RunnerEvent(
                        kind=RunnerEventKind.PROMISE_RETRY,
                        promise=promise,
                        payload={
                            "reason": result.reason,
                            "attempt": promise.attempts,
                        },
                    )
                    return
        except Exception as exc:  # noqa: BLE001 — surface act failures as rejection
            log.warning(
                "mission_act_fn_raised",
                promise_id=promise.id,
                error=str(exc),
                exc_info=True,
            )
            promise.status = PromiseStatus.REJECTED
            promise.evidence = f"act error: {exc}"
            yield RunnerEvent(
                kind=RunnerEventKind.PROMISE_REJECTED,
                promise=promise,
                payload={"reason": promise.evidence, "error": True},
            )

        # act_fn returned without emitting a terminal event — treat as cannot_fulfill
        if promise.status == PromiseStatus.IN_PROGRESS:
            promise.status = PromiseStatus.REJECTED
            promise.evidence = "act_fn exited without an outcome event"
            yield RunnerEvent(
                kind=RunnerEventKind.PROMISE_REJECTED,
                promise=promise,
                payload={"reason": promise.evidence},
            )

    # ------------------------------------------------------------------
    # Tail replacement
    # ------------------------------------------------------------------

    def _replace_tail(
        self,
        mission: list[Promise],
        after: Promise,
        replacement: list[Promise],
    ) -> bool:
        """Replace every pending top-level promise after ``after``.

        Existing fulfilled / rejected promises are preserved so the
        mission log keeps its history. Only promises still PENDING (and
        positioned after ``after`` in the top-level list) are dropped.

        Returns True if any changes were made.
        """
        try:
            idx = mission.index(after)
        except ValueError:
            return False
        # Split around ``after``: keep everything up to and including it.
        head = mission[: idx + 1]
        tail = mission[idx + 1 :]
        # Keep tail promises that are already resolved or in-progress so
        # we don't lose history or stomp running state.
        kept = [
            p for p in tail
            if p.status != PromiseStatus.PENDING
        ]
        if not replacement and not tail:
            return False
        mission.clear()
        mission.extend(head + kept + list(replacement))
        return True

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def _next_runnable(
        self, mission: list[Promise],
    ) -> tuple[Promise | None, list[Promise]]:
        """DFS for the next pending promise.

        A parent with children is runnable only after every child is
        FULFILLED. A parent with any REJECTED child is blocked (and
        will be promoted to REJECTED during the terminal cascade).
        """
        for p in mission:
            found, ancestors = self._descend(p, [])
            if found is not None:
                return found, ancestors
        return None, []

    def _descend(
        self, p: Promise, ancestors: list[Promise],
    ) -> tuple[Promise | None, list[Promise]]:
        if p.children:
            for child in p.children:
                found, found_ancestors = self._descend(child, ancestors + [p])
                if found is not None:
                    return found, found_ancestors
            # No runnable descendant — check if parent is unblocked.
            statuses = {c.status for c in p.children}
            if PromiseStatus.REJECTED in statuses:
                return None, []  # blocked; terminal cascade will reject p
            all_fulfilled = (
                bool(statuses)
                and all(s == PromiseStatus.FULFILLED for s in statuses)
            )
            if all_fulfilled and p.status == PromiseStatus.PENDING:
                return p, ancestors
            return None, []
        if p.status == PromiseStatus.PENDING:
            return p, ancestors
        return None, []

    # ------------------------------------------------------------------
    # Cascade
    # ------------------------------------------------------------------

    def _cascade_rejections(self, mission: list[Promise]) -> None:
        """Walk bottom-up: any parent with a rejected child is rejected too."""
        for p in mission:
            self._cascade_one(p)

    def _cascade_one(self, p: Promise) -> None:
        for child in p.children:
            self._cascade_one(child)
        has_rejected_child = any(
            c.status == PromiseStatus.REJECTED for c in p.children
        )
        if (
            p.children
            and p.status != PromiseStatus.REJECTED
            and has_rejected_child
        ):
            p.status = PromiseStatus.REJECTED
            if not p.evidence:
                p.evidence = "child promise rejected"


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _flatten(mission: list[Promise]) -> Iterator[Promise]:
    for p in mission:
        yield from _flatten_subtree(p)


def _flatten_subtree(p: Promise) -> Iterator[Promise]:
    for child in p.children:
        yield from _flatten_subtree(child)
    yield p
