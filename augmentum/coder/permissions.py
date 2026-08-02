"""Permission-approval registry for the coder agent.

Under ``AUGMENTUM_CODER_PERMISSIONS=confirm_mutations`` the hybrid loop
gates approval-required tools (see ``_APPROVAL_REQUIRED_TOOLS`` in
``modes/coder/handler.py``) behind an async callback. This module is the
bridge between that callback and the UI:

- The callback (installed on each ``CoderHandler`` via handler_factory)
  calls ``registry.request(user_id, tool_name, tool_input)``. That
  creates a ``PermissionRequest`` with an ``asyncio.Future`` and returns
  when the UI resolves it (or the timeout fires and it resolves False).

- The UI polls ``GET /v1/coder/permissions/pending`` every couple of
  seconds while coder mode is active. For each request it shows a modal;
  clicking Allow/Deny hits the approve/deny endpoint which resolves the
  registered future.

Stale requests are cleaned up by their own timeout task, so even if the
UI never responds the callback returns (False) and the loop continues.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Audit sink contract: called with the same kwargs as
# ``PermissionAuditStore.record``. Must never raise (the store's
# record() is best-effort); the registry guards anyway.
AuditSink = Callable[..., Awaitable[None]]


@dataclass
class PermissionRequest:
    """A single pending approval request.

    The ``future`` resolves to True (allow), False (deny), or False on
    timeout. Once resolved, the request is removed from the registry.
    """

    id: str
    user_id: str
    tool_name: str
    tool_input: dict
    created_at: float
    future: asyncio.Future = field(repr=False)
    workspace_id: str = ""

    def to_dict(self) -> dict:
        """Serialisable shape for the /pending API response."""
        return {
            "id": self.id,
            "tool_name": self.tool_name,
            "tool_input": self.tool_input,
            "created_at": self.created_at,
            "age_seconds": max(0.0, time.time() - self.created_at),
        }


class PermissionRegistry:
    """Process-wide registry of pending tool-permission requests.

    Keyed by request_id; ``pending_for(user_id)`` filters by owner so
    users only see their own requests. Concurrency: asyncio.Future is
    the sync primitive — no locks needed because everything runs on the
    same event loop.
    """

    def __init__(
        self,
        default_timeout: float = 60.0,
        audit_sink: AuditSink | None = None,
    ) -> None:
        self._pending: dict[str, PermissionRequest] = {}
        self._default_timeout = default_timeout
        self._audit_sink = audit_sink

    async def _audit(
        self,
        req: PermissionRequest,
        *,
        decision: str,
        decided_by: str,
    ) -> None:
        """Write one durable audit row. Best-effort, never raises."""
        if self._audit_sink is None:
            return
        try:
            await self._audit_sink(
                tool_name=req.tool_name,
                decision=decision,
                decided_by=decided_by,
                user_id=req.user_id,
                workspace_id=req.workspace_id,
                tool_input=req.tool_input,
            )
        except Exception as exc:
            log.warning(
                "coder.permission_audit_sink_failed",
                request_id=req.id,
                tool_name=req.tool_name,
                error=str(exc),
            )

    async def request(
        self,
        user_id: str,
        tool_name: str,
        tool_input: dict,
        timeout: float | None = None,
        workspace_id: str = "",
    ) -> bool:
        """Register a request and block until resolved (or timed out).

        Returns True (allow) or False (deny / timeout). Never raises —
        any internal failure resolves to False so the caller sees a
        structured denial rather than an exception tearing up the loop.
        """
        req_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()

        req = PermissionRequest(
            id=req_id,
            user_id=user_id or "",
            tool_name=tool_name,
            tool_input=tool_input,
            created_at=time.time(),
            future=future,
            workspace_id=workspace_id or "",
        )
        self._pending[req_id] = req
        log.info(
            "coder.permission_requested",
            request_id=req_id,
            user_id=user_id,
            tool_name=tool_name,
        )

        effective_timeout = timeout if timeout is not None else self._default_timeout
        try:
            result = await asyncio.wait_for(future, timeout=effective_timeout)
            await self._audit(
                req,
                decision="allowed" if result else "denied",
                decided_by="user",
            )
            return bool(result)
        except TimeoutError:
            log.info(
                "coder.permission_timeout",
                request_id=req_id,
                tool_name=tool_name,
                timeout=effective_timeout,
            )
            await self._audit(req, decision="denied", decided_by="timeout")
            return False
        except asyncio.CancelledError:
            # Task cancellation (client disconnected, session closed) — deny.
            # Fire-and-forget: awaiting the sink here would race a second
            # cancellation during teardown and could swallow the re-raise.
            log.debug("coder.permission_cancelled", request_id=req_id)
            if self._audit_sink is not None:
                asyncio.get_running_loop().create_task(
                    self._audit(req, decision="denied", decided_by="disconnect"),
                )
            raise
        finally:
            self._pending.pop(req_id, None)

    def pending_for(self, user_id: str) -> list[PermissionRequest]:
        """Return all pending requests belonging to ``user_id``.

        An empty ``user_id`` sees every request in the registry (useful
        for single-tenant dev setups without auth).
        """
        if not user_id:
            return list(self._pending.values())
        return [r for r in self._pending.values() if r.user_id == user_id]

    def get(self, request_id: str) -> PermissionRequest | None:
        return self._pending.get(request_id)

    def resolve(self, request_id: str, approved: bool) -> bool:
        """Resolve a pending request. Returns True if the request existed
        and was successfully resolved, False if unknown or already
        settled."""
        req = self._pending.get(request_id)
        if req is None:
            return False
        if req.future.done():
            return False
        req.future.set_result(bool(approved))
        return True

    def size(self) -> int:
        return len(self._pending)
