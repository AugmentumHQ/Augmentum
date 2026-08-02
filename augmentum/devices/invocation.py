"""Invocation, event, and pairing types.

These are the things drivers receive from the registry and return back.
Kept small and dataclass-shaped so they're easy to serialize to the route
layer and the SSE event bus.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import httpx


class InvocationError(Exception):
    """Raised by drivers to surface a structured error to the registry."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "driver_error",
        retryable: bool = False,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.details = dict(details or {})


@dataclass(slots=True)
class InvocationContext:
    """Per-call context handed to a driver.

    Carries shared resources (httpx client) plus the user/session metadata
    drivers need to honor scoping. Drivers must not stash this — it's
    valid only for the lifetime of one call.
    """

    user_id: str
    http_client: httpx.AsyncClient | None = None
    session_id: str = ""
    request_id: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class InvocationResult:
    """Return shape from `DeviceDriver.invoke`."""

    ok: bool
    state: dict[str, Any] = field(default_factory=dict)
    session_id: str = ""
    message: str = ""
    code: str = ""
    retryable: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def success(
        cls,
        *,
        state: dict[str, Any] | None = None,
        session_id: str = "",
        extra: dict[str, Any] | None = None,
    ) -> InvocationResult:
        return cls(
            ok=True,
            state=dict(state or {}),
            session_id=session_id,
            extra=dict(extra or {}),
        )

    @classmethod
    def failure(
        cls,
        message: str,
        *,
        code: str = "driver_error",
        retryable: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> InvocationResult:
        return cls(
            ok=False,
            message=message,
            code=code,
            retryable=retryable,
            extra=dict(extra or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "state": dict(self.state or {}),
            "session_id": self.session_id,
            "message": self.message,
            "code": self.code,
            "retryable": bool(self.retryable),
            "extra": dict(self.extra or {}),
        }


@dataclass(slots=True)
class PairResult:
    """Return shape from `pair_start` / `pair_complete`."""

    state: str  # 'pending' | 'active' | 'expired' | 'failed'
    requires_user_action: bool = False
    completes_with_code: bool = False
    instructions: str = ""
    expires_at: float | None = None
    message: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "requires_user_action": bool(self.requires_user_action),
            "completes_with_code": bool(self.completes_with_code),
            "instructions": self.instructions,
            "expires_at": self.expires_at,
            "message": self.message,
            "extra": dict(self.extra or {}),
        }


@dataclass(slots=True)
class Event:
    """Push event emitted by a driver and routed through the bus."""

    device_id: str
    capability_id: str
    type: str
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)
    user_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "capability_id": self.capability_id,
            "type": self.type,
            "data": dict(self.data or {}),
            "ts": float(self.ts or 0.0),
        }
