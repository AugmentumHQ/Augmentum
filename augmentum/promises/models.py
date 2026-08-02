"""Dataclasses for the Promise runtime.

A Promise has four states: pending, in_progress, fulfilled, rejected.
Every promise carries a ``Verification`` spec that the runner uses to
decide whether an attempt actually fulfilled the promise — no
self-reported completion.

Events flow two ways:
- ``ActEvent`` — emitted by the caller's ``act_fn`` back to the runner
- ``RunnerEvent`` — emitted by the runner for the caller to stream
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class PromiseStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    FULFILLED = "fulfilled"
    REJECTED = "rejected"


class VerificationKind(str, Enum):
    """Verification types the runner knows how to dispatch.

    ``any_of`` is a semantic escape hatch: a planner that can't predict
    the exact post-state (e.g., "a repo was cloned — don't know the dir
    name yet") can list multiple candidate checks and pass if any one
    passes. This decouples the plan from literal path assumptions.
    """

    SHELL = "shell"
    FILE = "file"
    ALWAYS = "always"
    ANY_OF = "any_of"
    # Reserved for phase 2:
    HTTP = "http"
    USER_CONFIRM = "user_confirm"
    LLM_JUDGE = "llm_judge"


class ActEventKind(str, Enum):
    """Kinds of events an ``act_fn`` may emit back to the runner."""

    PROGRESS = "progress"
    ATTEMPT_COMPLETE = "attempt_complete"
    NEEDS_DECOMPOSITION = "needs_decomposition"
    CANNOT_FULFILL = "cannot_fulfill"


class RunnerEventKind(str, Enum):
    """Kinds of events the runner emits to its caller."""

    MISSION_STARTED = "mission_started"
    PROMISE_STARTED = "promise_started"
    PROMISE_PROGRESS = "promise_progress"
    PROMISE_VERIFYING = "promise_verifying"
    PROMISE_FULFILLED = "promise_fulfilled"
    PROMISE_RETRY = "promise_retry"
    PROMISE_REJECTED = "promise_rejected"
    PROMISE_DECOMPOSED = "promise_decomposed"
    MISSION_REPLANNED = "mission_replanned"
    MISSION_COMPLETED = "mission_completed"
    MISSION_FAILED = "mission_failed"


@dataclass
class Verification:
    """How the runner verifies a promise was fulfilled."""

    kind: VerificationKind
    spec: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind.value, "spec": dict(self.spec)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Verification:
        return cls(kind=VerificationKind(d["kind"]), spec=dict(d.get("spec") or {}))

    @classmethod
    def always(cls) -> Verification:
        return cls(kind=VerificationKind.ALWAYS)


@dataclass
class Promise:
    """A commitment to a future state transition, with a verification contract."""

    description: str = ""
    verify: Verification = field(default_factory=Verification.always)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    status: PromiseStatus = PromiseStatus.PENDING
    evidence: str | None = None
    attempts: int = 0
    max_attempts: int = 3
    children: list[Promise] = field(default_factory=list)
    # Per-attempt fingerprints of the tool calls the act_fn emitted.
    # The handler uses this to detect "retrying with the same exact tool
    # call" and force a strategy change instead of burning attempts.
    attempt_fingerprints: list[str] = field(default_factory=list)
    parent_id: str | None = None
    created_at: float = field(default_factory=time.time)
    fulfilled_at: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "verify": self.verify.to_dict(),
            "status": self.status.value,
            "evidence": self.evidence,
            "attempts": self.attempts,
            "max_attempts": self.max_attempts,
            "attempt_fingerprints": list(self.attempt_fingerprints),
            "children": [c.to_dict() for c in self.children],
            "parent_id": self.parent_id,
            "created_at": self.created_at,
            "fulfilled_at": self.fulfilled_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Promise:
        return cls(
            id=d.get("id") or str(uuid.uuid4()),
            description=d.get("description", ""),
            verify=Verification.from_dict(d["verify"]) if d.get("verify") else Verification.always(),
            status=PromiseStatus(d.get("status", PromiseStatus.PENDING.value)),
            evidence=d.get("evidence"),
            attempts=d.get("attempts", 0),
            max_attempts=d.get("max_attempts", 3),
            attempt_fingerprints=list(d.get("attempt_fingerprints") or []),
            children=[cls.from_dict(c) for c in (d.get("children") or [])],
            parent_id=d.get("parent_id"),
            created_at=d.get("created_at") or time.time(),
            fulfilled_at=d.get("fulfilled_at"),
            metadata=dict(d.get("metadata") or {}),
        )


@dataclass
class PromiseContext:
    """Scoped context passed to the caller's ``act_fn`` for each attempt.

    The act_fn receives the current promise plus the whole mission (for
    rendering the mission log) and the list of ancestor promises (for
    decomposition scoping).
    """

    current: Promise
    mission: list[Promise]
    ancestors: list[Promise] = field(default_factory=list)


@dataclass
class ActEvent:
    """Event emitted by ``act_fn`` back to the runner.

    - ``PROGRESS``: informational — passed through to the caller stream
      (tool call metadata, partial output, TTS chunks).
    - ``ATTEMPT_COMPLETE``: this attempt is done; ``evidence`` holds the
      observed outcome. Runner will now invoke the verifier.
    - ``NEEDS_DECOMPOSITION``: payload is ``list[Promise]`` to install as
      children. The parent reverts to pending and will be re-picked
      after all children fulfill.
    - ``CANNOT_FULFILL``: give up on this promise. Runner marks rejected
      immediately, no retry.
    """

    kind: ActEventKind
    payload: Any = None
    evidence: str | None = None


@dataclass
class RunnerEvent:
    """Event emitted by the runner for the caller to stream."""

    kind: RunnerEventKind
    promise: Promise | None = None
    payload: Any = None
