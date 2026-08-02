"""Promise-based mission runtime for agentic modes.

A Promise is a handle to a future state transition with an explicit
verification contract. Missions are lists of Promises that drive an
agent toward a goal. The ``MissionRunner`` is domain-neutral: it owns
control flow (next-pending, verify, retry, decompose), while callers
plug in an ``act_fn`` (how to attempt a promise) and verifier
implementations (how to check a promise is fulfilled).

This runtime backs the Coder mode's multi-step execution and is
designed for reuse by the Voice controller and any future agentic
surface.
"""
from __future__ import annotations

from augmentum.promises.models import (
    ActEvent,
    ActEventKind,
    Promise,
    PromiseContext,
    PromiseStatus,
    RunnerEvent,
    RunnerEventKind,
    Verification,
    VerificationKind,
)
from augmentum.promises.parse import parse_mission_json, parse_prose_plan
from augmentum.promises.render import render_mission_log
from augmentum.promises.runner import MissionRunner
from augmentum.promises.verify import (
    VerifyFn,
    VerifyResult,
    always_pass,
    file_verifier,
    shell_verifier,
)

__all__ = [
    "ActEvent",
    "ActEventKind",
    "MissionRunner",
    "Promise",
    "PromiseContext",
    "PromiseStatus",
    "RunnerEvent",
    "RunnerEventKind",
    "Verification",
    "VerificationKind",
    "VerifyFn",
    "VerifyResult",
    "always_pass",
    "file_verifier",
    "parse_mission_json",
    "parse_prose_plan",
    "render_mission_log",
    "shell_verifier",
]
