"""External agentic-coder drivers (Claude Code, Codex).

A shared backend so the companion (autonomous) AND the user (direct, via the
coder UI) can drive an external coding agent that runs inside a sandboxed coder
workspace container — supervised, consent-gated, budget-capped — with outcomes
recorded to the companion's engineering-continuity ledger.

See docs/superpowers/specs/2026-06-21-companion-external-coder-drivers-design.md.
Slice 1 (this package): the engine core — driver ABC + normalized event model +
ClaudeCodeDriver + registry. Live container spawn / companion verb / UI selector
are wired in later slices (need on-device auth + container verification).
"""

from __future__ import annotations

from augmentum.coder.external.base import (
    CoderEvent,
    ExternalCoderDriver,
    ExternalRunResult,
    ExternalTask,
    to_engineering_record,
)
from augmentum.coder.external.claude_auth import (
    SETUP_TOKEN_CMD,
    auth_env,
    parse_auth_url,
    parse_setup_token,
)
from augmentum.coder.external.registry import available_drivers, select_driver

__all__ = [
    "SETUP_TOKEN_CMD",
    "CoderEvent",
    "ExternalCoderDriver",
    "ExternalRunResult",
    "ExternalTask",
    "auth_env",
    "available_drivers",
    "parse_auth_url",
    "parse_setup_token",
    "select_driver",
    "to_engineering_record",
]
