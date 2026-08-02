"""Schema validation tests for the game-agent log entries."""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter, ValidationError

from augmentum.game_agent.schema import (
    EventPayload,
    LogEntry,
    PlanAction,
    PlanPayload,
    SessionPayload,
    SurfaceCapsPayload,
)

_Adapter = TypeAdapter(LogEntry)


def test_session_entry_validates() -> None:
    """@example: a well-formed session header round-trips."""

    raw = {
        "t": 0,
        "kind": "session",
        "payload": {
            "session_id": "s_test",
            "surface": "mock",
            "objective": "do the thing",
            "schema_version": "game_agent.v1",
            "started_at_unix_ms": 1_700_000_000_000,
        },
    }
    entry = _Adapter.validate_python(raw)
    assert entry.kind == "session"
    assert isinstance(entry.payload, SessionPayload)
    assert entry.payload.objective == "do the thing"


def test_kind_discriminator_routes_correctly() -> None:
    """@example: ``kind`` selects the right payload subclass."""

    raw = {
        "t": 1000,
        "kind": "event",
        "payload": {"channel": "log", "data": {"hello": "world"}},
    }
    entry = _Adapter.validate_python(raw)
    assert entry.kind == "event"
    assert isinstance(entry.payload, EventPayload)
    assert entry.payload.channel == "log"


def test_plan_action_bounds() -> None:
    """@example: duration_ms is bounded to [10, 2000]."""

    PlanAction(semantic="jump", duration_ms=10)
    PlanAction(semantic="jump", duration_ms=2000)
    with pytest.raises(ValidationError):
        PlanAction(semantic="jump", duration_ms=9)
    with pytest.raises(ValidationError):
        PlanAction(semantic="jump", duration_ms=2001)


def test_plan_payload_rejects_too_many_actions() -> None:
    """@example: actions list is capped at 8 (matches prompt constraint)."""

    actions = [PlanAction(semantic="a", duration_ms=50) for _ in range(9)]
    with pytest.raises(ValidationError):
        PlanPayload(
            observations=[],
            state_update="",
            actions=actions,
            confidence=0.5,
            next_check_in_ms=1000,
        )


def test_surface_caps_requires_non_empty_inputs() -> None:
    """@example: a surface with no semantic_inputs is rejected."""

    with pytest.raises(ValidationError):
        SurfaceCapsPayload(
            semantic_inputs=[],
            log_schema="mock.v1",
            observation_modalities=["log"],  # type: ignore[list-item]
        )


def test_unknown_kind_rejected() -> None:
    """@example: an entry with an unknown discriminator value is rejected."""

    with pytest.raises(ValidationError):
        _Adapter.validate_python({"t": 0, "kind": "wat", "payload": {}})
