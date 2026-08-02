"""LiveLog write + read + validate round-trips."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from augmentum.game_agent.log import LiveLog, SessionClock, read_log, validate_entries


def test_session_clock_starts_at_zero() -> None:
    """@example: elapsed before start raises."""

    clock = SessionClock()
    with pytest.raises(RuntimeError):
        clock.elapsed_ms()


def test_session_clock_monotonic(tmp_path: Path) -> None:
    """@example: elapsed_ms is non-decreasing across two reads."""

    clock = SessionClock()
    clock.start()
    a = clock.elapsed_ms()
    b = clock.elapsed_ms()
    assert b >= a >= 0


def test_live_log_writes_ndjson_and_reads_back(tmp_path: Path) -> None:
    """@example: appended entries are valid NDJSON we can stream back."""

    path = tmp_path / "session.ndjson"
    with LiveLog(path) as live:
        live.append(
            {
                "t": 0,
                "kind": "session",
                "payload": {
                    "session_id": "s1",
                    "surface": "mock",
                    "objective": "x",
                    "schema_version": "game_agent.v1",
                    "started_at_unix_ms": 1,
                },
            }
        )
        live.append(
            {
                "t": 10,
                "kind": "event",
                "payload": {"channel": "log", "data": {"hello": "world"}},
            }
        )

    entries = list(read_log(path))
    assert len(entries) == 2
    assert entries[0]["kind"] == "session"
    assert entries[1]["payload"]["data"]["hello"] == "world"
    # Every line is exactly one JSON object terminated by a newline.
    lines = path.read_text().splitlines()
    assert all(json.loads(line)["kind"] in ("session", "event") for line in lines)


def test_live_log_rejects_invalid_entry(tmp_path: Path) -> None:
    """@example: a bad payload raises and is not written."""

    path = tmp_path / "session.ndjson"
    live = LiveLog(path)
    try:
        with pytest.raises(ValidationError):
            live.append({"t": 0, "kind": "session", "payload": {"missing": "fields"}})
        # File exists (opened) but should be empty -- the bad write was rejected.
        live.flush()
        assert path.read_text() == ""
    finally:
        live.close()


def test_validate_entries_passes_for_clean_log(tmp_path: Path) -> None:
    """@example: a clean log round-trips through validate_entries."""

    path = tmp_path / "ok.ndjson"
    with LiveLog(path) as live:
        live.append(
            {
                "t": 0,
                "kind": "session",
                "payload": {
                    "session_id": "s",
                    "surface": "mock",
                    "objective": "ok",
                    "schema_version": "game_agent.v1",
                    "started_at_unix_ms": 1,
                },
            }
        )
    validated = validate_entries(read_log(path))
    assert len(validated) == 1
