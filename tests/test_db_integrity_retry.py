"""Unit tests for ``_run_quick_check_with_retry``.

Covers:
  * Clean DB returns ``["ok"]`` immediately.
  * Transient errors (disk i/o, locked, busy) trigger backoff retry.
  * Recovery after one transient error logs ``db_integrity_check_recovered``.
  * Non-transient errors propagate without consuming retry budget.
  * All retries exhausted re-raises the last transient error.
  * Sleep is dependency-injected so tests run instantly (no real wait).
  * Custom backoff schedules honored.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from augmentum.proxy.server import (
    _TRANSIENT_SQLITE_FRAGMENTS,
    _run_quick_check_with_retry,
)


class _FakeLogger:
    def __init__(self) -> None:
        self.info_calls: list[tuple[str, dict[str, Any]]] = []
        self.warning_calls: list[tuple[str, dict[str, Any]]] = []

    def info(self, event: str, **kw: Any) -> None:
        self.info_calls.append((event, kw))

    def warning(self, event: str, **kw: Any) -> None:
        self.warning_calls.append((event, kw))


@pytest.fixture
def clean_db(tmp_path: Path) -> str:
    """A fresh, valid SQLite DB file. PRAGMA quick_check returns ['ok']."""
    p = tmp_path / "clean.db"
    conn = sqlite3.connect(str(p))
    conn.execute("CREATE TABLE t (x INTEGER)")
    conn.execute("INSERT INTO t VALUES (1)")
    conn.commit()
    conn.close()
    return str(p)


def test_clean_db_returns_ok(clean_db: str) -> None:
    sleeps: list[float] = []
    logger = _FakeLogger()
    rows = _run_quick_check_with_retry(
        clean_db, sleep=sleeps.append, logger=logger,
    )
    assert rows == ["ok"]
    assert sleeps == []  # no retries needed
    assert logger.info_calls == []  # nothing surprising to log


@pytest.mark.parametrize("transient_msg", list(_TRANSIENT_SQLITE_FRAGMENTS))
def test_transient_errors_are_retried(clean_db: str, transient_msg: str) -> None:
    """Every classified transient fragment triggers retry, not raise."""
    sleeps: list[float] = []
    logger = _FakeLogger()

    real_connect = sqlite3.connect
    call_count = {"n": 0}

    def flaky_connect(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise sqlite3.OperationalError(transient_msg)
        return real_connect(*args, **kwargs)

    with patch("sqlite3.connect", side_effect=flaky_connect):
        rows = _run_quick_check_with_retry(
            clean_db, sleep=sleeps.append, logger=logger,
        )

    assert rows == ["ok"]
    assert sleeps == [2.0]  # one retry consumed first backoff slot
    # Recovery is logged, with the prior error attached.
    assert any(
        e == "db_integrity_check_recovered" for e, _ in logger.info_calls
    )
    assert any(
        e == "db_integrity_check_transient_retry" for e, _ in logger.info_calls
    )


def test_two_transient_then_success(clean_db: str) -> None:
    """Two consecutive transient errors consume both backoff slots,
    then success on the third attempt — full retry budget exercised."""
    sleeps: list[float] = []
    logger = _FakeLogger()
    real_connect = sqlite3.connect
    call_count = {"n": 0}

    def flaky_connect(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] <= 2:
            raise sqlite3.OperationalError("disk I/O error")
        return real_connect(*args, **kwargs)

    with patch("sqlite3.connect", side_effect=flaky_connect):
        rows = _run_quick_check_with_retry(
            clean_db, sleep=sleeps.append, logger=logger,
        )

    assert rows == ["ok"]
    assert sleeps == [2.0, 8.0]  # both backoff slots used
    assert call_count["n"] == 3  # three connect attempts


def test_all_retries_exhausted_raises(tmp_path: Path) -> None:
    """When every attempt sees a transient error, the last one is
    re-raised — caller (the integrity loop) downgrades to a warning."""
    sleeps: list[float] = []
    logger = _FakeLogger()

    def always_fails(*_args, **_kwargs):
        raise sqlite3.OperationalError("disk I/O error")

    with patch("sqlite3.connect", side_effect=always_fails), \
         pytest.raises(sqlite3.OperationalError, match="disk I/O"):
        _run_quick_check_with_retry(
            str(tmp_path / "x.db"),
            sleep=sleeps.append,
            logger=logger,
        )

    # 3 attempts, 2 sleeps in between. The third attempt fails too,
    # at which point we raise rather than sleeping again.
    assert sleeps == [2.0, 8.0]


def test_non_transient_error_propagates_immediately(tmp_path: Path) -> None:
    """``no such table`` / syntax errors / etc are not transient — they
    indicate a real problem, not a flake. Don't retry; just raise."""
    sleeps: list[float] = []
    logger = _FakeLogger()

    def schema_error(*_args, **_kwargs):
        raise sqlite3.OperationalError("no such table: bogus")

    with patch("sqlite3.connect", side_effect=schema_error), \
         pytest.raises(sqlite3.OperationalError, match="no such table"):
        _run_quick_check_with_retry(
            str(tmp_path / "x.db"),
            sleep=sleeps.append,
            logger=logger,
        )

    assert sleeps == [], (
        "non-transient errors must not consume any retry budget"
    )


def test_custom_backoff_schedule(clean_db: str) -> None:
    """Backoff is parametrised so callers can tune for their workload."""
    sleeps: list[float] = []
    real_connect = sqlite3.connect
    call_count = {"n": 0}

    def flaky_connect(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise sqlite3.OperationalError("database is locked")
        return real_connect(*args, **kwargs)

    with patch("sqlite3.connect", side_effect=flaky_connect):
        _run_quick_check_with_retry(
            clean_db,
            backoff_schedule=(0.5, 1.5, 4.0),
            sleep=sleeps.append,
        )

    assert sleeps == [0.5]


def test_no_logger_argument_works_silently(clean_db: str) -> None:
    """Passing ``logger=None`` must not raise — caller may not have
    structlog wired (e.g. a maintenance script)."""
    sleeps: list[float] = []
    real_connect = sqlite3.connect
    call_count = {"n": 0}

    def flaky_connect(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise sqlite3.OperationalError("database is busy")
        return real_connect(*args, **kwargs)

    with patch("sqlite3.connect", side_effect=flaky_connect):
        rows = _run_quick_check_with_retry(
            clean_db, sleep=sleeps.append, logger=None,
        )

    assert rows == ["ok"]
