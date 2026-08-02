"""Engineering continuity ledger — write → recall round-trip.

Verifies the persistence loop that lets a stateless coding agent feel
continuous: a completed run is recorded into companion_journal and resurfaces
as a prompt-ready line next session. Mirrors the commitments.py pattern (zero
new tables). Light scaffolding — a minimal in-memory companion_journal + a fake
runtime/memory — so no migration machinery is needed.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import aiosqlite
import pytest

from augmentum.companion_runtime import engineering_log as el

pytestmark = pytest.mark.asyncio


def _iso(days_ago: float = 0.0) -> str:
    ts = datetime.now(UTC) - timedelta(days=days_ago)
    return ts.strftime("%Y-%m-%d %H:%M:%S")


class _FakeBackend:
    def __init__(self, conn):
        self.conn = conn


class _FakeSM:
    def __init__(self, conn):
        self.backend = _FakeBackend(conn)


class _FakeMemory:
    """Stand-in for runtime.memory — safe_journal just inserts a row."""
    def __init__(self, conn):
        self._conn = conn

    async def safe_journal(self, content, *, source, user_id, entry_type,
                           embed=False, origin=None):
        cur = await self._conn.execute(
            "INSERT INTO companion_journal "
            "(companion_id, user_id, entry_type, content, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("becca", user_id, entry_type, content, _iso(0)),
        )
        await self._conn.commit()
        return cur.lastrowid


class _FakeRuntime:
    companion_id = "becca"

    def __init__(self, conn):
        self.state_manager = _FakeSM(conn)
        self.memory = _FakeMemory(conn)


async def _make_conn():
    conn = await aiosqlite.connect(":memory:")
    await conn.execute(
        "CREATE TABLE companion_journal ("
        " id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " companion_id TEXT, user_id TEXT, entry_type TEXT,"
        " content TEXT, created_at TEXT)"
    )
    await conn.commit()
    return conn


async def _insert(conn, *, user_id, content, days_ago=0.0,
                  entry_type=el.ENTRY_TYPE, companion_id="becca"):
    await conn.execute(
        "INSERT INTO companion_journal "
        "(companion_id, user_id, entry_type, content, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (companion_id, user_id, entry_type, content, _iso(days_ago)),
    )
    await conn.commit()


async def test_record_then_recall_roundtrip():
    conn = await _make_conn()
    rt = _FakeRuntime(conn)
    eid = await el.record_engineering_outcome(
        rt, user_id="u_a", task="refactor the media store",
        outcome="landed, tests green", engine="Claude Code",
        framing="the cast images were slow",
    )
    assert eid is not None
    lines = await el.recent_engineering(rt, user_id="u_a")
    assert len(lines) == 1
    line = lines[0]
    assert "refactor the media store" in line
    assert "Claude Code" in line          # engine label
    assert "landed, tests green" in line  # outcome
    assert "cast images were slow" in line  # the user's framing
    await conn.close()


async def test_recall_skips_stale_past_ttl():
    conn = await _make_conn()
    rt = _FakeRuntime(conn)
    await _insert(conn, user_id="u_a", content="fresh work", days_ago=1)
    await _insert(conn, user_id="u_a", content="ancient work", days_ago=20)
    lines = await el.recent_engineering(rt, user_id="u_a", limit=5)
    joined = " ".join(lines)
    assert "fresh work" in joined
    assert "ancient work" not in joined  # > 14d TTL
    await conn.close()


async def test_recall_is_user_scoped():
    conn = await _make_conn()
    rt = _FakeRuntime(conn)
    await _insert(conn, user_id="u_a", content="A's project")
    await _insert(conn, user_id="u_b", content="B's project")
    a_lines = await el.recent_engineering(rt, user_id="u_a", limit=5)
    assert any("A's project" in line for line in a_lines)
    assert not any("B's project" in line for line in a_lines)
    await conn.close()


async def test_recall_respects_limit_newest_first():
    conn = await _make_conn()
    rt = _FakeRuntime(conn)
    await _insert(conn, user_id="u_a", content="oldest", days_ago=3)
    await _insert(conn, user_id="u_a", content="middle", days_ago=2)
    await _insert(conn, user_id="u_a", content="newest", days_ago=1)
    lines = await el.recent_engineering(rt, user_id="u_a", limit=2)
    assert len(lines) == 2
    assert any("newest" in line for line in lines)
    assert any("middle" in line for line in lines)
    assert not any("oldest" in line for line in lines)
    await conn.close()


async def test_native_engine_has_no_engine_label():
    conn = await _make_conn()
    rt = _FakeRuntime(conn)
    await el.record_engineering_outcome(
        rt, user_id="u_a", task="add a setting", engine="",
    )
    line = (await el.recent_engineering(rt, user_id="u_a"))[0]
    assert "add a setting" in line
    assert "We had work on" in line  # no engine label injected
    await conn.close()


async def test_record_noops_without_task():
    conn = await _make_conn()
    rt = _FakeRuntime(conn)
    assert await el.record_engineering_outcome(rt, user_id="u_a", task="") is None
    assert await el.recent_engineering(rt, user_id="u_a") == []
    await conn.close()
