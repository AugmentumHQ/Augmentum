"""Tests for the presence audio-history substrate (audio_history.py).

Pins:
  - append_becca_turn / append_user_turn round-trip with correct fields
  - turn_index is monotonic per (user_id, session_id), starts at 0
  - recent_window returns oldest-first within the budget
  - max_seconds budget walks newest→oldest until accumulated duration_ms hits cap
  - max_turns is a hard cap that overrides max_seconds
  - Cross-user invariant: User B sees empty list for User A's session_id
  - Cross-session invariant: same user, different session_id is isolated
  - Empty user_id / session_id raises ValueError (loud failure on scope leak)
  - Mimi token blob: serialize → store → reload → deserialize → identical ndarray
  - User turns omit mimi_tokens in v1 (column is null on disk)
  - sweep_old_turns drops rows past the retention horizon
  - clear_session is user+session scoped (doesn't bleed across users)
"""
from __future__ import annotations

import asyncio

import pytest


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture
def db():
    """In-memory SQLite with migrations applied + two seeded users."""
    from augmentum.auth.session_manager import SessionManager
    from augmentum.state.backends.sqlite import SQLiteBackend

    backend = SQLiteBackend(":memory:")
    _run(backend.connect())
    sm = SessionManager(backend._conn)
    alice = _run(sm.create_user("alice", "pw_for_alice_pls", role="user"))
    bob = _run(sm.create_user("bob", "pw_for_bob_pls", role="user"))
    yield backend, alice, bob
    _run(backend.close())


@pytest.fixture
def store(db):
    from augmentum.companion.presence.audio_history import MimiAudioHistory
    backend, _alice, _bob = db
    return MimiAudioHistory(backend._conn)


# ── Construction guards ─────────────────────────────────────────


class TestGuards:
    def test_append_requires_user_id(self, store):
        with pytest.raises(ValueError, match="user_id"):
            _run(store.append_becca_turn(
                session_id="sess_1", user_id="", transcript="hi",
            ))

    def test_append_requires_session_id(self, store, db):
        _, alice, _ = db
        with pytest.raises(ValueError, match="session_id"):
            _run(store.append_becca_turn(
                session_id="", user_id=alice.id, transcript="hi",
            ))

    def test_recent_window_requires_scope(self, store, db):
        _, alice, _ = db
        with pytest.raises(ValueError):
            _run(store.recent_window(session_id="sess_1", user_id=""))
        with pytest.raises(ValueError):
            _run(store.recent_window(session_id="", user_id=alice.id))


# ── Append + round-trip ─────────────────────────────────────────


class TestAppendRoundTrip:
    def test_becca_turn_persists(self, store, db):
        _, alice, _ = db
        turn = _run(store.append_becca_turn(
            session_id="sess_1",
            user_id=alice.id,
            transcript="Hello there.",
            duration_ms=1200,
        ))
        assert turn.turn_index == 0
        assert turn.speaker == "becca"
        assert turn.transcript == "Hello there."
        assert turn.user_id == alice.id

    def test_user_turn_persists_without_mimi(self, store, db):
        _, alice, _ = db
        turn = _run(store.append_user_turn(
            session_id="sess_1",
            user_id=alice.id,
            transcript="what's the weather?",
            duration_ms=850,
        ))
        assert turn.speaker == "user"
        assert turn.mimi_tokens is None

    def test_turn_index_is_monotonic_per_session(self, store, db):
        _, alice, _ = db
        t0 = _run(store.append_becca_turn(
            session_id="sess_1", user_id=alice.id, transcript="one",
        ))
        t1 = _run(store.append_user_turn(
            session_id="sess_1", user_id=alice.id, transcript="two",
        ))
        t2 = _run(store.append_becca_turn(
            session_id="sess_1", user_id=alice.id, transcript="three",
        ))
        assert t0.turn_index == 0
        assert t1.turn_index == 1
        assert t2.turn_index == 2

    def test_turn_index_resets_per_session(self, store, db):
        """Two sessions for the same user have independent turn_index counters."""
        _, alice, _ = db
        ta0 = _run(store.append_becca_turn(
            session_id="sess_A", user_id=alice.id, transcript="A.0",
        ))
        ta1 = _run(store.append_becca_turn(
            session_id="sess_A", user_id=alice.id, transcript="A.1",
        ))
        tb0 = _run(store.append_becca_turn(
            session_id="sess_B", user_id=alice.id, transcript="B.0",
        ))
        assert ta0.turn_index == 0
        assert ta1.turn_index == 1
        assert tb0.turn_index == 0  # independent counter


# ── recent_window ───────────────────────────────────────────────


class TestRecentWindow:
    def test_empty_session_returns_empty(self, store, db):
        _, alice, _ = db
        window = _run(store.recent_window(
            session_id="sess_unknown", user_id=alice.id,
        ))
        assert window == []

    def test_returns_oldest_first(self, store, db):
        _, alice, _ = db
        _run(store.append_becca_turn(
            session_id="sess_1", user_id=alice.id,
            transcript="first", duration_ms=500,
        ))
        _run(store.append_user_turn(
            session_id="sess_1", user_id=alice.id,
            transcript="second", duration_ms=500,
        ))
        _run(store.append_becca_turn(
            session_id="sess_1", user_id=alice.id,
            transcript="third", duration_ms=500,
        ))
        window = _run(store.recent_window(
            session_id="sess_1", user_id=alice.id,
        ))
        assert [t.transcript for t in window] == ["first", "second", "third"]

    def test_max_seconds_budget_truncates_newest_wins(self, store, db):
        """When max_seconds budget is tight, oldest turns fall off."""
        _, alice, _ = db
        # 5 turns × 2s each = 10s of total history
        for i in range(5):
            _run(store.append_becca_turn(
                session_id="sess_1", user_id=alice.id,
                transcript=f"turn-{i}", duration_ms=2000,
            ))
        # Ask for 3s budget — should get the most recent 2 turns
        # (newest-first walk accumulates 2s, then 4s ≥ 3s budget stops there)
        window = _run(store.recent_window(
            session_id="sess_1", user_id=alice.id, max_seconds=3.0,
        ))
        # Walking newest-first: turn-4 (2s), turn-3 (4s ≥ 3s → stop).
        # Returned oldest-first.
        assert [t.transcript for t in window] == ["turn-3", "turn-4"]

    def test_max_turns_hard_cap(self, store, db):
        _, alice, _ = db
        for i in range(10):
            _run(store.append_becca_turn(
                session_id="sess_1", user_id=alice.id,
                transcript=f"t-{i}", duration_ms=100,
            ))
        window = _run(store.recent_window(
            session_id="sess_1", user_id=alice.id, max_turns=3,
        ))
        # Last 3 turns, oldest-first
        assert [t.transcript for t in window] == ["t-7", "t-8", "t-9"]


# ── Multi-tenant invariants ─────────────────────────────────────


class TestMultiTenant:
    def test_user_b_cannot_see_user_a_session(self, store, db):
        """Cross-user invariant — same session_id is isolated per user."""
        _, alice, bob = db
        _run(store.append_becca_turn(
            session_id="sess_shared", user_id=alice.id,
            transcript="alice secret",
        ))
        # Bob asking for the same session_id sees nothing
        window = _run(store.recent_window(
            session_id="sess_shared", user_id=bob.id,
        ))
        assert window == []

    def test_turn_count_user_scoped(self, store, db):
        _, alice, bob = db
        for _ in range(3):
            _run(store.append_becca_turn(
                session_id="sess_1", user_id=alice.id, transcript="x",
            ))
        for _ in range(5):
            _run(store.append_becca_turn(
                session_id="sess_1", user_id=bob.id, transcript="y",
            ))
        assert _run(store.turn_count(
            session_id="sess_1", user_id=alice.id,
        )) == 3
        assert _run(store.turn_count(
            session_id="sess_1", user_id=bob.id,
        )) == 5

    def test_clear_session_user_scoped(self, store, db):
        """Bob clearing sess_1 must NOT clear Alice's sess_1."""
        _, alice, bob = db
        _run(store.append_becca_turn(
            session_id="sess_1", user_id=alice.id, transcript="alice",
        ))
        _run(store.append_becca_turn(
            session_id="sess_1", user_id=bob.id, transcript="bob",
        ))
        dropped = _run(store.clear_session(
            session_id="sess_1", user_id=bob.id,
        ))
        assert dropped == 1
        # Alice still has hers
        assert _run(store.turn_count(
            session_id="sess_1", user_id=alice.id,
        )) == 1


# ── Mimi token serialization ────────────────────────────────────


class TestMimiSerialization:
    def test_serialize_deserialize_roundtrip(self):
        import numpy as np
        from augmentum.companion.presence.audio_history import (
            deserialize_mimi_tokens,
            serialize_mimi_tokens,
        )
        # 8 codebooks × 125 frames = 1 second at Mimi rate
        original = np.random.randint(
            0, 1024, size=(8, 125), dtype=np.int16,
        )
        blob = serialize_mimi_tokens(original)
        assert isinstance(blob, bytes)
        assert len(blob) > 0

        restored = deserialize_mimi_tokens(blob)
        np.testing.assert_array_equal(original, restored)

    def test_serialize_requires_2d(self):
        import numpy as np
        from augmentum.companion.presence.audio_history import (
            serialize_mimi_tokens,
        )
        with pytest.raises(ValueError, match="2D"):
            serialize_mimi_tokens(np.zeros(100, dtype=np.int16))

    def test_stored_mimi_blob_round_trips_via_store(self, store, db):
        """Future capture path: when tokens are supplied, they store + reload."""
        import numpy as np
        from augmentum.companion.presence.audio_history import (
            deserialize_mimi_tokens,
            serialize_mimi_tokens,
        )
        _, alice, _ = db
        tokens = np.random.randint(
            0, 1024, size=(8, 250), dtype=np.int16,
        )
        blob = serialize_mimi_tokens(tokens)
        _run(store.append_becca_turn(
            session_id="sess_1", user_id=alice.id,
            transcript="with tokens", mimi_tokens=blob,
            duration_ms=2000,
        ))
        window = _run(store.recent_window(
            session_id="sess_1", user_id=alice.id,
        ))
        assert len(window) == 1
        assert window[0].mimi_tokens == blob
        restored = deserialize_mimi_tokens(window[0].mimi_tokens)
        np.testing.assert_array_equal(tokens, restored)


# ── Retention sweep ─────────────────────────────────────────────


class TestRetentionSweep:
    def test_sweep_drops_only_old_rows(self, store, db):
        _, alice, _ = db
        # Insert a row with a manually-set old timestamp
        _run(store.append_becca_turn(
            session_id="sess_1", user_id=alice.id, transcript="fresh",
        ))
        _run(store._conn.execute(
            "INSERT INTO companion_audio_history "
            "(id, user_id, session_id, turn_index, speaker, transcript, "
            " duration_ms, created_at) "
            "VALUES ('ancient_id', ?, 'sess_old', 0, 'becca', 'ancient', 0, "
            " datetime('now', '-60 days'))",
            (alice.id,),
        ))
        _run(store._conn.commit())

        dropped = _run(store.sweep_old_turns(retention_days=30))
        assert dropped == 1
        # Fresh row survives
        assert _run(store.turn_count(
            session_id="sess_1", user_id=alice.id,
        )) == 1

    def test_sweep_rejects_invalid_retention(self, store):
        with pytest.raises(ValueError, match="positive"):
            _run(store.sweep_old_turns(retention_days=0))
        with pytest.raises(ValueError, match="positive"):
            _run(store.sweep_old_turns(retention_days=-5))
