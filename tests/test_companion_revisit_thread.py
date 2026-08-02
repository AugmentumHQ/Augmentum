"""Tests for Piece 9' — the revisit_thread candidate.

Becca proactively picks up unresolved journal threads, runs the
resolver against them, writes findings back as a noticing. This is
the first autonomous use of the resolver capability — she queries
without being asked.

Resource gates verified here (each one prevents a resolver-call loop):
* Daily cap (count of revisited_at within 24h)
* Per-thread cooldown (revisited_at column on the thread row)
* Queue-presence gate at score time
* Primary-busy / user-recent / hush gates at score time
* No user_id on thread → skip (can't tenant-scope retrieval)
"""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


async def _boot_backend():
    from augmentum.state.backends.sqlite import SQLiteBackend

    backend = SQLiteBackend(":memory:")
    await backend.connect()
    # Insert a user so user_id-not-null thread filter has a real target
    await backend.conn.execute(
        "INSERT INTO users (id, username, password_hash, created_at) "
        "VALUES (?, ?, ?, datetime('now'))",
        ("usr_test", "tester", "x"),
    )
    await backend.conn.commit()
    return backend


def _fake_runtime(backend, *, role: str = "self"):
    """Minimal runtime that activity_selector sees."""
    runtime = MagicMock()
    runtime.backend = backend
    runtime.companion_id = "becca"
    # Real string owner so the owner-scope clause is a pass-through ("")
    # rather than binding a MagicMock into SQL (audit 2026-06-17).
    runtime.owner_user_id = ""
    runtime.user_cooldown_until = 0.0
    runtime.heavy_quiet_until = 0.0
    runtime.last_initiative_score_at = 0.0
    runtime._app_state = MagicMock()
    runtime._app_state.file_index = None  # Use journal-only retrieval for tests
    runtime._app_state.llama_manager = None  # not busy

    # Real CompanionMemory on the test backend so journal writes work.
    from augmentum.companion_runtime.memory import CompanionMemory
    runtime.memory = CompanionMemory(backend, "becca")
    return runtime


async def _insert_thread(backend, *, content: str, user_id: str = "usr_test",
                         entry_type: str = "wondering", revisited_at=None):
    """Insert a single journal row for testing."""
    if revisited_at:
        await backend.conn.execute(
            "INSERT INTO companion_journal "
            "(companion_id, user_id, entry_type, content, revisited_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("becca", user_id, entry_type, content, revisited_at),
        )
    else:
        await backend.conn.execute(
            "INSERT INTO companion_journal "
            "(companion_id, user_id, entry_type, content) "
            "VALUES (?, ?, ?, ?)",
            ("becca", user_id, entry_type, content),
        )
    await backend.conn.commit()


async def _insert_pending_proposal(backend, kind: str = "revisit_thread"):
    """Insert a queue row so the score gate finds something."""
    await backend.conn.execute(
        "INSERT INTO companion_initiative_queue "
        "(companion_id, proposed_at, kind, payload, score, status) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("becca", time.time(), kind, "{}", 0.65, "pending"),
    )
    await backend.conn.commit()


# ── _score_revisit_thread ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_score_low_when_no_queue():
    """No pending proposals → low score, regardless of role."""
    from augmentum.companion_runtime.behavior import activity_selector
    backend = await _boot_backend()
    runtime = _fake_runtime(backend)
    score = await activity_selector._score_revisit_thread(runtime, role="self")
    assert score == 0.05


@pytest.mark.asyncio
async def test_score_high_when_queue_pending():
    from augmentum.companion_runtime.behavior import activity_selector
    backend = await _boot_backend()
    runtime = _fake_runtime(backend)
    await _insert_pending_proposal(backend)
    score = await activity_selector._score_revisit_thread(runtime, role="self")
    assert score >= 0.70


@pytest.mark.asyncio
async def test_score_low_when_collaborator_role():
    from augmentum.companion_runtime.behavior import activity_selector
    backend = await _boot_backend()
    runtime = _fake_runtime(backend)
    await _insert_pending_proposal(backend)
    score = await activity_selector._score_revisit_thread(runtime, role="collaborator")
    assert score == 0.05


@pytest.mark.asyncio
async def test_score_low_when_primary_busy():
    from augmentum.companion_runtime.behavior import activity_selector
    backend = await _boot_backend()
    runtime = _fake_runtime(backend)
    await _insert_pending_proposal(backend)

    # Simulate primary busy by patching the gate
    with patch(
        "augmentum.companion_runtime.gates.is_primary_busy",
        return_value=True,
    ):
        score = await activity_selector._score_revisit_thread(runtime, role="self")
    assert score == 0.05


@pytest.mark.asyncio
async def test_score_excluded_proposals_dont_count():
    """Executed proposals must not keep returning a high score."""
    from augmentum.companion_runtime.behavior import activity_selector
    backend = await _boot_backend()
    runtime = _fake_runtime(backend)
    # Insert executed proposal — should not count
    await backend.conn.execute(
        "INSERT INTO companion_initiative_queue "
        "(companion_id, proposed_at, kind, payload, score, status) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("becca", time.time(), "revisit_thread", "{}", 0.65, "executed"),
    )
    await backend.conn.commit()
    score = await activity_selector._score_revisit_thread(runtime, role="self")
    assert score == 0.05


# ── _perform_revisit_thread ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_perform_skips_when_no_thread():
    """No unresolved threads → mark proposal executed, no journal write."""
    from augmentum.companion_runtime.behavior import activity_selector
    backend = await _boot_backend()
    runtime = _fake_runtime(backend)
    await _insert_pending_proposal(backend)

    await activity_selector._perform_revisit_thread(runtime)

    # Proposal marked executed
    cur = await backend.conn.execute(
        "SELECT status FROM companion_initiative_queue WHERE kind = 'revisit_thread'"
    )
    row = await cur.fetchone()
    await cur.close()
    assert row[0] == "executed"

    # No new journal entries (just the one we'd inserted if any)
    cur = await backend.conn.execute(
        "SELECT COUNT(*) FROM companion_journal"
    )
    assert (await cur.fetchone())[0] == 0


@pytest.mark.asyncio
async def test_perform_skips_when_daily_cap_hit():
    """6 revisits in last 24h → skip, no resolver call."""
    from augmentum.companion_runtime.behavior import activity_selector
    backend = await _boot_backend()
    runtime = _fake_runtime(backend)
    await _insert_pending_proposal(backend)

    # Insert 6 journal entries with recent revisited_at
    for i in range(6):
        await backend.conn.execute(
            "INSERT INTO companion_journal "
            "(companion_id, user_id, entry_type, content, revisited_at) "
            "VALUES (?, ?, ?, ?, datetime('now'))",
            ("becca", "usr_test", "wondering", f"thread {i}"),
        )
    await backend.conn.commit()

    # Insert a fresh unresolved thread
    await _insert_thread(backend, content="should not be resolved")

    # Patch resolver so we can assert it was NOT called
    with patch("augmentum.resolver.resolve_moments") as resolver:
        resolver_mock = AsyncMock()
        resolver.side_effect = resolver_mock

        await activity_selector._perform_revisit_thread(runtime)

        resolver_mock.assert_not_called()


@pytest.mark.asyncio
async def test_perform_skips_threads_within_cooldown():
    """Thread revisited 2h ago (under 6h cooldown) → not picked."""
    from augmentum.companion_runtime.behavior import activity_selector
    backend = await _boot_backend()
    runtime = _fake_runtime(backend)
    await _insert_pending_proposal(backend)

    # Insert one thread revisited 2h ago (within cooldown)
    await backend.conn.execute(
        "INSERT INTO companion_journal "
        "(companion_id, user_id, entry_type, content, revisited_at) "
        "VALUES (?, ?, ?, ?, datetime('now', '-2 hours'))",
        ("becca", "usr_test", "wondering", "recent revisit"),
    )
    await backend.conn.commit()

    await activity_selector._perform_revisit_thread(runtime)

    # No new noticing should have been written
    cur = await backend.conn.execute(
        "SELECT COUNT(*) FROM companion_journal WHERE entry_type = 'noticing'"
    )
    assert (await cur.fetchone())[0] == 0


@pytest.mark.asyncio
async def test_perform_skips_suppressed_threads():
    from augmentum.companion_runtime.behavior import activity_selector
    backend = await _boot_backend()
    runtime = _fake_runtime(backend)
    await _insert_pending_proposal(backend)
    await backend.conn.execute(
        "INSERT INTO companion_journal "
        "(companion_id, user_id, entry_type, content, suppressed) "
        "VALUES (?, ?, ?, ?, 1)",
        ("becca", "usr_test", "wondering", "self-corrected"),
    )
    await backend.conn.commit()

    await activity_selector._perform_revisit_thread(runtime)

    # No noticing was written (only suppressed thread existed)
    cur = await backend.conn.execute(
        "SELECT COUNT(*) FROM companion_journal WHERE entry_type = 'noticing'"
    )
    assert (await cur.fetchone())[0] == 0


@pytest.mark.asyncio
async def test_perform_skips_threads_without_user_id():
    """user_id IS NULL → can't tenant-scope resolver, skip."""
    from augmentum.companion_runtime.behavior import activity_selector
    backend = await _boot_backend()
    runtime = _fake_runtime(backend)
    await _insert_pending_proposal(backend)
    # Insert thread with NULL user_id
    await backend.conn.execute(
        "INSERT INTO companion_journal "
        "(companion_id, user_id, entry_type, content) "
        "VALUES (?, NULL, ?, ?)",
        ("becca", "wondering", "no user attached"),
    )
    await backend.conn.commit()

    await activity_selector._perform_revisit_thread(runtime)

    cur = await backend.conn.execute(
        "SELECT COUNT(*) FROM companion_journal WHERE entry_type = 'noticing'"
    )
    assert (await cur.fetchone())[0] == 0


@pytest.mark.asyncio
async def test_perform_happy_path_writes_noticing():
    """An eligible thread gets resolved, noticing is written with
    content_refs, original thread is marked revisited."""
    from augmentum.companion_runtime.behavior import activity_selector
    from augmentum.resolver.core import Moment

    backend = await _boot_backend()
    runtime = _fake_runtime(backend)
    await _insert_pending_proposal(backend)
    await _insert_thread(backend, content="that document I read last week")

    # Patch resolver to return one moment
    fake_moment = Moment(
        id="fi_42",
        kind="file",
        score=0.5,
        snippet="found document about thing",
        title="The Document",
        created_at="2026-05-19",
        content_refs=[],
        legs=["file_vec"],
        raw={},
    )
    with patch(
        "augmentum.resolver.resolve_moments",
        new=AsyncMock(return_value=[fake_moment]),
    ):
        await activity_selector._perform_revisit_thread(runtime)

    # A noticing was written
    cur = await backend.conn.execute(
        "SELECT content, content_refs FROM companion_journal "
        "WHERE entry_type = 'noticing'"
    )
    row = await cur.fetchone()
    await cur.close()
    assert row is not None
    # Field-note prose, not the legacy "Came back to: X\nFound: Y" log
    # template — pin the new shape.
    assert "looped back to" in row[0]
    assert "The Document" in row[0]
    # content_refs is JSON-encoded
    import json
    refs = json.loads(row[1])
    assert any(r.get("id") == "fi_42" for r in refs)

    # Original thread's revisited_at was set
    cur = await backend.conn.execute(
        "SELECT revisited_at FROM companion_journal "
        "WHERE entry_type = 'wondering' AND content = ?",
        ("that document I read last week",),
    )
    row = await cur.fetchone()
    await cur.close()
    assert row[0] is not None

    # Proposal marked executed
    cur = await backend.conn.execute(
        "SELECT status FROM companion_initiative_queue WHERE kind = 'revisit_thread'"
    )
    row = await cur.fetchone()
    await cur.close()
    assert row[0] == "executed"


@pytest.mark.asyncio
async def test_perform_with_no_results_still_marks_thread():
    """Resolver returns 0 moments → still write a 'nothing surfaced'
    noticing and mark thread revisited, so we don't re-fire on it."""
    from augmentum.companion_runtime.behavior import activity_selector

    backend = await _boot_backend()
    runtime = _fake_runtime(backend)
    await _insert_pending_proposal(backend)
    await _insert_thread(backend, content="totally unknown reference")

    with patch(
        "augmentum.resolver.resolve_moments",
        new=AsyncMock(return_value=[]),
    ):
        await activity_selector._perform_revisit_thread(runtime)

    # Noticing was still written
    cur = await backend.conn.execute(
        "SELECT content FROM companion_journal WHERE entry_type = 'noticing'"
    )
    row = await cur.fetchone()
    await cur.close()
    assert row is not None
    # Honest-silence prose: "Letting <thread> sit for now."
    assert row[0].startswith("Letting ")
    assert "sit for now" in row[0]

    # Thread revisited_at set so we don't re-fire
    cur = await backend.conn.execute(
        "SELECT revisited_at FROM companion_journal "
        "WHERE entry_type = 'wondering'"
    )
    row = await cur.fetchone()
    await cur.close()
    assert row[0] is not None


@pytest.mark.asyncio
async def test_perform_drops_journal_kind_moments():
    """Resolver moments of kind='journal' (other wonderings/noticings)
    are dropped before synthesis — connecting a wondering to a copy of
    itself produced meta-commentary about the journal, not a finding.
    All-journal results route to the honest-silence path."""
    from augmentum.companion_runtime.behavior import activity_selector
    from augmentum.resolver.core import Moment

    backend = await _boot_backend()
    runtime = _fake_runtime(backend)
    await _insert_pending_proposal(backend)
    await _insert_thread(backend, content="the helpful versus right thread")

    journal_moment = Moment(
        id="7355",
        kind="journal",
        score=0.9,
        snippet="the helpful versus right thread",
        title="wondering",
        created_at="2026-06-08",
        content_refs=[],
        legs=["journal_vec"],
        raw={},
    )
    with patch(
        "augmentum.resolver.resolve_moments",
        new=AsyncMock(return_value=[journal_moment]),
    ):
        await activity_selector._perform_revisit_thread(runtime)

    cur = await backend.conn.execute(
        "SELECT content, content_refs FROM companion_journal "
        "WHERE entry_type = 'noticing'"
    )
    row = await cur.fetchone()
    await cur.close()
    assert row is not None
    # Honest silence, not "looped back to wondering"
    assert row[0].startswith("Letting ")
    import json
    refs = json.loads(row[1] or "[]")
    assert not any(r.get("kind") == "journal" for r in refs)


def test_clip_phrase_cuts_at_word_boundary():
    """No mid-word slices in drawer prose ("…need in the brow")."""
    from augmentum.companion_runtime.behavior.activity_selector import _clip_phrase

    text = (
        "why does the algorithm treat the news cycle and the adult "
        "industry as if they serve the same structural need in the browser"
    )
    clipped = _clip_phrase(text, 120)
    assert len(clipped) <= 121  # limit + ellipsis char
    assert clipped.endswith("…")
    # The cut lands on a whole word, never inside one
    last_word = clipped[:-1].split()[-1]
    assert last_word in text.split()
    # Short text passes through untouched
    assert _clip_phrase("short note", 120) == "short note"
