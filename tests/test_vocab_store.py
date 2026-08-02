"""VocabStore round-trip tests (pure SQLite, no app stack).

Mirrors migration 145's schema locally — the migration file is the
source of truth; this keeps the unit test fast and self-contained.
"""

from __future__ import annotations

import aiosqlite
import pytest

_SCHEMA_SQL = """
CREATE TABLE users (id TEXT PRIMARY KEY);
CREATE TABLE vocab_state (
    user_id          TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    lang_code        TEXT NOT NULL,
    word_id          TEXT NOT NULL,
    fsrs_difficulty  REAL NOT NULL DEFAULT 5.0,
    fsrs_stability   REAL NOT NULL DEFAULT 0.0,
    fsrs_due_at      TEXT NOT NULL,
    fsrs_reps        INTEGER NOT NULL DEFAULT 0,
    fsrs_lapses      INTEGER NOT NULL DEFAULT 0,
    fsrs_last_grade  INTEGER,
    mastery_state    TEXT NOT NULL DEFAULT 'new',
    first_seen_at    TEXT NOT NULL DEFAULT (datetime('now')),
    last_reviewed_at TEXT,
    source_surface   TEXT NOT NULL,
    source_ref       TEXT,
    exposure_input   INTEGER NOT NULL DEFAULT 0,
    exposure_output  INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (user_id, lang_code, word_id)
);
CREATE INDEX idx_vocab_state_due ON vocab_state(user_id, lang_code, fsrs_due_at);
CREATE INDEX idx_vocab_state_mastery ON vocab_state(user_id, lang_code, mastery_state);
"""

_FAR_FUTURE = "2999-01-01 00:00:00"
_FAR_PAST = "2000-01-01 00:00:00"


async def _mkstore():
    from augmentum.state.vocab_store import VocabStore

    conn = await aiosqlite.connect(":memory:")
    await conn.executescript(_SCHEMA_SQL)
    await conn.execute("INSERT INTO users (id) VALUES ('u1')")
    await conn.execute("INSERT INTO users (id) VALUES ('u2')")
    await conn.commit()
    return VocabStore(conn), conn


@pytest.mark.asyncio
async def test_add_word_roundtrip_and_idempotent():
    store, _ = await _mkstore()
    added = await store.add_word(
        user_id="u1", lang_code="ja", word_id="1358280",
        source_surface="browse", source_ref="https://ja.wikipedia.org/x",
    )
    assert added is True
    again = await store.add_word(
        user_id="u1", lang_code="ja", word_id="1358280", source_surface="browse",
    )
    assert again is False  # re-clicking is a no-op

    row = await store.get_word(user_id="u1", lang_code="ja", word_id="1358280")
    assert row is not None
    assert row["mastery_state"] == "new"
    assert row["source_surface"] == "browse"
    assert row["source_ref"] == "https://ja.wikipedia.org/x"
    assert row["exposure_input"] == 1
    assert row["fsrs_reps"] == 0


@pytest.mark.asyncio
async def test_add_word_requires_ids():
    store, _ = await _mkstore()
    with pytest.raises(ValueError):
        await store.add_word(user_id="", lang_code="ja", word_id="w", source_surface="browse")
    with pytest.raises(ValueError):
        await store.add_word(user_id="u1", lang_code="", word_id="w", source_surface="browse")


@pytest.mark.asyncio
async def test_user_isolation():
    store, _ = await _mkstore()
    await store.add_word(user_id="u1", lang_code="ja", word_id="w", source_surface="browse")
    assert await store.get_word(user_id="u2", lang_code="ja", word_id="w") is None
    # Same word_id in another user's queue is fully independent.
    assert await store.add_word(user_id="u2", lang_code="ja", word_id="w", source_surface="browse") is True


@pytest.mark.asyncio
async def test_due_queue_respects_cutoff():
    store, _ = await _mkstore()
    await store.add_word(user_id="u1", lang_code="ja", word_id="a", source_surface="browse")
    await store.add_word(user_id="u1", lang_code="ja", word_id="b", source_surface="browse")
    # Freshly-added words are due ~tomorrow, not in the past.
    assert await store.count_due(user_id="u1", lang_code="ja", now=_FAR_PAST) == 0
    assert await store.get_due(user_id="u1", lang_code="ja", now=_FAR_PAST) == []
    # With a far-future cutoff, both count as due.
    due = await store.get_due(user_id="u1", lang_code="ja", now=_FAR_FUTURE)
    assert {r["word_id"] for r in due} == {"a", "b"}
    assert await store.count_due(user_id="u1", lang_code="ja", now=_FAR_FUTURE) == 2
    # Other users see nothing.
    assert await store.count_due(user_id="u2", lang_code="ja", now=_FAR_FUTURE) == 0


@pytest.mark.asyncio
async def test_update_after_grade():
    store, _ = await _mkstore()
    await store.add_word(user_id="u1", lang_code="ja", word_id="w", source_surface="browse")
    ok = await store.update_after_grade(
        user_id="u1", lang_code="ja", word_id="w",
        difficulty=5.28, stability=3.17, due_at=_FAR_FUTURE,
        reps=1, lapses=0, grade=3, mastery_state="learning",
    )
    assert ok is True
    row = await store.get_word(user_id="u1", lang_code="ja", word_id="w")
    assert row["fsrs_reps"] == 1
    assert row["fsrs_last_grade"] == 3
    assert row["mastery_state"] == "learning"
    assert row["fsrs_difficulty"] == pytest.approx(5.28)
    assert row["fsrs_stability"] == pytest.approx(3.17)
    assert row["last_reviewed_at"] is not None
    assert row["exposure_input"] == 1  # unchanged by grading
    assert await store.count_due(user_id="u1", lang_code="ja") == 0  # pushed to future

    # Grading a word that isn't queued is a no-op.
    miss = await store.update_after_grade(
        user_id="u1", lang_code="ja", word_id="nope",
        difficulty=5.0, stability=1.0, due_at=_FAR_FUTURE,
        reps=1, lapses=0, grade=3, mastery_state="learning",
    )
    assert miss is False


@pytest.mark.asyncio
async def test_seed_words():
    store, _ = await _mkstore()
    n = await store.seed_words(user_id="u1", lang_code="ja", word_ids=["a", "b", "c", "a"])
    assert n == 3  # the duplicate within the batch collapses
    assert await store.seed_words(user_id="u1", lang_code="ja", word_ids=["a", "d"]) == 1
    row = await store.get_word(user_id="u1", lang_code="ja", word_id="a")
    assert row["source_surface"] == "seeded"
    assert row["exposure_input"] == 0  # not encountered in context yet
    assert await store.seed_words(user_id="u1", lang_code="ja", word_ids=[]) == 0


@pytest.mark.asyncio
async def test_seed_words_are_due_immediately():
    store, _ = await _mkstore()
    assert await store.seed_words(user_id="u1", lang_code="ja", word_ids=["a", "b"]) == 2
    # Seeded words are due now (an explicit "drill these" action), unlike a
    # clicked word which lands tomorrow.
    assert await store.count_due(user_id="u1", lang_code="ja") == 2
    await store.add_word(user_id="u1", lang_code="ja", word_id="c", source_surface="browse")
    assert await store.count_due(user_id="u1", lang_code="ja") == 2  # 'c' not due today


@pytest.mark.asyncio
async def test_counts_by_mastery_and_remove():
    store, _ = await _mkstore()
    await store.add_word(user_id="u1", lang_code="ja", word_id="a", source_surface="browse")
    await store.add_word(user_id="u1", lang_code="ja", word_id="b", source_surface="browse")
    await store.update_after_grade(
        user_id="u1", lang_code="ja", word_id="b",
        difficulty=5.0, stability=30.0, due_at=_FAR_FUTURE,
        reps=2, lapses=0, grade=4, mastery_state="mature",
    )
    assert await store.counts_by_mastery(user_id="u1", lang_code="ja") == {"new": 1, "mature": 1}
    assert await store.remove_word(user_id="u1", lang_code="ja", word_id="a") is True
    assert await store.remove_word(user_id="u1", lang_code="ja", word_id="a") is False
    assert await store.counts_by_mastery(user_id="u1", lang_code="ja") == {"mature": 1}


@pytest.mark.asyncio
async def test_bump_exposure():
    store, _ = await _mkstore()
    await store.add_word(user_id="u1", lang_code="ja", word_id="w", source_surface="browse")
    assert await store.bump_exposure(user_id="u1", lang_code="ja", word_id="w") is True
    assert (await store.get_word(user_id="u1", lang_code="ja", word_id="w"))["exposure_input"] == 2
    assert await store.bump_exposure(user_id="u1", lang_code="ja", word_id="w", output=True) is True
    assert (await store.get_word(user_id="u1", lang_code="ja", word_id="w"))["exposure_output"] == 1
    # Not-queued word: no-op.
    assert await store.bump_exposure(user_id="u1", lang_code="ja", word_id="x") is False


@pytest.mark.asyncio
async def test_list_all():
    store, _ = await _mkstore()
    for wid in ("a", "b", "c"):
        await store.add_word(user_id="u1", lang_code="ja", word_id=wid, source_surface="browse")
    await store.add_word(user_id="u1", lang_code="es", word_id="z", source_surface="browse")
    rows = await store.list_all(user_id="u1", lang_code="ja")
    assert {r["word_id"] for r in rows} == {"a", "b", "c"}  # es row excluded
    assert await store.list_all(user_id="u2", lang_code="ja") == []


def test_timestamp_helpers():
    from augmentum.state import vocab_store as vs

    now = vs.now_ts()
    assert len(now) == 19 and now[4] == "-" and now[10] == " "
    assert vs.future_ts(1) > now
    assert vs.future_ts(-1) < now
    assert vs.parse_ts(now) is not None
    assert vs.parse_ts(None) is None
    assert vs.parse_ts("not a timestamp") is None
    assert vs.parse_ts("2026-05-11T12:00:00+00:00") is not None  # tolerate ISO
    assert vs.elapsed_days_since(None) == 0.0
    assert vs.elapsed_days_since(_FAR_PAST) > 8000.0
    assert vs.elapsed_days_since(_FAR_FUTURE) == 0.0  # clamped at 0
