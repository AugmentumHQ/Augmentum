"""Tests for DreamCompactor + on-write consolidation + count-gated time trim.

Focuses on the new compaction surface area added in 2026-05-02; uses an
in-memory aiosqlite fixture (no vec extension — the compactor's vec-backed
paths are tested via direct cosine math).
"""
from __future__ import annotations

import json
import struct
import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import aiosqlite
import pytest

from augmentum.dream.compactor import DreamCompactor
from augmentum.dream.consolidator import try_consolidate_dream
from augmentum.dream.journal import DreamJournal
from augmentum.dream.models import DreamEntry, DreamEntryType
from augmentum.utils.vector import cosine_similarity, parse_merged_response


UID = "user_test"
OTHER_UID = "user_other"


def _vec_to_blob(vec: list[float]) -> bytes:
    return struct.pack(f"<{len(vec)}f", *vec)


def _make_entry(
    *, content: str, embedding_vec: list[float] | None = None,
    user_id: str = UID, persona_id: str = "default",
    pinned: bool = False, created_at: str | None = None,
) -> dict:
    """Returns a dict matching what the journal would store. For test seeding."""
    return {
        "id": uuid.uuid4().hex[:16],
        "persona_id": persona_id,
        "content": content,
        "entry_type": DreamEntryType.REFLECTION.value,
        "source_memories": json.dumps([]),
        "source_sessions": json.dumps([]),
        "context_window": json.dumps({}),
        "embedding": _vec_to_blob(embedding_vec) if embedding_vec else None,
        "weight": 1.0,
        "pinned": 1 if pinned else 0,
        "dream_cycle_id": "test_cycle",
        "user_id": user_id,
        "created_at": created_at or datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S"),
        "expires_at": None,
    }


@pytest.fixture
async def journal(tmp_path):
    """Bare DreamJournal pointed at a temp DB with the dream tables created
    (no vec extension — keeps the test fast and avoids FastEmbed).
    """
    db_path = str(tmp_path / "compactor.db")
    async with aiosqlite.connect(db_path) as db:
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS dream_entries (
                id TEXT PRIMARY KEY,
                persona_id TEXT NOT NULL DEFAULT 'default',
                content TEXT NOT NULL,
                entry_type TEXT NOT NULL DEFAULT 'reflection',
                source_memories TEXT NOT NULL DEFAULT '[]',
                source_sessions TEXT NOT NULL DEFAULT '[]',
                context_window TEXT NOT NULL DEFAULT '{}',
                embedding BLOB,
                weight REAL NOT NULL DEFAULT 1.0,
                pinned INTEGER NOT NULL DEFAULT 0,
                dream_cycle_id TEXT NOT NULL,
                user_id TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                expires_at TEXT
            );
            CREATE TABLE IF NOT EXISTS dream_portraits (id TEXT PRIMARY KEY, persona_id TEXT, voice_notes TEXT, active_threads TEXT, impressions TEXT, source_entries TEXT, is_current INTEGER, checkpoint_name TEXT, user_id TEXT, created_at TEXT);
            CREATE TABLE IF NOT EXISTS dream_cycles (id TEXT PRIMARY KEY, persona_id TEXT, trigger_reason TEXT, memories_count INTEGER, entries_count INTEGER, model_used TEXT, tokens_used INTEGER, duration_ms INTEGER, status TEXT, error TEXT, started_at TEXT, completed_at TEXT, user_id TEXT);
            CREATE TABLE IF NOT EXISTS dream_memory_log (memory_id TEXT, dream_cycle_id TEXT, persona_id TEXT, user_id TEXT, created_at TEXT, PRIMARY KEY(memory_id, dream_cycle_id));
        """)
        await db.commit()
    j = DreamJournal(db_path)
    await j.initialize()
    return j


async def _seed_entries(journal, rows: list[dict]) -> None:
    """Insert raw test rows directly so we don't trigger on-write consolidation."""
    cols = ("id", "persona_id", "content", "entry_type", "source_memories",
            "source_sessions", "context_window", "embedding", "weight",
            "pinned", "dream_cycle_id", "user_id", "created_at", "expires_at")
    placeholders = ",".join("?" * len(cols))
    async with journal._connect() as db:
        for row in rows:
            await db.execute(
                f"INSERT INTO dream_entries ({','.join(cols)}) VALUES ({placeholders})",
                tuple(row[c] for c in cols),
            )
        await db.commit()


# ──────────────────────────────────────────────────────────────────────
# Pure-function tests (no DB / no LLM)
# ──────────────────────────────────────────────────────────────────────


def test_cosine_similarity_identical_vectors():
    a = [1.0, 0.5, 0.3]
    assert abs(cosine_similarity(a, a) - 1.0) < 1e-9


def test_cosine_similarity_orthogonal_vectors():
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_zero_norm():
    # Zero-norm input returns 0.0 rather than raising (callers treat as
    # "unknown" and skip). Verifies the defensive branch.
    assert cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_parse_merged_response_handles_merged_key():
    raw = '{"merged": "I noticed Alex loves his cat.", "importance": 0.8}'
    result = parse_merged_response(raw)
    assert result is not None
    assert result[0] == "I noticed Alex loves his cat."
    assert result[1] == 0.8


def test_parse_merged_response_handles_summary_key():
    raw = '{"summary": "Recurring theme of comfort with Whiskers.", "importance": 0.6}'
    result = parse_merged_response(raw)
    assert result is not None
    assert result[0] == "Recurring theme of comfort with Whiskers."


def test_parse_merged_response_strips_markdown_fences():
    # 5+ char body required (parser rejects garbage shorter than that)
    raw = '```json\n{"merged": "Hello world", "importance": 0.7}\n```'
    result = parse_merged_response(raw)
    assert result is not None
    assert result[0] == "Hello world"


def test_parse_merged_response_rejects_garbage():
    assert parse_merged_response("not json") is None
    assert parse_merged_response('{"merged": "ab"}') is None  # too short


def test_parse_merged_response_clamps_importance():
    out_of_range = parse_merged_response('{"merged": "valid text", "importance": 99}')
    assert out_of_range is not None
    assert out_of_range[1] == 1.0


# ──────────────────────────────────────────────────────────────────────
# Journal tests for compaction-supporting helpers
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_compact_journal_count_gate_skips_below_threshold(journal):
    """Below the count_threshold, time-trim is gated even if entries are old."""
    old_when = (datetime.now(UTC) - timedelta(days=60)).strftime("%Y-%m-%d %H:%M:%S")
    rows = [_make_entry(content=f"old entry {i}", created_at=old_when) for i in range(5)]
    await _seed_entries(journal, rows)

    stats = await journal.compact_journal(
        persona_id="default", max_age_days=30,
        user_id=UID, count_threshold=200,
    )
    assert stats["gated"] is True
    assert stats["compacted"] == 0
    # Confirm none were soft-deleted
    async with journal._connect() as db:
        cur = await db.execute(
            "SELECT COUNT(*) FROM dream_entries WHERE expires_at IS NOT NULL",
        )
        assert (await cur.fetchone())[0] == 0


@pytest.mark.asyncio
async def test_compact_journal_count_gate_proceeds_above_threshold(journal):
    """Above threshold, time-trim soft-deletes old unpinned entries."""
    old_when = (datetime.now(UTC) - timedelta(days=60)).strftime("%Y-%m-%d %H:%M:%S")
    new_when = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
    rows = [_make_entry(content=f"old {i}", created_at=old_when) for i in range(5)]
    rows += [_make_entry(content=f"new {i}", created_at=new_when) for i in range(5)]
    await _seed_entries(journal, rows)

    stats = await journal.compact_journal(
        persona_id="default", max_age_days=30,
        user_id=UID, count_threshold=3,  # 10 active > 3 → gate opens
    )
    assert stats["gated"] is False
    assert stats["compacted"] == 5  # the old ones


@pytest.mark.asyncio
async def test_compact_journal_count_gate_default_preserves_existing_behavior(journal):
    """Without count_threshold, behavior matches pre-compactor (always trim)."""
    old_when = (datetime.now(UTC) - timedelta(days=60)).strftime("%Y-%m-%d %H:%M:%S")
    await _seed_entries(journal, [_make_entry(content="old", created_at=old_when)])
    stats = await journal.compact_journal(persona_id="default", user_id=UID)
    assert stats["gated"] is False
    assert stats["compacted"] == 1


@pytest.mark.asyncio
async def test_merge_entries_updates_keep_and_soft_deletes_drop(journal):
    """merge_entries replaces keep's content and marks drop expired."""
    a = _make_entry(content="Original keep")
    b = _make_entry(content="Original drop")
    await _seed_entries(journal, [a, b])

    ok = await journal.merge_entries(
        keep_id=a["id"], drop_id=b["id"],
        merged_content="Merged content here", user_id=UID,
    )
    assert ok is True

    async with journal._connect() as db:
        cur = await db.execute(
            "SELECT content, expires_at FROM dream_entries WHERE id = ?", (a["id"],),
        )
        keep_row = await cur.fetchone()
        assert keep_row[0] == "Merged content here"
        assert keep_row[1] is None  # not expired

        cur = await db.execute(
            "SELECT expires_at FROM dream_entries WHERE id = ?", (b["id"],),
        )
        drop_row = await cur.fetchone()
        assert drop_row[0] is not None  # soft-deleted


@pytest.mark.asyncio
async def test_merge_entries_refuses_cross_user(journal):
    """A merge call with the wrong user_id finds no row and returns False."""
    a = _make_entry(content="user A", user_id=UID)
    await _seed_entries(journal, [a])

    ok = await journal.merge_entries(
        keep_id=a["id"], drop_id="bogus",
        merged_content="hijacked", user_id=OTHER_UID,
    )
    assert ok is False
    # Original content unchanged
    async with journal._connect() as db:
        cur = await db.execute(
            "SELECT content FROM dream_entries WHERE id = ?", (a["id"],),
        )
        assert (await cur.fetchone())[0] == "user A"


@pytest.mark.asyncio
async def test_merge_entries_requires_user_id(journal):
    with pytest.raises(ValueError, match="merge requires user_id"):
        await journal.merge_entries(
            keep_id="x", drop_id="y", merged_content="z", user_id="",
        )


# ──────────────────────────────────────────────────────────────────────
# try_consolidate_dream — pure LLM-merge logic with mocked backend
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_try_consolidate_dream_returns_merged_when_in_range():
    target = MagicMock(id="entry_1", content="Existing reflection about Whiskers.")
    candidates = [(target, 0.75)]  # in [0.65, 0.85]

    backend = MagicMock()
    backend.chat = AsyncMock(return_value=MagicMock(
        message=MagicMock(content='{"merged": "Combined reflection.", "importance": 0.7}'),
    ))

    result = await try_consolidate_dream(
        new_content="New reflection about Whiskers.",
        candidates=candidates, backend=backend, model="test-model",
        sim_low=0.65, sim_high=0.85,
    )
    assert result is not None
    merged_text, importance, target_id = result
    assert merged_text == "Combined reflection."
    assert target_id == "entry_1"


@pytest.mark.asyncio
async def test_try_consolidate_dream_skips_outside_range():
    """Below sim_low or above sim_high: no consolidation, return None."""
    target = MagicMock(id="entry_1", content="Existing")

    backend = MagicMock(chat=AsyncMock())
    # Below low
    out = await try_consolidate_dream(
        new_content="x", candidates=[(target, 0.5)],
        backend=backend, model=None, sim_low=0.65, sim_high=0.85,
    )
    assert out is None
    # Above high
    out = await try_consolidate_dream(
        new_content="x", candidates=[(target, 0.95)],
        backend=backend, model=None, sim_low=0.65, sim_high=0.85,
    )
    assert out is None
    # No LLM calls in either case
    assert backend.chat.call_count == 0


@pytest.mark.asyncio
async def test_try_consolidate_dream_skips_empty_candidates():
    backend = MagicMock(chat=AsyncMock())
    out = await try_consolidate_dream(
        new_content="x", candidates=[], backend=backend, model=None,
        sim_low=0.65, sim_high=0.85,
    )
    assert out is None
    assert backend.chat.call_count == 0


# ──────────────────────────────────────────────────────────────────────
# DreamCompactor — orchestration with mocked LLM
# ──────────────────────────────────────────────────────────────────────


def _mock_backend_returning(merged_text: str = "Merged version") -> MagicMock:
    """Helper: backend whose chat() returns canned merge JSON."""
    backend = MagicMock()
    backend.chat = AsyncMock(return_value=MagicMock(
        message=MagicMock(content=f'{{"merged": "{merged_text}", "importance": 0.7}}'),
    ))
    return backend


def _mock_registry_with(backend: object, model: str = "utility-model") -> MagicMock:
    registry = MagicMock()
    registry.resolve_model_for_role = AsyncMock(return_value=(backend, model))
    return registry


@pytest.mark.asyncio
async def test_dream_compactor_dedup_pair_merge(journal, monkeypatch):
    """Two near-duplicate entries (high cosine) get LLM-merged into one."""
    # Enable the feature via monkeypatched settings
    from augmentum import config
    monkeypatch.setattr(config.settings, "dream_compaction_enabled", True)
    monkeypatch.setattr(config.settings, "dream_dedup_threshold", 0.9)

    # Two entries with NEARLY-identical embeddings (cosine ~0.9999)
    near = [1.0, 0.0, 0.0, 0.0]
    near2 = [0.99, 0.01, 0.0, 0.0]
    other = [0.0, 1.0, 0.0, 0.0]
    await _seed_entries(journal, [
        _make_entry(content="Whiskers blocked the keyboard again.", embedding_vec=near),
        _make_entry(content="The cat sat on my hands.", embedding_vec=near2),
        _make_entry(content="The truck needed an oil change.", embedding_vec=other),
    ])

    backend = _mock_backend_returning("Whiskers and the keyboard incident.")
    registry = _mock_registry_with(backend)

    compactor = DreamCompactor(journal=journal, registry=registry)
    stats = await compactor.compact(user_id=UID)

    assert stats["deduped_pairs"] == 1
    # One LLM call for the pair (cluster phase finds none since only 2 in cluster + min_size=3)
    assert backend.chat.call_count >= 1

    # Confirm one of the near-duplicates was soft-deleted
    async with journal._connect() as db:
        cur = await db.execute(
            "SELECT COUNT(*) FROM dream_entries WHERE expires_at IS NOT NULL"
        )
        assert (await cur.fetchone())[0] == 1


@pytest.mark.asyncio
async def test_dream_compactor_cluster_summarize(journal, monkeypatch):
    """Three thematically-similar entries summarize into one."""
    from augmentum import config
    monkeypatch.setattr(config.settings, "dream_compaction_enabled", True)
    monkeypatch.setattr(config.settings, "dream_dedup_threshold", 0.99)  # avoid pair-merge
    monkeypatch.setattr(config.settings, "dream_cluster_threshold", 0.7)
    monkeypatch.setattr(config.settings, "dream_cluster_min_size", 3)

    # Three entries similar enough to cluster (cosine ~0.8-0.95 between all
    # pairs) but NOT close enough to trip the dedup_threshold=0.99 gate set
    # above. Picked so cluster phase fires cleanly without any pair-merge
    # firing first and shrinking the candidate pool.
    similar_a = [1.0, 0.5, 0.0]
    similar_b = [1.0, 0.0, 0.5]
    similar_c = [1.0, 0.3, 0.3]
    odd = [0.0, 0.0, 1.0]
    await _seed_entries(journal, [
        _make_entry(content="Whiskers is a comforting presence.", embedding_vec=similar_a),
        _make_entry(content="Watching Whiskers nap is grounding.", embedding_vec=similar_b),
        _make_entry(content="Whiskers's purr is the best background noise.", embedding_vec=similar_c),
        _make_entry(content="The truck is in good shape.", embedding_vec=odd),
    ])

    backend = _mock_backend_returning("Whiskers as a recurring source of comfort.")
    registry = _mock_registry_with(backend)

    compactor = DreamCompactor(journal=journal, registry=registry)
    stats = await compactor.compact(user_id=UID)

    assert stats["summarized_clusters"] >= 1
    assert stats["summarized_entries"] >= 3

    # Confirm the cluster originals are soft-deleted, summary inserted, odd entry untouched
    async with journal._connect() as db:
        cur = await db.execute(
            "SELECT COUNT(*) FROM dream_entries WHERE content = 'The truck is in good shape.' AND expires_at IS NULL"
        )
        assert (await cur.fetchone())[0] == 1
        cur = await db.execute(
            "SELECT COUNT(*) FROM dream_entries WHERE expires_at IS NOT NULL"
        )
        assert (await cur.fetchone())[0] >= 3


@pytest.mark.asyncio
async def test_dream_compactor_per_user_isolation(journal, monkeypatch):
    """Compacting user A's journal doesn't touch user B's entries."""
    from augmentum import config
    monkeypatch.setattr(config.settings, "dream_compaction_enabled", True)
    monkeypatch.setattr(config.settings, "dream_dedup_threshold", 0.9)

    near_a = [1.0, 0.0, 0.0]
    near_b = [0.99, 0.01, 0.0]
    # User A: 2 near-duplicates
    await _seed_entries(journal, [
        _make_entry(content="A keep", embedding_vec=near_a, user_id=UID),
        _make_entry(content="A drop", embedding_vec=near_b, user_id=UID),
    ])
    # User B: 2 near-duplicates of their own
    await _seed_entries(journal, [
        _make_entry(content="B keep", embedding_vec=near_a, user_id=OTHER_UID),
        _make_entry(content="B drop", embedding_vec=near_b, user_id=OTHER_UID),
    ])

    backend = _mock_backend_returning("merged")
    registry = _mock_registry_with(backend)

    compactor = DreamCompactor(journal=journal, registry=registry)
    await compactor.compact(user_id=UID)

    # Only one of A's entries should be soft-deleted; B's both remain active.
    async with journal._connect() as db:
        cur = await db.execute(
            "SELECT user_id, expires_at FROM dream_entries ORDER BY user_id"
        )
        rows = await cur.fetchall()
    a_expired = sum(1 for r in rows if r[0] == UID and r[1] is not None)
    b_expired = sum(1 for r in rows if r[0] == OTHER_UID and r[1] is not None)
    assert a_expired == 1
    assert b_expired == 0  # B untouched


@pytest.mark.asyncio
async def test_dream_compactor_disabled_returns_early(journal, monkeypatch):
    from augmentum import config
    monkeypatch.setattr(config.settings, "dream_compaction_enabled", False)

    compactor = DreamCompactor(journal=journal)
    stats = await compactor.compact(user_id=UID)
    assert stats == {"enabled": False}


@pytest.mark.asyncio
async def test_dream_compactor_no_backend_skips_llm_phases(journal, monkeypatch):
    """Without a registry, time-trim still runs but dedup/cluster skip."""
    from augmentum import config
    monkeypatch.setattr(config.settings, "dream_compaction_enabled", True)

    await _seed_entries(journal, [
        _make_entry(content="entry", embedding_vec=[1.0, 0.0]),
    ])
    compactor = DreamCompactor(journal=journal, registry=None)
    stats = await compactor.compact(user_id=UID)
    assert stats["deduped_pairs"] == 0
    assert stats["summarized_clusters"] == 0


@pytest.mark.asyncio
async def test_dream_compactor_excludes_pinned_from_compaction(journal, monkeypatch):
    """Pinned entries are protected even when they'd otherwise cluster."""
    from augmentum import config
    monkeypatch.setattr(config.settings, "dream_compaction_enabled", True)
    monkeypatch.setattr(config.settings, "dream_dedup_threshold", 0.9)

    near_a = [1.0, 0.0]
    near_b = [0.99, 0.01]
    await _seed_entries(journal, [
        _make_entry(content="pinned reflection", embedding_vec=near_a, pinned=True),
        _make_entry(content="non-pinned twin", embedding_vec=near_b, pinned=False),
    ])
    backend = _mock_backend_returning("merged")
    registry = _mock_registry_with(backend)

    compactor = DreamCompactor(journal=journal, registry=registry)
    stats = await compactor.compact(user_id=UID)
    # Pinned entries are excluded from the load, so the only candidate has
    # no pair to merge with — 0 dedup
    assert stats["deduped_pairs"] == 0


@pytest.mark.asyncio
async def test_dream_compactor_requires_user_id(journal):
    compactor = DreamCompactor(journal=journal)
    with pytest.raises(ValueError, match="requires user_id"):
        await compactor.compact(user_id="")
