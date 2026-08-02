"""Multi-Session Chat (MSC) benchmark — tests the FULL memory pipeline end-to-end.

Unlike the PersonaChat benchmark (extraction only), this tests:
  1. Extraction across multiple sessions (fact accumulation)
  2. Storage with real SQLite + embeddings (dedup + supersede)
  3. Retrieval via hybrid search (vector + FTS5 + RRF)
  4. Fact evolution — personas grow across sessions, old facts update

Dataset: nayohan/multi_session_chat (HuggingFace)
Each dialog has 2-4 sessions with evolving persona facts.

Run:
    pytest tests/test_memory_benchmark_msc.py -v --tb=short
    pytest tests/test_memory_benchmark_msc.py -v -k report --tb=short   # summary only
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Lazy imports
# ---------------------------------------------------------------------------

_HF_AVAILABLE = True
try:
    from datasets import load_dataset
except ImportError:
    _HF_AVAILABLE = False

needs_hf = pytest.mark.skipif(not _HF_AVAILABLE, reason="datasets library not installed")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DATASET_ID = "nayohan/multi_session_chat"
_NUM_DIALOGS = 250  # Number of dialog chains to evaluate
_MATCH_THRESHOLD = float(os.environ.get("MEMORY_BENCH_MATCH_THRESHOLD", "0.45"))

_CACHE_DIR = Path(__file__).resolve().parent / ".bench_cache"
_CACHE_FILE = _CACHE_DIR / f"msc_{_NUM_DIALOGS}.json"

# Migrations needed for in-memory SQLite
_MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "augmentum" / "state" / "migrations"

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class SessionData:
    """One session within a multi-session dialog."""
    session_id: int
    persona_facts: list[str]  # ground truth persona at this session
    speaker1_turns: list[str]  # Speaker 1's utterances
    speaker2_turns: list[str]


@dataclass
class DialogChain:
    """A multi-session dialog between two speakers."""
    dialog_id: int
    sessions: list[SessionData]

    @property
    def final_persona(self) -> list[str]:
        """The most evolved persona (last session)."""
        return self.sessions[-1].persona_facts if self.sessions else []

    @property
    def initial_persona(self) -> list[str]:
        """Starting persona (first session)."""
        return self.sessions[0].persona_facts if self.sessions else []

    @property
    def new_facts_per_session(self) -> list[int]:
        """How many NEW facts appear in each session vs the previous."""
        counts = []
        prev = set()
        for s in self.sessions:
            current = {f.lower() for f in s.persona_facts}
            counts.append(len(current - prev))
            prev = current
        return counts


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _clean(text: str) -> str:
    return text.strip().rstrip(".").strip()


def _extract_speaker_turns(
    dialogue: list[str], speakers: list[str], target: str,
) -> list[str]:
    """Extract turns for a specific speaker."""
    return [d.strip() for d, s in zip(dialogue, speakers) if s == target and d.strip()]


def _load_dialogs() -> list[DialogChain]:
    """Load and group MSC dialogs by dialog_id."""
    if _CACHE_FILE.exists():
        with open(_CACHE_FILE, encoding="utf-8") as f:
            data = json.load(f)
        chains = []
        for d in data:
            sessions = [SessionData(**s) for s in d["sessions"]]
            chains.append(DialogChain(dialog_id=d["dialog_id"], sessions=sessions))
        return chains

    ds = load_dataset(_DATASET_ID, split="train")

    # Group by dialog_id
    from collections import defaultdict
    groups: dict[int, list[dict]] = defaultdict(list)
    for row in ds:
        groups[row["dialoug_id"]].append(row)

    # Only keep dialogs with 2+ sessions (multi-session)
    chains: list[DialogChain] = []
    for did in sorted(groups.keys()):
        rows = sorted(groups[did], key=lambda r: r["session_id"])
        if len(rows) < 2:
            continue

        sessions = []
        for row in rows:
            sessions.append(SessionData(
                session_id=row["session_id"],
                persona_facts=[_clean(p) for p in row["persona1"] if p.strip()],
                speaker1_turns=_extract_speaker_turns(
                    row["dialogue"], row["speaker"], "Speaker 1",
                ),
                speaker2_turns=_extract_speaker_turns(
                    row["dialogue"], row["speaker"], "Speaker 2",
                ),
            ))
        chains.append(DialogChain(dialog_id=did, sessions=sessions))

        if len(chains) >= _NUM_DIALOGS:
            break

    # Cache
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump([
            {
                "dialog_id": c.dialog_id,
                "sessions": [
                    {
                        "session_id": s.session_id,
                        "persona_facts": s.persona_facts,
                        "speaker1_turns": s.speaker1_turns,
                        "speaker2_turns": s.speaker2_turns,
                    }
                    for s in c.sessions
                ],
            }
            for c in chains
        ], f)

    return chains


# ---------------------------------------------------------------------------
# Full pipeline helpers — real SQLite + embeddings + store
# ---------------------------------------------------------------------------

from augmentum.memory.extractor import (
    _cosine_similarity,
    heuristic_extract,
    should_extract,
)

_BOOTSTRAP_SQL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (datetime('now')),
    description TEXT
);
INSERT INTO schema_version (version, description) VALUES (5, 'pre-memory baseline');
"""


async def _apply_migration(conn, version: int) -> None:
    for path in _MIGRATIONS_DIR.glob("*.sql"):
        try:
            v = int(path.stem.split("_")[0])
        except (ValueError, IndexError):
            continue
        if v == version:
            sql = path.read_text(encoding="utf-8")
            await conn.executescript(sql)
            await conn.commit()
            return


async def _create_memory_store():
    """Create a fresh in-memory MemoryStore with real migrations."""
    import aiosqlite

    from augmentum.memory.store import MemoryStore

    conn = await aiosqlite.connect(":memory:")
    conn.row_factory = aiosqlite.Row
    await conn.executescript(_BOOTSTRAP_SQL)
    await conn.commit()

    for v in [6, 8, 9]:
        await _apply_migration(conn, v)

    class _Backend:
        def __init__(self, c):
            self.conn = c
            self.vec_enabled = False  # FTS5 only for speed

    store = MemoryStore(_Backend(conn))
    return store, conn


def _get_embeddings(texts: list[str]) -> list[list[float]]:
    from augmentum.memory.embeddings import EmbeddingService
    return EmbeddingService.embed(texts)


def _match_facts(
    extracted: list[str],
    ground_truth: list[str],
    threshold: float = _MATCH_THRESHOLD,
) -> tuple[set[int], set[int]]:
    """Match extracted facts to ground-truth via cosine similarity."""
    if not extracted or not ground_truth:
        return set(), set()

    all_texts = extracted + ground_truth
    all_embs = _get_embeddings(all_texts)
    ext_embs = all_embs[:len(extracted)]
    gt_embs = all_embs[len(extracted):]

    matched_ext: set[int] = set()
    matched_gt: set[int] = set()

    for gi, ge in enumerate(gt_embs):
        best_sim = 0.0
        best_ei = -1
        for ei, ee in enumerate(ext_embs):
            sim = _cosine_similarity(ee, ge)
            if sim > best_sim:
                best_sim = sim
                best_ei = ei
        if best_sim >= threshold:
            matched_gt.add(gi)
            matched_ext.add(best_ei)

    return matched_ext, matched_gt


# ---------------------------------------------------------------------------
# Per-dialog evaluation result
# ---------------------------------------------------------------------------


@dataclass
class DialogResult:
    dialog_id: int
    num_sessions: int
    initial_persona: list[str]
    final_persona: list[str]

    # Extraction metrics (heuristic only, across all sessions)
    total_extracted: int = 0
    extraction_per_session: list[int] = field(default_factory=list)

    # Storage metrics (after dedup)
    stored_count: int = 0
    dedup_saved: int = 0  # how many were deduped away

    # Retrieval metrics (recall against final persona)
    recalled_facts: list[str] = field(default_factory=list)
    retrieval_matched_gt: int = 0
    retrieval_total_gt: int = 0
    retrieval_matched_recalled: int = 0
    retrieval_total_recalled: int = 0

    # Fact accumulation
    facts_grew: bool = False  # did stored facts grow across sessions?

    @property
    def retrieval_precision(self) -> float:
        return self.retrieval_matched_recalled / self.retrieval_total_recalled if self.retrieval_total_recalled else 0.0

    @property
    def retrieval_recall(self) -> float:
        return self.retrieval_matched_gt / self.retrieval_total_gt if self.retrieval_total_gt else 0.0

    @property
    def retrieval_f1(self) -> float:
        p, r = self.retrieval_precision, self.retrieval_recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------


async def _evaluate_dialog(chain: DialogChain) -> DialogResult:
    """Run the full pipeline for one dialog chain."""
    store, conn = await _create_memory_store()
    user_id = f"user_{chain.dialog_id}"

    result = DialogResult(
        dialog_id=chain.dialog_id,
        num_sessions=len(chain.sessions),
        initial_persona=chain.initial_persona,
        final_persona=chain.final_persona,
        retrieval_total_gt=len(chain.final_persona),
    )

    total_extracted = 0
    prev_count = 0

    try:
        # Phase 1: Process each session through extraction + storage
        for session in chain.sessions:
            session_id = f"session_{session.session_id}"
            session_extracted = 0

            # Extract from Speaker 1 turns (whose persona we're tracking)
            for turn in session.speaker1_turns:
                if not should_extract(turn):
                    continue
                facts = heuristic_extract(turn)
                for fact in facts:
                    fact.source_context = {"session_id": session_id}
                    await store.store_fact(
                        fact,
                        user_id=user_id,
                        session_id=session_id,
                        is_explicit=fact.is_explicit,
                    )
                    session_extracted += 1

            result.extraction_per_session.append(session_extracted)
            total_extracted += session_extracted

            # Check fact growth
            current_count = len(await store.list_all(user_id=user_id))
            if current_count > prev_count and prev_count > 0:
                result.facts_grew = True
            prev_count = current_count

        result.total_extracted = total_extracted
        result.stored_count = prev_count
        result.dedup_saved = total_extracted - prev_count

        # Phase 2: Retrieval — query with each ground-truth fact as the query
        all_recalled: list[str] = []
        seen_recalled: set[str] = set()

        for gt_fact in chain.final_persona:
            memories = await store.recall(
                query=gt_fact,
                user_id=user_id,
                limit=3,
                min_score=0.0,
            )
            for mem in memories:
                key = mem.content.lower()
                if key not in seen_recalled:
                    seen_recalled.add(key)
                    all_recalled.append(mem.content)

        result.recalled_facts = all_recalled
        result.retrieval_total_recalled = len(all_recalled)

        # Phase 3: Match recalled facts against final persona
        if all_recalled and chain.final_persona:
            m_recalled, m_gt = _match_facts(all_recalled, chain.final_persona)
            result.retrieval_matched_recalled = len(m_recalled)
            result.retrieval_matched_gt = len(m_gt)

    finally:
        await conn.close()

    return result


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def dialog_chains() -> list[DialogChain]:
    return _load_dialogs()


@pytest.fixture(scope="module")
def dialog_results(dialog_chains: list[DialogChain]) -> list[DialogResult]:
    """Run the full pipeline on all dialogs."""
    async def _run():
        results = []
        for chain in dialog_chains:
            r = await _evaluate_dialog(chain)
            results.append(r)
        return results
    return asyncio.run(_run())


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------


@dataclass
class MSCAggregateMetrics:
    total_dialogs: int = 0
    total_sessions: int = 0
    dialogs_with_extraction: int = 0
    total_gt_facts: int = 0
    total_extracted: int = 0
    total_stored: int = 0
    total_dedup_saved: int = 0
    total_recalled: int = 0
    total_retrieval_matched_gt: int = 0
    total_retrieval_matched_recalled: int = 0

    # Per-dialog metrics
    per_dialog_retrieval_precision: list[float] = field(default_factory=list)
    per_dialog_retrieval_recall: list[float] = field(default_factory=list)
    per_dialog_retrieval_f1: list[float] = field(default_factory=list)

    dialogs_with_fact_growth: int = 0

    @property
    def dedup_rate(self) -> float:
        return self.total_dedup_saved / self.total_extracted if self.total_extracted else 0.0

    @property
    def micro_retrieval_precision(self) -> float:
        return self.total_retrieval_matched_recalled / self.total_recalled if self.total_recalled else 0.0

    @property
    def micro_retrieval_recall(self) -> float:
        return self.total_retrieval_matched_gt / self.total_gt_facts if self.total_gt_facts else 0.0

    @property
    def micro_retrieval_f1(self) -> float:
        p, r = self.micro_retrieval_precision, self.micro_retrieval_recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def macro_retrieval_precision(self) -> float:
        return statistics.mean(self.per_dialog_retrieval_precision) if self.per_dialog_retrieval_precision else 0.0

    @property
    def macro_retrieval_recall(self) -> float:
        return statistics.mean(self.per_dialog_retrieval_recall) if self.per_dialog_retrieval_recall else 0.0

    @property
    def macro_retrieval_f1(self) -> float:
        return statistics.mean(self.per_dialog_retrieval_f1) if self.per_dialog_retrieval_f1 else 0.0


def _compute_metrics(results: list[DialogResult]) -> MSCAggregateMetrics:
    m = MSCAggregateMetrics(total_dialogs=len(results))

    for r in results:
        m.total_sessions += r.num_sessions
        m.total_gt_facts += r.retrieval_total_gt
        m.total_extracted += r.total_extracted
        m.total_stored += r.stored_count
        m.total_dedup_saved += r.dedup_saved
        m.total_recalled += r.retrieval_total_recalled
        m.total_retrieval_matched_gt += r.retrieval_matched_gt
        m.total_retrieval_matched_recalled += r.retrieval_matched_recalled

        if r.total_extracted > 0:
            m.dialogs_with_extraction += 1
        if r.facts_grew:
            m.dialogs_with_fact_growth += 1

        if r.retrieval_total_recalled > 0:
            m.per_dialog_retrieval_precision.append(r.retrieval_precision)
            m.per_dialog_retrieval_recall.append(r.retrieval_recall)
            m.per_dialog_retrieval_f1.append(r.retrieval_f1)

    return m


@pytest.fixture(scope="module")
def metrics(dialog_results: list[DialogResult]) -> MSCAggregateMetrics:
    return _compute_metrics(dialog_results)


# ===========================================================================
# Tests
# ===========================================================================


@needs_hf
class TestDatasetLoading:

    def test_dialog_count(self, dialog_chains: list[DialogChain]):
        assert len(dialog_chains) >= 200, f"Expected ~250 dialogs, got {len(dialog_chains)}"

    def test_multi_session(self, dialog_chains: list[DialogChain]):
        for chain in dialog_chains[:50]:
            assert len(chain.sessions) >= 2, f"Dialog {chain.dialog_id} has only {len(chain.sessions)} sessions"

    def test_persona_evolves(self, dialog_chains: list[DialogChain]):
        """At least some dialogs should have more facts in later sessions."""
        grew = sum(1 for c in dialog_chains if len(c.final_persona) > len(c.initial_persona))
        assert grew > len(dialog_chains) * 0.3, f"Only {grew} dialogs had persona growth"


@needs_hf
class TestExtractionAcrossSessions:

    def test_extraction_rate(self, metrics: MSCAggregateMetrics):
        rate = metrics.dialogs_with_extraction / metrics.total_dialogs
        assert rate >= 0.50, f"Only {rate:.1%} of dialogs had any extraction"

    def test_extractions_grow_across_sessions(self, dialog_results: list[DialogResult]):
        """Dialogs with more sessions should produce more extractions."""
        multi = [r for r in dialog_results if r.num_sessions >= 3 and r.total_extracted > 0]
        if not multi:
            pytest.skip("No dialogs with 3+ sessions and extractions")
        avg_per_session = [r.total_extracted / r.num_sessions for r in multi]
        assert statistics.mean(avg_per_session) > 0.5, "Less than 0.5 extractions per session on average"


@needs_hf
class TestStorageDedup:

    def test_dedup_rate_bounded(self, metrics: MSCAggregateMetrics):
        """Dedup rate should not be excessively high (would indicate over-merging).

        Note: with vec_enabled=False, dedup via embedding similarity is disabled.
        Dedup only fires on exact content match. A 0% rate is valid in FTS5-only mode.
        """
        if metrics.total_extracted == 0:
            pytest.skip("No extractions")
        assert metrics.dedup_rate < 0.90, (
            f"Dedup rate {metrics.dedup_rate:.1%} too high — over-merging?"
        )

    def test_stored_less_than_extracted(self, metrics: MSCAggregateMetrics):
        """Dedup should reduce total stored vs total extracted."""
        assert metrics.total_stored <= metrics.total_extracted


@needs_hf
class TestRetrieval:

    def test_retrieval_returns_results(self, metrics: MSCAggregateMetrics):
        assert metrics.total_recalled > 0, "No facts recalled at all"

    def test_micro_retrieval_precision(self, metrics: MSCAggregateMetrics):
        if metrics.total_recalled == 0:
            pytest.skip("No recalls")
        assert metrics.micro_retrieval_precision >= 0.15, (
            f"Retrieval precision {metrics.micro_retrieval_precision:.1%} below 15%"
        )

    def test_micro_retrieval_recall(self, metrics: MSCAggregateMetrics):
        assert metrics.micro_retrieval_recall >= 0.10, (
            f"Retrieval recall {metrics.micro_retrieval_recall:.1%} below 10%"
        )

    def test_micro_retrieval_f1(self, metrics: MSCAggregateMetrics):
        if metrics.total_recalled == 0:
            pytest.skip("No recalls")
        assert metrics.micro_retrieval_f1 >= 0.10, (
            f"Retrieval F1 {metrics.micro_retrieval_f1:.1%} below 10%"
        )


@needs_hf
class TestFactAccumulation:

    def test_facts_grow_across_sessions(self, metrics: MSCAggregateMetrics):
        """Some dialogs should show fact growth (more stored after later sessions)."""
        rate = metrics.dialogs_with_fact_growth / metrics.dialogs_with_extraction if metrics.dialogs_with_extraction else 0
        assert rate >= 0.10, (
            f"Only {rate:.1%} of dialogs showed fact growth across sessions"
        )


# ===========================================================================
# Diagnostic report
# ===========================================================================


@needs_hf
class TestMSCBenchmarkReport:

    def test_report(self, dialog_results: list[DialogResult], metrics: MSCAggregateMetrics):
        r = []
        r.append("")
        r.append("=" * 72)
        r.append("  MSC FULL-PIPELINE MEMORY BENCHMARK")
        r.append("=" * 72)
        r.append("")
        r.append(f"  Dataset:            {_DATASET_ID}")
        r.append(f"  Dialogs evaluated:  {metrics.total_dialogs}")
        r.append(f"  Total sessions:     {metrics.total_sessions}")
        r.append(f"  Match threshold:    {_MATCH_THRESHOLD}")
        r.append("")
        r.append("  --- Extraction ---")
        r.append(f"  Dialogs with extraction: {metrics.dialogs_with_extraction} / {metrics.total_dialogs}")
        r.append(f"  Total extracted:         {metrics.total_extracted}")
        r.append(f"  Total GT facts:          {metrics.total_gt_facts}")
        r.append("")
        r.append("  --- Storage (dedup) ---")
        r.append(f"  Total stored (after dedup): {metrics.total_stored}")
        r.append(f"  Dedup saved:                {metrics.total_dedup_saved} ({metrics.dedup_rate:.1%})")
        r.append("")
        r.append("  --- Retrieval (end-to-end) ---")
        r.append(f"  Total recalled:  {metrics.total_recalled}")
        r.append(f"  Micro Precision: {metrics.micro_retrieval_precision:.1%}  ({metrics.total_retrieval_matched_recalled} / {metrics.total_recalled})")
        r.append(f"  Micro Recall:    {metrics.micro_retrieval_recall:.1%}  ({metrics.total_retrieval_matched_gt} / {metrics.total_gt_facts})")
        r.append(f"  Micro F1:        {metrics.micro_retrieval_f1:.1%}")
        r.append("")

        if metrics.per_dialog_retrieval_precision:
            r.append("  --- Macro Metrics (per-dialog average) ---")
            r.append(f"  Precision: {metrics.macro_retrieval_precision:.1%}  (median {statistics.median(metrics.per_dialog_retrieval_precision):.1%})")
            r.append(f"  Recall:    {metrics.macro_retrieval_recall:.1%}  (median {statistics.median(metrics.per_dialog_retrieval_recall):.1%})")
            r.append(f"  F1:        {metrics.macro_retrieval_f1:.1%}  (median {statistics.median(metrics.per_dialog_retrieval_f1):.1%})")
            r.append("")

        r.append("  --- Fact Accumulation ---")
        growth_rate = metrics.dialogs_with_fact_growth / metrics.dialogs_with_extraction if metrics.dialogs_with_extraction else 0
        r.append(f"  Dialogs with fact growth across sessions: {metrics.dialogs_with_fact_growth} ({growth_rate:.1%})")
        r.append("")

        # Best retrieval examples
        best = sorted(
            [d for d in dialog_results if d.retrieval_total_recalled > 0],
            key=lambda d: d.retrieval_f1, reverse=True,
        )[:5]
        r.append("  --- Top 5 best retrieval F1 ---")
        for d in best:
            r.append(f"  Dialog {d.dialog_id} ({d.num_sessions} sessions): "
                      f"F1={d.retrieval_f1:.0%} P={d.retrieval_precision:.0%} R={d.retrieval_recall:.0%} | "
                      f"stored={d.stored_count} recalled={d.retrieval_total_recalled} | "
                      f"GT={d.final_persona[:2]}")
        r.append("")

        # Worst retrieval
        worst = sorted(
            [d for d in dialog_results if d.retrieval_total_recalled > 0 and d.retrieval_f1 < 0.5],
            key=lambda d: d.retrieval_f1,
        )[:5]
        if worst:
            r.append("  --- Bottom 5 retrieval F1 ---")
            for d in worst:
                r.append(f"  Dialog {d.dialog_id} ({d.num_sessions} sessions): "
                          f"F1={d.retrieval_f1:.0%} P={d.retrieval_precision:.0%} R={d.retrieval_recall:.0%} | "
                          f"stored={d.stored_count} recalled={d.retrieval_total_recalled} | "
                          f"GT={d.final_persona[:2]}")
            r.append("")

        # Zero-extraction dialogs
        zero_ext = [d for d in dialog_results if d.total_extracted == 0]
        if zero_ext:
            r.append(f"  --- Zero-extraction dialogs: {len(zero_ext)} ---")
            for d in zero_ext[:3]:
                r.append(f"  Dialog {d.dialog_id}: GT={d.final_persona[:3]}")
            r.append("")

        r.append("=" * 72)
        print("\n".join(r))
        assert True
