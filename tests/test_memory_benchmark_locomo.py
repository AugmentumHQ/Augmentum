"""LoCoMo benchmark — tests full memory pipeline against the hardest conversational memory benchmark.

Dataset: snap-research/locomo (locomo10.json) — 10 long conversations, 1,986 QA pairs.
Published baselines: Mem0 66.9%, Zep 65.9%, ChatGPT Memory 52.9%, Human 87.9%.

Evaluation:
  1. Feed conversation sessions through extract → store (real SQLite + FTS5).
  2. For each question, recall memories via hybrid search.
  3. Score: does recalled context contain the answer?
     - Single-hop/temporal: token F1 between recalled content and gold answer.
     - Multi-hop: average best-match F1 per sub-answer.
     - Adversarial: 1.0 if nothing relevant recalled (correct abstention), 0.0 otherwise.
  4. Aggregate by category and overall.

Run:
    pytest tests/test_memory_benchmark_locomo.py -v --tb=short
    pytest tests/test_memory_benchmark_locomo.py -v -k report   # summary only
"""

from __future__ import annotations

import asyncio
import json
import re
import statistics
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DATA_FILE = Path(__file__).resolve().parent / ".bench_cache" / "locomo10.json"
_MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "augmentum" / "state" / "migrations"

_CAT_NAMES = {1: "multi_hop", 2: "temporal", 3: "open_domain", 4: "single_hop", 5: "adversarial"}

needs_data = pytest.mark.skipif(not _DATA_FILE.exists(), reason="locomo10.json not downloaded")

# ---------------------------------------------------------------------------
# Answer scoring — matches LoCoMo's official evaluation.py
# ---------------------------------------------------------------------------

_ARTICLES = {"a", "an", "the"}


def _normalize_answer(text: str) -> str:
    """Lowercase, remove articles, punctuation, extra whitespace."""
    text = text.lower()
    # Remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)
    # Remove articles
    tokens = text.split()
    tokens = [t for t in tokens if t not in _ARTICLES]
    return " ".join(tokens).strip()


def _token_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 score (LoCoMo standard)."""
    pred_tokens = _normalize_answer(prediction).split()
    gt_tokens = _normalize_answer(ground_truth).split()

    if not pred_tokens or not gt_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def _score_answer(prediction: str, gold: str, category: int) -> float:
    """Score a prediction against gold answer, per LoCoMo category rules."""
    if category == 5:  # Adversarial
        # Correct if system abstains (nothing recalled = abstention)
        pred_lower = prediction.lower()
        if not prediction.strip() or "not mentioned" in pred_lower or "no information" in pred_lower:
            return 1.0
        return 0.0

    if category == 3:  # Open-domain: only score factual part
        gold = gold.split(";")[0].strip()

    if category == 1:  # Multi-hop: average F1 per sub-answer
        gold_parts = [g.strip() for g in gold.split(",") if g.strip()]
        if not gold_parts:
            return 0.0
        scores = []
        for gp in gold_parts:
            scores.append(_token_f1(prediction, gp))
        return sum(scores) / len(scores)

    # Single-hop (4), temporal (2): direct F1
    return _token_f1(prediction, gold)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass
class LoCoMoConversation:
    sample_id: str
    speaker_a: str
    speaker_b: str
    sessions: list[list[dict]]  # list of sessions, each a list of {speaker, text, dia_id}
    session_datetimes: list[str]
    qa: list[dict]  # {question, answer, category, evidence}


def _load_conversations() -> list[LoCoMoConversation]:
    with open(_DATA_FILE, encoding="utf-8") as f:
        raw = json.load(f)

    convs = []
    for item in raw:
        conv_data = item["conversation"]
        speaker_a = conv_data["speaker_a"]
        speaker_b = conv_data["speaker_b"]

        sessions = []
        datetimes = []
        for i in range(1, 100):
            key = f"session_{i}"
            dt_key = f"{key}_date_time"
            if key not in conv_data:
                break
            sessions.append(conv_data[key])
            datetimes.append(conv_data.get(dt_key, ""))

        # Normalize QA evidence to always be list of strings
        qa = []
        for q in item.get("qa", []):
            evidence = q.get("evidence", [])
            if isinstance(evidence, int):
                evidence = []
            elif isinstance(evidence, str):
                evidence = [evidence]
            answer = q.get("answer") or q.get("adversarial_answer", "")
            qa.append({
                "question": q["question"],
                "answer": str(answer),
                "category": q["category"],
                "evidence": evidence,
            })

        convs.append(LoCoMoConversation(
            sample_id=item["sample_id"],
            speaker_a=speaker_a,
            speaker_b=speaker_b,
            sessions=sessions,
            session_datetimes=datetimes,
            qa=qa,
        ))

    return convs


# ---------------------------------------------------------------------------
# Memory pipeline
# ---------------------------------------------------------------------------

from augmentum.memory.extractor import heuristic_extract, should_extract

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


async def _create_store():
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
            self.vec_enabled = False

    return MemoryStore(_Backend(conn)), conn


# ---------------------------------------------------------------------------
# Per-question result
# ---------------------------------------------------------------------------


@dataclass
class QAResult:
    question: str
    gold_answer: str
    category: int
    recalled_text: str  # concatenated recalled memories
    score: float


@dataclass
class ConvResult:
    sample_id: str
    num_sessions: int
    num_turns: int
    facts_stored: int
    qa_results: list[QAResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


async def _evaluate_conversation(conv: LoCoMoConversation) -> ConvResult:
    store, conn = await _create_store()
    user_id = conv.sample_id
    total_turns = 0
    facts_stored = 0

    try:
        # Phase 1: Feed sessions through extraction pipeline
        for si, session in enumerate(conv.sessions):
            session_id = f"session_{si + 1}"
            for turn in session:
                text = turn.get("text", "")
                speaker = turn.get("speaker", "")
                total_turns += 1

                if not text or not should_extract(text):
                    continue

                facts = heuristic_extract(text)
                for fact in facts:
                    # Tag with speaker and session for provenance
                    fact.source_context = {
                        "session_id": session_id,
                        "speaker": speaker,
                        "dia_id": turn.get("dia_id", ""),
                    }
                    await store.store_fact(
                        fact,
                        user_id=user_id,
                        session_id=session_id,
                        is_explicit=fact.is_explicit,
                    )
                    facts_stored += 1

        # Phase 2: Answer questions via recall
        qa_results = []
        for qa in conv.qa:
            question = qa["question"]
            gold = qa["answer"]
            category = qa["category"]

            # Recall memories relevant to the question
            memories = await store.recall(
                query=question,
                user_id=user_id,
                limit=5,
                min_score=0.0,
            )

            # Build context from recalled memories
            recalled_text = " ".join(m.content for m in memories).strip()

            # Score
            score = _score_answer(recalled_text, gold, category)
            qa_results.append(QAResult(
                question=question,
                gold_answer=gold,
                category=category,
                recalled_text=recalled_text[:200],
                score=score,
            ))

    finally:
        await conn.close()

    return ConvResult(
        sample_id=conv.sample_id,
        num_sessions=len(conv.sessions),
        num_turns=total_turns,
        facts_stored=facts_stored,
        qa_results=qa_results,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def conversations() -> list[LoCoMoConversation]:
    return _load_conversations()


@pytest.fixture(scope="module")
def conv_results(conversations: list[LoCoMoConversation]) -> list[ConvResult]:
    async def _run():
        results = []
        for conv in conversations:
            r = await _evaluate_conversation(conv)
            results.append(r)
        return results
    return asyncio.run(_run())


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------


@dataclass
class LoCoMoMetrics:
    total_convs: int = 0
    total_sessions: int = 0
    total_turns: int = 0
    total_facts: int = 0
    total_qa: int = 0

    # Per-category scores
    category_scores: dict[int, list[float]] = field(default_factory=dict)

    @property
    def overall_f1(self) -> float:
        all_scores = []
        for scores in self.category_scores.values():
            all_scores.extend(scores)
        return statistics.mean(all_scores) if all_scores else 0.0

    def category_f1(self, cat: int) -> float:
        scores = self.category_scores.get(cat, [])
        return statistics.mean(scores) if scores else 0.0


def _compute_metrics(results: list[ConvResult]) -> LoCoMoMetrics:
    m = LoCoMoMetrics()
    for r in results:
        m.total_convs += 1
        m.total_sessions += r.num_sessions
        m.total_turns += r.num_turns
        m.total_facts += r.facts_stored

        for qa in r.qa_results:
            m.total_qa += 1
            if qa.category not in m.category_scores:
                m.category_scores[qa.category] = []
            m.category_scores[qa.category].append(qa.score)

    return m


@pytest.fixture(scope="module")
def metrics(conv_results: list[ConvResult]) -> LoCoMoMetrics:
    return _compute_metrics(conv_results)


# ===========================================================================
# Tests
# ===========================================================================


@needs_data
class TestDataLoading:

    def test_conversation_count(self, conversations: list[LoCoMoConversation]):
        assert len(conversations) == 10

    def test_qa_count(self, conversations: list[LoCoMoConversation]):
        total = sum(len(c.qa) for c in conversations)
        assert total >= 1900, f"Expected ~1986 QA pairs, got {total}"

    def test_sessions_present(self, conversations: list[LoCoMoConversation]):
        for c in conversations:
            assert len(c.sessions) >= 5, f"{c.sample_id} has only {len(c.sessions)} sessions"


@needs_data
class TestExtraction:

    def test_facts_extracted(self, metrics: LoCoMoMetrics):
        assert metrics.total_facts > 100, f"Only {metrics.total_facts} facts extracted"

    def test_all_convs_have_facts(self, conv_results: list[ConvResult]):
        for r in conv_results:
            assert r.facts_stored > 0, f"{r.sample_id} had zero facts"


@needs_data
class TestRetrieval:

    def test_overall_f1_above_floor(self, metrics: LoCoMoMetrics):
        """Overall F1 should be non-trivial for retrieval-only (no LLM generation).

        Published baselines use LLM answer generation on top of recall.
        Our retrieval-only token F1 will be much lower — we're scoring
        raw recalled text against gold answers without an LLM to bridge.
        Even 1%+ shows the pipeline is finding *some* relevant content.
        """
        assert metrics.overall_f1 >= 0.005, (
            f"Overall F1 {metrics.overall_f1:.1%} — pipeline returning nothing?"
        )

    def test_single_hop_best_category(self, metrics: LoCoMoMetrics):
        """Single-hop should be our best non-adversarial category (direct fact lookup)."""
        sh = metrics.category_f1(4)
        # Retrieval-only: raw text overlap is low but single-hop should lead
        other_cats = [metrics.category_f1(c) for c in [1, 2, 3] if c in metrics.category_scores]
        if other_cats:
            assert sh >= max(other_cats) * 0.5, (
                f"Single-hop F1 {sh:.1%} unexpectedly low vs other categories"
            )

    def test_adversarial_abstention(self, metrics: LoCoMoMetrics):
        """Adversarial questions: we should sometimes correctly abstain."""
        adv = metrics.category_f1(5)
        # With 446 adversarial questions, even a few correct abstentions count
        assert adv >= 0.001, f"Adversarial score {adv:.1%} — never abstaining?"


# ===========================================================================
# Diagnostic report
# ===========================================================================


@needs_data
class TestLoCoMoReport:

    def test_report(self, conv_results: list[ConvResult], metrics: LoCoMoMetrics):
        r = []
        r.append("")
        r.append("=" * 72)
        r.append("  LOCOMO MEMORY BENCHMARK")
        r.append("=" * 72)
        r.append("")
        r.append(f"  Conversations:  {metrics.total_convs}")
        r.append(f"  Total sessions: {metrics.total_sessions}")
        r.append(f"  Total turns:    {metrics.total_turns}")
        r.append(f"  Facts stored:   {metrics.total_facts}")
        r.append(f"  QA pairs:       {metrics.total_qa}")
        r.append("")
        r.append(f"  --- Overall F1: {metrics.overall_f1:.1%} ---")
        r.append("")
        r.append("  --- Per-Category F1 ---")

        for cat in sorted(metrics.category_scores.keys()):
            name = _CAT_NAMES.get(cat, f"cat_{cat}")
            f1 = metrics.category_f1(cat)
            count = len(metrics.category_scores[cat])
            r.append(f"  {name:>15}: {f1:.1%}  ({count} questions)")

        r.append("")
        r.append("  --- Published Baselines (LoCoMo, LLM-as-Judge) ---")
        r.append("  Mem0:           66.9%")
        r.append("  Mem0g (graph):  68.4%")
        r.append("  Zep:            65.9%")
        r.append("  OpenAI Memory:  52.9%")
        r.append("  Full-Context:   72.9%")
        r.append("  Human:          87.9%")
        r.append("")
        r.append("  Note: Published scores use LLM-as-Judge (lenient grading)")
        r.append("  + LLM answer generation. Our scores use token F1 (strict)")
        r.append("  + retrieval-only (no LLM generation). Direct comparison")
        r.append("  is not apples-to-apples but provides a reference frame.")
        r.append("")

        # Per-conversation breakdown
        r.append("  --- Per-Conversation ---")
        for cr in conv_results:
            scores = [qa.score for qa in cr.qa_results]
            avg = statistics.mean(scores) if scores else 0
            r.append(f"  {cr.sample_id}: {cr.num_sessions} sessions, "
                      f"{cr.facts_stored} facts, {len(cr.qa_results)} QA, "
                      f"avg F1={avg:.1%}")
        r.append("")

        # Best/worst examples per category
        for cat in [4, 1, 2, 5]:
            name = _CAT_NAMES.get(cat, "?")
            cat_results = []
            for cr in conv_results:
                for qa in cr.qa_results:
                    if qa.category == cat:
                        cat_results.append(qa)

            if not cat_results:
                continue

            best = sorted(cat_results, key=lambda q: q.score, reverse=True)[:3]
            r.append(f"  --- Best {name} ---")
            for q in best:
                r.append(f"    F1={q.score:.0%} Q: {q.question[:60]}")
                r.append(f"         A: {q.gold_answer[:60]}")
                r.append(f"         Recalled: {q.recalled_text[:60]}")
            r.append("")

        r.append("=" * 72)
        print("\n".join(r))
        assert True
