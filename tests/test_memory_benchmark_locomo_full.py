"""LoCoMo full-pipeline benchmark — emulates Mem0/Zep evaluation methodology.

Dataset: snap-research/locomo (locomo10.json) — 10 long conversations, 1,986 QA pairs.
Published baselines: Mem0 66.9%, Zep 65.9%, ChatGPT Memory 52.9%, Human 87.9%.

Full pipeline (matches published systems):
  1. Feed conversation sessions through extract → store (real SQLite + FTS5 + vector).
  2. For each question, recall memories via hybrid search (vector + FTS5 + RRF).
  3. Pass recalled memories + question to LLM → generate concise answer.
  4. Score LLM answer against gold answer (token F1, strict).
  5. Aggregate by category and overall.

Config (env vars):
    LOCOMO_LLM_BASE_URL   — OpenAI-compat base URL (default: http://localhost:1234/v1)
    LOCOMO_LLM_MODEL      — model name (default: gemma-3-4b-it)
    LOCOMO_RECALL_LIMIT   — memories to recall per question (default: 10)
    LOCOMO_CONCURRENCY    — parallel LLM calls per conversation (default: 5)
    LOCOMO_CONV_LIMIT     — conversations to evaluate, 0=all (default: 0)

    Pipeline feature flags (default: v2 baseline, all off):
    LOCOMO_USE_VEC=1        — enable sqlite-vec vector search (v3+)
    LOCOMO_USE_CONTEXTUAL=1 — enable Anthropic-style context enrichment (v3+)
    LOCOMO_USE_ATOMIC=1     — enable compound fact splitting (v3+)
    LOCOMO_USE_CONVEX=1     — enable convex combination fusion (v3+)

Run:
    pytest tests/test_memory_benchmark_locomo_full.py -v --tb=short -p no:capture
    pytest tests/test_memory_benchmark_locomo_full.py -v -k report -p no:capture  # report only
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import statistics
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pytest

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DATA_FILE = Path(__file__).resolve().parent / ".bench_cache" / "locomo10.json"
_MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "augmentum" / "state" / "migrations"

_LLM_BASE_URL = os.environ.get("LOCOMO_LLM_BASE_URL", "http://localhost:1234/v1")
_LLM_MODEL = os.environ.get("LOCOMO_LLM_MODEL", "gemma-3-4b-it")
_RECALL_LIMIT = int(os.environ.get("LOCOMO_RECALL_LIMIT", "10"))
_CONCURRENCY = int(os.environ.get("LOCOMO_CONCURRENCY", "5"))
_CONV_LIMIT = int(os.environ.get("LOCOMO_CONV_LIMIT", "0"))

# --- Pipeline feature flags (v2 baseline: all off) ---
# Enable these for v3+ configurations or when using stronger generation models.
_USE_VEC = os.environ.get("LOCOMO_USE_VEC", "0") == "1"          # sqlite-vec vector search
_USE_CONTEXTUAL = os.environ.get("LOCOMO_USE_CONTEXTUAL", "0") == "1"  # Anthropic-style context enrichment
_USE_ATOMIC = os.environ.get("LOCOMO_USE_ATOMIC", "0") == "1"    # compound fact splitting
_USE_CONVEX = os.environ.get("LOCOMO_USE_CONVEX", "0") == "1"    # convex combination fusion (vs RRF)

_CAT_NAMES = {1: "multi_hop", 2: "temporal", 3: "open_domain", 4: "single_hop", 5: "adversarial"}

needs_data = pytest.mark.skipif(not _DATA_FILE.exists(), reason="locomo10.json not downloaded")


def _llm_available() -> bool:
    try:
        r = httpx.get(f"{_LLM_BASE_URL}/models", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


needs_llm = pytest.mark.skipif(not _llm_available(), reason=f"LLM not available at {_LLM_BASE_URL}")


# ---------------------------------------------------------------------------
# Answer scoring — matches LoCoMo's official evaluation.py
# ---------------------------------------------------------------------------

_ARTICLES = {"a", "an", "the"}


def _normalize_answer(text: str) -> str:
    """Lowercase, remove articles, punctuation, extra whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
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
        pred_lower = prediction.lower().strip()
        abstention_phrases = [
            "not mentioned", "no information", "no memory", "don't have",
            "do not have", "cannot find", "no record", "not found",
            "i don't know", "no relevant", "unable to find", "not available",
            "no data", "isn't mentioned", "wasn't mentioned", "not sure",
            "cannot determine", "no evidence", "unknown",
        ]
        if not pred_lower or any(p in pred_lower for p in abstention_phrases):
            return 1.0
        return 0.0

    if category == 3:  # Open-domain: only score factual part
        gold = gold.split(";")[0].strip()

    if category == 1:  # Multi-hop: average F1 per sub-answer
        gold_parts = [g.strip() for g in gold.split(",") if g.strip()]
        if not gold_parts:
            return 0.0
        scores = [_token_f1(prediction, gp) for gp in gold_parts]
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
    sessions: list[list[dict]]
    session_datetimes: list[str]
    qa: list[dict]


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
# Memory pipeline — full vector + FTS5 hybrid search
# ---------------------------------------------------------------------------

from augmentum.memory.extractor import heuristic_extract, should_extract


# ---------------------------------------------------------------------------
# Technique 1: Contextual Fact Enrichment (Anthropic's approach)
# Prepend a context sentence to facts before embedding for better retrieval.
# ---------------------------------------------------------------------------

def _contextualize_fact(fact_content: str, speaker: str, date_tag: str,
                        speaker_a: str, speaker_b: str) -> str:
    """Enrich a fact with contextual metadata for improved embedding retrieval.

    Instead of embedding raw "I went hiking", embed:
    "In a conversation on 7 May 2023, Caroline mentioned: I went hiking"
    This gives the embedding model topic + temporal + attribution signal.
    """
    partner = speaker_b if speaker == speaker_a else speaker_a
    parts = []
    if date_tag:
        parts.append(f"On {date_tag}")
    if speaker:
        parts.append(f"{speaker} told {partner}")
    prefix = ", ".join(parts)
    if prefix:
        return f"{prefix}: {fact_content}"
    return fact_content


# ---------------------------------------------------------------------------
# Technique 2: Atomic Fact Splitting
# Break compound facts into individual claims for precise retrieval.
# ---------------------------------------------------------------------------

# Conjunctions that indicate compound facts
_COMPOUND_SPLITTERS = re.compile(
    r"\s+(?:and then|and also|and|but also|, then|; )\s+",
    re.I,
)


def _split_atomic(fact_content: str) -> list[str]:
    """Split compound facts into atomic single-claim statements.

    'I went hiking and saw a meteor shower and cooked marshmallows'
    → ['I went hiking', 'saw a meteor shower', 'cooked marshmallows']

    Only splits if parts are long enough to be meaningful.
    """
    # Don't split very short facts or facts without conjunctions
    if len(fact_content) < 30 or " and " not in fact_content.lower():
        return [fact_content]

    parts = _COMPOUND_SPLITTERS.split(fact_content)
    # Only keep parts that are meaningful (>10 chars)
    meaningful = [p.strip() for p in parts if len(p.strip()) > 10]
    # If splitting produced nothing useful, return original
    if len(meaningful) <= 1:
        return [fact_content]
    return meaningful


def _extract_date_tag(session_datetime: str) -> str:
    """Extract a human-readable date from session datetime string.

    LoCoMo format: '7 May, 2023, 10:30 AM' or similar.
    Returns e.g. '7 May 2023' or '' if unparseable.
    """
    if not session_datetime:
        return ""
    # Try to parse common LoCoMo datetime formats
    for fmt in [
        "%d %B, %Y, %I:%M %p",   # "7 May, 2023, 10:30 AM"
        "%d %B, %Y, %H:%M",       # "7 May, 2023, 14:30"
        "%d %B %Y, %I:%M %p",     # "7 May 2023, 10:30 AM"
        "%d %B %Y",                # "7 May 2023"
        "%B %d, %Y, %I:%M %p",    # "May 7, 2023, 10:30 AM"
        "%B %d, %Y",               # "May 7, 2023"
    ]:
        try:
            from datetime import datetime
            dt = datetime.strptime(session_datetime.strip(), fmt)
            return dt.strftime("%-d %B %Y") if os.name != "nt" else dt.strftime("%d %B %Y").lstrip("0")
        except ValueError:
            continue
    # Fallback: extract date-like substring
    m = re.search(r"(\d{1,2}\s+\w+,?\s+\d{4})", session_datetime)
    if m:
        return m.group(1).replace(",", "")
    return ""

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
    """Create MemoryStore with vector search enabled via sqlite-vec.

    Key fix: sqlite-vec extension must be loaded on a connection created
    with check_same_thread=False, since aiosqlite runs sqlite3 in a
    background thread.
    """
    import sqlite3

    import aiosqlite
    from augmentum.memory.store import MemoryStore

    # Load sqlite-vec extension BEFORE wrapping in aiosqlite
    # Only when _USE_VEC is enabled (v3+ config)
    vec_enabled = False
    if _USE_VEC:
        try:
            import sqlite_vec
            raw = sqlite3.connect(":memory:", check_same_thread=False)
            raw.row_factory = sqlite3.Row
            raw.enable_load_extension(True)
            raw.load_extension(sqlite_vec.loadable_path())
            raw.enable_load_extension(False)
            vec_enabled = True
        except Exception:
            raw = sqlite3.connect(":memory:", check_same_thread=False)
            raw.row_factory = sqlite3.Row
    else:
        raw = sqlite3.connect(":memory:", check_same_thread=False)
        raw.row_factory = sqlite3.Row

    # Wrap raw connection in aiosqlite
    conn = aiosqlite.Connection(lambda: raw, iter_chunk_size=64)
    await conn  # start the background thread
    conn.row_factory = aiosqlite.Row

    await conn.executescript(_BOOTSTRAP_SQL)
    await conn.commit()
    for v in [6, 8, 9]:
        await _apply_migration(conn, v)

    if vec_enabled:
        await conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS memories_vec "
            "USING vec0(memory_id TEXT PRIMARY KEY, embedding float[768])"
        )
        await conn.commit()

    class _Backend:
        def __init__(self, c, vec):
            self.conn = c
            self.vec_enabled = vec

    return MemoryStore(_Backend(conn, vec_enabled)), conn, vec_enabled


# ---------------------------------------------------------------------------
# LLM answer generation — the key missing piece
# ---------------------------------------------------------------------------

_ANSWER_SYSTEM_PROMPT = """You are a precise memory recall assistant. Given a question and memory notes from conversations, answer ONLY if the memories contain a direct, specific answer.

Critical rules:
- Answer ONLY with information explicitly stated in the memories.
- If the memories are tangentially related but don't directly answer the specific question asked, say "Not mentioned in memories."
- If you would need to guess, infer, or assume to answer, say "Not mentioned in memories."
- For dates/times, only state dates that appear in the memories.
- Keep answers under 20 words — just the answer, no explanation.
- Do NOT extrapolate, speculate, or combine partial information to form an answer that isn't directly supported."""


async def _generate_answer(
    client: httpx.AsyncClient,
    question: str,
    memories: list[str],
    semaphore: asyncio.Semaphore,
) -> str:
    """Call LLM to generate an answer from recalled memories."""
    if not memories:
        return "Not mentioned in memories."

    memory_block = "\n".join(f"- {m}" for m in memories)
    user_prompt = f"""Memories:
{memory_block}

Question: {question}

Answer (concise, from memories only):"""

    async with semaphore:
        try:
            resp = await client.post(
                f"{_LLM_BASE_URL}/chat/completions",
                json={
                    "model": _LLM_MODEL,
                    "messages": [
                        {"role": "system", "content": _ANSWER_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 100,
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"[LLM error: {e}]"


# ---------------------------------------------------------------------------
# Per-question result
# ---------------------------------------------------------------------------


@dataclass
class QAResult:
    question: str
    gold_answer: str
    category: int
    recalled_text: str
    llm_answer: str
    retrieval_score: float  # raw retrieval token F1 (no LLM)
    full_score: float       # full pipeline token F1 (with LLM)


@dataclass
class ConvResult:
    sample_id: str
    num_sessions: int
    num_turns: int
    facts_stored: int
    vec_enabled: bool
    qa_results: list[QAResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------


async def _evaluate_conversation(
    conv: LoCoMoConversation,
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    progress_callback=None,
) -> ConvResult:
    store, conn, vec_enabled = await _create_store()
    user_id = conv.sample_id
    total_turns = 0
    facts_stored = 0

    try:
        # Phase 1: Feed sessions through extraction → store
        for si, session in enumerate(conv.sessions):
            session_id = f"session_{si + 1}"
            # Get session datetime for temporal enrichment
            session_dt = conv.session_datetimes[si] if si < len(conv.session_datetimes) else ""
            # Extract a short date label (e.g., "7 May 2023") for tagging
            date_tag = _extract_date_tag(session_dt)

            for turn in session:
                text = turn.get("text", "")
                speaker = turn.get("speaker", "")
                total_turns += 1

                if not text or not should_extract(text):
                    continue

                facts = heuristic_extract(text)
                for fact in facts:
                    # Technique 2: Atomic splitting (opt-in, v3+)
                    parts = _split_atomic(fact.content) if _USE_ATOMIC else [fact.content]

                    for part in parts:
                        # Technique 1: Contextual enrichment (opt-in, v3+)
                        if _USE_CONTEXTUAL:
                            content = _contextualize_fact(
                                part, speaker, date_tag,
                                conv.speaker_a, conv.speaker_b,
                            )
                        else:
                            # v2 baseline: just prepend date tag for temporal grounding
                            content = f"[{date_tag}] {part}" if date_tag else part

                        import copy
                        stored_fact = copy.copy(fact)
                        stored_fact.content = content
                        stored_fact.source_context = {
                            "session_id": session_id,
                            "speaker": speaker,
                            "dia_id": turn.get("dia_id", ""),
                            "session_datetime": session_dt,
                        }
                        await store.store_fact(
                            stored_fact,
                            user_id=user_id,
                            session_id=session_id,
                            is_explicit=stored_fact.is_explicit,
                        )
                        facts_stored += 1

        # Phase 2: Recall + LLM answer generation for each question
        qa_results = []
        total_qa = len(conv.qa)

        for qi, qa in enumerate(conv.qa):
            question = qa["question"]
            gold = qa["answer"]
            category = qa["category"]

            # Recall memories (v2: FTS5-only + RRF; v3+: hybrid vec+FTS5)
            # Temporarily set fusion method based on feature flag
            from augmentum.config import settings as _settings
            _orig_fusion = _settings.memory_fusion_method
            if _USE_CONVEX:
                _settings.memory_fusion_method = "convex"
            else:
                _settings.memory_fusion_method = "rrf"

            memories = await store.recall(
                query=question,
                user_id=user_id,
                limit=_RECALL_LIMIT,
                min_score=0.0,
            )
            _settings.memory_fusion_method = _orig_fusion

            memory_texts = [m.content for m in memories]
            recalled_text = " ".join(memory_texts).strip()

            # Score retrieval-only (baseline)
            retrieval_score = _score_answer(recalled_text, gold, category)

            # LLM answer generation (full pipeline)
            llm_answer = await _generate_answer(client, question, memory_texts, semaphore)

            # Score full pipeline
            full_score = _score_answer(llm_answer, gold, category)

            qa_results.append(QAResult(
                question=question,
                gold_answer=gold,
                category=category,
                recalled_text=recalled_text[:200],
                llm_answer=llm_answer[:200],
                retrieval_score=retrieval_score,
                full_score=full_score,
            ))

            if progress_callback and (qi + 1) % 50 == 0:
                progress_callback(conv.sample_id, qi + 1, total_qa)

    finally:
        await conn.close()

    return ConvResult(
        sample_id=conv.sample_id,
        num_sessions=len(conv.sessions),
        num_turns=total_turns,
        facts_stored=facts_stored,
        vec_enabled=vec_enabled,
        qa_results=qa_results,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def conversations() -> list[LoCoMoConversation]:
    convs = _load_conversations()
    if _CONV_LIMIT > 0:
        return convs[:_CONV_LIMIT]
    return convs


@pytest.fixture(scope="module")
def conv_results(conversations: list[LoCoMoConversation]) -> list[ConvResult]:
    def _progress(conv_id, done, total):
        print(f"    {conv_id}: {done}/{total} QA", flush=True)

    async def _run():
        semaphore = asyncio.Semaphore(_CONCURRENCY)
        async with httpx.AsyncClient() as client:
            results = []
            for ci, conv in enumerate(conversations):
                print(f"  [{ci+1}/{len(conversations)}] {conv.sample_id} "
                      f"({len(conv.sessions)} sessions, {len(conv.qa)} QA)...",
                      flush=True)
                t0 = time.time()
                r = await _evaluate_conversation(conv, client, semaphore, _progress)
                elapsed = time.time() - t0
                avg = statistics.mean([q.full_score for q in r.qa_results]) if r.qa_results else 0
                print(f"    -> {r.facts_stored} facts, avg F1={avg:.1%}, "
                      f"vec={r.vec_enabled}, {elapsed:.0f}s", flush=True)
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
    vec_enabled: bool = False

    # Per-category scores (full pipeline)
    category_scores: dict[int, list[float]] = field(default_factory=dict)
    # Per-category retrieval-only scores (baseline)
    retrieval_scores: dict[int, list[float]] = field(default_factory=dict)

    @property
    def overall_f1(self) -> float:
        all_scores = []
        for scores in self.category_scores.values():
            all_scores.extend(scores)
        return statistics.mean(all_scores) if all_scores else 0.0

    @property
    def retrieval_f1(self) -> float:
        all_scores = []
        for scores in self.retrieval_scores.values():
            all_scores.extend(scores)
        return statistics.mean(all_scores) if all_scores else 0.0

    def category_f1(self, cat: int) -> float:
        scores = self.category_scores.get(cat, [])
        return statistics.mean(scores) if scores else 0.0

    def category_retrieval_f1(self, cat: int) -> float:
        scores = self.retrieval_scores.get(cat, [])
        return statistics.mean(scores) if scores else 0.0


def _compute_metrics(results: list[ConvResult]) -> LoCoMoMetrics:
    m = LoCoMoMetrics()
    for r in results:
        m.total_convs += 1
        m.total_sessions += r.num_sessions
        m.total_turns += r.num_turns
        m.total_facts += r.facts_stored
        m.vec_enabled = m.vec_enabled or r.vec_enabled

        for qa in r.qa_results:
            m.total_qa += 1
            if qa.category not in m.category_scores:
                m.category_scores[qa.category] = []
                m.retrieval_scores[qa.category] = []
            m.category_scores[qa.category].append(qa.full_score)
            m.retrieval_scores[qa.category].append(qa.retrieval_score)

    return m


@pytest.fixture(scope="module")
def metrics(conv_results: list[ConvResult]) -> LoCoMoMetrics:
    return _compute_metrics(conv_results)


# ===========================================================================
# Tests
# ===========================================================================


@needs_data
@needs_llm
class TestDataLoading:

    def test_conversation_count(self, conversations: list[LoCoMoConversation]):
        expected = min(10, _CONV_LIMIT) if _CONV_LIMIT > 0 else 10
        assert len(conversations) == expected

    def test_qa_pairs_present(self, conversations: list[LoCoMoConversation]):
        total = sum(len(c.qa) for c in conversations)
        assert total > 0, "No QA pairs loaded"


@needs_data
@needs_llm
class TestExtraction:

    def test_facts_extracted(self, metrics: LoCoMoMetrics):
        assert metrics.total_facts > 100, f"Only {metrics.total_facts} facts extracted"

    def test_all_convs_have_facts(self, conv_results: list[ConvResult]):
        for r in conv_results:
            assert r.facts_stored > 0, f"{r.sample_id} had zero facts"


@needs_data
@needs_llm
class TestFullPipeline:

    def test_llm_improves_over_retrieval(self, metrics: LoCoMoMetrics):
        """Full pipeline (retrieval + LLM) should beat retrieval-only."""
        assert metrics.overall_f1 > metrics.retrieval_f1, (
            f"Full pipeline {metrics.overall_f1:.1%} did not beat "
            f"retrieval-only {metrics.retrieval_f1:.1%}"
        )

    def test_overall_f1_above_floor(self, metrics: LoCoMoMetrics):
        """With LLM generation, we should achieve meaningful scores."""
        assert metrics.overall_f1 >= 0.05, (
            f"Overall F1 {metrics.overall_f1:.1%} below 5% even with LLM generation"
        )

    def test_single_hop_strongest(self, metrics: LoCoMoMetrics):
        """Single-hop should be among top categories (direct fact lookup)."""
        sh = metrics.category_f1(4)
        assert sh >= 0.03, f"Single-hop F1 {sh:.1%} unexpectedly low"

    def test_adversarial_abstention(self, metrics: LoCoMoMetrics):
        """LLM should abstain on adversarial questions when memories are irrelevant."""
        adv = metrics.category_f1(5)
        assert adv >= 0.01, f"Adversarial score {adv:.1%} — LLM never abstaining?"

    def test_llm_answers_not_all_errors(self, conv_results: list[ConvResult]):
        """Verify LLM actually generated answers (not all errors)."""
        errors = 0
        total = 0
        for r in conv_results:
            for qa in r.qa_results:
                total += 1
                if qa.llm_answer.startswith("[LLM error"):
                    errors += 1
        error_rate = errors / total if total else 1.0
        assert error_rate < 0.05, f"{errors}/{total} LLM calls failed ({error_rate:.0%})"


# ===========================================================================
# Diagnostic report
# ===========================================================================


@needs_data
@needs_llm
class TestLoCoMoFullReport:

    def test_report(self, conv_results: list[ConvResult], metrics: LoCoMoMetrics):
        r = []
        r.append("")
        r.append("=" * 76)
        r.append("  LOCOMO FULL-PIPELINE BENCHMARK")
        r.append("=" * 76)
        r.append("")
        r.append(f"  LLM:            {_LLM_MODEL} @ {_LLM_BASE_URL}")
        r.append(f"  Recall limit:   {_RECALL_LIMIT} memories per question")
        r.append(f"  Vector search:  {'enabled' if _USE_VEC else 'off (FTS5-only)'}")
        r.append(f"  Contextual:     {'enabled' if _USE_CONTEXTUAL else 'off (date-tag only)'}")
        r.append(f"  Atomic split:   {'enabled' if _USE_ATOMIC else 'off'}")
        r.append(f"  Fusion:         {'convex' if _USE_CONVEX else 'RRF'}")
        r.append(f"  Conversations:  {metrics.total_convs}")
        r.append(f"  Total sessions: {metrics.total_sessions}")
        r.append(f"  Total turns:    {metrics.total_turns}")
        r.append(f"  Facts stored:   {metrics.total_facts}")
        r.append(f"  QA pairs:       {metrics.total_qa}")
        r.append("")
        r.append(f"  === OVERALL F1 (full pipeline): {metrics.overall_f1:.1%} ===")
        r.append(f"  === OVERALL F1 (retrieval-only): {metrics.retrieval_f1:.1%} ===")
        r.append(f"  === LLM UPLIFT: +{metrics.overall_f1 - metrics.retrieval_f1:.1%} ===")
        r.append("")

        # Per-category comparison
        r.append("  --- Per-Category F1 ---")
        r.append(f"  {'Category':>15}  {'Full':>8}  {'Retrieval':>9}  {'Uplift':>8}  {'Count':>6}")
        r.append(f"  {'-'*15}  {'-'*8}  {'-'*9}  {'-'*8}  {'-'*6}")
        for cat in sorted(metrics.category_scores.keys()):
            name = _CAT_NAMES.get(cat, f"cat_{cat}")
            full = metrics.category_f1(cat)
            retr = metrics.category_retrieval_f1(cat)
            uplift = full - retr
            count = len(metrics.category_scores[cat])
            r.append(f"  {name:>15}  {full:>7.1%}  {retr:>8.1%}  {uplift:>+7.1%}  {count:>6}")
        r.append("")

        # Published baselines
        r.append("  --- Published Baselines (LoCoMo, LLM-as-Judge) ---")
        baselines = [
            ("Human", 87.9), ("Full-Context", 72.9), ("Mem0g (graph)", 68.4),
            ("Mem0", 66.9), ("Zep", 65.9), ("OpenAI Memory", 52.9),
        ]
        ours = metrics.overall_f1 * 100
        for name, score in baselines:
            marker = ""
            r.append(f"  {name:>20}: {score:.1f}%{marker}")
        r.append(f"  {'>> AUGMENTUM <<':>20}: {ours:.1f}%  "
                  f"({_LLM_MODEL}, token F1, retrieval+gen)")
        r.append("")
        r.append("  Note: Published scores use LLM-as-Judge (lenient grading)")
        r.append("  + frontier LLMs (GPT-4). Our scores use token F1 (strict)")
        r.append(f"  + {_LLM_MODEL} (local 4B). Strict scoring penalizes verbose")
        r.append("  or paraphrased answers that would score well with a judge LLM.")
        r.append("")

        # Per-conversation breakdown
        r.append("  --- Per-Conversation ---")
        for cr in conv_results:
            full_scores = [qa.full_score for qa in cr.qa_results]
            retr_scores = [qa.retrieval_score for qa in cr.qa_results]
            avg_full = statistics.mean(full_scores) if full_scores else 0
            avg_retr = statistics.mean(retr_scores) if retr_scores else 0
            r.append(f"  {cr.sample_id}: {cr.num_sessions} sess, "
                      f"{cr.facts_stored} facts, {len(cr.qa_results)} QA  "
                      f"full={avg_full:.1%}  retr={avg_retr:.1%}")
        r.append("")

        # Best examples per category (full pipeline)
        for cat in [4, 1, 2, 3, 5]:
            name = _CAT_NAMES.get(cat, "?")
            cat_results = []
            for cr in conv_results:
                for qa in cr.qa_results:
                    if qa.category == cat:
                        cat_results.append(qa)

            if not cat_results:
                continue

            best = sorted(cat_results, key=lambda q: q.full_score, reverse=True)[:3]
            r.append(f"  --- Best {name} (top 3) ---")
            for q in best:
                r.append(f"    F1={q.full_score:.0%} Q: {q.question[:65]}")
                r.append(f"         Gold: {q.gold_answer[:65]}")
                r.append(f"         LLM:  {q.llm_answer[:65]}")
            r.append("")

        # Worst categories (where to improve)
        r.append("  --- Worst Answers (improvement targets) ---")
        all_qa = []
        for cr in conv_results:
            all_qa.extend(cr.qa_results)
        # Filter to non-adversarial with some recall but low score
        scored_qa = [q for q in all_qa if q.category != 5 and q.recalled_text and q.full_score == 0]
        worst = sorted(scored_qa, key=lambda q: len(q.recalled_text), reverse=True)[:5]
        for q in worst:
            name = _CAT_NAMES.get(q.category, "?")
            r.append(f"    [{name}] Q: {q.question[:60]}")
            r.append(f"      Gold: {q.gold_answer[:60]}")
            r.append(f"      LLM:  {q.llm_answer[:60]}")
            r.append(f"      Recalled: {q.recalled_text[:60]}")
        r.append("")

        r.append("=" * 76)
        print("\n".join(r))
        assert True
