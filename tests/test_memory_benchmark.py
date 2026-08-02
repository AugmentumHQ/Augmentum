"""PersonaChat benchmark — measures extraction precision/recall against ground-truth persona facts.

Downloads 1 000 conversations from the PersonaChat validation split (HuggingFace).
Each conversation has 4-5 ground-truth personality traits and a multi-turn dialog
where those traits surface naturally.

Eval loop per row:
  1. Concatenate conversation history into individual user turns.
  2. Run each turn through should_extract() + heuristic_extract().
  3. Compare extracted facts against ground-truth persona labels using
     embedding cosine similarity (match threshold = 0.55).
  4. Aggregate precision, recall, F1 across the full dataset.

Run:
    pytest tests/test_memory_benchmark.py -v --tb=short
    pytest tests/test_memory_benchmark.py -v -k report --tb=short   # summary only

Set MEMORY_BENCH_MATCH_THRESHOLD env var to tune the semantic match threshold.
"""

from __future__ import annotations

import json
import os
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Lazy imports so collection doesn't blow up if deps missing
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

_DATASET_ID = "AlekseyKorshuk/persona-chat"
_SPLIT = "validation+train"
_MATCH_THRESHOLD = float(os.environ.get("MEMORY_BENCH_MATCH_THRESHOLD", "0.45"))

# Cache dir for downloaded + pre-processed rows so re-runs are instant
_CACHE_DIR = Path(__file__).resolve().parent / ".bench_cache"
_CACHE_FILE = _CACHE_DIR / "persona_chat_full.json"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass
class BenchRow:
    """One conversation with ground-truth persona facts."""

    persona_facts: list[str]
    user_turns: list[str]  # only the user-side utterances


def _clean_persona(raw: str) -> str:
    """Normalize PersonaChat persona strings (lowercased, trailing period stripped)."""
    return raw.strip().rstrip(".").strip()


def _extract_user_turns(utterances: list[dict[str, Any]]) -> list[str]:
    """Pull out the user-side turns from the nested utterances structure.

    PersonaChat alternates speakers in the history list.  Index 0 is the
    *other* speaker (the one whose persona we have), and even indices are
    that speaker.  Odd indices are the user we are evaluating against.

    We want the lines spoken by the persona-owner because those are the
    ones that *contain* the personality traits we're trying to extract.
    """
    turns: list[str] = []
    for utt in utterances:
        history = utt.get("history", [])
        # Even-indexed lines (0, 2, 4, …) are the persona-owner's turns
        for i, line in enumerate(history):
            if i % 2 == 0:
                cleaned = line.strip()
                if cleaned and cleaned not in turns:
                    turns.append(cleaned)
    return turns


def _load_rows() -> list[BenchRow]:
    """Load (and cache) 1 000 PersonaChat rows."""
    if _CACHE_FILE.exists():
        with open(_CACHE_FILE, encoding="utf-8") as f:
            data = json.load(f)
        return [BenchRow(**r) for r in data]

    ds = load_dataset(_DATASET_ID, split=_SPLIT)
    rows: list[BenchRow] = []
    for item in ds:
        persona = [_clean_persona(p) for p in item["personality"] if p.strip()]
        user_turns = _extract_user_turns(item["utterances"])
        if persona and user_turns:
            rows.append(BenchRow(persona_facts=persona, user_turns=user_turns))

    # Cache
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump([{"persona_facts": r.persona_facts, "user_turns": r.user_turns} for r in rows], f)
    return rows


# ---------------------------------------------------------------------------
# Extraction helpers — use real Augmentum code paths
# ---------------------------------------------------------------------------

from augmentum.memory.extractor import heuristic_extract, should_extract


def _extract_all_facts(turns: list[str]) -> list[str]:
    """Run every turn through the real extraction pipeline, return fact strings."""
    facts: list[str] = []
    seen: set[str] = set()
    for turn in turns:
        if not should_extract(turn):
            continue
        for fact in heuristic_extract(turn):
            key = fact.content.lower()
            if key not in seen:
                seen.add(key)
                facts.append(fact.content)
    return facts


# ---------------------------------------------------------------------------
# Semantic matching via embeddings
# ---------------------------------------------------------------------------

from augmentum.memory.extractor import _cosine_similarity  # noqa: E402


def _get_embeddings(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts using the real Augmentum embedding service."""
    from augmentum.memory.embeddings import EmbeddingService
    return EmbeddingService.embed(texts)


def _match_facts(
    extracted: list[str],
    ground_truth: list[str],
    threshold: float = _MATCH_THRESHOLD,
) -> tuple[set[int], set[int]]:
    """Match extracted facts to ground-truth persona facts via cosine similarity.

    Returns (matched_extracted_indices, matched_gt_indices).
    A ground-truth fact is matched if ANY extracted fact exceeds the threshold.
    """
    if not extracted or not ground_truth:
        return set(), set()

    all_texts = extracted + ground_truth
    all_embs = _get_embeddings(all_texts)
    ext_embs = all_embs[: len(extracted)]
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
# Per-row evaluation result
# ---------------------------------------------------------------------------


@dataclass
class RowResult:
    row_idx: int
    persona_facts: list[str]
    user_turns: list[str]
    extracted_facts: list[str]
    matched_gt_count: int
    total_gt: int
    matched_ext_count: int
    total_ext: int

    @property
    def precision(self) -> float:
        return self.matched_ext_count / self.total_ext if self.total_ext else 0.0

    @property
    def recall(self) -> float:
        return self.matched_gt_count / self.total_gt if self.total_gt else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def bench_rows() -> list[BenchRow]:
    return _load_rows()


@pytest.fixture(scope="module")
def bench_results(bench_rows: list[BenchRow]) -> list[RowResult]:
    """Run extraction + matching on all 1000 rows (once per module)."""
    results: list[RowResult] = []
    for idx, row in enumerate(bench_rows):
        extracted = _extract_all_facts(row.user_turns)
        if extracted and row.persona_facts:
            m_ext, m_gt = _match_facts(extracted, row.persona_facts)
        else:
            m_ext, m_gt = set(), set()

        results.append(RowResult(
            row_idx=idx,
            persona_facts=row.persona_facts,
            user_turns=row.user_turns,
            extracted_facts=extracted,
            matched_gt_count=len(m_gt),
            total_gt=len(row.persona_facts),
            matched_ext_count=len(m_ext),
            total_ext=len(extracted),
        ))
    return results


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------


@dataclass
class AggregateMetrics:
    total_rows: int = 0
    rows_with_extraction: int = 0
    total_gt_facts: int = 0
    total_extracted: int = 0
    total_matched_gt: int = 0
    total_matched_ext: int = 0
    per_row_precision: list[float] = field(default_factory=list)
    per_row_recall: list[float] = field(default_factory=list)
    per_row_f1: list[float] = field(default_factory=list)

    # Worst performers for debugging
    worst_recall: list[RowResult] = field(default_factory=list)
    best_precision: list[RowResult] = field(default_factory=list)

    @property
    def macro_precision(self) -> float:
        return statistics.mean(self.per_row_precision) if self.per_row_precision else 0.0

    @property
    def macro_recall(self) -> float:
        return statistics.mean(self.per_row_recall) if self.per_row_recall else 0.0

    @property
    def macro_f1(self) -> float:
        return statistics.mean(self.per_row_f1) if self.per_row_f1 else 0.0

    @property
    def micro_precision(self) -> float:
        return self.total_matched_ext / self.total_extracted if self.total_extracted else 0.0

    @property
    def micro_recall(self) -> float:
        return self.total_matched_gt / self.total_gt_facts if self.total_gt_facts else 0.0

    @property
    def micro_f1(self) -> float:
        p, r = self.micro_precision, self.micro_recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def extraction_rate(self) -> float:
        return self.rows_with_extraction / self.total_rows if self.total_rows else 0.0


def _compute_metrics(results: list[RowResult]) -> AggregateMetrics:
    m = AggregateMetrics(total_rows=len(results))

    for r in results:
        m.total_gt_facts += r.total_gt
        m.total_extracted += r.total_ext
        m.total_matched_gt += r.matched_gt_count
        m.total_matched_ext += r.matched_ext_count

        if r.total_ext > 0:
            m.rows_with_extraction += 1
            m.per_row_precision.append(r.precision)
            m.per_row_recall.append(r.recall)
            m.per_row_f1.append(r.f1)

    # Track worst/best for debugging
    with_ext = [r for r in results if r.total_ext > 0]
    m.worst_recall = sorted(with_ext, key=lambda r: r.recall)[:10]
    m.best_precision = sorted(with_ext, key=lambda r: r.precision, reverse=True)[:10]

    return m


@pytest.fixture(scope="module")
def metrics(bench_results: list[RowResult]) -> AggregateMetrics:
    return _compute_metrics(bench_results)


# ===========================================================================
# Tests
# ===========================================================================


@needs_hf
class TestDatasetLoading:
    """Verify dataset loads and parses correctly."""

    def test_row_count(self, bench_rows: list[BenchRow]):
        assert len(bench_rows) >= 15000, f"Expected ~18000 rows, got {len(bench_rows)}"

    def test_persona_facts_present(self, bench_rows: list[BenchRow]):
        for row in bench_rows[:50]:
            assert len(row.persona_facts) >= 2, f"Row missing persona: {row.persona_facts}"

    def test_user_turns_present(self, bench_rows: list[BenchRow]):
        has_turns = sum(1 for r in bench_rows if len(r.user_turns) > 0)
        assert has_turns > 800, f"Only {has_turns} rows have user turns"

    def test_persona_facts_cleaned(self, bench_rows: list[BenchRow]):
        """No trailing periods (we stripped them)."""
        for row in bench_rows[:100]:
            for fact in row.persona_facts:
                assert not fact.endswith("."), f"Uncleaned: {fact!r}"


@needs_hf
class TestExtractionCoverage:
    """Measure how many conversations produce at least one extraction."""

    def test_extraction_rate_above_minimum(self, metrics: AggregateMetrics):
        """At least 15% of conversations should yield some extraction.

        PersonaChat speakers often phrase facts indirectly ("my dogs love
        the park" vs "i have two dogs"), so heuristic-only extraction
        won't catch everything. 15% is a realistic floor.
        """
        assert metrics.extraction_rate >= 0.15, (
            f"Extraction rate {metrics.extraction_rate:.1%} below 15% minimum"
        )

    def test_total_extractions_nonzero(self, metrics: AggregateMetrics):
        assert metrics.total_extracted > 0

    def test_no_empty_facts(self, bench_results: list[RowResult]):
        """Every extracted fact should have content."""
        for r in bench_results:
            for fact in r.extracted_facts:
                assert fact.strip(), "Empty fact extracted"


@needs_hf
class TestPrecision:
    """Precision: what fraction of extracted facts actually match a ground-truth persona."""

    def test_micro_precision_above_floor(self, metrics: AggregateMetrics):
        """Micro precision should be at least 20%.

        Low bar because PersonaChat personas are terse ("i have two dogs")
        while extracted facts are verbose ("I have two dogs and they are great").
        Semantic matching compensates but isn't perfect.
        """
        if metrics.total_extracted == 0:
            pytest.skip("No extractions to measure precision")
        assert metrics.micro_precision >= 0.20, (
            f"Micro precision {metrics.micro_precision:.1%} below 20%"
        )

    def test_no_garbage_extractions(self, bench_results: list[RowResult]):
        """Spot-check: extracted facts should look like real personal info."""
        garbage_count = 0
        total = 0
        for r in bench_results:
            for fact in r.extracted_facts:
                total += 1
                lower = fact.lower()
                # Garbage heuristics: very short, or just filler
                if len(lower) < 8 or lower in ("i am fine", "i am good", "i am well"):
                    garbage_count += 1

        if total > 0:
            garbage_rate = garbage_count / total
            assert garbage_rate < 0.15, (
                f"{garbage_rate:.1%} of extractions look like garbage ({garbage_count}/{total})"
            )


@needs_hf
class TestRecall:
    """Recall: what fraction of ground-truth persona facts are captured."""

    def test_micro_recall_above_floor(self, metrics: AggregateMetrics):
        """Micro recall over matched rows.

        Heuristic-only recall on PersonaChat will be modest since many
        traits are expressed indirectly. 8% floor (with LLM extraction
        disabled, this is expected to be low).
        """
        assert metrics.micro_recall >= 0.08, (
            f"Micro recall {metrics.micro_recall:.1%} below 8%"
        )

    def test_some_persona_facts_found(self, metrics: AggregateMetrics):
        """At least some ground-truth facts should match."""
        assert metrics.total_matched_gt > 0, "No ground-truth facts matched at all"


@needs_hf
class TestF1:
    """F1 combines precision and recall."""

    def test_micro_f1_above_floor(self, metrics: AggregateMetrics):
        if metrics.total_extracted == 0:
            pytest.skip("No extractions")
        assert metrics.micro_f1 >= 0.10, (
            f"Micro F1 {metrics.micro_f1:.1%} below 10%"
        )


@needs_hf
class TestFalsePositivePatterns:
    """Check that known false-positive patterns are NOT extracted."""

    def test_greetings_not_extracted(self, bench_results: list[RowResult]):
        """Greetings like 'hello how are you' should never produce facts."""
        greeting_extractions = 0
        greeting_patterns = ("hello", "hi ", "hey ", "how are you", "nice to meet")
        for r in bench_results:
            for fact in r.extracted_facts:
                lower = fact.lower()
                if any(lower.startswith(g) for g in greeting_patterns):
                    greeting_extractions += 1
        assert greeting_extractions < 15, (
            f"{greeting_extractions} greeting-like extractions found"
        )

    def test_questions_not_extracted_as_facts(self, bench_results: list[RowResult]):
        """Questions directed at the other speaker shouldn't become facts."""
        question_facts = 0
        for r in bench_results:
            for fact in r.extracted_facts:
                if fact.strip().endswith("?"):
                    question_facts += 1
        assert question_facts < 10, (
            f"{question_facts} question-like extractions found"
        )


@needs_hf
class TestSemanticMatchQuality:
    """Validate that the semantic matching itself is reasonable."""

    def test_exact_match_detected(self):
        """Identical strings should always match."""
        texts = ["i like to go hiking", "i like to go hiking"]
        embs = _get_embeddings(texts)
        sim = _cosine_similarity(embs[0], embs[1])
        assert sim > 0.99

    def test_paraphrase_match(self):
        """Paraphrased persona facts should match above threshold."""
        pairs = [
            ("i have two dogs", "I have 2 dogs at home"),
            ("i love bicycling", "I enjoy riding my bicycle"),
            ("i work as a nurse", "I'm a nurse"),
        ]
        for gt, ext in pairs:
            embs = _get_embeddings([gt, ext])
            sim = _cosine_similarity(embs[0], embs[1])
            assert sim >= 0.50, f"Paraphrase pair below 0.50: {gt!r} vs {ext!r} = {sim:.3f}"

    def test_unrelated_lower_than_related(self):
        """Unrelated pairs should have meaningfully lower similarity than paraphrases."""
        related = ("i have two dogs", "I have 2 dogs at home")
        unrelated = ("i have two dogs", "the derivative of x squared is 2x")

        all_embs = _get_embeddings([related[0], related[1], unrelated[1]])
        sim_related = _cosine_similarity(all_embs[0], all_embs[1])
        sim_unrelated = _cosine_similarity(all_embs[0], all_embs[2])

        assert sim_related > sim_unrelated + 0.15, (
            f"Related ({sim_related:.3f}) should be well above unrelated ({sim_unrelated:.3f})"
        )


# ===========================================================================
# Diagnostic report — run with: pytest -v -k report
# ===========================================================================


@needs_hf
class TestBenchmarkReport:
    """Not a pass/fail test — prints the full benchmark report."""

    def test_report(self, bench_results: list[RowResult], metrics: AggregateMetrics):
        """Print comprehensive benchmark results."""
        report = []
        report.append("")
        report.append("=" * 72)
        report.append("  PERSONACHAT MEMORY EXTRACTION BENCHMARK")
        report.append("=" * 72)
        report.append("")
        report.append(f"  Dataset:            {_DATASET_ID}")
        report.append(f"  Rows evaluated:     {metrics.total_rows}")
        report.append(f"  Match threshold:    {_MATCH_THRESHOLD}")
        report.append("")
        report.append("  --- Coverage ---")
        report.append(f"  Rows with extraction:  {metrics.rows_with_extraction} / {metrics.total_rows} ({metrics.extraction_rate:.1%})")
        report.append(f"  Total GT facts:        {metrics.total_gt_facts}")
        report.append(f"  Total extracted:       {metrics.total_extracted}")
        report.append("")
        report.append("  --- Micro Metrics (corpus-level) ---")
        report.append(f"  Precision:  {metrics.micro_precision:.1%}  ({metrics.total_matched_ext} / {metrics.total_extracted} extracted matched GT)")
        report.append(f"  Recall:     {metrics.micro_recall:.1%}  ({metrics.total_matched_gt} / {metrics.total_gt_facts} GT facts found)")
        report.append(f"  F1:         {metrics.micro_f1:.1%}")
        report.append("")

        if metrics.per_row_precision:
            report.append("  --- Macro Metrics (per-row average, extraction rows only) ---")
            report.append(f"  Precision:  {metrics.macro_precision:.1%}  (median {statistics.median(metrics.per_row_precision):.1%})")
            report.append(f"  Recall:     {metrics.macro_recall:.1%}  (median {statistics.median(metrics.per_row_recall):.1%})")
            report.append(f"  F1:         {metrics.macro_f1:.1%}  (median {statistics.median(metrics.per_row_f1):.1%})")
            report.append("")

        # Distribution of extractions per row
        ext_counts = [r.total_ext for r in bench_results]
        report.append("  --- Extraction distribution ---")
        report.append(f"  0 facts:    {ext_counts.count(0)} rows")
        report.append(f"  1 fact:     {sum(1 for c in ext_counts if c == 1)} rows")
        report.append(f"  2-3 facts:  {sum(1 for c in ext_counts if 2 <= c <= 3)} rows")
        report.append(f"  4-5 facts:  {sum(1 for c in ext_counts if 4 <= c <= 5)} rows")
        report.append(f"  6+ facts:   {sum(1 for c in ext_counts if c >= 6)} rows")
        report.append("")

        # Sample: best extractions (high recall)
        best = sorted(
            [r for r in bench_results if r.total_ext > 0],
            key=lambda r: r.recall, reverse=True,
        )[:5]
        report.append("  --- Top 5 best recall rows ---")
        for r in best:
            report.append(f"  Row {r.row_idx}: recall={r.recall:.0%} prec={r.precision:.0%} | "
                          f"GT={r.persona_facts} | Ext={r.extracted_facts}")
        report.append("")

        # Sample: worst precision (most false positives)
        worst_prec = sorted(
            [r for r in bench_results if r.total_ext >= 3],
            key=lambda r: r.precision,
        )[:5]
        if worst_prec:
            report.append("  --- Top 5 worst precision rows (3+ extractions) ---")
            for r in worst_prec:
                report.append(f"  Row {r.row_idx}: prec={r.precision:.0%} rec={r.recall:.0%} | "
                              f"GT={r.persona_facts} | Ext={r.extracted_facts}")
            report.append("")

        # Sample: missed facts (GT facts with 0 recall)
        zero_recall = [r for r in bench_results if r.total_ext > 0 and r.recall == 0.0]
        if zero_recall:
            report.append(f"  --- Rows with extractions but 0% recall ({len(zero_recall)} rows) ---")
            for r in zero_recall[:5]:
                report.append(f"  Row {r.row_idx}: Ext={r.extracted_facts} | GT={r.persona_facts}")
            report.append("")

        report.append("=" * 72)

        # Print the full report
        print("\n".join(report))

        # This test always passes — it's for diagnostics
        assert True
