"""RAG Pipeline v2 benchmark harness — measures retrieval accuracy.

Ingests the benchmark corpus into a temp SQLite database with real embeddings,
runs all queries through the scoring pipeline, and reports precision/recall/noise
metrics per category.

Usage:
    python -m pytest tests/test_rag_pipeline_bench.py -v -s
    python -m pytest tests/test_rag_pipeline_bench.py::test_rag_baseline -v -s
    python -m pytest tests/test_rag_pipeline_bench.py::test_rag_offline -v -s
    python -m pytest tests/test_rag_pipeline_bench.py::test_rag_compare -v -s
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import aiosqlite
import pytest

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_HERE = Path(__file__).resolve().parent
_CORPUS_DIR = _HERE / "rag_bench" / "corpus"
_CORPUS_V2_DIR = _HERE / "rag_bench" / "corpus_v2"
_QUERIES_PATH = _HERE / "rag_bench" / "queries.json"
_QUERIES_V2_PATH = _HERE / "rag_bench" / "queries_v2.json"
_RESULTS_DIR = _HERE / "rag_bench" / ".results"


# ---------------------------------------------------------------------------
# Settings patching
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _patch_settings(**overrides):
    """Temporarily override augmentum.config.settings attributes."""
    from augmentum.config import settings

    originals = {}
    for key, value in overrides.items():
        originals[key] = getattr(settings, key)
        setattr(settings, key, value)
    try:
        yield settings
    finally:
        for key, value in originals.items():
            setattr(settings, key, value)


# ---------------------------------------------------------------------------
# Minimal backend wrapper
# ---------------------------------------------------------------------------

class _BenchBackend:
    """Minimal backend wrapper so DocumentStore can use our temp DB."""

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn
        self.vec_enabled = False

    @property
    def conn(self) -> aiosqlite.Connection:  # noqa: D102
        return self._conn


# ---------------------------------------------------------------------------
# Lightweight LLM backend for query analyzer (OpenAI-compatible)
# ---------------------------------------------------------------------------

class _LMStudioBackend:
    """Minimal OpenAI-compatible backend for the query analyzer.

    Calls LM Studio's /v1/chat/completions endpoint directly.
    Only implements the .chat() interface that QueryAnalyzer needs.
    """

    def __init__(self, base_url: str = "http://localhost:1234", model: str = "") -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model

    async def chat(self, messages, system_prompt="", temperature=0.0, max_tokens=150, **kw):
        import httpx

        api_messages = []
        if system_prompt:
            api_messages.append({"role": "system", "content": system_prompt})
        for m in messages:
            api_messages.append({"role": m.role, "content": m.content})

        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{self._base_url}/v1/chat/completions",
                json={
                    "model": self._model,
                    "messages": api_messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        content = data["choices"][0]["message"]["content"]

        # Return a simple object with .content attribute
        class _Resp:
            pass
        r = _Resp()
        r.content = content
        return r


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL DEFAULT 'default',
    filename TEXT NOT NULL,
    mime_type TEXT NOT NULL,
    file_size INTEGER,
    chunk_count INTEGER,
    scope TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS document_chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    page_num INTEGER,
    char_offset INTEGER DEFAULT 0,
    token_count INTEGER DEFAULT 0,
    embedding BLOB,
    parent_id TEXT REFERENCES document_chunks(id),
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_chunks_document
    ON document_chunks(document_id, chunk_index);

CREATE VIRTUAL TABLE IF NOT EXISTS document_chunks_fts USING fts5(
    content,
    content=document_chunks,
    content_rowid=rowid
);

CREATE TRIGGER IF NOT EXISTS doc_chunks_fts_insert
    AFTER INSERT ON document_chunks BEGIN
    INSERT INTO document_chunks_fts(rowid, content)
        VALUES (new.rowid, new.content);
END;

CREATE TRIGGER IF NOT EXISTS doc_chunks_fts_delete
    AFTER DELETE ON document_chunks BEGIN
    INSERT INTO document_chunks_fts(document_chunks_fts, rowid, content)
        VALUES('delete', old.rowid, old.content);
END;

CREATE TRIGGER IF NOT EXISTS doc_chunks_fts_update
    AFTER UPDATE OF content ON document_chunks BEGIN
    INSERT INTO document_chunks_fts(document_chunks_fts, rowid, content)
        VALUES('delete', old.rowid, old.content);
    INSERT INTO document_chunks_fts(rowid, content)
        VALUES (new.rowid, new.content);
END;
"""

_VEC_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS doc_chunks_vec USING vec0(
    chunk_id TEXT PRIMARY KEY,
    embedding float[768]
);
"""

# ---------------------------------------------------------------------------
# Keyword matching
# ---------------------------------------------------------------------------

def _keyword_in_text(keyword: str, text: str) -> bool:
    """Case-insensitive word-boundary match.

    Multi-word phrases matched as exact sequences.
    """
    pattern = r"\b" + re.escape(keyword) + r"\b"
    return bool(re.search(pattern, text, re.IGNORECASE))


# ---------------------------------------------------------------------------
# Metric data classes
# ---------------------------------------------------------------------------

@dataclass
class QueryResult:
    query_id: str
    category: str
    difficulty: str
    chunks_injected: int
    keywords_found: list[str]
    keywords_missed: list[str]
    noise_detected: bool
    precision: float
    recall: float
    strategy_correct: bool | None
    context_tokens: int


@dataclass
class BenchReport:
    mode: str
    timestamp: str
    total_queries: int
    results: list[QueryResult]
    # Overall
    precision_at_k: float
    recall_at_k: float
    noise_rate: float
    strategy_accuracy: float  # -1 if N/A
    # Per category
    category_metrics: dict  # category -> {precision, recall, noise_rate, count}
    # Diagnostics
    avg_chunks_injected: float
    avg_context_tokens: float


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------

class RAGBenchHarness:
    """Self-contained RAG benchmark that runs without a live server."""

    def __init__(self, corpus_dir: Path | None = None, queries_path: Path | None = None) -> None:
        self._corpus_dir = corpus_dir or _CORPUS_DIR
        self._queries_path = queries_path or _QUERIES_PATH
        self._db_path: str | None = None
        self._conn: aiosqlite.Connection | None = None
        self._backend: _BenchBackend | None = None
        self._store = None  # DocumentStore
        self._queries: list[dict] = []

    # -- Setup / teardown ---------------------------------------------------

    async def setup(self) -> None:
        """Create temp DB, load schema, ingest corpus, load queries."""
        # Use a temp file (not :memory:) so aiosqlite handles it cleanly
        fd, self._db_path = tempfile.mkstemp(suffix=".db", prefix="rag_bench_")
        os.close(fd)

        self._conn = await aiosqlite.connect(self._db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.execute("PRAGMA journal_mode=WAL")
        await self._conn.execute("PRAGMA foreign_keys=ON")
        await self._conn.executescript(_SCHEMA_SQL)

        self._backend = _BenchBackend(self._conn)

        # Try to load sqlite-vec extension and create vec0 table
        try:
            import sqlite_vec
            await self._conn.enable_load_extension(True)
            await self._conn.load_extension(sqlite_vec.loadable_path())
            await self._conn.enable_load_extension(False)
            await self._conn.executescript(_VEC_SQL)
            self._backend.vec_enabled = True
            print("  sqlite-vec loaded — hybrid search enabled")
        except Exception:
            self._backend.vec_enabled = False
            print("  sqlite-vec unavailable — FTS-only mode")

        from augmentum.documents.store import DocumentStore

        self._store = DocumentStore(self._backend)

        # Ingest corpus
        await self._ingest_corpus()

        # Load queries
        with open(self._queries_path, encoding="utf-8") as f:
            self._queries = json.load(f)

    async def teardown(self) -> None:
        """Close DB and remove temp file."""
        if self._conn:
            await self._conn.close()
        if self._db_path and os.path.exists(self._db_path):
            os.unlink(self._db_path)

    async def _ingest_corpus(self) -> None:
        """Ingest all corpus files into the store."""
        mime_map = {
            ".md": "text/markdown",
            ".csv": "text/csv",
            ".txt": "text/plain",
        }
        files = sorted(self._corpus_dir.iterdir())
        print(f"\nIngesting {len(files)} corpus files...")
        t0 = time.perf_counter()

        for fpath in files:
            if fpath.name.startswith("."):
                continue
            mime = mime_map.get(fpath.suffix, "text/plain")
            data = fpath.read_bytes()

            # Disable contextual retrieval (no LLM needed) and reranker
            with _patch_settings(
                document_rag_contextual_retrieval=False,
                reranker_enabled=False,
            ):
                result = await self._store.ingest(
                    data=data,
                    filename=fpath.name,
                    mime_type=mime,
                    user_id="bench",
                )
            print(f"  {fpath.name}: {result['chunk_count']} chunks")

        elapsed = time.perf_counter() - t0
        print(f"Ingestion complete in {elapsed:.1f}s")

    # -- Query execution ----------------------------------------------------

    async def run_all(
        self,
        mode: str = "offline",
        llm_backend=None,
        use_query_expansion: bool = False,
        use_span_filter: bool = False,
        use_density_scoring: bool = False,
        use_topic_coverage: bool = False,
        use_reranker: bool = False,
    ) -> BenchReport:
        """Run all non-holdout queries and compute metrics.

        mode="offline": mechanical pipeline only, no query analyzer
        mode="full": includes query analyzer (requires llm_backend)
        mode="baseline": same as offline, labeled as baseline

        use_query_expansion: add embedding-based synonym expansion to FTS
        use_span_filter: filter chunk sentences by query relevance
        """
        from augmentum.config import settings
        from augmentum.documents.dedup import deduplicate
        from augmentum.documents.scoring import (
            apply_budget,
            cliff_detect,
            determine_sufficiency,
            score_gate,
        )

        use_analyzer = mode == "full" and llm_backend is not None
        analyzer = None
        if use_analyzer:
            from augmentum.documents.query_analyzer import QueryAnalyzer
            analyzer = QueryAnalyzer(backend=llm_backend)

        # Build vocab index for query expansion if requested
        if use_query_expansion:
            from augmentum.documents.query_expansion import build_vocab_index
            all_chunks = []
            rows = await self._conn.execute(
                "SELECT content FROM document_chunks WHERE chunk_index >= 0"
            )
            for row in await rows.fetchall():
                all_chunks.append(row["content"])
            n_terms = build_vocab_index(all_chunks)
            print(f"  Query expansion: {n_terms} vocab terms indexed")

        # Pre-compute answer density per chunk
        density_map: dict[str, float] = {}
        if use_density_scoring:
            from augmentum.documents.answer_density import compute_density

            rows = await self._conn.execute(
                "SELECT id, content FROM document_chunks WHERE chunk_index >= 0"
            )
            for row in await rows.fetchall():
                density_map[row["id"]] = compute_density(row["content"])
            avg_density = sum(density_map.values()) / len(density_map) if density_map else 0
            print(f"  Density scoring: {len(density_map)} chunks scored (avg={avg_density:.3f})")

        # Build topic coverage map
        if use_topic_coverage:
            from augmentum.documents.topic_coverage import build_topic_map

            docs_for_topics = []
            rows = await self._conn.execute("SELECT id, filename FROM documents")
            for row in await rows.fetchall():
                # Get full content for topic extraction
                chunks_rows = await self._conn.execute(
                    "SELECT content FROM document_chunks WHERE document_id = ? AND chunk_index >= 0 ORDER BY chunk_index",
                    (row["id"],),
                )
                all_content = "\n".join(r["content"] for r in await chunks_rows.fetchall())
                docs_for_topics.append({
                    "id": row["id"],
                    "filename": row["filename"],
                    "content": all_content,
                })
            n_mapped = build_topic_map(docs_for_topics)
            print(f"  Topic coverage: {n_mapped} documents mapped")

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        results: list[QueryResult] = []
        active_queries = [q for q in self._queries if not q.get("holdout", False)]

        print(f"\nRunning {len(active_queries)} queries (mode={mode})...")
        t0 = time.perf_counter()

        for q in active_queries:
            qid = q["id"]
            category = q["category"]
            difficulty = q.get("difficulty", "medium")
            expected_kw = q.get("expected_chunks_contain", [])
            unexpected_kw = q.get("expected_chunks_not_contain", [])
            should_retrieve = q.get("should_retrieve", True)

            # Topic coverage check — soft signal, not hard gate
            topic_coverage_score = 1.0  # default: fully covered
            if use_topic_coverage:
                from augmentum.documents.topic_coverage import check_topic_coverage

                coverage = check_topic_coverage(q["query"])
                topic_coverage_score = coverage["best_match_score"]

            # Query analyzer (full mode only)
            actual_strategy = None
            search_queries = [q["query"]]
            skip_search = False

            if use_analyzer and analyzer:
                analysis = await analyzer.analyze(
                    q["query"],
                    doc_names=[f.name for f in self._corpus_dir.iterdir() if f.is_file()],
                )
                actual_strategy = analysis.strategy
                if analysis.strategy == "skip":
                    skip_search = True
                elif analysis.queries:
                    search_queries = analysis.queries

            # Query expansion: add synonym terms to search queries
            if use_query_expansion and not skip_search:
                from augmentum.documents.query_expansion import expand_query_terms
                from augmentum.documents.store import _STOP_WORDS
                import re as _re

                expanded_queries = []
                for sq in search_queries:
                    words = _re.findall(r'\w+', sq.lower())
                    content_words = [w for w in words if w not in _STOP_WORDS and len(w) > 2]
                    if content_words:
                        expansions = expand_query_terms(content_words, top_k=3, min_similarity=0.5)
                        if expansions:
                            expanded_queries.append(sq + " " + " ".join(expansions))
                        else:
                            expanded_queries.append(sq)
                    else:
                        expanded_queries.append(sq)
                search_queries = expanded_queries

            # Search (reranker optional)
            raw_results: list[dict] = []
            if not skip_search:
                with _patch_settings(reranker_enabled=use_reranker):
                    if len(search_queries) == 1:
                        raw_results = await self._store.search(
                            search_queries[0], user_id="bench",
                            limit=settings.document_rag_recall_limit,
                        )
                    else:
                        # Multi-query: concurrent search + max-score merge
                        import asyncio as _aio
                        tasks = [
                            self._store.search(sq, user_id="bench",
                                               limit=settings.document_rag_recall_limit)
                            for sq in search_queries
                        ]
                        all_result_lists = await _aio.gather(*tasks)
                        merged_map: dict[str, dict] = {}
                        for sq_results in all_result_lists:
                            for r in sq_results:
                                cid = r["chunk_id"]
                                if cid not in merged_map or r.get("score", 0) > merged_map[cid].get("score", 0):
                                    merged_map[cid] = r
                        raw_results = sorted(
                            merged_map.values(),
                            key=lambda r: r.get("score", 0),
                            reverse=True,
                        )

            dual_source = getattr(self._store, "_last_search_dual_source", True)

            # Topic coverage soft signal: dampen scores when topic poorly covered
            if use_topic_coverage and raw_results and topic_coverage_score < 0.3:
                # Low topic coverage → multiply scores down so score gate filters more
                dampen = 0.3 + topic_coverage_score  # 0.3 to 0.6 range
                for r in raw_results:
                    r["score"] = r.get("score", 0) * dampen

            # Density boost: re-score results by information density
            if use_density_scoring and raw_results:
                from augmentum.documents.answer_density import boost_by_density

                for r in raw_results:
                    chunk_id = r.get("chunk_id", "")
                    density = density_map.get(chunk_id, 0.0)
                    if density > 0:
                        r["score"] = boost_by_density(r["score"], density, weight=0.5)
                # Re-sort after boosting
                raw_results.sort(key=lambda r: r.get("score", 0), reverse=True)

            # Scoring pipeline
            scored = score_gate(raw_results, reranker_enabled=use_reranker, dual_source=dual_source, query=q["query"])
            clipped = cliff_detect(
                scored,
                cliff_ratio=settings.document_rag_cliff_ratio,
                max_results=settings.document_rag_recall_limit,
            )
            deduped = deduplicate(clipped)

            # Span filtering: keep only query-relevant sentences within chunks
            if use_span_filter and deduped:
                from augmentum.documents.span_filter import filter_chunk_spans
                deduped = filter_chunk_spans(deduped, q["query"])

            budgeted = apply_budget(deduped, max_tokens=1500)
            sufficiency = determine_sufficiency(budgeted)

            # Combine chunk content for keyword matching
            combined = " ".join(
                sc.chunk.get("content", "") for sc in budgeted
            )
            context_tokens = len(combined) // 4  # approximate

            # Keyword evaluation
            kw_found = [kw for kw in expected_kw if _keyword_in_text(kw, combined)]
            kw_missed = [kw for kw in expected_kw if not _keyword_in_text(kw, combined)]

            # Noise: unexpected keywords present in retrieved chunks
            noise_kw = [kw for kw in unexpected_kw if _keyword_in_text(kw, combined)]

            # For skip/negative queries: any retrieval is noise
            if not should_retrieve and category == "skip":
                # Skip queries: noise if sufficiency != "none"
                noise_detected = sufficiency != "none"
            elif category == "negative":
                # Negative queries: noise if unexpected keywords found
                noise_detected = bool(noise_kw)
            else:
                noise_detected = bool(noise_kw)

            # Precision: fraction of injected chunks that contain any expected keyword
            chunks_with_hit = 0
            total_chunks = len(budgeted)
            for sc in budgeted:
                chunk_text = sc.chunk.get("content", "")
                if any(_keyword_in_text(kw, chunk_text) for kw in expected_kw):
                    chunks_with_hit += 1

            precision = chunks_with_hit / total_chunks if total_chunks > 0 else (
                1.0 if not should_retrieve else 0.0
            )

            # Recall: fraction of expected keywords found
            recall = (
                len(kw_found) / len(expected_kw) if expected_kw
                else (1.0 if not should_retrieve else 0.0)
            )

            results.append(QueryResult(
                query_id=qid,
                category=category,
                difficulty=difficulty,
                chunks_injected=total_chunks,
                keywords_found=kw_found,
                keywords_missed=kw_missed,
                noise_detected=noise_detected,
                precision=precision,
                recall=recall,
                strategy_correct=(
                    actual_strategy == q.get("expected_strategy")
                    if actual_strategy is not None else None
                ),
                context_tokens=context_tokens,
            ))

        elapsed = time.perf_counter() - t0
        print(f"Queries complete in {elapsed:.1f}s")

        # Aggregate metrics
        n = len(results)
        precision_at_k = sum(r.precision for r in results) / n if n else 0.0
        recall_at_k = sum(r.recall for r in results) / n if n else 0.0
        noise_rate = sum(1 for r in results if r.noise_detected) / n if n else 0.0

        # Strategy accuracy (only if tested)
        strat_tested = [r for r in results if r.strategy_correct is not None]
        strategy_accuracy = (
            sum(1 for r in strat_tested if r.strategy_correct) / len(strat_tested)
            if strat_tested else -1.0
        )

        # Per-category breakdown
        categories: dict[str, list[QueryResult]] = {}
        for r in results:
            categories.setdefault(r.category, []).append(r)

        category_metrics = {}
        for cat, cat_results in sorted(categories.items()):
            cn = len(cat_results)
            category_metrics[cat] = {
                "precision": sum(r.precision for r in cat_results) / cn,
                "recall": sum(r.recall for r in cat_results) / cn,
                "noise_rate": sum(1 for r in cat_results if r.noise_detected) / cn,
                "count": cn,
            }

        avg_chunks = sum(r.chunks_injected for r in results) / n if n else 0.0
        avg_tokens = sum(r.context_tokens for r in results) / n if n else 0.0

        return BenchReport(
            mode=mode,
            timestamp=timestamp,
            total_queries=n,
            results=results,
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            noise_rate=noise_rate,
            strategy_accuracy=strategy_accuracy,
            category_metrics=category_metrics,
            avg_chunks_injected=avg_chunks,
            avg_context_tokens=avg_tokens,
        )

    # -- Persistence --------------------------------------------------------

    def save_results(self, report: BenchReport) -> Path:
        """Save report JSON to .results directory."""
        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        fname = f"{report.mode}_{report.timestamp}.json"
        path = _RESULTS_DIR / fname
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2)
        print(f"\nResults saved to {path}")
        return path

    def load_latest_results(self, mode: str) -> BenchReport | None:
        """Load the most recent results file for a given mode."""
        if not _RESULTS_DIR.exists():
            return None
        files = sorted(
            _RESULTS_DIR.glob(f"{mode}_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not files:
            return None
        with open(files[0], encoding="utf-8") as f:
            data = json.load(f)
        # Reconstruct dataclasses
        results = [QueryResult(**r) for r in data.pop("results")]
        return BenchReport(results=results, **data)

    # -- Reporting ----------------------------------------------------------

    def print_report(self, report: BenchReport) -> None:
        """Print human-readable benchmark report."""
        print(f"\n{'=' * 70}")
        print(f"RAG BENCHMARK REPORT — mode={report.mode}")
        print(f"{'=' * 70}")
        print(f"Queries: {report.total_queries}")
        print(f"Precision@K:  {report.precision_at_k:.1%}")
        print(f"Recall@K:     {report.recall_at_k:.1%}")
        print(f"Noise Rate:   {report.noise_rate:.1%}")
        if report.strategy_accuracy >= 0:
            print(f"Strategy Acc: {report.strategy_accuracy:.1%}")
        print(f"Avg Chunks:   {report.avg_chunks_injected:.1f}")
        print(f"Avg Tokens:   {report.avg_context_tokens:.0f}")

        print(f"\n{'-' * 70}")
        print(f"{'Category':<18} {'Count':>5} {'Prec':>7} {'Recall':>7} {'Noise':>7}")
        print(f"{'-' * 70}")
        for cat, m in sorted(report.category_metrics.items()):
            print(
                f"{cat:<18} {m['count']:>5} "
                f"{m['precision']:>6.1%} {m['recall']:>6.1%} {m['noise_rate']:>6.1%}"
            )

        # Per-query details for failures
        failures = [r for r in report.results if r.keywords_missed or r.noise_detected]
        if failures:
            print(f"\n{'-' * 70}")
            print("ISSUES:")
            for r in failures:
                issues = []
                if r.keywords_missed:
                    issues.append(f"missed=[{', '.join(r.keywords_missed)}]")
                if r.noise_detected:
                    issues.append("NOISE")
                print(f"  {r.query_id} ({r.category}/{r.difficulty}): {', '.join(issues)}")

        print(f"{'=' * 70}\n")

    def print_comparison(self, baseline: BenchReport, current: BenchReport) -> None:
        """Print side-by-side comparison of two reports."""
        def _delta(a: float, b: float) -> str:
            d = b - a
            sign = "+" if d >= 0 else ""
            return f"{sign}{d:.1%}"

        print(f"\n{'=' * 70}")
        print(f"COMPARISON: {baseline.mode} vs {current.mode}")
        print(f"{'=' * 70}")
        print(f"{'Metric':<18} {'Baseline':>10} {'Current':>10} {'Delta':>10}")
        print(f"{'-' * 70}")
        print(
            f"{'Precision@K':<18} {baseline.precision_at_k:>9.1%} "
            f"{current.precision_at_k:>9.1%} {_delta(baseline.precision_at_k, current.precision_at_k):>10}"
        )
        print(
            f"{'Recall@K':<18} {baseline.recall_at_k:>9.1%} "
            f"{current.recall_at_k:>9.1%} {_delta(baseline.recall_at_k, current.recall_at_k):>10}"
        )
        print(
            f"{'Noise Rate':<18} {baseline.noise_rate:>9.1%} "
            f"{current.noise_rate:>9.1%} {_delta(baseline.noise_rate, current.noise_rate):>10}"
        )
        print(
            f"{'Avg Chunks':<18} {baseline.avg_chunks_injected:>9.1f} "
            f"{current.avg_chunks_injected:>9.1f} "
            f"{current.avg_chunks_injected - baseline.avg_chunks_injected:>+9.1f}"
        )

        # Per-category comparison
        all_cats = sorted(
            set(list(baseline.category_metrics) + list(current.category_metrics))
        )
        if all_cats:
            print(f"\n{'-' * 70}")
            print(f"{'Category':<18} {'B-Prec':>7} {'C-Prec':>7} {'B-Rec':>7} {'C-Rec':>7}")
            print(f"{'-' * 70}")
            for cat in all_cats:
                bm = baseline.category_metrics.get(cat, {})
                cm = current.category_metrics.get(cat, {})
                bp = bm.get("precision", 0)
                cp = cm.get("precision", 0)
                br = bm.get("recall", 0)
                cr = cm.get("recall", 0)
                print(f"{cat:<18} {bp:>6.1%} {cp:>6.1%} {br:>6.1%} {cr:>6.1%}")

        print(f"{'=' * 70}\n")


# ---------------------------------------------------------------------------
# Test functions
# ---------------------------------------------------------------------------

@pytest.fixture
async def harness():
    """Create and tear down the benchmark harness (v1 corpus)."""
    h = RAGBenchHarness()
    await h.setup()
    yield h
    await h.teardown()


@pytest.fixture
async def harness_v2():
    """Create and tear down the benchmark harness (v2 corpus)."""
    h = RAGBenchHarness(corpus_dir=_CORPUS_V2_DIR, queries_path=_QUERIES_V2_PATH)
    await h.setup()
    yield h
    await h.teardown()


@pytest.mark.asyncio
async def test_rag_baseline(harness: RAGBenchHarness):
    """Capture baseline -- run BEFORE implementation changes."""
    report = await harness.run_all(mode="baseline")
    harness.save_results(report)
    harness.print_report(report)


@pytest.mark.asyncio
async def test_rag_offline(harness: RAGBenchHarness):
    """Mechanical improvements only, no LLM query analyzer."""
    report = await harness.run_all(mode="offline")
    harness.save_results(report)
    harness.print_report(report)
    # CI gate thresholds — conservative floors to catch regressions.
    # Baseline FTS-only: ~42% precision, ~17% noise.  Tighten after improvements.
    assert report.precision_at_k >= 0.25, f"Precision {report.precision_at_k:.1%} below 25%"
    assert report.noise_rate <= 0.35, f"Noise rate {report.noise_rate:.1%} above 35%"


@pytest.mark.asyncio
async def test_rag_query_expansion(harness: RAGBenchHarness):
    """Test embedding-based query expansion (zero-LLM synonym discovery)."""
    report = await harness.run_all(mode="offline", use_query_expansion=True)
    harness.save_results(report)
    harness.print_report(report)

    offline = harness.load_latest_results("offline")
    if offline:
        print("\n--- Query Expansion vs Offline comparison ---")
        harness.print_comparison(offline, report)


@pytest.mark.asyncio
async def test_rag_span_filter(harness: RAGBenchHarness):
    """Test FILCO-style sentence-level span filtering."""
    report = await harness.run_all(mode="offline", use_span_filter=True)
    harness.save_results(report)
    harness.print_report(report)

    offline = harness.load_latest_results("offline")
    if offline:
        print("\n--- Span Filter vs Offline comparison ---")
        harness.print_comparison(offline, report)


@pytest.mark.asyncio
async def test_rag_combined(harness: RAGBenchHarness):
    """Test both query expansion + span filtering together."""
    report = await harness.run_all(
        mode="offline", use_query_expansion=True, use_span_filter=True,
    )
    harness.save_results(report)
    harness.print_report(report)

    offline = harness.load_latest_results("offline")
    if offline:
        print("\n--- Combined (expansion + span filter) vs Offline ---")
        harness.print_comparison(offline, report)


@pytest.mark.asyncio
async def test_rag_density(harness: RAGBenchHarness):
    """Test answer density scoring (boost info-dense chunks)."""
    report = await harness.run_all(mode="offline", use_density_scoring=True)
    harness.save_results(report)
    harness.print_report(report)

    offline = harness.load_latest_results("offline")
    if offline:
        print("\n--- Density Scoring vs Offline ---")
        harness.print_comparison(offline, report)


@pytest.mark.asyncio
async def test_rag_topic_coverage(harness: RAGBenchHarness):
    """Test topic coverage mapping (active negative detection)."""
    report = await harness.run_all(mode="offline", use_topic_coverage=True)
    harness.save_results(report)
    harness.print_report(report)

    offline = harness.load_latest_results("offline")
    if offline:
        print("\n--- Topic Coverage vs Offline ---")
        harness.print_comparison(offline, report)


@pytest.mark.asyncio
async def test_rag_novel_combined(harness: RAGBenchHarness):
    """Test density + topic coverage together."""
    report = await harness.run_all(
        mode="offline", use_density_scoring=True, use_topic_coverage=True,
    )
    harness.save_results(report)
    harness.print_report(report)

    offline = harness.load_latest_results("offline")
    if offline:
        print("\n--- Novel Combined (density + topic) vs Offline ---")
        harness.print_comparison(offline, report)


@pytest.mark.asyncio
async def test_rag_full(harness: RAGBenchHarness):
    """Full pipeline with query analyzer via LM Studio."""
    import httpx

    # Check LM Studio is running
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get("http://localhost:1234/v1/models")
            models = resp.json().get("data", [])
            if not models:
                pytest.skip("LM Studio has no models loaded")
            model_id = models[0]["id"]
            print(f"\nUsing LM Studio model: {model_id}")
    except Exception:
        pytest.skip("LM Studio not available on localhost:1234")

    # Prefer a small/fast model for query classification
    _PREFERRED = ["gemma-3-4b-it", "qwen3.5-4b-claude-4.6-opus-reasoning-distilled", "nvidia/nemotron-3-nano-4b"]
    chosen = model_id
    for pref in _PREFERRED:
        if any(m["id"] == pref for m in models):
            chosen = pref
            break
    print(f"Using model: {chosen}")

    backend = _LMStudioBackend(base_url="http://localhost:1234", model=chosen)

    # Increase timeout for benchmark (local models are slower than production API)
    with _patch_settings(document_rag_query_analysis_timeout=10.0):
        report = await harness.run_all(mode="full", llm_backend=backend)
    harness.save_results(report)
    harness.print_report(report)

    # Compare with offline results
    offline = harness.load_latest_results("offline")
    if offline:
        print("\n--- Full vs Offline comparison ---")
        harness.print_comparison(offline, report)


@pytest.mark.asyncio
async def test_rag_compare(harness: RAGBenchHarness):
    """Load saved baseline, run current, print comparison."""
    baseline = harness.load_latest_results("baseline")
    if not baseline:
        pytest.skip("No baseline results -- run test_rag_baseline first")
    current = await harness.run_all(mode="offline")
    harness.print_comparison(baseline, current)


# ---------------------------------------------------------------------------
# V2 corpus tests — fresh domains to validate against overfitting
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_v2_offline(harness_v2: RAGBenchHarness):
    """V2 corpus: mechanical pipeline only."""
    report = await harness_v2.run_all(mode="offline")
    harness_v2.save_results(report)
    harness_v2.print_report(report)


@pytest.mark.asyncio
async def test_v2_topic_coverage(harness_v2: RAGBenchHarness):
    """V2 corpus: soft topic coverage."""
    report = await harness_v2.run_all(mode="offline", use_topic_coverage=True)
    harness_v2.save_results(report)
    harness_v2.print_report(report)


@pytest.mark.asyncio
async def test_v2_query_expansion(harness_v2: RAGBenchHarness):
    """V2 corpus: embedding-based query expansion."""
    report = await harness_v2.run_all(mode="offline", use_query_expansion=True)
    harness_v2.save_results(report)
    harness_v2.print_report(report)


@pytest.mark.asyncio
async def test_v2_all_novel(harness_v2: RAGBenchHarness):
    """V2 corpus: topic coverage + query expansion combined."""
    report = await harness_v2.run_all(
        mode="offline", use_topic_coverage=True, use_query_expansion=True,
    )
    harness_v2.save_results(report)
    harness_v2.print_report(report)


# ---------------------------------------------------------------------------
# Reranker tests — cross-encoder precision boost
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rag_reranker(harness: RAGBenchHarness):
    """V1 corpus: reranker enabled (cross-encoder after RRF)."""
    report = await harness.run_all(mode="offline", use_reranker=True)
    harness.save_results(report)
    harness.print_report(report)


@pytest.mark.asyncio
async def test_v2_reranker(harness_v2: RAGBenchHarness):
    """V2 corpus: reranker enabled."""
    report = await harness_v2.run_all(mode="offline", use_reranker=True)
    harness_v2.save_results(report)
    harness_v2.print_report(report)


@pytest.mark.asyncio
async def test_rag_reranker_topic(harness: RAGBenchHarness):
    """V1 corpus: reranker + topic coverage (best production config)."""
    report = await harness.run_all(
        mode="offline", use_reranker=True, use_topic_coverage=True,
    )
    harness.save_results(report)
    harness.print_report(report)


@pytest.mark.asyncio
async def test_v2_reranker_topic(harness_v2: RAGBenchHarness):
    """V2 corpus: reranker + topic coverage (best production config)."""
    report = await harness_v2.run_all(
        mode="offline", use_reranker=True, use_topic_coverage=True,
    )
    harness_v2.save_results(report)
    harness_v2.print_report(report)
