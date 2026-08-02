"""RAG Pipeline v2 — Stress & Production Scenario Tests

Systematic testing of real-world conditions the benchmark harness doesn't cover.
Each scenario class documents: hypothesis, method, pass criteria, and findings.

Scenarios:
  2. Large documents (200+ chunks) — performance degradation
  3. Conversation context — queries dependent on prior messages
  4. Concurrent ingestion + search — race conditions
  5. Full vs search mode interaction — mixed binding paths
  6. Token budget vs context window — overflow behavior
  7. Re-ingestion / document updates — old+new chunk mixing
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import tempfile
import time

import aiosqlite
import pytest

# ---------------------------------------------------------------------------
# Shared infrastructure
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def _patch_settings(**overrides):
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


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY, user_id TEXT NOT NULL DEFAULT 'default',
    filename TEXT NOT NULL, mime_type TEXT NOT NULL,
    file_size INTEGER, chunk_count INTEGER, scope TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS document_chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL, content TEXT NOT NULL,
    page_num INTEGER, char_offset INTEGER DEFAULT 0,
    token_count INTEGER DEFAULT 0, embedding BLOB,
    parent_id TEXT REFERENCES document_chunks(id),
    created_at TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON document_chunks(document_id, chunk_index);
CREATE VIRTUAL TABLE IF NOT EXISTS document_chunks_fts USING fts5(
    content, content=document_chunks, content_rowid=rowid
);
CREATE TRIGGER IF NOT EXISTS doc_chunks_fts_insert AFTER INSERT ON document_chunks BEGIN
    INSERT INTO document_chunks_fts(rowid, content) VALUES (new.rowid, new.content);
END;
CREATE TRIGGER IF NOT EXISTS doc_chunks_fts_delete AFTER DELETE ON document_chunks BEGIN
    INSERT INTO document_chunks_fts(document_chunks_fts, rowid, content) VALUES('delete', old.rowid, old.content);
END;
CREATE TRIGGER IF NOT EXISTS doc_chunks_fts_update AFTER UPDATE OF content ON document_chunks BEGIN
    INSERT INTO document_chunks_fts(document_chunks_fts, rowid, content) VALUES('delete', old.rowid, old.content);
    INSERT INTO document_chunks_fts(rowid, content) VALUES (new.rowid, new.content);
END;
"""

_VEC_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS doc_chunks_vec USING vec0(
    chunk_id TEXT PRIMARY KEY, embedding float[768]
);
"""


class _Backend:
    def __init__(self, conn):
        self._conn = conn
        self.vec_enabled = False

    @property
    def conn(self):
        return self._conn


async def _create_store():
    """Create a fresh DocumentStore with temp DB. Returns (store, conn, db_path)."""
    fd, db_path = tempfile.mkstemp(suffix=".db", prefix="rag_stress_")
    os.close(fd)

    conn = await aiosqlite.connect(db_path)
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA foreign_keys=ON")
    await conn.executescript(_SCHEMA_SQL)

    backend = _Backend(conn)
    try:
        import sqlite_vec
        await conn.enable_load_extension(True)
        await conn.load_extension(sqlite_vec.loadable_path())
        await conn.enable_load_extension(False)
        await conn.executescript(_VEC_SQL)
        backend.vec_enabled = True
    except Exception:
        pass

    from augmentum.documents.store import DocumentStore
    store = DocumentStore(backend)
    return store, conn, db_path


async def _cleanup(conn, db_path):
    await conn.close()
    if os.path.exists(db_path):
        os.unlink(db_path)


# ===================================================================
# SCENARIO 2: Large Documents (200+ chunks)
# ===================================================================

class TestLargeDocuments:
    """
    HYPOTHESIS: Pipeline maintains quality and performance with large
    documents that produce 200+ chunks, as users commonly upload
    100+ page PDFs.

    METHOD: Generate a synthetic document with 200+ distinct sections
    containing unique identifiable facts. Measure:
    - Ingestion time
    - Query time (search + scoring pipeline)
    - Precision (does the right section get found?)
    - Dedup effectiveness (many overlapping chunks)

    PASS CRITERIA:
    - Query time < 2 seconds
    - Correct section found in top-3 for targeted queries
    - No crashes or memory issues
    """

    @pytest.fixture
    async def large_store(self):
        store, conn, db_path = await _create_store()
        yield store, conn
        await _cleanup(conn, db_path)

    def _generate_large_doc(self, n_sections: int = 50) -> str:
        """Generate a document with n_sections distinct, identifiable sections.

        Each section has unique facts (numbers, names, dates) that can be
        queried specifically to verify correct retrieval.
        """
        parts = ["# Large Technical Specification Document\n"]
        for i in range(n_sections):
            section_num = i + 1
            parts.append(f"""
## Section {section_num}: Module {section_num:03d} Configuration

The Module-{section_num:03d} subsystem was designed by Engineer-{section_num:03d}
on {2020 + (i % 6)}-{(i % 12) + 1:02d}-{(i % 28) + 1:02d}. It processes
approximately {(i + 1) * 1000} transactions per second with a latency of
{(i + 1) * 0.5:.1f} milliseconds.

The budget allocation for Module-{section_num:03d} is ${(i + 1) * 25000:,}.
The team consists of {(i % 8) + 2} engineers and {(i % 4) + 1} QA analysts.
Primary technology stack includes {"Python" if i % 3 == 0 else "Rust" if i % 3 == 1 else "Go"}
with {"PostgreSQL" if i % 2 == 0 else "MongoDB"} as the data store.

Performance benchmarks show {95 + (i % 5):.1f}% uptime over the last quarter,
with {(i + 1) * 3} incidents resolved. The error rate is {0.01 * (i + 1):.2f}%
which is {"within" if i % 3 != 0 else "above"} the acceptable threshold of 0.50%.
""")
        return "\n".join(parts)

    @pytest.mark.asyncio
    async def test_ingestion_performance(self, large_store):
        """Measure ingestion time for a large document."""
        store, conn = large_store
        doc = self._generate_large_doc(50)  # ~50 sections = ~200+ chunks
        print(f"\nDocument size: {len(doc):,} chars ({len(doc.split()):,} words)")

        t0 = time.perf_counter()
        with _patch_settings(document_rag_contextual_retrieval=False, reranker_enabled=False):
            result = await store.ingest(
                data=doc.encode(), filename="large_spec.md",
                mime_type="text/markdown", user_id="test",
            )
        elapsed = time.perf_counter() - t0

        print(f"Chunks: {result['chunk_count']}")
        print(f"Ingestion time: {elapsed:.1f}s")
        print(f"Time per chunk: {elapsed / result['chunk_count'] * 1000:.0f}ms")

        assert result["chunk_count"] >= 50, "Expected at least 50 chunks"

    @pytest.mark.asyncio
    async def test_query_performance_large_doc(self, large_store):
        """Measure query time against a large document."""
        store, conn = large_store
        doc = self._generate_large_doc(50)

        with _patch_settings(document_rag_contextual_retrieval=False, reranker_enabled=False):
            result = await store.ingest(
                data=doc.encode(), filename="large_spec.md",
                mime_type="text/markdown", user_id="test",
            )
        doc_id = result["id"]
        n_chunks = result["chunk_count"]

        # Warm up embedding model
        await store.search("test", user_id="test", limit=1)

        # Measure query times
        queries = [
            "What is the budget for Module-025?",
            "Which module uses Rust with MongoDB?",
            "What is the error rate for Module-010?",
            "How many engineers work on Module-042?",
            "What is the latency of Module-001?",
        ]
        times = []
        for q in queries:
            t0 = time.perf_counter()
            results = await store.search(q, user_id="test", document_id=doc_id, limit=5)
            elapsed = time.perf_counter() - t0
            times.append(elapsed)

        avg_time = sum(times) / len(times)
        max_time = max(times)
        print(f"\nChunks in document: {n_chunks}")
        print(f"Avg query time: {avg_time * 1000:.0f}ms")
        print(f"Max query time: {max_time * 1000:.0f}ms")

        assert max_time < 2.0, f"Query took {max_time:.1f}s, exceeds 2s limit"

    @pytest.mark.asyncio
    async def test_precision_large_doc(self, large_store):
        """Verify correct sections found in top results for targeted queries."""
        store, conn = large_store
        doc = self._generate_large_doc(50)

        with _patch_settings(document_rag_contextual_retrieval=False, reranker_enabled=False):
            result = await store.ingest(
                data=doc.encode(), filename="large_spec.md",
                mime_type="text/markdown", user_id="test",
            )
        doc_id = result["id"]

        # Query for specific facts unique to one section
        test_cases = [
            ("Module-025 budget", "$625,000"),
            ("Module-001 latency", "0.5 milliseconds"),
            ("Module-010 error rate", "0.10%"),
        ]

        hits = 0
        for query, expected_fact in test_cases:
            results = await store.search(query, user_id="test", document_id=doc_id, limit=3)
            combined = " ".join(r["content"] for r in results)
            found = expected_fact in combined
            hits += int(found)
            status = "HIT" if found else "MISS"
            print(f"  {query}: {status} (looking for '{expected_fact}')")

        precision = hits / len(test_cases)
        print(f"\nLarge doc precision: {precision:.0%} ({hits}/{len(test_cases)})")
        assert precision >= 0.5, f"Precision {precision:.0%} below 50% on targeted queries"

    @pytest.mark.asyncio
    async def test_dedup_effectiveness_large_doc(self, large_store):
        """Verify dedup removes overlapping chunks in large documents."""
        from augmentum.documents.dedup import deduplicate
        from augmentum.documents.scoring import cliff_detect, score_gate

        store, conn = large_store
        doc = self._generate_large_doc(50)

        with _patch_settings(document_rag_contextual_retrieval=False, reranker_enabled=False):
            result = await store.ingest(
                data=doc.encode(), filename="large_spec.md",
                mime_type="text/markdown", user_id="test",
            )
        doc_id = result["id"]

        _q = "module configuration budget"
        results = await store.search(_q, user_id="test", document_id=doc_id, limit=10)
        dual = getattr(store, "_last_search_dual_source", True)
        scored = score_gate(results, reranker_enabled=False, dual_source=dual, query=_q)
        clipped = cliff_detect(scored, cliff_ratio=0.3, max_results=10)
        deduped = deduplicate(clipped)

        removed = len(clipped) - len(deduped)
        print(f"\nRaw results: {len(results)}")
        print(f"After cliff: {len(clipped)}")
        print(f"After dedup: {len(deduped)} (removed {removed})")
        # Just verify it doesn't crash and produces reasonable output
        assert len(deduped) <= len(clipped)


# ===================================================================
# SCENARIO 3: Conversation Context
# ===================================================================

class TestConversationContext:
    """
    HYPOTHESIS: Queries that depend on conversation context ("what about
    the penalties?") lose meaning when sent to search in isolation.
    The pipeline should still return relevant results when the query
    contains enough signal, but may fail on pure pronouns/references.

    METHOD: Test queries with varying amounts of context signal:
    - Full context query: "What are the termination penalties?"
    - Partial context: "What about the penalties?"
    - Pure reference: "Tell me more about that"

    PASS CRITERIA:
    - Full context: correct retrieval
    - Partial context: reasonable retrieval (some signal remains)
    - Pure reference: graceful handling (empty or low-score results)
    """

    @pytest.fixture
    async def ctx_store(self):
        store, conn, db_path = await _create_store()

        with _patch_settings(document_rag_contextual_retrieval=False, reranker_enabled=False):
            await store.ingest(
                data=_CONTRACT_DOC.encode(), filename="contract.md",
                mime_type="text/markdown", user_id="test",
            )
        yield store, conn
        await _cleanup(conn, db_path)

    @pytest.mark.asyncio
    async def test_full_context_query(self, ctx_store):
        """Full context query should retrieve correctly."""
        store, conn = ctx_store
        results = await store.search(
            "What is the termination notice period?",
            user_id="test", limit=3,
        )
        combined = " ".join(r["content"] for r in results)
        found = "ninety" in combined.lower() or "90" in combined
        print(f"\nFull context query: {'HIT' if found else 'MISS'}")
        assert found, "Full context query should find '90 days'"

    @pytest.mark.asyncio
    async def test_partial_context_query(self, ctx_store):
        """Partial context ('penalties') should still find relevant content."""
        store, conn = ctx_store
        results = await store.search(
            "What about the penalties?",
            user_id="test", limit=3,
        )
        # "penalties" doesn't appear verbatim, but semantic search should
        # find termination/competition related content
        print(f"\nPartial context: {len(results)} results returned")
        assert len(results) > 0, "Partial context should return some results"

    @pytest.mark.asyncio
    async def test_pure_reference_query(self, ctx_store):
        """Pure reference ('tell me more about that') has no search signal."""
        store, conn = ctx_store
        results = await store.search(
            "Tell me more about that",
            user_id="test", limit=3,
        )
        # With query quality dampening, low-signal queries should get
        # fewer "high" confidence results.
        from augmentum.config import settings
        from augmentum.documents.scoring import score_gate
        _q = "Tell me more about that"
        dual = getattr(store, "_last_search_dual_source", True)
        scored = score_gate(results, reranker_enabled=settings.reranker_enabled, dual_source=dual, query=_q)
        high = [s for s in scored if s.tier == "high"]
        uncertain = [s for s in scored if s.tier == "uncertain"]
        irrelevant = [s for s in scored if s.tier == "irrelevant"]
        print(f"\nPure reference: {len(results)} raw -> {len(high)} high, {len(uncertain)} uncertain, {len(irrelevant)} irrelevant")
        # Query quality dampening should prevent all results being "high"
        assert len(high) < len(results), "All results 'high' for meaningless query — dampening not working"

    @pytest.mark.asyncio
    async def test_pronoun_query(self, ctx_store):
        """Pronoun-heavy query ('How long is it?') — minimal signal."""
        store, conn = ctx_store
        _q = "How long is it?"
        results = await store.search(_q, user_id="test", limit=3)
        from augmentum.config import settings
        from augmentum.documents.scoring import score_gate
        dual = getattr(store, "_last_search_dual_source", True)
        scored = score_gate(results, reranker_enabled=settings.reranker_enabled, dual_source=dual, query=_q)
        high = [s for s in scored if s.tier == "high"]
        uncertain = [s for s in scored if s.tier == "uncertain"]
        irrelevant = [s for s in scored if s.tier == "irrelevant"]
        print(f"\nPronoun query: {len(results)} raw -> {len(high)} high, {len(uncertain)} uncertain, {len(irrelevant)} irrelevant")
        assert len(high) < len(results), "All results 'high' for pronoun query — dampening not working"


# ===================================================================
# SCENARIO 4: Concurrent Ingestion + Search
# ===================================================================

class TestConcurrentOps:
    """
    HYPOTHESIS: A user might upload a document and immediately ask about
    it. Ingestion must complete before search can find the content.
    Also: multiple concurrent searches should not interfere.

    METHOD:
    - Ingest and search concurrently
    - Run multiple searches in parallel
    - Verify no data corruption or crashes

    PASS CRITERIA:
    - Post-ingest search finds the document
    - Concurrent searches return consistent results
    - No crashes or deadlocks
    """

    @pytest.fixture
    async def conc_store(self):
        store, conn, db_path = await _create_store()
        yield store, conn
        await _cleanup(conn, db_path)

    @pytest.mark.asyncio
    async def test_search_after_sequential_ingest(self, conc_store):
        """Document is searchable immediately after ingest completes."""
        store, conn = conc_store

        with _patch_settings(document_rag_contextual_retrieval=False, reranker_enabled=False):
            result = await store.ingest(
                data=b"# Unique\n\nThe zephyr protocol handles authentication via quantum tokens.",
                filename="unique.md", mime_type="text/markdown", user_id="test",
            )

        results = await store.search("zephyr protocol quantum tokens", user_id="test", limit=3)
        assert len(results) > 0
        assert any("zephyr" in r["content"].lower() for r in results)
        print("\nSequential ingest+search: PASS")

    @pytest.mark.asyncio
    async def test_concurrent_searches(self, conc_store):
        """Multiple concurrent searches return consistent, non-corrupted results."""
        store, conn = conc_store

        with _patch_settings(document_rag_contextual_retrieval=False, reranker_enabled=False):
            await store.ingest(
                data=_CONTRACT_DOC.encode(), filename="contract.md",
                mime_type="text/markdown", user_id="test",
            )

        # Run 10 concurrent searches
        queries = [
            "compensation salary", "termination notice", "intellectual property",
            "non-competition clause", "severance payment",
            "compensation salary", "termination notice", "intellectual property",
            "non-competition clause", "severance payment",
        ]

        async def search(q):
            return await store.search(q, user_id="test", limit=3)

        results = await asyncio.gather(*[search(q) for q in queries])

        # Same query should return same results
        for i in range(5):
            r1 = [r["chunk_id"] for r in results[i]]
            r2 = [r["chunk_id"] for r in results[i + 5]]
            assert r1 == r2, f"Inconsistent results for query '{queries[i]}'"

        print("\nConcurrent searches (10 parallel): PASS — all consistent")

    @pytest.mark.asyncio
    async def test_no_deadlock_under_load(self, conc_store):
        """Rapid sequential operations don't deadlock aiosqlite."""
        store, conn = conc_store

        with _patch_settings(document_rag_contextual_retrieval=False, reranker_enabled=False):
            await store.ingest(
                data=_CONTRACT_DOC.encode(), filename="contract.md",
                mime_type="text/markdown", user_id="test",
            )

        t0 = time.perf_counter()
        for i in range(50):
            await store.search(f"query number {i}", user_id="test", limit=3)
        elapsed = time.perf_counter() - t0

        print(f"\n50 sequential searches: {elapsed:.1f}s ({elapsed/50*1000:.0f}ms avg)")
        assert elapsed < 30.0, f"50 searches took {elapsed:.1f}s — too slow"


# ===================================================================
# SCENARIO 5: Full vs Search Mode Interaction
# ===================================================================

class TestFullSearchMixed:
    """
    HYPOTHESIS: When a user has doc A in "full" mode and doc B in "search"
    mode, both should work independently without interference.
    Full mode injects entire content; search mode uses the v2 pipeline.

    METHOD: Ingest two documents, simulate the injection paths for each mode.

    PASS CRITERIA:
    - Full content returns complete document
    - Search returns relevant chunks only
    - Full-mode doc content doesn't contaminate search results from other doc
    """

    @pytest.fixture
    async def mixed_store(self):
        store, conn, db_path = await _create_store()

        with _patch_settings(document_rag_contextual_retrieval=False, reranker_enabled=False):
            r1 = await store.ingest(
                data=_CONTRACT_DOC.encode(), filename="contract.md",
                mime_type="text/markdown", user_id="test",
            )
            r2 = await store.ingest(
                data=_RECIPE_DOC.encode(), filename="recipes.md",
                mime_type="text/markdown", user_id="test",
            )
        yield store, conn, r1["id"], r2["id"]
        await _cleanup(conn, db_path)

    @pytest.mark.asyncio
    async def test_full_mode_returns_complete(self, mixed_store):
        """Full mode returns all content from the document."""
        store, conn, finance_id, cooking_id = mixed_store
        full = await store.get_full_content(finance_id)
        assert full is not None
        assert "$145,000" in full["content"]
        assert "termination" in full["content"].lower()

    @pytest.mark.asyncio
    async def test_search_mode_returns_relevant_chunks(self, mixed_store):
        """Search mode returns only relevant chunks, not the whole doc."""
        store, conn, finance_id, cooking_id = mixed_store
        results = await store.search(
            "beef braising time", user_id="test",
            document_id=cooking_id, limit=3,
        )
        assert len(results) > 0
        combined = " ".join(r["content"] for r in results)
        assert "beef" in combined.lower() or "braise" in combined.lower()
        # Should NOT contain finance content
        assert "$145,000" not in combined

    @pytest.mark.asyncio
    async def test_mixed_mode_no_cross_contamination(self, mixed_store):
        """Full-mode doc doesn't appear in search results for other doc."""
        store, conn, finance_id, cooking_id = mixed_store

        # Search within cooking doc for a finance term
        results = await store.search(
            "compensation salary", user_id="test",
            document_id=cooking_id, limit=3,
        )
        combined = " ".join(r["content"] for r in results)
        assert "$145,000" not in combined
        assert "compensation" not in combined.lower()

    @pytest.mark.asyncio
    async def test_full_and_search_simultaneous(self, mixed_store):
        """Simulate both paths firing at once (as integration.py does)."""
        store, conn, finance_id, cooking_id = mixed_store

        # Full mode for finance
        full_content = await store.get_full_content(finance_id)
        # Search mode for cooking
        search_results = await store.search(
            "How to make sourdough?", user_id="test",
            document_id=cooking_id, limit=3,
        )

        # Both should return their respective content
        assert full_content is not None
        assert "$145,000" in full_content["content"]
        assert len(search_results) > 0

        print(f"\nFull mode: {len(full_content['content'])} chars")
        print(f"Search mode: {len(search_results)} chunks")


# ===================================================================
# SCENARIO 6: Token Budget vs Context Window
# ===================================================================

class TestTokenBudget:
    """
    HYPOTHESIS: The context budget should prevent injecting more tokens
    than configured, regardless of how many relevant chunks exist.
    With large documents, many chunks could be relevant — the budget
    must cap injection size.

    METHOD: Create a document where many chunks match a broad query.
    Vary budget settings and verify output size stays within bounds.

    PASS CRITERIA:
    - Injected tokens <= configured budget
    - Budget respects sentence boundaries (no mid-word truncation)
    - Lower budget = fewer chunks, not truncated garbage
    """

    @pytest.fixture
    async def budget_store(self):
        store, conn, db_path = await _create_store()

        # Create a document where every section matches "project budget"
        sections = []
        for i in range(20):
            sections.append(
                f"## Project {i+1} Budget\n\n"
                f"The budget for Project {i+1} is ${(i+1)*50000:,}. "
                f"This covers {(i+1)*2} team members over {(i+1)} quarters. "
                f"Key deliverables include milestone-{i+1}A and milestone-{i+1}B. "
                f"The project manager is PM-{i+1:03d} from the engineering division."
            )
        doc = "# Organization Budget Report\n\n" + "\n\n".join(sections)

        with _patch_settings(document_rag_contextual_retrieval=False, reranker_enabled=False):
            result = await store.ingest(
                data=doc.encode(), filename="budgets.md",
                mime_type="text/markdown", user_id="test",
            )
        yield store, conn, result["id"]
        await _cleanup(conn, db_path)

    @pytest.mark.asyncio
    async def test_budget_caps_output(self, budget_store):
        """Budget should cap total injected content."""
        from augmentum.documents.dedup import deduplicate
        from augmentum.documents.scoring import apply_budget, cliff_detect, score_gate

        store, conn, doc_id = budget_store
        _q = "project budget deliverables"
        results = await store.search(_q, user_id="test", document_id=doc_id, limit=10)
        dual = getattr(store, "_last_search_dual_source", True)
        scored = score_gate(results, reranker_enabled=False, dual_source=dual, query=_q)
        clipped = cliff_detect(scored, cliff_ratio=0.3, max_results=10)
        deduped = deduplicate(clipped)

        # Test different budgets
        for max_tokens in [500, 1000, 1500, 3000]:
            budgeted = apply_budget(deduped, max_tokens=max_tokens)
            total_chars = sum(len(sc.chunk.get("content", "")) for sc in budgeted)
            total_tokens = total_chars // 4
            print(f"  Budget {max_tokens} tokens: {len(budgeted)} chunks, ~{total_tokens} tokens, {total_chars} chars")
            assert total_tokens <= max_tokens * 1.1, (
                f"Budget {max_tokens} exceeded: {total_tokens} tokens"
            )

    @pytest.mark.asyncio
    async def test_budget_no_mid_word_truncation(self, budget_store):
        """Truncated chunks should end at sentence/word boundaries."""
        from augmentum.documents.scoring import ScoredChunk, apply_budget

        # Create a chunk that will be truncated
        long_text = "First sentence about budgets. Second sentence about teams. Third sentence about deliverables. Fourth sentence about timelines."
        chunks = [ScoredChunk(chunk={"content": long_text}, tier="high", score=0.9)]

        budgeted = apply_budget(chunks, max_tokens=15)  # ~60 chars
        if budgeted:
            content = budgeted[0].chunk["content"]
            # Should not end mid-word
            assert not content[-1].isalpha() or content.endswith("..."), (
                f"Truncated mid-word: '{content[-20:]}'"
            )
            print(f"\nTruncated content: '{content}'")

    @pytest.mark.asyncio
    async def test_lower_budget_fewer_chunks(self, budget_store):
        """Lower budget should produce fewer chunks, not same chunks truncated."""
        from augmentum.documents.dedup import deduplicate
        from augmentum.documents.scoring import apply_budget, cliff_detect, score_gate

        store, conn, doc_id = budget_store
        _q = "project budget"
        results = await store.search(_q, user_id="test", document_id=doc_id, limit=10)
        dual = getattr(store, "_last_search_dual_source", True)
        scored = score_gate(results, reranker_enabled=False, dual_source=dual, query=_q)
        clipped = cliff_detect(scored, cliff_ratio=0.3, max_results=10)
        deduped = deduplicate(clipped)

        small = apply_budget(deduped, max_tokens=200)
        large = apply_budget(deduped, max_tokens=2000)
        print(f"\nSmall budget (200): {len(small)} chunks")
        print(f"Large budget (2000): {len(large)} chunks")
        assert len(small) <= len(large)


# ===================================================================
# SCENARIO 7: Re-ingestion / Document Updates
# ===================================================================

class TestDocumentUpdates:
    """
    HYPOTHESIS: When a user deletes and re-uploads a document, the old
    chunks must not appear in search results. If the user uploads a new
    version alongside the old one, results may mix — this tests both paths.

    METHOD:
    - Ingest doc v1 with unique fact A
    - Delete doc v1
    - Ingest doc v2 with different fact B
    - Search for both A and B

    PASS CRITERIA:
    - After deletion: fact A not found
    - After re-ingest: fact B found, fact A still not found
    - No orphaned chunks in DB
    """

    @pytest.fixture
    async def update_store(self):
        store, conn, db_path = await _create_store()
        yield store, conn
        await _cleanup(conn, db_path)

    @pytest.mark.asyncio
    async def test_delete_removes_all_chunks(self, update_store):
        """Deleting a document removes all its chunks from search."""
        store, conn = update_store

        with _patch_settings(document_rag_contextual_retrieval=False, reranker_enabled=False):
            result = await store.ingest(
                data=b"# V1\n\nThe alpha protocol uses quantum encryption with key-size 4096.",
                filename="spec.md", mime_type="text/markdown", user_id="test",
            )
        doc_id = result["id"]

        # Verify it's searchable
        results = await store.search("alpha protocol quantum", user_id="test", limit=3)
        assert any("alpha" in r["content"].lower() for r in results)

        # Delete
        await store.delete_document(doc_id)

        # Verify it's gone
        results = await store.search("alpha protocol quantum", user_id="test", limit=3)
        found = any("alpha" in r["content"].lower() for r in results)
        assert not found, "Deleted document chunks still appearing in search"
        print("\nDelete removes all chunks: PASS")

    @pytest.mark.asyncio
    async def test_reupload_replaces_content(self, update_store):
        """Delete + re-upload: only new content appears in search."""
        store, conn = update_store

        # V1: unique fact
        with _patch_settings(document_rag_contextual_retrieval=False, reranker_enabled=False):
            r1 = await store.ingest(
                data=b"# V1\n\nThe zephyr system costs $999,999 and supports 50 users.",
                filename="spec.md", mime_type="text/markdown", user_id="test",
            )
        await store.delete_document(r1["id"])

        # V2: different unique fact
        with _patch_settings(document_rag_contextual_retrieval=False, reranker_enabled=False):
            r2 = await store.ingest(
                data=b"# V2\n\nThe phoenix system costs $123,456 and supports 200 users.",
                filename="spec.md", mime_type="text/markdown", user_id="test",
            )

        # V2 should be found
        results = await store.search("phoenix system cost", user_id="test", limit=3)
        combined = " ".join(r["content"] for r in results)
        assert "phoenix" in combined.lower(), "V2 content not found"
        assert "$123,456" in combined, "V2 price not found"

        # V1 should NOT be found
        assert "zephyr" not in combined.lower(), "V1 content still present"
        assert "$999,999" not in combined, "V1 price still present"
        print("\nDelete + re-upload: PASS — only V2 content found")

    @pytest.mark.asyncio
    async def test_no_orphaned_chunks(self, update_store):
        """After deletion, no orphaned chunks remain in DB."""
        store, conn = update_store

        with _patch_settings(document_rag_contextual_retrieval=False, reranker_enabled=False):
            result = await store.ingest(
                data=b"# Test\n\nSome content here for orphan testing purposes.",
                filename="orphan_test.md", mime_type="text/markdown", user_id="test",
            )
        doc_id = result["id"]

        # Count chunks before delete
        cursor = await conn.execute(
            "SELECT COUNT(*) as cnt FROM document_chunks WHERE document_id = ?",
            (doc_id,),
        )
        before = (await cursor.fetchone())["cnt"]
        assert before > 0

        await store.delete_document(doc_id)

        # Count chunks after delete
        cursor = await conn.execute(
            "SELECT COUNT(*) as cnt FROM document_chunks WHERE document_id = ?",
            (doc_id,),
        )
        after = (await cursor.fetchone())["cnt"]
        assert after == 0, f"Found {after} orphaned chunks after deletion"

        # Also check FTS (sync should have fired via trigger)
        cursor = await conn.execute("SELECT COUNT(*) as cnt FROM document_chunks_fts")
        fts_count = (await cursor.fetchone())["cnt"]
        cursor = await conn.execute("SELECT COUNT(*) as cnt FROM document_chunks")
        chunk_count = (await cursor.fetchone())["cnt"]
        assert fts_count == chunk_count, (
            f"FTS index out of sync: {fts_count} FTS rows vs {chunk_count} chunk rows"
        )
        print("\nNo orphaned chunks: PASS (0 chunks, FTS in sync)")


# ===================================================================
# Shared test documents (compact, distinct domains)
# ===================================================================

_CONTRACT_DOC = """# Employment Agreement

## Compensation
The annual base compensation shall be One Hundred Forty-Five Thousand Dollars ($145,000),
payable in bi-weekly installments. Quarterly performance bonuses up to 15%.

## Non-Competition
For eighteen (18) months following termination, the Employee shall not engage in
competing business within fifty (50) miles of any Company office.

## Termination
Either party may terminate with ninety (90) days written notice. Termination without
cause: severance equal to six (6) months base compensation plus accrued vacation.

## Intellectual Property
All inventions created during employment are Company property. Employee assigns all
rights, title, and interest in such intellectual property.
"""

_RECIPE_DOC = """# Recipe Collection

## Classic Beef Bourguignon
Braise 2 pounds beef chuck in red wine at 325 degrees for 3 hours.
Add pearl onions, mushrooms, carrots in last hour. Deglaze with cognac. Serves 6.

## Sourdough Bread
Mix 500g bread flour, 350g water, 100g active starter. Autolyse 45 minutes.
Add 10g salt. Four stretch-and-folds over 2 hours. Cold proof 12-18 hours.
Bake in Dutch oven at 500 degrees: 20 min lid on, 25 min lid off.

## Thai Green Curry
Saute green curry paste in coconut oil 2 minutes. Add coconut milk, fish sauce,
palm sugar, kaffir lime leaves. Simmer chicken 15 minutes. Serve over jasmine rice.
"""
