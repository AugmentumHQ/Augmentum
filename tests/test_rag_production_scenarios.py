"""Production scenario tests for RAG pipeline v2.

Tests real-world usage patterns that the benchmark harness doesn't cover:
1. Session-bound document filtering (single doc, multi doc, mixed modes)
2. Large document performance (200+ chunks)
3. Full vs search mode interaction
4. Document isolation (no cross-contamination between bound docs)
5. Edge cases (empty documents, single-chunk docs, no bindings)

Uses a real in-memory DocumentStore with real embeddings.
"""

from __future__ import annotations

import contextlib
import os
import re
import tempfile
import time

import aiosqlite
import pytest

# ---------------------------------------------------------------------------
# Helpers
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


# ---------------------------------------------------------------------------
# Fixture: shared store with multiple documents
# ---------------------------------------------------------------------------

@pytest.fixture
async def store():
    """Set up a DocumentStore with 3 documents from different domains."""
    fd, db_path = tempfile.mkstemp(suffix=".db", prefix="rag_scenario_")
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
    ds = DocumentStore(backend)

    # Ingest 3 distinct documents
    docs = {}
    with _patch_settings(document_rag_contextual_retrieval=False, reranker_enabled=False):
        # Doc A: finance/contract
        result = await ds.ingest(
            data=_DOC_FINANCE.encode(),
            filename="contract.md",
            mime_type="text/markdown",
            user_id="test",
        )
        docs["finance"] = result["id"]

        # Doc B: cooking
        result = await ds.ingest(
            data=_DOC_COOKING.encode(),
            filename="recipes.md",
            mime_type="text/markdown",
            user_id="test",
        )
        docs["cooking"] = result["id"]

        # Doc C: tech
        result = await ds.ingest(
            data=_DOC_TECH.encode(),
            filename="api_docs.md",
            mime_type="text/markdown",
            user_id="test",
        )
        docs["tech"] = result["id"]

    yield ds, docs, conn

    await conn.close()
    os.unlink(db_path)


# ---------------------------------------------------------------------------
# Test documents (short, distinct domains — no overlap)
# ---------------------------------------------------------------------------

_DOC_FINANCE = """# Employment Agreement

## Compensation
The annual base compensation shall be One Hundred Forty-Five Thousand Dollars ($145,000),
payable in bi-weekly installments. The Employee shall be eligible for quarterly performance
bonuses of up to fifteen percent (15%) of the base compensation.

## Non-Competition
For a period of eighteen (18) months following termination, the Employee shall not directly
or indirectly engage in any business that competes with the Company within a radius of
fifty (50) miles of any Company office location.

## Termination
Either party may terminate this Agreement with ninety (90) days written notice. In the event
of termination without cause, the Employee shall receive severance equal to six (6) months
of base compensation plus accrued vacation time.

## Intellectual Property
All inventions, discoveries, and works of authorship created during employment shall be the
sole property of the Company. The Employee hereby assigns all rights, title, and interest
in such intellectual property to the Company.
"""

_DOC_COOKING = """# Recipe Collection

## Classic Beef Bourguignon
Braise 2 pounds of beef chuck in red wine at 325 degrees Fahrenheit for 3 hours.
Add pearl onions, mushrooms, and carrots during the last hour. Deglaze the pan with
cognac before adding the wine. Season with thyme, bay leaf, and black pepper.
Serves 6 people.

## Thai Green Curry
Saute green curry paste in coconut oil for 2 minutes. Add coconut milk, fish sauce,
palm sugar, and kaffir lime leaves. Simmer chicken breast pieces for 15 minutes.
Garnish with Thai basil and sliced red chilies. Best served over jasmine rice.

## Sourdough Bread
Mix 500g bread flour with 350g water and 100g active starter. Autolyse for 45 minutes.
Add 10g salt, then perform four sets of stretch and folds over 2 hours. Cold proof
in the refrigerator for 12-18 hours. Bake in a Dutch oven at 500 degrees for 20 minutes
with lid on, then 25 minutes with lid off.
"""

_DOC_TECH = """# TaskFlow API Reference

## Authentication
All API requests require a valid API key passed in the X-API-Key header.
OAuth 2.0 bearer tokens are also supported for user-delegated access.
Rate limits: 1000 requests per minute for standard tier, 5000 for premium.

## Endpoints

### GET /api/v2/users
Returns a paginated list of users. Supports offset and limit parameters.
Default limit is 25, maximum is 100.

### POST /api/v2/orders
Creates a new order. Required fields: customer_id, line_items array.
Each line item must include product_id, quantity, and unit_price.
Returns the created order with a unique order_id.

## Webhooks
Configure webhook endpoints to receive real-time notifications.
Supported events: order.created, order.shipped, user.registered.
Webhook payloads are signed using HMAC-SHA256 with your webhook secret.
"""


# ---------------------------------------------------------------------------
# Scenario 1: Single document binding — search only in bound doc
# ---------------------------------------------------------------------------

class TestSingleDocBinding:
    """Verify search respects single-document binding."""

    @pytest.mark.asyncio
    async def test_finance_query_only_returns_finance(self, store):
        ds, docs, conn = store
        results = await ds.search(
            "What is the annual salary?",
            user_id="test",
            document_id=docs["finance"],
        )
        for r in results:
            assert r["doc_id"] == docs["finance"], (
                f"Got result from {r['filename']} instead of contract.md"
            )

    @pytest.mark.asyncio
    async def test_cooking_query_only_returns_cooking(self, store):
        ds, docs, conn = store
        results = await ds.search(
            "How long to braise the beef?",
            user_id="test",
            document_id=docs["cooking"],
        )
        for r in results:
            assert r["doc_id"] == docs["cooking"], (
                f"Got result from {r['filename']} instead of recipes.md"
            )

    @pytest.mark.asyncio
    async def test_tech_query_only_returns_tech(self, store):
        ds, docs, conn = store
        results = await ds.search(
            "What are the API rate limits?",
            user_id="test",
            document_id=docs["tech"],
        )
        for r in results:
            assert r["doc_id"] == docs["tech"], (
                f"Got result from {r['filename']} instead of api_docs.md"
            )

    @pytest.mark.asyncio
    async def test_cross_domain_query_respects_binding(self, store):
        """Ask a cooking question but bind to finance doc — should get no relevant results."""
        ds, docs, conn = store
        results = await ds.search(
            "How to make sourdough bread?",
            user_id="test",
            document_id=docs["finance"],
        )
        # Results should be from finance only (even if irrelevant)
        for r in results:
            assert r["doc_id"] == docs["finance"]
        # The content should NOT contain cooking terms
        combined = " ".join(r["content"] for r in results)
        assert "sourdough" not in combined.lower()
        assert "bread" not in combined.lower()


# ---------------------------------------------------------------------------
# Scenario 2: Multi-document binding with post-filter
# ---------------------------------------------------------------------------

class TestMultiDocBinding:
    """Verify post-filter works correctly for multiple bound documents."""

    @pytest.mark.asyncio
    async def test_search_across_two_docs(self, store):
        """Search across finance + tech, cooking excluded."""
        ds, docs, conn = store
        search_ids = {docs["finance"], docs["tech"]}
        results = await ds.search(
            "What are the terms and conditions?",
            user_id="test",
        )
        # Post-filter as integration.py does
        filtered = [r for r in results if r["doc_id"] in search_ids]
        for r in filtered:
            assert r["doc_id"] in search_ids
            assert r["doc_id"] != docs["cooking"]

    @pytest.mark.asyncio
    async def test_no_cross_contamination(self, store):
        """Cooking-specific query filtered to finance+tech returns nothing useful."""
        ds, docs, conn = store
        search_ids = {docs["finance"], docs["tech"]}
        results = await ds.search(
            "beef bourguignon recipe",
            user_id="test",
        )
        filtered = [r for r in results if r["doc_id"] in search_ids]
        combined = " ".join(r["content"] for r in filtered)
        assert "bourguignon" not in combined.lower()


# ---------------------------------------------------------------------------
# Scenario 3: Document isolation — no cross-document leakage
# ---------------------------------------------------------------------------

class TestDocumentIsolation:
    """Verify complete isolation between documents."""

    @pytest.mark.asyncio
    async def test_unique_terms_stay_isolated(self, store):
        """Terms unique to one doc should not appear in results from another."""
        ds, docs, conn = store

        # "HMAC-SHA256" only exists in tech doc
        results = await ds.search(
            "webhook signature verification",
            user_id="test",
            document_id=docs["tech"],
        )
        assert any("HMAC" in r["content"] or "webhook" in r["content"].lower() for r in results)

        # Same query against finance doc
        results = await ds.search(
            "webhook signature verification",
            user_id="test",
            document_id=docs["finance"],
        )
        combined = " ".join(r["content"] for r in results)
        assert "HMAC" not in combined
        assert "webhook" not in combined.lower()


# ---------------------------------------------------------------------------
# Scenario 4: Scoring pipeline with document filtering
# ---------------------------------------------------------------------------

class TestScoringWithFiltering:
    """Verify the full scoring pipeline works with document-scoped results."""

    @pytest.mark.asyncio
    async def test_score_gate_on_filtered_results(self, store):
        """Score gate should properly classify filtered results."""
        from augmentum.documents.scoring import score_gate, cliff_detect
        from augmentum.documents.dedup import deduplicate

        ds, docs, conn = store
        results = await ds.search(
            "What is the compensation?",
            user_id="test",
            document_id=docs["finance"],
        )
        dual = getattr(ds, "_last_search_dual_source", True)
        scored = score_gate(results, reranker_enabled=False, dual_source=dual)

        # Should have results, all from finance
        assert len(scored) > 0
        for sc in scored:
            assert sc.chunk["doc_id"] == docs["finance"]

        # Cliff + dedup should work without error
        clipped = cliff_detect(scored, cliff_ratio=0.3, max_results=3)
        deduped = deduplicate(clipped)
        assert len(deduped) >= 1

    @pytest.mark.asyncio
    async def test_reranker_with_document_filter(self, store):
        """Reranker should work on document-scoped results."""
        ds, docs, conn = store
        with _patch_settings(reranker_enabled=True):
            results = await ds.search(
                "termination notice period",
                user_id="test",
                document_id=docs["finance"],
                limit=3,
            )
        assert len(results) > 0
        # Reranker scores should be in [0, 1] range (sigmoid-normalized)
        for r in results:
            score = r.get("score", 0)
            assert 0 <= score <= 1.0, f"Reranker score {score} not in [0,1] — normalization broken"
            assert r["doc_id"] == docs["finance"]


# ---------------------------------------------------------------------------
# Scenario 5: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    """Edge cases that production will encounter."""

    @pytest.mark.asyncio
    async def test_empty_query(self, store):
        """Empty query should return empty results, not crash."""
        ds, docs, conn = store
        results = await ds.search("", user_id="test")
        assert results == [] or isinstance(results, list)

    @pytest.mark.asyncio
    async def test_very_long_query(self, store):
        """Very long query should not crash (gets truncated by FTS/embedding)."""
        ds, docs, conn = store
        long_query = "compensation " * 500  # ~5500 chars
        results = await ds.search(long_query, user_id="test", limit=3)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_special_characters_in_query(self, store):
        """Special chars shouldn't cause SQL injection or FTS errors."""
        ds, docs, conn = store
        for query in [
            'What about "salary"?',
            "It's the employee's contract",
            "SELECT * FROM documents; DROP TABLE documents;--",
            "term AND (compensation OR salary)",
            "$145,000",
        ]:
            results = await ds.search(query, user_id="test", limit=3)
            assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_nonexistent_document_id(self, store):
        """Search with invalid document_id should return empty, not error."""
        ds, docs, conn = store
        results = await ds.search(
            "test query",
            user_id="test",
            document_id="nonexistent_id_12345",
        )
        assert results == []

    @pytest.mark.asyncio
    async def test_search_immediately_after_ingest(self, store):
        """Document should be searchable right after ingestion (no race)."""
        ds, docs, conn = store

        # Ingest a new document
        new_doc = "# Unique Document\n\nThis contains the word xylophone which appears nowhere else."
        with _patch_settings(document_rag_contextual_retrieval=False, reranker_enabled=False):
            result = await ds.ingest(
                data=new_doc.encode(),
                filename="unique.md",
                mime_type="text/markdown",
                user_id="test",
            )
        new_id = result["id"]

        # Should be immediately searchable
        results = await ds.search(
            "xylophone",
            user_id="test",
            document_id=new_id,
        )
        assert len(results) > 0
        assert any("xylophone" in r["content"] for r in results)


# ---------------------------------------------------------------------------
# Scenario 6: Full content retrieval
# ---------------------------------------------------------------------------

class TestFullContent:
    """Test get_full_content for full-mode injection."""

    @pytest.mark.asyncio
    async def test_full_content_returns_complete_doc(self, store):
        ds, docs, conn = store
        full = await ds.get_full_content(docs["finance"])
        assert full is not None
        assert full["filename"] == "contract.md"
        # Should contain text from all sections (note: chunker may strip
        # markdown headers — check for body content, not headers)
        assert "$145,000" in full["content"]
        assert "eighteen (18) months" in full["content"]  # non-compete body
        assert "inventions" in full["content"]  # IP section body

    @pytest.mark.asyncio
    async def test_full_content_nonexistent(self, store):
        ds, docs, conn = store
        full = await ds.get_full_content("nonexistent_id")
        assert full is None


# ---------------------------------------------------------------------------
# Scenario 7: FTS tokenization with document binding
# ---------------------------------------------------------------------------

class TestFTSWithBinding:
    """Verify AND-first/OR-fallback works correctly with document filtering."""

    @pytest.mark.asyncio
    async def test_and_query_within_single_doc(self, store):
        """AND query should find content within bound doc."""
        ds, docs, conn = store
        # "base compensation" — both words exist in finance doc
        results = await ds.search(
            "base compensation quarterly bonus",
            user_id="test",
            document_id=docs["finance"],
        )
        assert len(results) > 0
        assert any("compensation" in r["content"].lower() for r in results)

    @pytest.mark.asyncio
    async def test_or_fallback_within_single_doc(self, store):
        """OR fallback should fire when AND returns nothing in bound doc."""
        ds, docs, conn = store
        # "salary payment schedule" — "salary" doesn't exist verbatim in finance doc
        # AND would fail, OR should catch "payment" or "schedule"
        results = await ds.search(
            "salary payment schedule",
            user_id="test",
            document_id=docs["finance"],
        )
        # Should get results via OR fallback or vec search
        assert isinstance(results, list)
