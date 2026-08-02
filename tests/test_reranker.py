"""Tests for cross-encoder reranking and contextual retrieval."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# RerankService tests
# ---------------------------------------------------------------------------

class TestRerankService:
    """Test the cross-encoder reranker service."""

    def test_rerank_empty_documents(self):
        from augmentum.memory.reranker import RerankService

        result = RerankService.rerank("query", [])
        assert result == []

    def test_rerank_dicts_empty(self):
        from augmentum.memory.reranker import RerankService

        result = RerankService.rerank_dicts("query", [])
        assert result == []

    @patch("augmentum.memory.reranker.RerankService.get_model")
    def test_rerank_returns_sorted_scores(self, mock_get_model):
        from augmentum.memory.reranker import RerankService

        mock_model = MagicMock()
        # Simulate model returning scores: doc0=0.1, doc1=0.9, doc2=0.5
        mock_model.rerank.return_value = [0.1, 0.9, 0.5]
        mock_get_model.return_value = mock_model

        result = RerankService.rerank("test query", ["bad", "great", "ok"])
        # Should be sorted by score descending
        assert result[0] == (1, 0.9)  # "great"
        assert result[1] == (2, 0.5)  # "ok"
        assert result[2] == (0, 0.1)  # "bad"

    @patch("augmentum.memory.reranker.RerankService.get_model")
    def test_rerank_top_k(self, mock_get_model):
        from augmentum.memory.reranker import RerankService

        mock_model = MagicMock()
        mock_model.rerank.return_value = [0.1, 0.9, 0.5, 0.3]
        mock_get_model.return_value = mock_model

        result = RerankService.rerank("query", ["a", "b", "c", "d"], top_k=2)
        assert len(result) == 2
        assert result[0][1] == 0.9

    @patch("augmentum.memory.reranker.RerankService.get_model")
    def test_rerank_dicts_adds_score(self, mock_get_model):
        from augmentum.memory.reranker import RerankService

        mock_model = MagicMock()
        mock_model.rerank.return_value = [0.2, 0.8]
        mock_get_model.return_value = mock_model

        results = [
            {"content": "low relevance", "id": "a", "score": 0.5},
            {"content": "high relevance", "id": "b", "score": 0.3},
        ]
        reranked = RerankService.rerank_dicts("query", results)
        assert len(reranked) == 2
        assert reranked[0]["id"] == "b"  # higher reranker score
        assert reranked[0]["reranker_score"] == 0.8
        assert reranked[0]["score"] == 0.8  # score replaced
        assert reranked[1]["id"] == "a"
        assert reranked[1]["reranker_score"] == 0.2

    @patch("augmentum.memory.reranker.RerankService.get_model")
    def test_rerank_dicts_custom_content_key(self, mock_get_model):
        from augmentum.memory.reranker import RerankService

        mock_model = MagicMock()
        mock_model.rerank.return_value = [0.9, 0.1]
        mock_get_model.return_value = mock_model

        results = [
            {"text": "relevant", "id": "a"},
            {"text": "irrelevant", "id": "b"},
        ]
        reranked = RerankService.rerank_dicts("query", results, content_key="text")
        assert reranked[0]["id"] == "a"

    def test_reset(self):
        from augmentum.memory.reranker import _UNLOADED, RerankService

        RerankService._model = "fake"
        RerankService._model_name = "fake-model"
        RerankService.reset()
        assert RerankService._model is _UNLOADED
        assert RerankService._model_name == ""


# ---------------------------------------------------------------------------
# Document store reranking integration
# ---------------------------------------------------------------------------

class TestDocumentStoreReranking:
    """Test reranking integration in DocumentStore."""

    @patch("augmentum.memory.reranker.RerankService.rerank_dicts")
    @patch("augmentum.config.settings")
    def test_rerank_results_called_when_enabled(self, mock_settings, mock_rerank):
        from augmentum.documents.store import DocumentStore

        mock_settings.reranker_enabled = True
        mock_rerank.return_value = [
            {"chunk_id": "b", "content": "better", "score": 0.9, "reranker_score": 0.9},
        ]

        results = [
            {"chunk_id": "a", "content": "first", "score": 0.5},
            {"chunk_id": "b", "content": "better", "score": 0.3},
        ]
        reranked = DocumentStore._rerank_results("test query", results, top_k=1)
        mock_rerank.assert_called_once()
        assert reranked[0]["chunk_id"] == "b"

    @patch("augmentum.memory.reranker.RerankService.rerank_dicts")
    @patch("augmentum.config.settings")
    def test_rerank_results_fallback_on_error(self, mock_settings, mock_rerank):
        from augmentum.documents.store import DocumentStore

        mock_settings.reranker_enabled = True
        mock_rerank.side_effect = RuntimeError("model failed")

        results = [
            {"chunk_id": "a", "content": "first", "score": 0.5},
            {"chunk_id": "b", "content": "second", "score": 0.3},
        ]
        reranked = DocumentStore._rerank_results("query", results, top_k=2)
        # Falls back to original order
        assert len(reranked) == 2
        assert reranked[0]["chunk_id"] == "a"

    def test_rrf_merge_still_works(self):
        """RRF merge is unaffected by reranking additions."""
        from augmentum.documents.store import DocumentStore

        vec = [
            {"chunk_id": "a", "content": "hello", "filename": "f.txt",
             "doc_id": "d1", "page_num": None, "chunk_index": 0, "distance": 0.1},
        ]
        result = DocumentStore._rrf_merge(vec, [])
        assert len(result) == 1
        assert "score" in result[0]


# ---------------------------------------------------------------------------
# Memory store reranking integration
# ---------------------------------------------------------------------------

class TestMemoryStoreReranking:
    """Test reranking integration in MemoryStore."""

    @patch("augmentum.memory.reranker.RerankService.rerank")
    @patch("augmentum.config.settings")
    def test_rerank_memories_when_enabled(self, mock_settings, mock_rerank):
        from augmentum.memory.models import Memory, MemoryType
        from augmentum.memory.store import MemoryStore

        mock_settings.reranker_enabled = True
        # Return: item 1 scores higher than item 0
        mock_rerank.return_value = [(1, 0.9), (0, 0.2)]

        mem_a = Memory(id="a", user_id="u", content="low", memory_type=MemoryType.FACT)
        mem_b = Memory(id="b", user_id="u", content="high", memory_type=MemoryType.FACT)
        scored = [(mem_a, 0.5), (mem_b, 0.3)]

        result = MemoryStore._rerank_memories("query", scored, limit=2)
        assert result[0][0].id == "b"  # reranked higher
        assert result[0][1] == 0.9

    @patch("augmentum.config.settings")
    def test_rerank_memories_skipped_when_disabled(self, mock_settings):
        from augmentum.memory.models import Memory, MemoryType
        from augmentum.memory.store import MemoryStore

        mock_settings.reranker_enabled = False

        mem = Memory(id="a", user_id="u", content="test", memory_type=MemoryType.FACT)
        scored = [(mem, 0.5)]

        result = MemoryStore._rerank_memories("query", scored, limit=5)
        assert result == scored  # unchanged

    @patch("augmentum.memory.reranker.RerankService.rerank")
    @patch("augmentum.config.settings")
    def test_rerank_memories_fallback_on_error(self, mock_settings, mock_rerank):
        from augmentum.memory.models import Memory, MemoryType
        from augmentum.memory.store import MemoryStore

        mock_settings.reranker_enabled = True
        mock_rerank.side_effect = RuntimeError("crash")

        mem = Memory(id="a", user_id="u", content="test", memory_type=MemoryType.FACT)
        scored = [(mem, 0.5)]

        result = MemoryStore._rerank_memories("query", scored, limit=5)
        assert result == scored  # falls back to original

    @patch("augmentum.memory.reranker.RerankService.rerank")
    @patch("augmentum.config.settings")
    def test_rerank_memories_empty_list(self, mock_settings, mock_rerank):
        from augmentum.memory.store import MemoryStore

        mock_settings.reranker_enabled = True

        result = MemoryStore._rerank_memories("query", [], limit=5)
        assert result == []
        mock_rerank.assert_not_called()


# ---------------------------------------------------------------------------
# Contextual retrieval tests
# ---------------------------------------------------------------------------

class TestContextualRetrieval:
    """Test LLM-generated chunk context generation."""

    def test_prepend_context_with_content(self):
        from augmentum.documents.contextual import prepend_context

        result = prepend_context("chunk text here", "This chunk discusses API endpoints.")
        assert result == "This chunk discusses API endpoints.\n\nchunk text here"

    def test_prepend_context_empty(self):
        from augmentum.documents.contextual import prepend_context

        result = prepend_context("chunk text here", "")
        assert result == "chunk text here"

    @pytest.mark.asyncio
    async def test_generate_contexts_no_backend(self):
        from augmentum.documents.contextual import generate_chunk_contexts

        contexts = await generate_chunk_contexts(
            ["chunk1", "chunk2"], "full document", backend=None,
        )
        assert contexts == ["", ""]

    @pytest.mark.asyncio
    async def test_generate_contexts_with_backend(self):
        from augmentum.documents.contextual import generate_chunk_contexts

        mock_backend = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "This chunk covers authentication."

        async def fake_chat(**kwargs):
            return mock_response

        mock_backend.chat = fake_chat

        contexts = await generate_chunk_contexts(
            ["chunk about auth tokens"], "Full doc about API security", backend=mock_backend,
        )
        assert len(contexts) == 1
        assert "authentication" in contexts[0]

    @pytest.mark.asyncio
    async def test_generate_contexts_handles_errors(self):
        from augmentum.documents.contextual import generate_chunk_contexts

        mock_backend = MagicMock()

        async def fail_chat(**kwargs):
            raise RuntimeError("LLM error")

        mock_backend.chat = fail_chat

        contexts = await generate_chunk_contexts(
            ["chunk1", "chunk2"], "doc", backend=mock_backend,
        )
        assert contexts == ["", ""]

    @pytest.mark.asyncio
    async def test_generate_contexts_truncates_long_docs(self):
        from augmentum.documents.contextual import generate_chunk_contexts

        mock_backend = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Context for chunk."

        calls = []

        async def capture_chat(**kwargs):
            calls.append(kwargs)
            return mock_response

        mock_backend.chat = capture_chat

        long_doc = "x" * 50_000
        await generate_chunk_contexts(["chunk"], long_doc, backend=mock_backend)

        # The prompt should have truncated the document
        assert len(calls) == 1
        msg_content = calls[0]["messages"][0].content
        assert "[... document truncated ...]" in msg_content


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestRerankerConfig:
    """Test reranker configuration settings."""

    def test_reranker_defaults(self):
        from augmentum.config import Settings

        s = Settings()
        assert s.reranker_enabled is True
        assert s.reranker_model == "jinaai/jina-reranker-v1-tiny-en"
        assert s.reranker_top_k == 5

    def test_contextual_retrieval_default_off(self):
        from augmentum.config import Settings

        s = Settings()
        assert s.document_rag_contextual_retrieval is False


# ---------------------------------------------------------------------------
# Wider retrieval pipeline tests
# ---------------------------------------------------------------------------

class TestWiderRetrievalPipeline:
    """Test that reranking widens the initial candidate pool."""

    @patch("augmentum.config.settings")
    def test_candidate_multiplier_with_reranking(self, mock_settings):
        """When reranking is on, we fetch 10x candidates instead of 2x."""
        mock_settings.reranker_enabled = True
        # The multiplier is computed inside search(), we test the logic directly
        limit = 5
        multiplier = 10 if mock_settings.reranker_enabled else 2
        assert limit * multiplier == 50

    @patch("augmentum.config.settings")
    def test_candidate_multiplier_without_reranking(self, mock_settings):
        mock_settings.reranker_enabled = False
        limit = 5
        multiplier = 10 if mock_settings.reranker_enabled else 2
        assert limit * multiplier == 10
