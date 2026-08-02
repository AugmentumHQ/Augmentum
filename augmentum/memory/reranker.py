"""Cross-encoder reranking service via FastEmbed ONNX runtime.

Reranks retrieval results using a cross-encoder model that scores
query-document pairs for semantic relevance. Plugs into both memory
recall and document search after the initial RRF merge stage.

Architecture mirrors EmbeddingService: lazy-loaded singleton, defaults to
CPU (see ``embedding_use_gpu`` in config for the rationale and the
operator escape hatch).
"""

from __future__ import annotations

import shutil
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from fastembed.rerank.cross_encoder import TextCrossEncoder

log = get_logger(__name__)

_UNLOADED = object()


class RerankService:
    """Lazy-loaded cross-encoder reranker using FastEmbed ONNX models.

    Model loads on first use (~130MB download). Thread-safe for inference.

    Supported models (via fastembed.rerank.cross_encoder):
    - Xenova/ms-marco-MiniLM-L-6-v2 (80MB, fast)
    - jinaai/jina-reranker-v1-tiny-en (130MB, good balance)
    - BAAI/bge-reranker-base (1GB, highest quality)
    """

    _model: TextCrossEncoder | object = _UNLOADED
    _model_name: str = ""
    _lock = threading.Lock()

    @classmethod
    def get_model(cls, model_name: str = "") -> TextCrossEncoder:
        """Lazy-load the reranker model on first call (thread-safe)."""
        from augmentum.config import settings

        target = model_name or settings.reranker_model
        if cls._model is not _UNLOADED and target == cls._model_name:
            return cls._model  # type: ignore[return-value]
        with cls._lock:
            if cls._model is not _UNLOADED and target == cls._model_name:
                return cls._model  # type: ignore[return-value]
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            log.info("reranker_model_loading", model=target)
            # Provider selection mirrors EmbeddingService — env var wins,
            # then ``embedding_use_gpu`` setting (the toggle covers both
            # services because they share a CUDA-context cost). CPU default
            # frees ~165 MiB of VRAM that bge-reranker-base would otherwise
            # hold permanently.
            import os
            _env_override = os.environ.get("ORT_PROVIDERS")
            _extra_kwargs: dict = {}
            if _env_override:
                _extra_kwargs["providers"] = [_env_override]
            elif not getattr(settings, "embedding_use_gpu", False):
                _extra_kwargs["providers"] = ["CPUExecutionProvider"]
            try:
                cls._model = TextCrossEncoder(model_name=target, **_extra_kwargs)
            except Exception as exc:
                if "NO_SUCHFILE" in str(exc) or "No such file" in str(exc):
                    # Corrupt/partial cache — clear and retry once, keeping
                    # the provider override so a CPU-only box doesn't
                    # silently relaunch on GPU after a cache wipe.
                    cache_dir = Path.home() / ".cache" / "fastembed"
                    model_slug = target.replace("/", "--")
                    for d in cache_dir.glob(f"models--{model_slug}*"):
                        log.warning("reranker_clearing_corrupt_cache", path=str(d))
                        shutil.rmtree(d, ignore_errors=True)
                    cls._model = TextCrossEncoder(model_name=target, **_extra_kwargs)
                else:
                    raise
            cls._model_name = target
            log.info("reranker_model_loaded", model=target, providers=_extra_kwargs.get("providers", "default"))
        return cls._model  # type: ignore[return-value]

    @classmethod
    def rerank(
        cls,
        query: str,
        documents: list[str],
        top_k: int | None = None,
    ) -> list[tuple[int, float]]:
        """Rerank documents against a query.

        Returns list of (original_index, score) tuples sorted by score
        descending. Scores are in [0, 1] range where higher = more relevant.

        If top_k is provided, only the top_k results are returned.
        """
        if not documents:
            return []

        model = cls.get_model()
        scores = list(model.rerank(query, documents, batch_size=64))

        # scores is a list of floats in the same order as documents
        # Normalize to [0, 1] — some cross-encoders (e.g., Jina) return
        # raw logits that can exceed 1.0. Use sigmoid for normalization.
        import math
        normalized = [1.0 / (1.0 + math.exp(-max(-500.0, min(500.0, s)))) for s in scores]
        indexed: list[tuple[int, float]] = list(enumerate(normalized))
        indexed.sort(key=lambda x: x[1], reverse=True)

        if top_k is not None:
            indexed = indexed[:top_k]

        return indexed

    @classmethod
    def rerank_dicts(
        cls,
        query: str,
        results: list[dict],
        content_key: str = "content",
        top_k: int | None = None,
    ) -> list[dict]:
        """Rerank a list of result dicts by their content field.

        Returns reranked results with 'reranker_score' added and
        'score' replaced by the reranker score.
        """
        if not results:
            return []

        documents = [r.get(content_key, "") for r in results]
        ranked = cls.rerank(query, documents, top_k=top_k)

        reranked: list[dict] = []
        for orig_idx, score in ranked:
            item = {**results[orig_idx], "reranker_score": score, "score": score}
            reranked.append(item)

        return reranked

    @classmethod
    def reset(cls) -> None:
        """Reset the model (for testing)."""
        cls._model = _UNLOADED
        cls._model_name = ""
