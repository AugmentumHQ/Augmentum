"""Lazy-loaded embedding service via FastEmbed (ONNX runtime).

Defaults to CPU because the shipped nomic-embed-text-v1.5-Q model is
INT8-quantized and ORT's CUDA EP can't keep it on-device (~156 memcpy
nodes per session = GPU↔CPU shuttling per inference). Operators with
VRAM to spare can flip ``embedding_use_gpu`` True or set
``ORT_PROVIDERS=CUDAExecutionProvider``; the env var still wins.
"""

from __future__ import annotations

import asyncio
import shutil
import struct
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from fastembed import TextEmbedding

log = get_logger(__name__)

# Sentinels for lazy init lifecycle. _UNLOADED = never tried;
# _LOAD_FAILED = tried at least once and failed terminally (e.g. cache
# corruption that survived the auto-wipe-and-retry path). When set,
# get_model() raises ``EmbeddingUnavailable`` immediately instead of
# re-attempting the ONNX load on every call — that produced thousands
# of NO_SUCHFILE tracebacks per minute under load in 2026-05-23 incident.
_UNLOADED = object()
_LOAD_FAILED = object()

# Permission errors (EACCES on ~/.cache) are infra, not model corruption:
# a container recreate resets /home/augmentum to root-owned and a live
# chown fixes it without a restart. Latching _LOAD_FAILED for those forced
# a process restart even after the fix (2026-07-02 voice incident), so
# they get a timed backoff instead — the model stays _UNLOADED and the
# next call after the interval retries the load for real. The backoff
# still prevents the per-request traceback spam the sentinel was built
# for (2026-05-23).
_PERM_RETRY_INTERVAL_S = 60.0


def _is_permission_error(exc: Exception) -> bool:
    return (
        isinstance(exc, PermissionError)
        or "Permission denied" in str(exc)
        or "EACCES" in str(exc)
    )

# Nomic requires task-type prefixes for optimal performance.
_QUERY_PREFIX = "search_query: "
_DOCUMENT_PREFIX = "search_document: "


class EmbeddingUnavailable(RuntimeError):
    """Raised when the embedding model cannot be loaded (broken cache,
    missing weights, ONNX runtime error). Distinct from generic
    RuntimeError so callers can decide whether the operation is
    optional (skip the embedding step, continue) or load-bearing
    (propagate / abort).

    Subclasses RuntimeError so existing ``except Exception`` paths
    catch it transparently. New code should ``except EmbeddingUnavailable``
    to distinguish "embedder is broken on this node" from other failures.
    """


class EmbeddingService:
    """Lazy-loaded CPU-local embeddings using nomic-embed-text-v1.5 (768-dim).

    Model loads on first use (~130MB download, ~2ms per query after warm-up).
    Thread-safe for read-only inference.

    Nomic uses task prefixes: ``search_query:`` for queries and
    ``search_document:`` for stored content.  The public methods handle
    this transparently — callers pass raw text.
    """

    _model: TextEmbedding | object = _UNLOADED
    _lock = threading.Lock()
    # monotonic deadline before which a permission-failed load won't retry
    _retry_at: float = 0.0
    MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5-Q"
    DIMENSION = 768
    # Hard cap on per-sequence tokens passed to the embedder. Nomic's model
    # supports 8192, but fastembed pads each batch to its longest member —
    # one 8192-token outlier in a code-heavy corpus blows attention scratch
    # to ~100 GB at modest batch sizes (batch * 12 heads * seq² * 4 bytes).
    # 1024 is generous for short-doc retrieval (embedding quality saturates
    # well below 512 tokens) and keeps worst-case attention bounded.
    MAX_SEQ_TOKENS = 1024

    @classmethod
    def get_model(cls) -> TextEmbedding:
        """Lazy-load the embedding model on first call (thread-safe).

        Raises ``EmbeddingUnavailable`` if the model can't be loaded
        after the auto-wipe-and-retry path. Callers that treat
        embedding as optional context (memory recall, knowledge pack
        retrieval) should catch this and skip the embedding step;
        callers that hard-require embedding (vector indexing) should
        propagate.
        """
        if cls._model is _LOAD_FAILED:
            raise EmbeddingUnavailable(
                f"embedding model {cls.MODEL_NAME!r} failed to load; "
                f"see earlier embedding_model_load_failed log for details"
            )
        if cls._model is not _UNLOADED:
            return cls._model  # type: ignore[return-value]
        if time.monotonic() < cls._retry_at:
            raise EmbeddingUnavailable(
                f"embedding model {cls.MODEL_NAME!r} load hit a permission "
                f"error; retrying after backoff — check ownership of "
                f"/home/augmentum"
            )
        with cls._lock:
            # Re-check both sentinels inside the lock — another thread
            # may have completed (or failed) the load between our
            # outer check and lock acquisition.
            if cls._model is _LOAD_FAILED:
                raise EmbeddingUnavailable(
                    f"embedding model {cls.MODEL_NAME!r} failed to load; "
                    f"see earlier embedding_model_load_failed log for details"
                )
            if cls._model is not _UNLOADED:
                return cls._model  # type: ignore[return-value]
            from fastembed import TextEmbedding

            log.info("embedding_model_loading", model=cls.MODEL_NAME)
            # Provider selection: env var wins (convert_worker still uses
            # it to avoid GPU OOM when embedding 1000+ knowledge-pack
            # chunks in a subprocess). Otherwise the ``embedding_use_gpu``
            # setting decides — default False because ORT inserts ~156
            # memcpy nodes for the INT8 Q model, making single-query
            # inference slower on GPU than CPU and burning ~500 MiB VRAM.
            import os

            from augmentum.config import settings as _settings
            _env_override = os.environ.get("ORT_PROVIDERS")
            _extra_kwargs: dict = {}
            if _env_override:
                _extra_kwargs["providers"] = [_env_override]
                log.info("embedding_provider_override", source="env", providers=_extra_kwargs["providers"])
            elif not getattr(_settings, "embedding_use_gpu", False):
                _extra_kwargs["providers"] = ["CPUExecutionProvider"]
                log.info("embedding_provider_override", source="setting", providers=["CPUExecutionProvider"])
            try:
                cls._model = TextEmbedding(cls.MODEL_NAME, **_extra_kwargs)
            except Exception as exc:
                if "NO_SUCHFILE" in str(exc) or "No such file" in str(exc):
                    # Cache corruption / partial download. Wipe matching
                    # snapshot directories and retry once with default
                    # kwargs. If THAT also fails, set the load-failed
                    # sentinel so subsequent get_model() calls bail
                    # immediately — otherwise every chat that touches
                    # the embedder would re-do the full download attempt
                    # cycle and spew tracebacks (production: 2026-05-23).
                    cache_dir = Path.home() / ".cache" / "fastembed"
                    model_slug = cls.MODEL_NAME.replace("/", "--")
                    for d in cache_dir.glob(f"models--{model_slug}*"):
                        log.warning("embedding_clearing_corrupt_cache", path=str(d))
                        shutil.rmtree(d, ignore_errors=True)
                    try:
                        # Preserve provider override on retry — otherwise a
                        # CPU-only deployment that hit cache corruption would
                        # silently relaunch on GPU and burn VRAM.
                        cls._model = TextEmbedding(cls.MODEL_NAME, **_extra_kwargs)
                    except Exception as retry_exc:
                        if _is_permission_error(retry_exc):
                            cls._raise_permission_backoff(retry_exc)
                        cls._model = _LOAD_FAILED
                        log.error(
                            "embedding_model_load_failed",
                            model=cls.MODEL_NAME,
                            error=str(retry_exc)[:300],
                            note="auto-wipe-and-retry also failed; embedder marked unavailable",
                        )
                        raise EmbeddingUnavailable(
                            f"embedding model {cls.MODEL_NAME!r} failed to load "
                            f"even after cache wipe: {retry_exc}"
                        ) from retry_exc
                else:
                    if _is_permission_error(exc):
                        cls._raise_permission_backoff(exc)
                    cls._model = _LOAD_FAILED
                    log.error(
                        "embedding_model_load_failed",
                        model=cls.MODEL_NAME,
                        error=str(exc)[:300],
                    )
                    raise EmbeddingUnavailable(
                        f"embedding model {cls.MODEL_NAME!r} failed to load: {exc}"
                    ) from exc
            cls._apply_seq_cap(cls._model)
            log.info("embedding_model_loaded", model=cls.MODEL_NAME)
        return cls._model  # type: ignore[return-value]

    @classmethod
    def _raise_permission_backoff(cls, exc: Exception) -> None:
        """EACCES-style failures stay retryable: leave ``_UNLOADED``, arm
        the backoff window, and raise. A live ``chown`` of /home/augmentum
        then heals the embedder on the next call — no restart needed."""
        cls._retry_at = time.monotonic() + _PERM_RETRY_INTERVAL_S
        log.error(
            "embedding_model_load_failed",
            model=cls.MODEL_NAME,
            error=str(exc)[:300],
            note=(
                "permission error — treated as transient infra; "
                f"will retry after {int(_PERM_RETRY_INTERVAL_S)}s. "
                "Fix: chown augmentum:augmentum /home/augmentum && chmod 755"
            ),
        )
        raise EmbeddingUnavailable(
            f"embedding model {cls.MODEL_NAME!r} blocked by a filesystem "
            f"permission error ({exc}); retrying after backoff"
        ) from exc

    @classmethod
    def _apply_seq_cap(cls, model: object) -> None:
        """Clamp the underlying tokenizer's truncation to MAX_SEQ_TOKENS.

        fastembed's PooledEmbedding wraps a HF tokenizer at ``model.model.
        tokenizer``. Calling ``enable_truncation`` overrides the model's
        default 8192 cap, bounding the worst-case attention scratch. Best-
        effort: if the wrapped layout changes in a future fastembed release,
        we log and continue with the default — the embedder still works,
        just without the bound.
        """
        try:
            inner = getattr(model, "model", None)
            tokenizer = getattr(inner, "tokenizer", None)
            if tokenizer is None:
                log.warning("embedding_seq_cap_skipped", reason="no tokenizer")
                return
            tokenizer.enable_truncation(max_length=cls.MAX_SEQ_TOKENS)
        except Exception as exc:
            log.warning("embedding_seq_cap_failed", error=str(exc))

    # ------------------------------------------------------------------
    # Document embeddings (for storage / indexing)
    # ------------------------------------------------------------------

    @classmethod
    def embed(cls, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts as *documents*. Use for storing content."""
        model = cls.get_model()
        prefixed = [f"{_DOCUMENT_PREFIX}{t}" for t in texts]
        return [e.tolist() for e in model.embed(prefixed)]

    @classmethod
    def embed_one(cls, text: str) -> list[float]:
        """Embed a single text string as a *document*."""
        return cls.embed([text])[0]

    @classmethod
    async def aembed_one(cls, text: str) -> list[float]:
        """Async wrapper for :meth:`embed_one`.

        ``embed_one`` runs synchronous ONNX inference and can block the
        event loop for hundreds of ms on cold model load. This is the
        ONLY sanctioned way to embed from inside an ``async def`` — the
        async_blocking scanner flags any direct ``embed_one(`` call made
        in an async body (audit 2026-06-17).
        """
        return await asyncio.to_thread(cls.embed_one, text)

    # ------------------------------------------------------------------
    # Query embeddings (for search / retrieval)
    # ------------------------------------------------------------------

    @classmethod
    def embed_query(cls, text: str) -> list[float]:
        """Embed a single text string as a *query*."""
        model = cls.get_model()
        prefixed = f"{_QUERY_PREFIX}{text}"
        return list(model.embed([prefixed]))[0].tolist()

    @classmethod
    async def aembed_query(cls, text: str) -> list[float]:
        """Async wrapper for :meth:`embed_query` — see :meth:`aembed_one`."""
        return await asyncio.to_thread(cls.embed_query, text)

    @classmethod
    def embed_queries(cls, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts as *queries*."""
        model = cls.get_model()
        prefixed = [f"{_QUERY_PREFIX}{t}" for t in texts]
        return [e.tolist() for e in model.embed(prefixed)]

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------

    @classmethod
    def to_blob(cls, vec: list[float]) -> bytes:
        """Pack a float vector into a little-endian float32 blob for sqlite-vec."""
        return struct.pack(f"<{len(vec)}f", *vec)

    @classmethod
    def from_blob(cls, blob: bytes) -> list[float]:
        """Unpack a sqlite-vec blob back to a float vector."""
        count = len(blob) // 4
        return list(struct.unpack(f"<{count}f", blob))

    @classmethod
    def reset(cls) -> None:
        """Reset the model (for testing)."""
        cls._model = _UNLOADED
