"""Reference Resolver core — RRF-merged hybrid retrieval.

Domain-agnostic. The resolver merges ranked results from heterogeneous
sources (files, journal entries) without baking in any assumption
about content type, language, or domain. Whatever the user has
indexed — documents, media, captioned images, autobiographical
moments — is reachable through the same NL query path.

Four retrieval legs run in parallel via :func:`asyncio.gather`:

* **file_index vec** — KNN over file_index_vec (mig 175)
* **file_index FTS5** — BM25 over file_index_fts (mig 074/097)
* **companion_journal vec** — KNN over companion_journal_vec (mig 177)
* **companion_journal FTS5** — BM25 over companion_journal_fts (mig 177)

Reciprocal Rank Fusion (RRF, k=60) merges the four ranked lists into
one. The k=60 constant matches the standard literature value used by
the existing PackManager and is intentionally not tuned to a specific
corpus — RRF's value is exactly that it's stable across heterogeneous
result distributions.

Cross-encoder rerank is reserved for a future iteration. The pure-RRF
path is fast enough (sub-100ms on a warm system) and tight enough
that an immediate rerank would over-fit small queries.

Failure modes are designed for graceful degradation:

* Embedding compute failure → skip vec legs, return FTS-only results
* vec0 extension missing → that leg returns empty, RRF still merges
* FTS5 query empty after sanitisation → skip FTS legs, return vec-only
* No matches anywhere → empty list, never raises

The shape of returned :class:`Moment` objects is the **contract**
the disambiguation card and active-items binding (Pieces 7+8) bind
to. Renaming any field is a breaking change.
"""

from __future__ import annotations

import asyncio
import struct
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from augmentum.memory.embeddings import EmbeddingService
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.companion_runtime.memory import CompanionMemory
    from augmentum.vfs.index import FileIndexService

log = get_logger(__name__)


# RRF constant — k=60 is the canonical value from Cormack et al.
_RRF_K = 60

# Snippet length cap. Long content is truncated for the disambiguation
# card so it fits in a single readable row.
_SNIPPET_MAX = 240


@dataclass
class Moment:
    """A single resolver result. Field set is the public contract for
    disambiguation UI (Piece 8) and active-items binding (Piece 7).
    """

    id: str
    kind: str  # 'file' | 'journal' (more kinds in future pieces)
    score: float
    snippet: str
    title: str = ""
    created_at: str = ""
    content_refs: list[dict] = field(default_factory=list)
    legs: list[str] = field(default_factory=list)  # which legs surfaced this
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "score": self.score,
            "snippet": self.snippet,
            "title": self.title,
            "created_at": self.created_at,
            "content_refs": self.content_refs,
            "legs": self.legs,
        }


def _encode_embedding(emb: list[float]) -> bytes:
    """Encode a 768-dim float vector as raw little-endian float32 bytes
    — the wire format vec0 expects."""
    return struct.pack(f"{len(emb)}f", *emb)


def _snippet(content: str, max_len: int = _SNIPPET_MAX) -> str:
    """Truncate content to a renderable snippet. Strips newlines so
    rendered cards stay one-line."""
    s = " ".join((content or "").split())
    if len(s) > max_len:
        return s[: max_len - 1].rstrip() + "…"
    return s


async def _embed_query(query: str) -> bytes | None:
    """Compute the query embedding, return None on failure.

    Run in a thread because EmbeddingService is sync. Doesn't block
    the event loop on the ONNX inference call.
    """
    if not query.strip():
        return None
    try:
        emb = await asyncio.to_thread(EmbeddingService.embed_query, query)
        return _encode_embedding(emb)
    except Exception as exc:
        log.warning("resolver_embed_failed", error=str(exc)[:200])
        return None


def _rrf_merge(leg_results: dict[str, list[tuple[str, dict]]]) -> list[Moment]:
    """Merge ranked leg results via Reciprocal Rank Fusion.

    ``leg_results`` maps leg name → ranked list of (composite_id, data)
    where composite_id is ``{kind}:{id}`` so file id 42 and journal id
    42 don't collide.

    Returns Moments sorted by descending RRF score. The score field
    on the Moment is the RRF sum; not directly comparable to leg-level
    similarities but stable for ranking.
    """
    accum: dict[str, dict[str, Any]] = {}
    for leg_name, ranked in leg_results.items():
        for rank, (composite_id, data) in enumerate(ranked, start=1):
            slot = accum.setdefault(composite_id, {
                "data": data,
                "rrf": 0.0,
                "legs": [],
            })
            slot["rrf"] += 1.0 / (_RRF_K + rank)
            slot["legs"].append(leg_name)

    moments: list[Moment] = []
    for composite_id, slot in accum.items():
        data = slot["data"]
        kind = composite_id.split(":", 1)[0]
        if kind == "file":
            moments.append(Moment(
                id=str(data.get("id") or ""),
                kind="file",
                score=float(slot["rrf"]),
                snippet=_snippet(
                    data.get("description")
                    or data.get("name")
                    or ""
                ),
                title=str(data.get("name") or ""),
                created_at=str(data.get("created_at") or ""),
                content_refs=[],
                legs=slot["legs"],
                raw=data,
            ))
        elif kind == "journal":
            moments.append(Moment(
                id=str(data.get("id") or ""),
                kind="journal",
                score=float(slot["rrf"]),
                snippet=_snippet(data.get("content") or ""),
                title=str(data.get("entry_type") or "journal"),
                created_at=str(data.get("created_at") or ""),
                content_refs=list(data.get("content_refs") or []),
                legs=slot["legs"],
                raw=data,
            ))
    moments.sort(key=lambda m: m.score, reverse=True)
    return moments


async def _file_vec_leg(
    file_index: FileIndexService | None,
    embedding: bytes | None,
    *,
    user_id: str,
    per_leg_limit: int,
) -> list[tuple[str, dict]]:
    if file_index is None or embedding is None:
        return []
    try:
        entries = await file_index.search_by_embedding(
            embedding, user_id=user_id, limit=per_leg_limit,
        )
    except Exception as exc:
        log.warning("resolver_file_vec_failed", error=str(exc)[:200])
        return []
    out: list[tuple[str, dict]] = []
    for e in entries:
        d = e.__dict__ if hasattr(e, "__dict__") else dict(e)
        eid = d.get("id") or ""
        if not eid:
            continue
        out.append((f"file:{eid}", d))
    return out


async def _file_fts_leg(
    file_index: FileIndexService | None,
    query: str,
    *,
    user_id: str,
    per_leg_limit: int,
) -> list[tuple[str, dict]]:
    if file_index is None or not query.strip():
        return []
    try:
        entries = await file_index.search(
            query, user_id=user_id, limit=per_leg_limit,
        )
    except Exception as exc:
        log.warning("resolver_file_fts_failed", error=str(exc)[:200])
        return []
    out: list[tuple[str, dict]] = []
    for e in entries:
        d = e.__dict__ if hasattr(e, "__dict__") else dict(e)
        eid = d.get("id") or ""
        if not eid:
            continue
        out.append((f"file:{eid}", d))
    return out


async def _journal_vec_leg(
    memory: CompanionMemory | None,
    embedding: bytes | None,
    *,
    user_id: str,
    per_leg_limit: int,
) -> list[tuple[str, dict]]:
    if memory is None or embedding is None:
        return []
    try:
        rows = await memory.search_journal_by_embedding(
            embedding, user_id=user_id, limit=per_leg_limit,
        )
    except Exception as exc:
        log.warning("resolver_journal_vec_failed", error=str(exc)[:200])
        return []
    return [(f"journal:{r['id']}", r) for r in rows if r.get("id")]


async def _journal_fts_leg(
    memory: CompanionMemory | None,
    query: str,
    *,
    user_id: str,
    per_leg_limit: int,
) -> list[tuple[str, dict]]:
    if memory is None or not query.strip():
        return []
    try:
        rows = await memory.search_journal_fts(
            query, user_id=user_id, limit=per_leg_limit,
        )
    except Exception as exc:
        log.warning("resolver_journal_fts_failed", error=str(exc)[:200])
        return []
    return [(f"journal:{r['id']}", r) for r in rows if r.get("id")]


async def resolve_moments(
    query: str,
    *,
    user_id: str,
    file_index: FileIndexService | None = None,
    memory: CompanionMemory | None = None,
    limit: int = 10,
    kinds: tuple[str, ...] = ("file", "journal"),
) -> list[Moment]:
    """Resolve a natural-language reference to ranked moments.

    Domain-agnostic: the user describes something, the resolver returns
    ranked candidates regardless of whether the underlying content is
    a document, image, media file, or journal moment. The retrieval
    legs do not branch on content type — only on source (file_index
    vs companion_journal). Higher layers decide what to do with the
    resolved item.

    ``kinds`` selects which retrieval legs run. Passing only ``("file",)``
    skips the journal legs; passing only ``("journal",)`` skips the
    file legs. Default runs both.

    Returns moments sorted by descending RRF score. Empty list on
    no matches or total retrieval failure — never raises.
    """
    if not query.strip():
        return []

    # Per-leg over-fetch: each leg returns more than the final limit
    # so RRF has room to reorder. 3× is the default RRF over-fetch
    # multiplier; doesn't blow up DB time because each leg is already
    # paginated at the SQL level.
    per_leg = max(limit * 3, 20)

    # Compute the query embedding once for the vec legs to share.
    # Failure here gracefully degrades to FTS-only.
    embedding = await _embed_query(query) if "file" in kinds or "journal" in kinds else None

    coros: dict[str, Any] = {}
    if "file" in kinds:
        coros["file_vec"] = _file_vec_leg(
            file_index, embedding, user_id=user_id, per_leg_limit=per_leg,
        )
        coros["file_fts"] = _file_fts_leg(
            file_index, query, user_id=user_id, per_leg_limit=per_leg,
        )
    if "journal" in kinds:
        coros["journal_vec"] = _journal_vec_leg(
            memory, embedding, user_id=user_id, per_leg_limit=per_leg,
        )
        coros["journal_fts"] = _journal_fts_leg(
            memory, query, user_id=user_id, per_leg_limit=per_leg,
        )

    if not coros:
        return []

    # Parallel fan-out — each leg uses its own SQL path so concurrent
    # access on the aiosqlite connection is just sequential under the
    # hood, but the embedding compute (the slow part) already happened
    # above and is shared between legs.
    names = list(coros.keys())
    results = await asyncio.gather(*coros.values(), return_exceptions=True)

    leg_results: dict[str, list[tuple[str, dict]]] = {}
    for name, result in zip(names, results):
        if isinstance(result, BaseException):
            log.warning(
                "resolver_leg_raised",
                leg=name, error=str(result)[:200],
            )
            leg_results[name] = []
        else:
            leg_results[name] = result

    merged = _rrf_merge(leg_results)
    return merged[:limit]
