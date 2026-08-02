"""Honest-gap mechanism.

Three stages keep Becca from confabulating beyond what she *actually*
knows:

1. **Deterministic summary** — what does memory return for this query?
   Bullet-list of facts, no model in the loop.
2. **LLM rephrase** — turn the bullets into natural voice. The LLM is
   allowed creativity in *register* and *flow*, not in *content*.
3. **Output lint** — compare the rephrase to the summary on the
   factual span. If the rephrase's embedding drifts ≥
   ``LINT_EMBEDDING_THRESHOLD`` (0.20) from the summary's, reject and
   retry up to ``MAX_RETRIES``. After exhausting retries, return the
   deterministic summary verbatim.

This is the mechanical floor under the honest-gap commitment — the
LLM is permitted to be warm but not to introduce facts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.companion_runtime.runtime import CompanionRuntime

log = get_logger(__name__)


LINT_EMBEDDING_THRESHOLD: float = 0.20
MAX_RETRIES: int = 2


@dataclass(slots=True)
class HonestResult:
    """The output of an honest-gap pass."""
    content: str
    used_fallback: bool          # True if we returned the deterministic summary verbatim
    retries: int = 0
    drift: float = 0.0
    sources: list[str] = field(default_factory=list)


async def _deterministic_summary(
    runtime: CompanionRuntime, query: str, *, k: int = 5,
) -> tuple[str, list[str]]:
    """Bullet-list of what we know. Returns (summary_text, source_ids)."""
    if runtime.memory._store is None:  # noqa: SLF001
        return ("(no memory store attached)", [])
    try:
        # CompanionMemory.recall is user-scoped: user_id is required and the
        # cap kwarg is `k`, not `limit`. Without the owner's id this would
        # raise (and, pre-fix, also reject `limit=`) — harmless only because
        # this path is currently unwired, but a landmine once it's called.
        hits = await runtime.memory.recall(
            query=query, user_id=runtime.owner_user_id, k=k,
        )
    except Exception as exc:
        log.warning("honest_gap_recall_failed", error=str(exc))
        return ("(memory recall failed)", [])

    bullets: list[str] = []
    sources: list[str] = []
    for h in hits or []:
        text = getattr(h, "content", None) or getattr(h, "text", None) or str(h)
        text = text.strip()
        if not text:
            continue
        bullets.append(f"- {text[:240]}")
        mid = getattr(h, "id", None)
        if mid is not None:
            sources.append(str(mid))
    if not bullets:
        return ("(no relevant memories found)", [])
    return ("\n".join(bullets), sources)


async def _rephrase(
    runtime: CompanionRuntime, query: str, summary: str,
) -> str:
    """LLM-driven natural-voice rephrase of the bullets. On any failure
    returns ``""`` — caller falls back to the summary."""
    try:
        from augmentum.companion_runtime import tiers
        from augmentum.models.base import InternalChatRequest
    except Exception:
        return ""
    try:
        backend, model_name = await tiers.utility(runtime)
    except Exception:
        return ""
    if not hasattr(backend, "chat"):
        return ""

    prompt = (
        "Rephrase the following bullet-pointed facts into one warm, "
        "natural answer. Do NOT add any facts that are not in the "
        "bullets. If a fact isn't there, say you don't know.\n\n"
        f"Question: {query}\n\nFacts:\n{summary}\n\nAnswer:"
    )
    req = InternalChatRequest(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=300,
    )
    try:
        resp = await backend.chat(req)
    except Exception as exc:
        log.warning("honest_gap_rephrase_failed", error=str(exc))
        return ""
    from augmentum.models.base import response_text
    return response_text(resp)


async def _embedding_drift(text_a: str, text_b: str) -> float:
    """Cosine distance between the two strings' embeddings. Returns
    ``0.0`` (no drift) when the embedding service is unavailable, so
    a missing model degrades to "always accept" rather than "always
    reject" — better to ship an unverified rephrase than refuse to
    speak at all."""
    if not text_a or not text_b:
        return 0.0
    try:
        import asyncio
        from augmentum.memory.embeddings import EmbeddingService
        emb_a = await asyncio.to_thread(EmbeddingService.embed_one, text_a)
        emb_b = await asyncio.to_thread(EmbeddingService.embed_one, text_b)
    except Exception:
        return 0.0
    if not emb_a or not emb_b or len(emb_a) != len(emb_b):
        return 0.0
    dot = sum(x * y for x, y in zip(emb_a, emb_b))
    na = sum(x * x for x in emb_a) ** 0.5
    nb = sum(x * x for x in emb_b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    cosine = max(-1.0, min(1.0, dot / (na * nb)))
    return 1.0 - cosine


async def answer(runtime: CompanionRuntime, query: str) -> HonestResult:
    """Return Becca's best honest answer to ``query``.

    Used by Sprint 4a's reach-out path and any future "answer the
    owner" flow that needs the mechanical floor under the honest-gap
    commitment.
    """
    summary, sources = await _deterministic_summary(runtime, query)

    last_rephrase = ""
    last_drift = 0.0
    for retry in range(MAX_RETRIES + 1):
        rephrase = await _rephrase(runtime, query, summary)
        if not rephrase:
            # Rephrase service unavailable — return summary verbatim
            return HonestResult(
                content=summary, used_fallback=True, retries=retry,
                drift=0.0, sources=sources,
            )
        drift = await _embedding_drift(summary, rephrase)
        last_rephrase, last_drift = rephrase, drift
        if drift < LINT_EMBEDDING_THRESHOLD:
            return HonestResult(
                content=rephrase, used_fallback=False, retries=retry,
                drift=drift, sources=sources,
            )
        log.info(
            "honest_gap_lint_rejected",
            drift=round(drift, 3), threshold=LINT_EMBEDDING_THRESHOLD,
            retry=retry,
        )

    log.warning(
        "honest_gap_fell_back_to_summary",
        drift=round(last_drift, 3), retries=MAX_RETRIES + 1,
    )
    return HonestResult(
        content=summary, used_fallback=True,
        retries=MAX_RETRIES + 1, drift=last_drift, sources=sources,
    )


__all__ = ["HonestResult", "LINT_EMBEDDING_THRESHOLD", "MAX_RETRIES", "answer"]
