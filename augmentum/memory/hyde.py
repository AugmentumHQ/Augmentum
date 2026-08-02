"""HyDE — Hypothetical Document Embedding for memory recall.

Short queries ("what was that thing I asked about gravity sims") embed
into vectors weaker than the memories they're trying to match (which are
declarative sentences with concrete nouns). HyDE asks a small LLM to
write a *hypothetical answer* to the query, embeds that, and uses both
vectors in the search. The hypothetical doesn't have to be factually
true — it just has to be the *shape* of an answer, which is what the
memory vectors actually look like.

Generic across user domains. The prompt asks the model to write what an
answer would look like; it doesn't impose vocabulary, structure, or
content priors specific to any user's interests.

Independent of:
- Phase 1 (cosine shadow-touch) — touches recall only, not ingest.
- Phase 2 (durability passthrough) — touches recall only, not ingest.
- Phase 3 (retroactive demotion) — touches recall, demotion path is
  orthogonal.

Gated by ``memory_hyde_enabled`` so it can ship dark, get evaluated on a
labeled query set, then flipped on per-user once the win is confirmed.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.models.base import ModelBackend

log = get_logger(__name__)


_HYDE_SYSTEM = """\
You are a query-expansion helper. Given a user's question or message, \
write a single short sentence that would PLAUSIBLY answer it — even if \
you don't know the actual answer. The goal is to produce text that \
matches the SHAPE of the answer, not the content.

Rules:
- One sentence, 8–25 words.
- Declarative shape. Not a question, not a list.
- If the query is too vague to imagine an answer, return the query unchanged.
- Don't add disclaimers, don't apologize, don't say "I think".

Examples:
Q: what was that thing I asked about gravity sims
A: The user was building a gravity-based browser game with a mobile-friendly viewport.

Q: my cat's name
A: The user has a pet cat whose name and birth date are recorded.

Q: where do I live
A: The user lives in a city, likely with a named state or country.

Q: hello
A: hello
"""


async def expand_query(
    query: str,
    backend: ModelBackend | None,
    model: str = "",
    *,
    max_chars: int = 400,
) -> str:
    """Produce a HyDE expansion for ``query``. Returns "" on any failure.

    Pure utility — the caller decides whether to feed the result into
    recall(hyde_text=...). Returns empty string (not the original query)
    on failure so the caller can distinguish "no expansion happened" from
    "expansion is identical to the query".
    """
    from augmentum.models.base import InternalChatRequest, Message

    if not query or not query.strip() or backend is None:
        return ""
    text = query.strip()
    if len(text) > max_chars:
        text = text[:max_chars]

    request = InternalChatRequest(
        model=model or "",
        messages=[
            Message(role="system", content=_HYDE_SYSTEM),
            Message(role="user", content=text),
        ],
        stream=False,
        temperature=0.3,
        max_tokens=80,
    )
    try:
        resp = await backend.chat(request)
    except Exception:
        log.debug("hyde_backend_failed", query=text[:60], exc_info=True)
        return ""
    out = ""
    try:
        out = (resp.message.content or "").strip()
    except Exception:
        return ""
    # Trim accidental markdown fences and quotes
    out = out.strip("`'\" ")
    if not out or out.lower() == text.lower():
        return ""
    # Guard against the model returning a multi-paragraph essay
    if len(out) > 400:
        out = out[:400]
    return out
