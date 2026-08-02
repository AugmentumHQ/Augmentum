"""Recall functions — the lookup layer for narrative-mode substrate.

The narrative engine has historically been inject-only: every turn folds
the full STATE snapshot + active plot threads + recent ledger into the
prompt as static text, regardless of whether the model needs that
specific state for the current turn. For SMALL local models with tight
context budgets this means a lot of the window is spent on stale-but-
re-injected state instead of fresh prose.

This module is the first half of closing that gap (audit 2026-05-31):
the lookup verbs. Each function queries the existing persistence layer
(entities / facts / plot_threads / archive_exchanges) and returns a
short structured text the model can paste into prose or the UI can
render in an inspector panel.

The *second* half — wiring these as LLM-callable tools so the model
can actually `recall_entity("Elena")` mid-turn instead of receiving
a full snapshot every turn — is a separate change requiring an
iterative tool-execution loop on top of NarrativeHandler. These
functions are deliberately decoupled from that wiring so the data
layer ships and gets exercised (via HTTP routes / tests / UI) before
the loop refactor lands.

Design contract
---------------

* **User-scoped.** Every recall passes ``user_id`` through to the
  persistence layer; cross-tenant reads are impossible by construction.
* **Empty result is not an error.** Missing entity / no facts matching
  / no thread by that id all return an ``RecallResult`` with empty
  ``items`` and a clear ``summary`` ("No entity matching X."). Callers
  never need a ``try/except`` for "not found."
* **Concise by default.** Each function takes an optional ``limit`` and
  caps output so the prose summary fits inside a small-model context
  budget (~300-600 tokens worst case for a wide recall).
* **Pure read.** No side effects. Recall does NOT advance the engine,
  modify the ledger, or persist anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from augmentum.state.narrative_persistence import NarrativePersistence
from augmentum.state.narrative_state import (
    Entity,
    EntityType,
    Fact,
    PlotStatus,
    PlotThread,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# Conservative caps — keep recall payloads tight enough that even a 7B
# model with a 4K-effective window can call multiple recalls in a turn
# without burning the prose budget.
_MAX_FACT_LIMIT = 10
_MAX_ARCHIVE_LIMIT = 5
_MAX_FACT_CONTENT_CHARS = 280
_MAX_ARCHIVE_CONTENT_CHARS = 400


@dataclass
class RecallResult:
    """Result of a single recall call.

    ``summary`` is the prose the model/UI consumes. ``items`` carries
    the structured data for callers that want to render their own view
    (e.g. an inspector panel listing each fact's domain + confidence).
    """

    summary: str
    items: list[dict[str, Any]] = field(default_factory=list)
    total_available: int = 0
    truncated: bool = False


# ---------------------------------------------------------------------------
# Entity recall
# ---------------------------------------------------------------------------


def _format_entity(entity: Entity) -> str:
    """One-paragraph human-readable entity card."""
    state = entity.state
    lines = [f"**{entity.name}** ({entity.entity_type.value})"]
    if entity.aliases:
        lines.append(f"Aliases: {', '.join(entity.aliases)}")
    if state.location:
        lines.append(f"Location: {state.location}")
    if state.emotional_state:
        lines.append(f"Emotional state: {state.emotional_state}")
    if state.physical_state:
        lines.append(f"Physical state: {state.physical_state}")
    if state.inventory:
        # Cap inventory list — long inventories are rarely all relevant.
        items = state.inventory[:8]
        more = len(state.inventory) - len(items)
        suffix = f" (+{more} more)" if more > 0 else ""
        lines.append(f"Inventory: {', '.join(items)}{suffix}")
    if state.relationships:
        rel_lines = [f"  - {who}: {note}" for who, note in list(state.relationships.items())[:8]]
        more = len(state.relationships) - len(rel_lines)
        if more > 0:
            rel_lines.append(f"  - (+{more} more relationships)")
        lines.append("Relationships:\n" + "\n".join(rel_lines))
    return "\n".join(lines)


async def recall_entity(
    persistence: NarrativePersistence,
    session_id: str,
    *,
    user_id: str,
    name: str,
) -> RecallResult:
    """Look up one entity by exact name or alias (case-insensitive).

    Returns the entity card if found; an empty RecallResult with a
    "not found" summary otherwise. Lookup is exact-or-alias — fuzzy
    matching is intentionally NOT included so the model gets a clean
    "not found" signal when its memory of a name is wrong (and can
    use ``recall_facts`` with the wrong-name fragment to find the
    correct spelling instead of getting a confident-but-wrong card).
    """
    needle = (name or "").strip()
    if not needle:
        return RecallResult(summary="recall_entity called without a name.")

    entities = await persistence._load_entities(session_id, user_id=user_id)
    needle_lc = needle.lower()
    for entity in entities.values():
        if entity.name.lower() == needle_lc:
            match = entity
            break
        if any(alias.lower() == needle_lc for alias in (entity.aliases or [])):
            match = entity
            break
    else:
        return RecallResult(
            summary=f"No entity matching '{needle}'. Known: {', '.join(sorted(e.name for e in entities.values())[:12])}.",
            total_available=len(entities),
        )

    return RecallResult(
        summary=_format_entity(match),
        items=[{
            "id": match.id,
            "name": match.name,
            "type": match.entity_type.value,
            "aliases": list(match.aliases or []),
            "state": match.state.to_dict(),
        }],
        total_available=1,
    )


async def list_entities(
    persistence: NarrativePersistence,
    session_id: str,
    *,
    user_id: str,
    entity_type: EntityType | None = None,
) -> RecallResult:
    """Enumerate entities (optionally filtered by type).

    Companion to ``recall_entity`` for cases where the model needs to
    discover what's in scope — e.g. "who is present?" / "what items
    have been introduced?" without committing to a specific name.
    """
    entities = list((await persistence._load_entities(session_id, user_id=user_id)).values())
    if entity_type is not None:
        entities = [e for e in entities if e.entity_type == entity_type]
    if not entities:
        label = entity_type.value if entity_type else "any type"
        return RecallResult(summary=f"No entities of {label} in this session.")

    entities.sort(key=lambda e: e.name.lower())
    lines = [f"{len(entities)} entities tracked:"]
    for e in entities[:24]:
        snippet = ""
        if e.state.location:
            snippet = f" @ {e.state.location}"
        elif e.state.emotional_state:
            snippet = f" ({e.state.emotional_state})"
        lines.append(f"- {e.name} [{e.entity_type.value}]{snippet}")
    truncated = len(entities) > 24
    if truncated:
        lines.append(f"…and {len(entities) - 24} more.")

    return RecallResult(
        summary="\n".join(lines),
        items=[
            {"id": e.id, "name": e.name, "type": e.entity_type.value}
            for e in entities
        ],
        total_available=len(entities),
        truncated=truncated,
    )


# ---------------------------------------------------------------------------
# Fact recall — substring + tag search over the fact store
# ---------------------------------------------------------------------------


# Common words that carry no retrieval signal — dropped from phrasal
# queries so "abilities of the dragon" keys on "abilities"/"dragon", not
# "of"/"the". Kept small: only true stopwords, never domain nouns.
_QUERY_STOPWORDS = frozenset({
    "a", "an", "and", "the", "of", "to", "in", "on", "at", "by", "for",
    "with", "is", "are", "was", "were", "be", "or", "as", "it", "its",
    "his", "her", "their", "s", "about", "any", "all",
})


def _query_terms(query: str) -> list[str]:
    """Tokenize a phrasal query into lowercase content terms.

    Splits on whitespace, lowercases, drops stopwords and 1-char tokens.
    Shared by the recall + lorebook search so multi-word queries match
    per-word (ANY, scored) instead of as one opaque substring.
    """
    out: list[str] = []
    seen: set[str] = set()
    for raw in query.split():
        t = raw.strip().lower().strip(".,!?;:\"'()[]")
        if len(t) < 2 or t in _QUERY_STOPWORDS or t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out


def _fact_match_score(fact: Fact, terms: list[str]) -> int:
    """Count of distinct query terms found in the fact (content or tags).

    ANY-term (scored), not ALL-term: a phrasal query like "Luna spirit
    dragon bond abilities" must still surface the facts about Luna even
    though no single fact contains every word. Callers rank by score so
    the facts matching the most terms come first; the limit trims noise
    from incidental single-term hits.
    """
    haystack = (fact.content + " " + " ".join(fact.tags)).lower()
    return sum(1 for term in set(terms) if term in haystack)


def _format_fact(fact: Fact) -> str:
    content = fact.content
    if len(content) > _MAX_FACT_CONTENT_CHARS:
        content = content[: _MAX_FACT_CONTENT_CHARS - 1].rstrip() + "…"
    pieces = [content]
    meta_bits: list[str] = []
    if fact.domain and fact.domain != "general":
        meta_bits.append(fact.domain)
    if fact.confidence and fact.confidence < 0.7:
        meta_bits.append(f"conf={fact.confidence:.2f}")
    if fact.tags:
        meta_bits.append("tags=" + ",".join(fact.tags[:4]))
    if meta_bits:
        pieces.append(f"[{', '.join(meta_bits)}]")
    return " ".join(pieces)


async def recall_facts(
    persistence: NarrativePersistence,
    session_id: str,
    *,
    user_id: str,
    query: str = "",
    limit: int = 5,
) -> RecallResult:
    """Substring + tag search over the session's facts.

    Empty ``query`` returns the most-recent facts. Limit is capped at
    ``_MAX_FACT_LIMIT`` so even a "give me everything" call doesn't
    flood the model's context.

    Why substring instead of embedding search: the fact store is small
    enough per session (low hundreds) that O(N) substring beats the
    embed-then-cosine round trip on both latency and dev complexity.
    A future enhancement could route via the existing vec layer when
    fact counts climb, but for now this is the right tradeoff.
    """
    limit = max(1, min(int(limit), _MAX_FACT_LIMIT))
    facts = await persistence._load_facts(session_id, user_id=user_id)

    # Exclude superseded facts so the model doesn't recall stale truths
    # that the engine has marked as overwritten.
    facts = [f for f in facts if not f.superseded_by]

    if query:
        terms = _query_terms(query)
        if terms:
            scored = [(f, _fact_match_score(f, terms)) for f in facts]
            scored = [(f, s) for f, s in scored if s > 0]
            # Rank: most terms matched first, then most recent.
            scored.sort(key=lambda fs: (-fs[1], -fs[0].established_at))
            matching = [f for f, _ in scored]
        else:
            # Query was all stopwords — fall back to recent facts.
            matching = sorted(facts, key=lambda f: f.established_at, reverse=True)
    else:
        # Sort by recency (established_at) descending — recent first.
        matching = sorted(facts, key=lambda f: f.established_at, reverse=True)
    total = len(matching)
    selected = matching[:limit]

    if not selected:
        if query:
            return RecallResult(
                summary=f"No facts matching '{query}'.",
                total_available=len(facts),
            )
        return RecallResult(
            summary="No facts established in this session yet.",
            total_available=0,
        )

    lines = [f"{total} fact(s) matching" + (f" '{query}'" if query else "") + f", showing {len(selected)}:"]
    for i, fact in enumerate(selected, 1):
        lines.append(f"{i}. {_format_fact(fact)}")

    return RecallResult(
        summary="\n".join(lines),
        items=[
            {
                "id": f.id,
                "content": f.content,
                "domain": f.domain,
                "confidence": f.confidence,
                "tags": list(f.tags),
                "established_at": f.established_at,
            }
            for f in selected
        ],
        total_available=total,
        truncated=total > len(selected),
    )


# ---------------------------------------------------------------------------
# Plot thread recall
# ---------------------------------------------------------------------------


def _format_plot_thread(thread: PlotThread) -> str:
    lines = [f"**{thread.title}** ({thread.status.value})"]
    if thread.description:
        desc = thread.description
        if len(desc) > 400:
            desc = desc[:399].rstrip() + "…"
        lines.append(desc)
    if thread.established_at:
        lines.append(f"Established at message {thread.established_at}")
    if thread.resolved_at:
        lines.append(f"Resolved at message {thread.resolved_at}")
    if thread.state:
        keys = list(thread.state.keys())[:6]
        lines.append("State keys: " + ", ".join(keys))
    return "\n".join(lines)


async def recall_plot_thread(
    persistence: NarrativePersistence,
    session_id: str,
    *,
    user_id: str,
    query: str,
) -> RecallResult:
    """Look up one plot thread by id or title-substring.

    Status-agnostic so resolved/abandoned threads are recallable —
    useful for "remind me what happened with the assassination
    investigation" after the thread closed.
    """
    needle = (query or "").strip()
    if not needle:
        return RecallResult(summary="recall_plot_thread called without a query.")

    threads = await persistence._load_plot_threads(session_id, user_id=user_id)
    needle_lc = needle.lower()
    # Try exact id match first
    by_id = next((t for t in threads if t.id == needle), None)
    if by_id is not None:
        return RecallResult(
            summary=_format_plot_thread(by_id),
            items=[{"id": by_id.id, "title": by_id.title, "status": by_id.status.value}],
            total_available=1,
        )
    # Then substring on title
    matches = [t for t in threads if needle_lc in t.title.lower()]
    if not matches:
        active_titles = sorted(t.title for t in threads if t.status == PlotStatus.ACTIVE)[:8]
        return RecallResult(
            summary=(
                f"No plot thread matching '{needle}'. "
                f"Active threads: {', '.join(active_titles) or '(none)'}."
            ),
            total_available=len(threads),
        )
    # If multiple match, list them concisely so the model can re-call with a sharper title.
    if len(matches) > 1:
        lines = [f"{len(matches)} threads match '{needle}'. Re-call with a more specific title:"]
        for t in matches[:10]:
            lines.append(f"- {t.title} [{t.status.value}]")
        return RecallResult(
            summary="\n".join(lines),
            items=[{"id": t.id, "title": t.title, "status": t.status.value} for t in matches],
            total_available=len(matches),
        )
    only = matches[0]
    return RecallResult(
        summary=_format_plot_thread(only),
        items=[{"id": only.id, "title": only.title, "status": only.status.value}],
        total_available=1,
    )


# ---------------------------------------------------------------------------
# Archive recall — semantic search over compacted ledger exchanges
# ---------------------------------------------------------------------------


def _format_archive_exchange(exchange: dict) -> str:
    """Format one archive exchange row for prose consumption."""
    user_text = (exchange.get("user_content") or "").strip()
    assistant_text = (exchange.get("assistant_content") or "").strip()
    if len(user_text) > _MAX_ARCHIVE_CONTENT_CHARS // 2:
        user_text = user_text[: _MAX_ARCHIVE_CONTENT_CHARS // 2 - 1].rstrip() + "…"
    if len(assistant_text) > _MAX_ARCHIVE_CONTENT_CHARS // 2:
        assistant_text = assistant_text[: _MAX_ARCHIVE_CONTENT_CHARS // 2 - 1].rstrip() + "…"
    turn = exchange.get("turn_number") or exchange.get("message_index") or "?"
    pieces = [f"[turn {turn}]"]
    if user_text:
        pieces.append(f"USER: {user_text}")
    if assistant_text:
        pieces.append(f"AI: {assistant_text}")
    return "\n".join(pieces)


async def recall_archive(
    persistence: NarrativePersistence,
    session_id: str,
    *,
    user_id: str,
    query: str,
    limit: int = 3,
) -> RecallResult:
    """Semantic search over compacted ledger exchanges (the ARCHIVE layer).

    Wraps the existing ``retrieve_archive_exchanges`` vec query so the
    model can fetch the actual narrative text from any prior point —
    useful for callbacks ("remind me what Elena said when she handed me
    the letter"), continuity checks ("did we ever actually meet the
    duke?"), or scene-setting ("describe the harvest festival again").
    """
    limit = max(1, min(int(limit), _MAX_ARCHIVE_LIMIT))
    needle = (query or "").strip()
    if not needle:
        return RecallResult(summary="recall_archive called without a query.")

    try:
        rows = await persistence.retrieve_archive_exchanges(
            session_id=session_id, user_id=user_id,
            query=needle, limit=limit,
        )
    except Exception as exc:
        log.warning("recall_archive_failed", session_id=session_id, error=str(exc)[:200])
        return RecallResult(summary=f"Archive recall failed: {exc}")

    if not rows:
        return RecallResult(
            summary=f"No archive matches for '{needle}'.",
            total_available=0,
        )

    lines = [f"{len(rows)} archive exchange(s) matching '{needle}':"]
    for i, row in enumerate(rows, 1):
        lines.append(f"---\n{_format_archive_exchange(row)}")

    return RecallResult(
        summary="\n".join(lines),
        items=[
            {
                "id": r.get("id"),
                "turn_number": r.get("turn_number") or r.get("message_index"),
                "user_content": r.get("user_content", "")[:600],
                "assistant_content": r.get("assistant_content", "")[:600],
            }
            for r in rows
        ],
        total_available=len(rows),
    )
