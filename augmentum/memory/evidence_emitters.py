"""Evidence emitters — turn user activity into Evidence (P2 activation).

The beachhead: the playlist emitter. A named playlist ("Study Jazz") is a
deliberate, high-trust signal. We match it CONSERVATIVELY against the user's
existing beliefs and, on a confident match, corroborate that belief through the
Evidence Bus — so a fact mentioned once in chat (PROVISIONAL) plus a matching
playlist (a second, INDEPENDENT channel) converges to ACTIVE. When a playlist
reveals an interest with NO existing belief, we never silently create durable
memory: we seed a PROVISIONAL candidate and surface it as a review-card OFFER
the user approves or dismisses.

Conservative by construction — miss rather than mis-corroborate. The matching
crux (a wrong signal→belief link = false confidence, the "knife") is contained
by requiring a real standalone-token overlap, corroborating only the SINGLE
best match, and never promoting a fresh interest without the user's nod.

See docs/superpowers/specs/2026-06-20-earned-understanding-design.md (P2).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.memory.evidence import EvidenceStore
    from augmentum.memory.store import MemoryStore

log = get_logger(__name__)

# Generic playlist vocabulary that carries no topical signal. A playlist named
# only out of these yields no subject and is skipped entirely.
_GENERIC = frozenset({
    "playlist", "playlists", "mix", "mixes", "favorites", "favourites", "favs",
    "my", "the", "a", "an", "songs", "song", "music", "tracks", "track", "list",
    "queue", "stuff", "misc", "untitled", "new", "vol", "part", "best", "of",
    "and", "to", "for", "playme", "play", "audio", "video", "videos", "collection", "set", "sets", "default",
})
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _subject_terms(name: str) -> list[str]:
    """Topical tokens from a playlist name — generic words and short tokens out."""
    return [
        t for t in _TOKEN_RE.findall((name or "").lower())
        if len(t) >= 3 and t not in _GENERIC
    ]


def _has_standalone_token(content: str, term: str) -> bool:
    """True only if ``term`` appears as a WHOLE word in content — guards
    against substring false positives (``art`` in ``cart``)."""
    return re.search(rf"\b{re.escape(term)}\b", (content or "").lower()) is not None


# Tier preference when several beliefs match: a PROVISIONAL belief benefits most
# from a second independent source (it can promote), so prefer it.
_TIER_RANK = {"provisional": 0, "active": 1, "core": 2, "archive": 3}


@dataclass(slots=True)
class PlaylistEmitResult:
    matched_memory_id: str = ""
    corroborated: bool = False        # recorded evidence against an existing belief
    promoted_checked: bool = False    # advanced the ladder (a new independent source)
    offered: bool = False             # seeded a PROVISIONAL candidate + review card
    candidate_memory_id: str = ""
    subject: str = ""
    terms: list[str] = field(default_factory=list)


async def _find_best_belief(
    memory_store: MemoryStore, *, user_id: str, terms: list[str],
) -> tuple[str, str] | None:
    """Conservative keyword match against the user's beliefs (incl. PROVISIONAL,
    which ``recall`` excludes). Returns (memory_id, tier) of the single best
    match, or None. Requires a standalone-token overlap, not a substring."""
    if not terms:
        return None
    likes = " OR ".join(["content LIKE ?"] * len(terms))
    params: list[Any] = [user_id, *[f"%{t}%" for t in terms]]
    cur = await memory_store._conn.execute(  # noqa: SLF001
        f"SELECT id, content, tier FROM memories "
        f"WHERE user_id = ? AND valid_until IS NULL AND ({likes})",
        params,
    )
    rows = await cur.fetchall()
    best: tuple[str, str] | None = None
    best_rank = 99
    for mid, content, tier in rows:
        if not any(_has_standalone_token(content, t) for t in terms):
            continue  # substring-only — reject
        rank = _TIER_RANK.get((tier or "").lower(), 5)
        if rank < best_rank:
            best, best_rank = (mid, tier or ""), rank
    return best


async def emit_playlist_evidence(
    memory_store: MemoryStore | None,
    evidence_store: EvidenceStore | None,
    *,
    user_id: str,
    playlist_name: str,
    companion_id: str | None = None,
    offer_new: bool = True,
) -> PlaylistEmitResult:
    """Record a playlist as evidence. Corroborate a matching belief, or offer a
    new candidate. Best-effort: callers wrap this and never let it break a save.
    """
    result = PlaylistEmitResult()
    if memory_store is None or evidence_store is None or not user_id:
        return result

    terms = _subject_terms(playlist_name)
    result.terms = terms
    result.subject = " ".join(terms)
    if not terms:
        return result  # nothing topical to learn from

    # Phrased as her observation so the review card reads naturally
    # ("You made a playlist … — want me to remember that?").
    claim = f'You made a playlist "{playlist_name[:80]}"'
    match = await _find_best_belief(memory_store, user_id=user_id, terms=terms)

    if match is not None:
        mid, _tier = match
        outcome = await evidence_store.corroborate_belief(
            memory_store, user_id=user_id, memory_id=mid,
            source="playlist", claim=claim, subject=result.subject,
            companion_id=companion_id,
        )
        result.matched_memory_id = mid
        result.corroborated = outcome.recorded
        result.promoted_checked = outcome.promoted_checked
        log.info(
            "evidence_playlist_corroborated",
            user_id=user_id, memory_id=mid, subject=result.subject,
            new_source=outcome.new_source, distinct=outcome.distinct_sources,
        )
        return result

    # No existing belief — a fresh interest candidate. Never durable without the
    # user's nod: seed a PROVISIONAL belief + a review-card offer.
    if not offer_new:
        return result
    candidate_id = await _offer_candidate(
        memory_store, evidence_store,
        user_id=user_id, subject=result.subject,
        playlist_name=playlist_name, claim=claim, companion_id=companion_id,
    )
    if candidate_id:
        result.offered = True
        result.candidate_memory_id = candidate_id
    return result


async def _offer_candidate(
    memory_store: MemoryStore,
    evidence_store: EvidenceStore,
    *,
    user_id: str,
    subject: str,
    playlist_name: str,
    claim: str,
    companion_id: str | None,
) -> str:
    """Seed a PROVISIONAL 'interested in X' belief from a playlist and queue a
    review-card offer. Returns the candidate memory id (or '' on failure)."""
    from augmentum.memory.models import MemoryType, SourceType

    content = f"Interested in {subject}"
    try:
        mid = await memory_store.store(
            content=content,
            memory_type=MemoryType.PREFERENCE,
            user_id=user_id,
            importance=0.5,
            confidence=0.5,
            source_type=SourceType.EXTRACTED,   # → PROVISIONAL under earned permanence
            source_context={"source": "playlist", "playlist": playlist_name[:80]},
        )
    except Exception as exc:  # noqa: BLE001
        log.warning("evidence_playlist_candidate_store_failed", error=str(exc)[:200])
        return ""

    # Record the playlist as the belief's first (single) source of evidence.
    try:
        await evidence_store.record(
            user_id=user_id, memory_id=mid, source="playlist",
            claim=claim, subject=subject, companion_id=companion_id,
        )
    except Exception:
        log.debug("evidence_playlist_candidate_record_failed", exc_info=True)

    # Surface it as a review card the user can approve (→ACTIVE) or dismiss.
    try:
        from augmentum.memory.notifications import queue_notification
        await queue_notification(
            memory_store._conn, mid, content,  # noqa: SLF001
            user_id=user_id,
            evidence=claim,
            tier="provisional",
            confidence=0.5,
            memory_type="preference",
        )
    except Exception:
        log.debug("evidence_playlist_candidate_notify_failed", exc_info=True)

    log.info("evidence_playlist_offer", user_id=user_id, subject=subject, memory_id=mid)
    return mid
