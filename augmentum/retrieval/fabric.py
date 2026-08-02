"""Retrieval fabric P1 — the resolve→act/offer/miss facade.

Spec: docs/superpowers/specs/2026-06-11-retrieval-fabric-design.md.

One lifecycle over everything retrievable. P1 sources:

  * ``index`` — the VFS ``file_index`` (the spine: media servers,
    documents, uploads, artifacts, bookmarks — already one catalog).
    Scoring REUSES the media resolver's field-tested helpers
    (title similarity + recency/progress boosts) and thresholds.
  * ``packs`` — knowledge-pack hybrid search via the existing
    ``PackManager.search`` (vector + FTS + ZIM, RRF-merged).

This module is a FACADE over existing functions — it owns the
lifecycle policy, not the searching:

  * Single-source resolution uses the media resolver's confidence
    gates verbatim: top score ≥ PLAY_THRESHOLD with PLAY_MARGIN over
    the runner-up → **act**; anything ≥ OFFER_THRESHOLD → **offer**
    (candidate cards); else → **miss** (honest, names the nearest).
  * Multi-source resolution NEVER auto-acts — cross-domain auto-fire
    is the misfire class the spec forbids ("asked for a movie, got a
    comic"). Legs merge via RRF (the PackManager pattern) and the
    outcome is offer or miss.

Never raises: every leg soft-fails to empty, a fully-failed resolve
returns a miss the calling verb can speak honestly.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from augmentum.media.resolver import (
    _KIND_TO_CONTENT,
    OFFER_THRESHOLD,
    PLAY_MARGIN,
    PLAY_THRESHOLD,
    _recency_boost,
    _strip_extension,
    _subtitle_for,
    _title_similarity,
    _tokens,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_RRF_K = 60
MAX_CANDIDATES = 5


@dataclass
class Item:
    """One retrievable item, source-agnostic."""

    id: str
    kind: str               # "audio" | "video" | "document" | "article" | ...
    source: str             # "index" | "packs" (leg id), plus origin detail
    title: str
    subtitle: str = ""
    uri: str = ""           # open/play target (file id, pack url, ...)
    actions: tuple[str, ...] = ()   # "play" | "open" | "infuse"
    score: float = 0.0      # leg-native score (similarity-ish for index)
    signals: dict[str, Any] = field(default_factory=dict)


@dataclass
class Resolution:
    outcome: str                    # "act" | "offer" | "miss"
    item: Item | None = None        # set when outcome == "act"
    candidates: list[Item] = field(default_factory=list)
    query: str = ""
    legs: tuple[str, ...] = ()      # which sources actually ran


# ── Legs ─────────────────────────────────────────────────────────────

async def _leg_index(
    query: str, *, user_id: str, app_state: Any,
    kinds: tuple[str, ...] | None, limit: int,
) -> list[Item]:
    file_index = getattr(app_state, "file_index", None) if app_state else None
    if file_index is None:
        return []
    try:
        hits = await file_index.search(query, user_id=user_id, limit=limit * 2)
    except Exception:  # noqa: BLE001 — legs soft-fail
        log.warning("fabric_index_leg_failed", exc_info=True)
        return []

    query_toks = _tokens(query)
    items: list[Item] = []
    for entry in hits or []:
        if getattr(entry, "is_directory", False):
            continue
        name = (getattr(entry, "name", "") or "").strip()
        file_id = (getattr(entry, "id", "") or "").strip()
        if not name or not file_id:
            continue
        kind = (getattr(entry, "kind", "") or "").strip()
        if kinds and kind not in kinds:
            continue
        sim = _title_similarity(query_toks, name)
        if sim <= 0.0:
            continue
        boost, in_progress, progress = _recency_boost(entry)
        items.append(Item(
            id=file_id,
            kind=kind,
            source="index",
            title=_strip_extension(name)[:120],
            subtitle=_subtitle_for(entry),
            uri=file_id,
            actions=("play", "open") if kind in _KIND_TO_CONTENT else ("open",),
            score=sim + boost,
            signals={"in_progress": in_progress, "progress_pct": progress},
        ))
    items.sort(key=lambda i: -i.score)
    return items[:limit]


async def _leg_packs(
    query: str, *, user_id: str, app_state: Any, limit: int,
) -> list[Item]:
    manager = getattr(app_state, "pack_manager", None) if app_state else None
    if manager is None:
        return []
    try:
        installed = manager.installed()
        pack_ids = [
            str(p.get("id") or "") for p in installed
            if p.get("id") and not p.get("disabled")
        ]
        if not pack_ids:
            return []
        results = await manager.search(query, pack_ids=pack_ids, limit=limit)
    except Exception:  # noqa: BLE001
        log.warning("fabric_packs_leg_failed", exc_info=True)
        return []

    items: list[Item] = []
    for r in results or []:
        title = (getattr(r, "title", "") or "").strip()
        if not title:
            continue
        items.append(Item(
            id=f"{getattr(r, 'pack_id', '')}:{getattr(r, 'url', '') or title}",
            kind="article",
            source="packs",
            title=title[:120],
            subtitle=(getattr(r, "section", "") or getattr(r, "pack_id", ""))[:80],
            uri=getattr(r, "url", "") or "",
            actions=("infuse", "open"),
            score=float(getattr(r, "score", 0.0) or 0.0),
            signals={"pack_id": getattr(r, "pack_id", "")},
        ))
    return items[:limit]


# ── Merge + lifecycle ────────────────────────────────────────────────

def _rrf_merge(legs: list[list[Item]], limit: int) -> list[Item]:
    """Reciprocal-rank fusion across legs (the PackManager pattern).

    Leg-native scores aren't comparable across sources; ranks are.
    """
    fused: dict[str, tuple[float, Item]] = {}
    for leg in legs:
        for rank, item in enumerate(leg):
            key = f"{item.source}:{item.id}"
            add = 1.0 / (_RRF_K + rank + 1)
            prev = fused.get(key)
            fused[key] = (prev[0] + add if prev else add, item)
    ranked = sorted(fused.values(), key=lambda t: -t[0])
    return [item for _, item in ranked[:limit]]


async def resolve(
    query: str,
    *,
    user_id: str,
    app_state: Any,
    sources: tuple[str, ...] = ("index", "packs"),
    kinds: tuple[str, ...] | None = None,
    limit: int = 8,
) -> Resolution:
    """Resolve a retrieval query through the fabric lifecycle."""
    query = (query or "").strip()
    if not query or not user_id:
        return Resolution(outcome="miss", query=query)

    tasks = []
    legs_run: list[str] = []
    if "index" in sources:
        legs_run.append("index")
        tasks.append(_leg_index(
            query, user_id=user_id, app_state=app_state,
            kinds=kinds, limit=limit,
        ))
    if "packs" in sources:
        legs_run.append("packs")
        tasks.append(_leg_packs(
            query, user_id=user_id, app_state=app_state, limit=limit,
        ))
    if not tasks:
        return Resolution(outcome="miss", query=query)

    leg_results = await asyncio.gather(*tasks)
    non_empty = [leg for leg in leg_results if leg]

    if not non_empty:
        return Resolution(outcome="miss", query=query, legs=tuple(legs_run))

    if len(legs_run) == 1:
        # Single source — leg-native scores carry meaning, so the
        # media resolver's confidence gates apply verbatim.
        items = non_empty[0]
        shortlist = [i for i in items if i.score >= OFFER_THRESHOLD]
        shortlist = shortlist[:MAX_CANDIDATES]
        if not shortlist:
            return Resolution(outcome="miss", query=query, legs=tuple(legs_run))
        top = shortlist[0]
        runner_up = shortlist[1].score if len(shortlist) > 1 else 0.0
        if top.score >= PLAY_THRESHOLD and (top.score - runner_up) >= PLAY_MARGIN:
            return Resolution(
                outcome="act", item=top, candidates=shortlist,
                query=query, legs=tuple(legs_run),
            )
        return Resolution(
            outcome="offer", candidates=shortlist,
            query=query, legs=tuple(legs_run),
        )

    # Multi-source: ranks fuse, confidence doesn't — never auto-act
    # across domains. Offer the fused shortlist.
    fused = _rrf_merge(list(leg_results), MAX_CANDIDATES)
    if not fused:
        return Resolution(outcome="miss", query=query, legs=tuple(legs_run))
    return Resolution(
        outcome="offer", candidates=fused, query=query, legs=tuple(legs_run),
    )
