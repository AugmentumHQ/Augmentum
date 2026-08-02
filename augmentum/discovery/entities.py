"""Consumption-entity resolution — the typed hop discovery was missing.

A consumption signal (``media_play``, ``comic_read``) is not a topic:
it's an encounter with a THING the user consumes — an audiobook, a
comic series, a show, an album. Treating its title as a topic string is
how the curator paired "My Quiet Blacksmith Life in Another World"
with a systems-programming blog post on the shared token "life"
(2026-06-12 incident; spec at docs/superpowers/specs/
2026-06-12-consumption-entity-discovery-design.md).

This module resolves such signals against the unified catalog
(``file_index.source_metadata`` — uniform across Audiobookshelf, Emby,
Jellyfin, Komga, Suwayomi, LibriVox) into WORK/SERIES-level entity
clusters (``interest_clusters.kind = 'entity'``):

- Series-level granularity: "Vol. 6" and "Vol. 7" are ONE interest.
- Entity clusters carry a structured ``entity_ref`` (kind, creators,
  series key, genres, local file refs) instead of relying on the name.
- They are minted WITHOUT a centroid embedding and never enter the
  vec index, so topic signals can't drift into them (and vice versa).
- They are excluded from every topic-shaped consumer: the curator's
  feed-polling loop and the For-You SearXNG recommender both filter
  ``kind = 'topic'``. Entities route down the catalog-first ladder in
  ``discovery/entity_recommender.py`` instead.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Signal types that represent consuming a catalog item. Anything here
# MUST carry ``metadata.file_id`` to resolve; without it the signal
# falls back to ordinary topic clustering.
ENTITY_SIGNAL_TYPES = frozenset({"media_play", "comic_read"})

# Volume/chapter/part suffixes stripped to reach the WORK-level key.
# Conservative: only trailing, clearly-enumerative fragments.
_VOLUME_SUFFIX_RE = re.compile(
    r"""
    [\s,:\-–—]*
    (?:
        (?:vol(?:ume)?|book|part|pt|ch(?:apter)?|episode|ep|season|s)
        \.?\s*\d+[a-z]?
        |
        \#\s*\d+
        |
        \(\s*\d{4}\s*\)          # trailing (year)
    )
    \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


def series_key(title: str, series_name: str = "") -> str:
    """Normalize to a work/series-level identity key.

    Prefers the provider's series_name (Komga/Suwayomi/ABS emit it);
    otherwise strips trailing volume/chapter enumerators from the
    title — repeatedly, so "Vol. 6 Part 2" collapses too. Lowercased,
    punctuation collapsed, whitespace normalized.
    """
    base = (series_name or "").strip() or (title or "").strip()
    if not base:
        return ""
    prev = None
    while prev != base:
        prev = base
        base = _VOLUME_SUFFIX_RE.sub("", base).strip()
    base = re.sub(r"[^\w\s]", " ", base.lower())
    return re.sub(r"\s+", " ", base).strip()


async def resolve_entity(
    conn: Any, *, user_id: str, file_id: str,
) -> dict | None:
    """Resolve a catalog file row into a ConsumptionEntity dict.

    Returns None when the row doesn't exist (or belongs to another
    user — the lookup is user-scoped, standard isolation rules).
    """
    if not (user_id and file_id):
        return None
    try:
        cur = await conn.execute(
            """SELECT name, source, mime_type, source_metadata
               FROM file_index WHERE id = ? AND user_id = ?""",
            (file_id, user_id),
        )
        row = await cur.fetchone()
        await cur.close()
    except Exception:
        log.warning("entity_resolve_query_failed", file_id=file_id[:40])
        return None
    if row is None:
        return None

    name, source, mime_type, meta_raw = row[0], row[1], row[2], row[3]
    try:
        meta = json.loads(meta_raw or "{}")
    except (json.JSONDecodeError, TypeError):
        meta = {}

    kind = str(meta.get("entity_kind") or "").strip().lower()
    if not kind:
        # Fall back on coarse inference — uniform enough for routing.
        mt = (mime_type or "").lower()
        if mt.startswith("audio"):
            kind = "audiobook"
        elif mt.startswith("video"):
            kind = "video"
        elif "comic" in (source or "") or mt in (
            "application/vnd.comicbook+zip", "application/x-cbz",
        ):
            kind = "comic"
        else:
            kind = "media"

    title = str(name or "").strip()
    s_name = str(meta.get("series_name") or "").strip()
    skey = series_key(title, s_name)
    if not skey:
        return None

    creators = [
        c for c in (
            str(meta.get("author_normalized") or "").strip(),
            str(meta.get("narrator_normalized") or "").strip(),
        ) if c
    ]
    genres = [
        str(g).strip() for g in (meta.get("genres") or [])
        if str(g).strip()
    ]
    return {
        "kind": kind,
        "title": title,
        "series_key": skey,
        "series_name": s_name,
        "creators": creators,
        "genres": genres[:8],
        "year": meta.get("year") or 0,
        "source": source or "",
        "local_refs": [file_id],
        "external_ids": {},   # filled by the P2 identity spine
    }


async def assign_entity_signal(
    store: Any,
    signal_id: str,
    *,
    user_id: str,
    file_id: str,
    fallback_title: str = "",
) -> str | None:
    """Route a consumption signal to its series-level entity cluster.

    Find-or-create by ``(user_id, kind='entity', series_key)`` equality
    — no embeddings involved. Returns the cluster_id, or None when
    resolution fails entirely (caller falls back to topic clustering,
    which keeps pre-migration behavior for unresolvable signals).
    """
    conn = getattr(store, "_conn", None)
    if conn is None:
        return None

    entity = await resolve_entity(conn, user_id=user_id, file_id=file_id)
    if entity is None:
        # Catalog miss (stale file_id, race with a re-sync). Still keep
        # it OUT of the topic lane when we can key on the title.
        skey = series_key(fallback_title)
        if not skey:
            return None
        entity = {
            "kind": "media", "title": fallback_title.strip(),
            "series_key": skey, "series_name": "", "creators": [],
            "genres": [], "year": 0, "source": "",
            "local_refs": [file_id] if file_id else [],
            "external_ids": {},
        }

    skey = entity["series_key"]
    try:
        cur = await conn.execute(
            """SELECT cluster_id, entity_ref FROM interest_clusters
               WHERE user_id = ? AND kind = 'entity'
                 AND json_extract(entity_ref, '$.series_key') = ?
               LIMIT 1""",
            (user_id, skey),
        )
        row = await cur.fetchone()
        await cur.close()
    except Exception:
        log.warning("entity_cluster_lookup_failed", series_key=skey[:60])
        return None

    if row is not None:
        cluster_id = row[0]
        # Refresh the ref — progress moved, maybe a new volume's
        # file_id joined the series.
        try:
            old = json.loads(row[1] or "{}")
        except (json.JSONDecodeError, TypeError):
            old = {}
        refs = list(dict.fromkeys(
            (old.get("local_refs") or []) + entity["local_refs"],
        ))[-20:]
        merged = {**old, **entity, "local_refs": refs}
        try:
            await conn.execute(
                """UPDATE interest_clusters
                   SET entity_ref = ?, signal_count = signal_count + 1,
                       updated_at = datetime('now')
                   WHERE cluster_id = ?""",
                (json.dumps(merged), cluster_id),
            )
            await conn.commit()
            await store.update_signal_cluster(signal_id, cluster_id)
        except Exception:
            log.warning("entity_cluster_update_failed", cluster_id=cluster_id)
        return cluster_id

    # Mint. Display name = series-level (no dangling "Vol."), centroid
    # deliberately ABSENT — entity clusters are matched by key equality
    # and must not absorb topic signals through the vec index.
    cluster_id = str(uuid.uuid4())
    display = entity["series_name"] or entity["title"]
    display = re.sub(r"\s+", " ", _VOLUME_SUFFIX_RE.sub("", display)).strip()
    try:
        await store.upsert_cluster({
            "cluster_id": cluster_id,
            "name": display[:80],
            "centroid_embedding": None,
            "frecency_short": 0.0,
            "frecency_long": 0.0,
            "depth_level": 1,
            "signal_count": 1,
            "narration": None,
            "knowledge_gaps": None,
            "adjacent_topics": None,
            "dampened": 0,
            "kind": "entity",
            "entity_ref": json.dumps(entity),
        }, user_id=user_id)
        await store.update_signal_cluster(signal_id, cluster_id)
    except Exception:
        log.warning("entity_cluster_create_failed", series_key=skey[:60])
        return None
    log.info(
        "entity_cluster_created",
        user_id=user_id, kind=entity["kind"], name=display[:60],
    )
    return cluster_id
