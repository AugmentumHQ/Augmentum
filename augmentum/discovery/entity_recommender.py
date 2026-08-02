"""Gate 1 of the consumption-entity ladder — the user's own catalog.

The cheapest, most-guaranteed recommendations don't need discovery at
all: the next unfinished volume in a series, the same author's unread
book, new chapters a sync just landed. All of it is SQL over
``file_index.source_metadata`` — fields every media provider already
writes uniformly (``author_normalized`` exists specifically to power
"other books by this author"; see media/sync.py).

Gates 2 (keyless aggregator relations — AniList/Open Library/
MusicBrainz/…) and 3 (guarded SearXNG) layer on per the spec; every
consumer of this module gets better recommendations when those land
WITHOUT changing shape, because picks carry their gate.

Relations, strongest first:
  continuation     — next unfinished item in this series
  new_arrival      — unconsumed items a sync added to a tracked series
  same_creator     — unconsumed work by the same author/narrator
  same_genre       — unstarted same-kind item sharing genres

Every pick carries a structured, speakable ``why`` — the grounding is
composed AT QUERY TIME from catalog facts, never free-associated.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_FINISHED_PCT = 95.0


@dataclass
class EntityPick:
    relation: str            # continuation | new_arrival | same_creator | same_genre
    gate: int                # 1 = own catalog (2/3 arrive with later phases)
    file_id: str
    title: str
    kind: str
    why: str                 # speakable grounding, catalog facts only
    creator: str = ""
    series_name: str = ""
    extra: dict = field(default_factory=dict)

    def to_payload(self) -> dict:
        return {
            "relation": self.relation, "gate": self.gate,
            "file_id": self.file_id, "title": self.title,
            "kind": self.kind, "why": self.why,
            "creator": self.creator, "series_name": self.series_name,
        }


def _meta(row_meta: str) -> dict:
    try:
        return json.loads(row_meta or "{}")
    except (json.JSONDecodeError, TypeError):
        return {}


def _consumed(meta: dict) -> bool:
    if meta.get("is_finished"):
        return True
    try:
        return float(meta.get("progress_pct") or 0.0) >= _FINISHED_PCT
    except (TypeError, ValueError):
        return False


def _started(meta: dict) -> bool:
    try:
        return float(meta.get("progress_pct") or 0.0) > 0.0
    except (TypeError, ValueError):
        return False


_NUM_RE = re.compile(r"(\d+)")


def _sequence_sort_key(name: str, meta: dict) -> tuple:
    """Order within a series: season/episode when present, else the
    first number in the name (vol/chapter), else the name itself."""
    season = meta.get("season_number") or 0
    episode = meta.get("episode_number") or 0
    if season or episode:
        return (0, int(season), int(episode), name.lower())
    m = _NUM_RE.search(name or "")
    if m:
        return (1, 0, int(m.group(1)), name.lower())
    return (2, 0, 0, (name or "").lower())


async def recommend_for_entity(
    conn: Any,
    entity: dict,
    *,
    user_id: str,
    limit: int = 5,
) -> list[EntityPick]:
    """Gate-1 picks for one resolved ConsumptionEntity. Never raises;
    degrades to fewer (or zero) picks on any query failure."""
    if not user_id or not isinstance(entity, dict):
        return []
    picks: list[EntityPick] = []
    seen_files = set(entity.get("local_refs") or [])
    s_name = (entity.get("series_name") or "").strip()
    kind = (entity.get("kind") or "").strip()
    creators = [c for c in (entity.get("creators") or []) if c]
    genres = {g.lower() for g in (entity.get("genres") or []) if g}

    # ── continuation + new_arrival: same series, unconsumed ─────────
    if s_name:
        rows = await _q(
            conn,
            """SELECT id, name, source_metadata FROM file_index
               WHERE user_id = ? AND is_directory = 0
                 AND json_extract(source_metadata, '$.series_name') = ?
               LIMIT 200""",
            (user_id, s_name),
        )
        in_series: list[tuple[str, str, dict]] = []
        unplayed_total = 0
        for fid, name, meta_raw in rows:
            meta = _meta(meta_raw)
            unplayed_total = max(
                unplayed_total, int(meta.get("unplayed_count") or 0),
            )
            if fid in seen_files or _consumed(meta):
                continue
            in_series.append((fid, name, meta))
        in_series.sort(key=lambda t: _sequence_sort_key(t[1], t[2]))
        if in_series:
            fid, name, meta = in_series[0]
            started = _started(meta)
            why = (
                f"next in {s_name} — you're partway in already"
                if started else f"next in {s_name}, ready in the library"
            )
            picks.append(EntityPick(
                relation="continuation", gate=1, file_id=fid,
                title=name, kind=kind, why=why,
                creator=creators[0] if creators else "",
                series_name=s_name,
            ))
        if unplayed_total > 0:
            picks.append(EntityPick(
                relation="new_arrival", gate=1,
                file_id=(in_series[0][0] if in_series else ""),
                title=s_name, kind=kind,
                why=f"{unplayed_total} new in {s_name} since the last sync",
                series_name=s_name,
                extra={"unplayed_count": unplayed_total},
            ))

    # ── same_creator: author/narrator equality, different series ────
    for creator in creators[:2]:
        if len(picks) >= limit:
            break
        rows = await _q(
            conn,
            """SELECT id, name, source_metadata FROM file_index
               WHERE user_id = ? AND is_directory = 0
                 AND (json_extract(source_metadata, '$.author_normalized') = ?
                      OR json_extract(source_metadata, '$.narrator_normalized') = ?)
               LIMIT 200""",
            (user_id, creator, creator),
        )
        for fid, name, meta_raw in rows:
            meta = _meta(meta_raw)
            if fid in seen_files or _consumed(meta):
                continue
            other_series = str(meta.get("series_name") or "").strip()
            if s_name and other_series == s_name:
                continue   # continuation already covers in-series
            display = str(meta.get("author") or "").strip() or creator
            picks.append(EntityPick(
                relation="same_creator", gate=1, file_id=fid,
                title=name, kind=kind,
                why=f"also by {display}, unread in the library",
                creator=creator, series_name=other_series,
            ))
            seen_files.add(fid)
            break   # one per creator — variety over exhaustiveness

    # ── same_genre: unstarted same-kind items sharing genres ────────
    if genres and kind and len(picks) < limit:
        rows = await _q(
            conn,
            """SELECT id, name, source_metadata FROM file_index
               WHERE user_id = ? AND is_directory = 0
                 AND json_extract(source_metadata, '$.entity_kind') = ?
               ORDER BY created_at DESC LIMIT 300""",
            (user_id, kind),
        )
        best: tuple[int, str, str, dict] | None = None
        for fid, name, meta_raw in rows:
            meta = _meta(meta_raw)
            if fid in seen_files or _started(meta) or _consumed(meta):
                continue
            if s_name and str(meta.get("series_name") or "").strip() == s_name:
                continue
            overlap = genres & {
                str(g).lower() for g in (meta.get("genres") or [])
            }
            if len(overlap) >= 1 and (best is None or len(overlap) > best[0]):
                best = (len(overlap), fid, name, meta)
        if best is not None:
            n_overlap, fid, name, meta = best
            shared = sorted(genres & {
                str(g).lower() for g in (meta.get("genres") or [])
            })
            picks.append(EntityPick(
                relation="same_genre", gate=1, file_id=fid,
                title=name, kind=kind,
                why=f"same shelf — {', '.join(shared[:2])} — never started",
                creator=str(meta.get("author") or ""),
                series_name=str(meta.get("series_name") or ""),
            ))

    return picks[:limit]


async def top_entity_picks(
    conn: Any,
    *,
    user_id: str,
    max_entities: int = 3,
    limit_per_entity: int = 3,
) -> list[tuple[dict, list[EntityPick]]]:
    """The user's most-active entities with their Gate-1 picks —
    shared entry point for the curator phase and the companion tool."""
    if not user_id:
        return []
    rows = await _q(
        conn,
        """SELECT name, entity_ref FROM interest_clusters
           WHERE user_id = ? AND kind = 'entity'
             AND COALESCE(dampened, 0) = 0 AND entity_ref != ''
           ORDER BY (frecency_short * 0.6 + frecency_long * 0.4) DESC,
                    updated_at DESC
           LIMIT ?""",
        (user_id, int(max_entities)),
    )
    out: list[tuple[dict, list[EntityPick]]] = []
    for _name, ref_raw in rows:
        entity = _meta(ref_raw)
        if not entity.get("series_key"):
            continue
        picks = await recommend_for_entity(
            conn, entity, user_id=user_id, limit=limit_per_entity,
        )
        if picks:
            out.append((entity, picks))
    return out


async def _q(conn: Any, sql: str, params: tuple) -> list:
    try:
        cur = await conn.execute(sql, params)
        rows = await cur.fetchall()
        await cur.close()
        return list(rows)
    except Exception:
        log.warning("entity_recommender_query_failed", exc_info=True)
        return []
