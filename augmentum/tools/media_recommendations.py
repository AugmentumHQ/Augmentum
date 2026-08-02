"""media_recommendations — her window into the consumption-entity ladder.

Data-returning tool (headless-first law): "what should I listen to
next?" pulls the user's most-active entities and their catalog-first
picks into the loop, and she narrates. No surface yank, no web call —
Gate 1 is pure SQL over the user's own library, so the answer is
instant and every line is groundable ("vol. 7 is sitting right there").

Gates 2/3 (keyless aggregator relations, guarded SearXNG) enrich the
same picks shape in later phases without changing this tool. Spec:
docs/superpowers/specs/2026-06-12-consumption-entity-discovery-design.md
"""

from __future__ import annotations

from typing import Any

from augmentum.tools.base import SurfaceExposure, Tool, ToolCategory, ToolResult
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_RELATION_LABEL = {
    "continuation": "continue",
    "new_arrival": "new since last sync",
    "same_creator": "more by the same author",
    "same_genre": "same shelf, never started",
    "for_you": "fits your taste",
    "fresh": "new in your library",
}


class MediaRecommendationsTool(Tool):
    """Catalog-grounded next-thing picks across everything they consume."""

    def __init__(self, app_state: Any = None) -> None:
        self._app_state = app_state

    @property
    def surfaces(self) -> SurfaceExposure:
        # Conversational-surface tool — hidden from reasoning-flow
        # steps (see SurfaceExposure.flow).
        return SurfaceExposure(flow=False)

    @property
    def name(self) -> str:
        return "media_recommendations"

    @property
    def description(self) -> str:
        return (
            "What they should pick up next, grounded in their OWN "
            "library — across movies, shows, audiobooks, podcasts, "
            "comics, books. Reads their whole catalog (not just what "
            "they've played), so it works even with no history: the "
            "next unfinished volume, an unread book by an author they "
            "like, or a never-started title matching the type/genre/"
            "theme they asked for. Pass kind/genre/query from what they "
            "said ('a funny movie' → kind=movie, genre=comedy); leave "
            "blank for 'surprise me'. Use when they ask what to watch / "
            "listen to / read next, or for a recommendation. Every pick "
            "carries WHY it fits — speak the why, never invent your own."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.SEARCH

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "description": (
                        "Content type when they name one: movie | show "
                        "| audiobook | podcast | comic | manga | book | "
                        "music | video. Omit for everything."
                    ),
                },
                "genre": {
                    "type": "string",
                    "description": (
                        "Single genre/mood word when specified — "
                        "'comedy', 'horror', 'sci-fi'. Omit otherwise."
                    ),
                },
                "query": {
                    "type": "string",
                    "description": (
                        "Free-text theme when the ask isn't a clean "
                        "genre — 'about space travel', 'a heist'. Their "
                        "words. Omit for a plain type/genre browse."
                    ),
                },
            },
        }

    @property
    def timeout(self) -> float:
        return 8.0

    @property
    def cacheable(self) -> bool:
        # Progress moves between calls — asking twice is legitimate.
        return False

    async def execute(self, **kwargs) -> ToolResult:
        user_id = self.extract_user_id(kwargs)
        if not user_id:
            return ToolResult(success=False, error="No user context.")
        kind_arg = str(kwargs.get("kind") or "").strip().lower()
        genre = str(kwargs.get("genre") or "").strip().lower()
        query = str(kwargs.get("query") or "").strip()

        conn = getattr(
            getattr(
                getattr(self._app_state, "state_manager", None),
                "backend", None,
            ),
            "conn", None,
        )
        if conn is None:
            return ToolResult(
                success=False, error="Library database isn't available.",
            )

        try:
            from augmentum.discovery.catalog_recommender import (
                kind_word_to_filters,
                recommend_picks,
            )
            file_kind, entity_kind = kind_word_to_filters(kind_arg)
            picks = await recommend_picks(
                conn,
                getattr(self._app_state, "file_index", None),
                user_id=user_id,
                kind=file_kind or None,
                entity_kind=entity_kind or None,
                genre=genre, query=query, limit=5,
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("media_recommendations_failed", error=str(exc))
            return ToolResult(
                success=False, error=f"recommendations failed: {exc}",
            )

        lines: list[str] = []
        payloads: list[dict] = []
        for p in picks:
            label = _RELATION_LABEL.get(p.relation, p.relation)
            lines.append(f"- [{label}] {p.title} — {p.why}")
            if p.file_id and len(payloads) < 4 and not any(
                c["file_id"] == p.file_id for c in payloads
            ):
                payloads.append({
                    "file_id": p.file_id,
                    "title": p.title,
                    "subtitle": p.why,
                    "kind": p.kind,
                    "content_kind": p.kind,
                    "in_progress": p.relation == "continuation",
                    "relation": p.relation,
                })

        # Quick-accept rail: park the picks (so "play the second one"
        # resolves through the router's offered-picks block onto
        # media.play's file_id fast-path) and queue the same tappable
        # cards media.play's offer shows — drained to the surface at
        # the next turn boundary. Best-effort: a missing referent
        # cache never blocks the narrated answer.
        ctx = kwargs.get("_context") or {}
        session_id = (
            str(ctx.get("session_id") or "") if isinstance(ctx, dict) else ""
        )
        if payloads:
            try:
                from augmentum.intent.dispatch import get_referent_cache
                refs = get_referent_cache(
                    self._app_state, user_id, session_id,
                )
                refs.pending_candidates = payloads
                import time as _t
                refs.pending_candidates_at = _t.time()
                refs.pending_surface_events.append({
                    "type": "intent_action",
                    "v": 1,
                    "action": "media.recommend",
                    "tier": 3,
                    "short_circuit": False,
                    "surface": {
                        "channel": "companion.candidates",
                        "payload": {
                            "intent": "media.recommend",
                            "query": kind_arg,
                            "candidates": payloads,
                        },
                    },
                })
            except Exception:  # noqa: BLE001
                log.debug(
                    "media_recommendations_card_park_failed", exc_info=True,
                )

        if not lines:
            scope = " ".join(w for w in (genre, kind_arg) if w).strip()
            scope = f" {scope}" if scope else ""
            return ToolResult(
                success=True,
                output=(
                    f"Their library has nothing{scope} that fits — it's "
                    "either not in their catalog or everything matching "
                    "is finished. Don't invent a title: say so honestly "
                    "and offer to widen it or look outside their library."
                ),
            )
        header = (
            "Library-grounded picks (their own collection — every line "
            "is a real, available item; speak the bracketed reason):"
        )
        if payloads:
            header += (
                "\nThese are also showing as tappable cards on their "
                "screen — they can tap one, or just name it ('the "
                "second one' works)."
            )
        return ToolResult(
            success=True,
            output=header + "\n" + "\n".join(lines)[:4000],
            metadata={"picks": len(lines), "cards": len(payloads)},
        )
