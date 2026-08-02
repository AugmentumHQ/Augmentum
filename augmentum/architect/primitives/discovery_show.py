"""discovery.show — open the discovery surface, optionally filtered.

User: "show me something to watch", "recommend a movie", "find me
a podcast about cooking", "what's worth reading" (latter would only
fire from imperative templates — bare WH questions skip via the
signal-routing gate). Imperative-only.

Implementation is intentionally thin: dispatch a surface event to
open discovery with an optional kind/topic filter. Discovery's own
ranking engine handles the recommendation work — we don't try to
re-derive it server-side. The user lands in discovery already
filtered to what they asked about.

Kinds we recognize: ``movie``, ``video``, ``podcast``, ``book``,
``audiobook``, ``music``, ``article``. Anything else flows through
as a free-text topic filter the discovery surface treats as a search.
"""

from __future__ import annotations

from typing import Any

from augmentum.intent.action import (
    ActionFanout,
    ActionResult,
    SessionContext,
)

# Tier-3-only: LLM picks based on intent + context. See [[no-regex-switchboard]].
_TIER3_ONLY = ActionFanout(tier1=False, tier2=False, tier3=True)
from augmentum.intent.registry import register_action
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# Kind-keyword recognition. The matcher captures whatever the user
# said into ``{kind}`` (or leaves it empty). We normalize here so
# the surface event carries a canonical kind discovery understands.
_KIND_KEYWORDS = {
    "movie": "movie", "movies": "movie", "film": "movie", "films": "movie",
    "video": "video", "videos": "video", "youtube": "video",
    "podcast": "podcast", "podcasts": "podcast",
    "audiobook": "audiobook", "audiobooks": "audiobook",
    "book": "book", "books": "book", "novel": "book", "novels": "book",
    "music": "music", "song": "music", "songs": "music", "track": "music",
    "article": "article", "articles": "article", "post": "article",
    "show": "video", "shows": "video", "series": "video",
}


def _normalize_kind(raw: str) -> str:
    if not raw:
        return ""
    s = raw.strip().lower()
    return _KIND_KEYWORDS.get(s, "")


async def _discovery_show_handler(
    text: str,
    session: SessionContext,
    args: dict[str, Any],
) -> ActionResult | None:
    if not session.user_id:
        return ActionResult(
            short_circuit=True,
            speak="I can't open discovery for a signed-out session.",
        )

    kind_raw = (args.get("kind") or "").strip()
    topic = (args.get("topic") or "").strip()
    kind = _normalize_kind(kind_raw)

    log.info(
        "architect_discovery_show",
        user_id=session.user_id, kind=kind or "(any)", topic=topic[:80],
    )

    # Spoken ack — keep it brief, mirror the user's words.
    if kind and topic:
        speak = f"Looking for {kind}s about {topic[:50]}."
    elif kind:
        speak = f"Showing you some {kind}s."
    elif topic:
        speak = f"Looking for something about {topic[:50]}."
    else:
        speak = "Opening discovery."

    return ActionResult(
        short_circuit=True,
        speak=speak,
        surface_emit={
            "channel": "discovery.open",
            "payload": {
                "kind": kind,
                "topic": topic,
            },
        },
    )


register_action(
    id="discovery.show",
    summary=(
        "Open the discovery surface, optionally filtered by content "
        "kind (movie / video / podcast / book / audiobook / music / "
        "article) and/or topic. Imperative forms only — bare "
        "'what's good' is conversational and routed to the LLM."
    ),
    examples=[
        "show me something to watch",
        "recommend a movie",
        "find me a podcast about cooking",
        "show me audiobooks about history",
        "recommend something to read",
        "find me music for studying",
        "show me what's new",
    ],
    handler=_discovery_show_handler,
    delivery="artifact",
    arg_schema={
        "kind": {
            "type": "string",
            "description": "Optional content kind filter.",
            "enum": list(set(_KIND_KEYWORDS.values())),
        },
        "topic": {
            "type": "string",
            "description": "Optional topic / theme to narrow recommendations.",
        },
    },
    surfaces=["becca", "chat"],
    stakes="trivial_reversible",
    fanout=_TIER3_ONLY,
)
