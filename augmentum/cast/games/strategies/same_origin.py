"""Strategy 1 — same-origin iframe + universal adapter chain.

Wraps the existing ``/ui/play/`` and ``/ui/play-web/`` surfaces shipped
in the browser-cast substrate (2026-06-04). The actual surface HTML +
adapter loader were already in place; this strategy just turns a
(title, profile) pair into the matching surface URL + adapter chain
that ``CastClassifier`` hands off to library2.

cost_rank=1 — cheapest. ``can_handle`` returns True for any title that
has either a numeric title_id (→ /ui/play/) OR an embed_url (→
/ui/play-web/). For cross-origin embed_urls this strategy still
"handles" the cast but the adapter chain inside the cross-origin
iframe is blind — Phase 3's proxy strategy is the fix for that.
"""

from __future__ import annotations

from typing import Any
from urllib.parse import quote

from augmentum.cast.games.models import (
    CastProfile,
    HostCapabilities,
    PreparedCast,
    STRATEGY_SHIM,
)
from augmentum.cast.games.strategies.base import CastStrategy


class SameOriginStrategy(CastStrategy):
    """Today's browser-cast substrate, factored behind the ABC."""

    id = STRATEGY_SHIM
    cost_rank = 1

    async def can_handle(
        self,
        title: dict[str, Any],
        host: HostCapabilities,
    ) -> bool:
        # We can always serve the same-origin iframe — even cross-origin
        # embeds work picture-wise; the adapter chain just doesn't reach
        # the inner realm. The classifier still picks us when nothing
        # else applies; the registry's profile records the input_chain
        # the user / probe expects to work.
        return _has_title_id(title) or bool(_embed_url(title))

    async def prepare(
        self,
        title: dict[str, Any],
        profile: CastProfile,
    ) -> PreparedCast:
        title_id = str(title.get("id") or title.get("title_id") or "")
        kind = str(title.get("kind") or "").lower()
        display = str(
            title.get("display_name")
            or title.get("title")
            or title.get("name")
            or "Game",
        )
        embed = _embed_url(title) or profile.embed_url

        # Kind decides the surface. Emulator/streamed go through
        # /ui/play/ (EmulatorJS / WebRTC viewer); everything web-shaped
        # rides /ui/play-web/ if it has an embed_url. Title_id alone
        # without a web-shaped kind also rides /ui/play/.
        web_kinds = {"js13k_game", "web_app"}
        if kind in web_kinds:
            if not embed:
                raise ValueError(
                    "SameOriginStrategy needs an embed_url for web kinds",
                )
            surface_url = (
                f"/ui/play-web/?embed_url={quote(embed, safe='')}"
                f"&title={quote(display, safe='')}&kiosk=1"
            )
        elif _has_title_id(title):
            surface_url = f"/ui/play/?title_id={quote(title_id)}&kiosk=1"
        elif embed:
            surface_url = (
                f"/ui/play-web/?embed_url={quote(embed, safe='')}"
                f"&title={quote(display, safe='')}&kiosk=1"
            )
        else:
            raise ValueError(
                "SameOriginStrategy needs either a title_id or embed_url",
            )

        return PreparedCast(
            title_id=title_id,
            strategy=self.id,
            surface_url=surface_url,
            surface_kind="html.generic",
            input_chain=profile.input_chain or ("gamepad_api",),
            keymap=profile.keymap,
            notes=profile.notes,
        )


def _has_title_id(title: dict[str, Any]) -> bool:
    return bool(title.get("id") or title.get("title_id"))


def _embed_url(title: dict[str, Any]) -> str:
    meta = title.get("metadata") if isinstance(title.get("metadata"), dict) else {}
    return str(meta.get("embed_url") or title.get("embed_url") or "")
