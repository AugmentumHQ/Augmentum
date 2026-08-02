"""App menu — the application's own buttons as companion action space.

The convergent platform pattern (Apple App Intents, Windows App
Actions, spec conversation 2026-06-10): surfaces declare their
user-nameable outcome actions as palette commands with agent metadata;
the client syncs the live catalog up; ONE verb (``app.act``) matches
free-form intent against the closed, stakes-capped list and fires the
existing client handler via ``palette.run``. No DOM driving, no
per-action verb files, no roster growth — the model's prompt never
sees the catalog.

Deliberately ephemeral, mirroring ``presence_context.AttentionStore``:
the catalog is a cache of CLIENT state. Restart → empty until the
client re-syncs (it re-syncs on every page load and registry change).
Per-user, latest-client-wins — same trade as the attention slots.

Filtering funnel (why a fuzzy match can't do damage):
  1. Registration is opt-in curation — surfaces register outcomes,
     not plumbing (the registerCommand ``agent`` field is the blessing).
  2. ``live`` flag — the client evaluates each command's ``when`` guard
     at sync time; dead actions never reach the candidate list.
  3. Stakes cap — only ``trivial_reversible`` entries are matchable.
  4. Closed-world pick — the utility model chooses from an enumerated
     menu or says none; it cannot invent an action.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Catalog bounds — a surface registering hundreds of "actions" is a
# bug (or an attack); cap and truncate rather than trust.
_MAX_ENTRIES = 64
_MAX_FIELD_CHARS = 160
_CATALOG_TTL_S = 6 * 3600.0

# v1 stakes cap: a wrong semantic match may only fire something a
# shrug undoes. Deletes/sends/purchases stay verb-only.
_MATCHABLE_STAKES = ("trivial_reversible",)


class MenuStore:
    """Per-user catalog of client-registered agent actions."""

    def __init__(self) -> None:
        self._catalogs: dict[str, tuple[float, list[dict[str, str]]]] = {}

    def update(self, user_id: str, entries: list[dict[str, Any]]) -> int:
        """Replace the user's catalog. Returns the accepted entry count."""
        if not user_id or not isinstance(entries, list):
            return 0
        cleaned: list[dict[str, str]] = []
        for raw in entries[:_MAX_ENTRIES]:
            if not isinstance(raw, dict):
                continue
            cid = str(raw.get("id") or "").strip()[:_MAX_FIELD_CHARS]
            desc = str(raw.get("description") or "").strip()[:_MAX_FIELD_CHARS]
            if not cid or not desc:
                continue
            cleaned.append({
                "id": cid,
                "description": desc,
                "keywords": str(raw.get("keywords") or "")[:_MAX_FIELD_CHARS],
                "stakes": str(raw.get("stakes") or "trivial_reversible"),
                "speak": str(raw.get("speak") or "")[:_MAX_FIELD_CHARS],
                "live": "1" if raw.get("live") else "",
            })
        self._catalogs[user_id] = (time.time(), cleaned)
        return len(cleaned)

    def catalog(self, user_id: str) -> list[dict[str, str]]:
        """Current matchable candidates: live + stakes-capped, TTL-fresh."""
        stamped = self._catalogs.get(user_id)
        if stamped is None:
            return []
        ts, entries = stamped
        if time.time() - ts > _CATALOG_TTL_S:
            self._catalogs.pop(user_id, None)
            return []
        return [
            e for e in entries
            if e["live"] and e["stakes"] in _MATCHABLE_STAKES
        ]

    def all_entries(self, user_id: str) -> list[dict[str, str]]:
        """Full synced catalog (including non-live) — for honest misses."""
        stamped = self._catalogs.get(user_id)
        return list(stamped[1]) if stamped else []

    def reset(self) -> None:
        self._catalogs.clear()


MENU = MenuStore()


def observe_commands(user_id: str, topic: str, payload: dict[str, Any]) -> None:
    """Topic mapper for ``surface.commands.catalog`` observe events."""
    if topic != "surface.commands.catalog":
        return
    count = MENU.update(user_id, payload.get("entries") or [])
    log.debug("app_menu_catalog_updated", user_id=user_id, entries=count)


# ── Closed-world matcher ──────────────────────────────────────────────

_MATCH_SYSTEM_PROMPT = (
    "You map a user's request to ONE action from a menu, or none.\n"
    "Reply with ONLY a JSON object: {\"choice\": \"<action id>\", "
    "\"confidence\": 0.0-1.0} or {\"choice\": \"none\"}.\n"
    "Rules: pick an action ONLY if the request clearly asks for that "
    "outcome. Paraphrases count ('love this song' = add to favorites). "
    "Topic overlap does NOT count ('what station is this' is a "
    "question, not a button). When unsure, choose none."
)

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


async def match_intent(
    text: str,
    candidates: list[dict[str, str]],
    *,
    app_state: Any,
    user_id: str = "",
    session_id: str = "",
) -> dict[str, str] | None:
    """Pick the menu entry the user is asking for, or None.

    Same utility-call contract as the architect router: temperature 0,
    thinking off, short timeout, soft-fail to None on every error path
    — a degraded matcher means an honest "I don't have a button for
    that", never a guess.
    """
    if not text.strip() or not candidates:
        return None
    registry = getattr(app_state, "provider_registry", None)
    if registry is None:
        return None

    from augmentum.config import settings
    from augmentum.models.base import InternalChatRequest, Message

    override = (getattr(settings, "architect_router_model", "") or "").strip()
    timeout_s = max(
        0.5,
        float(getattr(settings, "architect_router_timeout_ms", 4000)) / 1000.0,
    )

    try:
        backend, resolved_model = await registry.resolve_backend_with_fabric(
            override, user_id=user_id, session_id=session_id,
        )
    except Exception:  # noqa: BLE001
        log.debug("app_menu_resolve_failed", exc_info=True)
        return None
    if backend is None:
        return None

    menu_lines = "\n".join(
        f"- {e['id']}: {e['description']}"
        + (f" (keywords: {e['keywords']})" if e["keywords"] else "")
        for e in candidates
    )
    req = InternalChatRequest(
        model=resolved_model or override or "",
        messages=[
            Message(role="system", content=_MATCH_SYSTEM_PROMPT),
            Message(
                role="user",
                content=f"Menu:\n{menu_lines}\n\nRequest: {text.strip()}",
            ),
        ],
        stream=False,
        temperature=0.0,
        chat_template_kwargs={"enable_thinking": False},
        max_tokens=128,
    )

    try:
        resp = await asyncio.wait_for(backend.chat(req), timeout=timeout_s)
    except Exception:  # noqa: BLE001 — timeout and backend errors alike
        log.debug("app_menu_match_call_failed", exc_info=True)
        return None

    raw = getattr(getattr(resp, "message", None), "content", "") or ""
    m = _JSON_RE.search(raw)
    if m is None:
        log.debug("app_menu_match_no_json", raw=raw[:120])
        return None
    try:
        parsed = json.loads(m.group(0))
    except Exception:  # noqa: BLE001
        log.debug("app_menu_match_bad_json", raw=raw[:120])
        return None

    choice = str(parsed.get("choice") or "").strip()
    try:
        confidence = float(parsed.get("confidence") or 0.0)
    except (TypeError, ValueError):
        confidence = 0.0
    if not choice or choice.lower() == "none" or confidence < 0.55:
        return None
    # Closed world: the id must be on the menu we showed it.
    for entry in candidates:
        if entry["id"] == choice:
            log.info(
                "app_menu_matched",
                command_id=choice, confidence=confidence,
                text_preview=text[:60],
            )
            return entry
    log.info("app_menu_match_offlist", choice=choice[:80])
    return None
