"""weather.today — direct keyless weather, no search round-trip.

First consumer of the direct-sources layer (``augmentum/sources/``).
"What's the weather" used to ride LLM → web_search → SearXNG → scrape:
slow, lossy, and un-reusable. This verb hits Open-Meteo directly and
returns typed data into the loop — she narrates it.

Seamless home-location ladder (the personalization story):

  1. ``location`` arg — the model fills this from the ask itself OR
     from what it knows about the user (memory layers in her prompt);
     no transcript regex anywhere.
  2. Stored home blob — ``sources.home_location`` in the per-user
     settings store, written ONCE by any successful resolution below.
  3. The existing Settings → Location field ("Portland, OR") — already
     wired for geo-aware search; we geocode it once and promote it to
     the structured blob.
  4. Honest ask — "what city should I use?"; her reply re-enters as a
     located call, which step 1 catches and step 2 remembers.

Units are inferred from the resolved country (US → imperial) — nobody
should have to configure Fahrenheit.
"""
from __future__ import annotations

import json
from typing import Any

from augmentum.intent.action import ActionFanout, ActionResult, SessionContext
from augmentum.intent.registry import register_action
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_TIER3_ONLY = ActionFanout(tier1=False, tier2=False, tier3=True)

_HOME_KEY = "sources.home_location"


async def _load_home(store: Any, user_id: str) -> dict[str, Any] | None:
    try:
        raw = await store.get_user(user_id, _HOME_KEY)
        if raw:
            home = json.loads(raw)
            if isinstance(home, dict) and home.get("latitude") is not None:
                return home
    except Exception:  # noqa: BLE001 — unreadable blob = no home
        log.warning("weather_home_load_failed", user_id=user_id, exc_info=True)
    return None


async def _save_home(
    store: Any, user_id: str, place: dict[str, Any], *, seeded_from: str,
) -> None:
    try:
        await store.set_user(user_id, _HOME_KEY, json.dumps(place))
        log.info(
            "weather_home_saved",
            user_id=user_id, place=place.get("name", "")[:60],
            seeded_from=seeded_from,
        )
    except Exception:  # noqa: BLE001 — saving home is best-effort
        log.warning("weather_home_save_failed", user_id=user_id, exc_info=True)


async def _weather_today_handler(
    _text: str, session: SessionContext, args: dict[str, Any],
) -> ActionResult:
    if not session.user_id:
        return ActionResult(
            short_circuit=True,
            speak="I can't look up weather for a signed-out session.",
        )

    from augmentum.sources import open_meteo

    app_state = getattr(session, "app_state", None)
    store = getattr(app_state, "settings_store", None) if app_state else None

    loc_arg = (args.get("location") or "").strip()
    remember = str(args.get("remember_home", "")).strip().lower() in (
        "true", "1", "yes",
    )

    home = await _load_home(store, session.user_id) if store else None

    place: dict[str, Any] | None = None
    if loc_arg:
        place = await open_meteo.geocode(loc_arg)
        if place is None:
            return ActionResult(
                short_circuit=True,
                speak=f"I couldn't find a place called {loc_arg[:60]}.",
                # Invites a correction — park so "I meant Rochester,
                # New York" re-enters as a location fill, not a fresh
                # context-free utterance.
                clarify={"missing": ["location"], "args": {"location": ""}},
            )
        # First successful location becomes home; an explicit
        # "remember" updates it. A one-off "weather in Tokyo" when home
        # already exists does NOT clobber it.
        if store and (remember or home is None):
            await _save_home(
                store, session.user_id, place,
                seeded_from="explicit" if remember else "first_use",
            )
    elif home is not None:
        place = home
    else:
        # Settings → Location ("Portland, OR") — already user-stated,
        # already wired for geo-aware search. Promote it once. ONLY
        # the user-scoped value may become this user's home; the
        # install-wide config location is a household default that
        # answers the question without persisting (companion_eval
        # caught a fresh user inheriting the owner's city as their
        # saved home through the global fallback, 2026-06-11).
        loc_setting = ""
        if store:
            try:
                loc_setting = (
                    await store.get_user(session.user_id, "location")
                ) or ""
            except Exception:  # noqa: BLE001
                log.warning("weather_settings_location_failed", exc_info=True)
        if loc_setting:
            place = await open_meteo.geocode(loc_setting)
            if place is not None and store:
                await _save_home(
                    store, session.user_id, place, seeded_from="settings",
                )
        else:
            global_loc = ""
            try:
                from augmentum.config import settings as app_settings
                global_loc = (getattr(app_settings, "location", "") or "").strip()
            except Exception:  # noqa: BLE001
                global_loc = ""
            if global_loc:
                place = await open_meteo.geocode(global_loc)
        if place is None:
            return ActionResult(
                short_circuit=True,
                speak=(
                    "I don't have a home city for you yet — what city "
                    "should I use? I'll remember it."
                ),
                # Park the ask so the bare answer ("Rochester") fills
                # the location slot instead of facing the address gate
                # as unclassifiable noise.
                clarify={"missing": ["location"]},
            )

    imperial = str(place.get("country_code", "")).upper() == "US"
    fc = await open_meteo.forecast(
        float(place.get("latitude") or 0.0),
        float(place.get("longitude") or 0.0),
        imperial=imperial,
    )
    if fc is None:
        return ActionResult(
            short_circuit=True,
            speak=(
                f"I couldn't reach the weather service for "
                f"{place.get('name', 'your area')} just now."
            ),
        )

    summary = open_meteo.summarize(place, fc, imperial=imperial)
    data = {k: summary[k] for k in ("place", "unit", "now", "today", "tomorrow")}
    return ActionResult(
        short_circuit=True,
        speak=summary["spoken"],
        # Structured data rides the model-visible payload so follow-ups
        # ("what about tomorrow?", "should I bike?") answer from context
        # without a refetch.
        prompt_addendum=f"[weather data] {json.dumps(data)}",
    )


register_action(
    id="weather.today",
    summary=(
        "Current conditions plus today/tomorrow forecast from the "
        "weather service directly (no web search needed). Location is "
        "optional: omit it to use the user's home city. If you know "
        "where the user lives from memory or conversation, pass it as "
        "location. Set remember_home=true when the user says this is "
        "where they live or asks you to remember it."
    ),
    examples=[
        "what's the weather",
        "what's it like outside",
        "do I need a jacket today",
        "weather in portland tomorrow",
        "is it going to rain today",
        "remember that I live in Denver",
    ],
    handler=_weather_today_handler,
    # Artifact: the speak line IS the data, spoken VERBATIM. This was
    # "verbal" (synthesize over the result) until 2026-06-11, when a
    # geocode miss returned the "what city should I use?" ask and the
    # synthesis tier hallucinated "42 degrees" instead of relaying it.
    # Numbers from a data source must never pass through a model that
    # can invent numbers; follow-ups still work via prompt_addendum.
    delivery="artifact",
    arg_schema={
        "location": {
            "type": "string",
            "description": (
                "City or place name. Omit to use the user's saved home."
            ),
        },
        "remember_home": {
            "type": "string",
            "description": (
                "'true' when the user states this is their home city or "
                "asks to remember it."
            ),
        },
    },
    required=[],
    surfaces=["becca", "chat", "voice"],
    stakes="trivial_reversible",
    fanout=_TIER3_ONLY,
)
