"""``schedule_briefing`` tool — chat-LLM entrypoint for the standing-tasks
substrate.

When the user asks Becca to wake them up with a daily briefing — or any
other multi-topic, time-anchored recurring digest — the model calls this
tool. It wraps :func:`augmentum.companion_runtime.standing_tasks.add_task`
for the ``briefing`` kind with an anchored ``local_time`` so the next
occurrence is the user's wall-clock time, not a relative interval.

The tool is intentionally narrow: it only sets up briefings. Other
standing-task kinds (feed_digest, github_releases, url_watch,
recurring_search) currently surface through the topics+tasks modal UI,
where the user picks the kind explicitly. Adding chat affordances for
each kind is a per-kind decision — start with the one a user is most
likely to ask for in natural language.

See ``augmentum/companion_runtime/standing_tasks.py`` for the engine and
``augmentum/state/migrations/239_companion_standing_tasks.sql`` for the
storage shape.
"""

from __future__ import annotations

from typing import Any

from augmentum.tools._standing_common import (
    CRON_SCHEMA_PROPERTY,
    DELIVERY_SCHEMA_PROPERTY,
    parse_cron_param,
    parse_delivery_param,
    schedule_moment_error,
    standing_gate,
)
from augmentum.tools.base import (
    CoreVerbAutonomyClass,
    CoreVerbMetadata,
    CoreVerbSafetyClass,
    CostEnvelope,
    SurfaceExposure,
    Tool,
    ToolCategory,
    ToolResult,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


def _coerce_topic_list(value: Any) -> list[str]:
    """Accept either a list of strings or a comma/semicolon-separated
    string (the LLM sometimes prefers prose). Empty entries are dropped."""
    if value is None:
        return []
    if isinstance(value, list):
        return [str(t).strip() for t in value if str(t).strip()]
    if isinstance(value, str):
        parts: list[str] = []
        for chunk in value.replace(";", ",").split(","):
            chunk = chunk.strip()
            if chunk:
                parts.append(chunk)
        return parts
    return []


def _parse_local_time(value: Any) -> str | None:
    """Coerce a model-supplied local_time into canonical ``HH:MM``.

    Tiny models pass times in whatever shape they see in the user's
    request — '9', '9am', '9 pm', '9:00', '09:00', 'noon', '0900', etc.
    Rather than fail validation when the LLM forgets to format strictly,
    parse leniently and return canonical 'HH:MM'. Return None for
    genuinely unparseable input so the caller can reject.
    """
    if value is None:
        return None
    s = str(value).strip().lower()
    if not s:
        return None
    # Named times.
    if s in {"noon", "midday"}:
        return "12:00"
    if s in {"midnight", "0", "00"}:
        return "00:00"

    # Detect am/pm suffix (handles "9am", "9 am", "9 a.m.", "9pm.").
    is_pm = False
    is_am = False
    cleaned = s.replace(".", "").replace(" ", "")
    if cleaned.endswith("pm"):
        is_pm = True
        cleaned = cleaned[:-2]
    elif cleaned.endswith("am"):
        is_am = True
        cleaned = cleaned[:-2]

    hh: int | None = None
    mm: int = 0
    if ":" in cleaned:
        try:
            hh_s, mm_s = cleaned.split(":", 1)
            hh = int(hh_s)
            mm = int(mm_s[:2])
        except (ValueError, IndexError):
            return None
    elif cleaned.isdigit():
        if len(cleaned) <= 2:
            try:
                hh = int(cleaned)
            except ValueError:
                return None
        elif len(cleaned) in (3, 4):
            # "900" -> 9:00, "0900" -> 09:00, "1730" -> 17:30
            try:
                hh = int(cleaned[:-2])
                mm = int(cleaned[-2:])
            except ValueError:
                return None
        else:
            return None
    else:
        return None

    if hh is None:
        return None
    # Apply am/pm. 12am = 00, 12pm = 12, others +/- 12.
    if is_pm and hh < 12:
        hh += 12
    elif is_am and hh == 12:
        hh = 0
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        return None
    return f"{hh:02d}:{mm:02d}"


def _derive_default_title(local_time: str, topics: list[str]) -> str:
    """Synthesize a sensible briefing title when the LLM omits one.

    Small models often forget the ``title`` field — they extract topics
    + time from the user's prompt ("wake me at 9 with news") but don't
    invent a label. Rather than failing validation, derive one from
    time-of-day (matches natural English: 'Morning briefing', 'Evening
    briefing'). Falls back to '<HH:MM> briefing' on parse error.
    """
    try:
        hh = int(local_time.split(":", 1)[0])
    except (ValueError, AttributeError, IndexError):
        return "Daily briefing"
    if 5 <= hh < 11:
        return "Morning briefing"
    if 11 <= hh < 15:
        return "Midday briefing"
    if 15 <= hh < 18:
        return "Afternoon briefing"
    if 18 <= hh < 22:
        return "Evening briefing"
    return "Late briefing"


def _find_similar_briefings(
    existing: list,
    *,
    title: str,
    local_time: str,
    topics: list[str],
    location: str | None,
) -> list[dict[str, Any]]:
    """Return briefings the proposed one likely duplicates.

    A briefing is "similar" when any of:
      * title is a case-insensitive substring of an existing title (or
        vice versa) — covers "Morning briefing" vs "Morning"
      * same local_time AND >=50% topic overlap
      * same local_time AND same location

    The first signal catches near-identical titles. The latter two catch
    the "you already have a 9am thing covering this stuff" case where the
    user wouldn't realize they're double-scheduling.

    Returns up to 3 matches as plain dicts (id, title, local_time,
    topics, weekdays, location) so the caller can render them.
    """
    title_l = (title or "").strip().lower()
    requested_topics = {t.strip().lower() for t in (topics or []) if t.strip()}
    loc_l = (location or "").strip().lower()

    matches: list[tuple[int, dict[str, Any]]] = []
    for task in existing:
        if getattr(task, "kind", "") != "briefing":
            continue
        p = task.params or {}
        their_title = (task.title or "").strip().lower()
        their_time = (p.get("local_time") or "").strip()
        their_topics = {
            str(t).strip().lower() for t in (p.get("topics") or [])
            if str(t).strip()
        }
        their_loc = (p.get("location") or "").strip().lower()

        score = 0
        if title_l and their_title and (
            title_l in their_title or their_title in title_l
        ):
            score += 3
        if their_time and their_time == local_time:
            if requested_topics and their_topics:
                overlap = (
                    len(requested_topics & their_topics) / len(requested_topics)
                )
                if overlap >= 0.5:
                    score += 2
            if loc_l and their_loc and loc_l == their_loc:
                score += 2

        if score > 0:
            matches.append((score, {
                "id": task.id,
                "title": task.title,
                "local_time": p.get("local_time", ""),
                "topics": p.get("topics") or [],
                "weekdays": p.get("weekdays") or [],
                "location": p.get("location", ""),
            }))

    matches.sort(key=lambda m: m[0], reverse=True)
    return [m[1] for m in matches[:3]]


def _build_dup_review_result(
    *,
    similar: list[dict[str, Any]],
    requested: dict[str, Any],
):
    """Render the "this already exists, ask the user" soft-fail.

    Returns success=True so the LLM treats the call as resolved and
    presents the message to the user rather than retrying. Metadata
    carries the structured payload for the LLM to reason over;
    ``output`` is the conversational text.
    """
    bullets = []
    for s in similar:
        line = f"• '{s['title']}'"
        if s.get("local_time"):
            line += f" at {s['local_time']}"
        if s.get("topics"):
            line += f" covering {', '.join(s['topics'][:3])}"
        if s.get("location"):
            line += f" for {s['location']}"
        bullets.append(line)

    output = (
        "You already have a similar briefing:\n"
        + "\n".join(bullets)
        + "\n\nWant me to leave it as-is, replace it with the new one, "
        "or schedule this as an additional briefing? "
        "(Replace = cancel the existing one and create the new; "
        "additional = pass confirm_replace=true to force-create.)"
    )

    return ToolResult(
        success=True,
        output=output,
        metadata={
            "ok": True,
            "created": False,
            "reason": "similar_exists",
            "similar": similar,
            "requested": requested,
            "next_steps": {
                "leave_as_is": "Do nothing — user keeps existing briefing.",
                "replace": (
                    "Call cancel_briefing with the existing task_id, "
                    "then call schedule_briefing again with the same "
                    "params (the dup-check will pass since the existing "
                    "one is gone)."
                ),
                "additional": (
                    "Call schedule_briefing again with the same params "
                    "plus confirm_replace=true to bypass the dup check "
                    "and create a second briefing."
                ),
            },
        },
    )


_KNOWN_GATHER_TOOLS = {"searxng", "image_search", "youtube", "weather"}

# Synonym table — small models constantly emit shapes like "search",
# "google_images", "yt", and we shouldn't silently drop a meaningful
# intent because the canonical name didn't roll first. Maps lowercase
# alias → canonical tool name. Conservative on purpose: only add a
# synonym after it's been seen in the wild from a real model output.
_GATHER_TOOL_ALIASES: dict[str, str] = {
    # searxng (text search) — implicit anyway, but accept declarations
    "search": "searxng",
    "web": "searxng",
    "web_search": "searxng",
    "google": "searxng",
    "google_search": "searxng",
    # image_search
    "image": "image_search",
    "images": "image_search",
    "google_images": "image_search",
    "photo": "image_search",
    "photos": "image_search",
    "picture": "image_search",
    "pictures": "image_search",
    "pic": "image_search",
    "pics": "image_search",
    # youtube
    "yt": "youtube",
    "video": "youtube",
    "videos": "youtube",
    "youtube_search": "youtube",
    "video_search": "youtube",
    # weather (direct Open-Meteo gather — typed data, not SearXNG)
    "forecast": "weather",
    "open_meteo": "weather",
    "open-meteo": "weather",
    "weather_forecast": "weather",
}


def _normalize_gather_tools(
    value: Any,
) -> tuple[list[str], list[str], list[tuple[str, str]]]:
    """Resolve gather-tool names. Returns (canonical, dropped, aliased).

    * canonical — deduped list of canonical names safe to persist.
    * dropped — original tokens that matched nothing (model
      hallucinated a tool that doesn't exist).
    * aliased — (original_token, canonical_name) pairs for tokens
      mapped via the synonym table — surfaced so the LLM sees its
      input was interpreted, not taken verbatim.

    Empty canonical = text-only (same as omitting the field).
    ``searxng`` is implicit on the text path; listing it here is a
    no-op kept for declaration symmetry.
    """
    if value is None:
        return [], [], []
    if not isinstance(value, list):
        value = [value]
    canonical: list[str] = []
    dropped: list[str] = []
    aliased: list[tuple[str, str]] = []
    seen: set[str] = set()
    for v in value:
        original = str(v).strip()
        if not original:
            continue
        k = original.lower()
        if k in _KNOWN_GATHER_TOOLS:
            if k not in seen:
                canonical.append(k)
                seen.add(k)
            continue
        mapped = _GATHER_TOOL_ALIASES.get(k)
        if mapped is not None:
            if mapped not in seen:
                canonical.append(mapped)
                seen.add(mapped)
            aliased.append((original, mapped))
            continue
        dropped.append(original)
    return canonical, dropped, aliased


def _normalize_weekdays(value: Any) -> list[int]:
    """Accept ints, ISO ints (1=Mon..7=Sun), or short names ('mon','tue'…).
    Returns a list of canonical ISO ints. Empty list = every day."""
    if value is None:
        return []
    if not isinstance(value, list):
        value = [value]
    name_map = {
        "mon": 1, "monday": 1,
        "tue": 2, "tuesday": 2,
        "wed": 3, "wednesday": 3,
        "thu": 4, "thursday": 4,
        "fri": 5, "friday": 5,
        "sat": 6, "saturday": 6,
        "sun": 7, "sunday": 7,
    }
    out: list[int] = []
    for v in value:
        try:
            if isinstance(v, str):
                k = v.strip().lower()
                if k in name_map:
                    out.append(name_map[k])
                else:
                    n = int(k)
                    if 1 <= n <= 7:
                        out.append(n)
            elif isinstance(v, int | float):
                n = int(v)
                if 1 <= n <= 7:
                    out.append(n)
        except (ValueError, TypeError):
            continue
    return sorted(set(out))


class ScheduleBriefingTool(Tool):
    """Wire up a recurring briefing the user asked for in chat.

    The user says something like "wake me at 9 with news, weather, and
    traffic for my city" and the model calls this with parsed parameters.
    The tool creates a ``briefing`` standing task; the standing-tasks
    engine fires it at the next anchored time and every day thereafter,
    publishing the digest via the notifications hub (Web Push) and
    writing a note to the drawer.
    """

    def __init__(self, app_state: Any) -> None:
        self._app_state = app_state

    @property
    def name(self) -> str:
        return "schedule_briefing"

    @property
    def description(self) -> str:
        return (
            "Create a recurring briefing at a time of day. For 'wake "
            "me at 9 with X', 'every Monday digest Y', 'set up a "
            "daily news briefing'."
        )

    @property
    def model_hint(self) -> str:
        return (
            "Pick one_shot=true for 'remind me at X', 'tomorrow at Y', "
            "'in N hours/minutes' (fires once then deletes). Pick "
            "one_shot=false (default) for 'wake me daily', 'every "
            "morning', 'every Monday' (recurring). topics = display "
            "labels; for topics that aren't good search queries, write "
            "refined queries in search_queries (same index). Include "
            "location in queries only when locality matters (weather, "
            "traffic, news) — not for global topics (crypto, "
            "programming). local_time is HH:MM 24-hour; for cadences "
            "a time-of-day can't say ('every 2 hours', 'hourly "
            "9-5', 'the 1st of the month') pass cron instead. Pass "
            "gather_tools only for ad-hoc asks where the user implies "
            "richer media (recipes → ['image_search', 'youtube'], "
            "concerts/events → ['image_search'], product launches → "
            "['image_search']). When the user picked a named preset, "
            "DO NOT pass gather_tools — the preset declares the right "
            "tools and yours would override. Don't look up existing "
            "briefings first."
        )

    @property
    def category(self) -> ToolCategory:
        return ToolCategory.EXECUTE

    @property
    def surfaces(self) -> SurfaceExposure:
        # Chat + companion + voice (2026-07-07): the native FC loop does
        # the NL param extraction that used to block voice exposure.
        return SurfaceExposure(
            chat=True, coder=False, companion=True, flow=False,
            voice="disruptive",
            voice_capability_line="set up a recurring briefing or digest at a time of day (schedule_briefing)",
        )

    @property
    def core_verb(self) -> CoreVerbMetadata | None:
        # Phase 4 — explicit recurring scheduling; user must request it.
        return CoreVerbMetadata(
            safety_class=CoreVerbSafetyClass.WRITE_SELF,
            autonomy_class=CoreVerbAutonomyClass.EXPLICIT,
            cost_envelope=CostEnvelope(max_wallclock_ms=3_000, max_db_ops=6),
            cite_self_required=True,
        )

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def produces(self) -> list[str]:
        return ["text", "structured_data"]

    @property
    def error_hints(self) -> dict[str, str]:
        return {
            "already exists": (
                "A briefing with that title already exists. Ask the "
                "user whether to keep it, replace it (call "
                "cancel_briefing then re-schedule), or pick a "
                "different title."
            ),
            "local_time must be HH:MM": (
                "local_time must be HH:MM 24-hour (e.g. '09:00', "
                "'17:30'). Re-parse the user's request and retry."
            ),
            "topics is required": (
                "Empty topics list. Ask the user what they want covered "
                "in the briefing — at least one topic is required."
            ),
            "scheduling_disabled": (
                "The scheduling dispatcher is off in this install. "
                "Tell the user to enable scheduling_enabled (or the "
                "companion runtime) in settings."
            ),
            "standing_tasks_disabled": (
                "Standing tasks substrate is disabled. Tell the user "
                "to flip companion_standing_tasks_enabled in settings."
            ),
        }

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": (
                        "Optional short display title (e.g. 'Morning "
                        "briefing'). When omitted, the tool derives one "
                        "from the time of day ('Morning briefing' for "
                        "9am, 'Evening briefing' for 6pm, etc.). Pass "
                        "an explicit title only when the user named "
                        "the briefing."
                    ),
                },
                "topics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional list of topics to digest. Each becomes "
                        "one section. Short human-readable labels "
                        "('news', 'weather', 'Bitcoin price'). Defaults "
                        "to ['news']. Use search_queries to override the "
                        "actual search terms — topics is just the "
                        "display label."
                    ),
                },
                "search_queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional list of refined search queries, one "
                        "per topic. When provided, each query is "
                        "submitted to SearXNG verbatim instead of using "
                        "the topic label. Use when the topic label is a "
                        "natural-language phrase ('Check Bitcoin price') "
                        "that wouldn't make a good search query — pass "
                        "'Bitcoin price USD last 24 hours' as the "
                        "search_query while keeping the topic label "
                        "human-readable. Also: only include the location "
                        "in queries where it matters (weather, traffic, "
                        "news, local events) — NOT for global topics "
                        "(crypto, programming, science). Length must "
                        "match topics; missing entries fall back to the "
                        "cleaned topic."
                    ),
                },
                "local_time": {
                    "type": "string",
                    "description": (
                        "Time of day. Accepts '09:00', '17:30', '9am', "
                        "'5pm', '9 am', 'noon', 'midnight', or just '9'. "
                        "Pass whatever shape the user gave — the tool "
                        "canonicalizes to HH:MM 24-hour. Required "
                        "unless cron is set."
                    ),
                },
                "cron": CRON_SCHEMA_PROPERTY,
                "delivery": DELIVERY_SCHEMA_PROPERTY,
                "weekdays": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional. Restrict to specific weekdays. Accepts "
                        "short names ('mon', 'tue', ...) or ISO ints "
                        "(1=Mon, 7=Sun). Empty / omitted = every day. "
                        "Example: ['mon', 'tue', 'wed', 'thu', 'fri'] "
                        "for weekday-only."
                    ),
                },
                "location": {
                    "type": "string",
                    "description": (
                        "Optional. Appended to each topic's search query "
                        "to make results location-relevant. Pass a "
                        "city/region the user mentioned (e.g. '<city, "
                        "region>') so 'traffic' becomes 'traffic <city, "
                        "region>'."
                    ),
                },
                "confirm_replace": {
                    "type": "boolean",
                    "description": (
                        "Optional. Set to true ONLY after the user has "
                        "explicitly confirmed they want to create a new "
                        "briefing even though a similar one exists. "
                        "Bypasses the duplicate check. Default false."
                    ),
                },
                "gather_tools": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional. Extra gather phases to run at fire "
                        "time, beyond the default text search. ONLY two "
                        "values are accepted: 'image_search' (adds a "
                        "hero image) and 'youtube' (adds a video + "
                        "transcript). 'searxng' is always implicit — "
                        "you do not need to list it. Common synonyms "
                        "('images', 'pics', 'yt', 'video', 'videos') "
                        "are auto-mapped to the canonical names, but "
                        "prefer the canonical names directly. Generic "
                        "names like 'search' / 'web_search' / 'google' "
                        "are mapped to the implicit text path (a no-op). "
                        "Use ONLY for ad-hoc requests where the user's "
                        "intent implies media (recipe, concert, product "
                        "launch, travel, hobby projects). When the user "
                        "is picking a named preset, OMIT this field — "
                        "presets carry their own curated gather list. "
                        "If you pass an unknown name the tool reports it "
                        "back so you can correct on a follow-up call."
                    ),
                },
                "one_shot": {
                    "type": "boolean",
                    "description": (
                        "Optional. Set true when the user wants a "
                        "ONE-TIME delivery — 'remind me at 5pm', "
                        "'tomorrow at 9 tell me X', 'in 2 hours give "
                        "me Y'. The briefing fires ONCE at the next "
                        "scheduled time and then deletes itself. "
                        "Default false (recurring). Set false for "
                        "'wake me daily', 'every Monday', 'every "
                        "morning' patterns."
                    ),
                },
                "read_aloud": {
                    "type": "boolean",
                    "description": (
                        "Optional. Set true when the user wants the "
                        "briefing SPOKEN — 'read me my morning briefing', "
                        "'tell me out loud', 'narrate it'. The result "
                        "note then auto-plays through the user's TTS "
                        "voice when they open it. Default false (silent; "
                        "they read it themselves)."
                    ),
                },
            },
            # Either local_time or cron carries the schedule; enforcing
            # local_time here would force the model to invent one even
            # for "every 2 hours" asks. Validated in execute().
            "required": [],
        }

    async def execute(self, **kwargs) -> ToolResult:
        ok, gate_err, runtime = standing_gate(self._app_state)
        if not ok:
            return gate_err

        title = str(kwargs.get("title") or "").strip()
        topics = _coerce_topic_list(kwargs.get("topics"))
        search_queries = _coerce_topic_list(kwargs.get("search_queries"))
        weekdays = _normalize_weekdays(kwargs.get("weekdays"))
        location = str(kwargs.get("location") or "").strip() or None
        one_shot = bool(kwargs.get("one_shot"))
        gather_tools, gather_dropped, gather_aliased = _normalize_gather_tools(
            kwargs.get("gather_tools"),
        )
        if gather_dropped or gather_aliased:
            log.info(
                "schedule_briefing_gather_tools_normalized",
                accepted=gather_tools,
                dropped=gather_dropped[:8],
                aliased=[
                    {"from": a, "to": b}
                    for a, b in gather_aliased[:8]
                ],
            )

        cron_expr, cron_err = parse_cron_param(kwargs.get("cron"))
        if cron_err:
            return ToolResult(
                success=False, error=cron_err, validation_error=True,
            )
        delivery, delivery_err = parse_delivery_param(kwargs.get("delivery"))
        if delivery_err:
            return ToolResult(
                success=False, error=delivery_err, validation_error=True,
            )
        local_time = _parse_local_time(kwargs.get("local_time"))
        if local_time is None and cron_expr is None:
            return ToolResult(
                success=False,
                error=schedule_moment_error(),
                validation_error=True,
            )

        # Small models routinely drop fields they did get from the user.
        # Rather than fail validation, fill in sensible defaults and flag
        # the assumptions in the response so the LLM can confirm with the
        # user. "news" is the universal briefing default — applies at any
        # time of day and matches the most common ask.
        defaulted: list[str] = []
        if not topics:
            topics = ["news"]
            defaulted.append("topics")
        if not title:
            title = _derive_default_title(local_time, topics)
            defaulted.append("title")

        user_id = Tool.extract_user_id(kwargs)
        if not user_id:
            return ToolResult(
                success=False,
                error="user_id missing — briefings are per-user",
                metadata={"ok": False, "reason": "missing_user"},
            )

        params: dict[str, Any] = {
            "topics": topics,
            "title": title,
        }
        if local_time is not None:
            params["local_time"] = local_time
        if cron_expr:
            # Engine precedence: cron > local_time. Keep a co-supplied
            # local_time as the fallback rung, not a competitor.
            params["cron"] = cron_expr
        if weekdays:
            params["weekdays"] = weekdays
        if location:
            params["location"] = location
        if search_queries:
            # When the LLM provided refined queries, persist them on the
            # task so every future fire uses them. Length mismatch is
            # handled at fire time (missing entries fall back to the
            # cleaned topic) — no normalization needed here.
            params["search_queries"] = search_queries
        if one_shot:
            # Flag is what standing_tasks.step() reads to delete the row
            # after a successful fire. Only persisted when True so
            # legacy briefings (no flag at all) stay recurring.
            params["one_shot"] = True
        if gather_tools:
            # Persisted only when the model opted into extras. Preset
            # path (task #32) writes its own gather_tools and takes
            # precedence — see preset_id handling.
            params["gather_tools"] = gather_tools
        if delivery:
            # Explicit per-task delivery choice (alert|quiet) — read by
            # _surface_importance at fire time. Only persisted when the
            # user chose, so kind defaults keep applying otherwise.
            params["delivery"] = delivery
        if bool(kwargs.get("read_aloud")):
            # Per-briefing spoken-delivery toggle — read by _surface_result
            # (→ note origin) so the drawer auto-narrates on open. Persisted
            # only when True so silent briefings carry no flag.
            params["read_aloud"] = True

        from augmentum.companion_runtime import standing_tasks

        # Implicit dup-check: list the user's current briefings and bail
        # out with a "do you want to manage the existing one?" payload if
        # we'd be creating something the user already has.
        #
        # Two cases bypass the check entirely:
        # 1. ``confirm_replace=true`` — user explicitly said yes after a
        #    prior dup-review response.
        # 2. ``one_shot=true`` — a one-time check doesn't compete with a
        #    recurring schedule. "Give me a fresh bitcoin lookup right
        #    now" shouldn't be blocked by "check bitcoin every Thursday"
        #    — they're different temporal needs.
        #
        # When checking, we also exclude delivered one-shots (a one-shot
        # that already fired is done; nothing competes with it).
        confirm_replace = bool(kwargs.get("confirm_replace"))
        if not confirm_replace and not one_shot:
            existing_all = await standing_tasks.list_tasks(
                runtime.backend.conn,
                user_id=user_id, companion_id=runtime.companion_id,
            )
            existing_live = [
                t for t in existing_all
                if not (t.params or {}).get("delivered_at")
            ]
            similar = _find_similar_briefings(
                existing_live,
                title=title, local_time=local_time, topics=topics,
                location=location,
            )
            if similar:
                return _build_dup_review_result(
                    similar=similar,
                    requested={
                        "title": title, "local_time": local_time,
                        "topics": topics, "weekdays": weekdays,
                        "location": location,
                    },
                )

        user_tz = await standing_tasks._resolve_user_timezone(
            self._app_state, user_id,
        )
        try:
            task = await standing_tasks.add_task(
                runtime.backend.conn,
                user_id=user_id, companion_id=runtime.companion_id,
                title=title, kind="briefing", params=params,
                # Anchored schedules don't really use interval_seconds for
                # the first run, but it's the auto-fallback if local_time
                # parsing fails on a later run for any reason. 24h is the
                # right default for daily briefings.
                interval_seconds=86400,
                user_timezone=user_tz,
            )
        except ValueError as exc:
            return ToolResult(
                success=False,
                error=str(exc),
                validation_error=True,
            )
        except Exception as exc:
            log.warning("schedule_briefing_failed", error=str(exc)[:200])
            return ToolResult(
                success=False,
                error="failed to create briefing",
                metadata={"ok": False, "reason": "internal"},
            )

        if task is None:
            return ToolResult(
                success=False,
                error=f"briefing with title '{title}' already exists",
                metadata={"ok": False, "reason": "duplicate"},
            )

        weekday_words = ""
        if weekdays:
            names = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu",
                     5: "Fri", 6: "Sat", 7: "Sun"}
            weekday_words = " on " + "/".join(names[w] for w in weekdays)
        loc_words = f" for {location}" if location else ""
        if one_shot:
            cadence = "fires once"
            occurrence_label = "Fires at"
        else:
            cadence = "runs"
            occurrence_label = "First briefing"
        if cron_expr:
            from augmentum.utils.cron import describe
            when_words = f"on schedule '{describe(cron_expr)}'"
            weekday_words = ""  # cron owns the day pattern
        else:
            when_words = f"at {local_time}"
        summary = (
            f"Set up '{title}' — {cadence} {when_words}"
            f"{weekday_words}{loc_words}, "
            f"covering {len(topics)} topic{'s' if len(topics) != 1 else ''} "
            f"({', '.join(topics)}). "
            f"{occurrence_label}: {task.next_run_at} UTC."
        )
        if one_shot:
            summary += " (One-time — deletes itself after firing.)"
        if defaulted:
            summary += (
                f" (Assumed default {' and '.join(defaulted)} — "
                f"tell the user what was assumed and ask if they want "
                f"to adjust.)"
            )
        if gather_aliased:
            pairs = ", ".join(f"'{a}' → {b}" for a, b in gather_aliased)
            summary += f" (Mapped gather_tools: {pairs}.)"
        if gather_dropped:
            summary += (
                f" (Ignored unknown gather_tools "
                f"{gather_dropped} — allowed values are "
                f"'image_search' and 'youtube'. 'searxng' is implicit.)"
            )

        return ToolResult(
            success=True,
            output=summary,
            metadata={
                "ok": True,
                "task_id": task.id,
                "title": task.title,
                "local_time": local_time,
                "topics": topics,
                "weekdays": weekdays,
                "location": location,
                "next_run_at": task.next_run_at,
                "defaulted": defaulted,
                "one_shot": one_shot,
                "gather_tools": gather_tools,
                "gather_tools_dropped": gather_dropped,
                "gather_tools_aliased": [
                    {"from": a, "to": b} for a, b in gather_aliased
                ],
            },
        )
