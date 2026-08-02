"""Smart tool pre-filtering — reduce tool count for better LLM accuracy.

Each extra tool in the prompt degrades tool-calling accuracy by 3-5%
(per BFCL benchmarks).  This module analyses the query and returns
only the tools likely to be relevant.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from augmentum.tools.base import Tool

# Pattern → tool names that should be included when the pattern matches.
# The "web" unified tool is included alongside web_search/web_fetch for
# passthrough mode which registers the unified tool.
_QUERY_TOOL_PATTERNS: list[tuple[re.Pattern, set[str]]] = [
    # Web search / current info — explicit search intent
    (re.compile(
        r"(search|find|look\s*up|latest|current|today|news|recent|"
        r"who\s+is|what\s+happened|trending|update)",
        re.IGNORECASE,
    ), {"web", "web_search", "web_fetch"}),

    # General knowledge questions — these are search queries, not app requests.
    # Catches: "what's the weather", "how tall is", "who won", "where is", etc.
    (re.compile(
        r"(weather|forecast|temperature|humidity|"
        r"what(?:'s| is| are| was| were)\s+(?:the|a|an)?\s*\w|"
        r"how\s+(?:tall|big|far|long|much|many|old|fast|hot|cold|deep)|"
        r"who\s+(?:won|is|are|was|were|invented|discovered|founded|created)|"
        r"where\s+(?:is|are|was|were|can|do)|"
        r"when\s+(?:is|was|did|does|will)|"
        r"why\s+(?:is|are|was|were|did|does|do)|"
        r"price\s+of|cost\s+of|score|result|"
        r"tell\s+me\s+about|explain|what\s+does)",
        re.IGNORECASE,
    ), {"web", "web_search", "web_fetch"}),

    # URLs in the query
    (re.compile(r"https?://", re.IGNORECASE), {"web", "web_fetch"}),

    # YouTube
    (re.compile(
        r"(youtube\.com|youtu\.be|video\s+transcript|caption|subtitle|"
        r"watch\?v=|youtube\s+video)",
        re.IGNORECASE,
    ), {"youtube", "web"}),

    # Wikipedia / encyclopedic
    (re.compile(
        r"(wikipedia|wiki\s|encyclopedi|who\s+was|history\s+of|"
        r"biography|definition\s+of)",
        re.IGNORECASE,
    ), {"wikipedia", "web"}),

    # Math / calculation
    (re.compile(
        r"(calcul|comput|solve|equation|formula|integral|derivative|"
        r"sum\s+of|product\s+of|factorial|\d+\s*[\+\-\*/\^]\s*\d+|"
        r"math|algebra|trigonometr|logarithm|sqrt|square\s+root)",
        re.IGNORECASE,
    ), {"calculator", "math_verify", "python_exec"}),

    # Code / programming
    (re.compile(
        r"(code|script|program|python|javascript|function|algorithm|"
        r"implement|execute|run|compile|debug|regex|parse|sort)",
        re.IGNORECASE,
    ), {"python_exec"}),

    # Memory / personal
    (re.compile(
        r"(remember|recall|my\s+(name|preference|favorite)|"
        r"i\s+told\s+you|you\s+know\s+me|last\s+time)",
        re.IGNORECASE,
    ), {"memory_recall"}),

    # Image generation
    (re.compile(
        r"(draw|generate\s+(an?\s+)?image|picture\s+of|illustrat|"
        r"create\s+(an?\s+)?image|visuali[zs]e|render|sketch|"
        r"make\s+(an?\s+)?(image|picture|photo))",
        re.IGNORECASE,
    ), {"image_generation"}),

    # Image search (finding existing images)
    (re.compile(
        r"(find\s+(an?\s+)?image|image\s+of|photo\s+of|"
        r"search\s+for\s+(an?\s+)?(image|picture|photo))",
        re.IGNORECASE,
    ), {"image_search"}),

    # Date / time
    (re.compile(
        r"(what\s+(time|day|date)|timezone|current\s+date|"
        r"how\s+many\s+days|day\s+of\s+the\s+week)",
        re.IGNORECASE,
    ), {"datetime"}),

    # Standing briefings — schedule. Surface the lifecycle trio
    # together so the LLM has list/cancel in view alongside schedule.
    (re.compile(
        r"(wake\s+me|"
        r"(daily|morning|evening|weekly|nightly)\s+(briefing|digest|reminder|update)|"
        r"(briefing|digest|reminder)\s+(at|every|on)\b|"
        r"schedule\s+(me\s+|a\s+|the\s+)?(daily|morning|evening|weekly|briefing|digest|reminder|wake|recurring)|"
        r"set\s+up\s+(a\s+)?(briefing|digest|reminder|recurring|wake|standing)|"
        r"remind\s+me\b|"
        r"every\s+(morning|night|evening|afternoon|weekday|weekend|day|"
        r"monday|tuesday|wednesday|thursday|friday|saturday|sunday)|"
        # "at 5pm weekdays" / "at 9 every day" / "at 17:00 with news"
        r"at\s+\d{1,2}(:\d{2})?\s*(am|pm)?\s+"
        r"(every|each|with|give|tell|wake|send|bring|weekday|weekdays|daily))",
        re.IGNORECASE,
    ), {"schedule_briefing", "schedule_request", "list_briefings",
        "cancel_briefing"}),

    # Watches — "tell me when X changes / releases / drops below Y."
    (re.compile(
        r"((keep\s+an?\s+eye\s+on)|"
        r"(watch|monitor|track)\s+(this|that|the|a)\b|"
        r"(tell|notify|alert|let)\s+(me\s+)?(know\s+)?"
        r"(when|if)\s+(it|this|that|the|there)|"
        r"(price|stock|temperature|new\s+release).{0,24}"
        r"(drops?|falls?|goes|rises?|below|above|changes?))",
        re.IGNORECASE,
    ), {"watch_for", "list_briefings", "cancel_briefing"}),

    # Standing briefings — list / inspect.
    (re.compile(
        r"((my|any|current|scheduled)\s+briefings?|"
        r"list\s+briefings?|"
        r"what\s+briefings?|"
        r"(show|see|view)\s+(me\s+)?(my\s+)?briefings?|"
        r"when\s+(is|are)\s+(my\s+)?(next\s+)?briefing|"
        r"how\s+many\s+briefings?)",
        re.IGNORECASE,
    ), {"list_briefings", "cancel_briefing", "schedule_briefing"}),

    # Standing briefings — cancel / pause. Allow an optional
    # qualifier word between article and keyword ("the news briefing",
    # "my evening digest", "that morning reminder").
    (re.compile(
        r"(cancel|delete|stop|remove|disable|drop|"
        r"unsubscribe\s+from|unschedule)\s+"
        r"(my\s+|the\s+|that\s+)?(\w+\s+)?"
        r"(briefing|digest|reminder|wake[-_]?up|standing|recurring)",
        re.IGNORECASE,
    ), {"cancel_briefing", "list_briefings"}),

    # Deadline countdowns — "taxes are due April 15", "my passport
    # renews in March", "count me down to the launch". Distinct verbs
    # from the reminder patterns above: due/renew/expire/deadline talk
    # is about a DATE with lead-times, not a time-of-day.
    (re.compile(
        r"((is|are|it.s)\s+due\b|due\s+(on|by|in|date)\b|"
        r"deadline|expires?\b|expiring|renew(s|al)?\b|"
        r"count(?:\s+me)?\s*down|days\s+(left|until|before)|"
        r"nudge\s+me\s+(at|before|leading))",
        re.IGNORECASE,
    ), {"schedule_deadline", "list_briefings", "cancel_briefing"}),

    # Scheduled app actions — "at 5pm check the weather and tell me",
    # "every weekday at 9 open my notes", "pause the music at midnight".
    # A verb fire, not a digest: the wall-clock + an app action.
    (re.compile(
        r"(at\s+(\d{1,2}(:\d{2})?\s*(am|pm)?|noon|midnight)\s*,?\s+"
        r"(check|open|play|pause|stop|start|turn|run|launch)|"
        r"(every|each)\s+\w+\s+at\s+\d{1,2}(:\d{2})?\s*(am|pm)?\s+"
        r"(check|open|play|pause|stop|start|turn|run|launch))",
        re.IGNORECASE,
    ), {"schedule_action", "schedule_request", "list_briefings",
        "cancel_briefing"}),

    # Unit conversion
    (re.compile(
        r"(convert|celsius|fahrenheit|miles|kilometers|pounds|kilograms|"
        r"gallons|liters|inches|centimeters|ounces|grams)",
        re.IGNORECASE,
    ), {"unit_converter"}),

    # Ebook / storybook
    (re.compile(
        r"(ebook|e-book|storybook|children.s\s+book|epub|"
        r"write\s+(a\s+)?(story|book|tale)|illustrated\s+book)",
        re.IGNORECASE,
    ), {"create_ebook", "image_generation"}),

    # Documents / artifacts
    (re.compile(
        r"(create\s+(a\s+)?(document|pdf|report|docx)|write\s+(a\s+)?report)",
        re.IGNORECASE,
    ), {"create_document"}),
    (re.compile(
        r"(presentation|slides|pptx|powerpoint|create\s+(a\s+)?presentation)",
        re.IGNORECASE,
    ), {"create_presentation"}),
    (re.compile(
        r"(spreadsheet|xlsx|excel|create\s+(a\s+)?spreadsheet|tabulate)",
        re.IGNORECASE,
    ), {"create_spreadsheet"}),
    (re.compile(
        r"(chart|graph|plot|visuali[zs]e\s+data|bar\s+chart|pie\s+chart)",
        re.IGNORECASE,
    ), {"create_chart"}),

    # Build / app
    (re.compile(
        r"(build\s+(me\s+)?(a\s+)?(app|application|website|web\s+app|tool|game|calculator|dashboard)|"
        r"create\s+(a\s+)?(web|app|site|page)|make\s+(me\s+)?(a\s+)?(app|site|game))",
        re.IGNORECASE,
    ), {"build_application"}),

    # Export
    (re.compile(
        r"(export|save\s+as|download\s+as|\.md\b|\.csv\b|markdown\s+file)",
        re.IGNORECASE,
    ), {"export_markdown", "export_csv", "export_code"}),

    # File operations
    (re.compile(
        r"(read\s+file|write\s+file|list\s+files|file\s+contents|"
        r"save\s+to\s+file|open\s+file)",
        re.IGNORECASE,
    ), {"file_ops"}),
]

# Tools the model must ALWAYS be able to choose — exempt from the regex
# relevance filter. Image intent is phrased too many ways for a keyword regex
# to catch reliably ("make me a picture of…", "whip up a logo", "I'd love to
# see…"), and the failure mode is the worst kind: the schema gets dropped, the
# model can't see the tool, so it DENIES the capability and falls back to web
# search. Tier-3 native function-calling is the right arbiter — present the
# tool every turn and let the model decide whether to call it. This only forces
# inclusion of tools already present in the toolset (an unregistered/disabled
# image backend is NOT conjured), and the safe-default fallback below can no
# longer strip them since matched_names is never empty when one is available.
_ALWAYS_INCLUDE: frozenset[str] = frozenset({"image_generation", "image_search"})

# Scheduling/watch substrate. Force-available on EVERY passthrough turn
# whenever a scheduling dispatcher exists (companion runtime OR
# SchedulerService — see handler_factory): the model is the arbiter of
# whether to schedule, not a keyword regex (policy change 2026-07-02 —
# keyword gating made paraphrases outside the pattern vocabulary
# silently unschedulable). The patterns above still serve analytical's
# relevance filter; this set marks which names belong to the substrate
# so handler_factory can order/dedupe the injection.
_SCHEDULE_SUBSTRATE_TOOLS: frozenset[str] = frozenset({
    "schedule_briefing", "schedule_request", "watch_for",
    "schedule_deadline", "schedule_action",
    "list_briefings", "cancel_briefing",
})


# Injection priority: create-verbs first — the prompt lists tools
# sequentially and tiny models bias toward earlier items, so the most
# common intents (schedule/watch) must precede list/cancel.
_SCHEDULE_INJECTION_ORDER: tuple[str, ...] = (
    "schedule_briefing", "schedule_request", "watch_for",
    "schedule_deadline", "schedule_action",
    "list_briefings", "cancel_briefing",
)


def query_wants_schedule_tools(query: str) -> bool:
    """True when *query* matches a scheduling / watch intent.

    Same relevance signal :func:`filter_tools_for_query` uses (a pattern whose
    target tools intersect the substrate set), applied before injection so the
    briefing/watch tools surface only on messages that actually ask for them.
    """
    return bool(schedule_tools_for_query(query))


def schedule_tools_for_query(query: str | None) -> list[str]:
    """Substrate tools the message's matched patterns name, in injection
    order. ``None`` (caller didn't wire the message, e.g. voice) returns
    the full substrate — the legacy always-on set. Empty list = the
    message didn't ask for scheduling at all."""
    if query is None:
        return list(_SCHEDULE_INJECTION_ORDER)
    if not query:
        return []
    matched: set[str] = set()
    for pattern, tool_names in _QUERY_TOOL_PATTERNS:
        hit = tool_names & _SCHEDULE_SUBSTRATE_TOOLS
        if hit and pattern.search(query):
            matched.update(hit)
    return [n for n in _SCHEDULE_INJECTION_ORDER if n in matched]


def filter_tools_for_query(
    query: str,
    tools: list[Tool],
    min_tools: int = 3,
    max_tools: int = 8,
) -> list[Tool]:
    """Return a subset of *tools* relevant to *query*.

    If no patterns match, returns all tools (safe fallback).
    Always returns at least *min_tools* and at most *max_tools* tools.
    """
    if not query or not tools:
        return tools

    # Did the MESSAGE itself signal any tool? (always-include tools must not
    # count here — they ride along regardless, so letting them seed this would
    # wrongly suppress the "no signal → keep the general set" fallback below.)
    query_matched: set[str] = set()
    for pattern, tool_names in _QUERY_TOOL_PATTERNS:
        if pattern.search(query):
            query_matched.update(tool_names)

    # Always-include capabilities (image gen/search) ride along whenever they're
    # present — the model is the arbiter of whether to call them. The schedule
    # substrate rides the same way (2026-07-02 policy, injection landed
    # 2026-07-07): regex stays the fast-path, never the gate — natural
    # phrasings the patterns miss ("set a tracker", "every 15 minutes") must
    # still leave the model able to infer a scheduling call.
    matched_names = query_matched | _ALWAYS_INCLUDE | _SCHEDULE_SUBSTRATE_TOOLS

    if not query_matched:
        # No specific signals. If the tool set is large (user has "all" enabled),
        # default to web search + a few safe tools rather than dumping 12+ schemas
        # that confuse the LLM. If the tool set is small (user curated), keep all.
        # Either way an available always-include tool is never stripped, so the
        # model can still choose to make an image on an unsignalled turn.
        if len(tools) > 6:
            # Default safe set — web search covers most general queries
            _safe_defaults = {"web", "web_search", "web_fetch", "wikipedia", "python_exec"}
            filtered = [
                t for t in tools
                if t.name in _safe_defaults
                or t.name in _ALWAYS_INCLUDE
                or t.name in _SCHEDULE_SUBSTRATE_TOOLS
            ]
            if filtered:
                return filtered
        return tools

    # Build filtered list, preserving order
    filtered = [t for t in tools if t.name in matched_names]

    # Ensure minimum tool count — pad with remaining tools by original order
    if len(filtered) < min_tools:
        remaining = [t for t in tools if t.name not in matched_names]
        filtered.extend(remaining[: min_tools - len(filtered)])

    # Cap at max to keep schema noise down. Ride-alongs (always-include +
    # the schedule substrate) are exempt from eviction — a blind
    # ``[:max_tools]`` slice used to silently drop them whenever the
    # matched set ran long, which un-did the always-available guarantee
    # one tool at a time. General tools compete for the remaining budget,
    # original order preserved (create-verbs early for small models).
    if len(filtered) > max_tools:
        protected = {
            t.name for t in filtered
            if t.name in _ALWAYS_INCLUDE or t.name in _SCHEDULE_SUBSTRATE_TOOLS
        }
        budget = max(0, max_tools - len(protected))
        kept: list[Tool] = []
        used = 0
        for t in filtered:
            if t.name in protected:
                kept.append(t)
            elif used < budget:
                kept.append(t)
                used += 1
        filtered = kept

    return filtered
