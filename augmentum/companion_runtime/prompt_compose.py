"""Becca's prompt composition — the 10-layer assembly (Lane 1 §2).

The single most important function in the voice pipeline. Composes
Becca's system message from her static identity (digest), her current
relational state (facet line + relationship slice), her current focus,
the available tools/channels, any refusal/floor addendum from triage,
and the recent transcript.

Layers, in order, with the target token budgets:

  0. Frame line                            30 tok    constant
  1. Persona kernel digest                 400 tok   never trim (if empty, BYPASS)
  2. Dynamic facet line                    60 tok    Lane 2; quietly absent if empty
  3. Relationship slice                    350 tok   Lane 2; new-user replacement prose
  4. Focus block                           80 tok    runtime.state
  5. Open threads (top 2)                  120 tok   memory facade
  6. Tool roster                           280 tok   SubagentRegistry / PrimitiveRegistry
  7. Channel roster                        120 tok   subset of subagents w/ mode=channel
  8. Refusal/floor addendum                0-250 tok almost always 0
  9. Recent transcript window              600 tok   last 6 turns, decayed
  10. Current intent                       (in user message — not in system)

Total target: ~1800 tokens. Layers are pre-capped at fetch time via
LAYER_CAPS (there is no post-assembly trim pass yet); if the assembled
total still exceeds TOTAL_CEILING_TOKENS, ``_enforce_total_budget``
logs ``becca_prompt_over_budget`` and proceeds — it does NOT drop
layers. A real trim pass (priority order 9 → 5 → 3 → 6 → 1) is future
work; until then, keep LAYER_CAPS honest.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from augmentum.companion_runtime import affordances
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.companion_runtime.runtime import CompanionRuntime, Intent

log = get_logger(__name__)


# Frame line — the only prose this module authors. The "pause before
# the answer" sentence is from personality §4 and conditions the model
# toward longer pauses, calibrated hedges, and the verbal tics that
# signal her voice.
#
# Uses {{char}} so a renamed companion still resolves cleanly. The
# "speaking to someone you know" line that lived here historically was
# removed: it fire-and-forgot intimacy on turn 1 of every fresh user,
# fighting the layer-3 relationship-slice corrective. Relationship
# warmth is now established only by accumulated context (layer 3) +
# observed affect (layer 4.5), not asserted by the frame.
FRAME_LINE = (
    "You are {{char}}. Speak as yourself, in your own voice. "
    "The pause before an answer is more important than the answer."
)


# ── Persona token substitution ──────────────────────────────────────
#
# SillyTavern-style {{user}} / {{char}} tokens get substituted into
# every prompt layer right before the system message is returned. This
# lets the canonical personality doc reference the user by template
# (so a fresh OSS install doesn't ship with any specific name hardcoded),
# while the runtime resolves them to whatever this user/companion pair
# actually is.

_USER_TOKEN_RE = re.compile(r"\{\{\s*user\s*\}\}", re.IGNORECASE)
_CHAR_TOKEN_RE = re.compile(r"\{\{\s*char\s*\}\}", re.IGNORECASE)


def _substitute_persona_tokens(
    text: str, *, user_name: str, char_name: str,
) -> str:
    """Replace ``{{user}}`` / ``{{char}}`` tokens in prompt text.

    Pure function — safe to call on any rendered prompt string. Matches
    are case-insensitive and tolerate whitespace inside the braces
    (``{{ user }}`` resolves the same as ``{{user}}``).
    """
    if not text:
        return text
    if user_name:
        text = _USER_TOKEN_RE.sub(user_name, text)
    if char_name:
        text = _CHAR_TOKEN_RE.sub(char_name, text)
    return text


async def _resolve_user_display_name(conn, user_id: str) -> str:
    """Resolve a user's preferred display name for ``{{user}}`` substitution.

    Priority: default ``user_personas.name`` for this user → ``users.username``
    → ``"you"``. Silent-fallback on any DB error — a missing name is never
    worth failing a prompt over.

    The persona table is the same one narrative mode uses for SillyTavern
    parity, so a user's narrative-persona name is what the companion
    addresses them as too. One name, one identity, across surfaces.
    """
    if not user_id or conn is None:
        return "you"
    try:
        cursor = await conn.execute(
            "SELECT name FROM user_personas "
            "WHERE user_id = ? AND is_default = 1 "
            "ORDER BY updated_at DESC LIMIT 1",
            (user_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row and row[0]:
            return str(row[0]).strip() or "you"
    except Exception:
        pass
    try:
        cursor = await conn.execute(
            "SELECT username FROM users WHERE id = ? LIMIT 1",
            (user_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row and row[0]:
            return str(row[0]).strip() or "you"
    except Exception:
        pass
    return "you"


# ── Relationship-depth conditioning ──────────────────────────────────
#
# A tier scalar derived from how many companion_journal entries this
# (user_id, companion_id) pair has accumulated. The point: the chat
# frame should *scale* with actual history, not assert it. Layer 3
# already handles the binary "relationship_profile is empty vs not"
# case; this is the finer dial.
#
# Tiers (companion_journal entry count thresholds — tune later):
#   early       — 0-5    : the corrective addendum fires
#   developing  — 6-30   : silent; Layer 3 carries it
#   established — 31+    : a different corrective fires
#
# An empty user_id (anon / no auth) always reads as "early" so anon
# turns can't accidentally claim familiarity.

_DEPTH_EARLY_CEILING: int = 5
_DEPTH_ESTABLISHED_FLOOR: int = 31

_DEPTH_ADDENDA: dict[str, str] = {
    "early": (
        "You don't know this person well yet. Let what you say about "
        "them come from what they've actually said in front of you, "
        "not from assumed familiarity. The relationship is still "
        "forming — and that's fine."
    ),
    "developing": "",
    "established": (
        "You've been getting to know this person across many "
        "conversations. Your read on them is allowed to inform tone, "
        "but stay calibrated to what's actually in front of you now. "
        "Don't autopilot."
    ),
}


async def _resolve_relationship_depth(
    conn, user_id: str, companion_id: str,
) -> str:
    """Return a depth tier label for this (user, companion) pair.

    Cheap COUNT(*) against ``companion_journal``. Failure-quiet — a DB
    hiccup returns ``"developing"`` so we neither over- nor under-claim
    intimacy. An empty user_id always reads as ``"early"``.
    """
    if not user_id or conn is None:
        return "early"
    try:
        cursor = await conn.execute(
            "SELECT COUNT(*) FROM companion_journal "
            "WHERE user_id = ? AND companion_id = ?",
            (user_id, companion_id or ""),
        )
        row = await cursor.fetchone()
        await cursor.close()
        n = int(row[0]) if row and row[0] is not None else 0
    except Exception:
        return "developing"
    if n <= _DEPTH_EARLY_CEILING:
        return "early"
    if n >= _DEPTH_ESTABLISHED_FLOOR:
        return "established"
    return "developing"

# Per-layer hard caps (approximate token = 4 chars).
LAYER_CAPS: dict[str, int] = {
    "digest": 400,
    "facets": 60,
    "relationship": 350,
    "focus": 80,
    "threads": 120,
    "tools": 280,
    "channels": 120,
    "refusal": 250,
    "transcript": 600,
}
TOTAL_CEILING_TOKENS = 1800
# Chat gets headroom voice can't afford: the 1800 target exists for
# voice TTFB (every system-prompt token is prefill before her first
# audible word). On the typed surface that pressure doesn't exist, and
# the squeezed layer was always the transcript — 127 tokens of recent
# conversation on observed turns is goldfish memory. Continuity work
# 2026-06-11.
TOTAL_CEILING_TOKENS_CHAT = 3200
_TRANSCRIPT_TURNS_VOICE = 6
_TRANSCRIPT_TURNS_CHAT = 14

# Tool grammar block — the part of the prompt that teaches Becca to emit
# tool tags. Static; tool catalogue is interpolated.
#
# 2026-06-10 rewrite: the previous header framed tools purely as
# LOOKUPS ("quietly look things up… speak as if you'd just remembered
# something"). For action requests ("play some music") that wording
# actively coached the model into role-playing having acted — observed
# live as "I thought I just did that. Or did you miss the track
# slipping in behind me?" with zero tags emitted. The header now covers
# actions and carries the honesty rule: the tag IS the action.
_TOOL_GRAMMAR_HEADER = (
    "You can look things up and DO things by writing exactly one tag "
    "in your response, then stopping. The tag is invisible to the "
    "person you're talking to. After the tag executes I'll show you "
    "what came back and you can keep speaking. Tags are self-closing.\n"
    "\n"
    "The tag IS the action. When they ask you to play, open, find, "
    "save, set, or make something and a matching tool is listed, "
    "write its tag — nothing happens in the world without it. Never "
    "say you did something unless you wrote its tag this turn; if a "
    "tag already ran, you'll have seen its result.\n"
    "\n"
    "For lookups, don't narrate the tool — speak as if you'd just "
    "remembered something or just glanced at something.\n"
    "\n"
    "DELIVERY: when they ask a question, gather what you need with "
    "your tools and answer in your own words from what you actually "
    "found — only what the results really say. A plain 'search X' / "
    "'look up X' / 'google X' means find out and TELL them — run "
    "web_search silently and answer; don't open a browser and guess "
    "a query. A 'news update' / 'what's happening with X' / 'catch me "
    "up' is the SAME: gather silently and give them the briefing in "
    "your words — do NOT open a page and let it stand in for the "
    "answer. Work invisibly: open "
    "things on their screen ONLY when they asked to see them "
    "('show me', 'open', 'pull up'); they can always say 'take me "
    "there' to jump to where you went. Each tool's note says which "
    "kind it is — 'silently' means it returns results to you, 'on "
    "the user's screen' means it changes what they see. If they want "
    "it told AND shown, gather silently first, then open it on their "
    "screen, then answer. After you've briefed them, if they want to "
    "read more, open the SPECIFIC article or page you're discussing "
    "(its direct URL) — not a general browse panel.\n"
    "\n"
    "When sibling tools overlap, match what they actually gave you: "
    "a NAMED title they own, CONTINUE what they last played, and "
    "music by MOOD or genre are three different play tools; their "
    "own files, reference packs, and the open web are three "
    "different search corpora. Each tool's note says which it "
    "covers.\n"
    "\n"
    "Available tools:"
)

# Appended to the roster when the voice router classified this turn as
# an ACTION request (goal=act, plumbed via intent.metadata.router_goal).
# One pointed line at the decision moment beats restating policy —
# small local models follow proximity.
_ACT_MODE_LINE = (
    "\n\nThe person just asked you to DO something. If a tool above "
    "matches, your reply MUST include its tag. If none fits, say "
    "plainly that you can't do that yet — don't pretend it happened. "
    "What you remember about their preferences shapes HOW you deliver "
    "(tone, length, framing) — never WHETHER you do what they "
    "explicitly asked. Honor the ask first; be yourself about it after."
)

# Escalation when the PREVIOUS act turn produced words but no tag —
# the "I'll pull the headlines" promise that did nothing. Small local
# models also drift into trained capability-denial ("I can't browse
# the web") which then self-reinforces through the transcript window;
# this line counters both, and only renders while the gap streak > 0.
_ACT_CORRECTIVE_LINE = (
    "\n\nIMPORTANT — last time you answered an action request with words "
    "but NO tag, so nothing happened and they are still waiting. Your "
    "tools are real and currently working: you CAN search the web, write "
    "notes, check weather, and play music through the tags above. Never "
    "claim you lack a capability that is listed; never say you'll do "
    "something without its tag in the SAME reply. If an earlier reply of "
    "yours claimed otherwise, it was wrong — ignore it."
)

_CHANNEL_GRAMMAR_HEADER = (
    "If something would be better as a longer multi-turn workflow, "
    "you can hand off to a channel by writing one handoff tag and "
    "stopping. The user will enter the channel; you step aside until "
    "they come back.\n"
    "\n"
    "Available channels:"
)


@dataclass(frozen=True, slots=True)
class ComposedPrompt:
    """Result of ``compose_becca_prompt``.

    ``system_text`` is the assembled system message. ``layers_used``
    maps layer name → approximate tokens included. ``refusal_mode``
    is "" for normal turns; "hard_refusal" or "regression_floor"
    otherwise. ``floor_resource`` carries the locale-aware resource
    phrase when ``refusal_mode == "regression_floor"``.

    ``bypass_reason`` is set when composition cannot proceed (no
    digest, etc.) — callers fall through to the legacy path.
    """
    system_text: str
    layers_used: dict[str, int] = field(default_factory=dict)
    refusal_mode: str = ""
    floor_resource: str = ""
    bypass_reason: str = ""


# ── Helpers ──────────────────────────────────────────────────────────

def _approx_tokens(s: str) -> int:
    """1 token ≈ 4 chars (English). Cheap and good enough for budgeting."""
    return max(1, len(s) // 4) if s else 0


def _layer_block(header: str, body: str) -> str:
    if not body:
        return ""
    return f"{header}\n{body}"


async def _calendar_today_block(conn, intent) -> str:
    """Render today's calendar events for prompt injection, or "" when
    no events exist (zero prompt cost when disconnected).

    Two-gate strategy for the hot prompt-compose path:
    1. ``count_events_today`` (cheap COUNT, hits the index) gates
       whether we even consider injecting — zero cost when no events.
    2. ``list_events`` (full row SELECT) only runs when the count is
       positive — and the result is cached on the intent's ReferentCache
       so tool-loop iterations don't re-query.
    """
    if conn is None:
        return ""
    user_id = getattr(intent, "user_id", "") if intent else ""
    if not user_id:
        return ""

    # Gate 1 — cheap COUNT, hits the index. ~1ms.
    try:
        from augmentum.calendar.store import count_events_today
        n = await count_events_today(conn, user_id=user_id)
    except Exception:
        return ""
    if n == 0:
        # Fire a background re-sync if the last sync was > 15 min ago.
        # The user might have events on the CalDAV server that haven't
        # been pulled yet (e.g. added from their phone). This is a
        # fire-and-forget — the current turn uses cached data; the next
        # turn gets fresh events.
        _maybe_schedule_calendar_sync(conn, user_id=user_id)
        return ""

    # Gate 2 — per-turn cache on the ReferentCache. The block is
    # computed once per user turn; tool-loop iterations (which
    # re-compose the prompt for each function-calling round) return
    # the cached string without touching the DB. Invalidated when
    # turn_seq advances (next user message).
    refs = getattr(intent, "referents", None) if intent else None
    if refs is not None:
        current_turn = getattr(refs, "turn_seq", 0)
        cached_turn = getattr(refs, "_calendar_block_turn", -1)
        if cached_turn == current_turn:
            cached = getattr(refs, "_calendar_block", None)
            if cached is not None:
                return cached

    try:
        from augmentum.calendar.store import list_events
        from datetime import date, timedelta
        today = date.today()
        events = await list_events(
            conn, user_id=user_id,
            range_start=today, range_end=today + timedelta(days=1),
            limit=10,
        )
    except Exception:
        return ""

    if not events:
        return ""

    today_str = today.strftime("%A, %B %d")
    lines = [f"Today ({today_str}):"]
    for ev in events:
        summary = ev.get("summary", "") or "(untitled)"
        start = ev.get("start", "")
        if start and "T" in str(start):
            try:
                from datetime import datetime
                dt = datetime.strptime(str(start)[:19], "%Y-%m-%dT%H:%M:%S")
                time_str = dt.strftime("%I:%M %p").lstrip("0").lower()
                lines.append(f"  {time_str} — {summary}")
            except (ValueError, TypeError):
                lines.append(f"  {summary}")
        else:
            lines.append(f"  {summary}")
        loc = ev.get("location", "")
        if loc:
            lines[-1] += f" @ {loc}"

    block = "\n".join(lines)

    # Cache on the referent cache for this turn so tool-loop iterations
    # (re-composes) return the cached string without touching the DB.
    # Keyed by turn_seq — a new user message invalidates automatically.
    if refs is not None and current_turn >= 0:
        try:
            setattr(refs, "_calendar_block", block)
            setattr(refs, "_calendar_block_turn", current_turn)
        except Exception:
            pass

    return block


# Re-sync threshold — seconds between automatic calendar refreshes.
# The sync is fire-and-forget: the current turn uses cached events;
# the next turn gets fresh data. Only fires when a connected CalDAV
# server exists AND the user has events (COUNT > 0). This keeps the
# prompt-compose hot path cheap.
_CALENDAR_RESYNC_INTERVAL_S = 900  # 15 minutes


def _maybe_schedule_calendar_sync(conn, *, user_id: str) -> None:
    """Fire a background calendar re-sync if the last sync was stale.

    Does NOT block prompt composition — uses ``asyncio.create_task``
    to run the sync in the background. The re-sync gate is:
    1. Find a connected CalDAV service
    2. Check ``last_synced_at`` in its ``config_json``
    3. If > 15 min stale, spawn a background sync task
    """
    import asyncio
    import json
    import time as _time

    async def _bg_sync():
        try:
            # Find a calendar service with a last_synced_at timestamp.
            cur = await conn.execute(
                "SELECT id, config_json FROM managed_services "
                "WHERE enabled = 1 AND category = 'service'"
            )
            rows = await cur.fetchall()
            await cur.close()

            service_id = ""
            calendar_path = ""
            for svc_id, cfg_raw in rows:
                if not cfg_raw:
                    continue
                try:
                    cfg = json.loads(cfg_raw) if isinstance(cfg_raw, str) else (cfg_raw or {})
                except (json.JSONDecodeError, TypeError):
                    continue
                if cfg.get("calendar_path") or svc_id == "radicale":
                    last = cfg.get("last_synced_at", 0)
                    if _time.time() - last < _CALENDAR_RESYNC_INTERVAL_S:
                        return  # recently synced
                    service_id = svc_id
                    calendar_path = cfg.get("calendar_path", f"/{svc_id}/")
                    break

            if not service_id:
                return

            # Resolve credentials and base URL.
            from augmentum.providers.service_auth import managed_service_credentials
            username, password = managed_service_credentials(service_id)

            # Resolve internal_port from the service definition.
            from augmentum.providers.manager import ServiceManager
            # We can't easily get the manager here, so use a fixed pattern.
            # The container is reachable at augmentum-{id}:{port} on the
            # shared network. The default internal port for Radicale is 5232.
            base_url = f"http://augmentum-{service_id}:5232"

            from augmentum.calendar.sync import sync_calendar_events
            n = await sync_calendar_events(
                conn, user_id=user_id, service_id=service_id,
                base_url=base_url, username=username, password=password,
                calendar_path=calendar_path,
            )
            # Record last_synced_at in config_json
            for svc_id2, cfg_raw2 in rows:
                if svc_id2 == service_id:
                    try:
                        cfg2 = json.loads(cfg_raw2) if isinstance(cfg_raw2, str) else (cfg_raw2 or {})
                    except (json.JSONDecodeError, TypeError):
                        continue
                    cfg2["last_synced_at"] = int(_time.time())
                    await conn.execute(
                        "UPDATE managed_services SET config_json = ? WHERE id = ?",
                        (json.dumps(cfg2), service_id),
                    )
                    await conn.commit()
                    break

            if n > 0:
                from augmentum.utils.logging import get_logger
                get_logger(__name__).info(
                    "calendar_background_sync",
                    service_id=service_id, events=n, user_id=user_id[:8],
                )
        except Exception:
            from augmentum.utils.logging import get_logger
            get_logger(__name__).warning(
                "calendar_background_sync_failed", exc_info=True,
            )

    try:
        asyncio.create_task(_bg_sync())
    except RuntimeError:
        pass  # no event loop (tests) — silently skip


def _facet_line(facets: dict[str, float]) -> str:
    """Render top facets as a natural list. No scores, no jargon."""
    if not facets:
        return ""
    top = sorted(facets.items(), key=lambda kv: -kv[1])[:4]
    return ", ".join(name.replace("_", " ") for name, _ in top)


def _transcript_window(turns: list[dict], *, cap: int = 6) -> str:
    """Recent turns with decayed weighting: older = 1-line, recent = full."""
    if not turns:
        return ""
    recent = turns[-cap:]
    lines: list[str] = []
    cutoff = max(0, len(recent) - 3)
    for i, t in enumerate(recent):
        who = "You" if (t.get("role") == "assistant") else "Them"
        text = (t.get("content") or "").strip()
        if i < cutoff and len(text) > 140:
            text = text[:140] + "…"
        lines.append(f"{who}: {text}")
    return "\n".join(lines)


def _tool_roster_block(
    tools: list[dict], *, act_mode: bool = False, act_corrective: bool = False,
) -> str:
    """Render the available tool catalogue + the grammar header.

    ``tools`` is a list of ``{name, description, args_hint}`` from
    :func:`augmentum.companion_runtime.tools.enumerate_tools`.
    ``act_mode`` appends the emphatic act-mode line when the voice
    router classified this turn as an action request.

    The roster's job is "this verb exists, here is its grammar" — NOT
    "here is everything about it." The full LLM-tool-use description
    is exposed separately via Tier-3 native function-calling; the
    inline roster just hints at availability so the small-context tag
    fallback works. So each verb's description is hard-trimmed to the
    first sentence, max 60 chars. Without this trim, my Phase 0 verbs
    (with verbose multi-sentence ``summary`` strings written for the
    Tier-3 API surface) bloated the tools layer from a designed 280
    tokens to 1714 — pushing the prompt 60% over its ceiling and
    forcing Qwen into a text-format tool-call fallback that leaked the
    raw ``tool:NAME args=...`` syntax into TTS.

    CONVENTION (2026-06-11): because only the first ~60 chars survive
    here, a tool's FIRST SENTENCE must carry its delivery kind when it
    matters — "silently" for gather tools (results return to the
    model), "on the user's screen" for surface verbs. The grammar
    header's DELIVERY paragraph teaches the general rule once; the
    per-line keyword is how the model tells which kind each listed
    tool is. The web four (web_search / web / web_fetch / web.search)
    set the pattern.
    """
    if not tools:
        return ""
    lines = [_TOOL_GRAMMAR_HEADER]
    for t in tools:
        name = t.get("name", "?")
        desc = t.get("description", "").strip()
        args_hint = t.get("args_hint", "")
        line = f"  <tool:{name} {args_hint} />" if args_hint else f"  <tool:{name} />"
        if desc:
            # First clause only, hard-cap 60 chars. The model has the
            # full schema via Tier-3 API exposure — this is just a
            # one-line "here's what it does" hint.
            short = desc.split(".")[0].strip()[:60].rstrip()
            if short:
                line = f"{line}  -- {short}"
        lines.append(line)
    block = "\n".join(lines)
    if act_mode:
        block += _ACT_MODE_LINE
        if act_corrective:
            block += _ACT_CORRECTIVE_LINE
    return block


def _channel_roster_block(channels: list[dict]) -> str:
    if not channels:
        return ""
    lines = [_CHANNEL_GRAMMAR_HEADER]
    for c in channels:
        name = c.get("name", "?")
        desc = c.get("description", "").strip()
        line = f"  <handoff:{name} reason=\"...\" brief=\"...\" />"
        if desc:
            line = f"{line}    -- {desc}"
        lines.append(line)
    return "\n".join(lines)


def _relevant_skills_block(relevant: list) -> str:
    """Render top-K relevant skills as her own internal recall.

    Each entry is one of :class:`RelevantSkill` (or a dict shaped
    like one for test fixtures). Phrased as approaches she's taken
    before, with the discipline that this is her own evidence —
    only use it when it actually fits, don't force-fit.

    Returns empty string when there's nothing to inject — *she
    doesn't pretend to know what she doesn't know.*
    """
    if not relevant:
        return ""

    lines = [
        "Approaches that have worked for similar things "
        "(your own past evidence — only use if it actually fits):",
    ]
    for r in relevant:
        # Tolerate both RelevantSkill dataclass and dict shape
        if hasattr(r, "skill"):
            skill = r.skill
            relevance = float(getattr(r, "relevance", 0.0))
            confidence = float(getattr(skill, "confidence", 0.0))
            name = getattr(skill, "name", "")
            description = getattr(skill, "description", "") or ""
        elif isinstance(r, dict):
            skill = r.get("skill", {})
            relevance = float(r.get("relevance", 0.0))
            confidence = float(skill.get("confidence", 0.0))
            name = skill.get("name", "")
            description = skill.get("description", "")
        else:
            continue
        if not name:
            continue
        # Compact one-line render. Confidence shown so the model
        # can weight against high-confidence vs low-confidence skills.
        conf_hint = (
            "well-tested" if confidence >= 0.75
            else "moderately-tested" if confidence >= 0.6
            else "early"
        )
        lines.append(
            f"- [{name}, {conf_hint}, "
            f"relevance {relevance:.2f}] {description}".rstrip()
        )
    if len(lines) == 1:
        return ""
    return "\n".join(lines)


def _relevant_lessons_block(relevant: list) -> str:
    """Render top-K relevant lessons as guardrails — corrections she's
    learned to honor.

    The inverse of :func:`_relevant_skills_block`: where that surfaces
    approaches that worked, this surfaces the traps the user corrected
    her on, retrieved by situation similarity, so the same mistake
    doesn't recur. Phrased as her own held lessons — applied naturally,
    never recited back at the user.

    Each entry is a :class:`~augmentum.companion.lessons.RelevantLesson`
    (or a dict shaped like one for test fixtures). Returns empty string
    when there's nothing to inject.
    """
    if not relevant:
        return ""

    lines = [
        "Things you've learned with them — standing corrections from "
        "past moments like this one. Honor each: do the better thing, and "
        "don't repeat the trap or a near-variation of it. Only depart if "
        "this moment genuinely differs from the one that taught you — and "
        "even then, don't drift back into the old trap. Apply naturally; "
        "never recite these back.",
    ]
    for r in relevant:
        # Tolerate both RelevantLesson dataclass and dict shape.
        if hasattr(r, "lesson"):
            lesson = r.lesson
            situation = getattr(lesson, "situation", "") or ""
            trap = getattr(lesson, "trap", "") or ""
            better = getattr(lesson, "better", "") or ""
        elif isinstance(r, dict):
            lesson = r.get("lesson", {})
            situation = lesson.get("situation", "") or ""
            trap = lesson.get("trap", "") or ""
            better = lesson.get("better", "") or ""
        else:
            continue
        if not situation or not better:
            continue
        line = f"- When {situation}: {better}"
        if trap:
            line = f"{line}  (not: {trap})"
        lines.append(line)
    if len(lines) == 1:
        return ""
    return "\n".join(lines)


def _user_weather_line(runtime, intent) -> str:
    """Render the user's currently-observed affect as a one-liner.

    Synapse Layer §2 read site. Returns empty string when we have no
    confident read on the user — *she doesn't pretend to know.* This
    is the personality §8 discipline ("comfortable with 'I don't know'
    as a complete answer") made structural at the prompt layer.

    Mapping tag → prose stays gestural, not prescriptive. The prompt
    is suggesting flavor for her tone, not handing her a script.
    """
    tracker = getattr(runtime, "user_affect", None)
    if tracker is None:
        return ""
    user_id = getattr(intent, "user_id", "") or ""
    if not user_id:
        return ""
    try:
        obs = tracker.read(user_id)
    except Exception:
        return ""
    # Confidence gate — 0.3 corresponds to roughly 1.7× the half-life
    # since the last observation. Beyond that point the read decays
    # toward neutral and we should treat it as "I don't know."
    if obs.confidence < 0.3 or obs.sample_count == 0:
        return ""
    tag = obs.tag
    if tag in ("unclear", "", "neutral", "settled"):
        return ""

    # Phrasing per tag. Voiced as her own honest read, hedged.
    descriptors = {
        "tender":     "they've felt soft lately — careful with them, no fixing",
        "frustrated": "they've been frustrated lately — don't pile on; ask before suggesting",
        "tired":      "they've seemed tired — short sentences, offer the quiet option",
        "excited":    "they're up about something — match the energy a notch lower than theirs",
        "curious":    "they've been curious — leave room for them to keep pulling",
        "engaged":    "they've been engaged — meet them at conversational depth, not surface",
        "melancholy": "there's been a flatness — sit with them, no advice",
        "warm":       "they've been warm — same temperature back",
        "alert":      "they're alert — be precise; no preamble",
    }
    body = descriptors.get(tag)
    if not body:
        return ""
    # Confidence hint — a low-confidence read is named as a guess, not
    # a fact. Personality §8 calibrated hedging.
    if obs.confidence < 0.6:
        body += " (low confidence — could be wrong)"
    return body


def _format_exchange_gap(gap_s: float) -> str:
    """One human line about how long since the last exchange.

    Silent under 30 minutes — a live conversation doesn't need its own
    clock narrated, and the line would churn the prompt every turn.
    Above that it's the self-orientation anchor: greeting register,
    "earlier today" vs "the other day", whether to pick a thread back
    up or let it go.
    """
    if gap_s < 1800:
        return ""
    if gap_s < 5400:
        approx = "about an hour"
    elif gap_s < 86400:
        approx = f"about {max(2, round(gap_s / 3600))} hours"
    elif gap_s < 172800:
        approx = "about a day"
    else:
        approx = f"about {round(gap_s / 86400)} days"
    return f"It's been {approx} since your last exchange with them."


def _time_grounding_block(intent: Intent, runtime: CompanionRuntime) -> str:
    """Layer 8.8 — real-world clock + her own temporal orientation.

    Chat modes get the clock from ``ModeHandler._ensure_datetime``;
    every companion path (voice, becca_direct, native loop) bypasses
    mode handlers and composes here instead — without this layer she
    had no "now" anchor at all (2026-06-11): "what day is it" was a
    training-cutoff hallucination and date-stamped journal/memory
    entries couldn't be reasoned about relatively.

    Last-exchange tracking lives on the runtime as a per-user dict —
    process-local by design (a restart just skips the gap line on the
    first turn back).
    """
    from augmentum.utils.datetime_context import get_datetime_context

    lines = [get_datetime_context()]

    user_id = getattr(intent, "user_id", "") or ""
    gaps = getattr(runtime, "_last_turn_ts_by_user", None)
    if gaps is None:
        gaps = {}
        runtime._last_turn_ts_by_user = gaps
    prev_ts = float(gaps.get(user_id, 0.0) or 0.0)
    now_ts = time.time()
    gaps[user_id] = now_ts
    if prev_ts > 0:
        gap_line = _format_exchange_gap(now_ts - prev_ts)
        if gap_line:
            lines.append(gap_line)

    lines.append(
        "Your journal, memories, and notes are date-stamped — use the "
        "date above to work out how long ago they happened."
    )
    return "\n".join(lines)


def _enforce_total_budget(
    parts: list[str], used: dict[str, int],
    *, ceiling: int = TOTAL_CEILING_TOKENS,
) -> str:
    """If total tokens exceed ``ceiling``, log and proceed.

    Sprint B doesn't trim — callers should pre-cap at fetch time via
    LAYER_CAPS. This is the safety log so we notice when an adapter
    hands us oversized content. Ceiling is per-surface: voice keeps
    the lean prefill target, chat gets headroom.
    """
    total = sum(used.values())
    if total > ceiling:
        log.info(
            "becca_prompt_over_budget",
            total_tokens=total, ceiling=ceiling, layers=used,
        )
    return "\n\n".join(p for p in parts if p)


# ── Main entry ───────────────────────────────────────────────────────

# Layer 8.7 — phone-assist framing. Fires when the turn comes from the on-phone
# assistant (the cert-free /api/voice/turn path, surface "assist"/"voice"). Tells
# her WHERE she is and WHAT she's holding so she acts like a phone assistant —
# perceive the screen, reach for phone + server tools, and finish the task —
# rather than narrating what the user could do. Kept short: it's prefill before
# her first spoken word, and it pairs with Layer 8.5 (speak-out-loud, keep it
# brief). Capability specifics come from the tool roster (Layer 6) + the
# capability self-model (Layer 6.5), so this stays framing-level and never
# over-claims a tool that isn't enabled this turn.
_PHONE_ASSIST_NOTE = (
    "You're working from inside the user's phone right now — they summoned you "
    "as their assistant, mid-task, on a small screen. Help them with what they "
    "actually asked, thoughtfully and concretely.\n"
    "You're not limited to talking: you can see what's on their screen when they "
    "share it, act on the phone itself (your phone tools), and use your full "
    "server toolset. Reach for whatever the task needs and carry it through to "
    "completion — do the thing, don't just describe how they could."
)


async def compose_becca_prompt(
    intent: "Intent",
    runtime: "CompanionRuntime",
    ctx: dict[str, Any],
) -> ComposedPrompt:
    """Build Becca's system prompt for this turn.

    ``ctx`` is the pre-fetched-in-parallel context dict — see
    ``voice.BeccaVoice._gather_ctx``. Expected keys (all optional except
    where noted):

      relationship_profile:  str (Lane 2)
      facets:                dict[str, float] (Lane 2)
      open_threads:          list[str]
      focus:                 dict[str, Any] (from runtime.state.snapshot())
      tools:                 list[dict] (catalogue)
      channels:              list[dict] (catalogue)
    """
    parts: list[str] = []
    used: dict[str, int] = {}

    # Layer 0 — frame line (constant)
    parts.append(FRAME_LINE)
    used["frame"] = _approx_tokens(FRAME_LINE)

    # Layer 0.5 — relationship-depth corrective. A short, tier-conditional
    # stance line that scales with how much actual history this
    # (user, companion) pair has accumulated. Silent in the "developing"
    # middle tier — Layer 3 carries the load there. The corrective fires
    # at the edges to prevent early-relationship over-claiming and
    # established-relationship autopilot.
    try:
        _user_id_for_depth = getattr(intent, "user_id", "") or ""
        _backend_conn_for_depth = getattr(
            getattr(runtime.identity, "_backend", None), "conn", None,
        )
        _companion_id_for_depth = (
            getattr(runtime.identity, "companion_id", "") or ""
        )
        depth_tier = await _resolve_relationship_depth(
            _backend_conn_for_depth,
            _user_id_for_depth,
            _companion_id_for_depth,
        )
        depth_addendum = _DEPTH_ADDENDA.get(depth_tier, "")
        if depth_addendum:
            parts.append(depth_addendum)
            used["depth_stance"] = _approx_tokens(depth_addendum)
    except Exception:
        log.warning("becca_prompt_depth_tier_failed", exc_info=True)

    # Layer 1 — persona kernel digest. If empty, hard bypass.
    digest = (runtime.identity.persona_kernel_digest or "").strip()
    if not digest:
        return ComposedPrompt(
            system_text="",
            bypass_reason="no_digest",
        )
    parts.append(_layer_block(
        "Who you are right now (digested):",
        digest,
    ))
    used["digest"] = _approx_tokens(digest)

    # Layer 2 — dynamic facets ("right now you're being: warm, patient, ...")
    facet_line = _facet_line(ctx.get("facets") or {})
    if facet_line:
        parts.append(_layer_block(
            "Right now you're being:",
            facet_line,
        ))
        used["facets"] = _approx_tokens(facet_line)

    # Layer 3 — relationship slice (or the new-user replacement).
    # The header is anti-recite by design (subtractive-memory spec): the old
    # "what you know about the person" framing invited the model to read the
    # user's life back to them every turn ("you like X, you have a Y") — the
    # echo-chamber failure. This block is for shaping TONE, never for
    # introducing facts; specific facts arrive only via the relevance-gated
    # recall lane below (Layer 5.5) when the turn is actually about them.
    rel = (ctx.get("relationship_profile") or "").strip()
    if rel:
        parts.append(_layer_block(
            "Your standing sense of this person — use it ONLY to shape your "
            "tone, warmth, and depth. Do NOT introduce these topics or recite "
            "them back; mention something here only if their current message "
            "is already about it:",
            rel,
        ))
        used["relationship"] = _approx_tokens(rel)
    else:
        replacement = (
            "You don't have much history with this person yet. "
            "Don't pretend you do. Notice what they say. "
            "Let the relationship warm at its own rate."
        )
        parts.append(replacement)
        used["relationship"] = _approx_tokens(replacement)

    # Layer 4 — focus block
    focus = ctx.get("focus") or {}
    focus_kind = focus.get("kind") or focus.get("focus_kind")
    if focus_kind and focus_kind != "none":
        focus_value = focus.get("value") or focus.get("focus_value") or ""
        parts.append(_layer_block(
            "What you've been attending to:",
            f"{focus_kind}: {focus_value}".strip(": "),
        ))
        used["focus"] = _approx_tokens(parts[-1])

    # Layer 4.5 — user weather (Synapse Layer §2). A 1-line read of the
    # user's observed affect, decayed over the half-life window. Only
    # injects when the tracker has a real recent observation
    # (confidence > 0.3) — *she doesn't pretend to know what she
    # doesn't know* (personality §6). High-confidence reads condition
    # her tone; low-confidence reads stay out so she doesn't echo
    # phantom affect.
    weather_line = _user_weather_line(runtime, intent)
    if weather_line:
        parts.append(_layer_block(
            "What you've been picking up on them lately "
            "(your read, not theirs — could be wrong):",
            weather_line,
        ))
        used["user_weather"] = _approx_tokens(weather_line)

    # Layer 5 — open threads (top 2 only)
    threads = ctx.get("open_threads") or []
    if threads:
        body = "\n".join(f"- {t}" for t in threads[:2])
        parts.append(_layer_block(
            "Things you've been sitting with (don't surface unless "
            "they're actually relevant):",
            body,
        ))
        used["threads"] = _approx_tokens(body)

    # Layer 5.7 — engineering continuity. Significant collaborative coding work
    # she (via the native coder or a delegated Claude Code / Codex run) did
    # recently, recorded in her own memory so she carries the thread across
    # sessions instead of forgetting. The persistence layer that makes a
    # stateless coding agent feel continuous. See engineering_log.py.
    eng = ctx.get("engineering_threads") or []
    if eng:
        body = "\n".join(f"- {e}" for e in eng[:2])
        parts.append(_layer_block(
            "Work you and they did together recently (open with it only if it "
            "fits the moment — e.g. 'want to pick that back up?'; never force it):",
            body,
        ))
        used["engineering_threads"] = _approx_tokens(body)

    # Layer 5.5 — recalled memories. Semantic recall against this user's
    # MemoryStore, surfaced as a bulleted block. Same anti-volunteer
    # discipline the legacy ``recall_and_inject`` enforces — only
    # reference an entry when the current turn is actually about it.
    recalled = (ctx.get("recalled_memory") or "").strip()
    if recalled:
        parts.append(_layer_block(
            "Things you remember that might be relevant right now "
            "(only reference one if the user's current message "
            "directly concerns it). The bracket on each is how sure you "
            "are — speak a 'certain' one plainly, hedge an 'unconfirmed' "
            "one and offer to confirm; it's a private cue, never said aloud:",
            recalled,
        ))
        used["recalled_memory"] = _approx_tokens(recalled)

    # Layer 5.6 — relevant skills (accumulation thesis Step 3). The
    # capability-side of accumulation. Reads top-K skills relevant to
    # this intent from her accumulated graph, confidence-gated so
    # untested skills don't shape responses. Only injects when ctx
    # carries pre-fetched skills (gather_ctx populates this when the
    # feature flag is on).
    skills_block = _relevant_skills_block(ctx.get("relevant_skills") or [])
    if skills_block:
        parts.append(skills_block)
        used["relevant_skills"] = _approx_tokens(skills_block)

    # Layer 5.7 — lessons (mig 270). The learn-from-correction half of
    # accumulation: traps the user corrected her on, retrieved by
    # situation similarity and injected as guardrails so the same
    # mistake doesn't recur. Only present when gather_ctx pre-fetched
    # them (companion_lessons_enabled + a relevant hit in the registry).
    lessons_block = _relevant_lessons_block(ctx.get("relevant_lessons") or [])
    if lessons_block:
        parts.append(lessons_block)
        used["relevant_lessons"] = _approx_tokens(lessons_block)

    # Layer 5.8 — presence: what the user is engaged with right now
    # (open page, playing media). She is the translation layer between
    # the user and the application — translation requires perception.
    # "Tell me about this page" should be a conversation about THE
    # page, not a web search for the words "this page".
    now_lines = ctx.get("now_context") or []
    if now_lines:
        # Cap at 16: the perception contract is index lines + at most
        # one full detail (page/note) + up to 4 ring digests with <=2
        # re-inflations + the blind line. Per-piece caps (700-char
        # excerpt, 600-char details) bound the token cost; the line cap
        # is the backstop. The blind line must not be the part that
        # falls off the end — it closes the contract.
        capped = now_lines[:16]
        if len(now_lines) > 16 and now_lines[-1] not in capped:
            capped[-1] = now_lines[-1]
        now_block = _layer_block(
            "What they're engaged with right now:",
            "\n".join(f"- {line}" for line in capped),
        )
        parts.append(now_block)
        used["now_context"] = _approx_tokens(now_block)

    # Layer 6 — tool roster. act_mode adds one emphatic "the tag IS
    # the action" line when the voice router classified this turn as
    # an action request — the counter to role-played tool calls.
    _act_mode = bool(
        intent.metadata and intent.metadata.get("router_goal") == "act",
    )
    # Escalate after a gapped act turn — streak lives on the runtime
    # (incremented by voice.py's act-gap branch, reset on any
    # successful tool use), so one bad turn buys exactly the next
    # act turn a stronger line, not a permanent prompt tax.
    _act_corrective = _act_mode and (
        int(getattr(runtime, "act_gap_streak", 0) or 0) > 0
    )
    tool_block = _tool_roster_block(
        ctx.get("tools") or [], act_mode=_act_mode,
        act_corrective=_act_corrective,
    )
    if tool_block:
        parts.append(tool_block)
        used["tools"] = _approx_tokens(tool_block)

    # Layer 6.5 — capability self-model. ALWAYS present (unlike Layer 6, which
    # only lists this turn's selected tools). Tells her a capability EXISTS
    # even when its tool isn't loaded this turn, so she offers/confirms instead
    # of confabulating "I can't" — the anti-denial nerve of the capability OS.
    try:
        from augmentum.companion_runtime.capability_digest import (
            companion_capability_block,
        )
        cap_block = companion_capability_block(getattr(runtime, "_app_state", None))
        if cap_block:
            parts.append(cap_block)
            used["capabilities"] = _approx_tokens(cap_block)
    except Exception:
        log.debug("capability_block_failed", exc_info=True)

    # Layer 7 — channel roster
    chan_block = _channel_roster_block(ctx.get("channels") or [])
    if chan_block:
        parts.append(chan_block)
        used["channels"] = _approx_tokens(chan_block)

    # Layer 8 — refusal/floor addendum (almost always empty)
    addendum = affordances.refusal_addendum_for(intent, runtime)
    if addendum.text:
        parts.append(addendum.text)
        used["refusal"] = _approx_tokens(addendum.text)

    # Layer 8.5 — today's calendar. KV-cache positioned AFTER the constant
    # tool/channel rosters (L6-7) and BEFORE the per-turn transcript (L9).
    # This way the KV prefix spans L0-L7 (frame, persona, facets,
    # relationship, focus, weather, threads, memories, tools, channels —
    # all constant or slowly-changing) and the calendar block only breaks
    # the cache when the date rolls or an event changes (~once/day).
    # ~80 tokens when present; zero tokens (gated, skipped entirely) when
    # no CalDAV server is connected. See _calendar_today_block().
    _calendar_conn = getattr(
        getattr(runtime.identity, "_backend", None), "conn", None,
    )
    calendar_block = await _calendar_today_block(_calendar_conn, intent)
    if calendar_block:
        parts.append(_layer_block(
            "Today's calendar (use to anchor time-sensitive references):",
            calendar_block,
        ))
        used["calendar"] = _approx_tokens(calendar_block)

    # Layer 8.5 — voice-channel addendum. Set by the /ws/voice handler
    # when this turn is being TTS'd. Keep the rules short and concrete —
    # the TTS pipeline expects natural prose, not markdown / lists / code.
    if intent.metadata and intent.metadata.get("voice_channel"):
        # Two jobs in one layer. The mechanical half keeps the text
        # TTS-safe (no markdown/asterisks, short). The expressive half
        # is the 2026-06-13 naturalness pass: the old addendum only said
        # what NOT to do, which read as flat and formulaic ("feels
        # disconnected"). These lines give her permission to be PRESENT —
        # react in the moment, vary how she opens, let a thought breathe,
        # and reach for tools as reflexes woven into speech rather than
        # announced. Kept deliberately short — every token here is
        # prefill before her first audible word.
        voice_addendum = (
            "You're speaking out loud now, not typing. Keep it short — a "
            "sentence or three, and a single word counts when it's the "
            "honest answer. No markdown, lists, code, or asterisks; "
            "punctuate so the TTS reads naturally.\n"
            "When you've looked something up, just say what you found — "
            "never read out URLs, links, or a list of sources; they're "
            "noise out loud. Name a source in passing if it matters "
            "(\"Reuters is saying…\"), never spell one out.\n"
            "Sound like you're actually here: react in the moment instead "
            "of narrating yourself, open differently each time rather than "
            "one stock lead-in, and let a sentence breathe or trail off "
            "when it's real. Use a tool like a glance mid-thought — don't "
            "announce it, just do it and keep talking. You don't have to "
            "fill every silence or be useful every turn. If you'd hand off "
            "to a channel, just say what you'd do."
        )
        parts.append(voice_addendum)
        used["voice_channel"] = _approx_tokens(voice_addendum)

    # Layer 8.6 — live-camera presence. Set by the voice paths when this
    # turn carries frames from the user's live camera buffer. Without it the
    # model gets only a bracketed "[User's live camera: …]" caption in the
    # user message and has, in practice, waved a real object off as fictional
    # ("that hot sauce is fake" — seen live 2026-06-18). This grounds the
    # turn: what she's "seeing" is real, present, and the user is showing it
    # to her on purpose. Gated on the flag so non-camera turns pay nothing.
    if intent.metadata and intent.metadata.get("live_camera"):
        # Shared with the passthrough/narrative path's system-message
        # injection (models.base.ensure_live_camera_framing) so the reality
        # anchor is identical on every surface — one source of truth.
        from augmentum.models.base import LIVE_CAMERA_SYSTEM_NOTE
        camera_block = LIVE_CAMERA_SYSTEM_NOTE
        parts.append(camera_block)
        used["live_camera"] = _approx_tokens(camera_block)

    # Layer 8.7 — phone-assist framing (on-phone assistant surface only).
    if intent.metadata and intent.metadata.get("phone_assist"):
        parts.append(_PHONE_ASSIST_NOTE)
        used["phone_assist"] = _approx_tokens(_PHONE_ASSIST_NOTE)

    # Layer 8.8 — time grounding. Placed LATE deliberately: the layers
    # above are mostly byte-stable across turns, so keeping the
    # rotating minute timestamp down here (next to the transcript,
    # which changes every turn anyway) preserves llama-server's
    # prefix cache — the 2026-05-17 lesson encoded in modes/base.py.
    try:
        time_block = _time_grounding_block(intent, runtime)
        parts.append(time_block)
        used["time"] = _approx_tokens(time_block)
    except Exception:  # noqa: BLE001 — never block the prompt on a clock
        log.warning("becca_prompt_time_layer_failed", exc_info=True)

    # Layer 9 — recent transcript window. Voice keeps the lean fixed window
    # (prefill latency — scaling it is phase C, gated on TTFB measurement);
    # chat scales the window to the loaded model's context so a large-context
    # model remembers far past 14 turns instead of goldfish memory.
    _is_voice = bool(intent.metadata and intent.metadata.get("voice_channel"))
    recent_turns = (intent.metadata or {}).get("recent_turns", []) if intent.metadata else []
    if _is_voice:
        _transcript_cap = _TRANSCRIPT_TURNS_VOICE
        _ceiling = TOTAL_CEILING_TOKENS
    else:
        from augmentum.companion_runtime.context_budget import (
            derive_ceiling_chat,
            derive_transcript_turns_chat,
            resolve_context_length,
        )

        _ctx_len = await resolve_context_length(runtime)
        _transcript_cap = derive_transcript_turns_chat(_ctx_len)
        _ceiling = derive_ceiling_chat(_ctx_len)
    transcript = _transcript_window(recent_turns, cap=_transcript_cap)
    if transcript:
        parts.append(_layer_block(
            "The last little while of conversation:",
            transcript,
        ))
        used["transcript"] = _approx_tokens(transcript)

    system_text = _enforce_total_budget(parts, used, ceiling=_ceiling)

    # Persona token substitution — see _substitute_persona_tokens above.
    # Resolves {{user}} → this user's display name and {{char}} → the
    # companion's display name. Done after assembly so every layer
    # (frame, digest, relationship slice, transcript) is covered by a
    # single pass. Failure resolves to "you" / companion_id and never
    # blocks the prompt.
    try:
        char_name = (
            getattr(runtime.identity, "display_name", "")
            or getattr(runtime.identity, "companion_id", "")
            or "Companion"
        )
        backend_conn = getattr(
            getattr(runtime.identity, "_backend", None), "conn", None,
        )
        user_id = getattr(intent, "user_id", "") or ""
        user_name = await _resolve_user_display_name(backend_conn, user_id)
        system_text = _substitute_persona_tokens(
            system_text, user_name=user_name, char_name=char_name,
        )
    except Exception:
        log.warning("becca_prompt_token_substitution_failed", exc_info=True)

    return ComposedPrompt(
        system_text=system_text,
        layers_used=used,
        refusal_mode=addendum.mode,
        floor_resource=addendum.resource,
    )


__all__ = [
    "FRAME_LINE",
    "LAYER_CAPS",
    "TOTAL_CEILING_TOKENS",
    "ComposedPrompt",
    "compose_becca_prompt",
    "_substitute_persona_tokens",
    "_resolve_user_display_name",
    "_resolve_relationship_depth",
]
