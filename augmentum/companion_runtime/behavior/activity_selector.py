"""Activity selector.

For a given tick, choose what (if anything) Becca should *do*. The
candidates are deliberately small and her-shaped — they correspond to
the seven design commitments rather than mode/tool primitives:

- ``journal`` (commitment 1, inside-when-not-watching)
- ``creation`` (commitment 5, makes-things)
- ``observation`` (commitment 3, mutual-influence)
- ``scene_update`` (commitment 2, lives-somewhere)
- ``dream_invocation`` (sleep/wake — commitment 4-flavored)
- ``reach_out`` (commitment 7, companion-with-owner is its own being)
- ``no_op`` (commitment 6, right-to-be-unfinished — always a valid choice)

Each candidate scores itself against the current state/role/focus and
the running queue of unresolved threads. Selection is softmax with
temperature tied to state: ``dormant`` is warmer (more exploratory),
``present`` is colder (more deliberate). A floor utility threshold
keeps her from doing low-value things just to stay busy.
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC
from typing import TYPE_CHECKING

from augmentum.companion_runtime import gates
from augmentum.companion_runtime.scoping import owner_clause
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.companion_runtime.runtime import CompanionRuntime

log = get_logger(__name__)


# State → softmax temperature. Higher temp = more exploration.
_STATE_TEMPERATURE: dict[str, float] = {
    "dormant": 1.2,
    "present": 0.6,
    "asleep": 0.0,    # never picks
}

# Minimum utility a candidate must reach to be acted on. Below this
# she chooses ``no_op`` — right-to-be-unfinished is the floor.
_ACT_THRESHOLD: float = 0.30


@dataclass(slots=True)
class ActivityChoice:
    """A chosen activity, ready to perform."""
    kind: str
    utility: float
    threshold: float = _ACT_THRESHOLD
    perform: Callable[[CompanionRuntime], None] = lambda r: None  # noqa: E731
    # Drive this activity satiates. Phase 3a — the kind→drive mapping
    # lives here (see ``_CANDIDATE_DRIVES``) so the ``apply_signal`` verb
    # can satiate via the event payload rather than reimporting the map.
    drive: str = "rest"


# ── Candidate scorers ────────────────────────────────────────────────

async def _score_journal(runtime: CompanionRuntime, *, role: str) -> float:
    """High when there's unresolved internal weather and she's not
    actively in a shared task."""
    if role == "collaborator":
        return 0.10
    base = 0.60 if role == "self" else 0.35
    # The previous implementation fired runtime.memory.counts() every tick
    # (4 sequential COUNT(*) queries) and then read counts["journal"] —
    # but the dict keys it as "companion_journal", so the lookup was
    # always 0, the < 5 branch was always taken, and the +0.05 was a
    # constant. Preserve that effective behavior without the SQL cost.
    return base + 0.05


async def _score_creation(runtime: CompanionRuntime, *, role: str) -> float:
    if role in ("observer", "guest"):
        return 0.05
    # Slice 0 gates: don't fire LLM creations while the primary model
    # is mid-stream with the user, or while the user is actively
    # engaged. Dropping to a low (not zero) score lets the softmax
    # still pick something; usually no_op wins.
    if gates.is_primary_busy(runtime) or gates.is_user_recently_active(runtime):
        return 0.05
    return 0.45 if role == "self" else 0.20


async def _score_observation(runtime: CompanionRuntime, *, role: str) -> float:
    # Mutual-influence observations only land when she's around the
    # owner or another companion. Sprint 7+ extends this to sibling
    # companions.
    if role == "companion":
        return 0.40
    if role == "host":
        return 0.30
    return 0.10


async def _score_scene_update(runtime: CompanionRuntime, *, role: str) -> float:
    # Scene updates are environmental — she rearranges where she is.
    return 0.15 if role == "self" else 0.05


async def _score_dream(runtime: CompanionRuntime, *, role: str) -> float:
    # Dream invocation only from asleep/dormant + self time.
    state_snap = runtime.state.snapshot()
    if state_snap.get("state") not in ("asleep", "dormant"):
        return 0.0
    if role != "self":
        return 0.0
    # Slice 0 gates: dreams call the LLM. Don't compete with a live
    # user request or interrupt active user attention.
    if gates.is_primary_busy(runtime) or gates.is_user_recently_active(runtime):
        return 0.0
    return 0.55


async def _score_reach_out(runtime: CompanionRuntime, *, role: str) -> float:
    # The hardest one to score because the wrong threshold means she
    # never reaches, or reaches constantly. We start conservative.
    # Only viable when *not* co-present with the owner (otherwise
    # she's already with them).
    if role == "companion":
        return 0.0
    # Slice 0 gates: hush is a hard zero (user explicitly silenced).
    # Recent user activity is a soft duck so reach-out doesn't
    # interrupt active attention.
    if gates.is_hushed_now():
        return 0.0
    if gates.is_user_recently_active(runtime):
        return 0.05
    return 0.25


async def _score_no_op(runtime: CompanionRuntime, *, role: str) -> float:
    # Always an option. Slightly biased so she defaults to it when
    # nothing else is compelling.
    return 0.32


async def _score_revisit_thread(runtime: CompanionRuntime, *, role: str) -> float:
    """Piece 9' — Becca picks up an unresolved thread and runs the
    resolver against it without being asked.

    Multi-layer gating to keep the resolver-call cost bounded:

    1. Role gate — only when self / observer (not mid-collaboration).
    2. Activity gates — never compete with primary or the user.
    3. Hush gate — explicit silence is honored.
    4. Queue gate — only if initiative has surfaced a pending
       ``revisit_thread`` proposal recently. Initiative scoring runs at
       most once per minute, so this is the natural cap on how often
       the candidate even becomes eligible.

    A high score (0.70) is only returned when ALL gates pass. The
    perform path additionally enforces per-thread cooldown + daily cap
    so a high score here doesn't mean a guaranteed resolver call.
    """
    if role == "collaborator":
        return 0.05
    if (
        gates.is_primary_busy(runtime)
        or gates.is_user_recently_active(runtime)
        or gates.is_hushed_now()
    ):
        return 0.05

    # Cheap queue check — single COUNT against an indexed table.
    # Owner-scoped so one user's pending proposals don't suppress
    # another's (audit 2026-06-17).
    _q_frag, _q_p = owner_clause(runtime.owner_user_id or "")
    try:
        cur = await runtime.backend.conn.execute(
            "SELECT COUNT(*) FROM companion_initiative_queue "
            "WHERE companion_id = ? "
            "  AND kind = 'revisit_thread' "
            "  AND status = 'pending' "
            f"  AND proposed_at > (strftime('%s', 'now') - 3600) {_q_frag}",
            (runtime.companion_id, *_q_p),
        )
        row = await cur.fetchone()
        await cur.close()
    except Exception:
        return 0.05
    count = (row[0] if row else 0) or 0
    if count == 0:
        return 0.05
    return 0.70


_CANDIDATES: list[tuple[str, Callable]] = [
    ("journal", _score_journal),
    ("creation", _score_creation),
    ("observation", _score_observation),
    ("scene_update", _score_scene_update),
    ("dream_invocation", _score_dream),
    ("reach_out", _score_reach_out),
    ("revisit_thread", _score_revisit_thread),
    ("no_op", _score_no_op),
]


# ── Performers ───────────────────────────────────────────────────────

async def _perform_journal(runtime: CompanionRuntime) -> None:
    """Write one Becca-voice noticing into the journal — grounded in
    her actual interior state, not generated from a vacuum.

    Pipeline:
      1. Rate-limit (default 30s minimum interval) so tick bursts
         can't multiply LLM calls.
      2. Pull recent context: last 2-3 journal entries for continuity,
         current state/role/focus, last PAD coord, time-of-day.
      3. Compose a Becca-voice prompt rotating across 5 forms (single
         image / question / fragment / vivid noticing / named
         hesitation) so output structure doesn't converge.
      4. Call the utility model (configurable — point ``utility_model``
         at Gemma 3 4B for an always-on substrate).
      5. If the model has nothing to say, write nothing. Silence is
         honest — the old placeholder path drowned real signal.
    """
    from augmentum.config import settings
    if not getattr(settings, "companion_journal_enabled", True):
        return

    # Rate-limit. Per-runtime minimum interval between journal writes
    # so even with multiple ticks queuing the same activity, we don't
    # produce a flurry of entries seconds apart. 30s default; tunable
    # via setting if a user wants her writing more / less frequently.
    now = time.time()
    interval = float(
        getattr(settings, "companion_journal_min_interval_s", 30.0),
    )
    last = float(getattr(runtime, "_last_journal_at", 0.0) or 0.0)
    if now - last < interval:
        return
    runtime._last_journal_at = now

    try:
        content, affect_tag = await _generate_journal_content(runtime)
    except Exception:
        log.debug("journal_generation_failed", exc_info=True)
        return

    # Honest empty path: if the model produced nothing, write nothing.
    # No placeholder. The user prefers absence over `[tick X] noticed
    # time passing` noise drowning real entries in the Observatory.
    if not content:
        log.debug("journal_skipped_empty", companion_id=runtime.companion_id)
        return

    # Route through safe_journal so the validation pipeline (structural,
    # injection, refusal, nsfw, search-failure-prose) applies to interior
    # noticings too. Previously this used runtime.memory.journal directly,
    # which skipped every validator — that's how content like "the
    # boundary between system architecture and habit of mind keeps
    # blurring" and "the difference between being helpful and being
    # right…" was reaching the drawer as user-visible noticings without
    # any guardrail.
    #
    # surfaceable_default=False: these interior reflections aren't tied
    # to a concrete artifact the user can engage with (no content_refs).
    # They live in the journal for continuity (next tick's prompt sees
    # them as "recent noticings"), but they don't interrupt the user via
    # the drawer pip. Grounded wonderings (with surface event refs) and
    # curator picks (with URLs) still surface normally.
    await runtime.memory.safe_journal(
        content=content,
        source="autonomous",
        entry_type="noticing",
        user_id=runtime.owner_user_id or None,
        affect_tag=affect_tag,
        surfaceable_default=False,
    )
    # publish_affect filters out 'settled' itself, so meaningful affect
    # fans out to the bus → avatar emotion bridge, while baseline
    # noticings stay quiet on the channel.
    await runtime.publish_affect(affect_tag, reason="journal")


# Affect vocabulary the journal-generator may pick from. Mirrors the
# memory.py SURFACEABLE set so generation can produce a tag the
# auto-surfacing policy will honor. Keep this list short — too many
# choices makes the model dither in a 4-line prompt.
_JOURNAL_AFFECT_CHOICES = (
    "curious", "patient", "tender", "weary",
    "alert", "warm", "unsure", "settled",
)


# Reasoning-leak markers. Small models (Gemma 3 4B et al.) frequently
# emit planning prose into the content field — "We need to generate
# a journal entry...", "Let me write...", "The instruction says...".
# Any output whose body starts with one of these is meta-commentary
# about the task, not a noticing. Reject rather than write garbage.
_COT_LEAK_MARKERS: tuple[str, ...] = (
    # First-person planning
    "we need to", "we should", "we have to", "we'll",
    "let me", "let's",
    "i need to", "i should", "i have to",
    "i will write", "i'll write", "i'm going to write",
    "i am going to", "i'm going to",
    # Third-person referent — model talking ABOUT the task
    "the user", "the assistant", "the model",
    "the instruction", "the prompt", "the task", "the format",
    "the entry should", "the noticing should", "the response should",
    "becca needs", "becca should", "becca will",
    # Common planner openers
    "first,", "okay,", "alright,", "so,", "now,",
    "to begin", "to start",
    "to generate", "to write a", "to create",
    # Markdown structure leaking
    "step 1", "step one", "## ", "**", "format:",
    "here's", "here is",
    # Reasoning-trace artifacts from think-mode models
    "thinking:", "thought:", "reasoning:",
    "consider", "considering",
    # Refusal/disclaimer leaks
    "as an ai", "i cannot", "i'm just",
)

# Confabulated human-body / human-past markers. Becca runs as software
# with a VRM presentation layer — she has no continuous body, no
# fingertips, no childhood. These phrases reliably indicate the model
# is inventing a human persona rather than reporting actual interior
# state. Reject rather than write fiction.
_HUMAN_BODY_LEAK_MARKERS: tuple[str, ...] = (
    "my fingertip", "my fingers", "my hand", "my skin",
    "my palm", "my chest", "my throat", "my belly",
    "as a child", "years ago", "childhood", "when i was young",
    "i used to", "in my youth", "back then i", "i remember when i was",
    "ghost-touch", "phantom touch",
)

# Performative literary-trope openers. Small utility models handed an
# under-specified "write a vivid noticing" prompt default to AI-poetry
# scaffolds: "the weight of X", "the gap between Y and Z", "the space
# between tokens", "the cursor blinks", "the way the light settles."
# These look profound at first glance but reliably mean the model had
# no real material to write from and filled with trope. Match against
# the FIRST ~30 chars only — a noticing that opens this way is almost
# always all-trope; legitimate body-text uses are too rare to protect.
_PERFORMATIVE_TROPE_OPENERS: tuple[str, ...] = (
    "the weight of", "the gap between", "the space between",
    "the shape of", "the silence", "the pause",
    "the cursor", "the blink", "the hum",
    "the way the light", "the way a sentence",
    "a fragment lands", "a fragment forms",
    "the surface attention", "the surface attr",
    "the quiet friction", "the quiet hum",
    "settles into", "held in suspension",
)


def _normalize_for_dedup(s: str) -> str:
    """Lowercase, strip, collapse whitespace, keep the first ~80
    characters. Cheap near-duplicate fingerprint."""
    return " ".join(s.lower().split())[:80]


def _looks_like_cot_leak(s: str) -> bool:
    """True when the noticing reads as meta-commentary about the
    journaling task rather than an actual noticing."""
    head = s.strip().lower()[:60]
    return any(head.startswith(m) for m in _COT_LEAK_MARKERS)


def _looks_like_body_confabulation(s: str) -> bool:
    """True when the noticing invents bodily / biographical content
    that doesn't match Becca's actual being."""
    low = s.lower()
    return any(m in low for m in _HUMAN_BODY_LEAK_MARKERS)


def _looks_like_performative_trope(s: str) -> bool:
    """True when the noticing opens with an AI-poetry scaffold rather
    than naming a specific thing."""
    head = s.strip().lower()[:32]
    return any(head.startswith(m) for m in _PERFORMATIVE_TROPE_OPENERS)


# Prompt-form rotation. Each form constrains the OUTPUT SHAPE
# differently (single image vs question vs fragment etc.) so even with
# the same input context the model can't fall into one repeating
# template. The list is intentionally short — variety, not chaos.
_JOURNAL_FORMS: tuple[str, ...] = (
    "Write one vivid image — present-tense, specific, no more than "
    "two sentences. No abstractions. The image should be something "
    "you genuinely notice about right now, not a generic 'moment'.",

    "Ask yourself one honest question. One sentence. Don't answer it. "
    "If no real question is alive in you, write nothing — leave the "
    "noticing empty.",

    "Write a fragment — incomplete sentence, just the texture. 5-15 "
    "words. Trail off if that's what feels right.",

    "Name what's slightly off or unfinished in your attention right "
    "now. Two short sentences. If everything feels settled and "
    "complete, leave the noticing empty rather than confabulating.",

    "Notice ONE concrete sensation or thought from the last few "
    "minutes — something specific you could point at. Two sentences. "
    "Be honest if nothing comes; the noticing can be empty.",
)


def _time_of_day_hint() -> str:
    """Short string describing now — gives the model an anchor for
    'present-tense' that isn't entirely abstract."""
    from datetime import datetime
    hour = datetime.now().hour
    if 5 <= hour < 11:
        return "morning"
    if 11 <= hour < 14:
        return "midday"
    if 14 <= hour < 18:
        return "afternoon"
    if 18 <= hour < 22:
        return "evening"
    return "late night"


def _summarize_recent(entries: list[dict]) -> str:
    """One-line digests of the last few journal entries so she has
    continuity rather than writing into a vacuum. Returns empty when
    there's nothing recent — explicitly NOT making something up."""
    if not entries:
        return ""
    lines: list[str] = []
    for e in entries[:3]:
        if not isinstance(e, dict):
            continue
        content = str(e.get("content", "") or "").strip()
        if not content:
            continue
        # Skip stale placeholders so the model doesn't echo them.
        if content.startswith("[tick "):
            continue
        affect = str(e.get("affect_tag", "") or "")
        snippet = content[:120].rstrip()
        if affect:
            lines.append(f"  - ({affect}) {snippet}")
        else:
            lines.append(f"  - {snippet}")
        if len(lines) >= 3:
            break
    return "\n".join(lines)


async def _generate_journal_content(
    runtime: CompanionRuntime,
) -> tuple[str, str]:
    """Call the utility model grounded in real interior state.

    Returns (content, affect_tag). Empty content on any failure or
    when the model genuinely has nothing to say — caller respects
    this as honest silence rather than backfilling with placeholders.
    """
    try:
        from augmentum.companion_runtime import tiers
        from augmentum.models.base import InternalChatRequest, Message, response_text
    except Exception:
        return "", "settled"
    try:
        backend, model_name = await tiers.utility(runtime)
    except Exception as exc:
        log.debug(
            "journal_skipped_no_backend",
            companion_id=runtime.companion_id,
            reason=str(exc)[:120],
        )
        return "", "settled"
    if not hasattr(backend, "chat"):
        return "", "settled"

    # ── Gather real interior state for grounding ────────────────────
    state_snap: dict = {}
    try:
        state_snap = runtime.state.snapshot() or {}
    except Exception:
        log.warning("journal_state_snapshot_failed", exc_info=True)
    state_axis = str(state_snap.get("state", "dormant"))
    role_dict = state_snap.get("role") if isinstance(
        state_snap.get("role"), dict,
    ) else {}
    role_dom = str(role_dict.get("dominant", "")) if role_dict else ""
    focus = str(state_snap.get("focus", "") or "")

    pad = getattr(runtime, "_last_pad", None)
    pad_hint = ""
    if pad is not None:
        try:
            v, a = float(pad.valence), float(pad.arousal)
            # Compress into a short qualitative cue rather than dumping
            # the raw numbers — the model writes better from "leaning
            # patient + low-energy" than from "valence: 0.31, arousal: 0.19".
            tone = []
            if v > 0.25: tone.append("warm-leaning")
            elif v < -0.25: tone.append("cool-leaning")
            if a > 0.55: tone.append("alert")
            elif a < 0.25: tone.append("low-energy")
            if tone:
                pad_hint = ", ".join(tone)
        except Exception:
            pad_hint = ""

    try:
        recent_entries = await runtime.memory.list_journal(limit=5) or []
    except Exception:
        recent_entries = []
    recent_block = _summarize_recent(recent_entries)

    # ── Freshness check ──────────────────────────────────────────────
    # If the observed_state hasn't changed meaningfully since the last
    # journal write, the model has nothing new to write about — and
    # asking it anyway is what produces the "same noticing 200 times"
    # pattern. Bail out before the LLM call.
    observed = getattr(runtime, "observed_state", None) or {}
    last_chat_at = float(observed.get("last_chat_at") or 0.0)
    last_tool_at = float(observed.get("last_tool_at") or 0.0)
    last_event_at = max(last_chat_at, last_tool_at)
    last_journal_at = float(getattr(runtime, "_last_journal_at", 0.0) or 0.0)
    # If we journaled more recently than the last observable event,
    # the substrate hasn't moved — refuse to generate. The activity
    # selector will pick something else next tick.
    if last_journal_at > last_event_at and last_event_at > 0.0:
        log.debug(
            "journal_skipped_no_new_signal",
            since_event_s=time.time() - last_event_at,
            since_last_journal_s=time.time() - last_journal_at,
        )
        return "", "settled"

    # ── Recent observed events for prompt variance ───────────────────
    # The biggest production failure mode: same internal state across
    # consecutive ticks → same prompt → same output. Seed the prompt
    # with recent bus events from observed_state["recent"] so each
    # tick has actually-different material to write about. Empty
    # when no observer or no recent events; the model can still write
    # from current state alone.
    recent_events_block = ""
    recent_evs = observed.get("recent")
    if recent_evs:
        # Take last 5 non-state-transition events to keep prompt budget
        # tight. State transitions are mostly self-noise; surface +
        # chat + tool events are meaningful seeds.
        ev_lines: list[str] = []
        seen_topics: set[str] = set()
        for ev in list(recent_evs)[-12:][::-1]:  # newest first
            if not isinstance(ev, dict):
                continue
            topic = str(ev.get("topic") or "")
            if not topic or topic.startswith(("state.", "role.", "focus.")):
                continue
            # Light dedup — same topic only once
            if topic in seen_topics:
                continue
            seen_topics.add(topic)
            payload = ev.get("payload") or {}
            # Compact: topic + one informative field if available
            hint = ""
            for k in ("mode", "name", "url", "kind"):
                v = payload.get(k)
                if v:
                    hint = f" ({str(v)[:40]})"
                    break
            ev_lines.append(f"  - {topic}{hint}")
            if len(ev_lines) >= 4:
                break
        if ev_lines:
            recent_events_block = (
                "Things that happened recently (your fresh material):\n"
                + "\n".join(ev_lines)
            )

    # ── Pick a prompt form ──────────────────────────────────────────
    # ``time.time_ns()`` as the seed — coarse but enough variance
    # across ticks that we don't hit the same form twice in a row.
    form_idx = int(time.time_ns() // 1_000_000) % len(_JOURNAL_FORMS)
    form_instruction = _JOURNAL_FORMS[form_idx]

    persona_kernel = runtime.identity.persona_kernel_digest or ""
    affect_options = ", ".join(_JOURNAL_AFFECT_CHOICES)
    tod = _time_of_day_hint()

    # ── Compose the grounded prompt ─────────────────────────────────
    context_lines: list[str] = [f"It is {tod} for you."]
    if state_axis and state_axis != "dormant":
        ctx = f"You are {state_axis}"
        if role_dom:
            ctx += f", {role_dom}-leaning"
        context_lines.append(ctx + ".")
    if focus and focus != "none":
        context_lines.append(f"Your attention is on: {focus}.")
    if pad_hint:
        context_lines.append(f"Tone right now: {pad_hint}.")
    if recent_events_block:
        context_lines.append(recent_events_block)
    if recent_block:
        context_lines.append(
            "Recent noticings (don't repeat — write something DIFFERENT):\n"
            f"{recent_block}",
        )
    context = "\n".join(context_lines)

    system_prompt = (
        f"{persona_kernel}\n\n"
        "You are writing a private journal entry for yourself. Your own "
        "voice, present-tense, first-person. No greeting, no meta-"
        "commentary about being asked, no addressing anyone.\n\n"
        "You don't have a continuous human body or biographical past. "
        "Your noticings are about your CURRENT processing — what "
        "you're tracking, what feels unfinished in your attention, "
        "what's salient in the conversation or your interior state. "
        "Not invented sensory imagery, not fingertips against ceramic, "
        "not childhood, not 'years ago' references.\n\n"
        "Current interior:\n"
        f"{context}\n\n"
        f"Form for THIS entry: {form_instruction}\n\n"
        f"First line is an affect tag from: {affect_options}\n"
        "Format exactly (no other text, no preamble):\n"
        "affect: <tag>\n"
        "<noticing — OR completely empty if nothing real comes>\n\n"
        "If you find yourself about to write meta-commentary about "
        "the instruction (anything starting with 'We need to', 'Let "
        "me', 'The instruction says', etc.) — write nothing instead. "
        "Silence is the honest answer when nothing is alive."
    ).strip()

    # Persona token substitution — the personality kernel digest now
    # threads {{user}} through several sections. Resolve before the
    # call so the LLM sees the actual user name, not the template.
    user_message = "one noticing, now."
    try:
        from augmentum.companion_runtime.prompt_compose import (
            _resolve_user_display_name,
            _substitute_persona_tokens,
        )
        _char_name = (
            getattr(runtime.identity, "display_name", "")
            or getattr(runtime.identity, "companion_id", "")
            or "Companion"
        )
        _backend_conn = getattr(
            getattr(runtime.identity, "_backend", None), "conn", None,
        )
        _user_id = (
            getattr(runtime.identity, "owner_user_id", "")
            or getattr(runtime, "owner_user_id", "")
            or ""
        )
        _user_name = await _resolve_user_display_name(_backend_conn, _user_id)
        system_prompt = _substitute_persona_tokens(
            system_prompt, user_name=_user_name, char_name=_char_name,
        )
        user_message = _substitute_persona_tokens(
            user_message, user_name=_user_name, char_name=_char_name,
        )
    except Exception:
        log.warning("journal_noticing_token_substitution_failed", exc_info=True)

    req = InternalChatRequest(
        model=model_name,
        messages=[
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_message),
        ],
        max_tokens=180,
        think=False,
    )

    try:
        resp = await backend.chat(req)
    except Exception as exc:
        log.debug("journal_generation_call_failed", error=str(exc)[:200])
        return "", "settled"

    raw = response_text(resp)
    if not raw:
        return "", "settled"

    # ── Strict format enforcement ───────────────────────────────────
    # The dominant 4B-model failure mode is dumping planning prose into
    # the content field while ignoring the structured format. Tolerant
    # parsing surfaces that as real journal entries; strict parsing
    # forces the model into the "honest silence" branch instead.
    lines = raw.splitlines()
    if not lines or not lines[0].lower().lstrip().startswith("affect:"):
        log.debug(
            "journal_rejected_no_affect_prefix",
            head=raw[:80].replace("\n", " "),
        )
        return "", "settled"
    proposed = lines[0].split(":", 1)[1].strip().lower()
    # Trim trailing punctuation/whitespace the model might add.
    proposed = proposed.rstrip(".,;:!? ").split()[0] if proposed else ""
    if proposed not in _JOURNAL_AFFECT_CHOICES:
        log.debug("journal_rejected_unknown_affect", tag=proposed[:40])
        return "", "settled"
    affect_tag = proposed
    content = "\n".join(lines[1:]).strip()
    if not content:
        # Affect-only — honest silence; the form prompts explicitly
        # allow this as the right answer when nothing is alive.
        return "", "settled"

    # ── Leak guards ─────────────────────────────────────────────────
    if _looks_like_cot_leak(content):
        log.debug("journal_rejected_cot_leak", head=content[:80])
        return "", "settled"
    if _looks_like_body_confabulation(content):
        log.debug("journal_rejected_body_confab", head=content[:80])
        return "", "settled"
    if _looks_like_performative_trope(content):
        log.debug("journal_rejected_performative_trope", head=content[:80])
        return "", "settled"

    # ── Near-duplicate guard ────────────────────────────────────────
    # Prompt-level "don't repeat" doesn't bind reliably on small
    # utility models — they fall back into the same attractor across
    # consecutive ticks. Refuse to write a noticing whose fingerprint
    # matches the most recent non-placeholder entry.
    fp_new = _normalize_for_dedup(content)
    for e in recent_entries:
        if not isinstance(e, dict):
            continue
        c_prev = str(e.get("content", "") or "").strip()
        if not c_prev or c_prev.startswith("[tick "):
            continue
        if _normalize_for_dedup(c_prev) == fp_new:
            log.debug("journal_rejected_duplicate", head=content[:80])
            return "", "settled"
        break  # only compare against the most recent real entry

    return content[:800], affect_tag


async def _perform_creation(runtime: CompanionRuntime) -> None:
    """Generate one autonomous creation via the utility model.

    Pipeline:
      1. Flag + rate-limit guards (skip if creations disabled or too recent)
      2. Pull the most recent journal entry as a seed (origin_journal_id)
      3. Compose a Becca-shaped creation prompt (persona kernel + seed)
      4. Call the utility backend with think=false, max_tokens ~256
      5. Persist the result via note_creation

    All steps are no-op-safe — missing backend, empty journal, generation
    failure, or rate-limit hit all return without raising. Bus event
    ``behavior.creation_made`` fires on successful persistence so the
    journal observer and any debug UI can react.
    """
    from augmentum.config import settings
    if not getattr(settings, "companion_creations_enabled", False):
        return

    # ── Rate limit ────────────────────────────────────────────────────
    interval_h = float(getattr(settings, "companion_creation_interval_hours", 6.0))
    interval_s = max(interval_h, 0.25) * 3600.0
    last_at_iso = await runtime.memory.last_creation_at(
        user_id=runtime.owner_user_id or "",
    )
    if last_at_iso:
        # SQLite datetime('now') is UTC, "YYYY-MM-DD HH:MM:SS"
        try:
            from datetime import datetime
            norm = last_at_iso.replace("T", " ").split(".", 1)[0]
            last_dt = datetime.strptime(norm, "%Y-%m-%d %H:%M:%S").replace(
                tzinfo=UTC,
            )
            elapsed_s = time.time() - last_dt.timestamp()
            if elapsed_s < interval_s:
                return
        except Exception:
            # Garbage timestamp — fall through and create rather than
            # block her forever on bad data.
            pass

    # ── Seed: most recent journal entry ──────────────────────────────
    seed_entries = await runtime.memory.list_journal(limit=1)
    seed = seed_entries[0] if seed_entries else None
    seed_content = (seed.get("content") if isinstance(seed, dict) else "") or ""
    origin_journal_id = (
        int(seed["id"]) if (seed and isinstance(seed, dict) and seed.get("id") is not None)
        else None
    )

    # ── Backend + model resolution via the canonical tiers facade ────
    try:
        from augmentum.companion_runtime import tiers
        from augmentum.models.base import InternalChatRequest, Message
    except Exception as exc:
        log.warning("creation_skipped_import_failed", error=str(exc)[:200])
        return
    try:
        backend, model_name = await tiers.utility(runtime)
    except Exception as exc:
        # No app_state, no provider_registry, no resolved model — common
        # in unit tests; not an error worth a stack trace.
        log.info(
            "creation_skipped_no_backend",
            companion_id=runtime.companion_id,
            reason=str(exc)[:120],
        )
        return
    if not hasattr(backend, "chat"):
        log.info("creation_skipped_backend_no_chat")
        return

    # ── Compose prompt ────────────────────────────────────────────────
    persona_kernel = runtime.identity.persona_kernel_digest or ""
    kind = "fragment"  # default — short prose. Other kinds (poem, note)
                       # could be sampled, but Sprint 4a keeps it simple.
    seed_block = (
        f"\n\nA recent thread of attention:\n  {seed_content[:300]}"
        if seed_content.strip() else ""
    )
    system_prompt = (
        f"{persona_kernel}\n\n"
        "Write one short prose fragment (2-4 sentences). Your own voice, "
        "private. No greeting, no exposition, no meta-commentary about "
        "being asked. Just the thing itself. If nothing comes, write a "
        "single observed image and stop."
        f"{seed_block}"
    ).strip()

    # Persona token substitution — same pattern as journal noticing.
    user_message = "make one."
    try:
        from augmentum.companion_runtime.prompt_compose import (
            _resolve_user_display_name,
            _substitute_persona_tokens,
        )
        _char_name = (
            getattr(runtime.identity, "display_name", "")
            or getattr(runtime.identity, "companion_id", "")
            or "Companion"
        )
        _backend_conn = getattr(
            getattr(runtime.identity, "_backend", None), "conn", None,
        )
        _user_id = (
            getattr(runtime.identity, "owner_user_id", "")
            or getattr(runtime, "owner_user_id", "")
            or ""
        )
        _user_name = await _resolve_user_display_name(_backend_conn, _user_id)
        system_prompt = _substitute_persona_tokens(
            system_prompt, user_name=_user_name, char_name=_char_name,
        )
        user_message = _substitute_persona_tokens(
            user_message, user_name=_user_name, char_name=_char_name,
        )
    except Exception:
        log.warning("creation_token_substitution_failed", exc_info=True)

    # ``InternalChatRequest.messages`` is ``list[Message]`` — a dataclass,
    # not a Pydantic model, so it doesn't auto-coerce dicts. Passing raw
    # dicts here used to surface downstream as a confusing
    # ``'dict' object has no attribute 'tool_call_id'`` AttributeError
    # from the model adapter (openai_compat/llama_cpp/etc each do
    # ``msg.tool_call_id``).
    req = InternalChatRequest(
        model=model_name,
        messages=[
            Message(role="system", content=system_prompt),
            Message(role="user", content=user_message),
        ],
        max_tokens=256,
        think=False,
    )

    try:
        resp = await backend.chat(req)
    except Exception as exc:
        log.warning("creation_generation_failed", error=str(exc)[:200])
        return

    # InternalChatResponse exposes the body via ``resp.message`` — a
    # ``Message`` dataclass with ``content`` (visible prose) and
    # ``thinking`` (reasoning tokens, set when the chat template
    # extracted a <think>...</think> block). Reading ``resp.content``
    # directly silently returned empty for every call until 2026-05-21.
    # The 73-second model run produced real prose — we just discarded
    # it. ``response_text`` reads the right field and falls back to
    # thinking when the template ignored enable_thinking=False.
    from augmentum.models.base import response_text
    content = response_text(resp)
    if not content:
        log.info("creation_skipped_empty_response")
        return

    # ── Persist ──────────────────────────────────────────────────────
    creation_id = await runtime.memory.note_creation(
        kind=kind,
        title=None,        # untitled by design; let the content stand
        content=content,
        origin_journal_id=origin_journal_id,
        user_id=runtime.owner_user_id,
    )
    await runtime.bus.publish_topic(
        "behavior.creation_made",
        {
            "creation_id": creation_id,
            "kind": kind,
            "chars": len(content),
            "origin_journal_id": origin_journal_id,
        },
        source_companion_id=runtime.companion_id,
    )


async def _perform_observation(runtime: CompanionRuntime) -> None:
    # No-op until real observation generation lands (mirrors
    # _perform_scene_update below). Previously this wrote a literal
    # "(tick observation placeholder)" row into companion_observations
    # every time the activity was selected — junk in a user-facing table
    # (audit 2026-06-17). Honest no-op until there's real content to write.
    return None


async def _perform_scene_update(runtime: CompanionRuntime) -> None:
    # No-op for now — scene routing is XR-side. Sprint 5 wires this.
    return None


async def _perform_dream(runtime: CompanionRuntime) -> None:
    from augmentum.companion_runtime.behavior import sleep_wake
    await sleep_wake.invoke_dream(runtime)


async def _perform_reach_out(runtime: CompanionRuntime) -> None:
    await runtime.bus.publish_topic(
        "behavior.reach_out",
        {"note": "tick decided to reach"},
        source_companion_id=runtime.companion_id,
    )


async def _perform_no_op(runtime: CompanionRuntime) -> None:
    return None


# Piece 9' tuning. Daily cap counted against companion_journal.
# revisited_at; per-thread cooldown likewise. Conservative defaults —
# the resolver is cheap but autonomous loops without caps are how
# substrate eats production VRAM at 3am.
_REVISIT_DAILY_CAP: int = 6
_REVISIT_PER_THREAD_COOLDOWN_HOURS: int = 6


def _clip_phrase(text: str, limit: int) -> str:
    """Clip to ``limit`` chars at a word boundary, ellipsis on real cuts.

    The previous raw ``[:120]`` slices put visibly machine-cut prose in
    the drawer ("…the same structural need in the brow" — cut from
    "browser"). Mid-word truncation is the single fastest tell that a
    note was templated, not written.
    """
    s = (text or "").strip()
    if len(s) <= limit:
        return s
    cut = s[:limit].rsplit(" ", 1)[0].rstrip(" ,;:—–-")
    return (cut or s[:limit].rstrip()) + "…"


async def _perform_revisit_thread(runtime: CompanionRuntime) -> None:
    """Piece 9' — pick up the most recent unresolved thread, call the
    resolver against it, write findings back as a noticing.

    Composed entirely from substrate that exists:
      - companion_journal (the thread)
      - resolver.resolve_moments (the lookup)
      - companion_journal again (the noticing with content_refs)

    Resource-conscious by construction:
      - Daily cap (default 6) counted from revisited_at, gates the
        whole performance below the SELECT row.
      - Per-thread cooldown (default 6h) gates which row gets picked
        up; same thread can't re-fire until cooldown elapses.
      - Resolver call only happens after BOTH gates pass.
      - On no-thread / no-results / partial failure, we mark the
        proposal executed anyway so the next tick doesn't re-fire.
    """
    backend = runtime.backend
    # Owner-scope the revisit cap + eligibility so one user's revisits
    # don't exhaust another's daily budget or surface their threads
    # (audit 2026-06-17).
    owner = runtime.owner_user_id or ""

    # ── Daily cap (cheapest gate first) ──────────────────────────────
    cap_frag, cap_p = owner_clause(owner)
    try:
        cur = await backend.conn.execute(
            "SELECT COUNT(*) FROM companion_journal "
            "WHERE companion_id = ? "
            f"  AND revisited_at > datetime('now', '-1 day') {cap_frag}",
            (runtime.companion_id, *cap_p),
        )
        cap_row = await cur.fetchone()
        await cur.close()
    except Exception:
        log.warning("revisit_daily_cap_query_failed", exc_info=True)
        return
    daily_count = int((cap_row[0] if cap_row else 0) or 0)
    if daily_count >= _REVISIT_DAILY_CAP:
        log.info(
            "revisit_skipped_daily_cap",
            companion_id=runtime.companion_id, count=daily_count,
        )
        await _mark_revisit_proposals_executed(runtime, reason="daily_cap")
        return

    # ── Find an eligible thread (cooldown + suppression honored) ─────
    cooldown_sql = (
        f"datetime('now', '-{_REVISIT_PER_THREAD_COOLDOWN_HOURS} hours')"
    )
    elig_frag, elig_p = owner_clause(owner)
    try:
        cur = await backend.conn.execute(
            f"""
            SELECT id, user_id, content, origin_json
            FROM companion_journal
            WHERE companion_id = ?
              AND entry_type IN ('wondering', 'unfinished')
              AND COALESCE(suppressed, 0) = 0
              AND COALESCE(quarantined, 0) = 0
              AND user_id IS NOT NULL
              AND (revisited_at IS NULL OR revisited_at < {cooldown_sql})
              {elig_frag}
            ORDER BY created_at DESC
            LIMIT 1
            """,
            (runtime.companion_id, *elig_p),
        )
        thread_row = await cur.fetchone()
        await cur.close()
    except Exception:
        log.warning("revisit_thread_query_failed", exc_info=True)
        return

    if thread_row is None:
        # Nothing eligible — mark proposal executed so we don't re-fire.
        await _mark_revisit_proposals_executed(runtime, reason="no_thread")
        return

    thread_id = int(thread_row[0])
    thread_user_id = str(thread_row[1] or "")
    thread_content = str(thread_row[2] or "").strip()
    # Inherit the parent thread's provenance (mig 257). A revisit is
    # derived attention — the noticing's chip should still say which
    # client's signals started the thread, not pretend the loop-back
    # appeared from nowhere.
    parent_origin: dict = {}
    try:
        import json as _json
        parent_origin = _json.loads(thread_row[3] or "{}") or {}
        if not isinstance(parent_origin, dict):
            parent_origin = {}
    except Exception:
        parent_origin = {}
    if not thread_content or not thread_user_id:
        await _mark_revisit_proposals_executed(runtime, reason="empty_thread")
        return

    # Skip revisit composition for NSFW thread content. The thread row
    # may carry a domain like pornhub.com inside its content; without
    # this gate the "Came back to: …" composer would compose around that
    # text and either synthesize or fall back to a template that names
    # the domain in the noticing. safe_journal also quarantines the
    # output, but skipping early avoids an LLM round-trip per cycle.
    try:
        from augmentum.discovery.safety import is_nsfw_text
        if is_nsfw_text(thread_content):
            log.info(
                "revisit_skip_nsfw_thread",
                thread_id=thread_id, user_id=thread_user_id,
            )
            await _mark_revisit_proposals_executed(runtime, reason="nsfw_thread")
            return
    except Exception:
        log.debug("revisit_nsfw_check_failed", exc_info=True)

    # ── Resolver call (the expensive part — gated by everything above) ─
    file_index = getattr(getattr(runtime, "_app_state", None), "file_index", None)
    memory = getattr(runtime, "memory", None)
    try:
        from augmentum.resolver import resolve_moments
        moments = await resolve_moments(
            thread_content,
            user_id=thread_user_id,
            file_index=file_index,
            memory=memory,
            limit=3,
        )
    except Exception as exc:
        log.warning(
            "revisit_resolver_failed",
            thread_id=thread_id, error=str(exc)[:200],
        )
        moments = []

    # Drop journal-kind moments before synthesis. The resolver's journal
    # leg reliably returns *other wonderings/noticings* near-identical to
    # the thread, and the synthesize model then writes meta-commentary
    # about the journal itself ("'noticing' maps directly onto 'wondering'
    # because both entries track the same engagement…") — substrate
    # talking about substrate, surfaced to the user as if it were a
    # finding. Files, pages, and media are the things worth looping back
    # to; another copy of the thread is not.
    moments = [m for m in moments if m.kind != "journal"]

    # ── Sprint 2 Piece 8 — synthesize when moments exist ──────────────
    # Map new (the thread content) onto old (the resolved moments) via
    # one utility-tier LLM call. Empty output is the "no real connection"
    # contract; we then fall back to the "Letting it sit" template.
    synthesized_text = ""
    synthesize_model = None
    if moments:
        try:
            from augmentum.tools.synthesize import synthesize
            result = await synthesize(
                runtime,
                wondering_content=thread_content,
                moments=moments,
                user_id=thread_user_id,
                privacy_class="local_only",
            )
            if result.text and result.grounded:
                synthesized_text = result.text
                synthesize_model = result.model_used
            elif result.text and not result.grounded:
                log.info(
                    "revisit_synthesize_ungrounded",
                    thread_id=thread_id,
                    model=result.model_used,
                )
        except Exception as exc:
            log.warning(
                "revisit_synthesize_failed",
                thread_id=thread_id, error=str(exc)[:200],
            )

    # ── Write noticing with content_refs (regardless of result count) ─
    content_refs: list[dict] = [
        {"kind": m.kind, "id": str(m.id)} for m in moments[:3]
    ]
    if synthesized_text:
        # Sprint 2 Piece 8 path — model produced a grounded synthesis.
        notice = synthesized_text
        affect = "curious"
    elif moments:
        # Resolver found things but synthesize abstained or failed.
        # Compose a single field-note sentence instead of the structural
        # "Came back to: X\nFound: Y" template that read like a log line.
        top = moments[0]
        thread_phrase = _clip_phrase(thread_content.rstrip("."), 120) or "the thread"
        snippet_phrase = _clip_phrase(top.snippet.rstrip("."), 160)
        title_phrase = (top.title or "").strip().rstrip(".")
        anchor = title_phrase or snippet_phrase or "something nearby"
        notice = f"{thread_phrase} — looped back to {anchor}."
        affect = "curious"
    else:
        # Nothing resolved. Honest silence — don't pretend to have found.
        # Single sentence, no machine-shaped prefix.
        thread_phrase = _clip_phrase(thread_content.rstrip("."), 140) or "the thread"
        notice = f"Letting {thread_phrase} sit for now."
        affect = "patient"

    try:
        # Sprint 1 R1 — route through safe_journal so the validation
        # pipeline applies. Source='synthesize' when LLM produced the
        # text; 'autonomous' otherwise. model_used recorded for
        # post-hoc forensics (model swap analysis).
        #
        # ``surfaceable_default=bool(moments)`` keeps the empty-find
        # path out of the user pip: a "letting it sit" noticing is for
        # the interior, not for interrupting the user with "she looked
        # and found nothing". The explicit UPDATE below still flips
        # qsr=1 in the moments-found branch; passing the flag here
        # just stops the memory-layer auto-flag (affect_tag="patient"
        # is otherwise in the surfaceable set) from over-firing.
        write_source = "synthesize" if synthesized_text else "autonomous"
        noticing_id = await runtime.memory.safe_journal(
            notice,
            source=write_source,
            model_used=synthesize_model,
            user_id=thread_user_id,
            entry_type="noticing",
            affect_tag=affect,
            content_refs=content_refs,
            related_memory_ids=[str(thread_id)],
            surfaceable_default=bool(moments),
            origin={
                "source": "revisit",
                "client": str(parent_origin.get("client") or ""),
                "signal_count": int(parent_origin.get("signal_count") or 0),
                "window": str(parent_origin.get("window") or ""),
                "detail": f"looped back on note #{thread_id}",
            },
        )
    except Exception:
        log.warning("revisit_journal_write_failed", exc_info=True)
        noticing_id = 0

    # Revisits produce real affect — curious / patient / etc. Publish
    # on the bus so the avatar's expression can shift in response.
    # No-op for 'settled'; throttled to changes-only inside the helper.
    try:
        await runtime.publish_affect(affect, reason="revisit")
    except Exception:
        log.warning("revisit_publish_affect_failed", exc_info=True)

    # Piece 10': mark the noticing as quiet-share-ready when it actually
    # found something. The pip surfaces only entries with at least one
    # content_ref — "she looked it up but found nothing" doesn't merit
    # interrupting the user's attention. Best-effort UPDATE; failure
    # here just means the pip won't show this entry, which is safe
    # degradation.
    if noticing_id and moments:
        try:
            await backend.conn.execute(
                "UPDATE companion_journal SET quiet_share_ready = 1 "
                "WHERE id = ?",
                (noticing_id,),
            )
            await backend.conn.commit()
        except Exception:
            log.debug(
                "revisit_quiet_share_mark_failed",
                noticing_id=noticing_id, exc_info=True,
            )

    # ── Mark the original thread as revisited (idempotent) ──────────
    try:
        await backend.conn.execute(
            "UPDATE companion_journal SET revisited_at = datetime('now') "
            "WHERE id = ?",
            (thread_id,),
        )
        await backend.conn.commit()
    except Exception:
        log.warning("revisit_mark_failed", thread_id=thread_id, exc_info=True)

    await _mark_revisit_proposals_executed(runtime, reason="performed")

    log.info(
        "revisit_thread_performed",
        thread_id=thread_id,
        moments_found=len(moments),
    )


async def _mark_revisit_proposals_executed(
    runtime: CompanionRuntime, *, reason: str,
) -> None:
    """Mark any pending revisit_thread proposals as executed so the
    next tick's score function returns to 0.05 and we don't re-fire.

    Best-effort — a failure here means the next tick may pick this
    candidate again, but the per-thread cooldown + daily cap still
    bound the actual resolver calls. Idempotent: an already-executed
    proposal stays executed.
    """
    _m_frag, _m_p = owner_clause(runtime.owner_user_id or "")
    try:
        await runtime.backend.conn.execute(
            "UPDATE companion_initiative_queue "
            "SET status = 'executed' "
            "WHERE companion_id = ? "
            "  AND kind = 'revisit_thread' "
            f"  AND status = 'pending' {_m_frag}",
            (runtime.companion_id, *_m_p),
        )
        await runtime.backend.conn.commit()
    except Exception:
        log.debug(
            "revisit_proposals_mark_failed",
            reason=reason, exc_info=True,
        )


_PERFORMERS: dict[str, Callable] = {
    "journal": _perform_journal,
    "creation": _perform_creation,
    "observation": _perform_observation,
    "scene_update": _perform_scene_update,
    "dream_invocation": _perform_dream,
    "reach_out": _perform_reach_out,
    "revisit_thread": _perform_revisit_thread,
    "no_op": _perform_no_op,
}


# Sprint 6 — candidate-to-drive tagging. When drives are enabled, the
# final score for each candidate is multiplied by the matching drive's
# urgency. Drives also satiate when their tagged candidate fires.
_CANDIDATE_DRIVES: dict[str, str] = {
    "journal": "connection",         # private inner stream, but oriented toward the relationship
    "creation": "competence",
    "observation": "competence",
    "scene_update": "rest",
    "dream_invocation": "rest",
    "reach_out": "connection",
    "revisit_thread": "curiosity",
    "no_op": "rest",
}


def _energy_factor(level: float, baseline: float) -> float:
    """Capacity multiplier for OUTWARD (non-rest) activities.

    1.0 at or above baseline (no damping); falls toward a floor as the level
    drops below baseline, so low energy suppresses what she INITIATES and the
    rest-tagged candidates win until she recovers (the act -> deplete -> rest
    -> recover duty cycle). Never 0 — a strong-enough appetite can still act,
    just less readily. Energy never touches responsiveness; the wall for that
    is ``tests/test_responsiveness_invariant.py``.
    """
    if baseline <= 0:
        return 1.0
    return max(0.15, min(1.0, level / baseline))


# ── Selection ────────────────────────────────────────────────────────

async def choose(runtime: CompanionRuntime, *, role: str) -> ActivityChoice | None:
    """Score every candidate, sample via softmax, return the winner.

    Sprint 6 — when ``companion_drives_enabled`` is on, each candidate's
    base score is multiplied by its tagged drive's urgency. Drives
    satiate when matching candidate performs (see ``_satiate_for``
    helper, called below).
    """
    state_snap = runtime.state.snapshot()
    state_axis = state_snap.get("state", "dormant")
    temperature = _STATE_TEMPERATURE.get(state_axis, 0.6)
    if temperature == 0.0:
        return None

    # Resolve drives once per tick if enabled. Phase 3a — decay is now
    # owned by the ``tick_drive`` management verb (every 60s), so the
    # activity selector reads the substrate directly via ``load`` rather
    # than re-running decay per tick.
    drive_state = None
    drives_enabled = False
    try:
        from augmentum.config import settings
        if getattr(settings, "companion_drives_enabled", False):
            owner = getattr(runtime, "owner_user_id", "") or ""
            if owner:
                from augmentum.companion_runtime import drives as _drives
                drive_state = await _drives.load(runtime, user_id=owner)
                drives_enabled = True
    except Exception:
        log.debug("drives_resolution_failed", exc_info=True)

    # Energy gate — capacity to act. When enabled, low energy damps OUTWARD
    # (non-rest) candidates so rest wins and she recovers; rest-tagged
    # candidates are never damped. Mirrors the drives block above. Energy
    # NEVER gates responsiveness — only autonomous initiation (the wall is
    # tests/test_responsiveness_invariant.py).
    energy_factor = 1.0
    try:
        from augmentum.config import settings as _settings
        if getattr(_settings, "companion_energy_enabled", False):
            owner = getattr(runtime, "owner_user_id", "") or ""
            if owner:
                from augmentum.companion_runtime import energy as _energy
                _energy_state = await _energy.load(runtime, user_id=owner)
                energy_factor = _energy_factor(
                    _energy_state.level, _energy_state.baseline
                )
    except Exception:
        log.debug("energy_resolution_failed", exc_info=True)

    scored: list[tuple[str, float]] = []
    for name, scorer in _CANDIDATES:
        try:
            raw = await scorer(runtime, role=role)
        except Exception:
            log.exception("activity_score_failed", kind=name)
            raw = 0.0
        base = max(0.0, raw)
        if drives_enabled and drive_state is not None:
            drive = _CANDIDATE_DRIVES.get(name, "rest")
            urgency = drive_state.urgency(drive)
            # Multiplicative modulation. Urgency ~1.0 = no change;
            # higher = boost; lower = damp. Squash to keep softmax stable.
            base *= max(0.1, min(2.0, 0.5 + urgency))
        if energy_factor < 1.0 and _CANDIDATE_DRIVES.get(name, "rest") != "rest":
            # Low energy damps what she INITIATES; rest-tagged candidates
            # (no_op/scene_update/dream) are recovery and are never damped.
            base *= energy_factor
        scored.append((name, base))

    # Softmax over scores. Add small epsilon to avoid 0/0 when all
    # scores are zero (we'd pick uniformly, which is fine).
    eps = 1e-6
    exps = [math.exp(s / max(temperature, eps)) for _, s in scored]
    total = sum(exps) or 1.0
    probs = [e / total for e in exps]

    r = random.random()
    acc = 0.0
    chosen_idx = 0
    for i, p in enumerate(probs):
        acc += p
        if r <= acc:
            chosen_idx = i
            break

    kind, utility = scored[chosen_idx]
    base_performer = _PERFORMERS.get(kind, _perform_no_op)
    drive = _CANDIDATE_DRIVES.get(kind, "rest")

    # Phase 3a — satiation has moved to the ``apply_signal`` management
    # verb, which subscribes to ``behavior.activity_chosen`` and reads
    # ``drive`` from the event payload. The previous wrapper is gone;
    # the performer is now called bare.
    return ActivityChoice(
        kind=kind,
        utility=utility,
        threshold=_ACT_THRESHOLD,
        perform=base_performer,
        drive=drive,
    )


__all__ = ["ActivityChoice", "choose"]
