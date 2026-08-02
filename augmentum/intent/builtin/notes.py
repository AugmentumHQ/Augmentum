"""Notes / memory primitive verbs.

These are the first batch of "compose-able" actions Becca can use to
take notes, capture thoughts, and persist into memory. Each is a single
verb with a focused arg schema — the LLM (Phase 9) becomes the
orchestrator that picks + chains them based on what the user said.

For v1 they ALSO accept light regex matches (auto-derived from
``examples``) so the user can reach them via direct phrasing today
while the LLM-tool layer is being wired up.

Session-level state — the active note ID and capture mode — lives on
the per-session ReferentCache so subsequent verbs default sensibly
("append" without an explicit note_id appends to the most recently
created one).
"""

from __future__ import annotations

import time
import uuid
from datetime import UTC, datetime
from typing import Any

from augmentum.intent.action import ActionFanout, ActionResult, SessionContext
from augmentum.intent.dispatch import get_referent_cache
from augmentum.intent.registry import register_action
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Capture mode auto-exits after this many seconds of inactivity. The
# deadline is refreshed on every captured utterance — only persistent
# silence (e.g., user navigates away, browser idle) triggers the
# auto-exit. Keeps the server from holding capture state forever for
# a session that the user moved on from.
_CAPTURE_IDLE_TIMEOUT_S = 300.0  # 5 minutes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _notes_store(session: SessionContext):
    """Resolve the per-app notes store, or None if app state missing."""
    if session.app_state is None:
        return None
    return getattr(session.app_state, "notes_store", None)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _new_note_id() -> str:
    return uuid.uuid4().hex[:12]


def _refs(session: SessionContext):
    return get_referent_cache(
        session.app_state, session.user_id, session.session_id,
    )


# ---------------------------------------------------------------------------
# create_note
# ---------------------------------------------------------------------------

async def _create_note(
    text: str, session: SessionContext, args: dict[str, Any],
) -> ActionResult | None:
    if not session.user_id:
        # Refuse to write to the anon row — multi-tenant invariant.
        log.warning("intent_create_note_no_user")
        return ActionResult(
            short_circuit=True,
            speak="I'm not sure who to save that for.",
        )
    store = _notes_store(session)
    if store is None:
        return ActionResult(
            short_circuit=True,
            speak="I can't reach the notes store right now.",
        )
    title = (args.get("title") or "").strip() or "Untitled"
    content = (args.get("content") or "").strip()

    # Idempotency — same (title, content) within the recent window
    # returns the existing note instead of creating a duplicate. This
    # catches LLM tool-call retries and double-trigger from the user.
    refs = _refs(session)
    fingerprint = f"{title}|{content}"
    existing_id = refs.recent_note_fingerprints.get(fingerprint)
    if existing_id:
        log.info("intent_create_note_idempotent", note_id=existing_id)
        refs.active_note_id = existing_id
        refs.active_note_title = title
        return ActionResult(
            short_circuit=True,
            surface_emit={
                "channel": "note.open_sticky",
                "payload": {
                    "note_id": existing_id,
                    "title": title,
                    "content": content,
                },
            },
            speak="Here it is.",
        )

    note = {
        "id": _new_note_id(),
        "title": title,
        "content": content,
        "tags": [],
        "source_url": "",
        "source_title": "",
        "format": "note",
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
        # Provenance, not silos: her notes live in the same list as
        # the user's, distinguished only by origin (UI filter chip).
        "origin": "companion",
    }
    try:
        await store.create(note, user_id=session.user_id)
    except Exception as exc:
        log.warning("intent_create_note_failed", error=str(exc))
        return ActionResult(
            short_circuit=True,
            speak="Something went wrong saving that note.",
        )

    refs.active_note_id = note["id"]
    refs.active_note_title = title
    # FIFO-bound the fingerprint map at 32 entries so it can't grow
    # indefinitely over a long session.
    refs.recent_note_fingerprints[fingerprint] = note["id"]
    if len(refs.recent_note_fingerprints) > 32:
        # Drop the oldest entry (insertion order is preserved in dict).
        oldest_key = next(iter(refs.recent_note_fingerprints))
        refs.recent_note_fingerprints.pop(oldest_key, None)

    return ActionResult(
        short_circuit=True,
        surface_emit={
            "channel": "note.open_sticky",
            "payload": {
                "note_id": note["id"],
                "title": title,
                "content": content,
            },
        },
        speak="Sure, here you go.",
    )


register_action(
    id="note.create",
    summary="Create a new note. Surfaces as a draggable sticky-note overlay.",
    examples=[
        "open a new note",
        "make me a new note",
        "create a note",
        "jot something down",
        "open a new note for me please",
        "start a fresh note",
    ],
    patterns=[
        # "open a new note", "open up a new note", "make me a quick note",
        # "create a fresh note", "pull up a note", "start a new note"
        r"\b(?:open|make|create|start|pull)\s+(?:up\s+)?(?:me\s+)?"
        r"(?:a\s+)?(?:new|fresh|quick|blank)?\s*note\b",
        # "jot something down", "jot this down", "jot down a note"
        r"\bjot(?:\s+(?:something|this|that|it))?\s+down\b",
        r"\bjot\s+down\b",
        # "take a note" — careful: distinct from "take notes on this" which
        # is the capture-mode trigger handled below.
        r"\btake\s+a\s+note\b",
    ],
    arg_schema={
        "title": {
            "type": "string",
            "description": (
                "Optional title for the note. If omitted, a default is "
                "used and the note shows as 'Untitled' until the user "
                "or the model renames it."
            ),
        },
        "content": {
            "type": "string",
            "description": (
                "Optional initial body. Useful when the user already "
                "named the topic in the same breath as the request."
            ),
        },
    },
    handler=_create_note,
    # The sticky appearing on screen is the feedback; her own short
    # speak line ("Sure, here you go.") is the only voiced ack.
    delivery="artifact",
)


# ---------------------------------------------------------------------------
# append_to_note
# ---------------------------------------------------------------------------

async def _append_to_note(
    text: str, session: SessionContext, args: dict[str, Any],
) -> ActionResult | None:
    if not session.user_id:
        log.warning("intent_append_note_no_user")
        return None
    store = _notes_store(session)
    if store is None:
        return None
    refs = _refs(session)
    note_id = (args.get("note_id") or refs.active_note_id or "").strip()
    if not note_id:
        return ActionResult(
            short_circuit=True,
            speak="I don't have an open note to add to.",
        )
    addition = (args.get("content") or "").strip()
    if not addition:
        return None

    try:
        existing = await store.get(note_id, user_id=session.user_id)
    except Exception:
        existing = None
    if not existing:
        return ActionResult(
            short_circuit=True,
            speak="I couldn't find that note.",
        )
    new_content = (existing.get("content") or "")
    if new_content and not new_content.endswith("\n"):
        new_content += "\n"
    new_content += addition

    try:
        await store.update(
            note_id,
            {"content": new_content, "updated_at": _now_iso()},
            user_id=session.user_id,
        )
    except Exception as exc:
        log.warning("intent_append_note_failed", error=str(exc))
        return ActionResult(
            short_circuit=True,
            speak="I couldn't update the note.",
        )

    return ActionResult(
        short_circuit=True,
        surface_emit={
            "channel": "note.update_sticky",
            "payload": {
                "note_id": note_id,
                "content": new_content,
                # Title rides along so a sticky surfaced BY this update
                # (closed or never opened) doesn't render blank-titled.
                "title": (existing.get("title") or "").strip(),
            },
        },
        # No spoken ack — appending mid-flow shouldn't interrupt.
    )


register_action(
    id="note.append",
    summary=(
        "Append content to a note. Defaults to the currently active "
        "sticky note when ``note_id`` is omitted."
    ),
    examples=[
        "add this to the note",
        "append to the note",
        "add to my note",
    ],
    patterns=[
        r"\b(?:add|append)\s+(?:this|that|it)?\s*(?:to (?:the |my )?note)\b",
    ],
    arg_schema={
        "content": {
            "type": "string",
            "description": "Text to append.",
        },
        "note_id": {
            "type": "string",
            "description": (
                "Specific note. Omit to append to the active note from "
                "the per-session referent cache."
            ),
        },
    },
    required=["content"],
    handler=_append_to_note,
    # Co-author register: the sticky note updating on screen IS the
    # feedback. Voice must not announce the write — the conversation
    # stays about the content.
    delivery="artifact",
)


# ---------------------------------------------------------------------------
# attach_image
# ---------------------------------------------------------------------------

async def _attach_image(
    text: str, session: SessionContext, args: dict[str, Any],
) -> ActionResult | None:
    """Append an image to a note as a markdown line.

    Markdown IS the canonical representation — no schema change, the
    full notes editor renders it, and deleting the line in the sticky
    textarea deletes the image. The sticky renders these lines as an
    attachment strip client-side. URLs come from upstream tool results
    in the same loop turn: image_generation reports its gallery url,
    web results carry theirs.
    """
    if not session.user_id:
        log.warning("intent_attach_image_no_user")
        return ActionResult(
            short_circuit=True,
            speak="I can't add images for a signed-out session.",
        )
    store = _notes_store(session)
    if store is None:
        return ActionResult(
            short_circuit=True,
            speak="I can't reach the notes store right now.",
        )
    refs = _refs(session)
    note_id = (args.get("note_id") or refs.active_note_id or "").strip()
    if not note_id:
        return ActionResult(
            short_circuit=True,
            speak="I don't have an open note to add that to.",
        )
    url = (args.get("url") or "").strip()
    if not url:
        return None
    # Relative gallery urls (/api/image/...) and absolute http(s) only —
    # anything else (javascript:, data: blobs) stays out of the note.
    if not (url.startswith("/") or url.startswith(("http://", "https://"))):
        return ActionResult(
            short_circuit=True,
            speak="That image link doesn't look like something I can attach.",
        )
    caption = (args.get("caption") or "").strip()
    # Keep the markdown well-formed: brackets/parens in user-ish text
    # would break the image line.
    caption = caption.replace("]", "").replace("[", "")[:120]
    url = url.replace(")", "%29").replace("(", "%28")

    try:
        existing = await store.get(note_id, user_id=session.user_id)
    except Exception:
        existing = None
    if not existing:
        return ActionResult(
            short_circuit=True,
            speak="I couldn't find that note.",
        )
    new_content = (existing.get("content") or "")
    if new_content and not new_content.endswith("\n"):
        new_content += "\n"
    new_content += f"![{caption or 'image'}]({url})\n"

    try:
        await store.update(
            note_id,
            {"content": new_content, "updated_at": _now_iso()},
            user_id=session.user_id,
        )
    except Exception as exc:
        log.warning("intent_attach_image_failed", error=str(exc))
        return ActionResult(
            short_circuit=True,
            speak="I couldn't add the image to the note.",
        )

    return ActionResult(
        short_circuit=True,
        surface_emit={
            "channel": "note.update_sticky",
            "payload": {
                "note_id": note_id,
                "content": new_content,
                "title": (existing.get("title") or "").strip(),
            },
        },
        # Silent like append — the thumbnail landing IS the feedback.
    )


register_action(
    id="note.attach_image",
    summary=(
        "Add an image to a note by URL — one the user asked you to "
        "generate, or one found while searching. The image appears in "
        "the note's attachment strip. Use the url reported by the "
        "image_generation or search tool result in this conversation."
    ),
    examples=[
        "add that image to the note",
        "put the picture in my note",
        "attach the image you made to the note",
        "save that photo to the note",
    ],
    arg_schema={
        "url": {
            "type": "string",
            "description": (
                "Image URL — the gallery url from image_generation "
                "(/api/image/...) or an http(s) image link from search."
            ),
        },
        "caption": {
            "type": "string",
            "description": "Optional short caption shown with the image.",
        },
        "note_id": {
            "type": "string",
            "description": (
                "Specific note. Omit to attach to the active note."
            ),
        },
    },
    required=["url"],
    fanout=ActionFanout(tier1=False, tier2=False, tier3=True),
    handler=_attach_image,
    delivery="artifact",
)


# ---------------------------------------------------------------------------
# show_sticky
# ---------------------------------------------------------------------------

async def _show_sticky(
    text: str, session: SessionContext, args: dict[str, Any],
) -> ActionResult | None:
    store = _notes_store(session)
    refs = _refs(session)
    note_id = (args.get("note_id") or refs.active_note_id or "").strip()
    if not note_id or store is None:
        return ActionResult(
            short_circuit=True,
            speak="There's no recent note to show.",
        )
    try:
        note = await store.get(note_id, user_id=session.user_id)
    except Exception:
        note = None
    if not note:
        return ActionResult(
            short_circuit=True,
            speak="I couldn't find that note.",
        )
    refs.active_note_id = note_id
    refs.active_note_title = note.get("title") or "Untitled"
    return ActionResult(
        short_circuit=True,
        surface_emit={
            "channel": "note.open_sticky",
            "payload": {
                "note_id": note_id,
                "title": note.get("title") or "Untitled",
                "content": note.get("content") or "",
            },
        },
    )


register_action(
    id="note.show_sticky",
    summary="Surface a note as a draggable sticky-note overlay.",
    examples=[
        "show me the note",
        "bring up that note",
        "show the sticky",
    ],
    arg_schema={
        "note_id": {
            "type": "string",
            "description": (
                "Specific note to surface. Omit to show the active "
                "note from the per-session referent cache."
            ),
        },
    },
    handler=_show_sticky,
    delivery="artifact",
)


# ---------------------------------------------------------------------------
# start_note_capture / end_note_capture
# ---------------------------------------------------------------------------

async def _start_capture(
    text: str, session: SessionContext, args: dict[str, Any],
) -> ActionResult | None:
    refs = _refs(session)
    # If no active note, the model is expected to call note.create first
    # — but if it skipped that step, we create a quick one so the
    # capture mode has somewhere to land.
    if not refs.active_note_id:
        seed = await _create_note(text, session, {
            "title": (args.get("title") or "Captured thoughts"),
        })
        if seed is None or refs.active_note_id is None:
            return ActionResult(
                short_circuit=True,
                speak="I couldn't start a note for that.",
            )
    # Record the current note length so the end-of-capture cleanup pass
    # knows which slice was dictated (vs pre-existing content). Read
    # AFTER the auto-create-on-missing branch so a brand-new seed note
    # gets baseline=0 (its whole body is the capture). Fall through to
    # baseline=0 on any read failure — the cleanup pass treats 0 as
    # "no baseline recorded, skip cleanup" which is the safe default.
    refs.note_capture_baseline_chars = 0
    store = _notes_store(session)
    if store is not None and session.user_id:
        try:
            existing = await store.get(refs.active_note_id, user_id=session.user_id)
            if existing:
                refs.note_capture_baseline_chars = len(existing.get("content") or "")
        except Exception as exc:
            # Baseline=0 fallback means the end-of-capture cleanup will
            # skip (treats 0 as "no baseline recorded") so we won't
            # rewrite content we don't know the boundary of. Worth a
            # warning so operators notice if it happens often.
            log.warning("intent_capture_baseline_read_failed", error=str(exc)[:160])
    refs.note_capture_mode = True
    refs.note_capture_deadline = time.monotonic() + _CAPTURE_IDLE_TIMEOUT_S
    return ActionResult(
        short_circuit=True,
        surface_emit={
            "channel": "note.capture_started",
            "payload": {"note_id": refs.active_note_id},
        },
        speak="Go ahead. I'm listening.",
    )


register_action(
    id="note.start_capture",
    summary=(
        "Enter thought-capture mode. Subsequent utterances are appended "
        "to the active note (after light LLM cleanup) instead of being "
        "treated as conversational turns. Exits via note.end_capture or "
        "common exit phrases handled by Tier 1."
    ),
    examples=[
        "translate my thoughts",
        "capture my thoughts",
        "take notes on this",
        "I'm thinking out loud",
        "jot my thoughts down",
    ],
    patterns=[
        r"\b(?:translate|capture|take notes on|write down)\s+(?:my\s+)?thoughts?\b",
        r"\btake notes on (?:this|that|what)\b",
        r"\bjot\s+(?:my\s+)?(?:thoughts?|ideas?)\s+down\b",
        r"\bI'?m\s+thinking\s+(?:out\s+loud|about)\b",
        r"\b(?:start|begin)\s+(?:a\s+)?brain\s*storm\b",
    ],
    arg_schema={
        "title": {
            "type": "string",
            "description": "Optional title for an auto-created note.",
        },
    },
    handler=_start_capture,
    # "Go ahead. I'm listening." is the whole handshake — no synth pass.
    delivery="artifact",
)


async def _end_capture(
    text: str, session: SessionContext, args: dict[str, Any],
) -> ActionResult | None:
    refs = _refs(session)
    was_active = refs.note_capture_mode
    note_id = refs.active_note_id or ""
    baseline = refs.note_capture_baseline_chars or 0
    refs.note_capture_mode = False
    refs.note_capture_deadline = 0.0
    refs.note_capture_baseline_chars = 0
    if not was_active:
        return None

    # Run the LLM cleanup over the captured slice. Failures + the
    # operator-off setting both fall back to the raw transcript, so
    # this never blocks the end-capture acknowledgement.
    cleaned_content: str | None = None
    try:
        from augmentum.intent.capture_cleanup import apply_cleanup_to_note
        store = _notes_store(session)
        changed, new_content = await apply_cleanup_to_note(
            store,
            note_id,
            user_id=session.user_id,
            baseline_chars=baseline,
            app_state=session.app_state,
        )
        if changed:
            cleaned_content = new_content
    except Exception as exc:
        log.warning("intent_end_capture_cleanup_failed", error=str(exc)[:200])

    # Surface the post-cleanup content so the sticky overlay refreshes
    # in place. Frontend uses note.update_sticky for the live edit and
    # note.capture_ended for the lifecycle event — emit both when we
    # have cleaned content, just the lifecycle event otherwise.
    payload: dict[str, Any] = {"note_id": note_id}
    if cleaned_content is not None:
        payload["content"] = cleaned_content
    return ActionResult(
        short_circuit=True,
        surface_emit={
            "channel": "note.capture_ended",
            "payload": payload,
        },
        speak="Saved.",
    )


register_action(
    id="note.end_capture",
    summary="Exit thought-capture mode.",
    examples=[
        "save this",
        "okay that's enough",
        "stop noting",
        "we're done",
        "save the note",
    ],
    patterns=[
        # "save this", "save that", "save the note" — but NOT
        # "save this to memory" (memory.save's domain). Negative
        # lookahead refuses a continuation word.
        r"\bsave (?:this|the note|that)(?!\s+\w)",
        r"\bstop (?:noting|taking notes)\b",
        r"\bthat'?s enough\b",
        r"\bok(?:ay)? (?:we'?re )?done\b",
    ],
    handler=_end_capture,
    delivery="artifact",
)


# ---------------------------------------------------------------------------
# save_to_memory
# ---------------------------------------------------------------------------

async def _save_to_memory(
    text: str, session: SessionContext, args: dict[str, Any],
) -> ActionResult | None:
    if not session.user_id:
        log.warning("intent_save_memory_no_user")
        return ActionResult(
            short_circuit=True,
            speak="I'm not sure whose memory to update.",
        )
    # Defer to the existing memory subsystem. We import lazily so this
    # module stays import-safe in test contexts that don't pull in the
    # memory stack.
    content = (args.get("content") or "").strip()
    if not content:
        return ActionResult(
            short_circuit=True,
            speak="What should I remember?",
        )
    tags = args.get("tags") or []
    if not isinstance(tags, list):
        tags = []

    try:
        from augmentum.memory.models import MemoryType, SourceType
        from augmentum.memory.store import MemoryStore
        store = getattr(session.app_state, "memory_store", None)
        if not isinstance(store, MemoryStore):
            store = None
    except Exception:
        store = None

    if store is None:
        return ActionResult(
            short_circuit=True,
            speak="The memory store isn't available.",
        )

    # Update-don't-duplicate (wiring program Phase 2, from the Claude
    # write-discipline comparison): a near-identical existing fact
    # gets SUPERSEDED instead of piled. Targets the retry/double-
    # trigger class only — same-subject-new-value changes flow through
    # the extractor's supersede lane, not a loosened threshold here.
    duplicate_of = None
    try:
        from augmentum.intent.builtin.memory_admin import near_duplicate
        existing = await store.recall(
            content, user_id=session.user_id, limit=3,
        )
        for hit in existing or []:
            if near_duplicate(content, getattr(hit, "content", "") or ""):
                duplicate_of = hit
                break
    except Exception as exc:  # noqa: BLE001 — dedup is best-effort
        log.warning("intent_save_memory_dedup_failed", error=str(exc))

    try:
        if duplicate_of is not None:
            await store.supersede(
                duplicate_of.id,
                content,
                user_id=session.user_id,
                memory_type=MemoryType.FACT,
                session_id=session.session_id or None,
                importance=0.6,
                confidence=1.0,
                source_type=SourceType.EXPLICIT,
                source_context={"tags": list(tags)} if tags else None,
            )
            log.info(
                "intent_save_memory_superseded",
                user_id=session.user_id, old_id=duplicate_of.id,
            )
            return ActionResult(
                short_circuit=True,
                speak="Got it — updated what I had.",
                toast="Memory updated",
            )
        await store.store(
            content=content,
            memory_type=MemoryType.FACT,
            user_id=session.user_id,
            session_id=session.session_id or None,
            importance=0.6,
            confidence=1.0,
            source_type=SourceType.EXPLICIT,
            source_context={"tags": list(tags)} if tags else None,
        )
    except Exception as exc:
        log.warning("intent_save_memory_failed", error=str(exc))
        return ActionResult(
            short_circuit=True,
            speak="I couldn't save that.",
        )

    return ActionResult(
        short_circuit=True,
        speak="Got it. I'll remember.",
        toast="Saved to memory",
    )


register_action(
    id="memory.save",
    summary=(
        "Persist a fact / preference / context-of-record into long-term "
        "memory so future turns can recall it without re-asking."
    ),
    examples=[
        "remember that I prefer dark mode",
        "save this to memory",
        "remember this",
        "make a note of that",
    ],
    patterns=[
        r"\bremember\s+(?:that|this)\b",
        r"\bsave (?:this|that) to memory\b",
        r"\bmake a note of (?:that|this)\b",
    ],
    arg_schema={
        "content": {
            "type": "string",
            "description": "The fact / preference to remember.",
        },
        "tags": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Optional tags for later recall scoping.",
        },
    },
    required=["content"],
    handler=_save_to_memory,
    # "Got it. I'll remember." is the complete authored ack — a
    # synthesize pass on top is double-speak. memory.recall stays
    # verbal: recalled facts are data she composes from.
    delivery="artifact",
)


# ---------------------------------------------------------------------------
# recall
# ---------------------------------------------------------------------------

# Generic self-reference "queries" a small model emits for a topic-less
# "what do you remember about me?" — these route to the standing-set
# read-back, not hybrid recall (hybrid recall on "me" is noise).
_GENERIC_SELF_QUERIES = frozenset({
    "me", "about me", "myself", "everything", "anything", "all",
    "everything about me", "anything about me", "us", "about us",
})
# The earned standing set worth reading back, in priority order. CORE
# first ("holding close"), then ACTIVE. PROVISIONAL (unproven, never
# injected) and ARCHIVE (deliberately tucked away) are excluded — the
# subtractive bar means she recites only what's earned its place.
_STANDING_TIERS = ("core", "active")
_STANDING_SET_MAX = 10


async def _recall_standing_set(
    session: SessionContext, store,
) -> ActionResult:
    """Topic-less "what do you remember about me?" — the subtractive pull.

    Reads back the earned standing set (CORE + ACTIVE) grouped and
    honest, instead of always injecting it every turn. honest_gap-
    aligned when the pile is empty: say the true thing, don't invent.
    """
    hits: list = []
    try:
        for tier in _STANDING_TIERS:
            rows = await store.list_all(
                user_id=session.user_id, tier=tier, limit=_STANDING_SET_MAX,
            )
            hits.extend(rows)
            if len(hits) >= _STANDING_SET_MAX:
                break
    except Exception as exc:
        log.warning("intent_recall_standing_failed", error=str(exc))
        return ActionResult(
            short_circuit=True,
            speak="I couldn't reach my memory just now.",
        )

    hits = hits[:_STANDING_SET_MAX]
    if not hits:
        return ActionResult(
            short_circuit=True,
            speak=(
                "Honestly, not much yet — only what you've actually told me "
                "sticks, and it earns its place over time. Tell me what "
                "matters and I'll hold onto it."
            ),
            digest="recall about-me — nothing earned yet",
        )

    lines = []
    for h in hits:
        tier = getattr(h, "tier", "")
        tier = tier if isinstance(tier, str) else getattr(tier, "value", "")
        weight = "holding close" if tier == "core" else "remember"
        content = (getattr(h, "content", "") or "")[:160]
        if content:
            lines.append(f"[{weight}] {content}")
    addendum = (
        "<what_you_remember>\n"
        "The user asked what you remember about them. These are the facts "
        "you actually have saved — share them in your own voice, naturally "
        "and grouped, NOT as a numbered list or data dump. 'holding close' "
        "matters most. Mention the few that feel relevant; you needn't "
        "recite all of them, and be plainly honest about how much you have. "
        "Don't invent anything beyond this list.\n"
        + "\n".join(lines)
        + "\n</what_you_remember>"
    )
    return ActionResult(short_circuit=False, prompt_addendum=addendum)


async def _recall(
    text: str, session: SessionContext, args: dict[str, Any],
) -> ActionResult | None:
    query = (args.get("query") or "").strip()
    store = getattr(session.app_state, "memory_store", None)
    if store is None:
        return None
    # Topic-less "what do you remember about me?" → the earned standing
    # set. A small model often passes a generic self-reference ("me")
    # instead of omitting the arg; treat those as topic-less too.
    if not query or query.lower() in _GENERIC_SELF_QUERIES:
        return await _recall_standing_set(session, store)
    try:
        # MemoryStore.recall does hybrid vec + FTS retrieval. Cap at 5
        # for a voice-shaped reply.
        hits = await store.recall(
            query, user_id=session.user_id, limit=5,
        )
    except Exception as exc:
        log.warning("intent_recall_failed", error=str(exc))
        return ActionResult(
            short_circuit=True,
            speak="I couldn't search memory right now.",
        )

    if not hits:
        return ActionResult(
            short_circuit=True,
            speak=f"I don't have anything saved about {query}.",
        )

    # Soft augmentation — let the model summarize the hits naturally.
    # Returning prompt_addendum (not short_circuit) so the LLM still
    # composes the spoken reply with context. Each hit carries its
    # age (Phase 2 staleness honesty): "you told me this in March"
    # hedging beats confidently asserting a stale preference.
    from augmentum.intent.builtin.memory_admin import age_phrase
    lines = []
    for h in hits[:5]:
        c = getattr(h, "content", None) or (
            h.get("content") if isinstance(h, dict) else ""
        )
        if not c:
            continue
        created = getattr(h, "created_at", "") or (
            h.get("created_at", "") if isinstance(h, dict) else ""
        )
        when = age_phrase(created)
        lines.append(f"- {c}" + (f" (saved {when})" if when else ""))
    addendum = (
        "<recall_hits>\n"
        f"User asked about: {query}\n"
        + "\n".join(lines)
        + "\nOlder facts may be stale — hedge or ask if it matters."
        + "\n</recall_hits>"
    )
    return ActionResult(
        short_circuit=False,
        prompt_addendum=addendum,
    )


register_action(
    id="memory.recall",
    summary=(
        "Read back facts saved ABOUT the user. With a query, pulls the "
        "matching facts ('what did I say about that book?'). With NO "
        "query, reads back your earned standing set for a general 'what "
        "do you remember about me?' — only what's earned its place "
        "(never unproven or archived memories), grouped and honest. The "
        "model composes the spoken reply naturally. Siblings: your OWN "
        "inner life (wonderings, dreams, things you've noticed) is "
        "companion.introspect."
    ),
    examples=[
        "what did I say about that book",
        "do you remember the project name",
        "recall what we discussed about X",
        "what do you remember about me",
        "what do you know about me",
        "tell me what you remember about me",
    ],
    arg_schema={
        "query": {
            "type": "string",
            "description": (
                "What to recall, in the user's words. LEAVE EMPTY for a "
                "general 'what do you remember about me?' — that reads "
                "back the standing set rather than searching a topic."
            ),
        },
    },
    handler=_recall,
)
