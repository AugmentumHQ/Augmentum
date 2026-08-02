"""Cardsmith API — co-design pipeline for new character cards.

Routes
------
- ``POST /api/characters/cardsmith/start``     create a session, return its id
- ``POST /api/characters/cardsmith/turn``      SSE stream — one Cardsmith reply
- ``POST /api/characters/cardsmith/finalize``  save accumulated card explicitly
- ``POST /api/characters/cardsmith/cancel``    drop the session without saving

The frontend calls /start once when the user picks "Describe with AI" in the
new-card modal, then opens an SSE per /turn for every back-and-forth message.
On the model's final reply (after the user confirms) the model emits the
literal sentinel ``[CARDSMITH_DONE]`` — the route detects it, runs finalize
inline, and emits a ``finalized`` event with the saved char_id.

Streaming event protocol (over SSE, ``text/event-stream``):
    data: {"type":"delta","text":"<visible text chunk>"}
    data: {"type":"field","path":"description","value":"..."}
    data: {"type":"finalized","char_id":"ch_...","name":"Lyra"}
    data: {"type":"error","error":"..."}
    data: [DONE]
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from augmentum.config import settings
from augmentum.knowledge.content_extractor import (
    ContentDoc,
    ContentExtractError,
    fetch_content_doc,
    fetch_path,
)
from augmentum.models.base import InternalChatRequest, Message
from augmentum.modes.narrative.cardsmith import (
    build_character_payload,
    drop_session,
    get_or_create_session,
    get_prompt,
    get_session,
)
from augmentum.modes.narrative.cardsmith.parser import StreamingFieldParser
from augmentum.modes.narrative.cardsmith.scratchpad import (
    ScratchEntry,
    add_to_scratchpad,
    build_reference_index,
    deserialize_scratchpad,
    recall_for_turn,
    render_scratchpad_block,
    serialize_scratchpad,
)
from augmentum.modes.narrative.cardsmith.state import (
    register_session,
    serialize_for_disk,
    session_from_row,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/characters/cardsmith", tags=["cardsmith"])


_SUPPORTED_TYPES = {"single", "ensemble", "world_rpg"}  # Phase 4: all three card types live.
_SUPPORTED_SOURCES = {"describe", "blank", "wiki"}  # Phase 2: Wiki lane wired.


# Prepended to the Cardsmith system prompt when a scratchpad is non-empty.
# Teaches the model: there are external sources in your working memory,
# you can request more via fetch_targets[], the recall layer surfaces
# anything the user mentions by name.
_SCRATCHPAD_ADDENDUM = """\
# External sources

The user has attached one or more external sources. Their content lives
in the <scratchpad> block below — your working memory across the
conversation. Each turn you receive the same scratchpad with whatever
documents you've fetched so far.

Zones in the scratchpad:
  active   — full content. Recently fetched, not yet used in a card field.
  indexed  — title + 1-line digest only. Older, still queryable.
  consumed — already used in a lorebook/card field. Reference only.

When the user mentions something specific, the server auto-injects a
<recalled> block surfacing matching scratchpad docs — even ones zoned
to indexed/consumed. Use that for accurate canon cross-reference.

# Requesting more sources

When you need a document you don't have yet, emit a fetch_targets[]
commit at the END of your reply. Same-host only — the server resolves
paths against the original source's host:

    <commit>
    {"fetch_targets[]": [
      {"path": "Sapin_Kingdom"},
      {"path": "/wiki/Mana"},
      {"url": "https://same-host.example.com/specific-page"}
    ]}
    </commit>

Bound your requests: 3-7 targets per turn is healthy. The server fetches
them in parallel (max 5 concurrent) between this turn and the next, and
the new docs appear as ``active`` in the scratchpad on your next turn.
The user sees a "Fetching N references…" indicator during the gap.

The system prompt may include a ``<recently_fetched>`` hint listing paths
already in the scratchpad. Do NOT re-emit ``fetch_targets[]`` for those —
they're already available; just reference their content directly.

# Working effectively

1. On the first turn, READ the active source thoroughly. Identify points
   of interest. If the source has <links>, scan them for relevant topics.
2. Don't blind-fetch lists of links — pick deliberately. 3-7 high-signal
   targets per turn beats 20 noisy ones.
3. When you commit a lorebook entry that draws from a specific scratchpad
   doc, name the doc's path in the entry's `source_path` field — the
   server can mark that doc as consumed (keeps active set lean):
       {"keys": ["Sapin"], "content": "...", "source_path": "/wiki/Sapin_Kingdom"}
4. Don't quote scratchpad content verbatim. Interpret + adapt + ask the
   user how they want it twisted from canon.
"""


# Phase 2 backward-compat — kept as a no-op alias since the addendum was
# folded into _SCRATCHPAD_ADDENDUM. Old code paths that read this still work.
_WIKI_ADDENDUM = """\
# Wiki source attached

The user pasted a fan-wiki or encyclopedia URL. The structured block below
contains canonical reference material — title, infobox key/value pairs,
section excerpts, and category tags. Use it to inform every draft.

When opening (Q_HOOK), acknowledge what the wiki establishes — name 1-2
specific canonical traits ("Sharingan user, Uchiha clan survivor") so the
user knows you've read it. Then ask the FIRST real question: "Are we going
canon-faithful, twisting it, or going full AU?" If the user provided a
twist in their seed prompt (look for it in the first user turn), reflect
that twist in your acknowledgment.

For Q_PHYSICAL, draw from the ``Appearance`` section if present, plus
infobox fields like hair / eye / build / age. Surface canon vs. twist as
a real choice ("wiki has him at 5'10" lean — keep, adjust, or reimagine?").

For Q_PERSONALITY, draw from ``Personality`` section. Same canon-vs-twist
framing.

For Q_DEPTH, draw from ``Background``, ``Abilities``, ``History`` sections.

For Q_RELATIONSHIP_AND_OPENING, the wiki rarely tells you how the
character meets {{user}} — that's the user's invention. Suggest 1-2
plausible scenarios drawn from the wiki's setting, but treat the answer
as theirs to give.

Never blindly transcribe wiki prose into the card. Always interpret +
adapt + ask. The wiki is reference material, not the deliverable.

If the wiki classifier flagged a type mismatch (ensemble or world_rpg
detected but card_type=single), open by acknowledging it: "This is
technically [a team / a setting], but Single cards are what's available
today — pick one figure to focus on, or we treat the group as a single
composite voice. Your call."
"""


# ── Helpers ────────────────────────────────────────────────────────────────

def _user_id(request: Request) -> str:
    user = request.scope.get("user")
    return user.id if user else ""


def _backend(request: Request):
    from augmentum.state.backends.sqlite import SQLiteBackend
    sm = getattr(request.app.state, "state_manager", None)
    if not sm or not isinstance(sm.backend, SQLiteBackend):
        return None
    return sm.backend


# ── Cardsmith session persistence (write-through to cardsmith_sessions) ───
#
# The in-memory OrderedDict in modes/narrative/cardsmith/state.py is the
# read path for hot sessions; this layer adds a durable disk row so a
# container restart, OOM, or fresh worker doesn't drop a multi-turn
# conversation in progress.
#
# Failure mode: all helpers swallow exceptions and log a warning. Cardsmith
# is a user-product feature, not a critical write path — losing one row to
# a transient SQLite error shouldn't crash the SSE turn. Worst case the
# session falls back to memory-only semantics (the prior behavior).

async def _persist_session(sess, be) -> None:
    """Write-through UPSERT for a cardsmith session.

    Called from the route layer at every durable checkpoint:
      - after the user message lands (append_user)
      - after the assistant turn completes (append_assistant + field commits)
      - after the agentic fetch loop updates the scratchpad
      - after `finalized = True`
      - after start-session meta is populated (wiki context, seed)

    The in-memory representation (sess.messages / .fields / .meta) is the
    source of truth during a turn; this serializes a snapshot of that
    state and stamps it onto disk so a restart reads it back unchanged.
    """
    if not be or not sess:
        return
    try:
        row = serialize_for_disk(sess)
        await be.conn.execute(
            """INSERT INTO cardsmith_sessions
                 (session_id, user_id, card_type, source, created_at,
                  last_active_at, messages, fields, meta, finalized)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(session_id) DO UPDATE SET
                   user_id        = excluded.user_id,
                   card_type      = excluded.card_type,
                   source         = excluded.source,
                   last_active_at = excluded.last_active_at,
                   messages       = excluded.messages,
                   fields         = excluded.fields,
                   meta           = excluded.meta,
                   finalized      = excluded.finalized""",
            (
                row["session_id"], row["user_id"], row["card_type"], row["source"],
                row["created_at"], row["last_active_at"],
                row["messages"], row["fields"], row["meta"], row["finalized"],
            ),
        )
        await be.conn.commit()
    except Exception as exc:
        log.warning("cardsmith_persist_failed",
                    session_id=getattr(sess, "session_id", "?"), error=str(exc))


async def _resolve_session(session_id: str, *, user_id: str, be) -> object | None:
    """Return a CardsmithSession by id — in-memory first, disk fallback.

    `get_session` enforces user ownership + TTL eviction on the in-memory
    path. When the in-memory dict misses (server restart, LRU eviction),
    we look up the row in cardsmith_sessions, parse it, register it back
    into the OrderedDict, then re-call get_session so the same ownership
    and TTL checks apply to the rehydrated session.

    Returns None if the session doesn't exist on disk, the row belongs to
    a different user, or the row exists but has aged past TTL (in which
    case the rehydration will fail the TTL check inside get_session and
    drop_session will be invoked).
    """
    sess = get_session(session_id, user_id=user_id)
    if sess is not None:
        return sess
    if not be or not session_id:
        return None
    try:
        cursor = await be.conn.execute(
            """SELECT session_id, user_id, card_type, source,
                      created_at, last_active_at,
                      messages, fields, meta, finalized
                 FROM cardsmith_sessions
                WHERE session_id = ?""",
            (session_id,),
        )
        row = await cursor.fetchone()
        await cursor.close()
    except Exception as exc:
        log.warning("cardsmith_rehydrate_failed", session_id=session_id, error=str(exc))
        return None
    if row is None:
        return None
    # Map the tuple row into the dict shape session_from_row expects.
    row_dict = {
        "session_id": row[0], "user_id": row[1], "card_type": row[2],
        "source": row[3], "created_at": row[4], "last_active_at": row[5],
        "messages": row[6], "fields": row[7], "meta": row[8],
        "finalized": row[9],
    }
    # Cross-tenant guard: a user guessing another user's session_id should
    # not get a rehydrated session back. Mirrors the in-memory check in
    # get_session so the disk path doesn't open a side channel.
    if row_dict["user_id"] and user_id and row_dict["user_id"] != user_id:
        return None
    try:
        sess = session_from_row(row_dict)
    except Exception as exc:
        log.warning("cardsmith_parse_failed", session_id=session_id, error=str(exc))
        return None
    register_session(sess)
    log.info("cardsmith_session_rehydrated",
             session_id=session_id, message_count=len(sess.messages),
             finalized=sess.finalized)
    # Re-run get_session to apply the TTL check (which will evict + return
    # None if the row aged past the in-memory TTL after restart).
    return get_session(session_id, user_id=user_id)


async def _drop_persisted_session(session_id: str, be) -> None:
    """Delete the disk row for a session that's being explicitly dropped.

    Called after `drop_session` (memory eviction) so a Cancel button or
    finalize-then-cleanup doesn't leave orphan rows accumulating.
    """
    if not be or not session_id:
        return
    try:
        await be.conn.execute(
            "DELETE FROM cardsmith_sessions WHERE session_id = ?",
            (session_id,),
        )
        await be.conn.commit()
    except Exception as exc:
        log.warning("cardsmith_drop_persisted_failed",
                    session_id=session_id, error=str(exc))


def _sse_event(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


# ── Agentic fetch helpers ─────────────────────────────────────────────────

# Hard cap per turn — bounds latency between commit and the next turn.
_MAX_FETCHES_PER_TURN = 5


def _coerce_fetch_targets(raw: list) -> list[dict]:
    """Normalize fetch_targets[] commit values into [{path, url, title}] dicts.

    Accepts the model's preferred shape ``{"path": "X"}`` or ``{"url": "Y"}``
    plus a few forgiving variants (``{"target": ...}``, plain strings).
    Drops malformed entries silently.
    """
    out: list[dict] = []
    for raw_entry in raw[:_MAX_FETCHES_PER_TURN]:
        if isinstance(raw_entry, str):
            entry = {"url": raw_entry} if raw_entry.startswith(("http://", "https://")) else {"path": raw_entry}
        elif isinstance(raw_entry, dict):
            entry = dict(raw_entry)
            # Forgiving aliases.
            if "target" in entry and "path" not in entry and "url" not in entry:
                t = str(entry["target"])
                entry["url" if t.startswith(("http://", "https://")) else "path"] = t
        else:
            continue
        if not (entry.get("path") or entry.get("url")):
            continue
        out.append(entry)
    return out


async def _process_fetch_targets(
    sess, targets: list[dict],
) -> int:
    """Fetch each target and merge into the session scratchpad.

    Targets with `path` are resolved against `sess.meta["wiki_host"]`. URL
    targets are fetched directly but only if they're on the same host as
    the original source (cross-host blocked for v1).

    Returns the count of NEW entries added (deduped on existing paths).
    """
    base_host = sess.meta.get("wiki_host", "") or ""
    scratchpad = deserialize_scratchpad(sess.meta.get("scratchpad") or [])

    async def _fetch_one(target: dict) -> ScratchEntry | None:
        path = (target.get("path") or "").strip()
        url = (target.get("url") or "").strip()
        try:
            if path:
                if not base_host:
                    return None
                doc = await fetch_path(path, base_host=base_host)
                key = path if path.startswith("/") else f"/wiki/{path.replace(' ', '_')}"
            else:
                # URL targets — enforce same-host.
                from urllib.parse import urlparse
                parsed = urlparse(url)
                target_host = (parsed.hostname or "").lower()
                if base_host and target_host and target_host != base_host:
                    log.info(
                        "cardsmith_fetch_target_cross_host_blocked",
                        session_id=sess.session_id,
                        url=url,
                        base_host=base_host,
                    )
                    return None
                doc = await fetch_content_doc(url)
                key = url
            return ScratchEntry.from_content_doc(doc, path=key)
        except ContentExtractError as exc:
            log.info(
                "cardsmith_fetch_target_failed",
                session_id=sess.session_id,
                target=target,
                error=str(exc),
            )
            return None
        except Exception as exc:
            log.warning(
                "cardsmith_fetch_target_unexpected",
                session_id=sess.session_id,
                target=target,
                error=str(exc),
                exc_info=True,
            )
            return None

    results = await asyncio.gather(*(_fetch_one(t) for t in targets))
    added = 0
    for entry in results:
        if entry is not None and add_to_scratchpad(scratchpad, entry):
            added += 1

    sess.meta["scratchpad"] = serialize_scratchpad(scratchpad)
    log.info(
        "cardsmith_fetch_targets_processed",
        session_id=sess.session_id,
        requested=len(targets),
        added=added,
        scratchpad_size=len(scratchpad),
    )
    return added


# ── /start ─────────────────────────────────────────────────────────────────

class StartBody(BaseModel):
    card_type: str = "single"
    source: str = "describe"
    seed_prompt: str = ""
    wiki_url: str = ""  # required when source == "wiki"
    # Continuation token — when set, the new session inherits the parent's
    # scratchpad and wiki context. Used after a finalize to build another
    # character in the same universe without re-pasting the wiki URL.
    parent_session_id: str = ""


@router.post("/start")
async def cardsmith_start(body: StartBody, request: Request) -> JSONResponse:
    """Create a new Cardsmith session and return its id.

    When ``source == "wiki"``, fetches the wiki context (using the in-process
    LRU cache populated by /wiki-preview if the user just viewed it) and
    stashes it on ``session.meta.wiki_context``. The /turn route reads this
    and prepends a structured context block to the system prompt so the
    Cardsmith opens with knowledge of the canonical source.

    When ``parent_session_id`` is set, the new session inherits the parent's
    scratchpad + wiki host/title/context. Validates ownership; the parent
    must belong to the same user. Used to chain multiple cards in the same
    universe (e.g. an ensemble's named characters after the ensemble card,
    or several NPCs in a wiki-grounded setting).
    """
    if body.card_type not in _SUPPORTED_TYPES:
        return JSONResponse(
            {"error": f"card_type '{body.card_type}' not yet available"},
            status_code=400,
        )
    if body.source not in _SUPPORTED_SOURCES:
        return JSONResponse(
            {"error": f"source '{body.source}' not yet available"},
            status_code=400,
        )

    uid = _user_id(request)
    be = _backend(request)

    # ── Continuation path ─────────────────────────────────────────────────
    parent_meta: dict | None = None
    parent_universe_saves: list[dict] = []
    if body.parent_session_id.strip():
        parent_id = body.parent_session_id.strip()
        parent_sess = await _resolve_session(parent_id, user_id=uid, be=be)
        if parent_sess is None:
            return JSONResponse(
                {"error": "Parent session not found"},
                status_code=404,
            )
        # We inherit metadata regardless of finalized state — the parent's
        # wiki source and scratchpad are useful continuation context even
        # when the user chained mid-session. Finalize-only would be more
        # restrictive than necessary.
        parent_meta = dict(parent_sess.meta or {})
        # Track previously-saved char_ids so the system prompt can mention
        # them in continuation context (built up below).
        prev_saves = parent_meta.get("universe_saves")
        if isinstance(prev_saves, list):
            parent_universe_saves = [
                dict(s) for s in prev_saves if isinstance(s, dict)
            ]

    # Wiki-source path is mutually exclusive with parent — picking a parent
    # means reusing its wiki context, not pasting a new URL.
    wiki_context: ContentDoc | None = None
    if body.source == "wiki" and parent_meta is None:
        if not body.wiki_url.strip():
            return JSONResponse(
                {"error": "wiki_url required when source='wiki'"},
                status_code=400,
            )
        try:
            wiki_context = await fetch_content_doc(body.wiki_url.strip())
        except ContentExtractError as exc:
            return JSONResponse(
                {"error": f"Source fetch failed: {exc}"},
                status_code=400,
            )
        except Exception as exc:
            log.warning(
                "wiki_fetch_unexpected_error",
                url=body.wiki_url,
                error=str(exc),
            )
            return JSONResponse(
                {"error": "Source fetch failed unexpectedly"},
                status_code=500,
            )

    sess = get_or_create_session(
        user_id=uid,
        card_type=body.card_type,
        source=body.source if parent_meta is None else (
            parent_meta.get("source") or body.source
        ),
        seed_prompt=body.seed_prompt.strip(),
    )

    if wiki_context is not None:
        # Stash diagnostic / UI metadata.
        sess.meta["wiki_context_data"] = wiki_context.to_dict()
        sess.meta["wiki_url"] = wiki_context.url
        sess.meta["wiki_title"] = wiki_context.title
        # Derive the host so the agentic fetch loop can resolve relative paths.
        from urllib.parse import urlparse
        parsed = urlparse(wiki_context.url)
        sess.meta["wiki_host"] = (parsed.hostname or "").lower()
        # Seed the scratchpad with the user-pasted source as the first
        # active doc. Subsequent fetch_targets[] commits add to this list.
        seed_entry = ScratchEntry.from_content_doc(
            wiki_context, path=wiki_context.url,
        )
        sess.meta["scratchpad"] = serialize_scratchpad([seed_entry])
        # Type-mismatch note for UX.
        if wiki_context.detected_type != body.card_type:
            sess.meta["wiki_type_mismatch"] = wiki_context.detected_type

    # Inherit wiki/scratchpad/universe state from the parent session. Done
    # AFTER the wiki-context branch so parent inheritance can't be
    # accidentally clobbered when the model later re-fetches the original
    # source via fetch_targets[].
    if parent_meta is not None:
        for key in ("wiki_url", "wiki_title", "wiki_host", "wiki_context_data"):
            if key in parent_meta and parent_meta[key]:
                sess.meta[key] = parent_meta[key]
        parent_scratchpad = parent_meta.get("scratchpad")
        if isinstance(parent_scratchpad, list) and parent_scratchpad:
            # Deep-copy via the round-trip helpers so the child can mutate
            # zones independently of the parent's snapshot.
            inherited = deserialize_scratchpad(parent_scratchpad)
            sess.meta["scratchpad"] = serialize_scratchpad(inherited)
        sess.meta["chained_from"] = body.parent_session_id.strip()
        sess.meta["universe_saves"] = list(parent_universe_saves)

    log.info(
        "cardsmith_session_started",
        session_id=sess.session_id,
        card_type=body.card_type,
        source=body.source,
        user_id=uid,
        has_seed=bool(body.seed_prompt.strip()),
        has_wiki=wiki_context is not None,
        chained_from=body.parent_session_id.strip() or "",
        wiki_host=wiki_context.host_kind if wiki_context else (
            (parent_meta or {}).get("wiki_host", "") if parent_meta else ""
        ),
    )
    await _persist_session(sess, be)
    return JSONResponse({"session_id": sess.session_id})


# ── /wiki-preview ──────────────────────────────────────────────────────────

class WikiPreviewBody(BaseModel):
    url: str


@router.post("/wiki-preview")
async def cardsmith_wiki_preview(
    body: WikiPreviewBody, request: Request,
) -> JSONResponse:
    """Fetch + classify any source URL, return a slim summary for the
    launcher confirmation step. Caches the full ContentDoc server-side so
    /start doesn't re-fetch when the user clicks Begin.
    """
    if not body.url.strip():
        return JSONResponse({"error": "URL is required"}, status_code=400)

    try:
        ctx = await fetch_content_doc(body.url.strip())
    except ContentExtractError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    except Exception as exc:
        log.warning("wiki_preview_unexpected_error", url=body.url, error=str(exc))
        return JSONResponse(
            {"error": "Source fetch failed unexpectedly"},
            status_code=500,
        )

    # Build a slim summary for the UI — full WikiContext stays server-side.
    section_headings = list(ctx.sections.keys())
    warning = ""
    if ctx.detected_type != "single":
        type_label = {
            "ensemble": "a group / team",
            "world_rpg": "a setting / world",
        }.get(ctx.detected_type, ctx.detected_type)
        warning = (
            f"This looks like {type_label}, but {type_label.split('/')[0].strip()} "
            "card types aren't built yet. We'll create a Single card based on it for now."
        )

    return JSONResponse({
        "url": ctx.url,
        "host_kind": ctx.host_kind,
        "title": ctx.title,
        "summary": ctx.summary[:400],
        "thumbnail_url": ctx.thumbnail_url,
        "detected_type": ctx.detected_type,
        "confidence": ctx.confidence,
        "section_count": len(ctx.sections),
        "section_headings": section_headings[:8],
        "infobox_field_count": len(ctx.infobox),
        "categories": ctx.categories[:6],
        "warning": warning,
    })


# ── /turn (SSE) ────────────────────────────────────────────────────────────

class TurnBody(BaseModel):
    session_id: str
    user_message: str = ""
    model: str = ""


@router.post("/turn")
async def cardsmith_turn(body: TurnBody, request: Request) -> StreamingResponse:
    """Run one Cardsmith conversational turn — SSE stream of deltas + fields.

    Serializes via ``sess.lock`` so two concurrent /turn calls on the same
    session can't interleave message appends or field commits. A second
    concurrent caller is rejected with 409 rather than queued — the user
    almost certainly didn't mean to fire two simultaneously, and silent
    serialization would feel like the second click hung.

    Lock ownership: acquired here in the route, released in the streaming
    generator's ``finally``. Any synchronous setup after the acquire runs
    under the protection of an outer try/except that releases on exception.
    The earlier design acquired inside the generator — that race-leaked the
    409 path because two concurrent callers could both pass ``locked()``
    before either started the generator.
    """
    uid = _user_id(request)
    be = _backend(request)
    sess = await _resolve_session(body.session_id, user_id=uid, be=be)
    if sess is None:
        return JSONResponse(
            {"error": "Cardsmith session not found or expired"},
            status_code=404,
        )

    # Synchronous ``locked()`` check + immediate ``acquire()`` is race-free
    # under cooperative asyncio: nothing yields between the two statements,
    # and acquiring a free Lock returns without yielding.
    if sess.lock.locked():
        return JSONResponse(
            {"error": "Another turn is already in progress for this session"},
            status_code=409,
        )
    await sess.lock.acquire()

    def _release_lock_safe() -> None:
        if sess.lock.locked():
            try:
                sess.lock.release()
            except RuntimeError:
                # Released from a different task — extremely unlikely under
                # the single-stream-per-session invariant.
                log.warning(
                    "cardsmith_lock_release_unexpected",
                    session_id=sess.session_id,
                )

    try:
        # First-turn kickstart: if the user hasn't sent anything and there's
        # no seed prompt, give the model a polite "begin" nudge.
        user_msg = body.user_message.strip()
        if not user_msg and not sess.messages:
            seed = sess.meta.get("seed_prompt", "")
            user_msg = seed or "Let's design a new character together."
        if user_msg:
            sess.append_user(user_msg)
            # Durable checkpoint: the user's message is now on disk so a
            # crash mid-stream doesn't lose the prompt they just typed.
            await _persist_session(sess, be)

        provider_reg = getattr(request.app.state, "provider_registry", None)
        if not provider_reg or not getattr(provider_reg, "backends", None):
            _release_lock_safe()
            return JSONResponse(
                {"error": "No LLM backend available"}, status_code=503,
            )

        try:
            backend, resolved_model = await provider_reg.resolve_model_for_role(
                "utility",
                override=body.model or "",
                settings=settings,
            )
        except Exception as exc:
            log.warning("cardsmith_model_resolve_failed", error=str(exc))
            _release_lock_safe()
            return JSONResponse(
                {"error": "Failed to resolve LLM"}, status_code=503,
            )

        system_prompt = get_prompt(sess.card_type)

        # Universe-chain addendum — when this session inherited context from
        # a parent session (chained_from set), tell the model: prior cards
        # exist in this same setting; build on that continuity rather than
        # re-asking baseline canon questions.
        if sess.meta.get("chained_from"):
            saves = sess.meta.get("universe_saves") or []
            prior_names: list[str] = []
            if isinstance(saves, list):
                for s in saves:
                    if isinstance(s, dict):
                        nm = s.get("name")
                        if isinstance(nm, str) and nm.strip():
                            prior_names.append(nm.strip())
            if prior_names:
                names_csv = ", ".join(prior_names[-8:])
                system_prompt += (
                    "\n\n# Continuation context\n\n"
                    "The user has been building characters in this universe "
                    f"with you. Already saved this session: {names_csv}. "
                    "Do NOT re-establish baseline canon — the wiki scratchpad "
                    "below is the same one those characters drew from. Open "
                    "by acknowledging which character we're building next "
                    "and how (if at all) they relate to the prior ones. "
                    "Offer the user a quick choice: a known canonical figure, "
                    "a related OC, or someone unrelated in the same setting."
                )

        # Scratchpad rendering: any source the user pasted (Phase 2 wiki lane)
        # plus all docs the agentic fetch loop has pulled in subsequent turns.
        # The recall layer auto-injects scratchpad entries the user mentioned
        # in their message even when they're zoned to indexed/consumed.
        scratchpad = deserialize_scratchpad(sess.meta.get("scratchpad") or [])
        if scratchpad:
            # Build recall against last user message + last 2 assistant turns.
            last_user = user_msg or ""
            tail_pieces: list[str] = []
            for m in sess.messages[-4:]:
                if m.get("role") == "assistant":
                    tail_pieces.append(m.get("content", ""))
            tail = " ".join(tail_pieces)

            index = build_reference_index(scratchpad)
            recalled = recall_for_turn(last_user, tail, index, scratchpad)
            scratchpad_block = render_scratchpad_block(scratchpad, recalled=recalled)
            system_prompt = (
                system_prompt + "\n\n" + _SCRATCHPAD_ADDENDUM + "\n\n" + scratchpad_block
            )

        messages = [Message(role="system", content=system_prompt)]
        for m in sess.messages:
            messages.append(Message(role=m["role"], content=m["content"]))

        chat_request = InternalChatRequest(
            model=resolved_model,
            messages=messages,
            stream=True,
            temperature=0.85,
            max_tokens=2400,
        )
    except BaseException:
        # Synchronous setup failed after we acquired the lock. Release so the
        # session isn't stranded for the rest of the TTL window.
        _release_lock_safe()
        raise

    async def _stream():
        # Lock is already held by the route handler. We just need to release
        # it when the stream finishes (or the client disconnects, or anything
        # below raises).
        parser = StreamingFieldParser()
        visible_accum: list[str] = []
        finalized_emitted = False
        committed_paths: list[str] = []
        client_disconnected = False

        async def _client_alive() -> bool:
            """Best-effort disconnect probe via Starlette's request.is_disconnected.

            Returns True if the client is still connected. Probe is non-blocking
            (relies on the receive queue's has-data signal). On any error we
            assume connected to avoid false-positive aborts.
            """
            try:
                return not await request.is_disconnected()
            except Exception:
                return True

        try:
            async for chunk in backend.chat_stream(chat_request):
                # Periodically check whether the client has disconnected. If so,
                # bail out of the stream early so we stop spending tokens on a
                # reply the user will never see. The model backend's chat_stream
                # is left to wind down naturally on the next chunk yield.
                if not await _client_alive():
                    client_disconnected = True
                    log.info(
                        "cardsmith_client_disconnected_mid_stream",
                        session_id=sess.session_id,
                    )
                    break
                # Forward reasoning chunks so the UI can show that the model
                # is working. Reasoning-capable model families (GLM-4.x,
                # DeepSeek V3.2/V4, EXAONE 4.x, Qwen 3.x in thinking mode,
                # etc.) often spend many seconds on reasoning before any
                # visible content streams — without this the chat looks
                # hung after the second turn once context gets non-trivial.
                if chunk.thinking_delta:
                    yield _sse_event({
                        "type": "thinking",
                        "text": chunk.thinking_delta,
                    })
                if chunk.content_delta:
                    step = parser.feed(chunk.content_delta)
                    if step.visible:
                        visible_accum.append(step.visible)
                        yield _sse_event({"type": "delta", "text": step.visible})
                    for emission in step.emissions:
                        sess.commit_field(emission.path, emission.value)
                        committed_paths.append(emission.path)
                        log.debug(
                            "cardsmith_field_commit",
                            session_id=sess.session_id,
                            path=emission.path,
                            value_preview=str(emission.value)[:80],
                        )
                        yield _sse_event({
                            "type": "field",
                            "path": emission.path,
                            "value": emission.value,
                        })
                    if step.done and not finalized_emitted:
                        finalized_emitted = True
                        # Don't break the loop — let the model finish whatever
                        # trailing whitespace it sends. We finalize after the
                        # stream completes.
                if chunk.done:
                    break

            # Drain any held-back text in the parser
            tail = parser.flush()
            if tail.visible:
                visible_accum.append(tail.visible)
                yield _sse_event({"type": "delta", "text": tail.visible})
            for emission in tail.emissions:
                sess.commit_field(emission.path, emission.value)
                committed_paths.append(emission.path)
                yield _sse_event({
                    "type": "field",
                    "path": emission.path,
                    "value": emission.value,
                })
            if tail.done:
                finalized_emitted = True

            # Persist the visible reply on the conversation log so the next
            # /turn call sees the assistant context. Skip if the client
            # disconnected mid-stream — the partial reply isn't useful for
            # the next turn and may confuse the model.
            if not client_disconnected:
                sess.append_assistant("".join(visible_accum).strip())
            # Durable checkpoint — runs even on disconnect so field commits
            # that landed mid-stream (name, paragraph slots, lorebook entries)
            # are preserved. The visible partial reply is intentionally NOT
            # appended on disconnect: feeding a truncated assistant turn back
            # into the next /turn confuses the model. The fields it committed
            # are still authoritative, so the user resumes with everything
            # the model managed to extract — they just have to re-prompt for
            # whatever the model was mid-sentence on.
            await _persist_session(sess, be)

            log.info(
                "cardsmith_turn_complete",
                session_id=sess.session_id,
                committed_this_turn=committed_paths,
                total_field_count=len(sess.fields),
                accumulated_paths=sorted(sess.fields.keys()),
                client_disconnected=client_disconnected,
            )

            # Agentic fetch loop — between turns, fulfill any fetch_targets[]
            # the model committed this turn. The new content lands in the
            # scratchpad and shows up as ``active`` in the next turn's
            # system prompt. Skipped if the client disconnected (no point
            # spending tokens/bandwidth on a reply the user won't see).
            if not client_disconnected and not finalized_emitted:
                fetch_targets = _coerce_fetch_targets(sess.fields.get("fetch_targets") or [])
                if fetch_targets:
                    yield _sse_event({
                        "type": "fetching",
                        "count": len(fetch_targets),
                        "targets": [t.get("title") or t.get("path") or t.get("url") for t in fetch_targets],
                    })
                    fetched_count = 0
                    try:
                        fetched_count = await _process_fetch_targets(sess, fetch_targets)
                    except Exception as exc:
                        log.warning(
                            "cardsmith_fetch_targets_failed",
                            session_id=sess.session_id,
                            error=str(exc),
                            exc_info=True,
                        )
                    yield _sse_event({
                        "type": "fetched",
                        "count": fetched_count,
                    })
                    # Clear the request so re-emitting fetch_targets[] on a
                    # subsequent turn doesn't re-run the previous batch.
                    sess.fields["fetch_targets"] = []
                    # Durable checkpoint: scratchpad now holds the fetched
                    # docs. Re-running the fetch loop after a crash would
                    # cost tokens + upstream requests we already paid for.
                    await _persist_session(sess, be)

            # If the model emitted [CARDSMITH_DONE], finalize in-line and emit
            # the saved char_id so the frontend can navigate to the editor.
            if finalized_emitted:
                try:
                    saved = await _finalize_and_save(sess, request)
                    yield _sse_event({
                        "type": "finalized",
                        "char_id": saved["char_id"],
                        "name": saved["name"],
                        # Frontend uses this to decide whether to offer
                        # "Add another in this universe" continuation.
                        "has_universe": saved.get("has_universe", False),
                        "session_id": sess.session_id,
                    })
                except Exception as exc:
                    log.warning(
                        "cardsmith_inline_finalize_failed",
                        session_id=sess.session_id,
                        error=str(exc),
                    )
                    yield _sse_event({
                        "type": "error",
                        "error": f"Save failed: {exc}",
                    })

            yield "data: [DONE]\n\n"

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("cardsmith_stream_error", error=str(exc), exc_info=True)
            # The local llama_cpp backend raises "No model selected. Available
            # models: <local GGUFs>" when it gets a request before its
            # llama-server has loaded anything. When cardsmith resolves to
            # that backend (because primary_chat_model points at a local
            # model that isn't currently loaded, or utility_model isn't set
            # and the chain falls through to the local default), the raw
            # GGUF list leaking through is confusing — the user picked a
            # cloud model in chat and has no idea why local files are being
            # listed. Wrap with context about *which* backend was resolved
            # and how to point cardsmith somewhere else.
            err_str = str(exc)
            if "No model selected" in err_str and "Available models" in err_str:
                backend_kind = type(backend).__name__
                yield _sse_event({
                    "type": "error",
                    "error": (
                        f"Cardsmith resolved to the local engine ({backend_kind}) "
                        f"using model '{resolved_model}', but that model isn't "
                        f"loaded right now. Either load it from the model "
                        f"manager, set Settings > utility_model to a cloud "
                        f"model you have configured (e.g. DeepSeek), or "
                        f"re-select your preferred chat model from the chat "
                        f"composer so primary_chat_model is in sync."
                    ),
                })
            else:
                yield _sse_event({"type": "error", "error": err_str})
            yield "data: [DONE]\n\n"
        finally:
            # Always release the per-session lock so the next turn can proceed,
            # even if the stream errored or the client disconnected.
            _release_lock_safe()

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ── /session/{id} (resume preview) ───────────────────────────────────────


@router.get("/session/{session_id}")
async def cardsmith_get_session(session_id: str, request: Request) -> JSONResponse:
    """Return a hydrated snapshot of an in-flight session.

    Used by the frontend's resume flow: when the browser reloads and the
    in-memory `state.sessionId` is gone, the launcher checks localStorage
    for a saved sessionId and calls this endpoint to confirm the session
    is still alive server-side and to fetch the conversation history +
    accumulated fields needed to re-render the chat thread.

    Returns 404 with the same shape as `/turn` and `/finalize` for any of:
      - unknown session_id
      - session belongs to a different user
      - session aged past TTL (the in-memory check inside get_session)

    `meta.scratchpad` can be large (full text of fetched wiki docs); we
    ship only its length + entry titles to keep the resume payload small.
    The full scratchpad stays server-side and re-engages on the next /turn.
    """
    uid = _user_id(request)
    be = _backend(request)
    sess = await _resolve_session(session_id, user_id=uid, be=be)
    if sess is None:
        return JSONResponse(
            {"error": "Cardsmith session not found or expired"},
            status_code=404,
        )

    # Trim scratchpad to a summary — full content stays server-side.
    meta_summary = dict(sess.meta)
    scratchpad = meta_summary.pop("scratchpad", None)
    if isinstance(scratchpad, list):
        meta_summary["scratchpad_summary"] = {
            "entry_count": len(scratchpad),
            "titles": [
                (e.get("title") or e.get("path") or "")[:120]
                for e in scratchpad[:20]
                if isinstance(e, dict)
            ],
        }

    return JSONResponse({
        "session_id": sess.session_id,
        "card_type": sess.card_type,
        "source": sess.source,
        "created_at": sess.created_at,
        "last_active_at": sess.last_active_at,
        "messages": sess.messages,
        "fields": sess.fields,
        "meta": meta_summary,
        "finalized": sess.finalized,
    })


# ── /sessions (drafts list) ───────────────────────────────────────────────


@router.get("/sessions")
async def cardsmith_list_sessions(request: Request) -> JSONResponse:
    """List all in-progress (non-finalized) cardsmith sessions for the user.

    Powers the launcher modal's "in-progress drafts" picker so users see
    every unfinished card, not just the most recent one (which is all the
    localStorage resume token tracks).

    Returns slim per-session rows:
      - session_id, card_type, source
      - last_active_at (unix seconds, REAL — match the disk schema)
      - message_count
      - friendly_label: name field if committed, else "Based on <wiki_title>",
        else "<N>-turn draft" — gives the picker a meaningful label even
        before the model has emitted a name commit
      - has_universe: True when meta.scratchpad is non-empty — UI hides
        the chain-to-universe button for sessions without one

    Source of truth is the disk table; the in-memory OrderedDict is a
    cache. Every durable checkpoint writes through, so the disk row is
    at most a few seconds stale on an active session.
    """
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    be = _backend(request)
    if not be:
        return JSONResponse({"sessions": []})

    try:
        cursor = await be.conn.execute(
            """SELECT session_id, card_type, source, last_active_at,
                      messages, fields, meta
                 FROM cardsmith_sessions
                WHERE user_id = ? AND finalized = 0
                ORDER BY last_active_at DESC
                LIMIT 20""",
            (uid,),
        )
        rows = await cursor.fetchall()
        await cursor.close()
    except Exception as exc:
        log.warning("cardsmith_list_sessions_failed", user_id=uid, error=str(exc))
        return JSONResponse({"sessions": []})

    sessions: list[dict] = []
    for row in rows:
        try:
            messages_json = row[4] or "[]"
            fields_json = row[5] or "{}"
            meta_json = row[6] or "{}"
            messages = json.loads(messages_json)
            fields = json.loads(fields_json)
            meta = json.loads(meta_json)
        except (ValueError, TypeError):
            # Corrupt row — skip rather than break the whole list.
            continue
        msg_count = len(messages) if isinstance(messages, list) else 0
        name = ""
        if isinstance(fields, dict):
            raw_name = fields.get("name")
            if isinstance(raw_name, str):
                name = raw_name.strip()
        wiki_title = ""
        scratchpad_size = 0
        if isinstance(meta, dict):
            raw_title = meta.get("wiki_title")
            if isinstance(raw_title, str):
                wiki_title = raw_title.strip()
            sp = meta.get("scratchpad")
            if isinstance(sp, list):
                scratchpad_size = len(sp)
        if name:
            label = name
        elif wiki_title:
            label = f"Based on {wiki_title}"
        else:
            label = f"{msg_count}-turn draft" if msg_count else "Empty draft"
        sessions.append({
            "session_id": row[0],
            "card_type": row[1],
            "source": row[2],
            "last_active_at": row[3],
            "message_count": msg_count,
            "friendly_label": label,
            "has_universe": scratchpad_size > 0,
        })
    return JSONResponse({"sessions": sessions})


# ── /finalize ──────────────────────────────────────────────────────────────

class FinalizeBody(BaseModel):
    session_id: str


@router.post("/finalize")
async def cardsmith_finalize(
    body: FinalizeBody, request: Request,
) -> JSONResponse:
    """Save the accumulated card explicitly (e.g. user clicked Drop to Editor).

    Idempotent: calling twice returns the same char_id from the second call,
    but the first save is the source of truth — the session is dropped after.
    """
    uid = _user_id(request)
    be = _backend(request)
    sess = await _resolve_session(body.session_id, user_id=uid, be=be)
    if sess is None:
        return JSONResponse(
            {"error": "Cardsmith session not found or expired"},
            status_code=404,
        )
    try:
        saved = await _finalize_and_save(sess, request)
        # Include the just-finalized session_id so the frontend can pass
        # it as parent_session_id on a chained /start call.
        return JSONResponse({
            "ok": True,
            "session_id": sess.session_id,
            **saved,
        })
    except Exception as exc:
        log.warning(
            "cardsmith_finalize_failed",
            session_id=body.session_id,
            error=str(exc),
            exc_info=True,
        )
        return JSONResponse(
            {"error": f"Save failed: {exc}"}, status_code=500,
        )


# ── /cancel ────────────────────────────────────────────────────────────────

class CancelBody(BaseModel):
    session_id: str


@router.post("/cancel")
async def cardsmith_cancel(body: CancelBody, request: Request) -> JSONResponse:
    """Drop a Cardsmith session without saving anything.

    Verifies ownership before dropping — defense in depth so a session
    can't be cancelled by a different tenant who guessed its id. Idempotent:
    cancelling an unknown / already-dropped id returns ok without surfacing
    whether the id ever existed.
    """
    uid = _user_id(request)
    be = _backend(request)
    # Resolve via disk fallback — a cancel after a server restart should
    # still drop the persisted row even if memory has nothing.
    sess = await _resolve_session(body.session_id, user_id=uid, be=be)
    if sess is not None:
        drop_session(body.session_id)
        await _drop_persisted_session(body.session_id, be)
    return JSONResponse({"ok": True})


# ── Save plumbing ──────────────────────────────────────────────────────────

# Fields whose presence indicates the inline-tag protocol succeeded. If none
# of these landed during the conversation, the model likely didn't follow the
# tag protocol — recovery extraction is invoked over the conversation log.
_HEALTH_KEYS = (
    "description", "desc_physical", "desc_personality", "desc_depth",
    "personality", "scenario", "greeting",
)


def _state_is_sparse(sess) -> bool:
    """Return True when accumulated session state lacks meaningful card content."""
    for k in _HEALTH_KEYS:
        v = sess.fields.get(k)
        if isinstance(v, str) and v.strip():
            return False
        if isinstance(v, list) and v:
            return False
    return True


def _examples_missing_with_content(sess) -> bool:
    """Return True when the card otherwise has content but `examples` is empty.

    Audit found ``examples`` lands far less often than other fields — the
    Q_EXAMPLES question runs late in the script and weaker models skip it.
    When the rest of the card is populated, run a targeted recovery pass
    rather than ship a card with no example dialogue (which makes
    downstream chat quality noticeably worse).
    """
    examples = sess.fields.get("examples")
    if isinstance(examples, str) and examples.strip():
        return False
    # Only trigger if the card actually has content elsewhere — sparse
    # state will be handled by the broader recovery path.
    has_content = any(
        isinstance(sess.fields.get(k), str) and sess.fields[k].strip()
        for k in ("description", "desc_physical", "personality", "greeting")
    )
    return has_content


_RECOVERY_PROMPT = """\
You are extracting a character card from a Cardsmith design conversation.

Read the FULL conversation below carefully. The user and the Cardsmith have
been co-designing a character. Now extract the final card from what was
established. Be faithful to the conversation — do not invent details that
weren't discussed.

Output ONLY a single JSON object with these keys (omit any you genuinely
cannot fill from the conversation):

{
  "name": "...",
  "description": "...full multi-paragraph description, 6-paragraph structure...",
  "personality": "...1-2 sentence distilled personality summary...",
  "scenario": "...the encounter scene...",
  "greeting": "...the character's first message to {{user}}...",
  "examples": "...((user))/((char)) dialogue examples...",
  "visualTraits": "...comma-separated SD-friendly tokens (8-20 of them)...",
  "imageStyle": "anime|painterly|photorealistic|watercolor|pixel|comic|dark|fantasy|scifi|ukiyoe|noir|cozy",
  "tags": ["tag1", "tag2"],
  "alternateGreetings": ["..."],
  "lorebook": [{"keys": ["..."], "content": "...", "priority": 100}]
}

Output ONLY the JSON. No code fences. No preamble. No commentary.
"""


async def _recover_fields_from_conversation(sess, request: Request) -> int:
    """Last-resort: extract card fields from the conversation log via one LLM call.

    Triggered when accumulated session state is sparse (model didn't follow
    the inline-tag protocol). Merges results into sess.fields, only filling
    keys that aren't already populated. Never overwrites tag-derived state.

    Returns count of fields recovered (0 on failure).
    """
    if not sess.messages:
        return 0

    provider_reg = getattr(request.app.state, "provider_registry", None)
    if not provider_reg or not getattr(provider_reg, "backends", None):
        return 0

    try:
        backend, resolved_model = await provider_reg.resolve_model_for_role(
            "utility", settings=settings,
        )
    except Exception as exc:
        log.warning("cardsmith_recovery_resolve_failed", error=str(exc))
        return 0

    convo_lines = []
    for m in sess.messages:
        role = m.get("role", "?").upper()
        content = m.get("content", "")
        if content:
            convo_lines.append(f"{role}: {content}")
    convo = "\n\n".join(convo_lines)

    chat_req = InternalChatRequest(
        model=resolved_model,
        messages=[
            Message(role="system", content=_RECOVERY_PROMPT),
            Message(role="user", content=f"<conversation>\n{convo}\n</conversation>"),
        ],
        stream=False,
        temperature=0.2,
        max_tokens=3000,
    )

    try:
        resp = await backend.chat(chat_req)
        raw = (resp.message.content or "").strip()
    except Exception as exc:
        log.warning("cardsmith_recovery_call_failed", error=str(exc))
        return 0

    # Strip code fences if the model wrapped despite instructions.
    if raw.startswith("```"):
        first_nl = raw.find("\n")
        last_fence = raw.rfind("```")
        if first_nl != -1 and last_fence > first_nl:
            raw = raw[first_nl + 1:last_fence].strip()

    try:
        extracted = json.loads(raw)
    except (ValueError, TypeError):
        log.warning(
            "cardsmith_recovery_json_parse_failed",
            session_id=sess.session_id,
            raw_preview=raw[:200],
        )
        return 0

    if not isinstance(extracted, dict):
        return 0

    # Merge into session state — only fill missing/empty keys. Never
    # overwrite something the tag protocol already committed.
    recovered = 0
    for k, v in extracted.items():
        if v is None:
            continue
        existing = sess.fields.get(k)
        is_empty = (
            existing is None
            or (isinstance(existing, str) and not existing.strip())
            or (isinstance(existing, list) and not existing)
        )
        if is_empty:
            sess.fields[k] = v
            recovered += 1

    log.info(
        "cardsmith_recovered_fields",
        session_id=sess.session_id,
        count=recovered,
        keys=sorted(extracted.keys()),
    )
    return recovered


async def _finalize_and_save(sess, request: Request) -> dict:
    """Build the character payload and persist it.

    Writes to:
      - ``ui_characters`` via the same _upsert_char helper character_routes uses
      - ``regex_scripts`` for any character-scoped scripts the model produced

    Returns ``{"char_id": "...", "name": "..."}`` on success. Drops the
    Cardsmith session on success or hard error.

    If the inline-tag protocol failed (sparse field state), runs a recovery
    extraction pass over the conversation log before building the payload.
    """
    from augmentum.proxy.character_routes import _upsert_char

    be = _backend(request)
    if not be:
        raise RuntimeError("No database backend")

    uid = sess.user_id

    sparse = _state_is_sparse(sess)
    examples_gap = _examples_missing_with_content(sess)
    if sparse or examples_gap:
        log.info(
            "cardsmith_recovery_triggered",
            session_id=sess.session_id,
            reason="sparse" if sparse else "examples_missing",
            accumulated_keys=sorted(sess.fields.keys()),
        )
        try:
            await _recover_fields_from_conversation(sess, request)
        except Exception as exc:
            log.warning(
                "cardsmith_recovery_unexpected_error",
                session_id=sess.session_id,
                error=str(exc),
                exc_info=True,
            )

    payload = build_character_payload(sess)
    char_id = payload["char_id"]
    name = payload["name"]
    data = payload["data"]
    avatar = payload["avatar"]

    await _upsert_char(be, char_id, name, data, avatar, uid=uid)

    # Insert regex_scripts rows
    if payload.get("regex_scripts"):
        now = datetime.now(UTC).isoformat()
        for row in payload["regex_scripts"]:
            try:
                if uid:
                    await be.conn.execute(
                        "INSERT INTO regex_scripts "
                        "(id, name, find_regex, replace_string, placement, "
                        " enabled, order_num, character_name, created_at, "
                        " updated_at, user_id) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            row["id"], row["name"], row["find_regex"],
                            row["replace_string"], row["placement"],
                            1 if row["enabled"] else 0, row["order_num"],
                            row["character_name"], now, now, uid,
                        ),
                    )
                else:
                    await be.conn.execute(
                        "INSERT INTO regex_scripts "
                        "(id, name, find_regex, replace_string, placement, "
                        " enabled, order_num, character_name, created_at, updated_at) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            row["id"], row["name"], row["find_regex"],
                            row["replace_string"], row["placement"],
                            1 if row["enabled"] else 0, row["order_num"],
                            row["character_name"], now, now,
                        ),
                    )
            except Exception as exc:
                log.warning(
                    "cardsmith_regex_insert_failed",
                    char_id=char_id,
                    name=row.get("name"),
                    error=str(exc),
                )
        await be.conn.commit()

    # ── Ensemble: persist character_groups row ──
    group_payload = payload.get("character_group")
    if group_payload:
        try:
            await _upsert_character_group(be, group_payload, uid=uid)
        except Exception as exc:
            log.warning(
                "cardsmith_character_group_insert_failed",
                char_id=char_id,
                error=str(exc),
                exc_info=True,
            )

    log.info(
        "cardsmith_card_saved",
        session_id=sess.session_id,
        char_id=char_id,
        name=name,
        user_id=uid,
        card_type=sess.card_type,
        regex_count=len(payload.get("regex_scripts") or []),
        lorebook_count=len(data.get("lorebook") or []),
        alt_greeting_count=len(data.get("alternateGreetings") or []),
        member_count=len((group_payload or {}).get("member_names") or []),
    )

    # Stamp the just-saved character onto the session's universe trail so
    # any chained continuation can reference what's already been built. The
    # disk row is preserved (not dropped) so a child session created via
    # parent_session_id can inherit this list.
    saves: list = []
    raw_prev = sess.meta.get("universe_saves")
    if isinstance(raw_prev, list):
        saves = [s for s in raw_prev if isinstance(s, dict)]
    saves.append({"char_id": char_id, "name": name})
    sess.meta["universe_saves"] = saves

    sess.finalized = True
    # Compute has_universe BEFORE the in-memory drop so the caller doesn't
    # have to re-read meta from a dropped session. True when the saved
    # session carries wiki context the next chained card can inherit.
    sp = sess.meta.get("scratchpad")
    has_universe = bool(sess.meta.get("wiki_host")) or bool(
        isinstance(sp, list) and sp
    )
    # Drop the in-memory entry to free the slot — disk path will rehydrate
    # on demand if a chained child resolves the parent via _resolve_session.
    drop_session(sess.session_id)
    # Persist the finalized row so chained children can inherit meta. A
    # separate periodic sweep (see migration 185 comment) prunes finalized
    # rows older than TTL — out of scope for this change.
    await _persist_session(sess, _backend(request))

    return {
        "char_id": char_id, "name": name, "has_universe": has_universe,
    }


async def _upsert_character_group(be, group: dict, *, uid: str) -> None:
    """INSERT (or replace) a character_groups row matching the cardsmith output.

    The group's ``name`` doubles as the join key with ``ui_characters.name``
    (the engine matches by name, not by id). Existing rows with the same
    name are replaced so users can iterate on a group via the editor.
    """
    import json as _json
    now = datetime.now(UTC).isoformat()
    group_id = "grp_" + uuid.uuid4().hex[:12]
    member_names_json = _json.dumps(group.get("member_names") or [])
    member_summaries_json = _json.dumps(group.get("member_summaries") or {})
    muted_names_json = _json.dumps(group.get("muted_names") or [])
    description = (group.get("description") or "")[:2000]
    generation_mode = group.get("generation_mode") or "llm_decide"
    group_avatar = group.get("avatar") or ""
    name = group.get("name") or ""

    # Drop any existing row with this group name (per user) so users can
    # re-roll the group via Cardsmith without leaking stale member lists.
    if uid:
        await be.conn.execute(
            "DELETE FROM character_groups WHERE name = ? AND user_id = ?",
            (name, uid),
        )
        await be.conn.execute(
            "INSERT INTO character_groups (id, name, description, member_names, "
            " generation_mode, member_summaries, avatar, muted_names, created_at, "
            " updated_at, user_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                group_id, name, description, member_names_json,
                generation_mode, member_summaries_json, group_avatar,
                muted_names_json, now, now, uid,
            ),
        )
    else:
        await be.conn.execute(
            "DELETE FROM character_groups WHERE name = ?",
            (name,),
        )
        await be.conn.execute(
            "INSERT INTO character_groups (id, name, description, member_names, "
            " generation_mode, member_summaries, avatar, muted_names, created_at, "
            " updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                group_id, name, description, member_names_json,
                generation_mode, member_summaries_json, group_avatar,
                muted_names_json, now, now,
            ),
        )
    await be.conn.commit()
