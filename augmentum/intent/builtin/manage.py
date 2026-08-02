"""Content-management verbs — wiring program Phase 5.

Playlist create/rename/delete, file favorite, chat rename, library
save rename. All wrap stores/tables the UI already uses (playlists,
file_index, ui_sessions, library_publications) — same rows, same
surfaces; companion-created playlists carry ``origin='companion'``
(provenance principle 2; the playlist dropdown suffixes the label).

Deliberately deferred:
- ``playlist.add`` — items carry a typed schema (youtube/file refs)
  that needs the CLIENT's current-track context; the playlist UI's
  add-buttons are app.act's lane (context-bound, arg-less).
- ``game.pin`` — pinning needs the browse context (source/source_id)
  only the games surface has; app.act candidate, not a verb.
- ``chat.delete`` — stakes (Phase C ladder), per the program spec.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from augmentum.intent.action import ActionFanout, ActionResult, SessionContext
from augmentum.intent.registry import register_action
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_TIER3_ONLY = ActionFanout(tier1=False, tier2=False, tier3=True)


def _conn(session: SessionContext):
    sm = getattr(session.app_state, "state_manager", None) if session.app_state else None
    backend = getattr(sm, "backend", None)
    return getattr(backend, "conn", None)


def _gate(session: SessionContext) -> ActionResult | None:
    if not session.user_id:
        return ActionResult(
            short_circuit=True,
            speak="I can't manage content for a signed-out session.",
        )
    if _conn(session) is None:
        return ActionResult(
            short_circuit=True,
            speak="I can't reach storage right now.",
        )
    return None


async def _find_playlist(conn, user_id: str, query: str):
    """Resolve a playlist by name containment. Returns (row|None, matches)."""
    cur = await conn.execute(
        "SELECT id, name FROM playlists WHERE user_id = ? "
        "ORDER BY updated_at DESC LIMIT 50",
        (user_id,),
    )
    rows = await cur.fetchall()
    q = query.strip().lower()
    matches = [
        r for r in rows
        if q and (q in str(r[1]).lower() or str(r[1]).lower() in q)
    ]
    if len(matches) == 1:
        return matches[0], matches
    return None, matches


# ---------------------------------------------------------------------------
# playlist.create / rename / delete
# ---------------------------------------------------------------------------

async def _playlist_create(
    text: str, session: SessionContext, args: dict[str, Any],
) -> ActionResult:
    blocked = _gate(session)
    if blocked:
        return blocked
    name = str(args.get("name") or "").strip()[:120]
    if not name:
        return ActionResult(
            short_circuit=True,
            speak="What should the playlist be called?",
            clarify={"missing": ["name"], "args": {}},
        )
    conn = _conn(session)
    pl_id = uuid.uuid4().hex[:12]
    await conn.execute(
        "INSERT INTO playlists (id, user_id, name, items_json, origin) "
        "VALUES (?, ?, ?, ?, 'companion')",
        (pl_id, session.user_id, name, json.dumps([])),
    )
    await conn.commit()
    log.info("playlist_create_verb", user_id=session.user_id, playlist_id=pl_id)
    return ActionResult(
        short_circuit=True,
        speak=f"Made a playlist called {name}.",
        toast=f"Playlist: {name}",
        digest=f"playlist created: {name}",
    )


register_action(
    id="playlist.create",
    summary=(
        "Silently create a new, empty playlist with the given name in "
        "the user's music library (Grove). Adding items happens from "
        "the player UI. Siblings: renaming is playlist.rename, "
        "removing one is playlist.delete."
    ),
    examples=[
        "make me a playlist called late night coding",
        "new playlist for workout music",
    ],
    arg_schema={
        "name": {"type": "string", "description": "The playlist name."},
    },
    fanout=_TIER3_ONLY,
    handler=_playlist_create,
    delivery="verbal",
)


async def _playlist_rename(
    text: str, session: SessionContext, args: dict[str, Any],
) -> ActionResult:
    blocked = _gate(session)
    if blocked:
        return blocked
    query = str(args.get("query") or "").strip()
    new_name = str(args.get("new_name") or "").strip()[:120]
    if not query or not new_name:
        missing = [k for k, v in (("query", query), ("new_name", new_name)) if not v]
        return ActionResult(
            short_circuit=True,
            speak="Which playlist, and what's the new name?",
            clarify={"missing": missing, "args": dict(args)},
        )
    conn = _conn(session)
    row, matches = await _find_playlist(conn, session.user_id, query)
    if row is None:
        if matches:
            names = ", or ".join(str(m[1]) for m in matches[:4])
            return ActionResult(
                short_circuit=True,
                speak=f"Which one — {names}?",
                clarify={"missing": ["query"], "args": dict(args)},
            )
        return ActionResult(
            short_circuit=True,
            speak=f"I don't see a playlist like {query[:50]}.",
        )
    await conn.execute(
        "UPDATE playlists SET name = ?, updated_at = datetime('now') "
        "WHERE id = ? AND user_id = ?",
        (new_name, row[0], session.user_id),
    )
    await conn.commit()
    return ActionResult(
        short_circuit=True,
        speak=f"Renamed {row[1]} to {new_name}.",
        digest=f"playlist renamed: {new_name}",
    )


register_action(
    id="playlist.rename",
    summary=(
        "Silently rename one of the user's playlists. Siblings: "
        "playlist.create makes a new one, playlist.delete removes one."
    ),
    examples=[
        "rename my chill playlist to evening wind-down",
        "call that playlist road trip instead",
    ],
    arg_schema={
        "query": {"type": "string", "description": "The current playlist name (or part of it)."},
        "new_name": {"type": "string", "description": "The new name."},
    },
    fanout=_TIER3_ONLY,
    handler=_playlist_rename,
    delivery="verbal",
)


async def _playlist_delete(
    text: str, session: SessionContext, args: dict[str, Any],
) -> ActionResult:
    blocked = _gate(session)
    if blocked:
        return blocked
    conn = _conn(session)
    playlist_id = str(args.get("playlist_id") or "").strip()
    confirm = str(args.get("confirm") or "").strip()
    if playlist_id and confirm:
        from augmentum.intent.builtin.memory_admin import _reads_assent
        if not _reads_assent(confirm):
            return ActionResult(short_circuit=True, speak="Kept it.")
        await conn.execute(
            "DELETE FROM playlists WHERE id = ? AND user_id = ?",
            (playlist_id, session.user_id),
        )
        await conn.commit()
        log.info(
            "playlist_delete_verb",
            user_id=session.user_id, playlist_id=playlist_id,
        )
        return ActionResult(
            short_circuit=True,
            speak="Deleted.",
            digest="playlist deleted at the user's request",
        )
    query = str(args.get("query") or "").strip()
    if not query:
        return ActionResult(
            short_circuit=True,
            speak="Which playlist should I delete?",
            clarify={"missing": ["query"], "args": {}},
        )
    row, matches = await _find_playlist(conn, session.user_id, query)
    if row is None:
        if matches:
            names = ", or ".join(str(m[1]) for m in matches[:4])
            return ActionResult(
                short_circuit=True,
                speak=f"Which one — {names}?",
                clarify={"missing": ["query"], "args": dict(args)},
            )
        return ActionResult(
            short_circuit=True,
            speak=f"I don't see a playlist like {query[:50]}.",
        )
    return ActionResult(
        short_circuit=True,
        speak=f"Delete the playlist {row[1]} — its items go with it. Sure?",
        clarify={
            "missing": ["confirm"],
            "args": {"playlist_id": row[0], "query": query},
        },
    )


register_action(
    id="playlist.delete",
    summary=(
        "Delete one of the user's playlists — names the playlist back "
        "and only deletes after the user confirms (recall-then-confirm, "
        "same contract as memory.forget). Sibling: playlist.rename "
        "keeps it."
    ),
    examples=[
        "delete my old workout playlist",
        "get rid of that test playlist",
    ],
    arg_schema={
        "query": {"type": "string", "description": "Which playlist, in the user's words."},
        "playlist_id": {"type": "string", "description": "Internal — set by the confirm flow."},
        "confirm": {"type": "string", "description": "Internal — the user's answer."},
    },
    fanout=_TIER3_ONLY,
    handler=_playlist_delete,
    delivery="verbal",
    stakes="disruptive",
)


# ---------------------------------------------------------------------------
# file.favorite
# ---------------------------------------------------------------------------

async def _file_favorite(
    text: str, session: SessionContext, args: dict[str, Any],
) -> ActionResult:
    blocked = _gate(session)
    if blocked:
        return blocked
    conn = _conn(session)
    on = args.get("off") is not True

    file_id = ""
    label = ""
    query = str(args.get("query") or "").strip()
    if query:
        from augmentum.media.resolver import resolve_media
        result = await resolve_media(
            getattr(session, "app_state", None),
            user_id=session.user_id, query=query,
        )
        if result.decision == "play" and result.top is not None:
            file_id = result.top.file_id
            label = result.top.title
        elif result.candidates:
            names = "; ".join(c.title for c in result.candidates[:3])
            return ActionResult(
                short_circuit=True,
                speak=f"A few match — {names}. Which one?",
                clarify={"missing": ["query"], "args": dict(args)},
            )
    else:
        refs = getattr(session, "referents", None)
        file_id = getattr(refs, "last_file_id", "") or "" if refs else ""
    if not file_id:
        return ActionResult(
            short_circuit=True,
            speak="Which file or title do you mean?",
            clarify={"missing": ["query"], "args": dict(args)},
        )
    cur = await conn.execute(
        "UPDATE file_index SET is_favorite = ?, updated_at = datetime('now') "
        "WHERE id = ? AND user_id = ?",
        (1 if on else 0, file_id, session.user_id),
    )
    await conn.commit()
    if cur.rowcount == 0:
        return ActionResult(
            short_circuit=True,
            speak="I couldn't find that file in the index.",
        )
    word = "Favorited" if on else "Unfavorited"
    spoken = f"{word} {label}." if label else f"{word}."
    log.info(
        "file_favorite_verb",
        user_id=session.user_id, file_id=file_id, on=on,
    )
    return ActionResult(
        short_circuit=True,
        speak=spoken,
        digest=f"file {'favorited' if on else 'unfavorited'}: {label or file_id}",
    )


register_action(
    id="file.favorite",
    summary=(
        "Silently mark a library file as a favorite (or unmark it) — "
        "'favorite this' uses whatever just played; a named title gets "
        "resolved against the library. Shows up in the same Favorites "
        "the user's own stars feed."
    ),
    examples=[
        "favorite this", "add that audiobook to my favorites",
        "unfavorite the foundation",
    ],
    arg_schema={
        "query": {
            "type": "string",
            "description": "Title to favorite. Omit to use what just played.",
        },
        "off": {"type": "boolean", "description": "True to UNfavorite."},
    },
    fanout=_TIER3_ONLY,
    handler=_file_favorite,
    delivery="verbal",
)


# ---------------------------------------------------------------------------
# chat.rename
# ---------------------------------------------------------------------------

async def _chat_rename(
    text: str, session: SessionContext, args: dict[str, Any],
) -> ActionResult:
    blocked = _gate(session)
    if blocked:
        return blocked
    title = str(args.get("title") or "").strip()[:120]
    if not title:
        return ActionResult(
            short_circuit=True,
            speak="What should this chat be called?",
            clarify={"missing": ["title"], "args": {}},
        )
    conn = _conn(session)
    cur = await conn.execute(
        "UPDATE ui_sessions SET title = ?, updated_at = datetime('now') "
        "WHERE id = ? AND user_id = ?",
        (title, session.session_id, session.user_id),
    )
    await conn.commit()
    if cur.rowcount == 0:
        return ActionResult(
            short_circuit=True,
            speak="This conversation isn't saved as a chat yet — it'll pick the name up once it is.",
        )
    return ActionResult(
        short_circuit=True,
        speak=f"Renamed this chat to {title}.",
        digest=f"chat renamed: {title}",
        surface_emit={
            "channel": "chat.renamed",
            "payload": {"session_id": session.session_id, "title": title},
        },
    )


register_action(
    id="chat.rename",
    summary=(
        "Rename the CURRENT chat session — the title shown in the "
        "chat list. Call for 'rename this chat to X', 'call this "
        "conversation Y'. Sibling: starting fresh is chat.new."
    ),
    examples=[
        "rename this chat to garden planning",
        "call this conversation budget talk",
    ],
    arg_schema={
        "title": {"type": "string", "description": "The new chat title."},
    },
    fanout=_TIER3_ONLY,
    handler=_chat_rename,
    delivery="verbal",
)


# ---------------------------------------------------------------------------
# library.rename
# ---------------------------------------------------------------------------

async def _library_rename(
    text: str, session: SessionContext, args: dict[str, Any],
) -> ActionResult:
    blocked = _gate(session)
    if blocked:
        return blocked
    query = str(args.get("query") or "").strip().lower()
    new_title = str(args.get("new_title") or "").strip()[:160]
    description = str(args.get("description") or "").strip()[:500]
    if not query or not (new_title or description):
        return ActionResult(
            short_circuit=True,
            speak="Which library save, and what should change?",
            clarify={"missing": ["query"], "args": dict(args)},
        )
    conn = _conn(session)
    cur = await conn.execute(
        "SELECT id, title FROM library_publications WHERE user_id = ? "
        "ORDER BY updated_at DESC LIMIT 50",
        (session.user_id,),
    )
    rows = await cur.fetchall()
    matches = [r for r in rows if query in str(r[1]).lower()]
    if not matches:
        return ActionResult(
            short_circuit=True,
            speak=f"I don't see a library save like {query[:50]}.",
        )
    if len(matches) > 1:
        names = ", or ".join(str(m[1]) for m in matches[:4])
        return ActionResult(
            short_circuit=True,
            speak=f"Which one — {names}?",
            clarify={"missing": ["query"], "args": dict(args)},
        )
    sets, params = [], []
    if new_title:
        sets.append("title = ?")
        params.append(new_title)
    if description:
        sets.append("description = ?")
        params.append(description)
    params += [matches[0][0], session.user_id]
    await conn.execute(
        f"UPDATE library_publications SET {', '.join(sets)}, "
        "updated_at = datetime('now') WHERE id = ? AND user_id = ?",
        params,
    )
    await conn.commit()
    changed = new_title or matches[0][1]
    return ActionResult(
        short_circuit=True,
        speak=f"Updated — it's {changed} now.",
        digest=f"library save updated: {changed}",
    )


register_action(
    id="library.rename",
    summary=(
        "Silently rename or re-describe one of the user's library "
        "saves (published apps/pages). Call for 'rename my saved X to "
        "Y', 'update the description on Z'."
    ),
    examples=[
        "rename my saved budget app to family budget",
        "update the description on my recipe page",
    ],
    arg_schema={
        "query": {"type": "string", "description": "Which save, by (partial) title."},
        "new_title": {"type": "string", "description": "New title (optional)."},
        "description": {"type": "string", "description": "New description (optional)."},
    },
    fanout=_TIER3_ONLY,
    handler=_library_rename,
    delivery="verbal",
)
