"""Character card CRUD — server-side storage for character cards."""

from __future__ import annotations

import base64
import json
import re
import time
from datetime import UTC, datetime
from html import unescape
from html.parser import HTMLParser

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from augmentum.proxy import system_events
from augmentum.state.write_guard import incoming_stamp, is_stale, stale_payload
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

router = APIRouter(prefix="/api/characters", tags=["characters"])


def _backend(request: Request):
    from augmentum.state.backends.sqlite import SQLiteBackend

    sm = getattr(request.app.state, "state_manager", None)
    if not sm or not isinstance(sm.backend, SQLiteBackend):
        return None
    return sm.backend


def _user_id(request: Request) -> str:
    """Extract user_id from authenticated request."""
    user = request.scope.get("user")
    return user.id if user else ""


# ── HTML cleaning ──────────────────────────────────────────────────────

def _is_safe_image_url(src: str) -> bool:
    """Check if an image URL is safe to preserve (https, http, or data URI)."""
    return bool(
        src
        and (
            src.startswith("https://")
            or src.startswith("http://")
            or src.startswith("data:image/")
        )
    )


# Regex to extract url(...) from inline CSS (background-image, etc.)
_BG_IMAGE_RE = re.compile(
    r"""background(?:-image)?\s*:\s*url\(\s*['"]?(https?://[^'")]+)['"]?\s*\)""",
    re.IGNORECASE,
)


class _HTMLStripper(HTMLParser):
    """Extract text from HTML, converting block elements to newlines.

    Image preservation:
    - ``<img>`` tags → markdown ``![alt](src)``
    - ``<a>`` wrapping an ``<img>`` → image only (link dropped)
    - ``<figure>``/``<figcaption>`` → image + caption text
    - Inline ``background-image: url(...)`` → markdown image
    - ``<style>`` blocks → fully stripped (prevents CSS garbage)

    Other media tags (video, audio, svg) are still dropped.
    """

    _BLOCK_TAGS = frozenset({
        "p", "div", "br", "hr", "li", "tr",
        "h1", "h2", "h3", "h4", "h5", "h6",
        "figure", "figcaption", "section", "article",
    })
    _SKIP_TAGS = frozenset({"video", "audio", "source", "picture", "svg", "path"})

    def __init__(self):
        super().__init__()
        self._parts: list[str] = []
        self._skip_depth = 0
        self._style_depth = 0  # inside <style> block

    def _emit_img(self, src: str, alt: str = "") -> None:
        """Emit a markdown image if the URL is safe."""
        if not _is_safe_image_url(src):
            return
        alt = (alt or "image").replace("[", "").replace("]", "")
        alt = alt.replace("(", "").replace(")", "")
        self._parts.append(f"\n![{alt}]({src})\n")

    def handle_starttag(self, tag: str, attrs: list) -> None:
        # <style> blocks — skip entirely (JanitorAI CSS)
        if tag == "style":
            self._style_depth += 1
            return
        if self._style_depth > 0:
            return

        if tag == "img":
            attr_dict = dict(attrs)
            self._emit_img(attr_dict.get("src", ""), attr_dict.get("alt", ""))
            return

        # <a> tags — transparent (just pass through, img inside handled above)
        if tag == "a":
            return

        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
            return

        if tag in self._BLOCK_TAGS:
            self._parts.append("\n")

        # Extract background-image from inline style (JanitorAI pattern)
        if tag == "div":
            attr_dict = dict(attrs)
            style = attr_dict.get("style", "")
            if style:
                m = _BG_IMAGE_RE.search(style)
                if m:
                    self._emit_img(m.group(1))

    def handle_endtag(self, tag: str) -> None:
        if tag == "style":
            self._style_depth = max(0, self._style_depth - 1)
            return
        if self._style_depth > 0:
            return
        if tag in ("img", "a"):
            return
        if tag in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in self._BLOCK_TAGS:
            self._parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0 and self._style_depth == 0:
            self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts)


def _clean_html(text: str) -> str:
    """Strip HTML tags from a field, keeping meaningful text content."""
    if not text or "<" not in text:
        return text  # Not HTML, return as-is

    stripper = _HTMLStripper()
    stripper.feed(text)
    result = unescape(stripper.get_text())

    # Collapse runs of whitespace on each line, then collapse blank lines
    lines = []
    for line in result.split("\n"):
        stripped = line.strip()
        if stripped:
            lines.append(stripped)
        elif lines and lines[-1] != "":
            lines.append("")  # Preserve one blank line as paragraph separator

    # Remove trailing blank lines
    while lines and lines[-1] == "":
        lines.pop()

    return "\n".join(lines)


# Patterns for JanitorAI creator credit lines.
# These appear at the end of personality, greeting, etc. fields.
# Matches: "created by Name on janitorai.com", "Created by Name 2026©",
# "made by Name on janitorai", "© Name on janitorai", etc.
# Uses \s* before "created" to handle inline (no newline) placement.
_CREATOR_PATTERNS: list[re.Pattern[str]] = [
    # "created/made/written by X on janitorai" or "created by X ©"
    re.compile(
        r"\s*(?:created|made|written|posted)\s+by\s+.{1,80}?(?:on\s+janitor|janitorai|©).*",
        re.IGNORECASE,
    ),
    # "© Name 2026 on janitorai" or standalone "© Name"
    re.compile(
        r"\s*©.{0,80}?(?:janitor|20\d{2}).*",
        re.IGNORECASE,
    ),
    # "on janitorai.com" at end
    re.compile(
        r"\s+on\s+janitorai\.com\s*$",
        re.IGNORECASE | re.MULTILINE,
    ),
]


def _strip_creator_lines(text: str) -> str:
    """Remove JanitorAI creator credit lines from text fields."""
    if not text:
        return text
    for pat in _CREATOR_PATTERNS:
        text = pat.sub("", text)
    # Clean up resulting blank lines
    lines = text.split("\n")
    out: list[str] = []
    for line in lines:
        if line.strip() == "" and out and out[-1].strip() == "":
            continue
        out.append(line)
    while out and out[-1].strip() == "":
        out.pop()
    while out and out[0].strip() == "":
        out.pop(0)
    return "\n".join(out)


# ── Card normalization helpers ─────────────────────────────────────────

def _normalize_card(raw: dict) -> dict | None:
    """Server-side mirror of the frontend normalizeCardData logic."""
    # V2/V3 with spec+data
    if raw.get("spec") and raw.get("data"):
        data = raw["data"]
        # V3 assets live at top level — carry them into the data dict
        if isinstance(raw.get("assets"), list) and "assets" not in data:
            data["assets"] = raw["assets"]
        return data
    # V2 without spec but nested data
    d = raw.get("data")
    if isinstance(d, dict) and (d.get("name") or d.get("char_name")):
        return d
    # Chub API response: { node: { definition: { ... } } } — users pasting
    # raw responses from api.chub.ai/api/characters/<full>?full=true land here
    node = raw.get("node")
    if isinstance(node, dict):
        node_defn = node.get("definition")
        if isinstance(node_defn, dict):
            return node_defn
    # Chub API response (already-unwrapped): definition at top level
    defn = raw.get("definition")
    if isinstance(defn, dict):
        return defn
    # Pygmalion format
    if raw.get("char_name"):
        return {
            "name": raw["char_name"],
            "description": raw.get("char_persona", ""),
            "personality": "",
            "scenario": raw.get("world_scenario", ""),
            "first_mes": raw.get("char_greeting", ""),
            "mes_example": raw.get("example_dialogue", ""),
        }
    # V1 flat / JanitorAI direct
    if raw.get("name") or raw.get("ch_name"):
        return raw
    return None


def _is_janitorai(data: dict) -> bool:
    """Detect JanitorAI-sourced cards by signature fields."""
    return (
        data.get("_source", "").startswith("janitorai")
        or data.get("_source") == "fetch_intercept"
        # JanitorAI uses 'first_message' + 'example_dialogs' (not TavernCard names)
        or ("first_message" in data and "example_dialogs" in data)
        # JanitorAI cards have creator_id / creator_name
        or "creator_id" in data
    )


def _resolve_alt_greetings(data: dict) -> list:
    """Extract alternate greetings from various card formats.

    TavernCard V2 uses ``alternate_greetings`` (list of strings).
    JanitorAI uses ``initial_messages`` — an array where index 0 is the
    primary greeting and subsequent entries are alternates.  Each entry
    can be a string or an object with a ``message`` or ``content`` key.
    """
    alt = data.get("alternate_greetings")
    if alt:
        return alt
    initial = data.get("initial_messages")
    if isinstance(initial, list) and len(initial) > 1:
        return initial[1:]
    return []


def _map_fields(data: dict) -> dict:
    """Map various card field names to Augmentum's canonical names.

    HTML is stripped from text fields (JanitorAI embeds rich HTML in descriptions).

    JanitorAI field remapping:
      - JanitorAI 'description' = user-facing showcase (images, links, FAQ) → creatorNotes
      - JanitorAI 'personality' = AI-facing character definition → description
    """
    janitor = _is_janitorai(data)

    raw_desc = data.get("description") or data.get("char_persona") or ""
    raw_pers = data.get("personality") or data.get("tavern_personality") or ""

    if janitor and raw_pers:
        # JanitorAI: personality has the real character definition
        # JanitorAI description is user-facing showcase — not character data,
        # but preserved in creator_notes for reference
        description = raw_pers
        personality = ""
        creator_notes = _clean_html(raw_desc)
    else:
        # Standard TavernCard / other formats: description is the character definition
        description = raw_desc
        personality = raw_pers
        creator_notes = data.get("creator_notes") or ""

    # For JanitorAI, apply creator-line stripping to all text fields
    clean = _clean_html
    if janitor:
        _clean = clean
        clean = lambda t: _strip_creator_lines(_clean(t))

    return {
        "name": data.get("name") or data.get("ch_name") or data.get("char_name") or "Unnamed",
        "description": clean(description),
        "personality": clean(personality),
        "scenario": clean(data.get("scenario") or data.get("world_scenario") or ""),
        "greeting": clean(
            data.get("first_mes") or data.get("first_message")
            or data.get("greeting_message") or data.get("char_greeting") or ""
        ),
        "examples": clean(
            data.get("mes_example") or data.get("example_dialogue")
            or data.get("example_dialogs") or ""
        ),
        "systemPrompt": clean(data.get("system_prompt") or ""),
        "postHistoryInstructions": clean(data.get("post_history_instructions") or ""),
        "creatorNotes": _clean_html(creator_notes),
        "tags": data.get("tags") or [],
        "alternateGreetings": [
            t for t in (
                clean(g) if isinstance(g, str)
                else clean(g.get("message", "") or g.get("content", ""))
                if isinstance(g, dict) else ""
                for g in _resolve_alt_greetings(data)
            ) if t and t.strip()
        ],
    }


def _build_char(fields: dict, data: dict) -> tuple[str, dict]:
    """Build a character object from normalized fields. Returns (char_id, char_dict)."""
    name = fields["name"]
    char_id = f"ch_{int(time.time() * 1000):x}{hash(name) % 0xFFF:03x}"

    char = {
        "id": char_id,
        "name": name,
        "description": fields["description"],
        "personality": fields["personality"],
        "scenario": fields["scenario"],
        "greeting": fields["greeting"],
        "examples": fields["examples"],
        "systemPrompt": fields["systemPrompt"],
        "postHistoryInstructions": fields["postHistoryInstructions"],
        "creatorNotes": fields["creatorNotes"],
        "tags": fields["tags"],
        "alternateGreetings": fields["alternateGreetings"],
        "lorebook": [],
        "createdAt": int(time.time() * 1000),
    }

    # Handle lorebook/character_book if present
    book = data.get("character_book") or data.get("embedded_lorebook")
    if isinstance(book, dict) and book.get("entries"):
        entries = book["entries"]
        if isinstance(entries, dict):
            entries = list(entries.values())
        char["lorebook"] = [
            {
                "name": e.get("name") or e.get("comment") or "",
                "keys": e.get("keys") or e.get("key") or [],
                "content": e.get("content") or "",
                "enabled": e.get("enabled", True),
                "priority": e.get("order") or e.get("priority") or 100,
            }
            for e in entries
            if isinstance(e, dict)
        ]

    # V3 assets: extract background image from structured assets array
    assets = data.get("assets")
    if isinstance(assets, list):
        for asset in assets:
            if not isinstance(asset, dict):
                continue
            asset_type = asset.get("type", "")
            uri = asset.get("uri", "")
            # Background images — store as backgroundImage on the character
            if asset_type == "background" and uri and _is_safe_image_url(uri):
                char["backgroundImage"] = uri
                break  # take first background

    return char_id, char


async def _upsert_char(be, char_id: str, name: str, char: dict, avatar: str = "", uid: str = ""):
    """Insert or update a character in the database.

    The ON CONFLICT branch carries a ``WHERE user_id`` guard so a known
    char_id from another tenant can't be clobbered by passing the same id
    in a PUT/POST. Rows with NULL user_id (legacy pre-auth data) are still
    writable by the first owner who claims them.
    """
    now = datetime.now(UTC).isoformat()
    data_json = json.dumps(char)
    if uid:
        await be.conn.execute(
            "INSERT INTO ui_characters (id, name, data, avatar, created_at, updated_at, user_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET name=?, data=?, avatar=?, updated_at=? "
            "WHERE ui_characters.user_id = ? OR ui_characters.user_id IS NULL",
            (char_id, name, data_json, avatar, now, now, uid,
             name, data_json, avatar, now,
             uid),
        )
    else:
        await be.conn.execute(
            "INSERT INTO ui_characters (id, name, data, avatar, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(id) DO UPDATE SET name=?, data=?, avatar=?, updated_at=? "
            "WHERE ui_characters.user_id IS NULL",
            (char_id, name, data_json, avatar, now, now,
             name, data_json, avatar, now),
        )
    await be.conn.commit()
    # Single create/update funnels through here (import-json + save). Tell the
    # caller's other devices to refetch /api/characters/ — user-scoped.
    system_events.publish("characters.changed", {"id": char_id}, user_id=uid)


# ── List ──────────────────────────────────────────────────────────────────

@router.get("/")
async def list_characters(request: Request):
    """Return all characters including avatars."""
    be = _backend(request)
    if not be:
        return JSONResponse({"characters": []})

    uid = _user_id(request)
    q = "SELECT id, name, data, avatar, created_at, updated_at FROM ui_characters"
    params: list = []
    if uid:
        q += " WHERE user_id = ?"
        params.append(uid)
    q += " ORDER BY updated_at DESC"
    cursor = await be.conn.execute(q, params)
    rows = await cursor.fetchall()
    chars = []
    for r in rows:
        char = json.loads(r[2])
        char["id"] = r[0]
        char["name"] = r[1]
        if r[3]:
            char["avatar"] = r[3]
        char["createdAt"] = r[4]
        char["updatedAt"] = r[5]
        chars.append(char)
    return JSONResponse({"characters": chars})


# ── Bulk import (localStorage migration) ─────────────────────────────────

@router.post("/import")
async def import_characters(request: Request):
    """Accept an array of character objects and upsert them all."""
    be = _backend(request)
    if not be:
        return JSONResponse({"error": "No database"}, status_code=503)

    body = await request.json()
    chars = body.get("characters", [])
    if not isinstance(chars, list):
        return JSONResponse({"error": "Expected {characters: [...]}"}, status_code=400)

    if len(chars) > 500:
        return JSONResponse({"error": "Too many characters (max 500)"}, status_code=400)

    now = datetime.now(UTC).isoformat()
    uid = _user_id(request)
    count = 0
    for char in chars:
        char_id = char.get("id")
        if not char_id:
            continue
        name = char.get("name", "Unnamed")
        avatar = char.pop("avatar", "") or ""
        # Enforce avatar size limit on bulk import
        if avatar and len(avatar) > _AVATAR_MAX_BYTES * 1.4:
            avatar = ""
        data_json = json.dumps(char)

        # ON CONFLICT carries a user_id guard — see _upsert_char for the
        # rationale (prevents cross-user clobber when char_id collides).
        if uid:
            await be.conn.execute(
                "INSERT INTO ui_characters (id, name, data, avatar, created_at, updated_at, user_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET name=?, data=?, avatar=?, updated_at=? "
                "WHERE ui_characters.user_id = ? OR ui_characters.user_id IS NULL",
                (char_id, name, data_json, avatar, now, now, uid,
                 name, data_json, avatar, now,
                 uid),
            )
        else:
            await be.conn.execute(
                "INSERT INTO ui_characters (id, name, data, avatar, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET name=?, data=?, avatar=?, updated_at=? "
                "WHERE ui_characters.user_id IS NULL",
                (char_id, name, data_json, avatar, now, now,
                 name, data_json, avatar, now),
            )
        count += 1

    await be.conn.commit()
    if count:
        system_events.publish("characters.changed", {"imported": count}, user_id=uid)
    log.info("characters_imported", count=count)
    return JSONResponse({"ok": True, "imported": count})


# ── Avatar download ───────────────────────────────────────────────────

_AVATAR_MAX_BYTES = 2 * 1024 * 1024  # 2 MB limit

async def _download_avatar(url: str) -> str:
    """Download an avatar URL and return as a base64 data URL, or '' on failure.

    Uses SafeHttpClient to block SSRF against internal/private IPs.
    """
    if not url or not url.startswith("http"):
        return ""
    try:
        from augmentum.utils.safe_http import SafeHttpClient

        client = SafeHttpClient(max_response_size=_AVATAR_MAX_BYTES)
        body, meta = await client.fetch_bytes(url, timeout=10.0)
        ct = (meta.get("content_type") or "image/jpeg").split(";")[0].strip()
        if not ct.startswith("image/"):
            return ""
        if len(body) > _AVATAR_MAX_BYTES:
            return ""
        b64 = base64.b64encode(body).decode("ascii")
        return f"data:{ct};base64,{b64}"
    except Exception as exc:
        log.debug("avatar_download_failed", url=url, error=str(exc))
        return ""


# ── Import raw JSON (bookmarklet / paste) ─────────────────────────────

@router.post("/import-json")
async def import_json(request: Request):
    """Accept raw character JSON (from bookmarklet, paste, any source) and create a character."""
    be = _backend(request)
    if not be:
        return JSONResponse({"error": "No database"}, status_code=503)

    try:
        raw = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    data = _normalize_card(raw)
    if not data:
        return JSONResponse({"error": "Unrecognized character card format"}, status_code=400)

    fields = _map_fields(data)
    char_id, char = _build_char(fields, data)

    # Download avatar if the source provides a URL
    # JanitorAI uses 'photo' or 'profile_image'; TavernCard uses 'avatar'
    # V3 uses assets array with type='icon'
    avatar = ""
    avatar_url = (
        data.get("avatar")
        or data.get("photo")
        or data.get("profile_image")
        or data.get("avatar_url")
        or ""
    )
    # V3 icon asset fallback
    if not avatar_url and isinstance(data.get("assets"), list):
        for asset in data["assets"]:
            if (
                isinstance(asset, dict)
                and asset.get("type") == "icon"
                and isinstance(asset.get("uri"), str)
                and asset["uri"].startswith("http")
            ):
                avatar_url = asset["uri"]
                break
    if isinstance(avatar_url, str) and avatar_url.startswith("http"):
        avatar = await _download_avatar(avatar_url)

    uid = _user_id(request)
    await _upsert_char(be, char_id, fields["name"], char, avatar, uid=uid)

    log.info("character_imported_json", id=char_id, name=fields["name"],
             avatar="yes" if avatar else "no")
    return JSONResponse({"ok": True, "id": char_id, "name": fields["name"]})


# ── Get single ────────────────────────────────────────────────────────────
# NOTE: /{char_id} routes MUST come after all specific path routes
# (import, import-json, janitor-fetch) to avoid catching them as char_id.

@router.get("/{char_id}")
async def get_character(char_id: str, request: Request):
    be = _backend(request)
    if not be:
        return JSONResponse({"error": "No database"}, status_code=503)

    uid = _user_id(request)
    q = "SELECT id, name, data, avatar, created_at, updated_at FROM ui_characters WHERE id = ?"
    params: list = [char_id]
    if uid:
        q += " AND user_id = ?"
        params.append(uid)
    cursor = await be.conn.execute(q, params)
    row = await cursor.fetchone()
    if not row:
        return JSONResponse({"error": "Not found"}, status_code=404)

    char = json.loads(row[2])
    char["id"] = row[0]
    char["name"] = row[1]
    if row[3]:
        char["avatar"] = row[3]
    return JSONResponse(char)


# ── Create / Update (upsert) ─────────────────────────────────────────────

async def _invalidate_narrative_caches_for_character(
    app_state, char_names: list[str], user_id: str,
) -> int:
    """Evict narrative engines + handlers whose session is linked to any of
    the given character names (for the given user). Called after a card is
    edited or deleted so the next request rehydrates with the fresh data
    from ``ui_characters`` instead of serving a stale in-memory copy.

    Returns the count of evicted cache entries (for logging).
    """
    if not char_names:
        return 0
    names_lc = {(n or "").lower() for n in char_names if n}
    if not names_lc:
        return 0

    engines = getattr(app_state, "narrative_engines", None)
    handlers = getattr(app_state, "narrative_handlers", None)
    evicted = 0

    def _key_matches(key) -> bool:
        # Cache keys are either bare session_id or (user_id, session_id).
        # Only evict entries belonging to this user — other users' sessions
        # are unaffected by one user's card edit.
        return isinstance(key, tuple) and len(key) == 2 and key[0] == user_id

    for cache in (engines, handlers):
        if not cache:
            continue
        to_remove = []
        for key, obj in list(cache.items()):
            if not _key_matches(key):
                continue
            # Engines: obj.state.character_card_name
            # Handlers: obj._engine.state.character_card_name
            state = getattr(obj, "state", None) or getattr(
                getattr(obj, "_engine", None), "state", None,
            )
            if state is None:
                continue
            name = (getattr(state, "character_card_name", "") or "").lower()
            if name and name in names_lc:
                to_remove.append(key)
        for k in to_remove:
            cache.pop(k, None)
            evicted += 1

    return evicted


async def _character_old_name(be, char_id: str, uid: str) -> str:
    """Fetch the pre-save name of a character so we can evict caches keyed
    on either the old or new name when the user renames a card."""
    try:
        q = "SELECT name FROM ui_characters WHERE id = ?"
        params: list = [char_id]
        if uid:
            q += " AND user_id = ?"
            params.append(uid)
        cur = await be.conn.execute(q, params)
        row = await cur.fetchone()
        return row[0] if row else ""
    except Exception:
        return ""


@router.put("/{char_id}")
async def save_character(char_id: str, request: Request):
    be = _backend(request)
    if not be:
        return JSONResponse({"error": "No database"}, status_code=503)

    body = await request.json()
    name = body.get("name", "Unnamed")
    avatar = body.pop("avatar", "") or ""
    # Enforce avatar size limit (base64 data URIs can be huge from paste/drag)
    if avatar and len(avatar) > _AVATAR_MAX_BYTES * 1.4:  # base64 ~1.37x raw
        avatar = ""  # silently drop oversized avatar
    uid = _user_id(request)

    # Stale-write guard — same contract as chats. Without this, a card open
    # in two tabs (or on phone + desktop) loses one side's edit silently:
    # the ON CONFLICT clause in _upsert_char is a TENANT guard, not a
    # staleness guard. Clients that send no ``updatedAt`` are accepted
    # unguarded, so older tabs keep working.
    if await is_stale(be.conn, "ui_characters", char_id,
                      incoming_stamp(body), user_id=uid):
        log.warning("character_save_stale_rejected", char_id=char_id)
        return JSONResponse(stale_payload(char_id), status_code=409)

    # Capture the pre-save name so renames evict caches keyed on either.
    old_name = await _character_old_name(be, char_id, uid) if uid else ""

    await _upsert_char(be, char_id, name, body, avatar, uid=uid)

    # Evict cached narrative engines/handlers tied to this character so the
    # next request rehydrates from the freshly-saved card (prevents stale
    # visual_traits / system_prompt / personality in scene image + inspector).
    try:
        evicted = await _invalidate_narrative_caches_for_character(
            request.app.state, [name, old_name], uid,
        )
        if evicted:
            log.info("narrative_cache_invalidated_on_char_save",
                     char_id=char_id, name=name, old_name=old_name,
                     evicted=evicted, user_id=uid)
    except Exception:
        log.warning("narrative_cache_invalidate_failed",
                    char_id=char_id, exc_info=True)

    return JSONResponse({"ok": True, "id": char_id})


# ── Avatar assignment ─────────────────────────────────────────────────────

@router.patch("/{character_id}/avatar")
async def assign_avatar_to_character(character_id: str, request: Request):
    """Assign an avatar to a character card.

    user_id is required: without it the AvatarStore UPDATE skips its
    ownership filter, letting any logged-in caller reassign someone
    else's avatar to a character_id they don't own.
    """
    uid = _user_id(request)
    if not uid:
        return JSONResponse({"error": "Unauthorized"}, status_code=401)
    body = await request.json()
    avatar_id = body.get("avatar_id")
    if not avatar_id:
        return JSONResponse({"error": "avatar_id required"}, status_code=400)
    from augmentum.avatar.store import AvatarStore
    be = _backend(request)
    if not be:
        return JSONResponse({"error": "No database"}, status_code=503)
    store = AvatarStore(be.conn)
    await store.assign_to_character(avatar_id, character_id, user_id=uid)
    return JSONResponse({"status": "ok"})


# ── Delete ────────────────────────────────────────────────────────────────

@router.delete("/{char_id}")
async def delete_character(char_id: str, request: Request):
    be = _backend(request)
    if not be:
        return JSONResponse({"error": "No database"}, status_code=503)

    uid = _user_id(request)
    old_name = await _character_old_name(be, char_id, uid) if uid else ""

    q = "DELETE FROM ui_characters WHERE id = ?"
    params: list = [char_id]
    if uid:
        q += " AND user_id = ?"
        params.append(uid)
    cursor = await be.conn.execute(q, params)
    await be.conn.commit()
    if cursor.rowcount == 0:
        return JSONResponse({"error": "Not found"}, status_code=404)
    system_events.publish("characters.changed", {"id": char_id, "deleted": True}, user_id=uid)

    # Evict cached engines/handlers tied to the deleted character so their
    # next request doesn't serve a phantom card from memory.
    try:
        if old_name:
            evicted = await _invalidate_narrative_caches_for_character(
                request.app.state, [old_name], uid,
            )
            if evicted:
                log.info("narrative_cache_invalidated_on_char_delete",
                         char_id=char_id, name=old_name, evicted=evicted)
    except Exception:
        log.warning("narrative_cache_invalidate_failed",
                    char_id=char_id, exc_info=True)

    return JSONResponse({"ok": True})
