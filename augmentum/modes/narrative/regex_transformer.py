"""Regex script transformer — input/output text processing pipeline.

Adapted from SillyTavern's applyRegexScripts() pattern.  Scripts execute
in order_num ascending order; each script's output feeds into the next.

Scripts can be global (no character_name) or character-specific.
Invalid regex patterns are caught and skipped — they never crash the pipeline.
"""

from __future__ import annotations

import asyncio
import re
import uuid
from dataclasses import dataclass

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class RegexScript:
    """A single regex find-replace rule."""

    id: str = ""
    name: str = ""
    find_regex: str = ""
    replace_string: str = ""
    placement: str = "output"  # "input", "output", "both"
    enabled: bool = True
    order_num: int = 100
    character_name: str | None = None  # None = global
    # Client edit-stamp (ms epoch) for the stale-write guard. Distinct from
    # the server's ``updated_at`` column — see augmentum/state/write_guard.py.
    client_updated_at: int = 0

    def __post_init__(self) -> None:
        if not self.id:
            self.id = uuid.uuid4().hex[:12]


class RegexScriptStore:
    """CRUD for regex scripts backed by SQLite.

    Every script is tenant-scoped via ``user_id`` so one user's regex can't
    match or mutate text in another user's roleplay pipeline.
    """

    def __init__(self, conn) -> None:
        self._conn = conn

    async def list_scripts(
        self,
        character_name: str | None = None,
        *,
        user_id: str = "",
    ) -> list[RegexScript]:
        """List scripts. If character_name is given, return global + that character's scripts."""
        where: list[str] = []
        params: list = []
        if character_name:
            where.append("(character_name IS NULL OR character_name = ?)")
            params.append(character_name)
        if user_id:
            where.append("user_id = ?")
            params.append(user_id)
        query = (
            "SELECT id, name, find_regex, replace_string, placement, "
            "enabled, order_num, character_name, client_updated_at "
            "FROM regex_scripts"
        )
        if where:
            query += " WHERE " + " AND ".join(where)
        query += " ORDER BY order_num"
        cursor = await self._conn.execute(query, params)
        rows = await cursor.fetchall()
        return [
            RegexScript(
                id=r[0], name=r[1], find_regex=r[2], replace_string=r[3],
                placement=r[4], enabled=bool(r[5]), order_num=r[6],
                character_name=r[7],
                client_updated_at=r[8] if len(r) > 8 and r[8] is not None else 0,
            )
            for r in rows
        ]

    async def save_script(
        self, script: RegexScript, *, user_id: str = "",
    ) -> RegexScript:
        if not user_id:
            raise ValueError("regex_scripts insert requires user_id")
        await self._conn.execute(
            "INSERT OR REPLACE INTO regex_scripts "
            "(id, name, find_regex, replace_string, placement, enabled, "
            "order_num, character_name, user_id, client_updated_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))",
            (
                script.id, script.name, script.find_regex,
                script.replace_string, script.placement,
                int(script.enabled), script.order_num, script.character_name,
                user_id, script.client_updated_at,
            ),
        )
        await self._conn.commit()
        return script

    async def delete_script(
        self, script_id: str, *, user_id: str = "",
    ) -> bool:
        if not user_id:
            raise ValueError("regex_scripts delete requires user_id")
        cursor = await self._conn.execute(
            "DELETE FROM regex_scripts WHERE id = ? AND user_id = ?",
            (script_id, user_id),
        )
        await self._conn.commit()
        return cursor.rowcount > 0

    async def toggle_script(
        self, script_id: str, enabled: bool, *, user_id: str = "",
    ) -> bool:
        if not user_id:
            raise ValueError("regex_scripts toggle requires user_id")
        cursor = await self._conn.execute(
            "UPDATE regex_scripts SET enabled = ?, updated_at = datetime('now') "
            "WHERE id = ? AND user_id = ?",
            (int(enabled), script_id, user_id),
        )
        await self._conn.commit()
        return cursor.rowcount > 0


def apply_regex_scripts(
    text: str,
    scripts: list[RegexScript],
    placement: str,
) -> str:
    """Apply regex scripts to text for the given placement stage.

    Parameters
    ----------
    text:
        The text to transform.
    scripts:
        All scripts (pre-sorted by order_num).
    placement:
        "input" or "output" — only scripts matching this placement (or "both") run.
    """
    if not text or not scripts:
        return text

    result = text
    for script in scripts:
        if not script.enabled:
            continue
        if script.placement != placement and script.placement != "both":
            continue
        if not script.find_regex:
            continue
        try:
            pattern = re.compile(script.find_regex)
            replacement = script.replace_string
            # Expand {{random:a,b,c}} macros in replacement string per match
            if "{{random:" in replacement:
                result = _sub_with_random(pattern, replacement, result)
            else:
                result = pattern.sub(replacement, result)
        except re.error:
            log.debug("regex_script_invalid", script_id=script.id, pattern=script.find_regex)
            continue
    return result


async def apply_regex_scripts_safe(
    text: str,
    scripts: list[RegexScript],
    placement: str,
    timeout: float = 2.0,
) -> str:
    """Async wrapper around :func:`apply_regex_scripts` with ReDoS protection.

    Runs the synchronous regex pipeline in a thread and enforces a timeout
    so that catastrophic backtracking patterns cannot block the event loop.
    Returns the original *text* unchanged on timeout.
    """
    if not text or not scripts:
        return text
    try:
        return await asyncio.wait_for(
            asyncio.to_thread(apply_regex_scripts, text, scripts, placement),
            timeout=timeout,
        )
    except TimeoutError:
        log.warning(
            "regex_scripts_timeout",
            placement=placement,
            script_count=len(scripts),
        )
        return text


# Pre-compiled pattern for {{random:...}} in replacement strings
_RANDOM_LIST_RE = re.compile(r"\{\{random:([^}]+)\}\}", re.IGNORECASE)


def _sub_with_random(pattern: re.Pattern, replacement: str, text: str) -> str:
    """Substitute with per-match {{random:a,b,c}} expansion.

    Each match independently picks a random item, so different matches
    in the same text can get different replacements.
    """
    import random

    def _replacer(match: re.Match) -> str:
        # First apply capture group backreferences
        expanded = match.expand(replacement)
        # Then expand {{random:...}} macros
        def _pick(m: re.Match) -> str:
            items = [item.strip() for item in m.group(1).split(",") if item.strip()]
            return random.choice(items) if items else ""
        return _RANDOM_LIST_RE.sub(_pick, expanded)

    return pattern.sub(_replacer, text)
