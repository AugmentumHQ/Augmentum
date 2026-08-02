"""Group chat manager — multi-character conversations with turn management.

Adapted from SillyTavern's group chat system.  Supports round-robin,
random, and manual speaker selection modes.

In Augmentum's proxy architecture, group chats work by:
  1. Storing member list and generation mode in the group definition
  2. Tracking the current speaker index per session
  3. Rewriting the system prompt for the current speaker's character card
  4. Prefixing responses with character name for UI attribution
"""

from __future__ import annotations

import json
import random
import uuid
from dataclasses import dataclass, field

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


VALID_GENERATION_MODES = ("round_robin", "random", "manual", "llm_decide")


@dataclass
class CharacterGroup:
    """A group of characters for multi-character conversations."""

    id: str = ""
    name: str = ""
    description: str = ""
    member_names: list[str] = field(default_factory=list)
    # round_robin | random | manual | llm_decide
    generation_mode: str = "round_robin"
    member_summaries: dict[str, str] = field(default_factory=dict)  # name → custom summary
    avatar: str = ""  # base64 data URL
    # Members excluded from speaker rotation but still included in the joint
    # prompt as scene context. Keeps a character "present but silent".
    muted_names: list[str] = field(default_factory=list)
    # Owner — every query goes through this predicate so a client can't load
    # another user's group via X-Augmentum-Group-Id.
    user_id: str = ""

    def __post_init__(self) -> None:
        if not self.id:
            self.id = uuid.uuid4().hex[:12]
        # Unknown modes fall back to round_robin so a typo in the DB doesn't
        # break the chat — dropdown is the source of truth going forward.
        if self.generation_mode not in VALID_GENERATION_MODES:
            self.generation_mode = "round_robin"

    def is_muted(self, name: str) -> bool:
        """Case-insensitive mute check."""
        lower = (name or "").lower()
        return any(m.lower() == lower for m in self.muted_names)

    def unmuted_members(self) -> list[str]:
        """Members eligible to speak."""
        return [n for n in self.member_names if not self.is_muted(n)]


class GroupStore:
    """CRUD for character groups backed by SQLite."""

    def __init__(self, conn) -> None:
        self._conn = conn

    _SELECT_COLS = (
        "id, name, description, member_names, generation_mode, "
        "member_summaries, avatar, muted_names, user_id"
    )

    @staticmethod
    def _parse_row(r) -> CharacterGroup:
        summaries = {}
        if len(r) > 5 and r[5]:
            try:
                summaries = json.loads(r[5])
            except (json.JSONDecodeError, TypeError):
                pass
        muted: list[str] = []
        if len(r) > 7 and r[7]:
            try:
                parsed = json.loads(r[7])
                if isinstance(parsed, list):
                    muted = [str(x) for x in parsed if x]
            except (json.JSONDecodeError, TypeError):
                pass
        return CharacterGroup(
            id=r[0], name=r[1], description=r[2],
            member_names=json.loads(r[3]) if r[3] else [],
            generation_mode=r[4],
            member_summaries=summaries,
            avatar=r[6] if len(r) > 6 else "",
            muted_names=muted,
            user_id=(r[8] if len(r) > 8 else "") or "",
        )

    async def list_groups(self, *, user_id: str = "") -> list[CharacterGroup]:
        """List groups owned by ``user_id``. Raises if uid is empty rather
        than silently returning every tenant's groups — auth middleware
        guarantees a uid on /api/narrative/*, so an empty value here means
        a caller bypassed that guarantee and we should fail loud."""
        if not user_id:
            raise ValueError("character_groups list requires user_id")
        cursor = await self._conn.execute(
            f"SELECT {self._SELECT_COLS} FROM character_groups "
            "WHERE user_id = ? ORDER BY name",
            (user_id,),
        )
        rows = await cursor.fetchall()
        return [self._parse_row(r) for r in rows]

    async def get_group(
        self, group_id: str, *, user_id: str = "",
    ) -> CharacterGroup | None:
        """Load a group scoped to ``user_id``. Returning None here is what
        blocks the ``X-Augmentum-Group-Id`` cross-tenant path — a forged id
        belonging to another user simply doesn't match the
        ``AND user_id = ?`` predicate. Raises on empty uid rather than
        degrading to a global lookup; see ``list_groups`` for rationale."""
        if not user_id:
            raise ValueError("character_groups get requires user_id")
        cursor = await self._conn.execute(
            f"SELECT {self._SELECT_COLS} FROM character_groups "
            "WHERE id = ? AND user_id = ?",
            (group_id, user_id),
        )
        r = await cursor.fetchone()
        if not r:
            return None
        return self._parse_row(r)

    async def save_group(
        self, group: CharacterGroup, *, user_id: str = "",
    ) -> CharacterGroup:
        # Stamp the owner so an unauthenticated-era row doesn't get saved
        # without one, and so renames/updates don't drop the owner.
        owner = user_id or group.user_id
        if not owner:
            raise ValueError("character_groups insert requires user_id")
        group.user_id = owner
        await self._conn.execute(
            "INSERT OR REPLACE INTO character_groups "
            "(id, name, description, member_names, generation_mode, "
            "member_summaries, avatar, muted_names, user_id, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))",
            (
                group.id, group.name, group.description,
                json.dumps(group.member_names),
                group.generation_mode,
                json.dumps(group.member_summaries),
                group.avatar,
                json.dumps(group.muted_names),
                owner,
            ),
        )
        await self._conn.commit()
        return group

    async def delete_group(
        self, group_id: str, *, user_id: str = "",
    ) -> bool:
        if not user_id:
            raise ValueError("character_groups delete requires user_id")
        cursor = await self._conn.execute(
            "DELETE FROM character_groups WHERE id = ? AND user_id = ?",
            (group_id, user_id),
        )
        await self._conn.commit()
        return cursor.rowcount > 0


class GroupTurnManager:
    """Tracks turn order for a group chat session."""

    def __init__(self, group: CharacterGroup) -> None:
        self._group = group
        self._current_index = 0
        self._last_speaker: str | None = None

    @property
    def current_speaker(self) -> str:
        """Name of the character whose turn it is."""
        if not self._group.member_names:
            return ""
        return self._group.member_names[self._current_index % len(self._group.member_names)]

    @property
    def last_speaker(self) -> str | None:
        return self._last_speaker

    def advance(self) -> str:
        """Advance to the next speaker and return their name.

        Muted members are skipped in round_robin and random modes. If every
        member is muted, we fall back to the first member (with a log warning
        upstream) — silencing the whole group would leave chat unable to
        progress, which is worse than slightly violating the mute intent.
        """
        if not self._group.member_names:
            return ""

        self._last_speaker = self.current_speaker
        unmuted = self._group.unmuted_members()

        # All-muted fallback: use the full list so the chat still advances.
        pool = unmuted if unmuted else list(self._group.member_names)

        if self._group.generation_mode == "round_robin":
            # Walk forward until we land on an unmuted member (if any exist).
            # Cap iterations at len(members) to avoid an infinite loop on the
            # all-muted path.
            members = self._group.member_names
            for _ in range(len(members)):
                self._current_index = (self._current_index + 1) % len(members)
                if members[self._current_index] in pool:
                    break
        elif self._group.generation_mode == "random":
            # Prefer candidates that aren't the last speaker, from the pool.
            candidates = [
                i for i, name in enumerate(self._group.member_names)
                if name in pool and name != self._last_speaker
            ]
            if not candidates:
                # Fall back to any pool member (even if it's last speaker).
                candidates = [
                    i for i, name in enumerate(self._group.member_names)
                    if name in pool
                ]
            if candidates:
                self._current_index = random.choice(candidates)  # noqa: S311
        # manual / llm_decide: caller is responsible for set_speaker before
        # the turn runs; advance() does not rotate automatically for them.

        return self.current_speaker

    def set_speaker(self, name: str) -> bool:
        """Manually set the current speaker. Allowed even if the member is
        muted — explicit user intent overrides the mute. Caller decides
        whether to log a warning."""
        for i, member in enumerate(self._group.member_names):
            if member.lower() == name.lower():
                self._current_index = i
                return True
        return False

    def to_dict(self) -> dict:
        return {
            "group_id": self._group.id,
            "current_index": self._current_index,
            "last_speaker": self._last_speaker,
            "current_speaker": self.current_speaker,
            "generation_mode": self._group.generation_mode,
            "members": self._group.member_names,
            "muted": list(self._group.muted_names),
        }
