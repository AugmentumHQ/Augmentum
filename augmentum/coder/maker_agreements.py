"""Working Agreements — durable, model-agnostic "how this maker works" memory.

The companion's skill graph remembers what *worked*; the lesson registry
remembers what she was *corrected* on. This is the third axis, for the
**assistant** relationship: standing operating principles for how the
coding assistant should work with a particular person — "tell me the
blast radius before irreversible changes", "finish one thing well over
starting three", "prefer the strong-foundation option over a shortcut
with UX caveats".

Unlike lessons (situational corrections, retrieved by embedding
similarity), agreements are *always-on*: every active agreement is
injected into the system prompt each turn, so they condition the whole
relationship rather than firing on a matched situation. That is the
point — the relationship stops living only in one model's private
scratchpad and becomes the user's own, server-persisted and
MODEL-AGNOSTIC: any local model they run, and any future instance of the
assistant, inherits how they think.

User-scoped. Ships empty — each user accrues their own; nothing is baked
in (OSS-clean, persona-agnostic).
"""

from __future__ import annotations

from dataclasses import dataclass

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Categories are advisory (free-text is allowed) but this is the curated
# vocabulary the UI/seed use, ordered roughly by how they read in a prompt.
CATEGORIES = (
    "scope",          # how big/small to go, what to finish
    "reliability",    # correctness, verification, don't-break-the-daily-driver
    "process",        # how to work — confirm-first, no-bare-commit, verify
    "communication",  # how to talk — blast radius first, honest about failures
    "aesthetics",     # taste — comfort over premium, register, naming
    "general",
)

_MAX_PRINCIPLE_CHARS = 280
_MAX_RATIONALE_CHARS = 400


@dataclass
class Agreement:
    id: int
    user_id: str | None
    principle: str
    rationale: str
    category: str
    source: str
    strength: float
    times_seen: int
    status: str

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "principle": self.principle,
            "rationale": self.rationale,
            "category": self.category,
            "source": self.source,
            "strength": self.strength,
            "times_seen": self.times_seen,
            "status": self.status,
        }


def _norm(text: str) -> str:
    """Normalised key for dedup — case/space-insensitive, trailing-period
    tolerant so 'Tell me first' and 'tell me first.' collapse."""
    return " ".join((text or "").lower().split()).rstrip(".")


class MakerAgreements:
    """CRUD + prompt rendering for the ``maker_agreements`` table.

    Every method is user-scoped: pass ``user_id``. With an empty
    ``user_id`` the reads return nothing and the writes refuse — an
    agreement must belong to someone, and we never read across users
    (the #1 multi-tenant invariant).

    Takes a raw aiosqlite connection (the coder threads one through as
    ``_resolve_archive_conn()``); pass ``backend.conn`` from a
    ``SQLiteBackend`` elsewhere.
    """

    def __init__(self, conn) -> None:
        self._conn = conn

    # ── Capture ───────────────────────────────────────────────────────

    async def add(
        self,
        *,
        principle: str,
        rationale: str = "",
        category: str = "general",
        source: str = "explicit",
        strength: float = 1.0,
        user_id: str = "",
    ) -> Agreement | None:
        """Add an agreement, or reinforce an existing near-identical one.

        Restating a principle the user already holds bumps ``times_seen``
        and firms ``strength`` rather than inserting a duplicate —
        recurrence is signal, not noise. Returns the row (new or
        reinforced), or ``None`` when ``user_id`` is empty (refused).
        """
        principle = (principle or "").strip()[:_MAX_PRINCIPLE_CHARS]
        if not principle:
            raise ValueError("MakerAgreements.add requires a principle")
        if not user_id:
            log.warning("maker_agreement_add_refused_no_user")
            return None
        rationale = (rationale or "").strip()[:_MAX_RATIONALE_CHARS]
        category = category if category in CATEGORIES else "general"

        existing = await self._fetch_by_norm(_norm(principle), user_id=user_id)
        if existing is not None:
            return await self._reinforce(existing.id, user_id=user_id)

        cursor = await self._conn.execute(
            "INSERT INTO maker_agreements "
            "(user_id, principle, rationale, category, source, strength) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, principle, rationale, category, source, _clamp01(strength)),
        )
        await self._conn.commit()
        row_id = cursor.lastrowid
        await cursor.close()
        if not row_id:
            raise RuntimeError("maker_agreement_add_failed: no row_id")
        log.info("maker_agreement_added", user_id=user_id, agreement_id=int(row_id),
                 category=category, source=source)
        got = await self._fetch_by_id(int(row_id), user_id=user_id)
        assert got is not None
        return got

    async def _reinforce(self, agreement_id: int, *, user_id: str) -> Agreement | None:
        """Firm an agreement on restatement (bounded EWMA toward 1.0)."""
        current = await self._fetch_by_id(agreement_id, user_id=user_id)
        if current is None:
            return None
        new_strength = _clamp01(current.strength + 0.15 * (1.0 - current.strength))
        await self._conn.execute(
            "UPDATE maker_agreements "
            "SET strength = ?, times_seen = times_seen + 1, "
            "    status = 'active', updated_at = datetime('now') "
            "WHERE id = ? AND user_id = ?",
            (new_strength, agreement_id, user_id),
        )
        await self._conn.commit()
        return await self._fetch_by_id(agreement_id, user_id=user_id)

    async def retire(self, agreement_id: int, *, user_id: str = "") -> bool:
        """Soft-delete (status='retired'). User-scoped; returns True on hit."""
        if not user_id:
            return False
        cursor = await self._conn.execute(
            "UPDATE maker_agreements SET status = 'retired', "
            "updated_at = datetime('now') WHERE id = ? AND user_id = ?",
            (agreement_id, user_id),
        )
        await self._conn.commit()
        changed = cursor.rowcount
        await cursor.close()
        return bool(changed)

    # ── Read ──────────────────────────────────────────────────────────

    async def list_active(
        self, *, user_id: str = "", limit: int = 50,
    ) -> list[Agreement]:
        """Active agreements for one user, strongest first. Empty user → []."""
        if not user_id:
            return []
        async with self._conn.execute(
            "SELECT id, user_id, principle, rationale, category, source, "
            "       strength, times_seen, status "
            "FROM maker_agreements "
            "WHERE user_id = ? AND status = 'active' "
            "ORDER BY strength DESC, updated_at DESC LIMIT ?",
            (user_id, max(1, int(limit))),
        ) as cursor:
            rows = await cursor.fetchall()
        return [_row_to_agreement(r) for r in rows]

    async def render_for_prompt(
        self, *, user_id: str = "", limit: int = 24,
    ) -> str:
        """Render active agreements as a compact, honored prompt block.

        Empty when the user has no agreements (the common first-run case)
        — so injecting this is a no-op until the relationship has actually
        accrued something. Grouped by category for readability; rationale
        is appended in parentheses only when short enough to stay terse.
        """
        agreements = await self.list_active(user_id=user_id, limit=limit)
        if not agreements:
            return ""
        lines: list[str] = []
        for cat in CATEGORIES:
            group = [a for a in agreements if a.category == cat]
            for a in group:
                line = f"- {a.principle.rstrip('.')}."
                if a.rationale and len(a.rationale) <= 120:
                    line += f" ({a.rationale.rstrip('.')})"
                lines.append(line)
        body = "\n".join(lines)
        return (
            "<working_agreements>\n"
            "How this person wants you to work — their standing preferences, "
            "accrued over time. Honor them unless they conflict with an "
            "explicit instruction in this turn (a request decides WHETHER; "
            "these shape HOW):\n"
            f"{body}\n"
            "</working_agreements>"
        )

    # ── Internals ─────────────────────────────────────────────────────

    async def _fetch_by_id(self, agreement_id: int, *, user_id: str) -> Agreement | None:
        async with self._conn.execute(
            "SELECT id, user_id, principle, rationale, category, source, "
            "       strength, times_seen, status "
            "FROM maker_agreements WHERE id = ? AND user_id = ?",
            (agreement_id, user_id),
        ) as cursor:
            row = await cursor.fetchone()
        return _row_to_agreement(row) if row else None

    async def _fetch_by_norm(self, norm_principle: str, *, user_id: str) -> Agreement | None:
        """Find an existing agreement whose normalised principle matches —
        the dedup-reinforce key. Compares in Python (small per-user set)
        so the normalisation rule lives in exactly one place."""
        for a in await self.list_active(user_id=user_id, limit=500):
            if _norm(a.principle) == norm_principle:
                return a
        return None


def _row_to_agreement(row) -> Agreement:
    return Agreement(
        id=int(row[0]),
        user_id=row[1],
        principle=row[2] or "",
        rationale=row[3] or "",
        category=row[4] or "general",
        source=row[5] or "explicit",
        strength=float(row[6]) if row[6] is not None else 1.0,
        times_seen=int(row[7]) if row[7] is not None else 1,
        status=row[8] or "active",
    )


def _clamp01(value: float) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 1.0
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v
