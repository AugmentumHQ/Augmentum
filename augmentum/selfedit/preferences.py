"""The learning loop — turning the never-pruned archive into judgment.

Every human Keep/Revert on a self-edit teaches the system which CHANGE-SHAPES it
can trust. A shape earns trust the anti-Westworld way — by **accumulation across
real verdicts**, conservatively (one keep proves nothing) — the same earned-not-
given philosophy as the evidence/convergence work ([[earned-understanding]]).

Trust surfaces as a *judgment-tier* signal: a consistently-kept shape lifts a
"no-regression-but-intent-unconfirmed" verdict from ``human_required`` to
``probable`` (via the existing honest verifier router). It is NEVER lifted to
``verified`` — learned taste is judgment, never mechanical — and ``probable`` is
not auto-promotable, so the loop never silently ships taste. It makes the system
*wiser* (better signal, better advice), and a separate explicit autonomy setting
could later let high-trust shapes auto-promote.

Lives in the isolated, durable growth DB (can't affect the main app).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from augmentum.selfedit.verifier import Verifier, judgment_verifier
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Conservative bars: trust is earned, not given on one good outcome.
MIN_SAMPLES = 3          # a shape must be judged at least this many times to count
TRUST_THRESHOLD = 0.8    # kept-rate at/above which a shape is "trusted"

DECISION_KEEP = "keep"
DECISION_REVERT = "revert"


def change_shape(surface: str, intent_class: str = "") -> str:
    """The unit the loop learns over — a normalized ``surface:intent`` signature
    (e.g. ``config:adaptation``, ``frontend:style``, ``backend:bugfix``). Groups
    similar changes so a verdict on one informs the next of the same shape."""
    s = (surface or "system").strip().lower()
    i = (intent_class or "").strip().lower()
    return f"{s}:{i}" if i else s


def confidence(kept: int, reverted: int) -> float:
    total = kept + reverted
    return (kept / total) if total else 0.0


def is_trusted(kept: int, reverted: int) -> bool:
    return (kept + reverted) >= MIN_SAMPLES and confidence(kept, reverted) >= TRUST_THRESHOLD


@dataclass
class PreferenceStat:
    shape: str
    kept: int
    reverted: int

    @property
    def samples(self) -> int:
        return self.kept + self.reverted

    @property
    def confidence(self) -> float:
        return confidence(self.kept, self.reverted)

    @property
    def trusted(self) -> bool:
        return is_trusted(self.kept, self.reverted)

    def to_dict(self) -> dict:
        return {
            "shape": self.shape, "kept": self.kept, "reverted": self.reverted,
            "samples": self.samples, "confidence": round(self.confidence, 3),
            "trusted": self.trusted,
        }


class PreferenceStore:
    """Keep/revert tallies per change-shape, on the isolated growth connection."""

    def __init__(self, conn: Any) -> None:
        self.conn = conn

    async def record(self, *, user_id: str, shape: str, kept: bool) -> None:
        """Tally one human verdict. Upsert (kept/reverted incremented in place)."""
        col = "kept" if kept else "reverted"
        await self.conn.execute(
            f"""
            INSERT INTO self_edit_preferences (user_id, shape, {col}, last_decision, updated_at)
            VALUES (?, ?, 1, ?, datetime('now'))
            ON CONFLICT(user_id, shape) DO UPDATE SET
                {col} = {col} + 1, last_decision = excluded.last_decision,
                updated_at = datetime('now')
            """,
            (user_id, shape, DECISION_KEEP if kept else DECISION_REVERT),
        )
        await self.conn.commit()
        log.info("selfedit_preference_recorded", shape=shape, kept=kept)

    async def stat(self, *, user_id: str, shape: str) -> PreferenceStat:
        cur = await self.conn.execute(
            "SELECT kept, reverted FROM self_edit_preferences WHERE user_id=? AND shape=?",
            (user_id, shape),
        )
        row = await cur.fetchone()
        return PreferenceStat(shape, int(row[0]), int(row[1])) if row else PreferenceStat(shape, 0, 0)

    async def summary(self, *, user_id: str) -> list[PreferenceStat]:
        cur = await self.conn.execute(
            "SELECT shape, kept, reverted FROM self_edit_preferences WHERE user_id=? "
            "ORDER BY (kept + reverted) DESC, kept DESC",
            (user_id,),
        )
        return [PreferenceStat(r[0], int(r[1]), int(r[2])) for r in await cur.fetchall()]


# --- the lift: a learned-trust judgment oracle ----------------------------

def preference_verifier(shape: str, *, store: PreferenceStore, user_id: str,
                        cost: int = 3) -> Verifier:
    """A judgment Verifier that confirms intent ONLY when ``shape`` is trusted —
    so a consistently-kept shape lifts ``human_required`` → ``probable``. An
    untrusted/insufficient shape SKIPs (no effect). Confidence is the kept-rate;
    the router's floor (≥0.7) still gates it, and it can never reach ``verified``."""
    async def _judge(_ctx: dict) -> tuple[bool, float, str]:
        st = await store.stat(user_id=user_id, shape=shape)
        if not st.trusted:
            # not enough earned evidence → skip (judgment_verifier turns a raise
            # into SKIP, so an untrusted shape leaves the verdict untouched).
            raise ValueError(f"shape '{shape}' not yet trusted ({st.kept}/{st.samples})")
        return True, st.confidence, f"learned: you keep '{shape}' ({st.kept}/{st.samples} kept)"

    return judgment_verifier("preference", _judge, cost=cost, required=False)
