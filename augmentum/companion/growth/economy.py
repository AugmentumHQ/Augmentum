"""Economy — mana + berry bookkeeping for the companion growth loop.

Wraps a :class:`~augmentum.companion.growth.store.EconomyAccount` row and
the per-(user, agent) transaction log. The store is pure CRUD; this class
owns the business rules: lazy mana regen, debit/earn semantics, vouch /
veto / sponsor entry points, and write-through to the audit log so every
movement is reconstructable.

Two currencies (spec §4):

* **Mana** regenerates with wall-clock time (~10/hour, soft cap 100). Every
  action debits some. The regen is computed lazily on read so we don't
  need a background ticker — the elapsed-since-last-tick fraction is
  added when somebody asks.
* **Berries** are earned from positive user outcomes and bank indefinitely
  (slow decay is Phase 5 — not implemented here). Big-swing actions
  require berry stock; small actions do not.

Anti-anthropomorphisation reminder: these are accounting numbers, not
feelings. The vocabulary is metaphor; the UI should translate.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from augmentum.companion.growth.store import (
    DEFAULT_AGENT_ID,
    EconomyAccount,
    GrowthStore,
)
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    pass

log = get_logger(__name__)


# Per-account serialization. Every balance mutation is a read-modify-write
# (get_or_create → mutate → save), so two concurrent debits/earns for the
# same account would both read the old balance and the second save would
# clobber the first → double-spend / negative balances (audit 2026-06-17).
# A process-local lock per (user, agent) serializes them. Augmentum runs
# single-process, so this is sufficient; a multi-worker deployment would
# additionally need an atomic conditional UPDATE in the store.
_ACCOUNT_LOCKS: dict[tuple[str, str], asyncio.Lock] = {}


def _account_lock(user_id: str, agent_id: str) -> asyncio.Lock:
    key = (user_id, agent_id)
    lock = _ACCOUNT_LOCKS.get(key)
    if lock is None:
        lock = asyncio.Lock()
        _ACCOUNT_LOCKS[key] = lock
    return lock


# A debit that would drive mana negative is rejected. Callers that want
# "go ahead even if it goes negative" must explicitly opt in via the
# ``allow_negative`` flag (used for the restraint-credit path that
# rewards inaction).
_DEFAULT_ALLOW_NEGATIVE = False


@dataclass(slots=True)
class DebitResult:
    """Outcome of a mana debit attempt."""

    ok: bool
    mana_after: float
    debited: float
    reason: str = ""


@dataclass(slots=True)
class EarnResult:
    """Outcome of a berry earn / spend."""

    ok: bool
    berries_after: float
    delta: float
    reason: str = ""


class Economy:
    """Owns the (user, agent) account row and writes transactions through.

    Stateless wrt the connection — every method round-trips to the store
    so two Economy handles for the same account stay coherent (the store
    is the source of truth).
    """

    def __init__(
        self,
        store: GrowthStore,
        *,
        user_id: str,
        agent_id: str = DEFAULT_AGENT_ID,
    ) -> None:
        if not user_id:
            raise ValueError("Economy requires user_id")
        self._store = store
        self.user_id = user_id
        self.agent_id = agent_id

    # ── Reads ─────────────────────────────────────────────────────────

    async def snapshot(self) -> EconomyAccount:
        """Return the current account state with regen applied."""
        async with _account_lock(self.user_id, self.agent_id):
            account = await self._store.get_or_create_economy(
                user_id=self.user_id, agent_id=self.agent_id,
            )
            ticked = self._apply_regen(account)
            if ticked:
                await self._store.save_economy(account)
                await self._store.append_tx(
                    user_id=self.user_id, agent_id=self.agent_id,
                    tx_type="mana_regen", amount=ticked,
                    reason="lazy_regen_on_read",
                )
            return account

    # ── Mana ──────────────────────────────────────────────────────────

    async def debit_mana(
        self,
        amount: float,
        *,
        growth_log_id: str | None = None,
        reason: str = "",
        allow_negative: bool = _DEFAULT_ALLOW_NEGATIVE,
    ) -> DebitResult:
        if amount < 0:
            raise ValueError("debit_mana amount must be non-negative")
        async with _account_lock(self.user_id, self.agent_id):
            account = await self._store.get_or_create_economy(
                user_id=self.user_id, agent_id=self.agent_id,
            )
            ticked = self._apply_regen(account)
            if not allow_negative and account.mana < amount:
                # Persist the regen tick even though the debit is rejected.
                if ticked:
                    await self._store.save_economy(account)
                    await self._store.append_tx(
                        user_id=self.user_id, agent_id=self.agent_id,
                        tx_type="mana_regen", amount=ticked,
                        reason="lazy_regen_on_debit_reject",
                    )
                return DebitResult(
                    ok=False, mana_after=account.mana, debited=0.0,
                    reason="insufficient_mana",
                )
            account.mana -= amount
            await self._store.save_economy(account)
            if ticked:
                await self._store.append_tx(
                    user_id=self.user_id, agent_id=self.agent_id,
                    tx_type="mana_regen", amount=ticked,
                    reason="lazy_regen_on_debit",
                    growth_log_id=growth_log_id,
                )
            await self._store.append_tx(
                user_id=self.user_id, agent_id=self.agent_id,
                tx_type="mana_debit", amount=amount,
                reason=reason or "action_dispatch",
                growth_log_id=growth_log_id,
            )
            return DebitResult(ok=True, mana_after=account.mana, debited=amount)

    # ── Berries ───────────────────────────────────────────────────────

    async def earn_berries(
        self,
        amount: float,
        *,
        signal_kind: str = "system",
        growth_log_id: str | None = None,
        reason: str = "",
        evidence_ref: str = "",
    ) -> EarnResult:
        if amount < 0:
            raise ValueError(
                "earn_berries amount must be non-negative; "
                "use spend_berries for negative movements",
            )
        async with _account_lock(self.user_id, self.agent_id):
            account = await self._store.get_or_create_economy(
                user_id=self.user_id, agent_id=self.agent_id,
            )
            account.berries += amount
            account.berries_lifetime += amount
            await self._store.save_economy(account)
            await self._store.append_tx(
                user_id=self.user_id, agent_id=self.agent_id,
                tx_type="berry_earn", amount=amount,
                reason=reason or "user_outcome",
                signal_kind=signal_kind,
                evidence_ref=evidence_ref,
                growth_log_id=growth_log_id,
            )
            return EarnResult(
                ok=True, berries_after=account.berries, delta=amount,
            )

    async def spend_berries(
        self,
        amount: float,
        *,
        growth_log_id: str | None = None,
        reason: str = "",
        allow_negative: bool = _DEFAULT_ALLOW_NEGATIVE,
    ) -> EarnResult:
        if amount < 0:
            raise ValueError("spend_berries amount must be non-negative")
        async with _account_lock(self.user_id, self.agent_id):
            account = await self._store.get_or_create_economy(
                user_id=self.user_id, agent_id=self.agent_id,
            )
            if not allow_negative and account.berries < amount:
                return EarnResult(
                    ok=False, berries_after=account.berries, delta=0.0,
                    reason="insufficient_berries",
                )
            account.berries -= amount
            await self._store.save_economy(account)
            await self._store.append_tx(
                user_id=self.user_id, agent_id=self.agent_id,
                tx_type="berry_spend", amount=amount,
                reason=reason or "big_swing_unlock",
                growth_log_id=growth_log_id,
            )
            return EarnResult(
                ok=True, berries_after=account.berries, delta=-amount,
            )

    # ── User-initiated grants ─────────────────────────────────────────

    async def vouch(
        self, amount: float, *, reason: str = "", evidence_ref: str = "",
    ) -> EarnResult:
        """Explicit user grant — bypasses the reward-signal pipeline."""
        return await self.earn_berries(
            amount, signal_kind="user_action",
            reason=reason or "user_vouch", evidence_ref=evidence_ref,
        )

    async def veto(
        self, amount: float, *, reason: str = "", evidence_ref: str = "",
    ) -> EarnResult:
        """Explicit user reversal — subtracts berries even past zero.

        Used when the automated reward signal said positive but the user
        actively disagrees. This signal is also the calibration anchor
        for tuning the implicit-signal weights later.
        """
        if amount < 0:
            raise ValueError("veto amount must be non-negative")
        async with _account_lock(self.user_id, self.agent_id):
            account = await self._store.get_or_create_economy(
                user_id=self.user_id, agent_id=self.agent_id,
            )
            account.berries -= amount
            # Don't decrement lifetime — vetoes are evidence about the user,
            # not removal of trust that was earned.
            await self._store.save_economy(account)
            await self._store.append_tx(
                user_id=self.user_id, agent_id=self.agent_id,
                tx_type="veto", amount=amount,
                reason=reason or "user_veto",
                signal_kind="user_action",
                evidence_ref=evidence_ref,
            )
            return EarnResult(
                ok=True, berries_after=account.berries, delta=-amount,
            )

    async def sponsor(
        self, amount: float, *, reason: str = "", evidence_ref: str = "",
    ) -> EarnResult:
        """Pre-fund a specific big swing — user fronts the berry cost."""
        return await self.earn_berries(
            amount, signal_kind="user_action",
            reason=reason or "user_sponsor", evidence_ref=evidence_ref,
        )

    # ── Regen ─────────────────────────────────────────────────────────

    def _apply_regen(self, account: EconomyAccount) -> float:
        """Mutate ``account`` to include elapsed regen.

        Returns the amount of mana credited (>= 0). Updates
        ``last_mana_tick`` to now. Capped at ``mana_cap``.
        """
        now = int(time.time())
        elapsed_sec = max(0, now - account.last_mana_tick)
        account.last_mana_tick = now
        if elapsed_sec <= 0 or account.mana_regen_per_hour <= 0:
            return 0.0
        if account.mana >= account.mana_cap:
            return 0.0
        gained = (elapsed_sec / 3600.0) * account.mana_regen_per_hour
        before = account.mana
        account.mana = min(account.mana_cap, account.mana + gained)
        return max(0.0, account.mana - before)
