"""Load balancer: virtual model routing across multiple backends."""

from __future__ import annotations

import contextvars
import random
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.state.balancer_store import BalancerConfig, BalancerMember

log = get_logger(__name__)

# Strategies
ROUND_ROBIN = "round_robin"
RANDOM = "random"
WEIGHTED_RANDOM = "weighted_random"
LEAST_RECENTLY_USED = "least_recently_used"
AB_TEST = "ab_test"

STRATEGIES = (ROUND_ROBIN, RANDOM, WEIGHTED_RANDOM, LEAST_RECENTLY_USED, AB_TEST)

LB_PREFIX = "lb/"


@dataclass
class BalancerResolution:
    """Context from a balancer resolution, available downstream for
    fallback retry and A/B metadata injection."""

    balancer_id: str
    balancer_name: str
    strategy: str
    member_id: int
    model_used: str
    backend_key: str
    fallback_enabled: bool
    fallback_members: list[BalancerMember]


# Per-request context carrying the resolution info
_balancer_ctx: contextvars.ContextVar[BalancerResolution | None] = contextvars.ContextVar(
    "balancer_ctx", default=None
)


def set_balancer_context(ctx: BalancerResolution | None) -> None:
    _balancer_ctx.set(ctx)


def get_balancer_context() -> BalancerResolution | None:
    return _balancer_ctx.get(None)


# Per-member cooldown (seconds). After a retryable failure a member is skipped
# by selection until it cools, so round-robin/random naturally rotate AROUND an
# exhausted free-tier key instead of re-hitting it every request. Used only when
# the provider gives no Retry-After/retryDelay hint; a parsed hint always wins.
_DEFAULT_COOLDOWN_S = 30.0
_MAX_COOLDOWN_S = 1800.0  # 30 min ceiling on the exponential blind backoff


class LoadBalancer:
    """Selects a member from a pool by strategy, skipping members in cooldown.

    Cooldown makes fallback *efficient*: a rate-limited member is benched until
    it recovers, so an N-key pool spreads load across the healthy members and
    stops burning requests on a key that's already returning 429.
    """

    def __init__(self, config: BalancerConfig, members: list[BalancerMember]) -> None:
        self.config = config
        self.members = [m for m in members if m.enabled]
        self._rr_index = 0
        # member.id -> monotonic cooldown deadline; member.id -> failure streak
        self._cooling: dict[int, float] = {}
        self._fail_streak: dict[int, int] = {}

    # ---- cooldown state -------------------------------------------------

    def is_cooling(self, member: BalancerMember) -> bool:
        return self._cooling.get(member.id, 0.0) > time.monotonic()

    def cooldown_remaining(self, member: BalancerMember) -> float:
        return max(0.0, self._cooling.get(member.id, 0.0) - time.monotonic())

    def note_failure(
        self, member: BalancerMember, retry_after_s: float | None = None
    ) -> None:
        """Cool ``member`` after a retryable failure. Honors the provider's
        Retry-After/retryDelay when supplied; else applies exponential blind
        backoff keyed on the member's consecutive-failure streak."""
        streak = self._fail_streak.get(member.id, 0) + 1
        self._fail_streak[member.id] = streak
        cool = (
            float(retry_after_s)
            if retry_after_s and retry_after_s > 0
            else min(_DEFAULT_COOLDOWN_S * (2 ** (streak - 1)), _MAX_COOLDOWN_S)
        )
        self._cooling[member.id] = time.monotonic() + cool

    def note_success(self, member: BalancerMember) -> None:
        """Clear a member's failure streak + cooldown once it serves content."""
        self._fail_streak.pop(member.id, None)
        self._cooling.pop(member.id, None)

    def _active_members(self) -> list[BalancerMember]:
        """Members not currently cooling; falls back to ALL when every member
        is cooling (better to attempt — one may have recovered — than hard-fail)."""
        now = time.monotonic()
        active = [m for m in self.members if self._cooling.get(m.id, 0.0) <= now]
        return active if active else self.members

    # ---- selection ------------------------------------------------------

    def select(self) -> BalancerMember:
        pool = self._active_members()
        if not pool:
            raise ValueError(f"Balancer '{self.config.name}' has no enabled members")

        strategy = self.config.strategy
        if strategy == ROUND_ROBIN:
            return self._round_robin(pool)
        if strategy in (RANDOM, AB_TEST):
            return self._random(pool)
        if strategy == WEIGHTED_RANDOM:
            return self._weighted_random(pool)
        if strategy == LEAST_RECENTLY_USED:
            return self._lru(pool)
        return self._round_robin(pool)

    def fallback_order(self, failed_member: BalancerMember) -> list[BalancerMember]:
        """After ``failed_member``: healthy members first (by priority), then
        cooling members (soonest-to-recover first) as a last resort — so the
        facade still attempts them if the healthy set is empty."""
        now = time.monotonic()
        remaining = [m for m in self.members if m.id != failed_member.id]
        healthy = sorted(
            (m for m in remaining if self._cooling.get(m.id, 0.0) <= now),
            key=lambda m: m.priority,
        )
        cooling = sorted(
            (m for m in remaining if self._cooling.get(m.id, 0.0) > now),
            key=lambda m: self._cooling[m.id],
        )
        return healthy + cooling

    def reload_members(self, members: list[BalancerMember]) -> None:
        self.members = [m for m in members if m.enabled]

    def _round_robin(self, pool: list[BalancerMember]) -> BalancerMember:
        member = pool[self._rr_index % len(pool)]
        self._rr_index += 1
        return member

    def _random(self, pool: list[BalancerMember]) -> BalancerMember:
        return random.choice(pool)

    def _weighted_random(self, pool: list[BalancerMember]) -> BalancerMember:
        weights = [m.weight for m in pool]
        return random.choices(pool, weights=weights, k=1)[0]

    def _lru(self, pool: list[BalancerMember]) -> BalancerMember:
        # last_used_at is ISO 8601 (e.g. "2026-03-21T12:00:00") — sorts
        # correctly as a string since ISO 8601 is lexicographically ordered.
        # Empty/None → empty string sorts before any timestamp (never-used first).
        return min(pool, key=lambda m: m.last_used_at or "")


class LoadBalancerRegistry:
    """In-memory cache of load balancers."""

    def __init__(self) -> None:
        self._balancers: dict[str, LoadBalancer] = {}
        self._name_map: dict[str, str] = {}  # lowercase name -> balancer id

    def register(self, balancer_id: str, lb: LoadBalancer) -> None:
        self._balancers[balancer_id] = lb
        self._name_map[lb.config.name.lower()] = balancer_id

    def unregister(self, balancer_id: str) -> None:
        lb = self._balancers.pop(balancer_id, None)
        if lb:
            self._name_map.pop(lb.config.name.lower(), None)

    def get(self, balancer_id: str) -> LoadBalancer | None:
        return self._balancers.get(balancer_id)

    def get_by_name(self, name: str) -> LoadBalancer | None:
        bid = self._name_map.get(name.lower())
        return self._balancers.get(bid) if bid else None

    def is_balancer_model(self, model_name: str) -> bool:
        return model_name.startswith(LB_PREFIX)

    def all_balancers(self) -> list[LoadBalancer]:
        return list(self._balancers.values())
