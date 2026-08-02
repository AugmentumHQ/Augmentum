"""Frecency scoring with dual half-life decay for discovery clusters."""
from __future__ import annotations

import math
from datetime import UTC, datetime

from augmentum.discovery.clustering import SIGNAL_WEIGHT_MAP
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SHORT_HALF_LIFE_DAYS: float = 7
LONG_HALF_LIFE_DAYS: float = 30
SHORT_WEIGHT: float = 0.6
LONG_WEIGHT: float = 0.4
DAMPEN_MULTIPLIER: float = 0.3


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def compute_decay(hours_ago: float, *, half_life_days: float) -> float:
    """Exponential decay: 0.5 ^ (hours_ago / (half_life_days * 24))."""
    return 0.5 ** (hours_ago / (half_life_days * 24))


def compute_combined_frecency(
    short: float,
    long: float,
    *,
    dampened: bool = False,
) -> float:
    """Combine short and long frecency scores.

    short * 0.6 + long * 0.4, multiplied by 0.3 if dampened.
    """
    score = short * SHORT_WEIGHT + long * LONG_WEIGHT
    if dampened:
        score *= DAMPEN_MULTIPLIER
    return score


def _saturation_factor(n: int) -> float:
    """Diminishing-returns multiplier for N signals from one source.

    Curve: ln(1 + n) / n. Asymptotically the per-source total saturates
    at ~ln(n+1) instead of growing linearly with n. Concretely:

    ===  =======   ===============================
     n   factor    effective total (factor * n)
    ===  =======   ===============================
     1   1.00      1.00  (special case, full weight)
     2   0.549     1.10
     5   0.358     1.79
    10   0.240     2.40
    100  0.046     4.62
    ===  =======   ===============================

    Without this, binge-watching 100 videos from one channel produces
    100× the frecency contribution of one focused visit, which crowds
    out diverse signals and biases interest_clusters toward the loudest
    source. Saturation closes the gap so a 100-item binge is roughly
    comparable to a 5-item focused session, not 20× larger.
    """
    if n <= 1:
        return 1.0
    return math.log(1 + n) / n


def compute_frecency_from_signals(
    signals: list[dict],
    *,
    now: datetime | None = None,
) -> tuple[float, float]:
    """Compute (short, long) frecency from a list of signal dicts.

    Each signal needs ``created_at`` (ISO string) and ``signal_type``.
    Optional ``source_domain`` enables per-domain saturation discounting
    (see :func:`_saturation_factor`) — signals without a domain fall into
    a single shared bucket which still saturates as a group.

    Returns (frecency_short, frecency_long).
    """
    if now is None:
        now = datetime.now(UTC)

    # Group decay-weighted contributions by source domain before summing
    # so we can apply per-source saturation. Without grouping, a binge-
    # watch of one channel would drown out diverse signals from many
    # sources because nothing in the math distinguishes "1 strong
    # signal" from "100 weak signals from the same place".
    by_domain: dict[str, list[tuple[float, float]]] = {}

    for s in signals:
        created_at_str = s.get("created_at", "")
        if not created_at_str:
            continue

        try:
            created = datetime.fromisoformat(created_at_str)
            if created.tzinfo is None:
                created = created.replace(tzinfo=UTC)
        except (ValueError, TypeError):
            continue

        hours_ago = max(0.0, (now - created).total_seconds() / 3600)
        sig_type = s.get("signal_type", "")
        weight = SIGNAL_WEIGHT_MAP.get(sig_type, s.get("weight", 1.0))

        short_part = weight * compute_decay(hours_ago, half_life_days=SHORT_HALF_LIFE_DAYS)
        long_part = weight * compute_decay(hours_ago, half_life_days=LONG_HALF_LIFE_DAYS)
        domain = s.get("source_domain", "") or ""
        by_domain.setdefault(domain, []).append((short_part, long_part))

    short_total = 0.0
    long_total = 0.0
    for contribs in by_domain.values():
        factor = _saturation_factor(len(contribs))
        for s_part, l_part in contribs:
            short_total += s_part * factor
            long_total += l_part * factor

    return short_total, long_total
