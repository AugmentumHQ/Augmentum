"""Energy gate — energy shapes what she INITIATES, never how she responds.

Step 2 of the companion-kernel build order. When ``companion_energy_enabled``
is on, low energy damps OUTWARD (non-rest) autonomous activities so rest wins
and she recovers (the act -> deplete -> rest -> recover duty cycle). Rest-tagged
candidates are never damped (``activity_selector.choose``) and never cost energy
(``spend_energy``) — both keyed on ``drive == "rest"`` so they stay symmetric.

Responsiveness is untouched by any of this — that wall is
``tests/test_responsiveness_invariant.py``.
"""
from __future__ import annotations

from augmentum.companion_runtime.behavior.activity_selector import (
    _CANDIDATE_DRIVES,
    _energy_factor,
)

# The rest-tagged candidates are the recovery set: never damped by low energy
# and never spend energy. Encoded here to guard the symmetry below.
_REST_KINDS = {"no_op", "scene_update", "dream_invocation"}


def test_no_damping_at_or_above_baseline():
    # At rest she sits at baseline -> factor 1.0 -> identical to energy-off.
    assert _energy_factor(0.6, 0.6) == 1.0
    # Above baseline still clamps to 1.0 (no super-charging of initiation).
    assert _energy_factor(0.9, 0.6) == 1.0


def test_damps_below_baseline_monotonically():
    f = _energy_factor(0.3, 0.6)
    assert 0.15 <= f < 1.0
    # Lower level -> more damping.
    assert _energy_factor(0.2, 0.6) < _energy_factor(0.4, 0.6)


def test_floored_never_zero():
    # Even at the energy floor a strong-enough appetite can still act.
    assert _energy_factor(0.05, 0.6) >= 0.15
    assert _energy_factor(0.0, 0.6) >= 0.15


def test_safe_on_zero_baseline():
    assert _energy_factor(0.5, 0.0) == 1.0


def test_rest_kinds_are_exactly_the_recovery_set():
    # The exemption in choose() and the skip in spend_energy both key on
    # drive == "rest". Guard that the recovery set is exactly the rest-tagged
    # candidates, so damping and spending never drift out of symmetry.
    rest_tagged = {k for k, d in _CANDIDATE_DRIVES.items() if d == "rest"}
    assert rest_tagged == _REST_KINDS
