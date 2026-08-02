"""Proactive offer — surface a small dismissable prompt at a chosen moment.

Catalog category D (proactive offer). See
``docs/superpowers/specs/2026-05-31-companion-action-catalog.md`` §D.

What this does:

  1. Take ``ctx.target_ref`` as the offer kind (e.g. ``"music"``,
     ``"break"``, ``"saved_recipe"``, ``"draft_reply"``).
  2. Pull the offer label + payload + optional dismiss-after-seconds
     from ``ctx.extras`` (``offer_label``, ``offer_payload``,
     ``dismiss_after_seconds``).
  3. Emit a ``growth.offer.surfaced`` surface event. The runtime / UI
     is the responsible party for *displaying* the offer and for
     routing the user's accept/dismiss back through the reward signal
     channels (``apply_explicit`` with the appropriate signal kind).
  4. Return ledger_delta so the growth log carries an "offer pending"
     counter that the verifier can match against accept/dismiss
     events later.

The handler itself does not block on the user's response — that's a
runtime concern. The verifier in Phase 2 closes the loop by matching
``offer_surfaced`` ledger entries against subsequent
``offer_accepted`` / ``offer_dismissed`` reward signals.

**Why this is a load-bearing category for "proactive yet reliable":**
Every major competitor (ChatGPT / Claude / Copilot) is responsive-not-
proactive. Replika/Character.AI are proactive but engagement-maxxing.
Proactive offers are how Becca demonstrates proactivity without the
engagement-trap shape — small, dismissable, calibrated against the
saturation guard the catalog spec requires.

Cost / tier: mana 2.0, tier 0 (offer itself is reversible; what's
*offered* may be higher tier and goes through its own gating when
the user accepts).

Reward signal (Phase 5 wires this): user accepts = +10; explicit
thanks = +15; dismissed = -3 (mild — wrong-time offers are not
free, since the interrupt itself is the cost).
"""

from __future__ import annotations

import time

from augmentum.companion.growth.actions import (
    ActionContext,
    ActionResult,
    register,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# Default dismiss-after window. Long enough the user can notice; short
# enough that a stale offer doesn't clutter the surface. UI overrides
# via ``ctx.extras["dismiss_after_seconds"]``.
_DEFAULT_DISMISS_AFTER_SECONDS = 5 * 60  # 5 minutes

# Hard cap on payload size — prevents an over-eager call site from
# stuffing a whole document into the offer surface event. The offer is
# meant to be a hint with enough context for the user to decide; the
# accepted action carries the full payload.
_MAX_OFFER_PAYLOAD_BYTES = 4 * 1024


class SurfaceProactiveOffer:
    """D — Proactive offer.

    Handler is intentionally thin: the *decision* to offer (timing
    against state, saturation, calibration) lives in the caller —
    usually the runtime's activity selector or initiative queue. This
    handler is the surfacing primitive that fans out the chosen offer
    consistently.
    """

    action_type = "proactive_offer"
    mana_cost = 2.0
    tier = 0

    async def run(self, ctx: ActionContext) -> ActionResult:
        if not ctx.target_ref:
            return ActionResult(
                ok=False, error="proactive_offer: empty target_ref (offer_kind)",
            )

        offer_kind = ctx.target_ref.strip()
        offer_label = str(ctx.extras.get("offer_label") or "").strip()
        if not offer_label:
            return ActionResult(
                ok=False,
                error="proactive_offer: extras['offer_label'] is required",
            )

        offer_payload = ctx.extras.get("offer_payload")
        if offer_payload is not None:
            # Size guard — surface events ride the bus + UI websocket;
            # a multi-MB payload would break the rest of the channel.
            payload_size = len(str(offer_payload).encode("utf-8"))
            if payload_size > _MAX_OFFER_PAYLOAD_BYTES:
                return ActionResult(
                    ok=False,
                    error=(
                        f"proactive_offer: payload too large "
                        f"({payload_size} > {_MAX_OFFER_PAYLOAD_BYTES} bytes)"
                    ),
                )

        try:
            dismiss_after = int(
                ctx.extras.get("dismiss_after_seconds")
                or _DEFAULT_DISMISS_AFTER_SECONDS
            )
        except (TypeError, ValueError):
            dismiss_after = _DEFAULT_DISMISS_AFTER_SECONDS
        # Clamp to a sane range — no zero, no day-long offers.
        dismiss_after = max(10, min(dismiss_after, 60 * 60))

        offer_id = f"offer_{int(time.time() * 1000)}_{offer_kind}"
        surface_event = {
            "topic": "growth.offer.surfaced",
            "payload": {
                "offer_id": offer_id,
                "offer_kind": offer_kind,
                "offer_label": offer_label[:280],
                "offer_payload": offer_payload,
                "dismiss_after_seconds": dismiss_after,
                "growth_log_id": ctx.growth_log_id,
                "rationale": ctx.rationale,
                "surfaced_at": int(time.time()),
            },
        }
        return ActionResult(
            ok=True,
            payload={
                "offer_id": offer_id,
                "offer_kind": offer_kind,
                "dismiss_after_seconds": dismiss_after,
            },
            surface_event=surface_event,
            # Counters the verifier reads later: every offer_surfaced
            # should eventually match an offer_accepted or
            # offer_dismissed signal (or time-out as ignored).
            ledger_delta={
                "offer_surfaced": 1,
                f"offer_surfaced_{offer_kind}": 1,
            },
            continue_loop=False,
        )


register(SurfaceProactiveOffer())
