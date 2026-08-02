"""Render foundation types.

Neutral primitives describing render work — what kind of output, for
which destination, with what payload. No coupling to fabric, no
coupling to a specific renderer, no I/O. The router director (when
fabric is enabled) consumes RenderJob to pick a node; the render
pipeline (future) consumes the same to dispatch the actual work.

Render kinds map onto ``CastRenderCapability`` boolean flags via
``capability_flag_for``. Adding a new kind is two lines: a constant
here and a flag on the capability dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── Render kinds ──────────────────────────────────────────────────


RENDER_HTML = "html"      # HTML/SVG → image; needs can_render_html
RENDER_VRM = "vrm"        # 3D VRM avatar frame / animation; needs can_render_vrm
RENDER_VIDEO = "video"    # Hardware video encode pass; needs can_encode_video
RENDER_WEBRTC = "webrtc"  # Live peer→TV stream setup; needs can_stream_webrtc


_KIND_TO_FLAG: dict[str, str] = {
    RENDER_HTML: "can_render_html",
    RENDER_VRM: "can_render_vrm",
    RENDER_VIDEO: "can_encode_video",
    RENDER_WEBRTC: "can_stream_webrtc",
}


# Render tiers — coarse signal from CastRenderCapability.tier. The
# router uses these to prefer higher-tier nodes for heavier jobs.
TIER_LITE = "lite"
TIER_STANDARD = "standard"
TIER_HEAVY = "heavy"

_TIER_RANK: dict[str, int] = {
    TIER_LITE: 0,
    TIER_STANDARD: 1,
    TIER_HEAVY: 2,
}


def capability_flag_for(kind: str) -> str:
    """Return the ``CastRenderCapability`` field name a kind requires.

    Empty string when the kind is unknown. Callers compare against the
    capability dataclass via ``getattr(cap, flag, False)`` so an
    unknown kind reads as 'no node can serve.'
    """
    return _KIND_TO_FLAG.get(kind, "")


def tier_rank(tier: str) -> int:
    """Coarse ordering for tier comparison. Unknown tiers rank lowest."""
    return _TIER_RANK.get(tier, -1)


# ── Job + route ───────────────────────────────────────────────────


@dataclass(frozen=True)
class RenderJob:
    """A pending render job — what to render, for which destination.

    ``kind`` is one of the ``RENDER_*`` constants. ``target_device_id``
    is the saved device (TV / receiver) the rendered output is bound
    for; the router uses it for future proximity scoring (prefer a
    peer on the same LAN segment as the TV). ``payload`` is opaque
    to the router; the renderer consumes it.

    Frozen so jobs are hashable + safe to pass across tasks.
    """

    kind: str
    target_device_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RenderRoute:
    """Where a render job will run.

    ``location`` is ``"local"`` or ``"peer"``. ``node_id`` is this
    node's identity for local routes or the chosen peer's node_id
    for delegated routes. ``tier`` carries the chosen node's tier so
    the renderer can make quality-of-service decisions (e.g. drop
    resolution on a lite-tier node).

    Returning ``None`` from the router means *nobody can serve this
    job* — distinct from ``RenderRoute(location="local")`` which
    means *local can serve, run it here*.
    """

    location: str
    node_id: str
    tier: str


@dataclass(frozen=True)
class RenderResult:
    """Outcome of a render dispatch.

    Successful renders carry an ``output_url`` the consumer (TV /
    receiver app / caller) can fetch. Failed renders carry a ``code``
    + ``message`` pair for diagnostics. ``location`` + ``node_id``
    let the caller report where the work actually ran — useful for
    diagnostic surfaces showing "rendered locally" vs "rendered on
    peer X."

    Frozen + dataclass so results round-trip through fabric envelopes
    without bespoke serializers.
    """

    ok: bool
    location: str = ""       # "local" | "peer" | "" on early failure
    node_id: str = ""
    output_url: str = ""
    code: str = ""           # error code; e.g. "no_capable_node"
    message: str = ""        # human-readable detail
    metadata: dict[str, Any] = field(default_factory=dict)
