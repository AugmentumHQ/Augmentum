"""Pin the Studio → cast payload shape against SurfaceSendRequest.

The Studio cast button assembles a request body whose ``state`` dict carries
``cast_source='studio'`` + the artifact preview URL. The /api/cast/send
endpoint accepts ``state`` as ``dict[str, Any]``, so the contract is loose
today, but a future tightening (TypedDict, stricter Pydantic model) could
silently break the Studio path. This smoke locks the shape.
"""
from __future__ import annotations

from augmentum.cast.receiver_protocol import SLOT_MAIN, SURFACE_HTML
from augmentum.proxy.cast_routes import SurfaceSendRequest


def test_studio_cast_payload_accepted():
    """Mirrors what ui/scripts/studio/cast.js castArtifactPreview() sends
    when invoked from the Studio cast button."""
    artifact_id = "art_abc123"
    body = SurfaceSendRequest(
        receiver_id="rcv_living_room",
        surface_kind=SURFACE_HTML,
        surface_url=f"/api/artifacts/{artifact_id}/preview",
        slot=SLOT_MAIN,
        state={
            "title": "Q4 Roadmap",
            "cast_source": "studio",
            "artifact_id": artifact_id,
            "cast_input_config": None,
            "cast_strategy": "shim",
        },
    )
    assert body.state["cast_source"] == "studio"
    assert body.state["artifact_id"] == artifact_id
    assert body.surface_url.endswith(f"/{artifact_id}/preview")
    assert body.surface_kind == SURFACE_HTML


def test_castgame_payload_still_accepted():
    """Pins the Library / games path's payload shape (cast_source='library'
    + input adapter chain). Studio and Games are independent cast flows;
    this guards against an accidental Pydantic tightening breaking either."""
    body = SurfaceSendRequest(
        receiver_id="rcv_living_room",
        surface_kind=SURFACE_HTML,
        surface_url="/ui/play/?title_id=abc&kiosk=1",
        slot=SLOT_MAIN,
        state={
            "title": "Sonic",
            "cast_source": "library",
            "artifact_id": "abc",
            "cast_input_config": {
                "adapters": ["gamepad_api", "keyboard"],
                "keymap": None,
            },
            "cast_strategy": "shim",
        },
    )
    assert body.state["cast_source"] == "library"
    assert body.state["cast_input_config"]["adapters"] == ["gamepad_api", "keyboard"]
