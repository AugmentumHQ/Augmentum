"""Live-vision wiring for the WS voice path (BeccaVoice).

Covers the seam that lets the always-listening companion SEE camera/image
frames carried on ``intent.metadata['images']``:

  - ``_apply_vision_to_intent`` reconciles the shared vision pipeline back
    onto the intent: a text-only primary's caption lands in ``intent.text``
    and the images are cleared; a VL primary keeps the frames for direct
    reading.
  - ``_call_primary`` / ``_consume_native_loop`` re-attach the surviving
    (VL) frames to the outbound user message so the model reads them.
  - No frames → every path is a no-op (never resolves a backend for vision).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from augmentum.companion_runtime import voice as voice_mod
from augmentum.companion_runtime.voice import BeccaVoice
from augmentum.models import base as base_mod


def _voice(app_state=None):
    return BeccaVoice(SimpleNamespace(
        bus=None, companion_id="becca", _app_state=app_state,
    ))


class _StreamBackend:
    """Records the request it was handed; yields nothing."""

    def __init__(self):
        self.seen_req = None

    async def chat_stream(self, req):
        self.seen_req = req
        return
        yield  # make this an async generator


# ── _apply_vision_to_intent ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_apply_vision_noop_without_frames(monkeypatch):
    """No images on the intent → no backend resolution, no mutation."""
    called = {"primary": False}

    async def _primary(_runtime):
        called["primary"] = True
        return object(), "m"
    monkeypatch.setattr(voice_mod.tiers, "primary", _primary)

    intent = SimpleNamespace(text="hello", metadata={"images": []})
    await _voice()._apply_vision_to_intent(intent)

    assert intent.text == "hello"
    assert called["primary"] is False


@pytest.mark.asyncio
async def test_apply_vision_text_only_inlines_caption(monkeypatch):
    """A text-only primary captions + strips frames: the caption is folded
    into intent.text and metadata images are cleared."""
    async def _primary(_runtime):
        return object(), "small-local"
    monkeypatch.setattr(voice_mod.tiers, "primary", _primary)

    async def _fake_pipeline(req, app_state, backend, **kwargs):
        # Mimic caption_via_router_fallback for a text-only primary.
        msg = req.messages[0]
        msg.content = f"[Image: a cat]\n\n{msg.content}"
        msg.images = None
    monkeypatch.setattr(base_mod, "apply_vision_pipeline", _fake_pipeline)

    intent = SimpleNamespace(
        text="what is this?", metadata={"images": ["data:image/jpeg;base64,AAA"]},
    )
    await _voice()._apply_vision_to_intent(intent)

    assert "a cat" in intent.text
    assert intent.metadata["images"] == []


@pytest.mark.asyncio
async def test_apply_vision_vl_keeps_frames(monkeypatch):
    """A VL primary leaves frames on the message → they survive on the
    intent for direct reading downstream."""
    async def _primary(_runtime):
        return object(), "qwen2.5-vl"
    monkeypatch.setattr(voice_mod.tiers, "primary", _primary)

    async def _fake_pipeline(req, app_state, backend, **kwargs):
        # VL primary: pipeline leaves content + images intact.
        return
    monkeypatch.setattr(base_mod, "apply_vision_pipeline", _fake_pipeline)

    frames = ["data:image/jpeg;base64,AAA"]
    intent = SimpleNamespace(text="what is this?", metadata={"images": list(frames)})
    await _voice()._apply_vision_to_intent(intent)

    assert intent.text == "what is this?"
    assert intent.metadata["images"] == frames


@pytest.mark.asyncio
async def test_apply_vision_never_raises(monkeypatch):
    """A pipeline blow-up degrades to a no-op, never breaking the turn."""
    async def _primary(_runtime):
        return object(), "m"
    monkeypatch.setattr(voice_mod.tiers, "primary", _primary)

    async def _boom(req, app_state, backend, **kwargs):
        raise RuntimeError("vision sidecar down")
    monkeypatch.setattr(base_mod, "apply_vision_pipeline", _boom)

    intent = SimpleNamespace(text="hi", metadata={"images": ["data:image/jpeg;base64,AAA"]})
    await _voice()._apply_vision_to_intent(intent)  # must not raise
    assert intent.text == "hi"


# ── frame re-attach on the outbound request ───────────────────────────


@pytest.mark.asyncio
async def test_call_primary_reattaches_vl_frames(monkeypatch):
    """Frames surviving on intent.metadata are re-attached to the user
    message handed to the backend (the VL-direct path)."""
    backend = _StreamBackend()

    async def _primary(_runtime):
        return backend, "qwen2.5-vl"
    monkeypatch.setattr(voice_mod.tiers, "primary", _primary)
    monkeypatch.setattr(BeccaVoice, "_attach_native_tools",
                        lambda self, req, b, m, intent: None)

    frames = ["data:image/jpeg;base64,AAA"]
    intent = SimpleNamespace(text="describe", user_id="u1", metadata={"images": frames})
    async for _ in _voice()._call_primary(
        "sys", intent, cancel=asyncio.Event(), invocation_id="iv",
    ):
        pass

    req = backend.seen_req
    assert req is not None
    user_msg = req.messages[1]
    assert user_msg.role == "user"
    assert user_msg.images == frames


@pytest.mark.asyncio
async def test_call_primary_unlocks_reasoning_on_vision(monkeypatch):
    """vision_reason in metadata flips req.think on for the response."""
    backend = _StreamBackend()

    async def _primary(_runtime):
        return backend, "qwen3.6"
    monkeypatch.setattr(voice_mod.tiers, "primary", _primary)
    monkeypatch.setattr(BeccaVoice, "_attach_native_tools",
                        lambda self, req, b, m, intent: None)

    intent = SimpleNamespace(
        text="what do you think?", user_id="u1",
        metadata={"images": [], "vision_reason": True},
    )
    async for _ in _voice()._call_primary(
        "sys", intent, cancel=asyncio.Event(), invocation_id="iv",
    ):
        pass
    assert backend.seen_req.think is True


@pytest.mark.asyncio
async def test_apply_vision_pipeline_reason_flag(monkeypatch):
    """apply_vision_pipeline(reason_on_vision=True) sets think only when
    the turn actually carried frames."""
    import augmentum.models.base as base

    async def _noop(*a, **k):
        return None
    monkeypatch.setattr(base, "resolve_chat_image_urls", _noop)
    monkeypatch.setattr(base, "caption_via_router_fallback", _noop)
    monkeypatch.setattr(base, "inject_vision_prompt", lambda req: None)

    with_img = base.InternalChatRequest(
        model="m",
        messages=[base.Message(role="user", content="hi", images=["data:image/png;base64,x"])],
    )
    await base.apply_vision_pipeline(with_img, object(), object(), reason_on_vision=True)
    assert with_img.think is True

    no_img = base.InternalChatRequest(
        model="m", messages=[base.Message(role="user", content="hi")],
    )
    await base.apply_vision_pipeline(no_img, object(), object(), reason_on_vision=True)
    assert no_img.think is False


@pytest.mark.asyncio
async def test_call_primary_no_images_sends_none(monkeypatch):
    """With no frames, the user message carries images=None (not [])."""
    backend = _StreamBackend()

    async def _primary(_runtime):
        return backend, "small-local"
    monkeypatch.setattr(voice_mod.tiers, "primary", _primary)
    monkeypatch.setattr(BeccaVoice, "_attach_native_tools",
                        lambda self, req, b, m, intent: None)

    intent = SimpleNamespace(text="hi", user_id="u1", metadata={})
    async for _ in _voice()._call_primary(
        "sys", intent, cancel=asyncio.Event(), invocation_id="iv",
    ):
        pass

    assert backend.seen_req.messages[1].images is None


def test_ensure_live_camera_framing_inserts_and_merges():
    """The reality anchor lands as a system message for the VL-direct path,
    merges into an existing system message, and is idempotent."""
    from augmentum.models.base import (
        LIVE_CAMERA_SYSTEM_NOTE,
        InternalChatRequest,
        Message,
        ensure_live_camera_framing,
    )
    # No system message → one is inserted at the front.
    r1 = InternalChatRequest(
        model="m", messages=[Message(role="user", content="what is this?")],
    )
    ensure_live_camera_framing(r1)
    assert r1.messages[0].role == "system"
    assert LIVE_CAMERA_SYSTEM_NOTE in r1.messages[0].content
    assert r1.messages[1].role == "user"

    # Existing system message → the anchor is prepended, original preserved.
    r2 = InternalChatRequest(model="m", messages=[
        Message(role="system", content="You are a helpful assistant."),
        Message(role="user", content="hi"),
    ])
    ensure_live_camera_framing(r2)
    assert LIVE_CAMERA_SYSTEM_NOTE in r2.messages[0].content
    assert "You are a helpful assistant." in r2.messages[0].content

    # Idempotent — a second call doesn't duplicate the note.
    ensure_live_camera_framing(r2)
    assert r2.messages[0].content.count("RIGHT NOW you can see") == 1
