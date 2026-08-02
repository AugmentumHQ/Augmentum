"""Delivery-class seam — artifact verbs stay quiet on the voice path.

Pins the 2026-06-10 co-author register fix: note writes must not get
latency affordances or synthesize confirmations in TTS. The sticky
note updating on screen is the feedback; the conversation stays about
the content.
"""
from __future__ import annotations

import pytest

# Importing the builtins package registers every verb; the architect
# primitives register on their own package import.
import augmentum.architect.primitives  # noqa: F401
import augmentum.intent  # noqa: F401
from augmentum.intent.registry import REGISTRY, register_action


def _get_action(action_id: str):
    for action in REGISTRY.all():
        if action.id == action_id:
            return action
    raise AssertionError(f"action {action_id} not registered")


# ── Verb declarations ─────────────────────────────────────────────────

@pytest.mark.parametrize("verb", [
    # Notes — the sticky updating is the feedback.
    "note.create",
    "note.append",
    "note.show_sticky",
    "note.start_capture",
    "note.end_capture",
    # Authored complete ack, no artifact needed.
    "memory.save",
    # Conversation control — instant, speak lines authored.
    "control.stop",
    "control.repeat",
    "control.slower",
    "control.louder",
    "control.goodbye",
    "control.nevermind",
    # Surface actions — the surface opening/changing is the feedback.
    "navigate.open_surface",
    "navigate.back",
    "chat.new",
    "chat.history",
    "companion.take_me_there",
    "discovery.show",
    "browse.find",
    "files.find",
    "web.search",
    "companion.today_recap",
    # Playback — transport state change is the feedback (pause/next/
    # previous deliberately speak nothing at all).
    "media.play",
    # media.search retired 2026-06-11 — duplicated search.local
    # byte-for-byte (same files.search_open emit); see builtin/media.py.
    "media.pause",
    "media.next",
    "media.previous",
    "media.resume",
    "grove.play_matching",
    "time.set_timer",
])
def test_surface_verbs_are_artifact_delivery(verb):
    assert _get_action(verb).delivery == "artifact"


@pytest.mark.parametrize("verb", [
    # Data verbs — results are material she composes from.
    "memory.recall",
    "search.local",
    "search.knowledge",
    # search.web retired 2026-06-11 — open-web screen search is
    # web.search, which is artifact-class by design (panel IS the
    # feedback), so it doesn't belong in this verbal list.
    # Slow generation — the latency affordance ("Hold on. Sketching.")
    # is earning its keep; keep the verbal wrap.
    "image.generate_with_defaults",
])
def test_data_verbs_stay_verbal(verb):
    assert _get_action(verb).delivery == "verbal"


def test_register_action_rejects_unknown_delivery():
    async def _noop(text, session, args):
        return None

    with pytest.raises(ValueError, match="delivery"):
        register_action(
            id="test.bad_delivery",
            summary="x",
            examples=["x"],
            handler=_noop,
            delivery="silent",  # not a valid class
        )


# ── Voice-side lookup ─────────────────────────────────────────────────

def test_delivery_for_tool_lookup():
    from augmentum.companion_runtime.tools import delivery_for_tool

    assert delivery_for_tool("note.append") == "artifact"
    assert delivery_for_tool("note.create") == "artifact"
    # Non-registry capability tools and unknown names stay verbal —
    # they're the slow gather tools the affordance deck exists for.
    assert delivery_for_tool("web_search") == "verbal"
    assert delivery_for_tool("definitely.not.a.tool") == "verbal"


# ── Deliver gate ──────────────────────────────────────────────────────

class _Synth:
    """Capture whether the synthesize pass ran."""

    def __init__(self):
        self.called = False

    async def _synthesize(self, result, promise, intent, composed, cancel):
        self.called = True
        return "SYNTHESIZED CONFIRMATION"


@pytest.mark.asyncio
async def test_deliver_result_artifact_ok_speaks_verbatim():
    from augmentum.companion_runtime.tool_protocol import ToolResult
    from augmentum.companion_runtime.voice import BeccaVoice

    synth = _Synth()
    deliver = BeccaVoice._deliver_result

    result = ToolResult(
        ok=True, tool="note.create", payload={"content": "Sure, here you go."},
        metadata={"delivery": "artifact", "speak": "Sure, here you go."},
    )
    out = await deliver(synth, result, None, None, None, None)
    assert out == "Sure, here you go. "
    assert not synth.called


@pytest.mark.asyncio
async def test_deliver_result_artifact_silent_append():
    from augmentum.companion_runtime.tool_protocol import ToolResult
    from augmentum.companion_runtime.voice import BeccaVoice

    synth = _Synth()
    result = ToolResult(
        ok=True, tool="note.append", payload={"content": "Done."},
        metadata={"delivery": "artifact", "speak": ""},
    )
    out = await BeccaVoice._deliver_result(
        synth, result, None, None, None, None,
    )
    assert out == ""
    assert not synth.called


@pytest.mark.asyncio
async def test_deliver_result_verbal_falls_through_to_synthesize():
    from augmentum.companion_runtime.tool_protocol import ToolResult
    from augmentum.companion_runtime.voice import BeccaVoice

    synth = _Synth()
    result = ToolResult(ok=True, tool="web_search", payload={"content": "x"})
    out = await BeccaVoice._deliver_result(
        synth, result, None, None, None, None,
    )
    assert out == "SYNTHESIZED CONFIRMATION"
    assert synth.called


@pytest.mark.asyncio
async def test_deliver_result_artifact_failure_still_synthesizes():
    from augmentum.companion_runtime.tool_protocol import ToolResult
    from augmentum.companion_runtime.voice import BeccaVoice

    synth = _Synth()
    result = ToolResult(
        ok=False, tool="note.append", payload=None,
        metadata={"delivery": "artifact", "speak": ""},
    )
    out = await BeccaVoice._deliver_result(
        synth, result, None, None, None, None,
    )
    # She still owns the miss in voice.
    assert out == "SYNTHESIZED CONFIRMATION"
    assert synth.called


# ── note.attach_image (image → note worksurface) ──────────────────────

class _FakeNotesStore:
    def __init__(self, notes):
        self.notes = notes
        self.updates = []

    async def get(self, note_id, *, user_id=""):
        return self.notes.get(note_id)

    async def update(self, note_id, fields, *, user_id=""):
        self.updates.append((note_id, fields))
        self.notes[note_id].update(fields)
        return self.notes[note_id]


_NOTE_TEST_SEQ = [0]


def _note_session(note_id="n1"):
    from types import SimpleNamespace
    from augmentum.intent.action import SessionContext
    # Unique user/session per call — the referent cache is keyed
    # globally, so reusing ids bleeds active_note_id across tests.
    _NOTE_TEST_SEQ[0] += 1
    uid = f"u_att{_NOTE_TEST_SEQ[0]}"
    sid = f"s_att{_NOTE_TEST_SEQ[0]}"
    store = _FakeNotesStore({
        "n1": {"id": "n1", "title": "Trip ideas", "content": "- Kyoto"},
    })
    app_state = SimpleNamespace(notes_store=store)
    session = SessionContext(
        user_id=uid, session_id=sid, mode=None, app_state=app_state,
    )
    from augmentum.intent.dispatch import get_referent_cache
    refs = get_referent_cache(app_state, uid, sid)
    if note_id:
        refs.active_note_id = note_id
    session.referents = refs
    return session, store


@pytest.mark.asyncio
async def test_attach_image_appends_markdown_and_emits():
    action = _get_action("note.attach_image")
    assert action.delivery == "artifact"
    session, store = _note_session()
    result = await action.handler(
        "", session, {"url": "/api/image/abc123", "caption": "temple sketch"},
    )
    assert result.surface_emit["channel"] == "note.update_sticky"
    content = store.notes["n1"]["content"]
    assert content.endswith("![temple sketch](/api/image/abc123)\n")
    assert content.startswith("- Kyoto\n")
    assert result.surface_emit["payload"]["title"] == "Trip ideas"
    assert not result.speak  # silent — the thumbnail IS the feedback


@pytest.mark.asyncio
async def test_attach_image_rejects_unsafe_url():
    action = _get_action("note.attach_image")
    session, store = _note_session()
    result = await action.handler(
        "", session, {"url": "javascript:alert(1)"},
    )
    assert result.surface_emit is None
    assert store.updates == []


@pytest.mark.asyncio
async def test_attach_image_sanitizes_markdown_breakers():
    action = _get_action("note.attach_image")
    session, store = _note_session()
    await action.handler(
        "", session,
        {"url": "https://x.test/a(1).png", "caption": "br[ack]ets"},
    )
    content = store.notes["n1"]["content"]
    assert "![brackets](https://x.test/a%281%29.png)" in content


@pytest.mark.asyncio
async def test_attach_image_no_active_note_honest_miss():
    action = _get_action("note.attach_image")
    session, _ = _note_session(note_id=None)
    result = await action.handler("", session, {"url": "/api/image/x"})
    assert "open note" in result.speak
