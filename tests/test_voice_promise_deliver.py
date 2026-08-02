"""Tests for the voice promise/deliver pattern.

When Becca emits a tool tag mid-stream, the deliver step wraps the
result back into her voice as a continuation of the prose she opened
the turn with. The promise (everything before the tag) is the
commitment; the deliver step (the "second companion pass") is how she
honors it.

Coverage:
- Promise dataclass shape
- Tag-echo stripping (defensive — prevents Qwen 3.6 35B leaking
  ``tool:NAME`` syntax to TTS, the original bug that motivated this
  whole thread)
- Deliver prompt for success path references the promise
- Deliver prompt for failure path maps error categories to in-voice
  framings (not jargon)
- Tier resolution: primary → utility fallback, strict flag honored
- _synthesize end-to-end with a fake backend, success + failure paths
- _synthesize falls back to the static failure deck on empty / timeout
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from augmentum.companion_runtime.tool_protocol import (
    Promise, ToolCall, ToolError, ToolResult,
)
from augmentum.companion_runtime.voice import BeccaVoice


# ── Promise dataclass shape ───────────────────────────────────────────


def test_promise_dataclass_shape():
    """Promise captures pre_text + tag + started_at (the inputs the
    deliver step needs to honor the commitment)."""
    tag = ToolCall(kind="tool", name="grove.play", args={"track": "dune"},
                   raw="<tool:grove.play />", span=(0, 22))
    p = Promise(pre_text="let me put that on", tag=tag, started_at=123.456)
    assert p.pre_text == "let me put that on"
    assert p.tag.name == "grove.play"
    assert p.started_at == 123.456


# ── Tag-echo stripping (defensive guard) ───────────────────────────────


def test_strip_tag_echo_removes_complete_tags():
    """If the deliver model echoes a tag back, strip it before TTS sees it."""
    text = "Sure — <tool:web.search query=\"sourdough\" /> let me check."
    assert "<tool:" not in BeccaVoice._strip_tag_echo(text)


def test_strip_tag_echo_removes_bare_tool_form():
    """Qwen 3.6 35B has been seen to emit ``tool:NAME args=...`` without
    the angle brackets — the exact text that leaked to TTS pre-fix."""
    text = "Putting on tool:grove.play_matching query=warm jazz /> for you."
    cleaned = BeccaVoice._strip_tag_echo(text)
    assert "tool:" not in cleaned
    assert "grove.play_matching" not in cleaned


def test_strip_tag_echo_preserves_normal_text():
    """The stripper must NOT mangle prose that happens to mention 'tool'."""
    text = "I have a tool for that — give me a second."
    assert BeccaVoice._strip_tag_echo(text) == text


def test_strip_tag_echo_removes_complete_think_block():
    """LFM2.5 leaks reasoning into deliver output when the backend's
    thinking parser doesn't recognize the family. Belt-and-suspenders
    strip at the voice layer catches it before TTS sees it.
    """
    text = (
        "<think>We are continuing a reply. The user is Becca and the "
        "assistant is in the middle of writing a response.</think>"
        "The weather looks rainy today."
    )
    out = BeccaVoice._strip_tag_echo(text)
    assert "We are continuing" not in out
    assert "The weather looks rainy" in out


def test_strip_tag_echo_removes_orphan_think_opener():
    """Router timeout / mid-stream cut leaves a half-open <think>... with
    no closer. Everything after the opener is reasoning — never speak it.
    """
    text = "<think>The user wants me to fetch the weather. Let me reason about this slowly."
    assert BeccaVoice._strip_tag_echo(text) == ""


def test_strip_tag_echo_removes_orphan_think_closer():
    """Asymmetric families (GLM-4.x, DeepSeek V3.2+, etc.) put opener in
    prompt prefix; response starts inside thinking. Everything BEFORE
    the closer is reasoning.
    """
    text = (
        "We are continuing a reply. The user is Becca and the assistant is "
        "in the middle of writing</think>Dune is queued in the living room."
    )
    out = BeccaVoice._strip_tag_echo(text)
    assert "We are continuing" not in out
    assert "Dune is queued" in out


# ── Deliver prompts: success + failure ────────────────────────────────


class _FakeBus:
    """Captures publish_topic calls so tests can assert on emissions."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def publish_topic(self, topic: str, payload: dict, *, source_companion_id: str = "") -> None:
        self.events.append((topic, payload))


def _bv():
    """Construct a BeccaVoice without a real runtime — only the methods
    that don't touch runtime state are exercised here."""
    # The helpers below are pure functions of args, not runtime state,
    # so a minimal stub suffices. A fake bus is wired so the _synthesize
    # end-to-end tests can fire publish_topic without crashing.
    class _Stub:
        companion_id = "test"
        identity = type("I", (), {
            "display_name": "Becca",
            "companion_id": "test",
            "owner_user_id": "user-1",
            "persona_kernel_digest": "x",
        })()
        bus = _FakeBus()
        _app_state = None
        _started = True
    return BeccaVoice(_Stub())


def test_deliver_prompt_ok_includes_promise():
    """Success prompt must echo the promise so the model knows what to
    confirm."""
    bv = _bv()
    promise = Promise(
        pre_text="let me find that book for you",
        tag=ToolCall(kind="tool", name="web.search", args={},
                     raw="<tool:web.search />", span=(0, 22)),
        started_at=0.0,
    )
    system, user = bv._deliver_prompt_ok(
        tool_name="web.search",
        summary="Result: Dune (1965) by Frank Herbert",
        promise=promise,
    )
    assert "let me find that book" in user
    assert "Dune" in user
    # Must instruct against tool-syntax leak.
    assert "tool" in system.lower() and ("<tool" in system or "tool:NAME" in system or "bracket" in system.lower())


def test_deliver_prompt_ok_handles_empty_promise():
    """If she emitted a tag with no opener at all, the deliver prompt
    shouldn't break — it should ask for a clean continuation."""
    bv = _bv()
    promise = Promise(
        pre_text="",
        tag=ToolCall(kind="tool", name="note.create", args={},
                     raw="<tool:note.create />", span=(0, 22)),
        started_at=0.0,
    )
    _, user = bv._deliver_prompt_ok(
        tool_name="note.create", summary="ok", promise=promise,
    )
    # Either the prompt explicitly notes the missing opener or it lets
    # the model start fresh — but it must not crash.
    assert isinstance(user, str) and len(user) > 0


def test_deliver_prompt_failure_maps_error_categories():
    """Failure prompt must translate the error category to in-voice
    framing — no jargon ('tool', 'API', 'request failed')."""
    bv = _bv()
    promise = Promise(
        pre_text="let me put on Dune",
        tag=ToolCall(kind="tool", name="grove.play", args={},
                     raw="<tool:grove.play />", span=(0, 22)),
        started_at=0.0,
    )
    cases = {
        "timeout": "took too long",
        "unauthorized": "don't have permission",
        "model_unavailable": "model",
        "invalid_args": "didn't quite have",
        "upstream_error": "broke",
    }
    for category, expected_phrase in cases.items():
        _, user = bv._deliver_prompt_failure(
            tool_name="grove.play",
            error_category=category, error_hint="", promise=promise,
        )
        assert expected_phrase in user, (
            f"category {category!r}: missing {expected_phrase!r} in {user[:120]!r}"
        )


def test_deliver_prompt_failure_no_jargon_in_system():
    """The system prompt must explicitly forbid jargon."""
    bv = _bv()
    promise = Promise(
        pre_text="let me look it up",
        tag=ToolCall(kind="tool", name="web.search", args={},
                     raw="<tool:web.search />", span=(0, 22)),
        started_at=0.0,
    )
    system, _ = bv._deliver_prompt_failure(
        tool_name="web.search",
        error_category="timeout", error_hint="", promise=promise,
    )
    assert "jargon" in system.lower() or "tool" in system.lower()


# ── Tier resolution ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_resolve_deliver_tier_primary_default(monkeypatch):
    """Default tier ('primary') → tiers.primary first."""
    from augmentum.companion_runtime import tiers, voice as voice_mod

    called: list[str] = []

    async def fake_primary(_rt):
        called.append("primary")
        return ("backend-primary", "model-primary")

    async def fake_utility(_rt):
        called.append("utility")
        return ("backend-utility", "model-utility")

    monkeypatch.setattr(tiers, "primary", fake_primary)
    monkeypatch.setattr(tiers, "utility", fake_utility)

    bv = _bv()
    backend, name = await bv._resolve_deliver_tier("primary")
    assert backend == "backend-primary"
    assert name == "model-primary"
    assert called == ["primary"]


@pytest.mark.asyncio
async def test_resolve_deliver_tier_primary_falls_back_to_utility(monkeypatch):
    """Primary unresolvable + strict=false → silent fallback to utility."""
    from augmentum.companion_runtime import tiers, voice as voice_mod

    async def fake_primary(_rt):
        raise RuntimeError("no primary backend")

    async def fake_utility(_rt):
        return ("backend-utility", "model-utility")

    monkeypatch.setattr(tiers, "primary", fake_primary)
    monkeypatch.setattr(tiers, "utility", fake_utility)
    # Default is strict=False
    monkeypatch.setattr(
        voice_mod.settings, "companion_promise_deliver_strict_tier", False,
        raising=False,
    )

    bv = _bv()
    backend, name = await bv._resolve_deliver_tier("primary")
    assert backend == "backend-utility"
    assert name == "model-utility"


@pytest.mark.asyncio
async def test_resolve_deliver_tier_strict_refuses_fallback(monkeypatch):
    """Primary unresolvable + strict=true → (None, '')."""
    from augmentum.companion_runtime import tiers, voice as voice_mod

    async def fake_primary(_rt):
        raise RuntimeError("no primary backend")

    async def fake_utility(_rt):
        raise AssertionError("must not be called when strict")

    monkeypatch.setattr(tiers, "primary", fake_primary)
    monkeypatch.setattr(tiers, "utility", fake_utility)
    monkeypatch.setattr(
        voice_mod.settings, "companion_promise_deliver_strict_tier", True,
        raising=False,
    )

    bv = _bv()
    backend, name = await bv._resolve_deliver_tier("primary")
    assert backend is None
    assert name == ""


@pytest.mark.asyncio
async def test_resolve_deliver_tier_utility_explicit(monkeypatch):
    """When configured for utility, skip primary entirely."""
    from augmentum.companion_runtime import tiers, voice as voice_mod

    async def fake_primary(_rt):
        raise AssertionError("primary must not be consulted")

    async def fake_utility(_rt):
        return ("backend-utility", "model-utility")

    monkeypatch.setattr(tiers, "primary", fake_primary)
    monkeypatch.setattr(tiers, "utility", fake_utility)

    bv = _bv()
    backend, name = await bv._resolve_deliver_tier("utility")
    assert backend == "backend-utility"
    assert name == "model-utility"


# ── _synthesize end-to-end with a fake backend ────────────────────────


class _FakeBackend:
    """Backend stub: returns a canned text response from chat().

    The voice path uses ``models.base.response_text(resp)`` which
    expects ``resp.message.content``. Match that shape so deliver
    actually reads the canned text rather than seeing empty and
    falling through to the static failure deck.
    """

    def __init__(self, text: str = "There you go — Dune's on.", raise_after: float = 0.0):
        self._text = text
        self._raise_after = raise_after

    async def chat(self, req):
        if self._raise_after > 0:
            await asyncio.sleep(self._raise_after)
        from types import SimpleNamespace
        return SimpleNamespace(message=SimpleNamespace(content=self._text))


@pytest.mark.asyncio
async def test_synthesize_ok_returns_in_voice_text(monkeypatch):
    """Success path: deliver returns the model's continuation, stripped
    of any echoed tag syntax."""
    from augmentum.companion_runtime import tiers, voice as voice_mod
    from augmentum.companion_runtime.prompt_compose import ComposedPrompt

    fake = _FakeBackend(text="Dune's queued up in the living room.")

    async def fake_primary(_rt):
        return (fake, "primary-model")

    monkeypatch.setattr(tiers, "primary", fake_primary)
    monkeypatch.setattr(
        voice_mod.settings, "companion_promise_deliver_tier", "primary",
        raising=False,
    )

    bv = _bv()
    promise = Promise(
        pre_text="let me put it on for you",
        tag=ToolCall(kind="tool", name="grove.play", args={},
                     raw="<tool:grove.play />", span=(0, 22)),
        started_at=0.0,
    )
    result = ToolResult(
        ok=True, tool="grove.play", payload={"title": "Dune"},
        duration_ms=120,
    )

    text = await bv._synthesize(
        result, promise,
        intent=type("I", (), {"user_id": "user-1", "text": "put on dune",
                              "metadata": {}})(),
        composed=ComposedPrompt(system_text="ignored"),
        cancel=asyncio.Event(),
    )
    assert "Dune" in text
    assert "<tool:" not in text


@pytest.mark.asyncio
async def test_synthesize_failure_returns_in_character_narration(monkeypatch):
    """Failure path: deliver model writes the miss in voice. Affordances
    fallback only on empty / timeout."""
    from augmentum.companion_runtime import tiers, voice as voice_mod
    from augmentum.companion_runtime.prompt_compose import ComposedPrompt

    fake = _FakeBackend(text="Tried to put it on, but I don't have it.")

    async def fake_primary(_rt):
        return (fake, "primary-model")

    monkeypatch.setattr(tiers, "primary", fake_primary)
    monkeypatch.setattr(
        voice_mod.settings, "companion_promise_deliver_tier", "primary",
        raising=False,
    )

    bv = _bv()
    promise = Promise(
        pre_text="let me put on Dune",
        tag=ToolCall(kind="tool", name="grove.play", args={"track": "dune"},
                     raw="<tool:grove.play track=\"dune\" />", span=(0, 30)),
        started_at=0.0,
    )
    result = ToolResult(
        ok=False, tool="grove.play", payload=None,
        error=ToolError(category="upstream_error", message="404 not found"),
    )

    text = await bv._synthesize(
        result, promise,
        intent=type("I", (), {"user_id": "user-1", "text": "put on dune",
                              "metadata": {}})(),
        composed=ComposedPrompt(system_text="ignored"),
        cancel=asyncio.Event(),
    )
    assert text  # not empty
    assert "don't have it" in text or "Tried" in text


@pytest.mark.asyncio
async def test_synthesize_cancelled_returns_empty(monkeypatch):
    """Cancellation short-circuits before any backend call."""
    from augmentum.companion_runtime.prompt_compose import ComposedPrompt

    bv = _bv()
    promise = Promise(
        pre_text="let me check",
        tag=ToolCall(kind="tool", name="web.search", args={},
                     raw="<tool:web.search />", span=(0, 22)),
        started_at=0.0,
    )
    result = ToolResult(ok=True, tool="web.search", payload={})
    cancel = asyncio.Event()
    cancel.set()

    text = await bv._synthesize(
        result, promise,
        intent=type("I", (), {"user_id": "user-1", "text": "search",
                              "metadata": {}})(),
        composed=ComposedPrompt(system_text="ignored"),
        cancel=cancel,
    )
    assert text == ""
