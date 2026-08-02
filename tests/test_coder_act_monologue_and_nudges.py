"""Tests for act-phase monologue strip + creation-verb thresholds + fallback suggestions.

Covers three improvements from the 2026-04-21 real-world transcript
audit (RetroArch docker-build task that looped in inspection):

1. ``_strip_act_monologue`` removes "Let me check...", "I'll look...",
   "The user wants..." tells that weak models emit as per-iteration
   prose. Empty substantive content → user sees nothing rather than
   confusion-looking narration.

2. Creation-verb goals (build / run / make / deploy / ...) tighten
   the inspection-loop thresholds so the break fires at ~iter 5
   instead of iter 8. Non-creation goals keep the looser defaults.

3. ``_render_fallback_summary`` emits a termination-reason-specific
   "Suggested next step" paragraph instead of the generic "try a
   stronger model" fallback line.
"""
from __future__ import annotations

import pytest

from augmentum.modes.coder.handler import CoderHandler
from augmentum.modes.coder.phase_act import _strip_act_monologue
from augmentum.models.base import InternalStreamChunk, Message

from tests.test_coder_handler import (
    _ExtendedContainerManager,
    _FakeBackend,
    _FakeChunk,
    _FakeTool,
    _force_native_tier,
    _make_request,
    _tc_delta,
)


# ---------------------------------------------------------------------------
# Monologue stripper
# ---------------------------------------------------------------------------


def test_strip_removes_let_me_tells():
    s = "Let me check if the Dockerfile exists. Here is the answer."
    out = _strip_act_monologue(s)
    assert "Let me check" not in out
    assert "Here is the answer" in out


def test_strip_removes_ill_tells():
    s = "I'll look at the setup script now. The build uses nginx."
    out = _strip_act_monologue(s)
    assert "I'll look" not in out
    assert "nginx" in out


def test_strip_removes_user_wants_tells():
    s = "The user wants to build Docker. First, the Dockerfile sets up Debian."
    out = _strip_act_monologue(s)
    assert "user wants" not in out
    assert "Debian" in out


def test_strip_removes_stuck_in_loop_narration():
    """2026-04-22 Pong transcript: the model narrated "I'm stuck in a
    loop" / "I can't see tool results" / "I realize I need to try
    again" between every iteration. Zero substantive content."""
    s = (
        "I'm stuck in a loop where tool results aren't being displayed. "
        "I see the issue. I realize I need to try a different approach. "
        "The file exists and compiled successfully."
    )
    out = _strip_act_monologue(s)
    assert "stuck in a loop" not in out
    assert "I see the issue" not in out
    assert "I realize I" not in out
    # Substantive claim survives
    assert "file exists and compiled" in out


def test_strip_removes_i_notice_tells():
    s = "I notice I keep getting the same result. Python 3.12 is installed."
    out = _strip_act_monologue(s)
    assert "I notice" not in out
    assert "I keep getting" not in out
    assert "Python 3.12 is installed" in out


def test_strip_removes_having_trouble_narration():
    s = (
        "I'm having trouble reading the output. "
        "I'm getting repeated reminders. "
        "The tkinter module is now installed."
    )
    out = _strip_act_monologue(s)
    assert "having trouble" not in out
    assert "I'm getting" not in out
    assert "tkinter module is now installed" in out


def test_strip_preserves_claim_of_action():
    """Claims of completed action ("I wrote X", "I installed Y") are
    NOT stripped — they contain substantive content the user wants to
    know. The hallucinated-claim issue is a separate bug to address;
    this stripper is for pure process talk."""
    s = "I wrote /workspace/pong.py. I installed tkinter via apt-get."
    out = _strip_act_monologue(s)
    assert "wrote" in out
    assert "installed" in out


def test_strip_preserves_substantive_content():
    # A real answer with no monologue tells should come through intact.
    s = (
        "The Dockerfile uses a two-stage build. Stage 1 installs "
        "nginx and downloads the RetroArch release. Stage 2 copies "
        "only the needed assets into a minimal runtime image."
    )
    out = _strip_act_monologue(s)
    assert out == s.strip()


def test_strip_empty_returns_empty():
    assert _strip_act_monologue("") == ""
    assert _strip_act_monologue("   \n  \n") == ""


def test_strip_all_monologue_returns_empty():
    s = (
        "Let me check the Dockerfile. I'll look at the assets. "
        "Let me see what's in the workspace."
    )
    out = _strip_act_monologue(s)
    # Every sentence was a tell → result is empty
    assert out == ""


def test_strip_keeps_code_blocks_fine():
    # Strip is regex-based on sentences; fenced code (already
    # removed by earlier pipeline stage) wouldn't appear here, but
    # for safety make sure multi-line structured content doesn't get
    # mangled.
    s = (
        "The Dockerfile shows:\n"
        "- Debian bullseye base\n"
        "- nginx + Python3\n"
        "Run with: docker build -t x ."
    )
    out = _strip_act_monologue(s)
    assert "Debian bullseye" in out
    assert "docker build -t x" in out


# ---------------------------------------------------------------------------
# Multi-tell block detector — real 2026-04-22 transcript
# ---------------------------------------------------------------------------


def test_multi_tell_block_resumes_at_discourse_marker():
    """Exact leaky prose from the basic-http-server networking turn.
    The per-sentence strip would miss this because the monologue is
    interleaved with non-tell sentences ("The answer is…", "They'd
    need to…"). The multi-tell detector (≥2 tells in first 500 chars)
    must nuke everything up to "Good question" — where the real
    response begins."""
    s = (
        "The user is asking whether they can access the server from "
        "their Windows machine if the server is running inside a "
        "Docker container on `127.0.0.1`. The answer is: no, because "
        "`127.0.0.1` (localhost) only exposes it to the container "
        "itself. They'd need to use `--public` flag which binds to "
        "`0.0.0.0`. Let me check how this container is set up. I can "
        "just explain the situation and offer to restart.\n\n"
        "Good question — right now it's bound to 127.0.0.1, which "
        "only works inside the container. Your Windows machine can't "
        "reach it."
    )
    out = _strip_act_monologue(s)
    # Preamble should be gone
    assert "The user is asking" not in out
    assert "Let me check how" not in out
    # The real answer must survive intact
    assert "Good question" in out
    assert "Your Windows machine can't reach it" in out


def test_multi_tell_block_with_no_marker_returns_empty():
    """If the whole response is monologue ABOUT the user with no
    user-facing turn marker, drop it all — the fallback summary will
    handle."""
    s = (
        "The user is asking me to do something. I need to figure out "
        "what. Let me think about it. I should check the files. Let "
        "me verify the setup. I realize I don't have context."
    )
    out = _strip_act_monologue(s)
    assert out == ""


def test_multi_tell_block_requires_user_reference_opener():
    """The block detector is scoped to "The user is X" openers. A
    slab of "Let me / I need / I'm stuck" tells WITHOUT the user-
    reference must fall through to per-sentence strip (preserving
    trailing substantive claims). Guards the "The tkinter module is
    installed" regression pattern."""
    s = (
        "Let me check the file. I need to verify the setup. "
        "I'm having trouble with the build. The tkinter module is "
        "now installed."
    )
    out = _strip_act_monologue(s)
    # Block detector should NOT have blanked this — no "The user" opener
    assert "tkinter module is now installed" in out
    # But per-sentence strip should still remove the tells
    assert "Let me check" not in out


def test_multi_tell_block_single_tell_untouched():
    """A single "Let me check" alone must NOT trigger the block
    detector — normal per-sentence strip handles it."""
    s = (
        "Let me check the file. The build completed successfully in "
        "2.3 seconds and produced /workspace/app."
    )
    out = _strip_act_monologue(s)
    assert "Let me check" not in out
    assert "build completed successfully" in out
    # Should NOT have been blanked out wholesale
    assert len(out) > 40


# ---------------------------------------------------------------------------
# Creation-verb tightened thresholds
# ---------------------------------------------------------------------------


def _make_inspection_handler(backend, monkeypatch):
    _force_native_tier(monkeypatch)
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [_FakeTool("file_read", output="ok")],
    )
    return CoderHandler(
        backend, session_id="sess-insp",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-insp",
    )


class _InspectsForever:
    """Emits a file_read every iteration, never writes, never stops."""

    def __init__(self):
        self.calls = 0

    async def chat_stream(self, request):
        self.calls += 1
        yield _FakeChunk(augmentum={"tool_calls": [
            _tc_delta(
                0, f"tc-{self.calls}", "file_read",
                {"path": f"/workspace/step{self.calls}.py"},
            ),
        ]})
        yield _FakeChunk(done=True, finish_reason="tool_calls")

    async def chat(self, request):
        return None


@pytest.mark.asyncio
async def test_creation_verb_breaks_earlier_than_defaults(monkeypatch):
    """A task with 'build' in it should trigger inspection_loop_break
    at iter 5 (1 grace + 3 nudge + 2 break delta)."""
    backend = _InspectsForever()
    handler = _make_inspection_handler(backend, monkeypatch)

    break_iter: int | None = None
    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_hybrid(
        _make_request("build and run the project"),
        workspace_context="",
    ):
        chunks.append(c)
        if (
            c.augmentum
            and c.augmentum.get("status") == "inspection_loop_break"
            and break_iter is None
        ):
            break_iter = backend.calls

    assert break_iter is not None, "Expected inspection_loop_break to fire"
    # Tightened thresholds: break at iter 3+2=5. Allow 5..6 for
    # the cold-start grace edge case (streak starts counting from
    # iter 2 or iter 3 depending on whether the grace check is
    # strictly > or >=).
    assert break_iter <= 6, (
        f"With creation verb, break should fire by iter 6; got {break_iter}"
    )


@pytest.mark.asyncio
async def test_non_creation_verb_uses_looser_thresholds(monkeypatch):
    """A non-creation task ('understand the codebase') should NOT
    trigger inspection_loop_break early — the detector requires a
    creation verb match to fire at all."""
    backend = _InspectsForever()
    handler = _make_inspection_handler(backend, monkeypatch)

    # Bound iterations so the test doesn't run to _HYBRID_MAX_ITERS.
    # The detector is gated on creation_verb so no break ever fires;
    # we care that 'understand' does NOT trigger tightened thresholds.
    monkeypatch.setattr(
        "augmentum.modes.coder.handler._HYBRID_OBSERVATION_EVERY", 100,
    )

    chunks: list[InternalStreamChunk] = []
    # Cap externally — we just want to confirm no inspection-loop
    # break fires in the first 6 iters for a non-creation goal.
    async def _stop_after(n):
        i = 0
        async for c in handler._act_hybrid(
            _make_request("understand the codebase structure"),
            workspace_context="",
        ):
            chunks.append(c)
            i += 1
            if i > n:
                break

    # We can't easily stop an async generator externally, so rely on
    # `_InspectsForever` behavior + the fact that break fires only on
    # creation verbs. Alternative: use a bounded backend.
    class _InspectsFew(_InspectsForever):
        async def chat_stream(self, request):
            self.calls += 1
            if self.calls <= 6:
                async for c in super().chat_stream(request):
                    # super() increments calls again — avoid
                    # double-counting by delegating differently
                    yield c
            else:
                yield _FakeChunk(done=True, finish_reason="stop")

    # Simpler: use a backend that always returns a stop after N iters
    # so the loop terminates naturally and we count break events.
    class _CountedBackend:
        def __init__(self, max_calls=6):
            self.calls = 0
            self.max = max_calls

        async def chat_stream(self, request):
            self.calls += 1
            if self.calls > self.max:
                yield _FakeChunk(done=True, finish_reason="stop")
                return
            yield _FakeChunk(augmentum={"tool_calls": [
                _tc_delta(
                    0, f"tc-{self.calls}", "file_read",
                    {"path": f"/workspace/s{self.calls}.py"},
                ),
            ]})
            yield _FakeChunk(done=True, finish_reason="tool_calls")

        async def chat(self, request):
            return None

    backend2 = _CountedBackend(max_calls=6)
    handler2 = _make_inspection_handler(backend2, monkeypatch)

    chunks2: list[InternalStreamChunk] = []
    async for c in handler2._act_hybrid(
        _make_request("understand the codebase structure"),
        workspace_context="",
    ):
        chunks2.append(c)

    break_chunks = [
        c for c in chunks2
        if c.augmentum and c.augmentum.get("status") == "inspection_loop_break"
    ]
    # No creation verb → no inspection_loop_break even after 6 iters
    assert break_chunks == [], (
        "Inspection-loop break should not fire on non-creation goals "
        "at these iter counts"
    )


# ---------------------------------------------------------------------------
# Fallback summary — termination-specific suggestions
# ---------------------------------------------------------------------------


def _render(handler, termination_reason: str, messages=None) -> str:
    return handler._render_fallback_summary(
        iteration=5,
        total_writes=0,
        termination_reason=termination_reason,
        same_file_edits={},
        messages=messages or [],
    )


def test_inspection_loop_break_suggests_explicit_command():
    h = CoderHandler(
        _FakeBackend([]), session_id="s",
        container_manager=None, workspace_id="w",
    )
    out = _render(h, "inspection_loop_break")
    assert "Suggested next step" in out
    assert "docker build" in out or "CREATE" in out or "explicit" in out


def test_same_file_edit_break_suggests_concrete_change():
    h = CoderHandler(
        _FakeBackend([]), session_id="s",
        container_manager=None, workspace_id="w",
    )
    out = _render(h, "same_file_edit_break")
    assert "Suggested next step" in out
    assert "thrashed" in out.lower() or "concrete" in out.lower() or (
        "start/end" in out.lower()
    )


def test_validation_error_streak_suggests_stronger_model():
    h = CoderHandler(
        _FakeBackend([]), session_id="s",
        container_manager=None, workspace_id="w",
    )
    out = _render(h, "validation_error_streak")
    assert "Suggested next step" in out
    # Suggestion mentions model capability as the root cause.
    assert "stronger model" in out.lower() or "model" in out.lower()


def test_test_failure_streak_suggests_fixture_investigation():
    h = CoderHandler(
        _FakeBackend([]), session_id="s",
        container_manager=None, workspace_id="w",
    )
    out = _render(h, "test_failure_streak")
    assert "Suggested next step" in out
    assert "test" in out.lower()


def test_unknown_termination_reason_uses_generic_fallback():
    h = CoderHandler(
        _FakeBackend([]), session_id="s",
        container_manager=None, workspace_id="w",
    )
    out = _render(h, "some_custom_reason_not_in_map")
    # No specific suggestion map entry → generic closing line
    assert "Suggested next step" not in out
    assert "didn't produce a narrative summary" in out
