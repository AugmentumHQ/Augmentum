"""Model-initiated compaction: the ``compact`` tool + handler consumption.

Covers the 2026-07-09 compact-tool wiring:

1. ``CompactTool.execute`` — signal-flag semantics: sets
   ``state.compact_requested`` + a four-line note, enforces required
   fields, the pending-guard, and the per-turn use cap.
2. ``_compaction_plan(force=True)`` — bypasses the enable gate and the
   token threshold; structural minimums still apply.
3. ``_compact_messages_with_synthesis`` — a pending model request is
   consumed exactly once, the model's note lands as ``### Synthesis``
   (no second-model call), and the flag clears even when the fold is
   structurally impossible.
4. Sticky-reminder context meter — rendered only when
   ``coder_compact_tool_enabled`` is on.
"""
from __future__ import annotations

from augmentum.coder.state import CoderState
from augmentum.coder.tools import CompactTool
from tests.test_coder_compaction_lifecycle import (
    _base_messages,
    _big_middle,
    _request,
    _tail,
)
from tests.test_coder_context_preservation import (
    _make_handler_for_compact,
    _trip_compaction_thresholds,
)


def _state() -> CoderState:
    return CoderState(session_id="s1", workspace_id="w1")


def _tool(state: CoderState) -> CompactTool:
    return CompactTool(
        container_manager=None, workspace_id="w1", state=state,
    )


_FIELDS = {
    "state": "read handler.py, found the loop",
    "decisions": "signal-flag pattern because finish_task uses it",
    "learnings": "none",
    "next": "wire the force branch",
}


# ---------------------------------------------------------------------------
# CompactTool.execute
# ---------------------------------------------------------------------------
async def test_compact_tool_sets_flags_and_note():
    st = _state()
    result = await _tool(st).execute(**_FIELDS)
    assert result.success
    assert st.compact_requested is True
    assert st.compact_tool_uses == 1
    # Note is the exact four-line synthesis shape.
    lines = st.compact_note.splitlines()
    assert [ln.split(":")[0] for ln in lines] == [
        "State", "Decisions", "Learnings", "Next",
    ]
    assert "signal-flag pattern" in st.compact_note


async def test_compact_tool_rejects_empty_fields():
    st = _state()
    result = await _tool(st).execute(
        state="done", decisions="", learnings="none", next="",
    )
    assert not result.success
    assert result.validation_error
    assert "Decisions" in result.error and "Next" in result.error
    assert st.compact_requested is False
    assert st.compact_tool_uses == 0


async def test_compact_tool_pending_guard():
    st = _state()
    assert (await _tool(st).execute(**_FIELDS)).success
    second = await _tool(st).execute(**_FIELDS)
    assert not second.success
    assert "already pending" in second.error
    assert st.compact_tool_uses == 1  # guard fired before the counter


async def test_compact_tool_per_turn_cap():
    st = _state()
    tool = _tool(st)
    for _ in range(2):
        assert (await tool.execute(**_FIELDS)).success
        # Simulate the loop consuming the fold between calls.
        st.compact_requested = False
        st.compact_note = ""
    third = await tool.execute(**_FIELDS)
    assert not third.success
    assert "2 times this turn" in third.error
    assert st.compact_requested is False


# ---------------------------------------------------------------------------
# _compaction_plan force semantics
# ---------------------------------------------------------------------------
def test_force_plan_bypasses_threshold_and_gate(monkeypatch):
    _trip_compaction_thresholds(monkeypatch, keep_recent=2)
    h = _make_handler_for_compact()
    # Raise the effective limit so the normal path is under threshold,
    # and disable auto compaction entirely — force must ignore both.
    h._coder_compact_token_limit = 10_000_000
    monkeypatch.setattr(
        "augmentum.config.settings.coder_compaction_auto_enabled",
        False, raising=False,
    )
    messages = [*_base_messages(), *_big_middle(), *_tail()]
    plan, _before = h._compaction_plan(messages)
    assert plan is None
    plan, _before = h._compaction_plan(messages, force=True)
    assert plan is not None


def test_force_plan_still_needs_a_middle(monkeypatch):
    _trip_compaction_thresholds(monkeypatch, keep_recent=2)
    h = _make_handler_for_compact()
    plan, _before = h._compaction_plan(_base_messages(), force=True)
    assert plan is None


# ---------------------------------------------------------------------------
# Model-note consumption in _compact_messages_with_synthesis
# ---------------------------------------------------------------------------
async def test_model_note_becomes_synthesis_without_llm_call(monkeypatch):
    _trip_compaction_thresholds(monkeypatch, keep_recent=2)
    h = _make_handler_for_compact()

    async def _must_not_call(*args, **kwargs):
        raise AssertionError("model-note path must not call the second model")

    import augmentum.coder.compaction_synthesis as cs
    monkeypatch.setattr(cs, "synthesize_compaction_segment", _must_not_call)

    h._state.compact_requested = True
    h._state.compact_note = (
        "State: probes green.\nDecisions: none.\n"
        "Learnings: mtime guard bites.\nNext: land the fix."
    )
    messages = [*_base_messages(), *_big_middle(), *_tail()]
    compacted, before, after = await h._compact_messages_with_synthesis(
        messages, _request(),
    )
    assert compacted
    assert after < before
    block = messages[2].content
    assert "### Synthesis" in block
    assert "mtime guard bites" in block
    # Consumed exactly once.
    assert h._state.compact_requested is False
    assert h._state.compact_note == ""


async def test_model_note_forces_fold_below_threshold(monkeypatch):
    """The whole point: the model folds at a seam, not at pressure."""
    _trip_compaction_thresholds(monkeypatch, keep_recent=2)
    h = _make_handler_for_compact()
    h._coder_compact_token_limit = 10_000_000  # far under threshold

    h._state.compact_requested = True
    h._state.compact_note = "State: a.\nDecisions: b.\nLearnings: c.\nNext: d."
    messages = [*_base_messages(), *_big_middle(), *_tail()]
    compacted, _b, _a = await h._compact_messages_with_synthesis(
        messages, _request(),
    )
    assert compacted
    assert "### Synthesis" in messages[2].content


async def test_flag_clears_even_when_fold_impossible(monkeypatch):
    _trip_compaction_thresholds(monkeypatch, keep_recent=2)
    h = _make_handler_for_compact()
    h._state.compact_requested = True
    h._state.compact_note = "State: a.\nDecisions: b.\nLearnings: c.\nNext: d."
    messages = _base_messages()  # no middle: structurally impossible
    compacted, _b, _a = await h._compact_messages_with_synthesis(
        messages, _request(),
    )
    assert not compacted
    assert h._state.compact_requested is False
    assert h._state.compact_note == ""


# ---------------------------------------------------------------------------
# Sticky-reminder context meter
# ---------------------------------------------------------------------------
def _inject(h, messages):
    h._inject_sticky_reminder(
        messages, goal="fix the bug", iteration=3, max_iters=40, writes=1,
    )
    return messages[-1].content


def test_meter_rendered_when_tool_enabled(monkeypatch):
    h = _make_handler_for_compact()
    monkeypatch.setattr(
        "augmentum.config.settings.coder_compact_tool_enabled",
        True, raising=False,
    )
    h._coder_compact_token_limit = 1_000
    content = _inject(h, [*_base_messages()])
    assert "context " in content and "% of budget" in content


def test_meter_absent_when_tool_disabled(monkeypatch):
    h = _make_handler_for_compact()
    monkeypatch.setattr(
        "augmentum.config.settings.coder_compact_tool_enabled",
        False, raising=False,
    )
    content = _inject(h, [*_base_messages()])
    assert "% of budget" not in content
    assert "Iteration 3/40" in content
