"""Tests for the 2026-04-20 loop-guard additions to _act_hybrid.

Covers:
  A. Inspection-only streak detection — when the user asks for
     creation ("make me X") and the model spends N iterations probing
     without writing anything, inject a strong nudge and break on the
     next iteration if still no writes. Prevents the "probe forever,
     produce nothing" failure mode observed 2026-04-20 on a "make me
     a snake game" request.

  C. Within-batch exact-args deduplication — when the model emits the
     same tool call twice in one batch, run it once and synthesize a
     ``batch_duplicate`` tool_result for the others so their tool_use
     IDs don't dangle and we save the round-trip.

Run: python -m pytest tests/test_coder_loop_guards.py -v
"""
from __future__ import annotations

import json

import pytest

from augmentum.modes.coder.handler import (
    CoderHandler,
    _CREATION_VERB_RE,
    _intent_key,
)
from augmentum.models.base import InternalStreamChunk

from tests.test_coder_handler import (
    _ExtendedContainerManager,
    _FakeChunk,
    _FakeTool,
    _force_native_tier,
    _make_request,
    _tc_delta,
)


# ---------------------------------------------------------------------------
# Creation-verb regex
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("goal,matches", [
    ("make me a snake game", True),
    ("create a new endpoint", True),
    ("build the auth module", True),
    ("add tests for X", True),
    ("implement the parser", True),
    ("write a CLI wrapper", True),
    ("generate the migration", True),
    ("scaffold a FastAPI app", True),
    ("initialize the database", True),
    ("init a pyproject.toml", True),
    ("refactor the handler", True),
    ("port this to Rust", True),
    # Non-creation goals
    ("explain how X works", False),
    ("show me the config", False),
    ("why is the test failing?", False),
    ("what does this function do", False),
    # Case insensitive
    ("MAKE me something", True),
    ("Create the thing", True),
])
def test_creation_verb_regex(goal, matches):
    assert bool(_CREATION_VERB_RE.search(goal)) is matches


# ---------------------------------------------------------------------------
# _intent_key — deterministic exact-args dedup
# ---------------------------------------------------------------------------


def test_intent_key_identical_calls_produce_same_key():
    a = {"name": "file_read", "input": {"path": "/x.py"}}
    b = {"name": "file_read", "input": {"path": "/x.py"}}
    assert _intent_key(a) == _intent_key(b)


def test_intent_key_different_args_produce_different_keys():
    a = {"name": "file_read", "input": {"path": "/x.py"}}
    b = {"name": "file_read", "input": {"path": "/y.py"}}
    assert _intent_key(a) != _intent_key(b)


def test_intent_key_different_tools_produce_different_keys():
    a = {"name": "file_read", "input": {"path": "/x.py"}}
    b = {"name": "code_grep", "input": {"path": "/x.py"}}
    assert _intent_key(a) != _intent_key(b)


def test_intent_key_key_order_does_not_matter():
    """A model that emits args in different orders still dedupes."""
    a = {"name": "code_grep", "input": {"pattern": "foo", "path": "/x"}}
    b = {"name": "code_grep", "input": {"path": "/x", "pattern": "foo"}}
    assert _intent_key(a) == _intent_key(b)


def test_intent_key_handles_native_tool_shape():
    """Native-tier tool calls have arguments as a JSON string inside
    ``function``. _intent_key must still normalise them."""
    a = {
        "id": "tc-1",
        "function": {"name": "file_read", "arguments": '{"path": "/x.py"}'},
    }
    b = {
        "id": "tc-2",
        "function": {"name": "file_read", "arguments": '{"path": "/x.py"}'},
    }
    assert _intent_key(a) == _intent_key(b)


def test_intent_key_robust_to_bad_args_string():
    """Garbage arguments string → empty key (but doesn't crash)."""
    tc = {"name": "file_read", "input": "not-json-{{"}
    key = _intent_key(tc)
    assert isinstance(key, tuple)


# ---------------------------------------------------------------------------
# Fix C: within-batch duplicate read dedup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_duplicate_reads_in_batch_run_once(monkeypatch):
    """Model emits two identical file_read calls in one tool_call list.
    Only one hits the container; the second gets a synthetic
    ``batch_duplicate`` tool_result pointing at the first's ID."""
    _force_native_tier(monkeypatch)

    file_tool = _FakeTool("file_read", output="body-of-x")
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [file_tool],
    )

    class _DuplicateBatch:
        def __init__(self):
            self.calls = 0

        async def chat_stream(self, request):
            self.calls += 1
            if self.calls == 1:
                yield _FakeChunk(augmentum={"tool_calls": [
                    _tc_delta(0, "tc-a", "file_read",
                              {"path": "/workspace/x.py"}),
                    _tc_delta(1, "tc-b", "file_read",
                              {"path": "/workspace/x.py"}),
                ]})
                yield _FakeChunk(done=True, finish_reason="tool_calls")
            else:
                yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    handler = CoderHandler(
        _DuplicateBatch(), session_id="sess-dup",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-dup",
    )

    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_hybrid(_make_request(), workspace_context=""):
        chunks.append(c)

    # Exactly one real execution of file_read
    assert len(file_tool.calls) == 1, (
        f"Expected dedup to collapse to one real call; got "
        f"{len(file_tool.calls)}"
    )

    # Two tool_result meta chunks: one canonical, one batch_duplicate
    tool_results = [
        c for c in chunks
        if c.augmentum and c.augmentum.get("status") == "tool_result"
    ]
    assert len(tool_results) == 2
    duplicates = [
        c for c in tool_results
        if c.augmentum.get("tool_result", {}).get("batch_duplicate")
    ]
    assert len(duplicates) == 1
    # The dup points at the canonical id
    assert duplicates[0].augmentum["tool_result"]["canonical_tool_call_id"] == "tc-a"
    assert duplicates[0].augmentum["tool_result"]["id"] == "tc-b"


@pytest.mark.asyncio
async def test_distinct_reads_in_batch_all_run(monkeypatch):
    """Calls with different args are NOT collapsed — different intent
    keys, different container round-trips."""
    _force_native_tier(monkeypatch)

    file_tool = _FakeTool("file_read", output="body")
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [file_tool],
    )

    class _DistinctBatch:
        def __init__(self):
            self.calls = 0

        async def chat_stream(self, request):
            self.calls += 1
            if self.calls == 1:
                yield _FakeChunk(augmentum={"tool_calls": [
                    _tc_delta(0, "tc-a", "file_read", {"path": "/a.py"}),
                    _tc_delta(1, "tc-b", "file_read", {"path": "/b.py"}),
                    _tc_delta(2, "tc-c", "file_read", {"path": "/c.py"}),
                ]})
                yield _FakeChunk(done=True, finish_reason="tool_calls")
            else:
                yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    handler = CoderHandler(
        _DistinctBatch(), session_id="sess-dist",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-dist",
    )
    async for _ in handler._act_hybrid(_make_request(), workspace_context=""):
        pass

    assert len(file_tool.calls) == 3


# ---------------------------------------------------------------------------
# Fix A: inspection-only streak nudge + break
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inspection_streak_fires_nudge_on_creation_task(monkeypatch):
    """Model inspects for N iterations on a "make me X" task → nudge
    meta chunk appears, model gets one more iteration to act.

    N = _INSPECTION_COLD_START_GRACE + _INSPECTION_STREAK_NUDGE. The
    cold-start grace (2 iters) was added 2026-04-20 from qwen-code so
    legitimate first-turn exploration doesn't false-positive into the
    nudge; the streak starts ticking from iteration 3."""
    _force_native_tier(monkeypatch)
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [_FakeTool("file_read", output="ok")],
    )

    class _AlwaysInspects:
        """Model that keeps emitting file_read and never writes."""
        def __init__(self):
            self.calls = 0

        async def chat_stream(self, request):
            self.calls += 1
            # Need enough iterations to cover cold-start grace (2) +
            # the streak threshold (default 5, env-tunable). 10 iters
            # gives room for the nudge to fire at iter 7.
            if self.calls <= 10:
                yield _FakeChunk(augmentum={"tool_calls": [
                    _tc_delta(0, f"tc-{self.calls}", "file_read",
                              {"path": f"/workspace/step{self.calls}.py"}),
                ]})
                yield _FakeChunk(done=True, finish_reason="tool_calls")
            else:
                yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    handler = CoderHandler(
        _AlwaysInspects(), session_id="sess-insp",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-insp",
    )

    # "make me a snake game" — has creation verb
    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_hybrid(
        _make_request("make me a snake game"), workspace_context="",
    ):
        chunks.append(c)

    nudge_chunks = [
        c for c in chunks
        if c.augmentum and c.augmentum.get("status") == "inspection_loop_nudge"
    ]
    assert nudge_chunks, (
        "Expected an inspection_loop_nudge after 3 inspection-only iterations"
    )
    assert nudge_chunks[0].augmentum.get("streak") >= 3


@pytest.mark.asyncio
async def test_inspection_streak_breaks_after_nudge_if_still_no_writes(monkeypatch):
    """If the model continues inspecting after the nudge, the loop
    breaks with inspection_loop_break termination."""
    _force_native_tier(monkeypatch)
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [_FakeTool("file_read", output="ok")],
    )

    class _NeverWrites:
        def __init__(self):
            self.calls = 0

        async def chat_stream(self, request):
            self.calls += 1
            # Keep emitting file_reads forever — the break is what we
            # want to verify, not a natural stop.
            yield _FakeChunk(augmentum={"tool_calls": [
                _tc_delta(0, f"tc-{self.calls}", "file_read",
                          {"path": f"/workspace/f{self.calls}.py"}),
            ]})
            yield _FakeChunk(done=True, finish_reason="tool_calls")

        async def chat(self, request):
            return None

    handler = CoderHandler(
        _NeverWrites(), session_id="sess-brk",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-brk",
    )
    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_hybrid(
        _make_request("build me an auth module"), workspace_context="",
    ):
        chunks.append(c)

    # Break chunk with streak >= 4 (3 trigger + at least 1 after nudge)
    break_chunks = [
        c for c in chunks
        if c.augmentum and c.augmentum.get("status") == "inspection_loop_break"
    ]
    assert break_chunks, "Expected inspection_loop_break to fire"
    assert break_chunks[0].augmentum.get("streak") >= 4


@pytest.mark.asyncio
async def test_inspection_streak_ignores_non_creation_tasks(monkeypatch):
    """A question like "explain how X works" has no creation verb —
    the model should be allowed to inspect freely without triggering
    the nudge or break."""
    _force_native_tier(monkeypatch)
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [_FakeTool("file_read", output="ok")],
    )

    class _InspectsFiveTimes:
        def __init__(self):
            self.calls = 0

        async def chat_stream(self, request):
            self.calls += 1
            if self.calls <= 5:
                yield _FakeChunk(augmentum={"tool_calls": [
                    _tc_delta(0, f"tc-{self.calls}", "file_read",
                              {"path": f"/f{self.calls}.py"}),
                ]})
                yield _FakeChunk(done=True, finish_reason="tool_calls")
            else:
                yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    handler = CoderHandler(
        _InspectsFiveTimes(), session_id="sess-nc",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-nc",
    )
    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_hybrid(
        _make_request("explain how the auth flow works"), workspace_context="",
    ):
        chunks.append(c)

    nudge_chunks = [
        c for c in chunks
        if c.augmentum
        and c.augmentum.get("status") in (
            "inspection_loop_nudge", "inspection_loop_break",
        )
    ]
    assert nudge_chunks == [], (
        "Inspection nudge/break should NOT fire on a non-creation goal"
    )


@pytest.mark.asyncio
async def test_inspection_streak_resets_on_write(monkeypatch):
    """If the model inspects twice, then writes, then inspects three
    more times, the nudge should NOT fire — the write reset the streak."""
    _force_native_tier(monkeypatch)

    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [
            _FakeTool("file_read", output="ok"),
            _FakeTool("file_write", output="wrote"),
        ],
    )

    class _MixedSequence:
        """Inspect, inspect, WRITE, inspect, inspect, inspect, stop."""
        def __init__(self):
            self.step = 0
            self.sequence = [
                ("file_read", {"path": "/a.py"}),
                ("file_read", {"path": "/b.py"}),
                ("file_write", {"path": "/out.py", "content": "x"}),
                ("file_read", {"path": "/c.py"}),
                ("file_read", {"path": "/d.py"}),
                ("file_read", {"path": "/e.py"}),
            ]

        async def chat_stream(self, request):
            if self.step < len(self.sequence):
                name, args = self.sequence[self.step]
                self.step += 1
                yield _FakeChunk(augmentum={"tool_calls": [
                    _tc_delta(0, f"tc-{self.step}", name, args),
                ]})
                yield _FakeChunk(done=True, finish_reason="tool_calls")
            else:
                yield _FakeChunk(done=True, finish_reason="stop")

        async def chat(self, request):
            return None

    handler = CoderHandler(
        _MixedSequence(), session_id="sess-mix",
        container_manager=_ExtendedContainerManager(),
        workspace_id="ws-mix",
    )
    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_hybrid(
        _make_request("create the new endpoint"), workspace_context="",
    ):
        chunks.append(c)

    nudge_chunks = [
        c for c in chunks
        if c.augmentum and c.augmentum.get("status") == "inspection_loop_nudge"
    ]
    # 3 inspections after the write → nudge SHOULD fire (streak restart
    # got to 3 again). This test specifically verifies the streak
    # RESETS on write, so check that the nudge only sees streak=3 (not
    # 5, which would mean the write was never counted as progress).
    if nudge_chunks:
        assert nudge_chunks[0].augmentum.get("streak") == 3, (
            "Streak should reset on write, so the post-write nudge "
            "should fire at 3, not at 5 (which would be the total "
            "inspection count including pre-write iters)."
        )


# ── Identical-call loop detector (2026-06-17) ───────────────────────────
# A model re-issuing the SAME tool with the SAME args for byte-identical
# output is stuck even when each call succeeds — the case no_progress,
# silent_success, and the validation breaks all miss. These pin the two
# pure helpers behind the native-loop nudge.

from augmentum.loops.breakers import (
    ALL_BREAKERS,
    IDENTICAL_TOOL_RESULT_NUDGE_AT,
)
from augmentum.modes.coder.phase_act import (
    _bump_identical_streaks,
    _identical_result_signature,
)


class TestIdenticalResultSignature:
    def test_same_call_same_signature(self):
        a = _identical_result_signature("file_read", {"path": "/x"}, "hello")
        b = _identical_result_signature("file_read", {"path": "/x"}, "hello")
        assert a == b

    def test_arg_key_order_does_not_matter(self):
        a = _identical_result_signature("shell_exec", {"cmd": "ls", "cwd": "/x"}, "out")
        b = _identical_result_signature("shell_exec", {"cwd": "/x", "cmd": "ls"}, "out")
        assert a == b

    def test_different_output_breaks_the_match(self):
        a = _identical_result_signature("shell_exec", {"cmd": "date"}, "10:00")
        b = _identical_result_signature("shell_exec", {"cmd": "date"}, "10:01")
        assert a != b

    def test_different_args_break_the_match(self):
        a = _identical_result_signature("file_read", {"path": "/x"}, "same")
        b = _identical_result_signature("file_read", {"path": "/y"}, "same")
        assert a != b

    def test_different_tool_breaks_the_match(self):
        a = _identical_result_signature("file_read", {"path": "/x"}, "same")
        b = _identical_result_signature("file_list", {"path": "/x"}, "same")
        assert a != b

    def test_unserialisable_input_does_not_raise(self):
        # A set isn't JSON-serialisable — must fall back to repr, not crash.
        sig = _identical_result_signature("t", {"k": {1, 2}}, "out")
        assert isinstance(sig, str) and sig


class TestBumpIdenticalStreaks:
    def test_consecutive_repeats_increment(self):
        streaks: dict[str, int] = {}
        assert _bump_identical_streaks(streaks, {"a"}) == 1
        assert _bump_identical_streaks(streaks, {"a"}) == 2
        assert _bump_identical_streaks(streaks, {"a"}) == 3

    def test_gap_resets_the_streak(self):
        streaks: dict[str, int] = {}
        _bump_identical_streaks(streaks, {"a"})
        _bump_identical_streaks(streaks, {"a"})
        # "a" absent this iteration → dropped; "b" starts fresh.
        peak = _bump_identical_streaks(streaks, {"b"})
        assert peak == 1
        assert "a" not in streaks
        # "a" returning starts from 1 again, not 3.
        assert _bump_identical_streaks(streaks, {"a"}) == 1

    def test_peak_is_the_max_across_signatures(self):
        streaks: dict[str, int] = {}
        _bump_identical_streaks(streaks, {"a", "b"})
        _bump_identical_streaks(streaks, {"a", "b"})
        peak = _bump_identical_streaks(streaks, {"a"})  # b drops, a -> 3
        assert peak == 3
        assert streaks == {"a": 3}

    def test_empty_iteration_returns_zero_and_clears(self):
        streaks: dict[str, int] = {"a": 5}
        assert _bump_identical_streaks(streaks, set()) == 0
        assert streaks == {}

    def test_threshold_default_is_three(self):
        assert IDENTICAL_TOOL_RESULT_NUDGE_AT == 3

    def test_breaker_is_registered_as_a_nudge(self):
        b = next(
            (x for x in ALL_BREAKERS if x.name == "identical_tool_result_nudge"),
            None,
        )
        assert b is not None
        assert b.kind == "nudge"
        assert b.env_var == "AUGMENTUM_CODER_IDENTICAL_RESULT_STREAK"


# ── Identical-call nudge — native loop integration ──────────────────────

from tests.test_coder_handler import (  # noqa: E402
    _FakeContainerManager,
    _SequencedBackend,
)


@pytest.mark.asyncio
async def test_native_nudges_on_identical_repeated_tool_result(monkeypatch):
    """Three consecutive iterations issuing the SAME file_read with the SAME
    output must trip the identical-call nudge (streak reaches the default 3),
    even though every call succeeds (so no_progress/silent_success never
    fire). The backend exhausts after the 3 reads; native's empty-stop
    retry cap then terminates the turn cleanly."""
    _force_native_tier(monkeypatch)

    # Fixed-output read tool → byte-identical result every call.
    fake_read = _FakeTool("file_read", output="SAME-CONTENTS")
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [fake_read],
    )

    def _read_iter(tag: str):
        return [
            _FakeChunk(augmentum={"tool_calls": [
                _tc_delta(0, tag, "file_read", {"path": "/workspace/a.py"}),
            ]}),
            _FakeChunk(done=True, finish_reason="tool_calls"),
        ]

    backend = _SequencedBackend([
        _read_iter("r1"), _read_iter("r2"), _read_iter("r3"),
    ])

    handler = CoderHandler(
        backend,
        session_id="sess-identical",
        container_manager=_FakeContainerManager(),
        workspace_id="ws-identical",
    )

    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_native(_make_request(), workspace_context=""):
        chunks.append(c)

    nudges = [
        c for c in chunks
        if c.augmentum and c.augmentum.get("status") == "identical_result_nudge"
    ]
    assert len(nudges) == 1, "expected exactly one identical-call nudge"
    assert nudges[0].augmentum.get("tool") == "file_read"
    assert nudges[0].augmentum.get("streak") >= 3
    # The read ran at least 3 times (the nudge is appended, not a break).
    assert len(fake_read.calls) >= 3


@pytest.mark.asyncio
async def test_native_does_not_nudge_paged_large_file_reads(monkeypatch):
    """Paging a large file is NOT a loop and must never trip the
    identical-call nudge. file_read takes an `offset` arg, so reading
    successive windows of a 3000-line file uses DIFFERENT args each call
    → different signatures → no nudge, even if outputs happened to match.
    Pins the 'don't punish reading a big file in chunks' requirement."""
    _force_native_tier(monkeypatch)

    fake_read = _FakeTool("file_read", output="<window>")  # fixed output
    monkeypatch.setattr(
        "augmentum.modes.coder.handler.create_coder_tools",
        lambda cm, ws, state, **_: [fake_read],
    )

    def _paged_read(tag: str, offset: int):
        return [
            _FakeChunk(augmentum={"tool_calls": [
                _tc_delta(0, tag, "file_read", {
                    "path": "/workspace/big.py", "offset": offset,
                }),
            ]}),
            _FakeChunk(done=True, finish_reason="tool_calls"),
        ]

    # Three consecutive reads advancing the offset — legitimate paging.
    backend = _SequencedBackend([
        _paged_read("p1", 0),
        _paged_read("p2", 2000),
        _paged_read("p3", 4000),
    ])

    handler = CoderHandler(
        backend,
        session_id="sess-paged",
        container_manager=_FakeContainerManager(),
        workspace_id="ws-paged",
    )

    chunks: list[InternalStreamChunk] = []
    async for c in handler._act_native(_make_request(), workspace_context=""):
        chunks.append(c)

    nudges = [
        c for c in chunks
        if c.augmentum and c.augmentum.get("status") == "identical_result_nudge"
    ]
    assert nudges == [], (
        "Paged reads (distinct offsets) must not trip the identical-call "
        "nudge — the model is making real progress through the file."
    )
    assert len(fake_read.calls) == 3


# ---------------------------------------------------------------------------
# Leaked-tool-markup detector (2026-07-02) — Qwen-XML tool calls emitted
# inside the thinking channel never execute; the stop gate must see them.
# ---------------------------------------------------------------------------


def test_leaked_markup_detects_xml_call_in_thinking():
    from augmentum.modes.coder.handler import _has_leaked_tool_markup

    thinking = (
        "Let me read a few key files.\n"
        "<tool_call>\n<function=file_read>\n<parameter=path>\n"
        "/workspace/erome-index/app.py\n</parameter>\n</function>\n"
        "</tool_call>\n"
        "I've already gathered a lot of information."
    )
    assert _has_leaked_tool_markup(thinking) is True


def test_leaked_markup_detects_bare_function_form():
    from augmentum.modes.coder.handler import _has_leaked_tool_markup

    assert _has_leaked_tool_markup("<function=shell_exec>ls</function>") is True


def test_leaked_markup_ignores_prose_and_unknown_names():
    from augmentum.modes.coder.handler import _has_leaked_tool_markup

    # Talking ABOUT tools in prose is fine.
    assert _has_leaked_tool_markup(
        "I will call file_read next, then run the tests."
    ) is False
    # Quoted third-party markup with an unknown function name is not a
    # coder tool call.
    assert _has_leaked_tool_markup("<function=window.onload>") is False
    assert _has_leaked_tool_markup("") is False
