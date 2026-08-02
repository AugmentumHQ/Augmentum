"""Tests for the turn-scoped backend-boundary capture context.

Covers the Step-1 foundation: the gate, multi-call accumulation into one turn
trace, FULL system-prompt capture (the fix vs the old hash), the snapshot
serializer, and that tag resolution now flows through the canonical primer map.
The context is not yet wired into the hot path — these exercise it directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from augmentum.config import settings
from augmentum.models.base import (
    InternalChatRequest,
    InternalChatResponse,
    InternalStreamChunk,
    Message,
    ModelBackend,
    ModelDetails,
)
from augmentum.prompts.primer import SURFACE_TAGS, tag_for
from augmentum.training import trace_context as tc


@pytest.fixture
def capture_on(tmp_path):
    """Enable capture + point the trace dir at tmp; restore the singleton after."""
    keys = (
        "training_capture_enabled",
        "training_capture_user_id",
        "training_capture_dir",
    )
    orig = {k: getattr(settings, k) for k in keys}
    object.__setattr__(settings, "training_capture_enabled", True)
    object.__setattr__(settings, "training_capture_user_id", "")
    object.__setattr__(settings, "training_capture_dir", str(tmp_path))
    try:
        yield tmp_path
    finally:
        for k, v in orig.items():
            object.__setattr__(settings, k, v)


def _read_traces(trace_dir) -> list[dict]:
    rows: list[dict] = []
    for f in Path(trace_dir).rglob("*.jsonl"):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def test_disabled_is_noop():
    orig = settings.training_capture_enabled
    object.__setattr__(settings, "training_capture_enabled", False)
    try:
        with tc.capture_turn(user_id="u1", mode="coder") as ctx:
            assert ctx is None
            assert tc.current_capture_context() is None
    finally:
        object.__setattr__(settings, "training_capture_enabled", orig)


def test_records_and_writes_one_turn(capture_on):
    with tc.capture_turn(user_id="u1", session_id="s1", mode="coder") as ctx:
        assert ctx is not None
        assert tc.current_capture_context() is ctx
        ctx.record(
            system_prompt="FULL SYSTEM WITH MEMORY INJECTED",
            messages=[{"role": "user", "content": "fix the bug"}],
            tools=None,
            response_text="looking",
            response_thinking="",
            model="m",
        )
        ctx.record(
            system_prompt="FULL SYSTEM WITH MEMORY INJECTED",
            messages=[
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {"id": "1", "name": "web_search", "arguments": {"q": "x"}}
                    ],
                }
            ],
            tools=[{"type": "function"}],
            response_text="",
            response_thinking="thought",
            model="m",
        )

    # context cleared after the scope
    assert tc.current_capture_context() is None

    rows = _read_traces(capture_on)
    assert len(rows) == 1
    row = rows[0]
    assert row["schema_version"] == tc.TRACE_SCHEMA_VERSION
    assert row["mode"] == "coder"
    assert row["tag"] == ":-"
    assert row["lane"] == ":-"  # coder is its own training lane
    assert row["num_calls"] == 2
    # FULL system prompt is stored, not a hash
    assert row["calls"][0]["system_prompt"] == "FULL SYSTEM WITH MEMORY INJECTED"
    assert "web_search" in row["tools_used"]
    assert row["calls"][1]["response_thinking"] == "thought"

    # v3: training-ready fields
    assert row["system_prompt"] == "FULL SYSTEM WITH MEMORY INJECTED"
    assert isinstance(row["chain"], list)
    assert row["chain"][-1]["role"] == "assistant"
    assert row["chain"][-1]["thinking"] == "thought"
    assert row["chain_depth"] == 1
    assert isinstance(row["tools_available"], list)
    assert row["final_thinking"] == "thought"


def test_lane_fold_records_coarse_capability_bucket(capture_on):
    """The 16 surface tags fold into 5 seed lanes (+ game/stream/direct).
    Each trace records ``lane`` alongside ``tag``+``mode`` so the fold is
    reversible, and the folded surfaces land in the lane's directory.
    """
    from augmentum.training.trace_context import _lane_for, _tag_dir

    # Fold table: (mode, expected tag, expected lane, expected dir).
    cases = [
        ("chat", ":C", ":C", "chat"),
        ("coder", ":-", ":-", "coder"),
        ("narrative", ":N", ":N", "narrative"),
        ("voice", ":V", ":V", "voice"),
        # companion/autonomy folds to the :U "you" lane — NOT chat. This
        # is the lane whose data can't be sourced from reactive chat.
        ("companion", ":B", ":U", "you"),
        ("becca_direct", ":B", ":U", "you"),
        # analytical/agentic/builder/knowledge/cast keep distinct tags but
        # fold INTO chat — same capability, harness re-supplies the scaffold.
        ("analytical", ":A", ":C", "chat"),
        ("agentic", ":T", ":C", "chat"),
        ("builder", ":W", ":C", "chat"),
        ("knowledge", ":K", ":C", "chat"),
        # game/stream stay outside the seed core (their own :G/:L configs).
        ("game", ":G", ":G", "game"),
        ("stream", ":L", ":L", "stream"),
        # direct stays isolated — external-harness pass-through, never seed data.
        ("direct", ":D", ":D", "direct"),
    ]
    for mode, exp_tag, exp_lane, exp_dir in cases:
        assert tag_for(mode) == exp_tag, mode
        assert _lane_for(exp_tag) == exp_lane, mode
        assert _tag_dir(exp_tag) == exp_dir, mode


def test_companion_trace_lands_in_you_lane(capture_on):
    with tc.capture_turn(user_id="u1", mode="companion") as ctx:
        assert ctx is not None
        ctx.record(
            system_prompt="sys",
            messages=[{"role": "user", "content": "hey"}],
            tools=None,
            response_text="thinking of you",
            response_thinking="",
            model="m",
        )
    rows = _read_traces(capture_on)
    assert len(rows) == 1
    assert rows[0]["tag"] == ":B"      # fine-grained tag preserved
    assert rows[0]["lane"] == ":U"     # but folds into the autonomy lane


def test_user_filter(capture_on):
    object.__setattr__(settings, "training_capture_user_id", "trainer")
    with tc.capture_turn(user_id="someone_else", mode="chat") as ctx:
        assert ctx is None
    with tc.capture_turn(user_id="trainer", mode="chat") as ctx:
        assert ctx is not None
        ctx.record(
            system_prompt="s",
            messages=[],
            tools=None,
            response_text="hi",
            response_thinking="",
            model="m",
        )
    rows = _read_traces(capture_on)
    assert len(rows) == 1
    assert rows[0]["user_id"] == "trainer"


def test_background_skipped(capture_on):
    with tc.capture_turn(user_id="u1", mode="chat", is_background=True) as ctx:
        assert ctx is None
    assert _read_traces(capture_on) == []


def test_no_calls_writes_no_file(capture_on):
    with tc.capture_turn(user_id="u1", mode="chat") as ctx:
        assert ctx is not None  # active, but we record nothing
    assert _read_traces(capture_on) == []


def test_error_recorded_and_reraised(capture_on):
    with pytest.raises(ValueError), tc.capture_turn(user_id="u1", mode="chat") as ctx:
        ctx.record(
            system_prompt="s",
            messages=[],
            tools=None,
            response_text="partial",
            response_thinking="",
            model="m",
        )
        raise ValueError("boom")
    rows = _read_traces(capture_on)
    assert len(rows) == 1
    assert rows[0]["error"] == "ValueError"
    assert rows[0]["num_calls"] == 1


def test_serialize_request_splits_system_and_preserves_tool_shape():
    msgs = [
        Message(role="system", content="SYSTEM FULL TEXT"),
        Message(role="user", content="do x"),
        Message(
            role="assistant",
            content="",
            tool_calls=[
                {"id": "a", "function": {"name": "web_search", "arguments": '{"q": "x"}'}}
            ],
            thinking="reasoning...",
        ),
        Message(role="tool", content="result body", tool_call_id="a"),
    ]
    system_prompt, chain = tc.serialize_request(msgs)
    assert system_prompt == "SYSTEM FULL TEXT"
    assert chain[0] == {"role": "user", "content": "do x"}
    asst = chain[1]
    assert asst["tool_calls"][0]["name"] == "web_search"
    assert asst["tool_calls"][0]["arguments"] == {"q": "x"}  # JSON args parsed
    assert asst["thinking"] == "reasoning..."
    assert chain[2]["tool_call_id"] == "a"


def test_serialize_request_drops_harness_synthesis_turn():
    msgs = [Message(role="user", content="Now respond to the user using the above.")]
    _system, chain = tc.serialize_request(msgs)
    assert chain == []


def test_serialize_request_drops_all_synthesis_prompt_variants():
    # Every synthesis prompt the tool loop actually injects (passthrough/
    # agentic handlers) must be stripped — a leaked one poisons the training
    # chain with a phantom user turn (found 2026-08-01 eyeballing a :C row).
    leaked = [
        "Use the tool results above to answer the user's question. "
        "Do NOT repeat the raw tool output — synthesize a natural response.",
        "Synthesize the results into a clear, well-structured response.",
        "Incorporate the tool results above into your response.",
    ]
    for content in leaked:
        _system, chain = tc.serialize_request([Message(role="user", content=content)])
        assert chain == [], f"synthesis prompt leaked into chain: {content[:40]!r}"


def test_serialize_request_keeps_extra_system_messages_in_chain():
    msgs = [
        Message(role="system", content="primary"),
        Message(role="user", content="hi"),
        Message(role="system", content="[STATE: mid-conversation injection]"),
    ]
    system_prompt, chain = tc.serialize_request(msgs)
    assert system_prompt == "primary"
    roles = [m["role"] for m in chain]
    assert roles == ["user", "system"]  # the 2nd system survives in-order


def test_begin_end_capture_matches_context_manager(capture_on):
    ctx, tok = tc.begin_capture(user_id="u1", session_id="s2", mode="analytical")
    assert ctx is not None
    assert tc.current_capture_context() is ctx
    ctx.record(
        system_prompt="sys",
        messages=[{"role": "user", "content": "analyze"}],
        tools=[{"function": {"name": "web_search"}}],
        response_text="done",
        response_thinking="hmm",
        model="m",
    )
    tc.end_capture(ctx, tok)
    assert tc.current_capture_context() is None
    rows = _read_traces(capture_on)
    assert len(rows) == 1
    assert rows[0]["tag"] == ":A"
    assert rows[0]["tools_available"] == ["web_search"]
    assert rows[0]["final_response"] == "done"


def test_begin_end_noop_when_disabled():
    orig = settings.training_capture_enabled
    object.__setattr__(settings, "training_capture_enabled", False)
    try:
        ctx, tok = tc.begin_capture(user_id="u1", mode="chat")
        assert ctx is None
        assert tok is None
        tc.end_capture(ctx, tok)  # no-op, no crash
    finally:
        object.__setattr__(settings, "training_capture_enabled", orig)


def test_traces_written_to_tag_subdirectory(capture_on):
    with tc.capture_turn(user_id="u1", mode="narrative") as ctx:
        ctx.record(
            system_prompt="char card",
            messages=[{"role": "user", "content": "enter tavern"}],
            tools=None,
            response_text="the door creaks",
            response_thinking="setting scene",
            model="m",
        )
    tag_dir = capture_on / "narrative"
    assert tag_dir.exists()
    jsonl_files = list(tag_dir.glob("*.jsonl"))
    assert len(jsonl_files) == 1


def test_sampling_flags_and_tool_schemas_captured(capture_on):
    tool_schema = {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
    with tc.capture_turn(user_id="u1", session_id="s1", mode="analytical") as ctx:
        ctx.record(
            system_prompt="sys",
            messages=[{"role": "user", "content": "research this"}],
            tools=[tool_schema],
            response_text="here are the results",
            response_thinking="let me search",
            model="m",
            sampling={"temperature": 0.3, "top_p": 0.9, "max_tokens": 512},
            flags={"think": True, "voice_input": True},
        )
    rows = _read_traces(capture_on)
    assert len(rows) == 1
    row = rows[0]
    # Full tool schemas preserved
    assert row["tool_schemas"] == [tool_schema]
    assert row["tools_available"] == ["web_search"]
    # Sampling captured
    assert row["sampling"]["temperature"] == 0.3
    assert row["sampling"]["max_tokens"] == 512
    # Flags captured
    assert row["flags"]["think"] is True
    assert row["flags"]["voice_input"] is True


def test_tag_resolution_uses_canonical_primer_map():
    # Every canonical surface resolves to its tag (the unification source).
    for surface, expected in SURFACE_TAGS.items():
        assert tag_for(surface) == expected
    # Already-a-tag passes through; unknown defaults to :C (raw mode preserved).
    assert tag_for(":N") == ":N"
    assert tag_for("totally_unknown_mode") == ":C"


# --------------------------------------------------------------------------
# Backend interception (install_capture_hook)
# --------------------------------------------------------------------------


class _FakeBackend(ModelBackend):
    """Minimal concrete backend for exercising the hook."""

    async def chat(self, request: InternalChatRequest) -> InternalChatResponse:
        return InternalChatResponse(
            message=Message(role="assistant", content="hello", thinking="pondered"),
            model=request.model,
        )

    async def chat_stream(self, request: InternalChatRequest):
        for piece, think in (("he", "t1"), ("llo", "t2")):
            yield InternalStreamChunk(
                content_delta=piece, thinking_delta=think, model=request.model
            )

    async def list_models(self):
        return []

    async def show_model(self, name: str) -> ModelDetails:
        return ModelDetails()


def _req(messages=None, **kw) -> InternalChatRequest:
    return InternalChatRequest(
        model="m", messages=messages or [Message(role="user", content="hi")], **kw
    )


def test_install_is_idempotent_and_isinstance_preserved():
    b = _FakeBackend()
    tc.install_capture_hook(b)
    first = b.chat
    tc.install_capture_hook(b)  # second call is a no-op
    assert b.chat is first
    # The whole point of method-interception over wrapping: class is untouched.
    assert isinstance(b, _FakeBackend)
    assert isinstance(b, ModelBackend)


async def test_chat_records_under_capture(capture_on):
    b = _FakeBackend()
    tc.install_capture_hook(b)
    with tc.capture_turn(user_id="u1", mode="coder") as ctx:
        resp = await b.chat(_req())
        assert resp.message.content == "hello"
        assert len(ctx.calls) == 1
        assert ctx.calls[0]["response_text"] == "hello"
        assert ctx.calls[0]["response_thinking"] == "pondered"
    rows = _read_traces(capture_on)
    assert len(rows) == 1
    assert rows[0]["num_calls"] == 1
    assert rows[0]["tag"] == ":-"


async def test_chat_stream_accumulates_deltas(capture_on):
    b = _FakeBackend()
    tc.install_capture_hook(b)
    with tc.capture_turn(user_id="u1", mode="analytical") as ctx:
        chunks = [c async for c in b.chat_stream(_req())]
        assert "".join(c.content_delta for c in chunks) == "hello"
        assert len(ctx.calls) == 1
        assert ctx.calls[0]["response_text"] == "hello"
        assert ctx.calls[0]["response_thinking"] == "t1t2"


async def test_no_capture_context_is_pure_passthrough():
    b = _FakeBackend()
    tc.install_capture_hook(b)
    assert tc.current_capture_context() is None  # capture off
    resp = await b.chat(_req())
    assert resp.message.content == "hello"
    chunks = [c async for c in b.chat_stream(_req())]
    assert len(chunks) == 2  # stream still flows unchanged


async def test_background_request_not_recorded(capture_on):
    b = _FakeBackend()
    tc.install_capture_hook(b)
    with tc.capture_turn(user_id="u1", mode="coder") as ctx:
        await b.chat(_req(is_background_task=True))
        assert ctx.calls == []


async def test_uninstall_restores_and_stops_recording(capture_on):
    b = _FakeBackend()
    tc.install_capture_hook(b)
    assert getattr(b, "_augmentum_capture_hooked", False) is True
    tc.uninstall_capture_hook(b)
    assert getattr(b, "_augmentum_capture_hooked", False) is False
    tc.uninstall_capture_hook(b)  # idempotent
    with tc.capture_turn(user_id="u1", mode="coder") as ctx:
        resp = await b.chat(_req())
        assert resp.message.content == "hello"  # still works
        assert ctx.calls == []  # but no longer recorded


def test_every_classifier_mode_has_a_distinct_surface_tag():
    """Every Mode the classifier can route to MUST map to a primer tag.

    An unmapped mode silently falls through ``tag_for``'s ``:C`` default and
    mis-tags that mode's training traces as chat — the agentic + direct bug
    (2026-06-28): both were missing, so agentic builds and direct external-
    harness turns were being captured under the ``:C`` chat tag. This locks
    the coverage so a future Mode addition can't reintroduce the silent drift.
    """
    from augmentum.classifier.router import Mode
    from augmentum.training.trace_context import (
        _LANE_DIRS,
        _LANE_FOR_TAG,
        _lane_for,
    )

    for m in Mode:
        assert m.value in SURFACE_TAGS, (
            f"Mode {m.value!r} is missing from SURFACE_TAGS — it would be "
            f"mis-tagged as :C (chat). Add it to augmentum/prompts/primer.py."
        )
        tag = tag_for(m.value)
        # Only passthrough/chat may legitimately resolve to :C.
        if m.value not in ("passthrough", "chat"):
            assert tag != ":C", f"Mode {m.value!r} mis-tagged as :C"
        # Every surface tag must fold to a training LANE (5 seed lanes +
        # game/stream/direct), and every lane must have a directory so a
        # write never lands in 'other'. This locks the fold: a new Mode
        # whose tag isn't in _LANE_FOR_TAG would silently default to the
        # :C chat lane, re-introducing the mis-tag drift at the lane level.
        assert tag in _LANE_FOR_TAG, (
            f"tag {tag!r} for {m.value!r} has no _LANE_FOR_TAG entry — it "
            f"would silently fold into the :C chat lane."
        )
        assert _lane_for(tag) in _LANE_DIRS, (
            f"lane {_lane_for(tag)!r} (tag {tag!r}) has no _LANE_DIRS entry"
        )
