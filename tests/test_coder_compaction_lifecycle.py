"""Context-lifecycle tests: supersession tombstoning + synthesis section.

2026-07-06 compaction upgrade, two layers:

1. Supersession tombstoning — a file_read result in the dropped region
   whose file is read again or edited LATER (later in the dropped
   region, or anywhere in the verbatim tail) is a stale copy; it
   renders as a one-line tombstone instead of a 1500-char preview.
   The NEWEST touch keeps carrying the grounded content.
2. LLM synthesis — ``_compact_messages_with_synthesis`` writes an LLM
   handoff note into the NEW segment as ``### Synthesis``; any failure
   on the synthesis path degrades to the mechanical segment (compaction
   itself must never fail because of it).
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from augmentum.models.base import InternalChatRequest, Message

# Reuse the scaffolding the other compaction tests use.
from tests.test_coder_context_preservation import (
    _make_handler_for_compact,
    _trip_compaction_thresholds,
)


def _assistant_with_tool(name: str, args: dict, tool_id: str) -> Message:
    return Message(
        role="assistant", content="",
        tool_calls=[{
            "id": tool_id,
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }],
    )


def _tool_result(tool_id: str, content: str) -> Message:
    return Message(role="tool", content=content, tool_call_id=tool_id)


def _base_messages() -> list[Message]:
    return [
        Message(role="system", content="sys" + "x" * 200),
        Message(role="user", content="task " + "w " * 1100),
    ]


def _tail() -> list[Message]:
    return [
        Message(role="assistant", content="tail"),
        Message(role="tool", content="tail", tool_call_id="tt1"),
        Message(role="assistant", content="tail2"),
        Message(role="tool", content="tail2", tool_call_id="tt2"),
    ]


def _request() -> InternalChatRequest:
    return InternalChatRequest(
        model="test-model",
        messages=[Message(role="user", content="task")],
        stream=True,
    )


# ---------------------------------------------------------------------------
# Supersession tombstoning
# ---------------------------------------------------------------------------

def test_reread_tombstones_older_copy(monkeypatch):
    """Two reads of the same file: the older one becomes a tombstone,
    the newer one keeps its grounded content."""
    _trip_compaction_thresholds(monkeypatch, keep_recent=2)
    h = _make_handler_for_compact()

    old_content = "OLD_COPY " + "a" * 1200
    new_content = "NEW_COPY " + "b" * 1200

    messages = [
        *_base_messages(),
        _assistant_with_tool("file_read", {"path": "/ws/a.py"}, "r1"),
        _tool_result("r1", old_content),
        _assistant_with_tool("file_read", {"path": "/ws/a.py"}, "r2"),
        _tool_result("r2", new_content),
        *_tail(),
    ]

    compacted, _b, _a = h._maybe_compact_messages(messages)
    assert compacted

    block = messages[2].content
    assert "stale copy dropped" in block
    assert "OLD_COPY" not in block
    assert "NEW_COPY" in block


def test_later_edit_tombstones_earlier_read(monkeypatch):
    """A read followed by an edit of the same file is stale."""
    _trip_compaction_thresholds(monkeypatch, keep_recent=2)
    h = _make_handler_for_compact()

    read_content = "READ_COPY " + "a" * 1200

    messages = [
        *_base_messages(),
        _assistant_with_tool("file_read", {"path": "/ws/a.py"}, "r1"),
        _tool_result("r1", read_content),
        _assistant_with_tool("code_edit", {"path": "/ws/a.py"}, "e1"),
        _tool_result("e1", "edited ok " + "c" * 1200),
        *_tail(),
    ]

    compacted, _b, _a = h._maybe_compact_messages(messages)
    assert compacted

    block = messages[2].content
    assert "stale copy dropped" in block
    assert "READ_COPY" not in block


def test_tail_touch_tombstones_dropped_read(monkeypatch):
    """A read in the dropped region is stale when the same file is
    touched again in the verbatim tail."""
    _trip_compaction_thresholds(monkeypatch, keep_recent=4)
    h = _make_handler_for_compact()

    read_content = "READ_COPY " + "a" * 1200

    messages = [
        *_base_messages(),
        _assistant_with_tool("file_read", {"path": "/ws/a.py"}, "r1"),
        _tool_result("r1", read_content),
        _assistant_with_tool("code_grep", {"pattern": "x"}, "g1"),
        _tool_result("g1", "match " + "m" * 1200),
        # Tail (keep_recent=4): re-reads the same file.
        _assistant_with_tool("file_read", {"path": "/ws/a.py"}, "r2"),
        _tool_result("r2", "fresh tail copy"),
        Message(role="assistant", content="tail prose"),
        Message(role="tool", content="tail", tool_call_id="tt1"),
    ]

    compacted, _b, _a = h._maybe_compact_messages(messages)
    assert compacted

    block = messages[2].content
    assert "stale copy dropped" in block
    assert "READ_COPY" not in block


def test_sole_read_keeps_grounded_content(monkeypatch):
    """A single un-superseded read keeps its content — the original
    grounded-content guarantee must not regress."""
    _trip_compaction_thresholds(monkeypatch, keep_recent=2)
    h = _make_handler_for_compact()

    read_content = "GROUNDED_FACT " + "a" * 1200

    messages = [
        *_base_messages(),
        _assistant_with_tool("file_read", {"path": "/ws/a.py"}, "r1"),
        _tool_result("r1", read_content),
        _assistant_with_tool("file_read", {"path": "/ws/other.py"}, "r2"),
        # Big enough to be clipped so the pass nets real savings and
        # the rollback guard (after >= before) doesn't fire.
        _tool_result("r2", "other " + "b" * 5000),
        *_tail(),
    ]

    compacted, _b, _a = h._maybe_compact_messages(messages)
    assert compacted

    block = messages[2].content
    assert "GROUNDED_FACT" in block
    assert "stale copy dropped" not in block


def test_error_read_result_never_tombstoned(monkeypatch):
    """ERROR results keep their recovery hint even when superseded —
    the error is the lesson, not a stale copy."""
    _trip_compaction_thresholds(monkeypatch, keep_recent=2)
    h = _make_handler_for_compact()

    messages = [
        *_base_messages(),
        _assistant_with_tool("file_read", {"path": "/ws/a.py"}, "r1"),
        _tool_result("r1", "ERROR: ENOENT no such file /ws/a.py" + " p" * 600),
        _assistant_with_tool("file_read", {"path": "/ws/a.py"}, "r2"),
        _tool_result("r2", "created later, content " + "b" * 1200),
        *_tail(),
    ]

    compacted, _b, _a = h._maybe_compact_messages(messages)
    assert compacted

    block = messages[2].content
    assert "ERROR: ENOENT" in block


# ---------------------------------------------------------------------------
# Synthesis section
# ---------------------------------------------------------------------------

def _big_middle() -> list[Message]:
    big = "x" * 3000
    return [
        _assistant_with_tool("file_read", {"path": "/ws/a.py"}, "t1"),
        _tool_result("t1", big),
        _assistant_with_tool("code_grep", {"pattern": "foo"}, "t2"),
        _tool_result("t2", big),
    ]


async def test_synthesis_section_lands_in_segment(monkeypatch):
    _trip_compaction_thresholds(monkeypatch, keep_recent=2)
    h = _make_handler_for_compact()

    async def _fake_synth(*args, **kwargs):
        return "State: half done.\nDecisions: used X because Y.\nLearnings: none.\nNext: wire Z."

    import augmentum.coder.compaction_synthesis as cs
    monkeypatch.setattr(cs, "synthesize_compaction_segment", _fake_synth)

    messages = [*_base_messages(), *_big_middle(), *_tail()]
    compacted, before, after = await h._compact_messages_with_synthesis(
        messages, _request(),
    )
    assert compacted
    assert after < before

    block = messages[2].content
    assert "### Synthesis" in block
    assert "used X because Y" in block
    # Section order: Synthesis sits before the mechanical Details.
    assert block.index("### Synthesis") < block.index("### Details")


async def test_synthesis_failure_degrades_to_mechanical(monkeypatch):
    """A raising synthesis path must not block compaction."""
    _trip_compaction_thresholds(monkeypatch, keep_recent=2)
    h = _make_handler_for_compact()

    async def _boom(*args, **kwargs):
        raise RuntimeError("provider down")

    import augmentum.coder.compaction_synthesis as cs
    monkeypatch.setattr(cs, "synthesize_compaction_segment", _boom)

    messages = [*_base_messages(), *_big_middle(), *_tail()]
    compacted, _b, _a = await h._compact_messages_with_synthesis(
        messages, _request(),
    )
    assert compacted
    block = messages[2].content
    assert "### Synthesis" not in block
    assert "### Details" in block


async def test_synthesis_disabled_by_setting(monkeypatch):
    _trip_compaction_thresholds(monkeypatch, keep_recent=2)
    h = _make_handler_for_compact()

    from augmentum.config import settings as _settings
    monkeypatch.setattr(
        _settings, "coder_compaction_synthesis_enabled", False,
    )

    called = False

    async def _fake_synth(*args, **kwargs):
        nonlocal called
        called = True
        return "should not appear"

    import augmentum.coder.compaction_synthesis as cs
    monkeypatch.setattr(cs, "synthesize_compaction_segment", _fake_synth)

    messages = [*_base_messages(), *_big_middle(), *_tail()]
    compacted, _b, _a = await h._compact_messages_with_synthesis(
        messages, _request(),
    )
    assert compacted
    assert not called
    assert "### Synthesis" not in messages[2].content


async def test_below_threshold_makes_no_synthesis_call(monkeypatch):
    """No compaction → no second-model call at all."""
    _trip_compaction_thresholds(monkeypatch, keep_recent=2)
    # Raise the trigger back up so the tiny history is under threshold.
    monkeypatch.setattr(
        "augmentum.modes.coder.handler._COMPACT_AT_TOKENS", 10_000_000,
    )
    h = _make_handler_for_compact()

    called = False

    async def _fake_synth(*args, **kwargs):
        nonlocal called
        called = True
        return "x"

    import augmentum.coder.compaction_synthesis as cs
    monkeypatch.setattr(cs, "synthesize_compaction_segment", _fake_synth)

    messages = [*_base_messages(), *_big_middle(), *_tail()]
    compacted, before, after = await h._compact_messages_with_synthesis(
        messages, _request(),
    )
    assert not compacted
    assert before == after
    assert not called


# ---------------------------------------------------------------------------
# The synthesis module itself
# ---------------------------------------------------------------------------

async def test_synthesize_reads_response_and_clears_kv_affinity():
    from augmentum.coder.compaction_synthesis import (
        synthesize_compaction_segment,
    )

    seen: dict = {}

    class _Backend:
        async def chat(self, request):
            seen["request"] = request
            return SimpleNamespace(
                message=Message(role="assistant", content="State: fine."),
            )

    source = InternalChatRequest(
        model="m", messages=[Message(role="user", content="go")],
        stream=True, kv_session_key="warm-slot",
    )
    out = await synthesize_compaction_segment(
        _Backend(),
        source_request=source,
        segment_preview="### Summary\n- Edited: a.py\n### Details\nT: x",
        user_goal="fix the bug",
    )
    assert out == "State: fine."
    req = seen["request"]
    assert req.kv_session_key == ""  # never evict the act loop's warm slot
    assert req.stream is False
    assert req.tools is None
    assert req.think is False


async def test_synthesize_fails_open_on_backend_error():
    from augmentum.coder.compaction_synthesis import (
        synthesize_compaction_segment,
    )

    class _Backend:
        async def chat(self, request):
            raise RuntimeError("boom")

    source = InternalChatRequest(
        model="m", messages=[Message(role="user", content="go")], stream=True,
    )
    out = await synthesize_compaction_segment(
        _Backend(), source_request=source,
        segment_preview="T: x", user_goal="g",
    )
    assert out is None


# ---------------------------------------------------------------------------
# Arg-preview digests: edit intent survives the fold
# ---------------------------------------------------------------------------

def _fold_block(monkeypatch, mid_messages: list[Message]) -> str:
    """Compact a history whose dropped middle is ``mid_messages`` and
    return the compacted block's text."""
    _trip_compaction_thresholds(monkeypatch, keep_recent=2)
    h = _make_handler_for_compact()
    # Ballast pair: compaction has a hard 1000-token minimum trigger
    # (`trigger_at = max(1_000, ...)`); the digest fixtures themselves
    # stay small so assertions read cleanly.
    ballast = [
        _assistant_with_tool("file_read", {"path": "/ws/ballast.py"}, "bal1"),
        _tool_result("bal1", "BALLAST " + "q " * 1500),
    ]
    messages = [*_base_messages(), *ballast, *mid_messages, *_tail()]
    compacted, _b, _a = h._maybe_compact_messages(messages)
    assert compacted
    return messages[2].content


def test_shell_command_survives_in_digest(monkeypatch):
    """shell_exec folds must carry the command string, not just the name."""
    block = _fold_block(monkeypatch, [
        _assistant_with_tool(
            "shell_exec", {"command": "pytest tests/test_foo.py -x -q"}, "s1",
        ),
        _tool_result("s1", "1 passed " + "z" * 200),
        _assistant_with_tool("shell_exec", {"command": "ls"}, "s2"),
        _tool_result("s2", "a.py"),
    ])
    assert "called shell_exec" in block
    assert "pytest tests/test_foo.py -x -q" in block


def test_code_edit_old_new_preview_survives(monkeypatch):
    """code_edit folds must carry path + an old->new preview."""
    block = _fold_block(monkeypatch, [
        _assistant_with_tool(
            "code_edit",
            {"path": "/ws/a.py", "old_string": "x = 1", "new_string": "x = 2"},
            "e1",
        ),
        _tool_result("e1", "edited ok " + "c" * 300),
        _assistant_with_tool("file_read", {"path": "/ws/b.py"}, "r1"),
        _tool_result("r1", "b contents"),
    ])
    assert "/ws/a.py" in block
    assert "x = 1 -> x = 2" in block


def test_giant_edit_args_get_line_count_shape(monkeypatch):
    """Huge old/new strings fold to a bounded +N/-N shape, and the
    rendered line stays under the digest cap."""
    old = "\n".join(f"old line {i}" for i in range(40))   # 40 lines
    new = "\n".join(f"new line {i}" for i in range(55))   # 55 lines
    assert len(old) + len(new) > 400
    block = _fold_block(monkeypatch, [
        _assistant_with_tool(
            "code_edit",
            {"path": "/ws/big.py", "old_string": old, "new_string": new},
            "e1",
        ),
        _tool_result("e1", "edited ok " + "c" * 300),
        _assistant_with_tool("shell_exec", {"command": "true"}, "s1"),
        _tool_result("s1", "ok"),
    ])
    line = next(
        ln for ln in block.splitlines()
        if "called code_edit" in ln
    )
    assert "/ws/big.py (-40/+55 lines)" in line
    assert "old line" not in line          # raw content did not leak
    # narration shape + name + bracketed digest, bounded
    assert len(line) < 400


def test_file_write_and_batch_digests(monkeypatch):
    """file_write carries path+length+head; batch carries count+paths;
    every digest respects the ~300-char digest cap."""
    from augmentum.modes.coder.handler import _DIGEST_CAP
    body = "print('hello world')\n" * 30
    block = _fold_block(monkeypatch, [
        _assistant_with_tool(
            "file_write", {"path": "/ws/new.py", "content": body}, "w1",
        ),
        _tool_result("w1", "written " + "c" * 300),
        _assistant_with_tool(
            "code_edit_batch",
            {"edits": [
                {"path": "/ws/one.py", "old_string": "a" * 500, "new_string": "b"},
                {"path": "/ws/two.py", "old_string": "c", "new_string": "d"},
            ]},
            "b1",
        ),
        _tool_result("b1", "batch ok"),
    ])
    assert f"/ws/new.py ({len(body)} chars)" in block
    assert "print('hello world')" in block
    assert "2 edits: /ws/one.py, /ws/two.py" in block
    for ln in block.splitlines():
        if "called " in ln and "[" in ln:
            digest = ln.rsplit("[", 1)[1]
            assert len(digest) <= _DIGEST_CAP + 1


def test_narration_shape_preserved_with_digest(monkeypatch):
    """The 2026-07-09 'A: <narration> — called <names>' shape must not
    regress; the digest appends after it in brackets."""
    m = _assistant_with_tool(
        "shell_exec", {"command": "npm test"}, "s1",
    )
    m.content = "Running the test suite to confirm the fix"
    block = _fold_block(monkeypatch, [
        m,
        _tool_result("s1", "all green " + "z" * 200),
        _assistant_with_tool("file_read", {"path": "/ws/x.py"}, "r1"),
        _tool_result("r1", "x"),
    ])
    assert (
        "A: Running the test suite to confirm the fix — called shell_exec "
        "[$ npm test]"
    ) in block
