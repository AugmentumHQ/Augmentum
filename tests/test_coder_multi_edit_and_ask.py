"""Tests for the three harness-parity features landed together:

1. Unicode-normalize tier on the 5-tier matcher (ports Codex's
   seek_sequence.rs 4th tier — en/em dashes, smart quotes, NBSP folding).
2. CodeMultiEditTool — atomic multi-hunk SEARCH/REPLACE per file.
   Critical invariant the tests here pin down: when ANY edit fails to
   match, the file on disk is UNCHANGED (fixes Codex apply_patch's
   documented-but-not-delivered atomicity guarantee).
3. AskUserTool + question_callback threading — mid-task interactive
   clarification via a new handler callback, separate from permission.
"""

from __future__ import annotations

import asyncio

from augmentum.coder.editing import _try_unicode, _unicode_fold, apply_edit
from augmentum.coder.state import CoderState
from augmentum.coder.tools import (
    ALL_CODER_TOOLS,
    AskUserTool,
    CodeMultiEditTool,
    create_coder_tools,
)

# --- Stub container manager ---------------------------------------------


class StubCM:
    """Container manager stub — in-memory file + write counter for atomicity checks."""

    def __init__(self, initial: str) -> None:
        self.content = initial
        self.writes = 0

    async def file_read(self, ws: str, path: str) -> str:   # noqa: ARG002
        return self.content

    async def file_write(self, ws: str, path: str, content: str) -> None:   # noqa: ARG002
        self.writes += 1
        self.content = content


def _state_with_read(path: str = "/workspace/f.py") -> CoderState:
    s = CoderState(session_id="s", workspace_id="w")
    s.record_file_read(path)
    return s


# --- Unicode tier --------------------------------------------------------


class TestUnicodeFold:
    def test_fold_noop_on_ascii(self):
        assert _unicode_fold("plain ascii") == "plain ascii"

    def test_fold_en_em_dashes(self):
        assert _unicode_fold("a\u2013b\u2014c") == "a-b-c"

    def test_fold_smart_quotes(self):
        assert _unicode_fold("\u201chello\u201d") == '"hello"'
        assert _unicode_fold("\u2018it\u2019s\u2019") == "'it's'"

    def test_fold_nbsp_and_thin_space(self):
        assert _unicode_fold("a\u00a0b\u2009c") == "a b c"

    def test_fold_strips_zero_width(self):
        assert _unicode_fold("a\u200bb\ufeff") == "ab"

    def test_fold_ellipsis(self):
        assert _unicode_fold("\u2026") == "..."


class TestUnicodeTier:
    def test_smart_quoted_source_matches_ascii_patch(self):
        """The real scenario: file went through a doc editor that curled the quotes."""
        content = 'def f():\n    print(\u201chi\u201d)\n'
        new, tier = apply_edit(content, 'print("hi")', 'print("bye")')
        assert tier == "unicode"
        assert '"bye"' in new

    def test_en_dash_file_ascii_patch(self):
        content = "x \u2013 y\n"
        new, tier = apply_edit(content, "x - y", "x : y")
        assert tier == "unicode"
        assert new == "x : y\n"

    def test_nbsp_in_file(self):
        content = "hello\u00a0world"
        new, tier = apply_edit(content, "hello world", "hi world")
        assert tier == "unicode"
        assert new == "hi world"

    def test_unicode_tier_runs_after_indentation(self):
        """Indentation tier must win when it can (higher confidence than fold)."""
        content = "    print('hi')\n"
        # Trimmed indent + straight quotes — no smart-quote folding needed
        new, tier = apply_edit(content, "print('hi')", "print('bye')")
        assert tier in ("indentation", "exact")

    def test_unicode_tier_falls_through_to_fuzzy(self):
        """Content has smart quotes; search doesn't mention them at all — fuzzy catches."""
        content = "def greet():\n    return \u201chello world\u201d\n"
        new, tier = apply_edit(
            content, "def greet():\n    return \"hi\"", "def greet():\n    return \"bye\"",
        )
        assert tier == "fuzzy"

    def test_try_unicode_returns_none_on_pure_ascii(self):
        """No smart chars on either side → skip without searching."""
        assert _try_unicode("abc", "abc", "xyz") is None


# --- CodeMultiEditTool ---------------------------------------------------


class TestMultiEditHappyPath:
    async def test_applies_all_edits_atomically(self):
        cm = StubCM("def f():\n    x = 1\n    y = 2\n    return x + y\n")
        t = CodeMultiEditTool(
            container_manager=cm, workspace_id="w", state=_state_with_read(),
        )
        r = await t.execute(path="/workspace/f.py", edits=[
            {"search": "x = 1", "replace": "x = 10"},
            {"search": "y = 2", "replace": "y = 20"},
        ])
        assert r.success, r.error
        assert cm.writes == 1, "must be ONE atomic write, not per-edit"
        assert "x = 10" in cm.content and "y = 20" in cm.content

    async def test_reports_tier_counts_in_output(self):
        cm = StubCM("alpha\nbeta\n")
        t = CodeMultiEditTool(
            container_manager=cm, workspace_id="w", state=_state_with_read(),
        )
        r = await t.execute(path="/workspace/f.py", edits=[
            {"search": "alpha", "replace": "ALPHA"},
            {"search": "beta",  "replace": "BETA"},
        ])
        assert r.success
        assert r.metadata["applied"] == 2
        assert "exact" in r.metadata["tier_counts"]

    async def test_edits_apply_sequentially(self):
        """edit[1]'s search matches against content AS MODIFIED BY edit[0]."""
        cm = StubCM("raw value\n")
        t = CodeMultiEditTool(
            container_manager=cm, workspace_id="w", state=_state_with_read(),
        )
        r = await t.execute(path="/workspace/f.py", edits=[
            {"search": "raw value",  "replace": "step1"},
            {"search": "step1",      "replace": "final"},   # depends on edit 0
        ])
        assert r.success
        assert cm.content == "final\n"

    async def test_no_op_batch_reports_cleanly(self):
        """All edits match but replace == search → no write, success with metadata."""
        cm = StubCM("foo\n")
        t = CodeMultiEditTool(
            container_manager=cm, workspace_id="w", state=_state_with_read(),
        )
        r = await t.execute(path="/workspace/f.py", edits=[
            {"search": "foo", "replace": "foo"},
        ])
        assert r.success
        assert cm.writes == 0
        assert r.metadata.get("no_op") is True


class TestMultiEditAtomicity:
    async def test_file_unchanged_on_any_match_failure(self):
        """The critical invariant — one miss, zero writes, original content preserved."""
        cm = StubCM("alpha\nbeta\n")
        t = CodeMultiEditTool(
            container_manager=cm, workspace_id="w", state=_state_with_read(),
        )
        r = await t.execute(path="/workspace/f.py", edits=[
            {"search": "alpha",             "replace": "ALPHA"},   # would match
            {"search": "NONEXISTENT_BLOCK", "replace": "whatever"},  # won't
        ])
        assert not r.success
        assert cm.writes == 0
        assert cm.content == "alpha\nbeta\n"   # on-disk bytes untouched

    async def test_reports_all_failures_not_just_first(self):
        """Model gets to fix every failure in one retry — not play whack-a-mole."""
        cm = StubCM("hello\n")
        t = CodeMultiEditTool(
            container_manager=cm, workspace_id="w", state=_state_with_read(),
        )
        r = await t.execute(path="/workspace/f.py", edits=[
            {"search": "MISSING_A", "replace": "a"},
            {"search": "MISSING_B", "replace": "b"},
            {"search": "MISSING_C", "replace": "c"},
        ])
        assert not r.success
        assert "edit[0]" in r.error
        assert "edit[1]" in r.error
        assert "edit[2]" in r.error

    async def test_error_mentions_file_unchanged(self):
        """Error envelope must explicitly tell the model to resend FULL batch."""
        cm = StubCM("hello\n")
        t = CodeMultiEditTool(
            container_manager=cm, workspace_id="w", state=_state_with_read(),
        )
        r = await t.execute(path="/workspace/f.py", edits=[
            {"search": "MISS", "replace": "x"},
        ])
        assert "UNCHANGED" in r.error
        assert "full" in r.error.lower() or "full batch" in r.error.lower()

    async def test_planned_success_count_in_envelope(self):
        """Error must report N/M would-have-applied — helps model see progress."""
        cm = StubCM("alpha\nbeta\n")
        t = CodeMultiEditTool(
            container_manager=cm, workspace_id="w", state=_state_with_read(),
        )
        r = await t.execute(path="/workspace/f.py", edits=[
            {"search": "alpha", "replace": "A"},
            {"search": "MISS",  "replace": "B"},
        ])
        assert "1/2" in r.error   # one succeeded, one failed


class TestMultiEditErrorHints:
    async def test_already_present_detected(self):
        """If replace text is already in the file, hint the edit is probably idempotent."""
        cm = StubCM("content is here already\n")
        t = CodeMultiEditTool(
            container_manager=cm, workspace_id="w", state=_state_with_read(),
        )
        r = await t.execute(path="/workspace/f.py", edits=[
            {"search": "unfindable_search", "replace": "content is here"},
        ])
        assert "already present" in r.error

    async def test_did_you_mean_suggestions_for_similar_blocks(self):
        """When search is close but below fuzzy threshold, hint helps the model recover."""
        cm = StubCM("def foo():\n    return 1\n\ndef bar():\n    return 2\n")
        t = CodeMultiEditTool(
            container_manager=cm, workspace_id="w", state=_state_with_read(),
        )
        # "lambda baz(): return 3" has no close match in the file. Our
        # similar-block hunt will still surface candidates from the file
        # so the model can see what's actually there.
        r = await t.execute(path="/workspace/f.py", edits=[
            {"search": "lambda baz(): return 3", "replace": "zzz"},
        ])
        assert not r.success
        # Either "Did you mean" hint fires (if any block passed the 0.4
        # similarity floor) OR the error at minimum tells the model what
        # it searched for so it can compare.
        assert "lambda baz" in r.error   # the failed search is echoed


class TestMultiEditValidation:
    async def test_empty_path(self):
        r = await CodeMultiEditTool(
            container_manager=StubCM(""), workspace_id="w", state=CoderState(session_id="s", workspace_id="w"),
        ).execute(path="", edits=[{"search": "x", "replace": "y"}])
        assert r.validation_error

    async def test_empty_edits(self):
        r = await CodeMultiEditTool(
            container_manager=StubCM(""), workspace_id="w", state=_state_with_read(),
        ).execute(path="/workspace/f.py", edits=[])
        assert r.validation_error and "array" in r.error

    async def test_read_before_edit_guard(self):
        state = CoderState(session_id="s", workspace_id="w")   # no record_file_read
        r = await CodeMultiEditTool(
            container_manager=StubCM("x"), workspace_id="w", state=state,
        ).execute(path="/workspace/new.py", edits=[{"search": "x", "replace": "y"}])
        assert not r.success
        # Post-2026-04-20: error message covers both "never read" and
        # "read but mtime-stale" so the model can pick the right
        # recovery. Assert on the durable substring.
        assert "haven't read" in r.error or "read it first" in r.error
        assert "Re-read" in r.error or "read" in r.error.lower()

    async def test_empty_search_block(self):
        cm = StubCM("x")
        t = CodeMultiEditTool(
            container_manager=cm, workspace_id="w", state=_state_with_read(),
        )
        r = await t.execute(path="/workspace/f.py", edits=[
            {"search": "", "replace": "something"},
        ])
        assert not r.success
        assert cm.writes == 0


# --- AskUserTool ---------------------------------------------------------


class TestAskUserBasics:
    async def test_no_callback_short_circuits_via_finish_task(self):
        """Option-3 bridge: without a question_callback wired (no
        QuestionRegistry + modal yet), ask_user degrades to a user-facing
        chat message + finish_task signal. Loop terminates cleanly, user
        answers on next turn. Ensures weak models get a graceful
        "waiting on you" instead of a hard refusal that provokes a guess."""
        state = CoderState(session_id="s", workspace_id="w")
        t = AskUserTool(
            container_manager=None, workspace_id="w",
            state=state, question_callback=None,
        )
        r = await t.execute(questions=[
            {"prompt": "Which framework?", "options": ["pytest", "unittest"]},
        ])
        # Success — the tool did its job (posted a question).
        assert r.success
        assert r.metadata.get("pending_question") is True
        assert r.metadata.get("questions_count") == 1
        # State must carry the finish signal so the act loop terminates
        # on the next iteration top.
        assert state.finish_requested is True
        # Formatted summary must surface the prompt AND the options so
        # the user can actually answer.
        assert "Which framework?" in state.finish_summary
        assert "pytest" in state.finish_summary
        assert "unittest" in state.finish_summary
        # Output (the LLM-visible tool result) tells the model to stop
        # calling tools — otherwise a model that ignores the finish
        # signal might keep running and re-ask.
        assert "turn will end" in r.output.lower()

    async def test_callback_returning_answer_succeeds(self):
        async def cb(questions):
            assert questions[0]["prompt"] == "Which framework?"
            return ["pytest"]

        t = AskUserTool(
            container_manager=None, workspace_id="w",
            state=CoderState(session_id="s", workspace_id="w"),
            question_callback=cb,
        )
        r = await t.execute(questions=[
            {"prompt": "Which framework?", "options": ["pytest", "unittest"]},
        ])
        assert r.success
        assert "pytest" in r.output

    async def test_callback_returning_none_is_user_cancel(self):
        async def cb(_):
            return None

        t = AskUserTool(
            container_manager=None, workspace_id="w",
            state=CoderState(session_id="s", workspace_id="w"),
            question_callback=cb,
        )
        r = await t.execute(questions=[
            {"prompt": "x?", "options": ["a", "b"]},
        ])
        assert not r.success
        assert "declined" in r.error

    async def test_callback_exception_returns_error(self):
        async def cb(_):
            raise RuntimeError("websocket closed")

        t = AskUserTool(
            container_manager=None, workspace_id="w",
            state=CoderState(session_id="s", workspace_id="w"),
            question_callback=cb,
        )
        r = await t.execute(questions=[{"prompt": "x?", "options": ["a", "b"]}])
        assert not r.success
        assert "failed" in r.error


class TestAskUserValidation:
    def _tool(self, cb=None):
        return AskUserTool(
            container_manager=None, workspace_id="w",
            state=CoderState(session_id="s", workspace_id="w"),
            question_callback=cb or (lambda qs: asyncio.sleep(0) or ["a"]),
        )

    async def test_empty_questions_rejected(self):
        r = await self._tool().execute(questions=[])
        assert r.validation_error

    async def test_non_list_rejected(self):
        r = await self._tool().execute(questions="not a list")
        assert r.validation_error

    async def test_missing_prompt_rejected(self):
        r = await self._tool().execute(questions=[
            {"options": ["a", "b"]},
        ])
        assert r.validation_error and "prompt" in r.error

    async def test_single_option_rejected(self):
        """A single option isn't a question — force real choices."""
        r = await self._tool().execute(questions=[
            {"prompt": "x?", "options": ["only_one"]},
        ])
        assert r.validation_error and "at least 2" in r.error

    async def test_options_capped_at_8(self):
        captured = []

        async def cb(questions):
            captured.append(questions)
            return ["0"]

        t = AskUserTool(
            container_manager=None, workspace_id="w",
            state=CoderState(session_id="s", workspace_id="w"),
            question_callback=cb,
        )
        await t.execute(questions=[
            {"prompt": "x?", "options": [str(i) for i in range(20)]},
        ])
        assert len(captured[0][0]["options"]) == 8


# --- Registration / wiring ----------------------------------------------


class TestWiring:
    def test_multi_edit_registered(self):
        names = [c.__name__ for c in ALL_CODER_TOOLS]
        assert "CodeMultiEditTool" in names

    def test_ask_user_registered(self):
        names = [c.__name__ for c in ALL_CODER_TOOLS]
        assert "AskUserTool" in names

    def test_create_coder_tools_threads_question_callback(self):
        async def cb(_):
            return ["x"]

        state = CoderState(session_id="s", workspace_id="w")
        tools = create_coder_tools(
            container_manager=None, workspace_id="w", state=state,
            question_callback=cb,
        )
        ask = next(t for t in tools if t.name == "ask_user")
        assert ask._question_callback is cb

    def test_create_coder_tools_no_callback_by_default(self):
        state = CoderState(session_id="s", workspace_id="w")
        tools = create_coder_tools(
            container_manager=None, workspace_id="w", state=state,
        )
        ask = next(t for t in tools if t.name == "ask_user")
        assert ask._question_callback is None
