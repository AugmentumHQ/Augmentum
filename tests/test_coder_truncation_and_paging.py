"""Tests for the truncation + pagination overhaul.

Covers:
  1. _truncate prepends AND appends the notice — prepend survives
     compaction's 160-char preview cap, which the old trailing-only
     notice didn't.
  2. FileReadTool offset + limit paging: explicit window, next_offset
     hint, past-EOF rejection, read-before-edit guard scoping to
     offset=0 only.
  3. CodeGrepTool / CodeGlobTool line-based caps with truthful metadata:
     matches_shown vs matches_found, explicit truncation notice in
     output.
"""

from __future__ import annotations

from augmentum.coder.state import CoderState
from augmentum.coder.tools import (
    _MAX_OUTPUT_CHARS,
    CodeGlobTool,
    CodeGrepTool,
    FileReadTool,
    _truncate,
)


class StubCM:
    """Minimal container-manager stub: canned file_read + _run_command."""

    def __init__(
        self,
        *,
        file_content: str = "",
        run_output: str = "",
    ) -> None:
        self.file_content = file_content
        self.run_output = run_output

    async def file_read(self, ws, path):   # noqa: ARG002
        return self.file_content

    async def run_command(self, *args, **kwargs):
        return await self._run_command(*args, **kwargs)

    async def _run_command(self, ws, cmd, timeout=None):   # noqa: ARG002
        return self.run_output


def _state():
    return CoderState(session_id="s", workspace_id="w")


# --- #1 _truncate dual-ended notice --------------------------------------


class TestTruncateDualNotice:
    def test_prepends_truncated_header(self):
        huge = "x" * (_MAX_OUTPUT_CHARS + 5000)
        out = _truncate(huge)
        assert out.startswith("[TRUNCATED"), "header missing from start"

    def test_appends_trailing_notice(self):
        huge = "x" * (_MAX_OUTPUT_CHARS + 5000)
        out = _truncate(huge)
        assert "total chars" in out.rsplit("\n", 1)[-1], "trailer missing"

    def test_header_survives_160char_preview(self):
        """Critical: compaction keeps only the first 160 chars of tool_result.
        The truncation signal MUST be in those 160 chars."""
        huge = "body " * 20000   # well over _MAX_OUTPUT_CHARS
        out = _truncate(huge)
        preview = out[:160]
        assert "TRUNCATED" in preview, (
            "truncation signal lost under compaction — "
            "this is the whole reason we prepend"
        )

    def test_noop_under_cap(self):
        small = "short content"
        assert _truncate(small) == small

    def test_reports_correct_total(self):
        huge = "y" * 100_000
        out = _truncate(huge)
        assert "100000" in out


# --- Universal tool_result byte-cap (the 614 KB transport-overflow backstop) --


class TestToolResultHardClamp:
    def test_clamps_oversized_result_under_cap(self):
        from augmentum.modes.coder.handler import (
            _TOOL_RESULT_HARD_CAP_BYTES,
            _clamp_tool_result_bytes,
        )
        # A tool that bypassed its own truncation (like the 614 KB doc fetch).
        huge = "z" * (_TOOL_RESULT_HARD_CAP_BYTES + 200_000)
        out = _clamp_tool_result_bytes(huge, "doc_fetch")
        # Result is safely under the cap (+ a small hint), never over transport.
        assert len(out.encode("utf-8")) <= _TOOL_RESULT_HARD_CAP_BYTES + 600
        assert "TRUNCATED" in out
        assert "code_search" in out, "must steer the model to the bounded recourse"

    def test_noop_under_cap(self):
        from augmentum.modes.coder.handler import _clamp_tool_result_bytes
        small = "fits fine"
        assert _clamp_tool_result_bytes(small, "file_read") == small

    def test_byte_accurate_with_multibyte(self):
        from augmentum.modes.coder.handler import (
            _TOOL_RESULT_HARD_CAP_BYTES,
            _clamp_tool_result_bytes,
        )
        # Multi-byte chars: the cap is on BYTES, so char-count would undershoot.
        huge = "★" * _TOOL_RESULT_HARD_CAP_BYTES  # 3 bytes each → ~3× the cap
        out = _clamp_tool_result_bytes(huge, "code_grep")
        assert len(out.encode("utf-8")) <= _TOOL_RESULT_HARD_CAP_BYTES + 600


# --- #1b _truncate does NOT give misleading file_read offsets ------------


class TestTruncateNoMisleadingOffset:
    """Regression: earlier revisions suggested ``offset=50000`` in the
    truncation header, but ``file_read``'s ``offset`` is a LINE index,
    not a byte count. That advice sent the model to a bogus past-EOF
    line, got rejected, and round-tripped back to ``offset=0`` — loop.
    """

    def test_header_does_not_name_byte_offset_for_file_read(self):
        huge = "x" * (_MAX_OUTPUT_CHARS + 5000)
        out = _truncate(huge)
        preview = out[:400]
        # The buggy copy specifically said ``offset=50000``. Never again.
        assert "offset=50000" not in preview, (
            "truncation header must not name _MAX_OUTPUT_CHARS as a "
            "file_read offset — offset is a LINE index, not a byte count"
        )

    def test_header_points_model_at_files_own_paging_hint(self):
        """Header should instead tell the model to read the tool
        output's built-in paging hint, which IS authoritative."""
        huge = "x" * (_MAX_OUTPUT_CHARS + 5000)
        out = _truncate(huge)
        # Phrasing can evolve; just assert we steer the model to the
        # correct source rather than naming a number.
        assert "own paging hint" in out[:400] or "line offset" in out[:400]


# --- #2 FileReadTool paging ----------------------------------------------


def _big_file(n_lines: int = 5000) -> str:
    return "\n".join(f"line{i}" for i in range(n_lines))


class TestFileReadPaging:
    async def test_default_window_first_2000_lines(self):
        cm = StubCM(file_content=_big_file(5000))
        r = await FileReadTool(
            container_manager=cm, workspace_id="w", state=_state(),
        ).execute(path="/f.py")
        assert r.success
        assert r.metadata["total_lines"] == 5000
        assert r.metadata["lines_shown"] == 2000
        assert r.metadata["next_offset"] == 2000
        assert "line0" in r.output
        assert "line1999" in r.output
        assert "line2000" not in r.output

    async def test_explicit_offset_and_limit(self):
        cm = StubCM(file_content=_big_file(5000))
        r = await FileReadTool(
            container_manager=cm, workspace_id="w", state=_state(),
        ).execute(path="/f.py", offset=2000, limit=500)
        assert r.success
        assert r.metadata["offset"] == 2000
        assert r.metadata["lines_shown"] == 500
        assert r.metadata["next_offset"] == 2500
        # Line numbers in output are 1-based file positions, not window-relative.
        # Padded to width 4, so 4-digit numbers have no leading space.
        assert "2001 | line2000" in r.output
        assert "2500 | line2499" in r.output

    async def test_next_offset_none_when_fully_read(self):
        """Small file → single call reads it all, next_offset is None."""
        cm = StubCM(file_content="only\ntwo\nlines")
        r = await FileReadTool(
            container_manager=cm, workspace_id="w", state=_state(),
        ).execute(path="/f.py")
        assert r.metadata["next_offset"] is None
        assert "next chunk" not in r.output

    async def test_long_lines_clip_by_line_not_mid_line(self):
        """File with very long lines (minified JS, generated SQL) must
        still clip on a line boundary. Without the pre-clip, the byte
        cap would chop mid-line and the paging hint would name an
        offset past what the model actually saw. The clipped window +
        its paging hint must always agree on where the visible content
        ends."""
        # 100 lines of 1000 chars each = 100k chars — well over 50k cap.
        long_line = "A" * 1000
        content = "\n".join(f"{long_line}" for _ in range(100))
        cm = StubCM(file_content=content)
        r = await FileReadTool(
            container_manager=cm, workspace_id="w", state=_state(),
        ).execute(path="/big.js")
        assert r.success
        # We clipped by line, so each line of output is a whole line
        # (not a cut-off one) — find the last line number shown.
        import re
        line_nums = [int(m) for m in re.findall(
            r"\s*(\d+) \| AAAA", r.output,
        )]
        assert line_nums, "no line-numbered output at all"
        last_shown = max(line_nums)
        # Paging hint should name ``offset=last_shown`` so the next
        # call continues at the right line.
        hint_pattern = rf"offset={last_shown}\b"
        assert re.search(hint_pattern, r.output), (
            f"paging hint should name offset={last_shown} (next line "
            f"after the last one shown); output tail was: "
            f"{r.output[-400:]}"
        )

    async def test_past_eof_rejected(self):
        cm = StubCM(file_content=_big_file(100))
        r = await FileReadTool(
            container_manager=cm, workspace_id="w", state=_state(),
        ).execute(path="/f.py", offset=500)
        assert not r.success
        assert r.validation_error
        assert "past end of file" in r.error

    async def test_read_before_edit_guard_only_on_offset_0(self):
        """A partial re-read at offset>0 implies prior read; don't re-arm guard."""
        st = _state()
        cm = StubCM(file_content=_big_file(5000))
        t = FileReadTool(container_manager=cm, workspace_id="w", state=st)

        # Partial read at offset>0 — should NOT set guard
        await t.execute(path="/f.py", offset=1000)
        assert "/f.py" not in st.files_read

        # Fresh read at offset=0 — sets guard
        await t.execute(path="/f.py")
        assert "/f.py" in st.files_read

    async def test_defensive_arg_coercion(self):
        """Weaker models sometimes pass strings or negatives — don't crash."""
        cm = StubCM(file_content="a\nb\nc\n")
        r = await FileReadTool(
            container_manager=cm, workspace_id="w", state=_state(),
        ).execute(path="/f.py", offset="oops", limit="-5")
        assert r.success   # coerced, not rejected

    async def test_next_offset_hint_in_output(self):
        cm = StubCM(file_content=_big_file(3000))
        r = await FileReadTool(
            container_manager=cm, workspace_id="w", state=_state(),
        ).execute(path="/f.py", limit=1000)
        assert "offset=1000" in r.output


# --- #3 CodeGrepTool line-based cap --------------------------------------


class TestCodeGrepLineCap:
    async def test_limits_to_default_200(self):
        """500 matches → first 200 shown + truncation notice + truthful metadata."""
        raw = "\n".join(f"/f{i}.py:1:match" for i in range(500))
        cm = StubCM(run_output=raw)
        r = await CodeGrepTool(
            container_manager=cm, workspace_id="w", state=_state(),
        ).execute(pattern="foo")
        assert r.success
        assert r.metadata["matches_shown"] == 200
        assert r.metadata["matches_found"] == 500
        assert "200 of 500" in r.output
        # Confirm line-based cut — every shown line is complete, no mid-line cut
        for line in r.output.split("\n"):
            if line.startswith("/f"):
                assert ":1:match" in line or line.endswith(":match")

    async def test_explicit_limit(self):
        raw = "\n".join(f"/f{i}.py:1:hit" for i in range(50))
        cm = StubCM(run_output=raw)
        r = await CodeGrepTool(
            container_manager=cm, workspace_id="w", state=_state(),
        ).execute(pattern="foo", limit=20)
        assert r.metadata["matches_shown"] == 20
        assert r.metadata["matches_found"] == 50

    async def test_no_truncation_under_cap(self):
        raw = "/a.py:1:match\n/b.py:2:match"
        cm = StubCM(run_output=raw)
        r = await CodeGrepTool(
            container_manager=cm, workspace_id="w", state=_state(),
        ).execute(pattern="foo")
        assert r.metadata["matches_found"] == 2
        assert "Showing first" not in r.output

    async def test_empty_results(self):
        cm = StubCM(run_output="")
        r = await CodeGrepTool(
            container_manager=cm, workspace_id="w", state=_state(),
        ).execute(pattern="foo")
        assert r.success
        assert r.metadata["matches_found"] == 0
        assert r.metadata["matches_shown"] == 0


# --- #3 CodeGlobTool line-based cap --------------------------------------


class TestCodeGlobLineCap:
    async def test_limits_to_default_200(self):
        raw = "\n".join(f"/path/f{i}.py" for i in range(500))
        cm = StubCM(run_output=raw)
        r = await CodeGlobTool(
            container_manager=cm, workspace_id="w", state=_state(),
        ).execute(pattern="*.py")
        assert r.metadata["files_shown"] == 200
        assert r.metadata["files_found"] == 500
        assert "200 of 500" in r.output

    async def test_explicit_limit(self):
        raw = "\n".join(f"/f{i}.py" for i in range(30))
        cm = StubCM(run_output=raw)
        r = await CodeGlobTool(
            container_manager=cm, workspace_id="w", state=_state(),
        ).execute(pattern="*.py", limit=10)
        assert r.metadata["files_shown"] == 10
        assert r.metadata["files_found"] == 30

    async def test_no_truncation_under_cap(self):
        raw = "/a.py\n/b.py\n/c.py"
        cm = StubCM(run_output=raw)
        r = await CodeGlobTool(
            container_manager=cm, workspace_id="w", state=_state(),
        ).execute(pattern="*.py")
        assert r.metadata["files_found"] == 3
        assert "Showing first" not in r.output

    async def test_backcompat_count_metadata(self):
        """Old callers read metadata['count']; keep it meaning total found."""
        raw = "/a.py\n/b.py"
        cm = StubCM(run_output=raw)
        r = await CodeGlobTool(
            container_manager=cm, workspace_id="w", state=_state(),
        ).execute(pattern="*.py")
        assert r.metadata.get("count") == 2
