"""CodeIntelAdoptionTracker — result-time find_symbol/file_outline nudges."""

from __future__ import annotations

from augmentum.coder.code_intel_nudge import (
    _DEFINITION_GREP_RE,
    CodeIntelAdoptionTracker,
)


def _kinds(nudges):
    return [k for k, _ in nudges]


class TestDefinitionGrepDetection:
    def test_definition_patterns_match(self):
        for pat in [
            "def search_index",
            "async def handle",
            "class NarrativeEngine",
            "function renderChart",
            "func main",
            "fn parse_config",
            r"def\s+search_index",
            "^class Foo",
            "interface Props",
            "struct Point",
        ]:
            assert _DEFINITION_GREP_RE.match(pat), pat

    def test_usage_patterns_do_not_match(self):
        for pat in [
            "search_index(",          # bare usage hunt
            "TODO",
            "import os",
            "self.classify",          # 'class' as substring
            "defaults =",             # 'def' as substring
            "error.*timeout",
        ]:
            assert not _DEFINITION_GREP_RE.match(pat), pat


class TestSymbolGrepNudge:
    def test_fires_once_at_threshold(self):
        t = CodeIntelAdoptionTracker(grep_nudge_at=2)
        t.observe("code_grep", {"pattern": "def foo"})
        assert _kinds(t.end_iteration()) == []
        t.observe("code_grep", {"pattern": "class Bar"})
        assert _kinds(t.end_iteration()) == ["symbol_grep_nudge"]
        # Never re-fires.
        t.observe("code_grep", {"pattern": "def baz"})
        assert _kinds(t.end_iteration()) == []

    def test_disarmed_by_find_symbol_use(self):
        t = CodeIntelAdoptionTracker(grep_nudge_at=2)
        t.observe("find_symbol", {"name": "foo"})
        t.observe("code_grep", {"pattern": "def foo"})
        t.observe("code_grep", {"pattern": "def bar"})
        assert _kinds(t.end_iteration()) == []

    def test_usage_greps_never_fire(self):
        t = CodeIntelAdoptionTracker(grep_nudge_at=2)
        for pat in ["foo(", "bar =", "TODO"]:
            t.observe("code_grep", {"pattern": pat})
        assert _kinds(t.end_iteration()) == []


class TestSingleReadStreakNudge:
    def test_fires_after_streak(self):
        t = CodeIntelAdoptionTracker(streak_nudge_at=3)
        for i in range(3):
            t.observe("file_read", {"path": f"/workspace/f{i}.py"})
            nudges = t.end_iteration()
        assert _kinds(nudges) == ["single_read_nudge"]

    def test_batch_read_disarms(self):
        t = CodeIntelAdoptionTracker(streak_nudge_at=3)
        t.observe("file_read", {"paths": ["/a.py", "/b.py"]})
        for i in range(4):
            t.observe("file_read", {"path": f"/f{i}.py"})
            assert _kinds(t.end_iteration()) == []

    def test_file_outline_disarms(self):
        t = CodeIntelAdoptionTracker(streak_nudge_at=3)
        t.observe("file_outline", {"paths": ["/a.py"]})
        for i in range(4):
            t.observe("file_read", {"path": f"/f{i}.py"})
            assert _kinds(t.end_iteration()) == []

    def test_parallel_fanout_resets_streak(self):
        t = CodeIntelAdoptionTracker(streak_nudge_at=3)
        t.observe("file_read", {"path": "/a.py"})
        t.end_iteration()
        t.observe("file_read", {"path": "/b.py"})
        t.end_iteration()
        # Two single reads in ONE iteration = parallel fanout → reset.
        t.observe("file_read", {"path": "/c.py"})
        t.observe("file_read", {"path": "/d.py"})
        t.end_iteration()
        t.observe("file_read", {"path": "/e.py"})
        assert _kinds(t.end_iteration()) == []

    def test_no_read_iterations_are_neutral(self):
        t = CodeIntelAdoptionTracker(streak_nudge_at=2)
        t.observe("file_read", {"path": "/a.py"})
        t.end_iteration()
        t.observe("shell_exec", {"command": "pytest"})
        t.end_iteration()  # neutral — no reset
        t.observe("file_read", {"path": "/b.py"})
        assert _kinds(t.end_iteration()) == ["single_read_nudge"]

    def test_fires_once(self):
        t = CodeIntelAdoptionTracker(streak_nudge_at=2)
        for i in range(5):
            t.observe("file_read", {"path": f"/f{i}.py"})
            nudges = t.end_iteration()
        assert _kinds(nudges) == []


class TestBothIndependent:
    def test_both_can_fire_in_one_turn(self):
        t = CodeIntelAdoptionTracker(grep_nudge_at=1, streak_nudge_at=1)
        t.observe("code_grep", {"pattern": "def foo"})
        t.observe("file_read", {"path": "/a.py"})
        kinds = _kinds(t.end_iteration())
        assert set(kinds) == {"symbol_grep_nudge", "single_read_nudge"}
