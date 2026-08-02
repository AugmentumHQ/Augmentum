"""Tests for the observation ledger — durable cross-session memory.

Layer 5 of the context kernel state model. Each observation is a
durable fact the agent learned ("pytest is the test runner", "auth
tokens live in /workspace/.env.local") persisted append-only to
``/workspace/.augmentum/observations.jsonl``.

Test surface:

  1. Observation dataclass — serialization round-trip, defensive
     parsing of malformed input, closed-set validation.
  2. JSONL ledger IO — parse_jsonl tolerant of bad lines,
     serialize_observations idempotent.
  3. Dedup-by-(category, fact) — same fact recorded twice updates
     timestamp without producing two entries.
  4. Query helpers — filter by category, confidence floor, limit.
  5. Container IO — read_ledger / append_observation against a fake
     container manager; failure modes preserved.
  6. Render — budget-respecting prompt block; priority categories
     surface first.

Tool-level tests live in ``test_coder_observations_tool.py`` (separate
file because they exercise the runtime_tools wiring).

Run: python -m pytest tests/test_coder_observations.py -v
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from augmentum.coder.observations import (
    CATEGORIES,
    CONFIDENCES,
    Observation,
    append_observation,
    merge_observation,
    parse_jsonl,
    query_observations,
    read_ledger,
    render_for_prompt,
    serialize_observations,
)

# ---------------------------------------------------------------------------
# Observation dataclass + parsing
# ---------------------------------------------------------------------------


class TestObservation:
    def test_to_jsonl_line_is_single_line(self):
        obs = Observation(
            ts=1.0, category="env", fact="auth in .env",
            source="turn 1", confidence="confirmed",
        )
        line = obs.to_jsonl_line()
        assert "\n" not in line
        # Round-trip via JSON parse confirms shape.
        parsed = json.loads(line)
        assert parsed["category"] == "env"
        assert parsed["fact"] == "auth in .env"

    def test_from_dict_validates_category(self):
        obs = Observation.from_dict({
            "ts": 1.0, "category": "nonsense",
            "fact": "x", "source": "turn 1",
        })
        # Invalid category rewrites to "other" rather than dropping
        # the observation entirely.
        assert obs is not None
        assert obs.category == "other"
        assert obs.fact == "x"

    def test_from_dict_validates_confidence(self):
        obs = Observation.from_dict({
            "ts": 1.0, "category": "env",
            "fact": "x", "source": "turn 1",
            "confidence": "guessing",
        })
        assert obs is not None
        assert obs.confidence == "confirmed"

    def test_from_dict_accepts_minimal_input(self):
        """Missing optional fields default to safe values rather than
        raising — robustness for partially-written or older-schema
        entries."""
        obs = Observation.from_dict({"fact": "minimal"})
        assert obs is not None
        assert obs.fact == "minimal"
        assert obs.category == "other"
        assert obs.confidence == "confirmed"

    def test_from_dict_returns_none_on_garbage(self):
        # ts=not_a_number → ValueError → returns None
        obs = Observation.from_dict({
            "ts": "not-a-number-and-not-a-float",
            "fact": "x",
        })
        assert obs is None


# ---------------------------------------------------------------------------
# JSONL serialize / parse
# ---------------------------------------------------------------------------


class TestJsonlRoundTrip:
    def test_empty_list_serializes_to_empty(self):
        assert serialize_observations([]) == ""

    def test_multiple_observations_round_trip(self):
        obs_list = [
            Observation(ts=1.0, category="build", fact="A", source="t1"),
            Observation(ts=2.0, category="env", fact="B", source="t2", confidence="tentative"),
            Observation(ts=3.0, category="api", fact="C", source="t3", confidence="user_asserted"),
        ]
        text = serialize_observations(obs_list)
        # Each on its own line + trailing newline.
        assert text.count("\n") == 3
        parsed = parse_jsonl(text)
        assert len(parsed) == 3
        assert [o.fact for o in parsed] == ["A", "B", "C"]
        assert parsed[1].confidence == "tentative"
        assert parsed[2].confidence == "user_asserted"

    def test_malformed_line_skipped_without_killing_ledger(self):
        text = (
            '{"ts": 1.0, "category": "build", "fact": "good", "source": "t1"}\n'
            'not json at all\n'
            '{"ts": 2.0, "category": "env", "fact": "also good", "source": "t2"}\n'
        )
        parsed = parse_jsonl(text)
        assert len(parsed) == 2
        assert [o.fact for o in parsed] == ["good", "also good"]

    def test_empty_lines_skipped(self):
        text = (
            '\n'
            '{"ts": 1.0, "category": "build", "fact": "x", "source": "t1"}\n'
            '\n\n'
        )
        assert len(parse_jsonl(text)) == 1

    def test_categories_closed_set(self):
        # The exposed CATEGORIES is what the tool validates against.
        # If we add a category to the dataclass without adding it to
        # the set, that drift fails this test.
        assert "build" in CATEGORIES
        assert "constraint" in CATEGORIES
        assert "other" in CATEGORIES
        # Pin the size so an accidental addition surfaces in review.
        assert len(CATEGORIES) == 10

    def test_confidences_closed_set(self):
        assert frozenset({
            "tentative", "confirmed", "user_asserted",
        }) == CONFIDENCES


# ---------------------------------------------------------------------------
# Dedup-by-(category, fact)
# ---------------------------------------------------------------------------


class TestMergeObservation:
    def test_new_fact_appended(self):
        existing = [
            Observation(ts=1.0, category="build", fact="A", source="t1"),
        ]
        merged = merge_observation(
            existing,
            Observation(ts=2.0, category="env", fact="B", source="t2"),
        )
        assert len(merged) == 2
        assert merged[-1].fact == "B"

    def test_same_category_same_fact_updates_timestamp(self):
        existing = [
            Observation(
                ts=1.0, category="build", fact="pytest is the runner",
                source="t1", confidence="tentative",
            ),
        ]
        merged = merge_observation(
            existing,
            Observation(
                ts=5.0, category="build", fact="pytest is the runner",
                source="t5", confidence="confirmed",
            ),
        )
        assert len(merged) == 1
        # Latest write wins — new ts, new confidence.
        assert merged[0].ts == 5.0
        assert merged[0].confidence == "confirmed"
        assert merged[0].source == "t5"

    def test_dedup_is_case_insensitive(self):
        existing = [
            Observation(
                ts=1.0, category="build", fact="Pytest IS the Runner",
                source="t1",
            ),
        ]
        merged = merge_observation(
            existing,
            Observation(
                ts=5.0, category="build", fact="pytest is the runner",
                source="t5",
            ),
        )
        assert len(merged) == 1

    def test_same_fact_different_category_kept_separately(self):
        """Two different categories with the same fact text are NOT the
        same observation. Real example: "production" might be both a
        deploy fact and a constraint."""
        existing = [
            Observation(ts=1.0, category="build", fact="X", source="t1"),
        ]
        merged = merge_observation(
            existing,
            Observation(ts=2.0, category="env", fact="X", source="t2"),
        )
        assert len(merged) == 2

    def test_dedup_preserves_chronological_order(self):
        """When a fact is refreshed, the refreshed entry goes to the
        END of the list — preserving "newest at the bottom" ordering
        that callers rely on for "latest N" queries."""
        existing = [
            Observation(ts=1.0, category="build", fact="A", source="t1"),
            Observation(ts=2.0, category="env", fact="B", source="t2"),
            Observation(ts=3.0, category="api", fact="C", source="t3"),
        ]
        # Refresh "A" — it should move from position 0 to position -1.
        merged = merge_observation(
            existing,
            Observation(ts=10.0, category="build", fact="A", source="t10"),
        )
        assert len(merged) == 3
        assert [o.fact for o in merged] == ["B", "C", "A"]
        assert merged[-1].ts == 10.0


# ---------------------------------------------------------------------------
# query_observations
# ---------------------------------------------------------------------------


class TestQueryObservations:
    @pytest.fixture
    def obs(self):
        return [
            Observation(ts=10.0, category="build", fact="newest build", source="t1"),
            Observation(ts=8.0, category="env", fact="env-2", source="t2", confidence="tentative"),
            Observation(ts=5.0, category="constraint", fact="node 18", source="t3", confidence="user_asserted"),
            Observation(ts=3.0, category="build", fact="oldest build", source="t4"),
            Observation(ts=1.0, category="other", fact="random", source="t5"),
        ]

    def test_no_filter_returns_all_ts_descending(self, obs):
        result = query_observations(obs)
        assert [o.fact for o in result] == [
            "newest build", "env-2", "node 18",
            "oldest build", "random",
        ]

    def test_filter_by_categories(self, obs):
        result = query_observations(obs, categories=["build"])
        assert len(result) == 2
        assert all(o.category == "build" for o in result)

    def test_limit_applies_after_sort(self, obs):
        result = query_observations(obs, limit=2)
        assert len(result) == 2
        # Most recent two by ts.
        assert [o.fact for o in result] == ["newest build", "env-2"]

    def test_min_confidence_excludes_tentative(self, obs):
        result = query_observations(obs, min_confidence="confirmed")
        # env-2 is tentative — excluded.
        assert "env-2" not in [o.fact for o in result]
        # user_asserted node-18 survives.
        assert "node 18" in [o.fact for o in result]

    def test_min_confidence_user_asserted_only(self, obs):
        result = query_observations(obs, min_confidence="user_asserted")
        assert len(result) == 1
        assert result[0].fact == "node 18"


# ---------------------------------------------------------------------------
# Container IO — read_ledger / append_observation
# ---------------------------------------------------------------------------


def _cm(files: dict[str, str] | None = None, fail: bool = False) -> MagicMock:
    cm = MagicMock()
    files = dict(files or {})
    writes: dict[str, str] = {}

    async def _file_read(workspace_id: str, path: str) -> str:
        if fail:
            raise RuntimeError("container down")
        if path in files:
            return files[path]
        raise FileNotFoundError(path)

    async def _file_write(workspace_id: str, path: str, content: str) -> None:
        if fail:
            raise RuntimeError("container down")
        writes[path] = content
        files[path] = content

    async def _run_command(workspace_id: str, cmd, timeout=None):
        if fail:
            raise RuntimeError("container down")
        return ""

    cm.file_read = AsyncMock(side_effect=_file_read)
    cm.file_write = AsyncMock(side_effect=_file_write)
    cm._run_command = AsyncMock(side_effect=_run_command)
    cm.run_command = cm._run_command
    cm._writes = writes  # exposed for assertions
    cm._files = files
    return cm


class TestReadLedger:
    @pytest.mark.asyncio
    async def test_missing_file_returns_empty(self):
        cm = _cm()
        assert await read_ledger(cm, "ws") == []

    @pytest.mark.asyncio
    async def test_present_file_parses(self):
        text = serialize_observations([
            Observation(ts=1.0, category="build", fact="x", source="t1"),
            Observation(ts=2.0, category="env", fact="y", source="t2"),
        ])
        cm = _cm(files={"/workspace/.augmentum/observations.jsonl": text})
        result = await read_ledger(cm, "ws")
        assert len(result) == 2
        assert [o.fact for o in result] == ["x", "y"]

    @pytest.mark.asyncio
    async def test_container_failure_returns_empty(self):
        cm = _cm(fail=True)
        assert await read_ledger(cm, "ws") == []


class TestAppendObservation:
    @pytest.mark.asyncio
    async def test_first_observation_creates_file(self):
        cm = _cm()
        ok = await append_observation(
            cm, "ws",
            Observation(ts=1.0, category="build", fact="x", source="t1"),
        )
        assert ok is True
        path = "/workspace/.augmentum/observations.jsonl"
        assert path in cm._writes
        # Single-line file.
        assert cm._writes[path].count("\n") == 1

    @pytest.mark.asyncio
    async def test_second_observation_preserves_first(self):
        cm = _cm()
        await append_observation(
            cm, "ws",
            Observation(ts=1.0, category="build", fact="A", source="t1"),
        )
        await append_observation(
            cm, "ws",
            Observation(ts=2.0, category="env", fact="B", source="t2"),
        )
        path = "/workspace/.augmentum/observations.jsonl"
        ledger = parse_jsonl(cm._writes[path])
        assert len(ledger) == 2
        assert [o.fact for o in ledger] == ["A", "B"]

    @pytest.mark.asyncio
    async def test_duplicate_fact_dedups_in_persisted_ledger(self):
        cm = _cm()
        await append_observation(
            cm, "ws",
            Observation(ts=1.0, category="build", fact="X", source="t1"),
        )
        await append_observation(
            cm, "ws",
            Observation(ts=5.0, category="build", fact="X", source="t5"),
        )
        ledger = parse_jsonl(
            cm._writes["/workspace/.augmentum/observations.jsonl"]
        )
        assert len(ledger) == 1
        assert ledger[0].ts == 5.0

    @pytest.mark.asyncio
    async def test_container_failure_returns_false(self):
        cm = _cm(fail=True)
        ok = await append_observation(
            cm, "ws",
            Observation(ts=1.0, category="build", fact="X", source="t1"),
        )
        assert ok is False

    @pytest.mark.asyncio
    async def test_no_container_manager_returns_false(self):
        ok = await append_observation(
            None, "ws",
            Observation(ts=1.0, category="build", fact="X", source="t1"),
        )
        assert ok is False


# ---------------------------------------------------------------------------
# render_for_prompt
# ---------------------------------------------------------------------------


class TestRenderForPrompt:
    def test_empty_returns_empty(self):
        assert render_for_prompt([]) == ""

    def test_priority_categories_render_first(self):
        obs = [
            Observation(ts=10.0, category="other", fact="other-recent", source="t1"),
            Observation(ts=5.0, category="constraint", fact="node 18 locked", source="t2"),
            Observation(ts=3.0, category="gotcha", fact="watch out for X", source="t3"),
        ]
        block = render_for_prompt(obs, budget_chars=500)
        # constraints + gotchas surface BEFORE other, even though other
        # has a more recent ts.
        idx_constraint = block.index("node 18 locked")
        idx_gotcha = block.index("watch out for X")
        idx_other = block.index("other-recent")
        assert idx_constraint < idx_other
        assert idx_gotcha < idx_other

    def test_budget_respected(self):
        obs = [
            Observation(ts=float(i), category="other", fact=f"observation number {i:03d} text", source=f"t{i}")
            for i in range(50)
        ]
        block = render_for_prompt(obs, budget_chars=300)
        # Hard upper bound — small slack for the header/footer tags.
        assert len(block) <= 320
        # Should still contain at least the wrapper tags.
        assert "<observations>" in block
        assert "</observations>" in block

    def test_zero_budget_returns_empty(self):
        obs = [Observation(ts=1.0, category="build", fact="x", source="t1")]
        assert render_for_prompt(obs, budget_chars=0) == ""
