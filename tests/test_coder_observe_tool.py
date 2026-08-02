"""Tests for the ``observe`` tool — model-facing surface for the
observation ledger. Exercises validation, source provenance, and the
WorkspaceKernel wrapper methods (``record_observation``,
``read_observations``).

These tests focus on the tool layer's contract — schema enforcement,
clear validation errors that teach the model the shape, error
propagation. Persistence semantics are covered in
``test_coder_observations.py``.

Run: python -m pytest tests/test_coder_observe_tool.py -v
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from augmentum.coder.observations import (
    Observation,
    parse_jsonl,
    serialize_observations,
)
from augmentum.coder.runtime_tools import ObserveTool
from augmentum.coder.state import CoderState
from augmentum.coder.workspace_kernel import (
    OBSERVATIONS_JSONL,
    WorkspaceKernel,
)


# ---------------------------------------------------------------------------
# Fakes — minimal container manager mirroring the surface the
# observations helpers touch.
# ---------------------------------------------------------------------------


def _cm() -> MagicMock:
    cm = MagicMock()
    files: dict[str, str] = {}
    writes: dict[str, str] = {}

    async def _file_read(workspace_id, path):
        if path in files:
            return files[path]
        raise FileNotFoundError(path)

    async def _file_write(workspace_id, path, content):
        writes[path] = content
        files[path] = content

    async def _run_command(workspace_id, cmd, timeout=None):
        return ""

    cm.file_read = AsyncMock(side_effect=_file_read)
    cm.file_write = AsyncMock(side_effect=_file_write)
    cm._run_command = AsyncMock(side_effect=_run_command)
    cm.run_command = cm._run_command
    cm._files = files
    cm._writes = writes
    return cm


def _state() -> CoderState:
    return CoderState(session_id="s", workspace_id="ws")


def _tool(cm=None, state=None) -> ObserveTool:
    return ObserveTool(
        container_manager=cm or _cm(),
        workspace_id="ws",
        state=state or _state(),
    )


# ---------------------------------------------------------------------------
# Validation — clear messages that teach the model the schema
# ---------------------------------------------------------------------------


class TestObserveToolValidation:
    @pytest.mark.asyncio
    async def test_missing_fact_is_validation_error(self):
        tool = _tool()
        result = await tool.execute(category="build", fact="")
        assert not result.success
        assert result.validation_error
        assert "fact" in result.error.lower()

    @pytest.mark.asyncio
    async def test_missing_category_is_validation_error(self):
        tool = _tool()
        result = await tool.execute(category="", fact="some fact")
        assert not result.success
        assert result.validation_error
        assert "category" in result.error.lower()

    @pytest.mark.asyncio
    async def test_invalid_category_lists_valid_options(self):
        tool = _tool()
        result = await tool.execute(category="bogus", fact="x")
        assert not result.success
        assert result.validation_error
        # Error message must enumerate the closed set so the model
        # can self-correct on the retry.
        assert "build" in result.error
        assert "constraint" in result.error
        assert "bogus" in result.error  # echoes back what was passed

    @pytest.mark.asyncio
    async def test_invalid_confidence_is_validation_error(self):
        tool = _tool()
        result = await tool.execute(
            category="build", fact="x", confidence="strong",
        )
        assert not result.success
        assert result.validation_error

    @pytest.mark.asyncio
    async def test_oversized_fact_refused(self):
        """Observations are one-sentence facts. A paragraph-length
        'fact' should split into multiple observations — refuse so
        the model fixes the shape rather than appending bloat."""
        tool = _tool()
        result = await tool.execute(category="build", fact="x" * 1000)
        assert not result.success
        assert result.validation_error
        assert "500" in result.error  # mentions the cap


# ---------------------------------------------------------------------------
# Happy-path persistence
# ---------------------------------------------------------------------------


class TestObserveToolPersist:
    @pytest.mark.asyncio
    async def test_records_observation_to_ledger(self):
        cm = _cm()
        tool = _tool(cm=cm)
        result = await tool.execute(
            category="build", fact="pytest is the test runner",
        )
        assert result.success
        assert "pytest is the test runner" in result.output
        # File written.
        assert OBSERVATIONS_JSONL in cm._writes
        ledger = parse_jsonl(cm._writes[OBSERVATIONS_JSONL])
        assert len(ledger) == 1
        assert ledger[0].fact == "pytest is the test runner"
        assert ledger[0].category == "build"
        assert ledger[0].confidence == "confirmed"

    @pytest.mark.asyncio
    async def test_tentative_confidence_recorded(self):
        cm = _cm()
        tool = _tool(cm=cm)
        result = await tool.execute(
            category="env", fact="auth tokens may be in .env.local",
            confidence="tentative",
        )
        assert result.success
        assert "tentative" in result.output.lower()
        ledger = parse_jsonl(cm._writes[OBSERVATIONS_JSONL])
        assert ledger[0].confidence == "tentative"

    @pytest.mark.asyncio
    async def test_source_includes_tool_call_count(self):
        """Source provenance is load-bearing for ledger audits — the
        ts alone doesn't tell you WHICH turn produced the fact."""
        state = _state()
        state.tool_calls_made = 7
        cm = _cm()
        tool = _tool(cm=cm, state=state)
        await tool.execute(category="build", fact="x")
        ledger = parse_jsonl(cm._writes[OBSERVATIONS_JSONL])
        assert "tool_calls=7" in ledger[0].source

    @pytest.mark.asyncio
    async def test_metadata_carries_observation_shape(self):
        cm = _cm()
        tool = _tool(cm=cm)
        result = await tool.execute(
            category="constraint", fact="node 18 locked",
        )
        meta = result.metadata["observation"]
        assert meta["category"] == "constraint"
        assert meta["fact"] == "node 18 locked"
        assert meta["confidence"] == "confirmed"

    @pytest.mark.asyncio
    async def test_persist_failure_returns_error_result(self):
        cm = _cm()
        cm.file_write = AsyncMock(side_effect=RuntimeError("disk full"))
        tool = _tool(cm=cm)
        result = await tool.execute(category="build", fact="x")
        assert not result.success
        assert "persist" in result.error.lower() or "kernel" in result.error.lower()


# ---------------------------------------------------------------------------
# Kernel wrapper methods — record_observation + read_observations
# ---------------------------------------------------------------------------


class TestKernelObservationMethods:
    @pytest.mark.asyncio
    async def test_record_observation_normalizes_invalid_category(self):
        """Defensive: even if a future caller bypasses the tool's
        validation, the kernel must not write garbage to disk."""
        cm = _cm()
        k = WorkspaceKernel(cm, "ws")
        ok = await k.record_observation(
            category="nonsense", fact="x", source="test",
        )
        assert ok is True
        ledger = parse_jsonl(cm._writes[OBSERVATIONS_JSONL])
        assert ledger[0].category == "other"

    @pytest.mark.asyncio
    async def test_record_observation_refuses_empty_fact(self):
        cm = _cm()
        k = WorkspaceKernel(cm, "ws")
        ok = await k.record_observation(
            category="build", fact="   ", source="test",
        )
        assert ok is False

    @pytest.mark.asyncio
    async def test_read_observations_returns_empty_on_fresh_workspace(self):
        cm = _cm()
        k = WorkspaceKernel(cm, "ws")
        assert await k.read_observations() == []

    @pytest.mark.asyncio
    async def test_read_observations_returns_full_ledger(self):
        cm = _cm()
        # Seed the ledger directly so the test doesn't depend on
        # append_observation succeeding first.
        text = serialize_observations([
            Observation(ts=1.0, category="build", fact="A", source="t1"),
            Observation(ts=2.0, category="env", fact="B", source="t2"),
        ])
        cm._files[OBSERVATIONS_JSONL] = text
        k = WorkspaceKernel(cm, "ws")
        obs = await k.read_observations()
        assert len(obs) == 2
        assert [o.fact for o in obs] == ["A", "B"]

    @pytest.mark.asyncio
    async def test_record_then_read_round_trip(self):
        cm = _cm()
        k = WorkspaceKernel(cm, "ws")
        await k.record_observation(
            category="env", fact="auth in .env.local", source="turn 9",
        )
        obs = await k.read_observations()
        assert len(obs) == 1
        assert obs[0].fact == "auth in .env.local"

    @pytest.mark.asyncio
    async def test_no_container_manager_record_returns_false(self):
        k = WorkspaceKernel(None, "ws")
        ok = await k.record_observation(
            category="env", fact="x", source="test",
        )
        assert ok is False

    @pytest.mark.asyncio
    async def test_no_container_manager_read_returns_empty(self):
        k = WorkspaceKernel(None, "ws")
        assert await k.read_observations() == []


# ---------------------------------------------------------------------------
# Cross-session continuity — the property the whole layer exists for
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cross_session_continuity():
    """Session 1 records facts; session 2 (fresh handler, same workspace)
    sees them. This is the load-bearing property — without it, the
    ledger isn't doing what it's supposed to."""
    cm = _cm()  # shared container = shared filesystem

    # Session 1: a series of observations.
    k1 = WorkspaceKernel(cm, "ws")
    await k1.record_observation(
        category="build", fact="pytest with -x", source="turn 1",
    )
    await k1.record_observation(
        category="env", fact="auth in .env.local", source="turn 3",
    )
    await k1.record_observation(
        category="constraint", fact="node 18 locked",
        source="turn 5", confidence="user_asserted",
    )

    # Session 2: brand new kernel instance, same container.
    k2 = WorkspaceKernel(cm, "ws")
    obs = await k2.read_observations()
    assert len(obs) == 3
    facts = {o.fact for o in obs}
    assert "pytest with -x" in facts
    assert "auth in .env.local" in facts
    assert "node 18 locked" in facts
    # Confidence carried through.
    node_obs = next(o for o in obs if "node 18" in o.fact)
    assert node_obs.confidence == "user_asserted"
