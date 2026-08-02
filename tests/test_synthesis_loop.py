"""Tests for the synthesis loop."""
from __future__ import annotations

import pytest

from augmentum.tools.constraint_schema import AppSpec, Element, Constraint
from augmentum.tools.synthesis_loop import SynthesisLoop, SynthesisResult


def _make_simple_spec():
    return AppSpec(
        name="Counter",
        state_schema={"count": "number"},
        elements=[
            Element(id="app", tag="div", role="container"),
            Element(id="count-display", tag="span", role="display"),
            Element(id="increment-btn", tag="button", role="action", label="Increment"),
        ],
        constraints=[
            Constraint(
                id="c1", behavior="render", description="App renders",
                type="structural", depends_on=[],
            ),
            Constraint(
                id="c2", behavior="increment",
                description="Clicking #increment-btn increases count",
                type="interaction",
                trigger={"event": "click", "target": "#increment-btn"},
                expected={"new_element": ".updated"},
                depends_on=["c1"],
            ),
        ],
    )


class TestSynthesisLoop:
    @pytest.mark.asyncio
    async def test_loop_completes_with_mock_llm(self):
        call_count = 0

        async def mock_llm(messages, **kw):
            nonlocal call_count
            call_count += 1
            return "(function() { 'use strict'; document.getElementById('count-display').textContent = '0'; })();"

        loop = SynthesisLoop(call_llm=mock_llm)
        result = await loop.run(_make_simple_spec())
        assert isinstance(result, SynthesisResult)
        assert result.skeleton != ""
        assert result.css != ""
        assert len(result.constraint_results) == 2

    @pytest.mark.asyncio
    async def test_loop_reports_failed_constraints(self):
        async def bad_llm(messages, **kw):
            return "this is not valid javascript }{}{}"

        loop = SynthesisLoop(call_llm=bad_llm, max_attempts=1)
        result = await loop.run(_make_simple_spec())
        assert isinstance(result, SynthesisResult)
        failed = [cr for cr in result.constraint_results if cr["status"] == "failed"]
        assert len(failed) > 0

    @pytest.mark.asyncio
    async def test_loop_respects_max_attempts(self):
        attempts = []

        async def counting_llm(messages, **kw):
            attempts.append(1)
            return "broken {"

        loop = SynthesisLoop(call_llm=counting_llm, max_attempts=2)
        result = await loop.run(_make_simple_spec())
        assert len(attempts) <= 4

    @pytest.mark.asyncio
    async def test_progress_callback_fires(self):
        progress_log = []

        async def mock_llm(messages, **kw):
            return "(function() { document.getElementById('count-display').textContent = '0'; })();"

        async def on_progress(data):
            progress_log.append(data)

        loop = SynthesisLoop(call_llm=mock_llm, max_attempts=1)
        result = await loop.run(_make_simple_spec(), progress_cb=on_progress)
        assert len(progress_log) > 0
