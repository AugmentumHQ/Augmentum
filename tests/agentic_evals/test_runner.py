"""Parametrized runner for agentic flow eval cases.

Each case YAML under ``cases/{flow}/*.yaml`` is loaded, executed through
``AgenticHandler._execute_flow_steps`` with a ScriptedBackend + MockTool
registry, then checked against declared property assertions.

Drives ``_execute_flow_steps`` directly to skip routing/approval
scaffolding — those belong in their own tests. This harness is focused
on generative-step + delivery quality.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from augmentum.models.base import InternalChatRequest, Message
from augmentum.modes.agentic.handler import AgenticHandler
from augmentum.modes.agentic.task_state import TaskState, TaskStatus
from augmentum.reasoning import templates as _tpl

from tests.agentic_evals.conftest import MockTool, MockToolRegistry, ScriptedBackend, ScriptedResponse
from tests.agentic_evals.properties import apply_assertions

CASES_DIR = Path(__file__).parent / "cases"


def _discover_cases() -> list[tuple[str, Path]]:
    cases = []
    if not CASES_DIR.exists():
        return cases
    for yaml_path in sorted(CASES_DIR.rglob("*.yaml")):
        rel = yaml_path.relative_to(CASES_DIR).with_suffix("")
        cases.append((str(rel).replace("\\", "/"), yaml_path))
    return cases


def _load_case(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_tools(spec: dict) -> list[MockTool]:
    tools = []
    for name, cfg in (spec or {}).items():
        tools.append(MockTool(
            name=name,
            output=cfg.get("output", ""),
            metadata=cfg.get("metadata", {}),
            success=cfg.get("success", True),
        ))
    return tools


def _collect_deliver_output(chunks: list, flow) -> str:
    """Concatenate chunks that carry delivered content to the user.

    AgenticHandler emits two kinds of chunks: step progress chunks (which
    include ``phase`` in the augmentum metadata) and streamed delivery
    chunks (which carry only ``{"mode": "agentic", "task_id": ...}``).
    The deliver step's text is the latter — collect those.
    """
    parts: list[str] = []
    for c in chunks:
        meta = getattr(c, "augmentum", None) or {}
        if meta.get("phase") is not None:
            continue
        if c.content_delta:
            parts.append(c.content_delta)
    return "".join(parts)


@pytest.mark.asyncio
@pytest.mark.parametrize("case_id,case_path", _discover_cases(), ids=lambda x: x if isinstance(x, str) else "")
async def test_agentic_flow_case(case_id: str, case_path: Path) -> None:
    case = _load_case(case_path)

    # Resolve flow factory: check templates first, then local eval fixtures.
    from tests.agentic_evals import conftest as _eval_conftest
    flow_factory = getattr(_tpl, case["flow"], None) or getattr(_eval_conftest, case["flow"], None)
    if flow_factory is None:
        pytest.skip(f"flow factory {case['flow']!r} not found")
    flow = flow_factory()

    # Build scripted backend
    backend = ScriptedBackend(
        responses=[ScriptedResponse(**r) for r in case.get("responses", [])]
    )

    # Build mock tool registry
    tools = _build_tools(case.get("mock_tools", {}))
    registry = MockToolRegistry(tools)

    handler = AgenticHandler(
        backend=backend,
        tool_registry=registry,
        session_id="eval",
    )

    task = TaskState(
        session_id="eval",
        flow_id=flow.id,
        status=TaskStatus.RUNNING,
        autonomy_level=4,
        original_query=case["query"],
        title=case.get("name", "eval"),
        plan_md="",
        total_steps=len([s for s in flow.steps if s.enabled]),
    )

    request = InternalChatRequest(
        model="mock",
        messages=[Message(role="user", content=case["query"])],
        stream=True,
        temperature=0.0,
    )

    chunks = []
    async for chunk in handler._execute_flow_steps(flow, task, case["query"], "mock", request):
        chunks.append(chunk)

    text = _collect_deliver_output(chunks, flow)
    failures = apply_assertions(text, case.get("assertions", []))
    assert not failures, (
        f"case {case_id!r} failed assertions:\n  - "
        + "\n  - ".join(failures)
        + f"\n\n--- deliver output ---\n{text[:2000]}"
    )
