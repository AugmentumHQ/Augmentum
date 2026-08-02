"""PassthroughSubagent adapter contract — Message objects, not dicts.

Regression for 2026-06-10: ``InternalChatRequest.__post_init__`` coerces
every message to the ``Message`` dataclass, but the adapter read the
synthesis seed with dict ``.get`` — so EVERY invocation crashed with
"'Message' object has no attribute 'get'" the moment the headless news
path started exercising it (search succeeded, gather failed).
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from augmentum.models.base import InternalChatRequest


class _Bus:
    async def publish_topic(self, *a, **kw):
        return None


def _runtime():
    return SimpleNamespace(
        _app_state=SimpleNamespace(tool_registry=object()),
        bus=_Bus(),
        companion_id="becca",
    )


@pytest.mark.asyncio
async def test_invoke_reads_message_dataclass_seed(monkeypatch):
    from augmentum.companion_runtime import tiers
    from augmentum.companion_runtime.subagents.base import SubagentContext
    from augmentum.companion_runtime.subagents.passthrough import (
        PassthroughSubagent,
    )
    from augmentum.modes.passthrough.orchestrator import SSOSOrchestrator

    async def _fake_primary(runtime):
        return object(), "test-model"

    # try_orchestrate returns a request whose messages were dict-shaped
    # at construction — __post_init__ coerces them to Message objects,
    # which is exactly the shape that crashed the old dict access.
    seeded = InternalChatRequest(
        model="test-model",
        messages=[
            {"role": "user", "content": "what's the latest news?"},
            {"role": "system", "content": "NEWS DIGEST FROM SEARCH"},
        ],
    )

    async def _fake_orchestrate(self, request):
        return seeded

    monkeypatch.setattr(tiers, "primary", _fake_primary)
    monkeypatch.setattr(SSOSOrchestrator, "try_orchestrate", _fake_orchestrate)

    runtime = _runtime()
    ctx = SubagentContext(
        intent=SimpleNamespace(
            text="what's the latest news?", user_id="u1", metadata={},
        ),
        runtime=runtime,
        bus=runtime.bus,
        companion_id="becca",
        invocation_id="inv1",
    )
    result = await PassthroughSubagent().invoke(ctx)

    assert not result.error
    assert result.content == "NEWS DIGEST FROM SEARCH"


@pytest.mark.asyncio
async def test_invoke_empty_seed_messages(monkeypatch):
    from augmentum.companion_runtime import tiers
    from augmentum.companion_runtime.subagents.base import SubagentContext
    from augmentum.companion_runtime.subagents.passthrough import (
        PassthroughSubagent,
    )
    from augmentum.modes.passthrough.orchestrator import SSOSOrchestrator

    async def _fake_primary(runtime):
        return object(), "test-model"

    async def _fake_orchestrate(self, request):
        return InternalChatRequest(model="test-model", messages=[])

    monkeypatch.setattr(tiers, "primary", _fake_primary)
    monkeypatch.setattr(SSOSOrchestrator, "try_orchestrate", _fake_orchestrate)

    runtime = _runtime()
    ctx = SubagentContext(
        intent=SimpleNamespace(text="x", user_id="u1", metadata={}),
        runtime=runtime,
        bus=runtime.bus,
        companion_id="becca",
        invocation_id="inv2",
    )
    result = await PassthroughSubagent().invoke(ctx)
    assert not result.error
    assert result.content == ""
