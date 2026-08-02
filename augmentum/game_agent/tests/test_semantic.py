"""SemanticInputResolver tests."""

from __future__ import annotations

import pytest

from augmentum.game_agent.semantic import SemanticInputResolver, UnknownSemanticError


async def _noop(_duration: int) -> None:
    return None


def test_bind_and_query() -> None:
    """@example: bind exposes the id; has() reports membership."""

    r = SemanticInputResolver()
    assert r.semantic_inputs() == []
    r.bind("jump", _noop)
    assert r.has("jump")
    assert r.semantic_inputs() == ["jump"]


def test_bind_rejects_bad_identifier() -> None:
    """@example: identifiers must be [a-z0-9_]+ and non-empty."""

    r = SemanticInputResolver()
    with pytest.raises(ValueError):
        r.bind("", _noop)
    with pytest.raises(ValueError):
        r.bind("has space", _noop)
    with pytest.raises(ValueError):
        r.bind("kebab-case", _noop)


@pytest.mark.asyncio
async def test_apply_invokes_bound_resolver() -> None:
    """@example: apply() forwards the duration to the bound callable."""

    seen: list[int] = []

    async def grab(duration: int) -> None:
        seen.append(duration)

    r = SemanticInputResolver()
    r.bind("press", grab)
    await r.apply("press", 250)
    assert seen == [250]


@pytest.mark.asyncio
async def test_apply_unknown_raises_with_context() -> None:
    """@example: missing semantic surfaces known-ids in the error."""

    r = SemanticInputResolver()
    r.bind("jump", _noop)
    with pytest.raises(UnknownSemanticError) as exc:
        await r.apply("attack", 100)
    assert exc.value.semantic == "attack"
    assert "jump" in exc.value.known
