"""Unit tests for foundry visual verification (injected captioner)."""
from __future__ import annotations

import pytest

from augmentum.coder.foundry.visual_verify import verify_image


def _cap(text):
    async def captioner(_img, _prompt):
        return text
    return captioner


@pytest.mark.asyncio
async def test_clean_render_yields_no_notes():
    notes = await verify_image(b"png", captioner=_cap("looks fine"), objective="a crate")
    assert notes == []


@pytest.mark.asyncio
async def test_problem_becomes_a_note():
    notes = await verify_image(
        b"png", captioner=_cap("The crate is flat and untextured."), objective="a crate",
    )
    assert notes == ["The crate is flat and untextured."]


@pytest.mark.asyncio
async def test_empty_image_skips():
    notes = await verify_image(b"", captioner=_cap("whatever"))
    assert notes == []


@pytest.mark.asyncio
async def test_blank_caption_yields_no_note():
    notes = await verify_image(b"png", captioner=_cap("   "))
    assert notes == []


@pytest.mark.asyncio
async def test_captioner_failure_is_swallowed():
    async def boom(_img, _prompt):
        raise RuntimeError("vision down")
    notes = await verify_image(b"png", captioner=boom)
    assert notes == []


@pytest.mark.asyncio
async def test_prompt_mentions_objective_and_kind():
    seen = {}
    async def captioner(_img, prompt):
        seen["p"] = prompt
        return "looks fine"
    await verify_image(b"png", captioner=captioner, objective="a red car", kind="frame")
    assert "a red car" in seen["p"]
    assert "running game" in seen["p"]
