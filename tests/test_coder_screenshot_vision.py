"""Screenshot vision feed — iterative comparison (verification spine follow-on).

The handler feeds browser_screenshot captures back to the agent: pixels
for VL drivers, a vision-router caption for text-only drivers. These
tests cover the COMPARISON rung added 2026-07-06: the second and later
screenshots must carry the previous iteration's context so the agent
diffs instead of re-glancing.

The handler class is huge; we test `_maybe_screenshot_image_message`
directly on a minimal stub object carrying only the attributes the
method reads — same approach as the goal-judge unit tests.
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from augmentum.modes.coder.handler import CoderHandler
from augmentum.tools.base import ToolResult


class _Stub:
    """Bare attribute bag bound to the real method under test."""

    _SCREENSHOT_IMAGE_MAX_BYTES = CoderHandler._SCREENSHOT_IMAGE_MAX_BYTES
    _maybe_screenshot_image_message = (
        CoderHandler._maybe_screenshot_image_message
    )

    def __init__(self, *, vision_capable: bool, caption: str = "a page"):
        self._workspace_id = "ws1"
        self._container_manager = AsyncMock()
        self._container_manager.file_download = AsyncMock(
            return_value=b"\x89PNG fake bytes",
        )
        self._vision_capable = vision_capable
        self._last_screenshot_note = None
        self._screenshot_pixel_feeds = 0
        router = AsyncMock()
        router.caption = AsyncMock(return_value=caption)
        self._vision_router = router

    async def _model_supports_vision(self, model: str) -> bool:
        return self._vision_capable


def _shot(path: str = "/workspace/.augmentum/screenshots/shot_1.png") -> ToolResult:
    return ToolResult(success=True, output="ok", metadata={"browser": {"path": path}})


@pytest.mark.asyncio
async def test_first_caption_feed_has_no_comparison():
    stub = _Stub(vision_capable=False, caption="header overlaps the nav")
    msg = await stub._maybe_screenshot_image_message(_shot(), "text-model")
    assert msg is not None
    assert "header overlaps the nav" in msg.content
    assert "for comparison" not in msg.content
    # Cached for the next iteration.
    assert stub._last_screenshot_note[1] == "header overlaps the nav"


@pytest.mark.asyncio
async def test_second_caption_feed_replays_previous_description():
    stub = _Stub(vision_capable=False, caption="first: broken layout")
    await stub._maybe_screenshot_image_message(_shot("/workspace/a.png"), "m")
    stub._vision_router.caption = AsyncMock(return_value="second: layout fixed")
    msg = await stub._maybe_screenshot_image_message(_shot("/workspace/b.png"), "m")
    assert "second: layout fixed" in msg.content
    assert "for comparison" in msg.content
    assert "first: broken layout" in msg.content
    assert "/workspace/a.png" in msg.content
    assert "what changed" in msg.content
    # No second vision call for the comparison — cached text only.
    stub._vision_router.caption.assert_awaited_once()


@pytest.mark.asyncio
async def test_vl_second_feed_prompts_pixel_comparison():
    stub = _Stub(vision_capable=True)
    first = await stub._maybe_screenshot_image_message(_shot(), "vl-model")
    second = await stub._maybe_screenshot_image_message(_shot(), "vl-model")
    first_text = first.content[0]["text"]
    second_text = second.content[0]["text"]
    assert "Compare" not in first_text
    assert "Compare it against the previous screenshot" in second_text
    # Pixels attached both times.
    for m in (first, second):
        assert m.content[1]["type"] == "image_url"
        assert m.content[1]["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.asyncio
async def test_feed_returns_none_outside_workspace_or_on_empty_caption():
    stub = _Stub(vision_capable=False, caption="")
    assert await stub._maybe_screenshot_image_message(
        _shot("/etc/passwd"), "m",
    ) is None
    # Empty caption from the router → no message, and no stale cache write.
    assert await stub._maybe_screenshot_image_message(_shot(), "m") is None
    assert stub._last_screenshot_note is None
