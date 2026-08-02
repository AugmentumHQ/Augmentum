"""Companion core tool pool — regression pins (wiring program P0).

CORE_TOOL_NAMES is the list of chat-registry tools the becca_direct
loop ALWAYS carries beyond her verb roster. The reference utilities
went missing from it for months while the voice manifest claimed them
as core — "what's 15% of 240" got small-model mental math (confirmed
live 2026-06-12). These pins make that drift loud.
"""

from __future__ import annotations

from augmentum.companion_runtime.native_loop import CORE_TOOL_NAMES
from augmentum.intent.manifest import VOICE_TOOLS_CORE


def test_reference_utilities_in_core_pool():
    for name in ("calculator", "unit_converter", "datetime"):
        assert name in CORE_TOOL_NAMES, f"{name} dropped from the loop's core pool"


def test_headless_capabilities_in_core_pool():
    for name in ("web_search", "web_fetch", "memory_recall",
                 "wikipedia", "context_peek", "image_generation"):
        assert name in CORE_TOOL_NAMES


def test_voice_core_chat_tools_are_loop_reachable():
    """Any CHAT-REGISTRY tool the voice manifest calls core must also
    be in the loop's pool — the original bug was exactly this mismatch.
    Intent verbs (dotted ids) reach the loop via the roster instead.
    """
    chat_tool_names = {n for n in VOICE_TOOLS_CORE if "." not in n}
    missing = chat_tool_names - set(CORE_TOOL_NAMES)
    assert not missing, (
        f"voice-core chat tools missing from CORE_TOOL_NAMES: {missing}"
    )
