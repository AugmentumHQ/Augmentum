"""KV prefix stability: datetime placement + late-system normalization.

Covers the 2026-07-09 class fix: the minute-resolution ``<current_time>``
block must never sit at (or be relocated to) the head of the token stream,
because it invalidates the whole KV prefix every turn. Contract:

  * modes/base.py appends it at the absolute END of the payload
  * llama_cpp late systems become in-place user carriers, never front-merges
  * llama_cpp relocates a block embedded in the leading system to the tail
"""
from __future__ import annotations

import pytest

from augmentum.models.llama_cpp import LlamaCppBackend

DT_BLOCK = (
    "<current_time>\nCurrent date: Wednesday, July 08, 2026. "
    "Current time: 20:09 UTC-04:00. trust it.\n</current_time>"
)


def _msgs(*pairs: tuple[str, str]) -> list[dict]:
    return [{"role": r, "content": c} for r, c in pairs]


# --- _normalize_system_messages ------------------------------------------


class TestNormalizeSystemMessages:
    def test_noop_single_leading_system_identity(self):
        messages = _msgs(("system", "sys"), ("user", "hi"))
        assert LlamaCppBackend._normalize_system_messages(messages) is messages

    def test_multiple_leading_systems_still_merge(self):
        out = LlamaCppBackend._normalize_system_messages(
            _msgs(("system", "a"), ("system", "b"), ("user", "hi"))
        )
        assert [m["role"] for m in out] == ["system", "user"]
        assert "a" in out[0]["content"] and "b" in out[0]["content"]

    def test_late_system_becomes_in_place_carrier_not_front_merge(self):
        out = LlamaCppBackend._normalize_system_messages(
            _msgs(
                ("system", "sys"),
                ("user", "turn 1"),
                ("assistant", "reply 1"),
                ("system", DT_BLOCK),
                ("user", "turn 2"),
            )
        )
        roles = [m["role"] for m in out]
        assert roles == ["system", "user", "assistant", "user", "user"]
        # head untouched: prefix stability
        assert out[0]["content"] == "sys"
        # block kept at the position it held, as a user carrier
        assert DT_BLOCK in out[3]["content"]
        assert "<current_time>" not in out[0]["content"]

    def test_trailing_system_stays_trailing(self):
        out = LlamaCppBackend._normalize_system_messages(
            _msgs(("system", "sys"), ("user", "hi"), ("system", DT_BLOCK))
        )
        assert out[0]["content"] == "sys"
        assert out[-1]["role"] == "user"
        assert DT_BLOCK in out[-1]["content"]


# --- _relocate_leading_datetime -------------------------------------------


class TestRelocateLeadingDatetime:
    def test_noop_without_block_identity(self):
        messages = _msgs(("system", "sys"), ("user", "hi"))
        assert LlamaCppBackend._relocate_leading_datetime(messages) is messages

    def test_block_moves_from_head_to_tail(self):
        out = LlamaCppBackend._relocate_leading_datetime(
            _msgs(("system", f"{DT_BLOCK}\n\nYou are helpful."), ("user", "hi"))
        )
        assert "<current_time>" not in out[0]["content"]
        assert out[0]["content"] == "You are helpful."
        assert out[-1]["role"] == "system"
        assert "<current_time>" in out[-1]["content"]

    def test_datetime_only_system_message_is_replaced(self):
        out = LlamaCppBackend._relocate_leading_datetime(
            _msgs(("system", DT_BLOCK), ("user", "hi"))
        )
        assert out[0]["role"] == "user"
        assert out[-1]["role"] == "system"
        assert "<current_time>" in out[-1]["content"]

    def test_noop_when_first_message_not_system(self):
        messages = _msgs(("user", f"{DT_BLOCK} hi"))
        assert LlamaCppBackend._relocate_leading_datetime(messages) is messages

    def test_relocate_then_normalize_keeps_prefix_stable(self):
        """End-to-end shape: embedded head block ends as a tail user carrier."""
        relocated = LlamaCppBackend._relocate_leading_datetime(
            _msgs(
                ("system", f"{DT_BLOCK}\n\nYou are helpful."),
                ("user", "turn 1"),
                ("assistant", "reply 1"),
                ("user", "turn 2"),
            )
        )
        out = LlamaCppBackend._merge_consecutive_same_role(
            LlamaCppBackend._ensure_user_first_after_system(
                LlamaCppBackend._normalize_system_messages(relocated),
            ),
        )
        # first three messages byte-identical to a stable history
        assert out[0] == {"role": "system", "content": "You are helpful."}
        assert out[1]["content"] == "turn 1"
        assert out[2]["content"] == "reply 1"
        # block rides at the very end, merged into the final user turn
        assert out[-1]["role"] == "user"
        assert "<current_time>" in out[-1]["content"]
        assert "turn 2" in out[-1]["content"]


# --- modes/base.py placement ----------------------------------------------


class TestEnsureDatetimePlacement:
    @pytest.mark.asyncio
    async def test_datetime_appended_after_last_user(self):
        from augmentum.models.base import InternalChatRequest, Message
        from augmentum.modes.base import ModeHandler

        class _Backend:
            supports_mid_conversation_system = True

        class _Handler(ModeHandler):
            async def _handle(self, *a, **k):  # pragma: no cover
                raise NotImplementedError

            async def _handle_stream(self, *a, **k):  # pragma: no cover
                raise NotImplementedError

        handler = _Handler.__new__(_Handler)
        handler._backend = _Backend()
        request = InternalChatRequest(
            model="m",
            messages=[
                Message(role="system", content="sys"),
                Message(role="user", content="turn 1"),
                Message(role="assistant", content="reply 1"),
                Message(role="user", content="turn 2"),
            ],
        )
        handler._ensure_datetime(request)
        assert request.messages[-1].role == "system"
        assert "<current_time>" in request.messages[-1].content
        # everything before the tail is untouched
        assert [m.content for m in request.messages[:-1]] == [
            "sys", "turn 1", "reply 1", "turn 2",
        ]

    # --- fallback path: backends WITHOUT mid-conversation-system support ---
    # (self-hosted OpenAI-compat engines + strict cloud APIs). These used to
    # PREPEND the datetime to the head, breaking the prefix cache every minute
    # on cache-sensitive self-hosted engines (2026-07-29 regression). The fix
    # folds the block into the TAIL of the last user message instead.

    @staticmethod
    def _fallback_handler():
        from augmentum.modes.base import ModeHandler

        class _Backend:
            supports_mid_conversation_system = False

        class _Handler(ModeHandler):
            async def _handle(self, *a, **k):  # pragma: no cover
                raise NotImplementedError

            async def _handle_stream(self, *a, **k):  # pragma: no cover
                raise NotImplementedError

        h = _Handler.__new__(_Handler)
        h._backend = _Backend()
        return h

    def test_fallback_folds_datetime_into_last_user_tail(self):
        from augmentum.models.base import InternalChatRequest, Message

        handler = self._fallback_handler()
        request = InternalChatRequest(
            model="m",
            messages=[
                Message(role="system", content="STABLE SYSTEM"),
                Message(role="user", content="hello"),
            ],
        )
        handler._ensure_datetime(request)
        # No new leading system; system[0] untouched (prefix preserved).
        assert request.messages[0].content == "STABLE SYSTEM"
        # Block folded into the tail of the last user turn.
        assert request.messages[-1].role == "user"
        assert request.messages[-1].content.startswith("hello")
        assert "<current_time>" in request.messages[-1].content
        assert len(request.messages) == 2  # no message inserted

    def test_fallback_prefix_byte_stable_across_minutes(self, monkeypatch):
        """The point of the fix: only the tail diverges minute-to-minute."""
        import augmentum.modes.base as base_mod
        from augmentum.models.base import InternalChatRequest, Message

        def run(block: str):
            monkeypatch.setattr(base_mod, "get_datetime_context", lambda: block)
            handler = self._fallback_handler()
            req = InternalChatRequest(
                model="m",
                messages=[
                    Message(role="system", content="STABLE SYSTEM PREFIX"),
                    Message(role="user", content="hello"),
                ],
            )
            handler._ensure_datetime(req)
            return req

        r1 = run("<current_time>\nCurrent time: 00:10\n</current_time>")
        r2 = run("<current_time>\nCurrent time: 00:11\n</current_time>")  # +1 min

        def prefix(r):
            # everything up to the (moving) datetime tail
            head = r.messages[0].content
            user = r.messages[-1].content.split("<current_time>")[0]
            return head + "\x00" + user

        assert prefix(r1) == prefix(r2)  # cacheable prefix is byte-identical
        assert r1.messages[-1].content != r2.messages[-1].content  # tail differs

    def test_fallback_no_trailing_user_preserves_prepend(self):
        """Unusual shape (ends in assistant) keeps the historical head placement."""
        from augmentum.models.base import InternalChatRequest, Message

        handler = self._fallback_handler()
        request = InternalChatRequest(
            model="m",
            messages=[
                Message(role="system", content="sys"),
                Message(role="user", content="hi"),
                Message(role="assistant", content="reply"),
            ],
        )
        handler._ensure_datetime(request)
        assert request.messages[0].role == "system"
        assert "<current_time>" in request.messages[0].content
        assert request.messages[-1].content == "reply"

    def test_idempotent_marker_detects_existing_block(self):
        """The repaired <current_time> marker prevents double-injection."""
        from augmentum.models.base import InternalChatRequest, Message

        handler = self._fallback_handler()
        request = InternalChatRequest(
            model="m",
            messages=[
                Message(role="system", content="sys"),
                Message(role="user", content="hello"),
            ],
        )
        handler._ensure_datetime(request)
        handler._ensure_datetime(request)  # second call must be a no-op
        assert request.messages[-1].content.count("<current_time>") == 1
