from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from augmentum.session.lifecycle import SessionLifecycle


class _Registry:
    def __init__(self, backend) -> None:
        self._backend = backend

    def get_backend(self, name: str | None = None):
        return self._backend


@pytest.mark.asyncio
async def test_save_kv_uses_backend_method():
    backend = type("Backend", (), {})()
    backend.save_session_state = AsyncMock(return_value=True)

    lifecycle = SessionLifecycle(provider_registry=_Registry(backend))
    lifecycle.touch("sess-1", "passthrough", "engine-model")

    await lifecycle._save_kv("sess-1")

    backend.save_session_state.assert_awaited_once_with("sess-1")
    stats = lifecycle.stats()
    assert stats["sessions"]["sess-1"]["kv_saved"] is True


@pytest.mark.asyncio
async def test_try_restore_kv_uses_backend_method():
    backend = type("Backend", (), {})()
    backend.restore_session_state = AsyncMock(return_value=True)

    lifecycle = SessionLifecycle(provider_registry=_Registry(backend))
    lifecycle.touch("sess-1", "narrative", "engine-model")

    restored = await lifecycle.try_restore_kv("sess-1", "engine-model")

    assert restored is True
    backend.restore_session_state.assert_awaited_once_with("sess-1")
