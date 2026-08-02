"""Hook reserved for chat-route-driven ``primary_chat_model`` updates.

Historical context: this used to auto-adopt ``primary_chat_model`` from
every chat request as a band-aid for "the frontend has multiple model-
select paths and only some push the value." That was wrong — a one-off
test chat with model X (cloud provider, niche local quant, anything)
would mutate the setting, and background workers reading the same value
on their interval (dream engine, portrait engine, distiller, every
"Auto — use Primary" role) would start trying to resolve that one-off
model. When it lived in the cloud and the workers' resolution path
didn't know about the ``@backend`` suffix, they'd fall through to the
local engine and fail noisily on every tick.

The corrected design: ``primary_chat_model`` is a stable user preference,
only updated by the explicit UI dropdown push (frontend's
``pushPrimaryChatModel`` → ``PUT /api/config/tools``). Chat requests do
not drift it. Engine-load events do not drift it. Background workers
read whatever the user has set; if the user hasn't set anything, the
resolver falls back to "first model on default backend."

This function is preserved as a no-op so the existing chat-route call
sites keep compiling without an edit pass. The right place to update
``primary_chat_model`` is the settings-write endpoint, full stop.
"""
from __future__ import annotations

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


async def adopt_primary_chat_model(app_state: object, model_name: str) -> None:
    """No-op. Kept for call-site back-compat.

    Replaced by the explicit-only UI dropdown push to
    ``/api/config/tools``. See module docstring for rationale.
    """
    return
