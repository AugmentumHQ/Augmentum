"""Model-swap offer catalog.

Pins a role's model as the user's primary chat model. The catalog
exposes *role names* (heavyweight / utility), not individual model
ids — what each role resolves to is install-configured via the
matching ``settings.<role>_model`` value. This keeps the curated
catalog stable while letting each install map roles to whatever
local model fits.

"Install" semantics: write ``user_settings['primary_chat_model']``
to the role's resolved model id. ``primary_chat_model`` is the
canonical "what the user is chatting with right now" mirror —
already consumed by the openai_compat layer, role resolution for
heavyweight/utility/classifier, anthropic alias fallback, etc.

If the role's model setting is empty (admin hasn't configured the
heavyweight model on this install), the accept handler returns
``ok=False`` with a clear ``unconfigured_role`` message rather than
silently writing an empty value.

Per-user. The install-level ``primary_chat_model`` is unaffected —
``user_settings`` overrides it via the standard read fallback in
``settings_store.get_user_or_global``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from augmentum.config import settings
from augmentum.offers.catalog.base import (
    CatalogEntry,
    OfferPreview,
    register_kind,
)
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from fastapi import Request


log = get_logger(__name__)


KIND: str = "model_swap"
SETTING_KEY: str = "primary_chat_model"


@dataclass(frozen=True)
class _RoleSpec:
    target_id: str
    role: str            # role name for log + display
    setting_attr: str    # settings attribute holding the model id
    label: str
    hint: str


_ROLES: list[_RoleSpec] = [
    _RoleSpec(
        target_id="heavyweight",
        role="heavyweight",
        setting_attr="heavyweight_model",
        label="Heavyweight model",
        hint=(
            "Pin the install's heavyweight model as your chat default. "
            "Slower but higher capability — good for complex reasoning."
        ),
    ),
    _RoleSpec(
        target_id="utility",
        role="utility",
        setting_attr="utility_model",
        label="Utility (fast) model",
        hint=(
            "Pin the install's utility model as your chat default. "
            "Lighter and faster — good for everyday assistant work."
        ),
    ),
]


def _resolved_model(spec: _RoleSpec) -> str:
    return (getattr(settings, spec.setting_attr, "") or "").strip()


def _make_entry(spec: _RoleSpec) -> CatalogEntry:
    async def _preview(_target_id: str, _user_id: str) -> OfferPreview | None:
        model = _resolved_model(spec)
        if not model:
            # Role is unconfigured on this install — don't surface
            # the offer at all rather than letting the user accept
            # something that fails.
            return None
        return OfferPreview(
            label=spec.label,
            hint=spec.hint,
            details={
                "scope": "user",
                "role": spec.role,
                "resolved_model": model,
                "writes_to": f"user_settings.{SETTING_KEY}",
            },
        )

    async def _accept(payload: dict[str, Any], request: Request) -> dict[str, Any]:
        store = getattr(request.app.state, "settings_store", None)
        user = request.scope.get("user")
        user_id = getattr(user, "id", "") if user is not None else ""

        if store is None:
            return {
                "ok": False,
                "error": "no_settings_store",
                "detail": "Settings store not attached to app.state.",
            }
        if not user_id:
            return {
                "ok": False,
                "error": "no_user",
                "detail": "Model pin is per-user; no authenticated user on the request.",
            }

        model = _resolved_model(spec)
        if not model:
            return {
                "ok": False,
                "error": "unconfigured_role",
                "detail": (
                    f"The {spec.role} role has no model assigned on this install. "
                    f"Set Settings → Models → {spec.label} first."
                ),
            }

        try:
            existing = await store.get_user(user_id, SETTING_KEY)
            already_pinned = existing == model
            if not already_pinned:
                await store.set_user(user_id, SETTING_KEY, model)
        except Exception as exc:
            return {
                "ok": False,
                "error": "store_failed",
                "detail": str(exc)[:200],
            }

        log.info(
            "offer_model_swap_pinned",
            role=spec.role, model=model, user_id=user_id,
            already_pinned=already_pinned,
        )
        return {
            "ok": True,
            "role": spec.role,
            "model": model,
            "already_pinned": already_pinned,
            "next_step": (
                f"Pinned {spec.label} ({model}) as your chat default. "
                f"Future turns route through it unless you pick a different "
                f"model in the chat composer. Settings → Models shows the "
                f"current selection."
            ),
        }

    return CatalogEntry(
        kind=KIND,
        target_id=spec.target_id,
        title=f"Switch to the {spec.label.lower()}?",
        scope="user",
        build_preview=_preview,
        accept=_accept,
    )


ENTRIES: list[CatalogEntry] = [_make_entry(s) for s in _ROLES]


if ENTRIES:
    register_kind(KIND, ENTRIES)
