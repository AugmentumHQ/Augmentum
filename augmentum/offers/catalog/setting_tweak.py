"""Setting-tweak offer catalog (whitelist-based).

A *strict* whitelist of safe-to-offer settings. The catalog is
code-not-config so adding a new offer-able setting requires a PR —
the model can NEVER propose toggling something not on this list.

Categorical exclusions (do NOT add to the whitelist):

* Anything that disables auth, sanitization, rate limits, content
  filters, or safety floors. The model's job is to surface user
  benefit, never to weaken the deployment's security posture.
* Anything that touches outbound credentials or trust-pinning.
* Anything irreversible without admin intervention.

Each entry writes its target value to either ``app_settings``
(install-wide → admin scope) or ``user_settings`` (per-user scope).
Admin-scope offers surface to non-admins as greyed-out with an
"ask your admin" hint rather than being hidden — the spec calls this
out so users still know the capability exists.

Currently shipped:

* ``ghost_text``         — enable inline LLM autocomplete in code editor (admin)
* ``knowledge_in_chat``  — surface knowledge packs in chat injectors (admin)
* ``emotion_aware_tts``  — extract RP emotion cues for TTS instruct (admin)
* ``voice_moonshine``    — local streaming STT engine (admin)
* ``companion_dispatch`` — let dispatcher pick chat mode (admin)
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


KIND: str = "setting_tweak"


@dataclass(frozen=True)
class _TweakSpec:
    """One whitelisted setting toggle.

    ``setting_key`` is the canonical snake_case name (must match an
    attribute on ``settings``). ``target_value`` is the value the
    offer proposes writing — stored as the string form even for
    bools (config_routes layer does the string→bool coercion).
    ``scope`` is "admin" when this is an install-wide app_settings
    write, "user" for a per-user override.
    """
    target_id: str
    setting_key: str
    target_value: str
    scope: str            # "admin" | "user"
    label: str
    hint: str


# Whitelist — adding entries here is the ONLY way the model can
# propose toggling these settings.
_TWEAKS: list[_TweakSpec] = [
    _TweakSpec(
        target_id="ghost_text",
        setting_key="ghost_text_enabled",
        target_value="true",
        scope="admin",
        label="Inline AI autocomplete",
        hint=(
            "Turn on LLM-powered inline suggestions in the code editor. "
            "Off by default — performance impact varies by model."
        ),
    ),
    _TweakSpec(
        target_id="knowledge_in_chat",
        setting_key="knowledge_library_in_chat",
        target_value="true",
        scope="admin",
        label="Knowledge packs in chat",
        hint=(
            "Surface installed knowledge packs (Wikipedia, ZIM, language packs) "
            "as retrieval context for chat turns."
        ),
    ),
    _TweakSpec(
        target_id="emotion_aware_tts",
        setting_key="tts_emotion_aware",
        target_value="true",
        scope="admin",
        label="Emotion-aware TTS",
        hint=(
            "Extract roleplay emotion cues from response text and pass them "
            "to the TTS engine's instruct parameter for more expressive voice."
        ),
    ),
    _TweakSpec(
        target_id="voice_moonshine",
        setting_key="voice_moonshine_enabled",
        target_value="true",
        scope="admin",
        label="Moonshine streaming STT (local, English)",
        hint=(
            "Use the bundled Moonshine model for local low-latency speech "
            "recognition on English. Falls back to other STT providers for "
            "other languages."
        ),
    ),
    _TweakSpec(
        target_id="companion_dispatch",
        setting_key="companion_dispatch_routes_chat",
        target_value="true",
        scope="admin",
        label="Companion-routed chat",
        hint=(
            "Let the companion dispatcher pick the chat mode using its richer "
            "feature set (persona affinity, runtime state, lexical similarity) "
            "instead of just the classifier."
        ),
    ),
]


def _already_at_target(spec: _TweakSpec) -> bool:
    """Cheap "is the live value already this?" check for previews.

    Settings is the live runtime object so this reflects whatever the
    last write left in place. Bool settings compare against the
    string-coerced form.
    """
    current = getattr(settings, spec.setting_key, None)
    if isinstance(current, bool):
        return ("true" if current else "false") == spec.target_value.lower()
    return str(current) == spec.target_value


def _make_entry(spec: _TweakSpec) -> CatalogEntry:
    async def _preview(_target_id: str, _user_id: str) -> OfferPreview | None:
        if _already_at_target(spec):
            return None  # already at target — don't surface
        return OfferPreview(
            label=spec.label,
            hint=spec.hint,
            details={
                "scope": spec.scope,
                "setting_key": spec.setting_key,
                "target_value": spec.target_value,
                "writes_to": (
                    f"app_settings.{spec.setting_key}"
                    if spec.scope == "admin"
                    else f"user_settings.{spec.setting_key}"
                ),
            },
        )

    async def _accept(payload: dict[str, Any], request: "Request") -> dict[str, Any]:
        store = getattr(request.app.state, "settings_store", None)
        user = request.scope.get("user")
        user_id = getattr(user, "id", "") if user is not None else ""
        is_admin = bool(getattr(user, "is_admin", False)) if user is not None else False

        if store is None:
            return {
                "ok": False,
                "error": "no_settings_store",
                "detail": "Settings store not attached to app.state.",
            }
        if not user_id:
            return {"ok": False, "error": "no_user"}
        if spec.scope == "admin" and not is_admin:
            return {
                "ok": False,
                "error": "admin_required",
                "detail": (
                    f"This setting is install-wide; an administrator has "
                    f"to enable {spec.label}. Ask your admin."
                ),
            }

        try:
            if spec.scope == "admin":
                existing = await store.get(spec.setting_key)
                if existing == spec.target_value:
                    already_set = True
                else:
                    already_set = False
                    await store.set(spec.setting_key, spec.target_value)
            else:
                existing = await store.get_user(user_id, spec.setting_key)
                if existing == spec.target_value:
                    already_set = True
                else:
                    already_set = False
                    await store.set_user(user_id, spec.setting_key, spec.target_value)
            # Mirror onto the live settings object so the change takes
            # effect this turn rather than waiting for a process
            # restart. Only meaningful for admin-scope writes — user
            # overrides are read via the store, not settings.<key>.
            if spec.scope == "admin":
                live_attr = getattr(settings, spec.setting_key, None)
                if isinstance(live_attr, bool):
                    object.__setattr__(
                        settings, spec.setting_key,
                        spec.target_value.lower() == "true",
                    )
                else:
                    object.__setattr__(
                        settings, spec.setting_key, spec.target_value,
                    )
        except Exception as exc:
            return {
                "ok": False,
                "error": "store_failed",
                "detail": str(exc)[:200],
            }

        log.info(
            "offer_setting_tweak_applied",
            setting_key=spec.setting_key,
            scope=spec.scope,
            already_set=already_set,
            user_id=user_id,
        )
        return {
            "ok": True,
            "setting_key": spec.setting_key,
            "target_value": spec.target_value,
            "scope": spec.scope,
            "already_set": already_set,
            "next_step": (
                f"{spec.label} is now enabled. "
                f"{'Visible to everyone on this install.' if spec.scope == 'admin' else 'Visible to you only.'} "
                f"Toggle back in Settings → Tools."
            ),
        }

    return CatalogEntry(
        kind=KIND,
        target_id=spec.target_id,
        title=f"Enable {spec.label}?",
        scope=spec.scope,
        build_preview=_preview,
        accept=_accept,
    )


ENTRIES: list[CatalogEntry] = [_make_entry(s) for s in _TWEAKS]


if ENTRIES:
    register_kind(KIND, ENTRIES)
