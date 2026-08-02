"""Mode-switch offer catalog.

Each entry corresponds to one value of :class:`augmentum.classifier.router.Mode`.
"Install" semantics: write ``default_mode = <mode>`` into the user's
``user_settings`` row. The classifier reads this via
``chat_router._read_default_mode`` and honors it as priority 3 — beats
content heuristics (narrative/complexity) and session continuity, but
loses to per-request override (model prefix, ``X-Augmentum-Mode``
header).

Per-user, not per-API-key. Phase 4 will add a per-API-key trio
(default_mode + allowed_models + cost_cap) on ``augmentum_api_keys`` so
external clients can pin without affecting the chat UI's default — until
then, ``d/`` prefix or the header is the right path for external API
keys, and this offer is for the chat-UI persona.

Catalog content is code-not-config: the entries here mirror Mode enum
values. The model's ``propose_offer`` call picks which target_id to
offer based on conversation context — e.g. repeated direct-API-style
requests → ``direct``; user keeps switching to coder workspace →
``coder``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from augmentum.classifier.router import MODE_MAP, Mode
from augmentum.offers.catalog.base import (
    CatalogEntry,
    OfferPreview,
    register_kind,
)
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from fastapi import Request


log = get_logger(__name__)


KIND: str = "mode_switch"
SETTING_KEY: str = "default_mode"


# Human-friendly descriptions for the preview chip. Kept short so the
# chip stays scannable — the model is expected to give a more nuanced
# ``reason`` in the propose_offer call that lives alongside the chip.
_MODE_DESCRIPTIONS: dict[Mode, tuple[str, str]] = {
    Mode.PASSTHROUGH: (
        "Passthrough mode",
        "Raw chat with no injectors. Default for general assistant work.",
    ),
    Mode.ANALYTICAL: (
        "Analytical mode",
        "Routes via reasoning flows — research, multi-step verification, structured answers.",
    ),
    Mode.NARRATIVE: (
        "Narrative mode",
        "Three-layer story memory + character cards. For roleplay and long-form fiction.",
    ),
    Mode.AGENTIC: (
        "Agentic mode",
        "Tool-using agent with web search, image gen, artifact authoring.",
    ),
    Mode.CODER: (
        "Coder mode",
        "Containerized coding agent: plan/act loop, workspaces, git review.",
    ),
    Mode.DIRECT: (
        "Direct mode",
        "Raw API pipe — no memory / packs / dream / files / SSOS injection. "
        "For external API clients (Claude Code, Aider, Cline, etc.).",
    ),
    Mode.BECCA_DIRECT: (
        "Becca-direct mode",
        "Chat routes through the companion's own prompt composer + tier stream.",
    ),
}


def _make_entry(mode: Mode) -> CatalogEntry:
    target_id = mode.value
    label, hint = _MODE_DESCRIPTIONS[mode]

    async def _preview(_target_id: str, user_id: str) -> OfferPreview | None:
        # No "already pinned" short-circuit at preview time — that
        # would mean reading user_settings without a Request. The
        # accept handler is idempotent (writing the same value is a
        # no-op) and reports ``already_pinned`` back to the UI.
        return OfferPreview(
            label=label,
            hint=hint,
            details={
                "scope": "user",
                "mode": target_id,
                "writes_to": f"user_settings.{SETTING_KEY}",
                "wins_over": "narrative + complexity heuristics + session continuity",
                "loses_to": "explicit prefix (e.g. 'd/') or X-Augmentum-Mode header",
            },
        )

    async def _accept(payload: dict[str, Any], request: "Request") -> dict[str, Any]:
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
                "detail": "Mode pin is per-user; no authenticated user on the request.",
            }

        try:
            existing = await store.get_user(user_id, SETTING_KEY)
            already_pinned = existing == target_id
            if not already_pinned:
                await store.set_user(user_id, SETTING_KEY, target_id)
        except Exception as exc:
            return {
                "ok": False,
                "error": "store_failed",
                "detail": str(exc)[:200],
            }

        log.info(
            "offer_mode_switch_pinned",
            mode=target_id, user_id=user_id,
            already_pinned=already_pinned,
        )
        return {
            "ok": True,
            "mode": target_id,
            "already_pinned": already_pinned,
            "next_step": (
                f"Default mode pinned to {label}. New chat turns will route "
                f"to this mode unless you use a prefix (e.g. 'd/') or "
                f"X-Augmentum-Mode header. Settings → General → Default "
                f"Mode shows the current pin and lets you clear it."
            ),
        }

    return CatalogEntry(
        kind=KIND,
        target_id=target_id,
        title=f"Pin {label} as your default?",
        scope="user",
        build_preview=_preview,
        accept=_accept,
    )


# Build entries for every Mode value MODE_MAP knows about. Iterating
# MODE_MAP (rather than Mode directly) keeps the catalog in sync with
# the classifier's accepted-mode-names list.
ENTRIES: list[CatalogEntry] = [
    _make_entry(mode) for _, mode in MODE_MAP.items()
]


if ENTRIES:
    register_kind(KIND, ENTRIES)
