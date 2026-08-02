"""Catalog protocol — what a per-kind offer entry has to provide.

Each offer kind (``mcp_server``, ``power``, …) is a small module that
defines ``CatalogEntry`` instances and registers them via
``register_kind``. The dispatcher looks an entry up by ``(kind,
target_id)`` and uses its ``build_preview`` to produce the chip body
and its ``accept`` to install / change / save on user click.

Catalog content is *code, not configuration*. Adding a new MCP server
means landing a PR that adds the entry — the model can only propose
entries that have been curated this way.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import Request


@dataclass(frozen=True)
class OfferPreview:
    """The structured preview block rendered inside the chip.

    ``label`` is the headline (e.g. "mcp/gmail-server v1.4"); ``hint``
    is one extra line of context (e.g. "18 MB install, 4 tools").
    ``details`` is a dict of free-form key/value pairs the UI may
    render in an expand-on-hover state — kept loose so phase 2 can
    grow into it.
    """

    label: str = ""
    hint: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if self.label:
            out["label"] = self.label
        if self.hint:
            out["hint"] = self.hint
        if self.details:
            out["details"] = dict(self.details)
        return out


# A preview builder takes the target_id + user_id and returns either
# an ``OfferPreview`` or ``None`` (the entry is not relevant for this
# user right now — e.g. the MCP server is already installed). The
# dispatcher uses ``None`` as the "don't surface" signal.
PreviewBuilder = Callable[[str, str], Awaitable[OfferPreview | None]]


# An accept handler runs when the user clicks Install. It receives
# the notification's stored payload, the action id (always "accept"
# in practice — snooze/never are handled by the dispatcher itself),
# and the FastAPI Request (so it can reach app.state to pull
# MCPClientManager / SettingsStore / etc.).
#
# It must return a dict the UI renders as the post-Accept state —
# typically ``{ok: True, ...}`` on success or ``{ok: False, error: ...}``
# on failure. The UI shows the error inline; the offer stays
# un-suppressed so the user can retry.
AcceptHandler = Callable[
    [dict[str, Any], "Request"], Awaitable[dict[str, Any]],
]


@dataclass(frozen=True)
class CatalogEntry:
    """One offerable target within a kind.

    ``kind`` is the namespace (``"mcp_server"``); ``target_id`` is the
    in-namespace identifier (``"gmail"``). Together they form the
    dedupe key used by the notification publish path.

    ``scope`` is the auth gate: ``"user"`` (anyone can accept) or
    ``"admin"`` (only admins; non-admins see a greyed-out chip).

    ``allowed_modes`` is the *context* gate — which chat modes the
    offer is sensible to surface from. Empty tuple (the default)
    means "any mode". A non-empty tuple restricts the dispatcher to
    only propose this entry when the calling handler is in one of
    the listed modes — e.g. ``("coder",)`` for powers and workspace
    profiles, which are coder-mode concepts and would surface a
    confusing chip in passthrough/narrative. The mode comes from
    the tool's ``_context['mode']`` which each mode handler
    populates when dispatching. Missing/empty mode in context falls
    open (proposes anyway) so non-handler call paths (tests,
    scripts) aren't broken.
    """

    kind: str
    target_id: str
    title: str
    scope: str = "user"  # "user" | "admin"
    build_preview: PreviewBuilder | None = None
    accept: AcceptHandler | None = None
    icon: str = ""
    allowed_modes: tuple[str, ...] = ()


# ── Registry ─────────────────────────────────────────────────────


# Global registry: kind -> target_id -> entry. Populated at module
# import time by each per-kind catalog module's top-level
# ``register_kind`` calls.
CATALOG: dict[str, dict[str, CatalogEntry]] = {}


def register_kind(kind: str, entries: list[CatalogEntry]) -> None:
    """Install entries under ``kind``.

    Re-registering the same kind replaces the previous map — that's
    how tests reset state. Production code calls this exactly once
    per kind, at import time.
    """

    if not kind:
        raise ValueError("kind is required")
    by_target: dict[str, CatalogEntry] = {}
    for entry in entries:
        if entry.kind != kind:
            raise ValueError(
                f"entry kind mismatch: {entry.kind!r} registered under {kind!r}",
            )
        if entry.target_id in by_target:
            raise ValueError(
                f"duplicate target_id {entry.target_id!r} for kind {kind!r}",
            )
        by_target[entry.target_id] = entry
    CATALOG[kind] = by_target


def get_entry(kind: str, target_id: str) -> CatalogEntry | None:
    """Look up an entry by (kind, target_id)."""

    return CATALOG.get(kind, {}).get(target_id)


def list_kinds() -> list[str]:
    """All registered kinds, in registration order."""

    return list(CATALOG.keys())


def list_targets(kind: str) -> list[str]:
    """All registered target_ids for one kind."""

    return list(CATALOG.get(kind, {}).keys())
