"""Live router catalog — the closed-world vocabulary of ``surface_emit`` targets.

The OpenRoom lesson (MiniMax's open-source agent OS): an agent authoring a
capability must be grounded in the *real* catalog of what the host can actually
do — their ``executeListApps`` advertises live ``app_id``s, ``meta.yaml`` declares
the real action vocabulary, ``SET_WALLPAPER`` documents the exact param name. The
model can't invent a target that doesn't exist because it's handed the truth.

Our capability synthesizer had the opposite shape: it asked the model to pick a
``channel`` and ``payload.surface`` cold, and our oracle only checks the channel
*string* (``res.surface_emit.channel == X``) — never whether the frontend acts on
it. So a verb could name ``navigate.open_surface`` with surface ``"workshop"``,
register cleanly, PASS its acceptance test, and then do nothing when dispatched,
because the router has no such target. A "verified but dead" verb — exactly the
failure the whole verification story exists to prevent.

This module parses the single source of truth — ``ui/scripts/intent-action-
router.js`` — for the channels it switches on and the ``navigate.open_surface``
targets it knows, so synthesis can (a) ground the model in the real vocabulary
and (b) reject a dead-target spec with a reason instead of authoring it.

Tolerant by construction: if the router file can't be read (a deployment without
the UI tree, a moved path), the catalog reports ``available is False`` and the
gate declines to block — the engine re-runs the real oracle at build time, and we
never want a parse miss to wedge the whole pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# capabilities -> selfedit -> augmentum -> <repo root>
_ROUTER_REL = "ui/scripts/intent-action-router.js"

# ``case 'channel.name':`` — the dispatch switch arms in routeIntentAction.
_CASE_RE = re.compile(r"case\s+'([a-z][a-z0-9_.]*)'\s*:")
# The ``const _NAV_TARGETS = { ... \n};`` block (first close-brace-at-line-start).
_NAV_BLOCK_RE = re.compile(r"const\s+_NAV_TARGETS\s*=\s*\{(.*?)\n\};", re.DOTALL)
# Top-level keys inside that block sit at exactly two-space indent; nested object
# keys (detail:, tab:) are deeper, so anchoring to two spaces excludes them.
_NAV_KEY_RE = re.compile(r"^ {2}([a-zA-Z_][a-zA-Z0-9_]*)\s*:", re.MULTILINE)


@dataclass(frozen=True)
class RouterCatalog:
    """What the frontend intent-action router actually handles."""

    channels: frozenset[str]
    nav_surfaces: frozenset[str]
    source: str = ""  # the path parsed, "" when not found / unreadable

    @property
    def available(self) -> bool:
        """True only when we read real channels — gates decline to block if False."""
        return bool(self.channels)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_router_catalog(text: str, *, source: str = "") -> RouterCatalog:
    """Parse channels + nav surfaces from router source text (pure; testable)."""
    channels = frozenset(_CASE_RE.findall(text))
    nav: frozenset[str] = frozenset()
    block = _NAV_BLOCK_RE.search(text)
    if block:
        nav = frozenset(_NAV_KEY_RE.findall(block.group(1)))
    return RouterCatalog(channels=channels, nav_surfaces=nav, source=source)


@lru_cache(maxsize=1)
def load_router_catalog() -> RouterCatalog:
    """Load + cache the catalog from the repo's router JS. Tolerant: an unreadable
    file yields an empty (``available is False``) catalog, never an exception."""
    path = _repo_root() / _ROUTER_REL
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        log.info("router_catalog_unavailable", path=str(path), error=repr(exc))
        return RouterCatalog(channels=frozenset(), nav_surfaces=frozenset())
    cat = parse_router_catalog(text, source=str(path))
    if not cat.available:
        log.warning("router_catalog_parsed_empty", path=str(path))
    return cat


def validate_emit_target(
    channel: str, payload: dict | None, *, catalog: RouterCatalog | None = None
) -> str:
    """Return a problem string if a ``surface_emit`` target is dead, else "".

    Tolerant: when the catalog couldn't be read, returns "" (don't block — the
    build-time oracle in the worktree remains the backstop)."""
    cat = catalog or load_router_catalog()
    if not cat.available:
        return ""
    if channel not in cat.channels:
        known = ", ".join(sorted(cat.channels))
        return (
            f"channel {channel!r} is not handled by the frontend router, so the "
            f"verb would register but do nothing. Use one of: {known}"
        )
    if channel == "navigate.open_surface":
        surface = (payload or {}).get("surface")
        if surface not in cat.nav_surfaces:
            known = ", ".join(sorted(cat.nav_surfaces))
            return (
                f"navigate.open_surface target {surface!r} is not a known surface; "
                f"opening it would no-op. Use one of: {known}"
            )
    if channel == "palette.run":
        # palette command ids are a PER-USER, ephemeral runtime catalog
        # (augmentum/intent/app_menu.py: "Restart -> empty until client re-syncs",
        # "latest-client-wins"). A permanent synthesized verb that hard-codes a
        # command_id can't reliably target one across users/sessions, and the
        # existing 'app.act' verb already matches intent to the LIVE catalog
        # dynamically. So palette.run is not synthesizable.
        return (
            "palette.run is not synthesizable: palette command ids are a per-user, "
            "ephemeral runtime catalog, so a permanent verb can't reliably target "
            "one. The existing 'app.act' verb already matches intent to live "
            "palette commands dynamically -- no new verb is needed."
        )
    return ""


# Payload keys each synth-eligible channel actually READS in the router. A
# declared arg merges into the surface_emit payload (the handler does
# ``{**payload, **(args or {})}``), so an arg the channel never reads is silently
# dropped -- "verified but inert". Curated for channels synthesis can emit;
# channels absent from this map skip the check (tolerant, like the catalog gate).
_SYNTH_CHANNEL_ARG_KEYS: dict[str, frozenset[str]] = {
    "navigate.open_surface": frozenset({"surface"}),
    "navigate.back": frozenset(),
}


def validate_declared_args(channel: str, arg_names: list[str] | tuple[str, ...]) -> str:
    """Return a problem if a declared arg can't be consumed by ``channel``, else "".

    Only enforced for channels in ``_SYNTH_CHANNEL_ARG_KEYS`` (the ones synthesis
    realistically emits); unknown channels are left tolerant so this never blocks
    a legitimate target we simply haven't curated yet."""
    consumed = _SYNTH_CHANNEL_ARG_KEYS.get(channel)
    if consumed is None:
        return ""
    dead = [a for a in arg_names if a not in consumed]
    if dead:
        allowed = ", ".join(sorted(consumed)) or "(none)"
        return (
            f"arg(s) {dead!r} are declared but channel {channel!r} never reads "
            f"them, so they'd be silently dropped. {channel!r} consumes: {allowed}. "
            "Remove the unused args or pick a channel that uses them."
        )
    return ""


def describe_for_prompt(catalog: RouterCatalog | None = None) -> str:
    """A grounding block for the synthesis prompt — the real, closed-world target
    vocabulary. Empty string when the catalog is unavailable (omit, don't lie)."""
    cat = catalog or load_router_catalog()
    if not cat.available:
        return ""
    channels = ", ".join(sorted(cat.channels))
    surfaces = ", ".join(sorted(cat.nav_surfaces))
    return (
        "GROUNDING — these are the ONLY surface_emit channels the frontend acts "
        "on. A verb with any other channel registers but does nothing:\n"
        f"  channels: {channels}\n"
        f'For channel "navigate.open_surface", payload.surface MUST be one of:\n'
        f"  {surfaces}\n"
        'Do NOT use "palette.run" — those commands are handled dynamically by the '
        'existing "app.act" verb; a synthesized one can\'t target them reliably.'
    )
