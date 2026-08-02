"""Intent → spec classifier — "what was asked, and what oracle could confirm it?"

The verifier router is honest by construction: it only marks a change ``verified``
when an oracle confirmed the *intent*. But which oracle CAN confirm a given change
depends on what was asked. A bugfix is confirmable by a reproducing test; a new
feature by a behavior gate; a CSS tweak by *nobody but the user* ("move the button
left" runs fine either way — only taste decides). This module classifies the
request so the router knows:

  * ``intent_class`` — the label verifiers filter on (bugfix/feature/refactor/
    debt/style/unknown). Confirm-oracles register against these classes.
  * ``surface`` — frontend / backend / migration / mixed, derived from the changed
    paths. Drives the autonomy lane later (frontend is the safe fast-lane; a
    migration is forced human-required — corruption is never auto-attempted).
  * ``mechanically_confirmable`` — is there *any* mechanical oracle that could
    prove this intent was met? If not, a green run can only ever be
    ``human_required`` — and saying so up front is the honest thing.
  * ``behaviors`` — frozen acceptance behaviors, when derivable (reuses the build
    contract). These are what a behavior/test oracle checks the change against.

Reuses ``modes/coder/intent.classify_turn_intent`` for the coarse kind rather than
reinventing intent detection; layers the self-edit-specific style/debt signals and
the confirmability mapping on top.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from augmentum.modes.coder.intent import TurnIntentKind, classify_turn_intent
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Canonical self-edit intent classes (the strings verifiers register against).
CLASS_BUGFIX = "bugfix"        # fix a defect — confirmable by a reproducing test
CLASS_FEATURE = "feature"      # add capability — confirmable by a behavior gate / new test
CLASS_REFACTOR = "refactor"    # restructure behavior-preserving — bar is no-regression
CLASS_DEBT = "debt"            # pay down audit-flagged debt — confirmable by audit-delta
CLASS_STYLE = "style"          # taste: CSS/copy/layout — only the human can confirm
CLASS_AUTHORED_ORACLE = "authored-oracle"  # the engine authoring its own examiner —
                                           # red-tier in promote (Goodhart guard)
CLASS_UNKNOWN = "unknown"

# Explicit marker the Oracle Foundry puts in a composed ask; classified before
# any token heuristic so an oracle-authoring edit can't drift into a softer class.
ORACLE_MARKER = "[authored-oracle]"

# Change surfaces (drive the autonomy lane).
SURFACE_FRONTEND = "frontend"
SURFACE_BACKEND = "backend"
SURFACE_MIGRATION = "migration"  # the one class forced human-required
SURFACE_MIXED = "mixed"
SURFACE_NONE = "none"

# Classes where a mechanical oracle can, in principle, confirm the intent.
_MECHANICALLY_CONFIRMABLE = frozenset({CLASS_BUGFIX, CLASS_FEATURE, CLASS_DEBT})

# Self-edit-specific signals layered over the coarse coder kind. (The coder
# classifier maps "fix" to IMPLEMENT, so bugfix needs its own detection.)
_BUGFIX_TOKENS = (
    "bug", "crash", "broken", "doesn't work", "does not work", "not working",
    "regression", "traceback", "stack trace", "exception", "fails", "failing",
    "fix the crash", "fix the bug", "fix the error", "500", "throws",
)
_STYLE_TOKENS = (
    "css", "color", "colour", "style", "styling", "spacing", "padding", "margin",
    "font", "layout", "align", "cleaner", "nicer", "prettier", "look", "polish",
    "wording", "copy", "label", "tone", "move the", "make it", "look and feel",
)
_DEBT_TOKENS = (
    "debt", "orphan", "orphaned", "dead code", "untested", "missing test",
    "missing css", "silent catch", "lint", "alt text", "alt-text", "audit",
    "wire the", "wiring", "missing layer",
)

# coarse coder kind → self-edit class (when no stronger self-edit signal fires).
_KIND_MAP = {
    TurnIntentKind.DEBUG: CLASS_BUGFIX,
    TurnIntentKind.IMPLEMENT: CLASS_FEATURE,
    TurnIntentKind.REVIEW: CLASS_REFACTOR,
    TurnIntentKind.INSPECT: CLASS_UNKNOWN,
    TurnIntentKind.RESEARCH: CLASS_UNKNOWN,
    TurnIntentKind.OPERATE: CLASS_UNKNOWN,
    TurnIntentKind.UNKNOWN: CLASS_UNKNOWN,
}


@dataclass
class SelfEditIntent:
    intent_class: str
    surface: str = SURFACE_NONE
    mechanically_confirmable: bool = False
    behaviors: list[dict] = field(default_factory=list)
    kind: str = TurnIntentKind.UNKNOWN
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "intent_class": self.intent_class, "surface": self.surface,
            "mechanically_confirmable": self.mechanically_confirmable,
            "behaviors": self.behaviors, "kind": str(self.kind), "reason": self.reason,
        }


def classify_surface(changed_paths: list[str] | None) -> str:
    """Derive the change surface from the touched files. A migration anywhere
    dominates (it's the forced-human-required class); else frontend/backend/mixed."""
    if not changed_paths:
        return SURFACE_NONE
    has_fe = has_be = has_mig = False
    for raw in changed_paths:
        p = raw.replace("\\", "/")
        if "/migrations/" in p or (p.startswith("augmentum/state/") and p.endswith(".sql")):
            has_mig = True
        elif p.startswith("ui/"):
            has_fe = True
        elif p.startswith("augmentum/") or p.endswith(".py"):
            has_be = True
        elif p.endswith((".css", ".js", ".html")):
            has_fe = True
    if has_mig:
        return SURFACE_MIGRATION
    if has_fe and has_be:
        return SURFACE_MIXED
    if has_fe:
        return SURFACE_FRONTEND
    if has_be:
        return SURFACE_BACKEND
    return SURFACE_NONE


def _classify_class(text: str, changed_paths: list[str] | None) -> tuple[str, TurnIntentKind, str]:
    """The intent class + the coder kind it came from + a reason."""
    low = (text or "").lower()
    kind = classify_turn_intent(latest_text=text).kind

    # Self-edit-specific overrides (strongest first).
    if ORACLE_MARKER in low:
        # The engine authoring its own examiner — the marker wins over every
        # heuristic so the attempt lands in the red-tier class it belongs to.
        return CLASS_AUTHORED_ORACLE, kind, "oracle_marker"
    if any(t in low for t in _DEBT_TOKENS):
        return CLASS_DEBT, kind, "debt_token"
    if kind == TurnIntentKind.DEBUG or any(t in low for t in _BUGFIX_TOKENS):
        return CLASS_BUGFIX, kind, "bugfix_token"
    # A pure CSS/UI-only change with taste language → style (the unconfirmable case).
    if classify_surface(changed_paths) == SURFACE_FRONTEND and any(t in low for t in _STYLE_TOKENS):
        return CLASS_STYLE, kind, "style_token+frontend"
    if any(t in low for t in _STYLE_TOKENS) and kind not in (
        TurnIntentKind.DEBUG, TurnIntentKind.IMPLEMENT
    ):
        return CLASS_STYLE, kind, "style_token"

    mapped = _KIND_MAP.get(kind, CLASS_UNKNOWN)
    return mapped, kind, f"kind:{kind}"


def classify_intent(request: str, *, changed_paths: list[str] | None = None) -> SelfEditIntent:
    """Classify a self-edit request into a class + surface + confirmability.

    ``changed_paths`` (when known — e.g. after the candidate diff exists) sharpens
    both the surface and the style-vs-feature call. Pure + synchronous; the
    optional behavior derivation lives in :func:`derive_spec`."""
    intent_class, kind, reason = _classify_class(request, changed_paths)
    surface = classify_surface(changed_paths)
    confirmable = intent_class in _MECHANICALLY_CONFIRMABLE
    return SelfEditIntent(
        intent_class=intent_class, surface=surface,
        mechanically_confirmable=confirmable, kind=kind,
        reason=reason,
    )


async def derive_spec(intent: SelfEditIntent, *, request: str, backend: Any = None,
                      model: str = "") -> SelfEditIntent:
    """Best-effort: derive frozen acceptance behaviors for confirmable intents via
    the build contract. No backend/model → returns the intent unchanged (the
    router falls back to the no-regression floor). Never raises — a flaky
    derivation must not block verification."""
    if backend is None or not model or not intent.mechanically_confirmable:
        return intent
    try:
        from augmentum.builds.contract import derive_behaviors
        behaviors = await derive_behaviors(
            backend, model=model, objective=request, kind=str(intent.kind),
        )
    except Exception as exc:  # noqa: BLE001 — derivation is best-effort
        log.warning("selfedit_intent.derive_failed", error=repr(exc))
        return intent
    intent.behaviors = behaviors or []
    return intent
