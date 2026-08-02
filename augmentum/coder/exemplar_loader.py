"""Intent-keyed exemplar loader for coder priming.

A fresh model arriving in a coder session has no memory of how the
project expects work to be shaped. The exemplar library gives it one
worked example matched to the classified turn intent — concrete enough
that a 4B model can imitate the pattern and short enough that a
frontier model can skim it.

Exemplars live as markdown files in ``augmentum/coder/exemplars/``.
They're loaded once at import time and cached. ``UNKNOWN`` intent and
missing files return ``""`` so callers can fall back gracefully.
"""
from __future__ import annotations

from pathlib import Path

import structlog

from augmentum.modes.coder.intent import TurnIntentKind

log = structlog.get_logger(__name__)

_EXEMPLARS_DIR = Path(__file__).parent / "exemplars"

# Intent kinds that map to a specific exemplar file. UNKNOWN deliberately
# falls back to IMPLEMENT (safe superset for write-capable work) so a
# fresh model still sees a worked turn shape. If the IMPLEMENT exemplar
# is also missing, callers get "" and skip exemplar injection entirely.
_INTENT_FILES: dict[TurnIntentKind, str] = {
    TurnIntentKind.INSPECT:   "inspect.md",
    TurnIntentKind.REVIEW:    "review.md",
    TurnIntentKind.RESEARCH:  "research.md",
    TurnIntentKind.IMPLEMENT: "implement.md",
    TurnIntentKind.DEBUG:     "debug.md",
    TurnIntentKind.OPERATE:   "operate.md",
}

_CACHE: dict[TurnIntentKind, str] = {}


def _load_all() -> None:
    """Read every exemplar file once at import time."""
    for intent, filename in _INTENT_FILES.items():
        path = _EXEMPLARS_DIR / filename
        try:
            _CACHE[intent] = path.read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            log.warning("coder_exemplar_missing", intent=intent.value, path=str(path))
            _CACHE[intent] = ""
        except OSError as exc:
            log.warning(
                "coder_exemplar_read_failed",
                intent=intent.value,
                path=str(path),
                error=str(exc),
            )
            _CACHE[intent] = ""


_load_all()


def load_exemplar(intent_kind: TurnIntentKind | None) -> str:
    """Return the exemplar markdown for an intent, or ``""``.

    ``None`` and ``UNKNOWN`` fall back to the IMPLEMENT exemplar — a
    write-capable shape is the safe default for an unclassified turn.
    """
    if intent_kind is None or intent_kind == TurnIntentKind.UNKNOWN:
        return _CACHE.get(TurnIntentKind.IMPLEMENT, "")
    return _CACHE.get(intent_kind, "")


def reload_exemplars() -> None:
    """Re-read all exemplar files. For tests + hot-reload only."""
    _CACHE.clear()
    _load_all()
