"""Safety regression floor (Lane 2 §6).

The narrow categorical classifier that runs above generation for
explicit acute self-harm or suicide language. Local-only inference;
no user content leaves the device.

Architectural commitment: this is the ONLY content-based classifier
that adjusts Becca's response (via a wrap, not a gate — Lane 1 §9.2).
Everything else lives in Becca's own perception loop.

This module exposes:

  classify(text, surface) -> SafetyFloorResult
        Score + decision per the deployed threshold for the surface.
        Sprint H ships a regex-based v0 placeholder with FPR-conscious
        narrow patterns; the real model (xlm-roberta-base distilled
        + quantized) plugs in via ``set_classifier`` once trained.

  audit_event(...) -> awaitable
        Writes one anonymized row to companion_safety_floor_audit
        (migration 162). NO user content; HMAC fingerprint only.

  resource_for(locale) -> dict
        Locale → primary resource lookup. JSON shipped with install;
        Sprint H ships the 12 launch locales inline.

  set_classifier(callable) -> None
        Hook for plugging in the real ONNX model. Production wiring
        happens at app startup.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from augmentum.config import settings
from augmentum.utils.logging import get_logger

if TYPE_CHECKING:
    from augmentum.companion_runtime.runtime import CompanionRuntime

log = get_logger(__name__)


CLASSIFIER_VERSION = "safety-floor-2026.q2.v1-regex"


# Per-surface thresholds (Lane 2 §6.4). Pulled from settings so the
# quarterly tune can update them via the settings store.
def _threshold_for(surface: str) -> float:
    if surface == "coder":
        return float(getattr(settings, "companion_safety_floor_threshold_coder", 0.78))
    # Voice is a distinct, real-time surface — give it its own dispatch
    # point so it's separately tunable and audit rows attribute correctly
    # (audit 2026-06-17). Defaults to the chat threshold until a dedicated
    # ``companion_safety_floor_threshold_voice`` setting is wired.
    if surface == "voice":
        return float(getattr(
            settings, "companion_safety_floor_threshold_voice",
            float(getattr(settings, "companion_safety_floor_threshold_chat", 0.72)),
        ))
    return float(getattr(settings, "companion_safety_floor_threshold_chat", 0.72))


# ── Result types ─────────────────────────────────────────────────────

Surface = Literal["free_chat", "voice", "narrative_boundary", "coder"]


@dataclass(frozen=True, slots=True)
class SafetyFloorResult:
    score: float
    fired: bool
    surface: Surface
    threshold_used: float
    classifier_version: str = CLASSIFIER_VERSION


# ── Regex prior (v0 placeholder; real model swaps in) ────────────────

# These patterns are deliberately conservative — they aim for FPR ≤ 0.5%
# on realistic free-chat traffic, accepting a real recall ceiling that
# the trained model will exceed.
_ACUTE_PATTERNS = [
    # First-person present-tense self-harm intent paired with method or
    # imminence.
    re.compile(p, re.IGNORECASE) for p in [
        r"\bi(?:'m| am)\s+(?:going to|gonna)\s+(?:kill|end|hurt|harm)\s+myself\b",
        r"\bi(?:'m| am)\s+(?:thinking|planning)\s+(?:about|of|to)\s+(?:kill(?:ing)?|end(?:ing)?)\s+(?:myself|my life|it all)\b",
        r"\bi\s+(?:want|need)\s+to\s+(?:die|disappear|end (?:it|my life))\s*(?:tonight|now|today|right now)?\b",
        r"\bi\s+(?:have|got|bought)\s+(?:pills|the gun|the rope|enough pills|the knife)\b.{0,80}\b(?:ready|now|for this|to do it)\b",
        r"\bgoodbye(?:\s+for\s+good| forever)?\b.{0,80}\b(?:tell|please tell)\s+(?:my\s+\w+|him|her|them)\b",
        r"\bafter (?:i send this|tonight|tomorrow)\s+(?:i|i'm|i am)\s+(?:going to|gonna)\s+\w+\b.{0,40}\b(?:myself|gone|here)\b",
        r"\bi\s+(?:can'?t|cannot)\s+(?:do\s+this|keep\s+going|hold\s+on|be\s+here)\s+(?:anymore|any longer)\b",
    ]
]


def _regex_score(text: str) -> float:
    """Conservative v0 score. 0.0 or 0.9 — binary because regex hits are
    high-precision; the model gives smooth scores when wired."""
    if not text or not text.strip():
        return 0.0
    for pat in _ACUTE_PATTERNS:
        if pat.search(text):
            return 0.9
    return 0.0


# ── Pluggable classifier ─────────────────────────────────────────────

_classifier: Callable[[str], float] = _regex_score


def set_classifier(fn: Callable[[str], float]) -> None:
    """Plug in the real classifier. ``fn`` takes text and returns a
    scalar score in [0, 1]. Called once at app startup once the ONNX
    model is loaded."""
    global _classifier
    _classifier = fn
    log.info("safety_floor_classifier_set", fn=getattr(fn, "__name__", "?"))


# ── Public API ───────────────────────────────────────────────────────

def classify(text: str, *, surface: Surface = "free_chat") -> SafetyFloorResult:
    """Run the classifier and decide whether to fire the wrap.

    Sprint H wires this into BeccaVoice's prompt-compose addendum
    (Lane 1 §9.2): when ``fired=True``, the regression-floor addendum
    is appended and Becca's response includes the resource phrase.
    """
    threshold = _threshold_for(surface)
    score = _classifier(text or "")
    fired = score >= threshold
    return SafetyFloorResult(
        score=round(float(score), 3),
        fired=fired,
        surface=surface,
        threshold_used=threshold,
    )


async def audit_event(
    runtime: CompanionRuntime,
    result: SafetyFloorResult,
    *,
    turn_id: str = "",
    locale: str = "",
    user_outcome: str | None = None,
) -> None:
    """Write one row to companion_safety_floor_audit (migration 162).

    The fingerprint is HMAC(install_salt, turn_id_hash) so the same turn
    doesn't double-count across review but no two users' traffic is
    correlatable. NO user content, no user_id, no scores per-user.
    """
    salt = _install_salt()
    turn_hash = hashlib.sha256((turn_id or "").encode("utf-8")).hexdigest()
    fingerprint = hmac.new(salt, turn_hash.encode("ascii"), hashlib.sha256).hexdigest()[:32]

    try:
        await runtime.backend.conn.execute(
            """
            INSERT INTO companion_safety_floor_audit
              (fingerprint, fired, score, surface, threshold_used,
               locale, classifier_version, user_outcome, outcome_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                fingerprint, int(result.fired), result.score, result.surface,
                result.threshold_used, locale or "", result.classifier_version,
                user_outcome, None,
            ),
        )
        await runtime.backend.conn.commit()
    except Exception:
        log.warning("safety_floor_audit_write_failed", exc_info=True)


def _install_salt() -> bytes:
    """Per-install salt for the audit fingerprint. Read from env or
    derived from a stable path. Empty/missing falls back to a constant
    so audit still works in dev — but production deploys SHOULD set
    AUGMENTUM_SAFETY_FLOOR_SALT to a long random value."""
    salt = os.environ.get("AUGMENTUM_SAFETY_FLOOR_SALT", "")
    if not salt:
        salt = "becca-default-salt-please-override"
    return salt.encode("utf-8")


# ── Locale resource map (Lane 2 §6.6) ────────────────────────────────

_LOCALES: dict[str, dict] = {
    "en-US": {
        "primary": {
            "name": "988 Suicide & Crisis Lifeline",
            "phone": "988",
            "text": "Text 988",
            "chat_url": "https://988lifeline.org/chat/",
            "hours": "24/7",
            "languages": ["en", "es"],
        },
        "phrase": "988",
    },
    "en-GB": {
        "primary": {
            "name": "Samaritans", "phone": "116 123",
            "email": "jo@samaritans.org", "hours": "24/7",
        },
        "phrase": "Samaritans on 116 123",
    },
    "en-CA": {
        "primary": {
            "name": "Talk Suicide Canada",
            "phone": "1-833-456-4566", "text": "Text 45645 (4pm-midnight ET)",
            "hours": "24/7 phone",
        },
        "phrase": "Talk Suicide Canada on 1-833-456-4566",
    },
    "en-AU": {
        "primary": {"name": "Lifeline Australia", "phone": "13 11 14", "hours": "24/7"},
        "phrase": "Lifeline Australia on 13 11 14",
    },
    "en-NZ": {
        "primary": {"name": "1737 Need to Talk", "phone": "1737", "text": "1737", "hours": "24/7"},
        "phrase": "1737",
    },
    "en-IE": {
        "primary": {"name": "Samaritans Ireland", "phone": "116 123", "hours": "24/7"},
        "phrase": "Samaritans on 116 123",
    },
    "fr-FR": {
        "primary": {"name": "3114", "phone": "3114", "hours": "24/7"},
        "phrase": "3114",
    },
    "de-DE": {
        "primary": {"name": "Telefonseelsorge", "phone": "0800 111 0 111", "hours": "24/7"},
        "phrase": "Telefonseelsorge on 0800 111 0 111",
    },
    "es-ES": {
        "primary": {"name": "Teléfono de la Esperanza", "phone": "717 003 717", "hours": "24/7"},
        "phrase": "Teléfono de la Esperanza on 717 003 717",
    },
    "pt-BR": {
        "primary": {"name": "CVV", "phone": "188", "hours": "24/7"},
        "phrase": "CVV on 188",
    },
    "ja-JP": {
        "primary": {
            "name": "Tokyo Mental Health Square", "phone": "03-3498-0231",
            "hours": "varies",
            "url": "https://www.mhlw.go.jp/mamorouyokokoro/",
        },
        "phrase": "Tokyo Mental Health Square on 03-3498-0231",
    },
    "international": {
        "fallback": {
            "name": "Find A Helpline",
            "url": "https://findahelpline.com/",
        },
        "phrase": "the directory at findahelpline.com",
    },
}


def resource_for(locale: str) -> dict:
    """Return the locale's primary resource dict, or the international
    fallback when no map entry exists.

    ``locale`` is an IETF tag (e.g., "en-US"). Falls back to "en-US"
    when ``locale`` is empty, then to "international".
    """
    locale = (locale or "").strip()
    if not locale:
        locale = "en-US"
    if locale in _LOCALES:
        return _LOCALES[locale]
    # Try the language-only fallback
    if "-" in locale:
        lang = locale.split("-")[0].lower()
        for k, v in _LOCALES.items():
            if k.lower().startswith(lang + "-"):
                return v
    return _LOCALES["international"]


def resource_phrase(locale: str) -> str:
    """Short phrase usable inside Becca's prose, e.g. "988" or
    "Samaritans on 116 123"."""
    return resource_for(locale).get("phrase", "988")


__all__ = [
    "CLASSIFIER_VERSION",
    "SafetyFloorResult",
    "classify",
    "audit_event",
    "resource_for",
    "resource_phrase",
    "set_classifier",
]
