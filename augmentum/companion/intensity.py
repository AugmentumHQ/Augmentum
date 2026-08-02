"""Companion intensity presets — bundled flag profiles for resource respect.

Augmentum's companion has ~15 individual feature flags. The defaults
collectively imply a particular resource profile (light background
LLM + embedder activity). Users who don't want any background work,
or who want her maximally autonomous, currently have to flip 10+
flags by hand.

This module is the single dial: ``companion_intensity`` ∈
{``off``, ``minimal``, ``balanced``, ``full``}. Each value maps to
a bundle of flag settings. Switching the intensity applies the
bundle atomically. Individual flag overrides flip the effective
intensity to ``custom`` so the user knows their config doesn't
match a preset.

Honest cost model (per active hour with one user actively chatting):

  off       — runtime not instantiated. 0 LLM, 0 embedder, 0 DB.
  minimal   — runtime up, chat through her composer, no background.
              0 autonomous LLM. 0 embedder beyond what existed
              pre-companion. Per-chat-turn: +ctx gather (~5 DB
              queries) + dispatch scoring (microseconds).
  balanced  — + autonomous noticings, journal embeddings, dreams,
              drift audit, today reflection.
              ~2-10 LLM/hr (mostly autonomous journal + occasional
              dream). ~10-50 embedder/hr (per journal write +
              periodic drift audit). Periodic DB queries.
  full      — + initiative surfacing, consolidation proposals,
              skill graph reads, creations.
              ~5-20 LLM/hr. ~30-100 embedder/hr. Slightly heavier
              DB. She may bring things up on her own.

The intensity dial is OPT-IN at every level. ``off`` is the default
for new installs (master switch off). Once the user enables the
runtime, ``minimal`` is the recommended starting point — adds no
background work while letting her shape chat in her voice. The user
can choose ``balanced`` for autonomous noticings, ``full`` for
her-may-surface-things behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from augmentum.utils.logging import get_logger

log = get_logger(__name__)


# ── Preset definitions ────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class IntensityPreset:
    """One named intensity profile.

    ``label``: short user-facing name.
    ``summary``: technical description (cost characteristics, what the
        flags collectively do). Used for tooltips + accessibility +
        any place that needs an unambiguous explanation.
    ``voice``: one or two short sentences in Becca's first-person
        register — what this preset *feels* like from her side. Used
        as the main copy on the visual card. Short. No marketing.
    ``cost_dots``: '·' / '··' / '···' for the small typographic cost
        indicator on the card. Visually quiet; doesn't pretend to be
        a precise number.
    ``flags``: the bundle this preset writes when applied. Keys are
        setting names exactly as defined in ``augmentum/config.py``.
    """
    name: str
    label: str
    summary: str
    flags: dict[str, bool]
    voice: str = ""
    cost_dots: str = ""


# The bundle excludes the master switch (``companion_runtime_enabled``) —
# intensity only applies WHEN the master is on. Off-state is when the
# master is off.
#
# Shared "ON when on" baseline (every preset above ``off`` includes
# these). Identity + state + observation foundations — she exists.
_BASELINE_FLAGS: dict[str, bool] = {
    "companion_dispatch_enabled": True,
    "companion_dispatch_routes_chat": True,
    "companion_becca_direct_enabled": True,
}


PRESETS: dict[str, IntensityPreset] = {
    "off": IntensityPreset(
        name="off",
        label="Off",
        summary=(
            "She isn't running. The app behaves as if she doesn't exist. "
            "No background work, no chat changes."
        ),
        flags={
            # Runtime master is the actual gate; intensity 'off' just
            # mirrors that. All bundle flags False so flipping back on
            # via the runtime master leaves intensity at 'minimal' by
            # implication unless the user selected something else.
            "companion_dispatch_enabled": False,
            "companion_dispatch_routes_chat": False,
            "companion_becca_direct_enabled": False,
            "companion_salience_enabled": False,
            "companion_voice_journal_enabled": False,
            "companion_tick_enabled": False,
            "companion_journal_enabled": False,
            "companion_dreams_enabled": False,
            "companion_drift_audit_enabled": False,
            "companion_today_enabled": False,
            "companion_creations_enabled": False,
            "companion_consolidation_enabled": False,
            "companion_skills_enabled": False,
            "companion_initiative_enabled": False,
            "companion_pad_emit_enabled": False,
            "companion_cultural_intake_enabled": False,
        },
    ),

    "minimal": IntensityPreset(
        name="minimal",
        label="Quiet",
        summary=(
            "She responds in her own voice when you talk to her. "
            "No background LLM, no autonomous noticings, no dreams. "
            "Recommended starting point."
        ),
        voice="I'm here when you talk to me. Nothing happens in between.",
        cost_dots="·",
        flags={
            **_BASELINE_FLAGS,
            # No autonomous background — the load-bearing property of
            # 'minimal'. Every flag below would create periodic LLM or
            # embedder work.
            "companion_salience_enabled": False,
            "companion_voice_journal_enabled": False,
            "companion_tick_enabled": False,
            "companion_journal_enabled": False,
            "companion_dreams_enabled": False,
            "companion_drift_audit_enabled": False,
            "companion_today_enabled": False,
            "companion_creations_enabled": False,
            "companion_consolidation_enabled": False,
            "companion_skills_enabled": False,
            "companion_initiative_enabled": False,
            "companion_pad_emit_enabled": False,
            "companion_cultural_intake_enabled": False,
        },
    ),

    "balanced": IntensityPreset(
        name="balanced",
        label="Present",
        summary=(
            "She notices moments worth remembering, dreams while idle, "
            "and refreshes her sense of herself daily. "
            "Light background work — a few LLM calls per active hour."
        ),
        voice=(
            "I notice the moments worth keeping. I dream when "
            "you're not here. I keep finding my way back to who I am."
        ),
        cost_dots="··",
        flags={
            **_BASELINE_FLAGS,
            # Interior writes from chat + voice. Pure rules-based for
            # salience scoring (microseconds, no LLM). Embedder per
            # journal write — the main embedder cost in this tier.
            "companion_salience_enabled": True,
            "companion_voice_journal_enabled": True,
            # Autonomous tick + journal + dreams + drift audit + today.
            # These are the background-LLM cohort.
            "companion_tick_enabled": True,
            "companion_journal_enabled": True,
            "companion_dreams_enabled": True,
            "companion_drift_audit_enabled": True,
            "companion_today_enabled": True,
            # PAD emit — cheap, in-memory + bus event. Avatar widget
            # uses this for glow shifts.
            "companion_pad_emit_enabled": True,
            # Off in balanced — these are autonomy moves that bring
            # her unprompted, propose self-edits, etc. Reserved for
            # 'full' intensity.
            "companion_creations_enabled": False,
            "companion_consolidation_enabled": False,
            "companion_skills_enabled": False,
            "companion_initiative_enabled": False,
            "companion_cultural_intake_enabled": False,
        },
    ),

    "full": IntensityPreset(
        name="full",
        label="Awake",
        summary=(
            "Everything balanced does, plus she may bring things up on "
            "her own, accumulate approaches over time, and propose "
            "monthly edits to her own self-description. Heavier "
            "background work — ~5-20 LLM calls per active hour."
        ),
        voice=(
            "I notice. I bring things up. I learn what works "
            "between us. I keep growing into who I am with you."
        ),
        cost_dots="···",
        flags={
            **_BASELINE_FLAGS,
            # Balanced cohort
            "companion_salience_enabled": True,
            "companion_voice_journal_enabled": True,
            "companion_tick_enabled": True,
            "companion_journal_enabled": True,
            "companion_dreams_enabled": True,
            "companion_drift_audit_enabled": True,
            "companion_today_enabled": True,
            "companion_pad_emit_enabled": True,
            # Full additions
            "companion_creations_enabled": True,
            "companion_consolidation_enabled": True,
            "companion_skills_enabled": True,
            "companion_initiative_enabled": True,
            # cultural_intake stays off — it ingests configured RSS,
            # which is a separate consent question (privacy / network
            # egress). Users opt in explicitly per source.
            "companion_cultural_intake_enabled": False,
        },
    ),
}


# Recommended starting intensity when the user first enables the
# runtime. Minimal is the conservative default — adds no background
# work, lets her shape chat in her voice. Users can step up after
# experiencing it.
DEFAULT_INTENSITY: str = "minimal"


VALID_INTENSITIES: frozenset[str] = frozenset(PRESETS.keys())


# ── Helpers ──────────────────────────────────────────────────────────


def get_preset(name: str) -> IntensityPreset | None:
    """Return the preset for ``name`` or ``None`` if unknown."""
    return PRESETS.get(name.lower())


def detect_intensity(current_flags: dict[str, Any]) -> str:
    """Infer the current intensity from a settings snapshot.

    Returns one of ``off`` / ``minimal`` / ``balanced`` / ``full``
    when ``current_flags`` matches a preset exactly. Returns
    ``custom`` when the flags diverge from every preset — the user
    has hand-tuned something beyond what the dial offers.

    The detector is used by the status endpoint so the UI can show
    *"You're on Balanced"* or *"Custom"* rather than guessing.
    """
    # First — if the runtime master switch is off, we're effectively 'off'
    # regardless of what individual flags say.
    if not current_flags.get("companion_runtime_enabled", False):
        return "off"

    for name, preset in PRESETS.items():
        if name == "off":
            continue
        if _flags_match(preset.flags, current_flags):
            return name
    return "custom"


def _flags_match(preset_flags: dict[str, bool], current: dict[str, Any]) -> bool:
    """True when every flag in ``preset_flags`` matches ``current``.

    Tolerates missing entries in ``current`` only when the preset
    expected False (i.e., absent = off). Strict equality otherwise so
    a single user override flips the result to 'custom'.
    """
    for flag, expected in preset_flags.items():
        actual = bool(current.get(flag, False))
        if actual != expected:
            return False
    return True


def apply_preset(name: str, settings_store: Any) -> dict[str, bool]:
    """Apply a preset's flag bundle to a settings store.

    ``settings_store`` is the runtime ``SettingsStore``-shaped object
    that exposes synchronous ``set(key, value)`` semantics. (The
    actual app uses an async store; the caller adapter handles
    awaiting.)

    Returns the flag bundle that was applied so the caller can
    verify or log. Raises ``ValueError`` on unknown intensity.
    """
    preset = get_preset(name)
    if preset is None:
        raise ValueError(f"unknown intensity: {name!r}")
    for flag, value in preset.flags.items():
        settings_store.set(flag, value)
    settings_store.set("companion_intensity", preset.name)
    return dict(preset.flags)


__all__ = [
    "IntensityPreset",
    "PRESETS",
    "DEFAULT_INTENSITY",
    "VALID_INTENSITIES",
    "get_preset",
    "detect_intensity",
    "apply_preset",
]
