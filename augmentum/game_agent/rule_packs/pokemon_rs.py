"""Fast-path rule pack for Pokémon Ruby/Sapphire (Gen-3, GBA).

The slow-path LLM is a strategic planner -- it reasons about objectives,
maps, party state. It runs at ~10-20 s per turn on a local 27B vision
model, which is far too slow for trivial reflex behavior like
"advance dialogue" or "continue battle text".

This module ships **reactive rules** that fire on probe-tick / input-ack
events (~250 ms latency) so the agent feels responsive on common
keep-pressing-A scenarios. Each rule:

* fires *only* on a positive trigger (no "I'm stuck, try something"
  heuristics -- those belong in the slow path);
* emits actions the surface already declares as valid semantics;
* respects a per-rule cooldown so a flapping condition doesn't spam
  the input wire.

V1 contains a single rule -- :func:`_auto_continue_dialog` -- because
it's the highest-leverage single behavior across all Pokémon games:
every cutscene, every dialog, every battle has a "press A to advance"
loop, and the slow path was spending 10-20 s per A-press to do this
deterministic chore. The rule chains follow-up A presses whenever
the previous A had visible effect AND the player is stationary,
self-terminating the moment effect_score drops below threshold
(textbox closed -> no pixel change -> rule no longer fires).
"""

from __future__ import annotations

from typing import Any

from augmentum.game_agent.rules import Rule, RuleEngine, RuleMatch
from augmentum.game_agent.schema import PlanAction, SurfaceCapsPayload

# ── Thresholds, all calibrated empirically ─────────────────────────────
#
# effect_score is sum-of-absolute-differences across 8x8 RGBA samples
# (see _fingerprintDelta in emulator-iframe-template.js). On the GBA
# 240x160 framebuffer, calibration:
#   ~ 0     identical
#   ~ 30    noise / single-frame animation tick
#   ~ 100   small UI change (cursor, single sprite)
#   ~ 250+  significant change (dialog appeared, menu opened, scene cut)
#
# We use 80 as the "real change" threshold so we don't refire on
# minor sprite shimmer.
_DIALOG_EFFECT_THRESHOLD = 80

# How long the auto-advance press is held. 120 ms matches the
# universal minimum we use everywhere else for input registration
# (~7 frames at 60 Hz, past every game's title-screen debounce).
_DIALOG_PRESS_DURATION_MS = 120

# Cooldown between auto-A presses. Total wall-clock between presses is
# this + the iframe-side hold (120 ms) + the post-release settle the
# input-ack tracer adds (150 ms) ~= 470 ms. That's a comfortable
# dialogue-advance cadence -- fast enough to clear long cutscenes but
# slow enough to leave the slow path a chance to interject if the
# situation has changed.
_DIALOG_COOLDOWN_MS = 200


def _is_player_stationary(window: list[dict[str, Any]], lookback: int = 6) -> bool:
    """Has the player not moved across the last few probe-tick events?

    Walks the event window newest-first, picks up to ``lookback`` recent
    ``ram`` probe events that carry both ``player_x`` and ``player_y``,
    and returns True iff every captured coordinate pair matches the
    most recent one. Returns ``False`` (not stationary, refuse to fire)
    if we have fewer than 2 readings -- the rule is conservative when
    it doesn't have evidence.

    ROOT CAUSE for the conservative default:
      A textbox can appear mid-walk on a step trigger. If we treat
      "no readings" as stationary and fire, we'd auto-mash A while the
      character is actually mid-stride, which races overworld input
      handling and can produce surprise jumps when the textbox closes.
    """

    coords: list[tuple[int, int]] = []
    for entry in reversed(window):
        if entry.get("kind") != "event":
            continue
        payload = entry.get("payload") or {}
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, dict):
            continue
        if data.get("event") != "ram":
            continue
        probes = data.get("probes")
        if not isinstance(probes, dict):
            continue
        if "player_x" not in probes or "player_y" not in probes:
            continue
        coords.append((int(probes["player_x"]), int(probes["player_y"])))
        if len(coords) >= lookback:
            break
    if len(coords) < 2:
        return False
    head = coords[0]
    return all(c == head for c in coords)


def _auto_continue_dialog(
    window: list[dict[str, Any]],
    caps: SurfaceCapsPayload,
) -> RuleMatch | None:
    """Chain another confirm-press when the previous one had real visible effect.

    Fires when:
    - Most recent log entry is an ``input_ack`` event for the confirm
      button (semantic "confirm"; legacy surfaces may use "a") with
      ``effect_score`` above :data:`_DIALOG_EFFECT_THRESHOLD` (something
      actually changed on screen after the press).
    - Player position has been stable across recent probe ticks (we're
      not in mid-step; the press was advancing text, not interacting
      with the overworld).

    The combination "confirm had effect AND we're stationary" is a
    strong signal for active dialogue / cutscene / battle text. By
    pressing confirm again, we let the rule engine handle long text
    strings (dozens of presses in Pokémon) without waking the slow path
    once per press. The rule self-terminates when:
    - the next press has effect_score below threshold (textbox closed),
    - the player position changes (back in overworld), or
    - the slow path emits a non-confirm action (higher-priority intent
      overrides the rule on the next tick).

    NOTE: the predicate previously checked ``button == "a"`` and emitted
    semantic ``"a"``. GBA emulatorjs sessions expose ``"confirm"`` as the
    A-button semantic (it maps to the hardware A at the translation
    layer). The old check caused this rule to silently return None on
    every tick for those sessions, leaving dialog advance entirely to the
    reactive dead-nav rule.
    """

    if not window:
        return None
    # Accept either semantic name — "confirm" for emulatorjs/GBA surfaces,
    # "a" for legacy surfaces that expose raw button names.
    confirm_sem = "confirm" if "confirm" in caps.semantic_inputs else (
        "a" if "a" in caps.semantic_inputs else None
    )
    if confirm_sem is None:
        return None
    last = window[-1]
    if last.get("kind") != "event":
        return None
    payload = last.get("payload") or {}
    if not isinstance(payload, dict):
        return None
    data = payload.get("data")
    if not isinstance(data, dict):
        return None
    if data.get("event") != "input_ack":
        return None
    # Match whichever semantic the surface uses for the confirm/A button.
    if data.get("button") not in ("confirm", "a"):
        return None
    effect = data.get("effect_score")
    if not isinstance(effect, int | float):
        return None
    if effect < _DIALOG_EFFECT_THRESHOLD:
        return None
    if not _is_player_stationary(window):
        return None
    return RuleMatch(
        rule_id="auto_continue_dialog",
        matched={
            "trigger": "input_ack_confirm_high_effect",
            "effect_score": int(effect),
        },
        actions=[PlanAction(semantic=confirm_sem, duration_ms=_DIALOG_PRESS_DURATION_MS)],
    )


def build_rule_engine() -> RuleEngine:
    """Construct a fresh :class:`RuleEngine` with the Pokémon RS pack."""

    engine = RuleEngine()
    engine.register(
        Rule(
            rule_id="auto_continue_dialog",
            predicate=_auto_continue_dialog,
            priority=10,
            cooldown_ms=_DIALOG_COOLDOWN_MS,
        )
    )
    return engine


__all__ = ["build_rule_engine"]
