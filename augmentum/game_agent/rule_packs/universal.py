"""Universal (game-agnostic) built-in reflex rules.

Unlike the per-game packs, these register on EVERY bridged session —
they encode interface physics that hold across games, keyed only on
generic signals (input_ack effect scores, ``*_text`` overlay probes).

Rule: dead-nav-during-dialog ("free turn")
------------------------------------------
While a dialog/text box is open, most games swallow movement input —
the press acks with effect_score ≈ 0 and nothing happens. The agent
used to burn an LLM turn discovering that, then often mis-learn it as
"navigation is broken". This rule intercepts the pattern at reflex
speed: nav press with ~zero effect while decoded text is on screen →
press confirm once (the input the screen actually wants). The freed
LLM turn sees ``reflex_did=confirm`` + the fx line and learns the
lesson without paying for it.
"""

from __future__ import annotations

from typing import Any

from augmentum.game_agent.progress import DEAD_INPUT_THRESHOLD
from augmentum.game_agent.rules import Rule, RuleMatch
from augmentum.game_agent.schema import PlanAction

# Effect scores below this are "the press did nothing". Real screen
# changes measure in the hundreds-to-thousands; idle-animation noise
# stays low double-digits on GBA-scale frames.
#
# Canonical definition lives in progress.py — the reflex layer and the
# end-of-session scorer MUST agree on what "that press worked" means, or
# the score would grade a run by a different rule than the agent acted
# on. Alias kept so this pack's local reads stay readable.
_DEAD_EFFECT_THRESHOLD = DEAD_INPUT_THRESHOLD


def dead_nav_during_dialog_rule() -> Rule:
    """Build the interceptor rule (fresh closure state per session)."""

    state: dict[str, Any] = {"last_handled": None}

    def _predicate(
        window: list[dict[str, Any]], _caps: Any
    ) -> RuleMatch | None:
        last_ack: dict[str, Any] | None = None
        dialog_open = False
        # One backwards pass: newest ack + freshest *_text state.
        for entry in reversed(window):
            if entry.get("kind") != "event":
                continue
            payload = entry.get("payload") or {}
            data = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(data, dict):
                continue
            if last_ack is None and data.get("event") == "input_ack":
                last_ack = data
            probes = data.get("probes")
            if isinstance(probes, dict):
                for key, value in probes.items():
                    if key.endswith("_text") and isinstance(value, str) and value.strip():
                        dialog_open = True
                        break
                if dialog_open:
                    break
        if last_ack is None or not dialog_open:
            return None
        button = str(last_ack.get("button") or "")
        if not button.startswith("nav_"):
            return None
        if int(last_ack.get("effect_score") or 0) >= _DEAD_EFFECT_THRESHOLD:
            return None
        marker = last_ack.get("request_id") or id(last_ack)
        if state["last_handled"] == marker:
            return None  # already answered this exact dead press
        state["last_handled"] = marker
        return RuleMatch(
            rule_id="universal:dead-nav-during-dialog",
            matched={
                "button": button,
                "effect_score": int(last_ack.get("effect_score") or 0),
                "reason": "movement swallowed by open dialog -> confirm",
            },
            actions=[PlanAction(semantic="confirm", duration_ms=120)],
        )

    return Rule(
        rule_id="universal:dead-nav-during-dialog",
        predicate=_predicate,
        priority=3,  # below model-authored reflexes (5), above pack rules
        cooldown_ms=1200,
    )


__all__ = ["dead_nav_during_dialog_rule"]
