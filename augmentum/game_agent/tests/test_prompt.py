"""Prompt + plan-output parser tests."""

from __future__ import annotations

import json

import pytest

from augmentum.game_agent.prompt import (
    SLOW_PATH_PROMPT,
    PlanParseError,
    build_slow_path_inputs,
    parse_plan_output,
)
from augmentum.game_agent.schema import SurfaceCapsPayload


def _caps() -> SurfaceCapsPayload:
    return SurfaceCapsPayload(
        semantic_inputs=["noop", "advance"],
        log_schema="mock.v1",
        observation_modalities=["log"],  # type: ignore[list-item]
    )


def test_prompt_body_is_agnostic() -> None:
    """@example: the strict prompt names no specific game."""

    forbidden = ["minecraft", "luanti", "dolphin", "stardew", "pokemon"]
    for word in forbidden:
        assert word.lower() not in SLOW_PATH_PROMPT.lower()


def test_inputs_block_includes_caps_and_objective() -> None:
    """@example: the runtime-inputs block injects the caps + objective verbatim."""

    block = build_slow_path_inputs(
        surface_kind="mock",
        caps=_caps(),
        objective="find food",
        state="",
        live_log_tail=[{"t": 0, "kind": "event"}],
        n_frames=0,
    )
    assert "OBJECTIVE: find food" in block
    assert '"semantic_inputs":["noop","advance"]' in block
    assert "FRAMES: <not provided this turn>" in block


def test_parse_plan_accepts_clean_json() -> None:
    """@example: a clean reply parses into a PlanPayload."""

    raw = json.dumps(
        {
            "observations": ["nothing yet"],
            "state_update": "still observing",
            "actions": [{"semantic": "noop", "duration_ms": 100}],
            "confidence": 0.5,
            "next_check_in_ms": 1000,
        }
    )
    plan = parse_plan_output(raw, _caps())
    assert plan.actions[0].semantic == "noop"


def test_parse_plan_strips_markdown_fences() -> None:
    """@example: models that wrap output in ```json fences still parse."""

    body = json.dumps(
        {
            "observations": [],
            "state_update": "",
            "actions": [],
            "confidence": 0.1,
            "next_check_in_ms": 500,
        }
    )
    raw = f"```json\n{body}\n```"
    plan = parse_plan_output(raw, _caps())
    assert plan.actions == []


def test_parse_plan_rejects_unknown_semantic() -> None:
    """@example: an action whose semantic is not in caps is rejected.

    ROOT CAUSE:
      Without this check, a slow-path model could emit an action no
      adapter can resolve; the orchestrator would then fail at
      resolver.apply time with a less precise error. We reject at
      parse time so the failure surfaces with the parse-error
      reason intact.
    """

    raw = json.dumps(
        {
            "observations": [],
            "state_update": "",
            "actions": [{"semantic": "flee", "duration_ms": 100}],
            "confidence": 0.5,
            "next_check_in_ms": 500,
        }
    )
    with pytest.raises(PlanParseError):
        parse_plan_output(raw, _caps())


def test_parse_plan_rejects_non_json() -> None:
    """@example: free-form prose is rejected with PlanParseError."""

    with pytest.raises(PlanParseError):
        parse_plan_output("I think we should wait and see.", _caps())


# ── OVERLAY block (Phase A: structured world state in the prompt) ──────


def test_inputs_block_omits_overlay_when_absent() -> None:
    """@example: surfaces without probes emit no OVERLAY line at all.

    ROOT CAUSE:
      A stray empty ``OVERLAY:`` line in the prompt would invite the
      model to hallucinate fields. Omitting the section entirely
      when no probes have fired keeps the contract honest.
    """

    block = build_slow_path_inputs(
        surface_kind="mock", caps=_caps(), objective="x",
        state="", live_log_tail=[], n_frames=0,
        overlay=None,
    )
    assert "OVERLAY" not in block

    block_empty = build_slow_path_inputs(
        surface_kind="mock", caps=_caps(), objective="x",
        state="", live_log_tail=[], n_frames=0,
        overlay={},
    )
    assert "OVERLAY" not in block_empty


def test_inputs_block_renders_overlay_above_live_log() -> None:
    """@example: OVERLAY block sits above LIVE_LOG_TAIL.

    The model is told to prefer OVERLAY for the freshest values; the
    physical order in the prompt reinforces that priority. Recency
    bias on most LLMs reads later tokens as authoritative, so OVERLAY
    must come BEFORE the (potentially noisy, long-history) tail.
    """

    block = build_slow_path_inputs(
        surface_kind="gba", caps=_caps(), objective="walk south",
        state="", live_log_tail=[{"t": 0, "kind": "event"}],
        n_frames=1,
        overlay={"player_x": 12, "player_y": 7, "map_num": 4},
    )
    assert "OVERLAY:" in block
    overlay_idx = block.find("OVERLAY:")
    log_idx = block.find("LIVE_LOG_TAIL:")
    assert 0 <= overlay_idx < log_idx
    # JSON shape sanity: keys sorted (stable hashing for KV cache reuse).
    assert '"map_num":4' in block
    assert '"player_x":12' in block


# ── FRAMES block (Phase B: temporal stacking) ─────────────────────────


def test_frames_block_omitted_when_n_frames_zero() -> None:
    """@example: n_frames=0 renders the honest 'not provided' line."""

    block = build_slow_path_inputs(
        surface_kind="mock", caps=_caps(), objective="x",
        state="", live_log_tail=[], n_frames=0,
    )
    assert "FRAMES: <not provided this turn>" in block
    # No "oldest -> newest" phrasing leaks through:
    assert "oldest" not in block


def test_frames_block_singular_when_n_frames_one() -> None:
    """@example: n_frames=1 reports a single attachment without the sequence note."""

    block = build_slow_path_inputs(
        surface_kind="mock", caps=_caps(), objective="x",
        state="", live_log_tail=[], n_frames=1,
    )
    assert "FRAMES: <1 attached>" in block
    assert "oldest" not in block


# ── INPUT_HINTS block (Phase G: universal control schema) ────────────


def _caps_with_hints() -> SurfaceCapsPayload:
    return SurfaceCapsPayload(
        semantic_inputs=["confirm", "cancel", "menu", "nav_up", "registered"],
        log_schema="pokemon_rs.v1",
        observation_modalities=["log", "frame"],  # type: ignore[list-item]
        input_hints={
            "confirm":    "Advance dialog one box.",
            "cancel":     "Cancel / back out.",
            "menu":       "Open main menu.",
            "nav_up":     "Walk one tile north.",
            "registered": "Use the registered key item.",
        },
        controller_profile="gba",
        game_profile="pokemon_rs",
    )


def test_inputs_block_omits_hints_when_absent() -> None:
    """@example: a surface without hints doesn't get a stray INPUT_HINTS line.

    ROOT CAUSE:
      An empty INPUT_HINTS block would invite the model to fill in
      meanings it can't actually know. Omitting it entirely keeps the
      contract honest.
    """

    block = build_slow_path_inputs(
        surface_kind="mock", caps=_caps(), objective="x",
        state="", live_log_tail=[], n_frames=0,
    )
    assert "INPUT_HINTS" not in block


def test_inputs_block_renders_hints_above_objective() -> None:
    """@example: INPUT_HINTS block sits between SURFACE_CAPS and OBJECTIVE.

    The cache-locality argument: hints are constant across the entire
    session (one controller+game composition). Putting them near the
    top means they get tokenized into the KV cache prefix once and
    never invalidate.
    """

    block = build_slow_path_inputs(
        surface_kind="gba", caps=_caps_with_hints(), objective="x",
        state="", live_log_tail=[], n_frames=0,
    )
    caps_idx = block.find("SURFACE_CAPS:")
    hints_idx = block.find("INPUT_HINTS:")
    obj_idx = block.find("OBJECTIVE:")
    assert 0 <= caps_idx < hints_idx < obj_idx


def test_inputs_block_hints_sort_universal_first() -> None:
    """@example: universal verbs render before game-specific extensions.

    Universal-first ordering keeps the most-portable verbs in the
    visual hot-spot of the hint block. The agent reaches for the
    portable verb if both exist for the same effect.
    """

    block = build_slow_path_inputs(
        surface_kind="gba", caps=_caps_with_hints(), objective="x",
        state="", live_log_tail=[], n_frames=0,
    )
    cancel_idx = block.find(" cancel ")
    confirm_idx = block.find(" confirm ")
    menu_idx = block.find(" menu ")
    nav_idx = block.find(" nav_up ")
    registered_idx = block.find(" registered ")
    # All universal verbs appear before the game-specific "registered".
    assert max(confirm_idx, cancel_idx, menu_idx, nav_idx) < registered_idx
    # confirm precedes cancel alphabetically within the universal block.
    assert cancel_idx < confirm_idx


def test_inputs_block_caps_includes_profile_ids_when_present() -> None:
    """@example: SURFACE_CAPS gains controller_profile + game_profile when set.

    Lets session replay / journal entries reference WHICH profile shaped
    the vocabulary the model was working with.
    """

    block = build_slow_path_inputs(
        surface_kind="gba", caps=_caps_with_hints(), objective="x",
        state="", live_log_tail=[], n_frames=0,
    )
    assert '"controller_profile":"gba"' in block
    assert '"game_profile":"pokemon_rs"' in block


def test_frames_block_announces_temporal_ordering_when_multi() -> None:
    """@example: n_frames>=2 tells the model the frames are time-ordered.

    ROOT CAUSE:
      Without an explicit ordering hint, models attend to multi-image
      prompts as unordered evidence and miss the "what changed?"
      signal — which is the whole point of the frame chunk. The
      phrase 'oldest -> newest' is the canonical cue used by video-VLM
      benchmarks.
    """

    block = build_slow_path_inputs(
        surface_kind="mock", caps=_caps(), objective="x",
        state="", live_log_tail=[], n_frames=3,
    )
    assert "FRAMES (oldest -> newest" in block
    assert "<3 attached>" in block


def test_inputs_block_overlay_is_sorted_json() -> None:
    """@example: overlay JSON is emitted with sort_keys for stable KV reuse.

    ROOT CAUSE:
      llama-server's slot cache prefix-matches at the token level. If
      we emit OVERLAY with non-deterministic key order, every turn
      invalidates the prefix from OVERLAY: forward and we pay full
      prefill each time. sort_keys=True keeps the cache hot.
    """

    block = build_slow_path_inputs(
        surface_kind="gba", caps=_caps(), objective="x",
        state="", live_log_tail=[], n_frames=0,
        overlay={"z_last": 1, "a_first": 2, "m_middle": 3},
    )
    # Find the substring after "OVERLAY: " up to the next newline.
    start = block.find("OVERLAY:") + len("OVERLAY: ")
    end = block.find("\n", start)
    overlay_json = block[start:end]
    # Sorted keys → "a_first" appears before "m_middle" appears before "z_last".
    assert overlay_json.index("a_first") < overlay_json.index("m_middle")
    assert overlay_json.index("m_middle") < overlay_json.index("z_last")
