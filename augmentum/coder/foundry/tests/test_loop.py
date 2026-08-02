"""Unit tests for the foundry loop orchestration (fake stages).

These verify the control flow — pass iteration, relay threading, score-delta,
contract-violation short-circuit, per-pass error isolation — without any stack.
"""
from __future__ import annotations

import pytest

from augmentum.coder.foundry.contract import GameBuildSpec
from augmentum.coder.foundry.loop import run_foundry

_VALID_FILES = {"index.html": "<canvas></canvas>"}


def _spec(dimension="2d"):
    return GameBuildSpec(slug="g", title="G", concept="c",
                         objective="win", dimension=dimension)


def _progress(score, *, reached_play=True, spm=None, duration_ms=90000):
    d = {
        "score": score, "reached_play": reached_play, "inputs_acked": 20,
        "inputs_effective": 15, "effective_input_ratio": 0.75,
        "goals_completed": 1, "duration_ms": duration_ms,
    }
    d["score_per_min"] = spm if spm is not None else score / (duration_ms / 60000.0)
    return d


@pytest.mark.asyncio
async def test_two_passes_run_and_thread_relay():
    seen_relays = []

    async def generate(spec):
        seen_relays.append(spec.relay)  # what this pass saw as feedback
        return {"slug": "g", "files": _VALID_FILES, "violations": [], "run_id": "r"}

    scores = iter([2.0, 5.0])

    async def play(slug, files, spec, secs):
        return _progress(next(scores))

    res = await run_foundry(_spec(), generate=generate, play=play, passes=2)
    assert len(res.passes) == 2
    # Pass 1 saw no relay; pass 2 saw the relay produced by pass 1.
    assert seen_relays[0] == ""
    assert seen_relays[1] != ""
    assert "Playtest feedback" in seen_relays[1]


@pytest.mark.asyncio
async def test_improved_true_when_score_climbs():
    async def generate(spec):
        return {"slug": "g", "files": _VALID_FILES, "violations": []}
    scores = iter([1.0, 4.0])
    async def play(slug, files, spec, secs):
        return _progress(next(scores))
    res = await run_foundry(_spec(), generate=generate, play=play, passes=2)
    assert res.improved is True


@pytest.mark.asyncio
async def test_improved_false_when_score_flat_or_drops():
    async def generate(spec):
        return {"slug": "g", "files": _VALID_FILES, "violations": []}
    scores = iter([5.0, 3.0])
    async def play(slug, files, spec, secs):
        return _progress(next(scores))
    res = await run_foundry(_spec(), generate=generate, play=play, passes=2)
    assert res.improved is False


@pytest.mark.asyncio
async def test_contract_violation_skips_play_and_feeds_back():
    played = []
    async def generate(spec):
        return {"slug": "g", "files": {}, "violations": ["missing <canvas> element"]}
    async def play(slug, files, spec, secs):
        played.append(slug)
        return _progress(1.0)
    res = await run_foundry(_spec(), generate=generate, play=play, passes=1)
    assert played == []  # play never ran for a non-playable build
    assert res.passes[0].progress is None
    assert "contract not satisfied" in res.passes[0].relay.lower()


@pytest.mark.asyncio
async def test_pass_error_is_isolated_not_fatal():
    async def generate(spec):
        raise RuntimeError("coder loop died")
    async def play(slug, files, spec, secs):
        return _progress(1.0)
    res = await run_foundry(_spec(), generate=generate, play=play, passes=2)
    assert len(res.passes) == 2
    assert all(p.error for p in res.passes)
    assert res.improved is False


@pytest.mark.asyncio
async def test_3d_runs_asset_and_verify_stages():
    calls = {"asset": 0, "verify": 0}
    async def asset(spec):
        calls["asset"] += 1
        return {"glb_asset": "assets/crate.glb", "render_png_bytes": b"PNG"}
    async def verify(img, objective):
        calls["verify"] += 1
        return ["crate looks flat"]
    async def generate(spec):
        # The generated game should have received the GLB path from the asset stage.
        assert spec.glb_asset == "assets/crate.glb"
        return {"slug": "g", "files": _VALID_FILES, "violations": []}
    async def play(slug, files, spec, secs):
        return _progress(3.0)
    res = await run_foundry(_spec("3d"), generate=generate, play=play,
                            asset=asset, verify=verify, passes=1)
    assert calls == {"asset": 1, "verify": 1}
    # Vision note flowed into the pass and its relay.
    assert res.passes[0].vision_notes == ["crate looks flat"]
    assert "crate looks flat" in res.passes[0].relay


@pytest.mark.asyncio
async def test_summary_renders():
    async def generate(spec):
        return {"slug": "g", "files": _VALID_FILES, "violations": []}
    async def play(slug, files, spec, secs):
        return _progress(2.0)
    res = await run_foundry(_spec(), generate=generate, play=play, passes=1)
    s = res.summary()
    assert "pass 1" in s and "improved:" in s
