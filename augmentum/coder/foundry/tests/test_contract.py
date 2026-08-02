"""Unit tests for the game-generation output contract + validator."""
from __future__ import annotations

from augmentum.coder.foundry.contract import (
    GameBuildSpec,
    contract_prompt,
    semantic_inputs_from,
    validate_generated_game,
)

_VALID_HTML = """
<!doctype html><html><body>
<canvas id="c" width="320" height="240"></canvas>
<script>
window.AUGMENTUM_GAME = {
  semantic_to_key: { "left": "ArrowLeft", "right": "ArrowRight", "action": "Space" },
  objective: "collect 3 coins"
};
const post = (m) => (window.parent||window).postMessage(m, "*");
post({type:'screen', label:'play'});
function tick(){ post({type:'progress', value: 0.5}); }
function win(){ post({type:'won'}); }
document.addEventListener('keydown', (e)=>{ /* ... */ });
</script>
</body></html>
"""


def test_valid_game_has_no_violations():
    assert validate_generated_game({"index.html": _VALID_HTML}) == []


def test_missing_index_flagged():
    v = validate_generated_game({"game.html": _VALID_HTML})
    assert any("entry file" in x for x in v)


def test_missing_augmentum_game_flagged():
    html = _VALID_HTML.replace("window.AUGMENTUM_GAME", "window.SOMETHING_ELSE")
    v = validate_generated_game({"index.html": html})
    assert any("AUGMENTUM_GAME" in x for x in v)


def test_missing_progress_hook_flagged():
    html = _VALID_HTML.replace("type:'progress'", "type:'other'")
    v = validate_generated_game({"index.html": html})
    assert any("progress" in x for x in v)


def test_missing_won_hook_flagged():
    html = _VALID_HTML.replace("type:'won'", "type:'done'")
    v = validate_generated_game({"index.html": html})
    assert any("won" in x for x in v)


def test_missing_canvas_flagged():
    html = _VALID_HTML.replace("<canvas", "<div")
    v = validate_generated_game({"index.html": html})
    assert any("canvas" in x for x in v)


def test_no_postmessage_at_all():
    html = "<canvas></canvas><script>window.AUGMENTUM_GAME={semantic_to_key:{a:'KeyA'}};</script>"
    v = validate_generated_game({"index.html": html})
    assert any("postMessage" in x for x in v)


def test_hooks_can_live_in_linked_js_file():
    # HTML has the canvas + AUGMENTUM_GAME; the postMessage hooks live in game.js.
    html = '<canvas></canvas><script src="game.js"></script>'
    js = (
        "window.AUGMENTUM_GAME={semantic_to_key:{'action':'Space'}};"
        "const p=m=>window.parent.postMessage(m,'*');"
        "p({type:'screen',label:'play'});p({type:'progress',value:1});p({type:'won'});"
    )
    assert validate_generated_game({"index.html": html, "game.js": js}) == []


def test_semantic_inputs_extraction():
    got = semantic_inputs_from({"index.html": _VALID_HTML})
    assert got == ["left", "right", "action"]


def test_semantic_inputs_empty_on_miss():
    assert semantic_inputs_from({"index.html": "<html></html>"}) == []


def test_contract_prompt_mentions_required_hooks_2d():
    spec = GameBuildSpec(slug="x", title="X", concept="c", objective="win", dimension="2d")
    p = contract_prompt(spec)
    assert "AUGMENTUM_GAME" in p and "semantic_to_key" in p
    assert "type:'progress'" in p and "type:'won'" in p
    assert "<canvas>" in p
    assert "2D game" in p


def test_contract_prompt_3d_mentions_gltf():
    spec = GameBuildSpec(slug="x", title="X", concept="c", objective="win",
                         dimension="3d", glb_asset="assets/crate.glb")
    p = contract_prompt(spec)
    assert "three.js" in p and "assets/crate.glb" in p
