"""The game-generation output contract + validator.

For the game foundry to close its loop, a generated game must be *agent-
playable*: the browser agent-bridge shim has to know the control vocabulary,
and :mod:`augmentum.game_agent.progress` needs the game to announce its own
state so play can be scored. Free-form generated HTML cannot be trusted to do
either, so we impose a small, explicit contract and **validate the emitted
files against it** before ever handing the build to the player (per the
"designed ≠ applied" discipline — a generated file is not trusted until the
required hooks are verified present).

The contract (both dimensions, "speak both"):

1. Entry ``index.html`` exists.
2. A JS global ``AUGMENTUM_GAME = { semantic_to_key: {...}, objective: "..." }``
   declares the control map (semantic action -> KeyboardEvent.code) and the
   goal. The bridge binds inputs from ``semantic_to_key``; the session uses
   its keys as the ``semantic_inputs`` vocabulary.
3. The game emits state via ``window.postMessage`` /
   ``window.parent.postMessage``:
     * ``{type:'progress', value: 0..1}``  as the player advances,
     * ``{type:'won'}``                     on win,
     * ``{type:'screen', label:'...'}``     on screen changes (title/play/…).
   These feed the productive-screen + goal signals in progress.py.
4. A ``<canvas>`` element exists (2D or WebGL) — the shim samples it to PNG.

``dimension`` is carried on the build spec: ``'3d'`` ⇒ three.js + a Blender-
made GLB asset; ``'2d'`` ⇒ a canvas game with no GLB. The contract is
identical for both; only the generation prompt differs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

Dimension = Literal["2d", "3d"]


@dataclass
class GameBuildSpec:
    """One game to generate. Fed to the generation driver and coder loop."""

    slug: str                       # dir-safe id, e.g. "coin-dash"
    title: str                      # human name
    concept: str                    # what the game is (prompt seed)
    objective: str                  # the player's goal (drives playtest)
    dimension: Dimension = "2d"
    # Suggested control map. The generated game is free to refine it, but the
    # contract requires SOME AUGMENTUM_GAME.semantic_to_key to exist.
    controls: dict[str, str] = field(default_factory=lambda: {
        "left": "ArrowLeft", "right": "ArrowRight",
        "up": "ArrowUp", "down": "ArrowDown", "action": "Space",
    })
    # Feedback from the previous playtest pass (relay_brief output). Empty on
    # the first pass; the driver injects it into the regeneration prompt.
    relay: str = ""
    # For the 3d path: path to a Blender-exported GLB the game should load.
    glb_asset: str = ""


# Files the contract cares about, and the markers each must contain.
INDEX_HTML = "index.html"


def contract_prompt(spec: GameBuildSpec) -> str:
    """Render the output-contract instructions injected into the coder task.

    Kept separate from the concept prompt so the contract text is identical
    across every generation and easy to keep in sync with the validator.
    """
    controls_json = ", ".join(f'"{k}": "{v}"' for k, v in spec.controls.items())
    dim_note = (
        "This is a 3D game: use three.js (loaded from a CDN or bundled) and "
        f"load the GLB asset at '{spec.glb_asset}' via GLTFLoader. Render into "
        "a <canvas>."
        if spec.dimension == "3d" else
        "This is a 2D game: render into a <canvas> with the 2D context."
    )
    return (
        "OUTPUT CONTRACT — the game MUST satisfy all of these or it cannot be "
        "playtested:\n"
        f"1. Provide an entry file named '{INDEX_HTML}'.\n"
        "2. Declare a JS global exactly like:\n"
        f"     window.AUGMENTUM_GAME = {{ semantic_to_key: {{ {controls_json} }}, "
        f'objective: "{spec.objective}" }};\n'
        "   semantic_to_key maps semantic action names to KeyboardEvent.code "
        "values. The game must listen for those key codes on document/window.\n"
        "3. Announce state to the host via postMessage (window.parent.postMessage "
        "when framed, else window.postMessage):\n"
        "     - postMessage({type:'screen', label:'play'}) once interactive "
        "(and 'title'/'loading' before);\n"
        "     - postMessage({type:'progress', value: <0..1>}) as the player "
        "advances;\n"
        "     - postMessage({type:'won'}) when the objective is met.\n"
        "4. Include a <canvas> element (the host samples it for frames).\n"
        f"{dim_note}\n"
        "Keep it a single self-contained game. Do not add a start menu that "
        "requires a mouse click to begin — boot straight into play so an "
        "automated player can reach it."
    )


# ── Validator ─────────────────────────────────────────────────────────

# Each check: (id, human message, predicate over the joined source text).
# Kept as regexes so minor formatting variance (spacing, quote style) passes
# while a genuinely-missing hook fails.
_AUGMENTUM_GAME_RE = re.compile(r"AUGMENTUM_GAME\s*=", re.I)
_SEMANTIC_TO_KEY_RE = re.compile(r"semantic_to_key\s*:", re.I)
_CANVAS_RE = re.compile(r"<canvas\b", re.I)
_POST_SCREEN_RE = re.compile(r"type\s*:\s*['\"]screen['\"]", re.I)
_POST_PROGRESS_RE = re.compile(r"type\s*:\s*['\"]progress['\"]", re.I)
_POST_WON_RE = re.compile(r"type\s*:\s*['\"]won['\"]", re.I)
_POSTMESSAGE_RE = re.compile(r"postMessage\s*\(", re.I)


def validate_generated_game(files: dict[str, str]) -> list[str]:
    """Return a list of contract violations. Empty list == valid.

    ``files`` maps relative path -> text content of the generated bundle.
    Non-text/binary assets (GLB, PNG) may be omitted; the validator only
    inspects text. All text is joined for the marker checks because the
    hooks may live in a linked ``.js`` file rather than inline in the HTML.
    """
    violations: list[str] = []

    # 1. Entry file present (case-insensitive match on basename).
    has_index = any(
        p.rsplit("/", 1)[-1].lower() == INDEX_HTML for p in files
    )
    if not has_index:
        violations.append(f"missing entry file '{INDEX_HTML}'")

    joined = "\n".join(files.values())

    # 2. AUGMENTUM_GAME + semantic_to_key.
    if not _AUGMENTUM_GAME_RE.search(joined):
        violations.append("missing 'window.AUGMENTUM_GAME = ...' control declaration")
    if not _SEMANTIC_TO_KEY_RE.search(joined):
        violations.append("AUGMENTUM_GAME is missing a 'semantic_to_key' map")

    # 3. postMessage state hooks. Require the call plus each state type so a
    #    game that only announces one of the three doesn't silently pass.
    if not _POSTMESSAGE_RE.search(joined):
        violations.append("game never calls postMessage — the host cannot score it")
    else:
        if not _POST_SCREEN_RE.search(joined):
            violations.append("missing postMessage({type:'screen', ...}) hook")
        if not _POST_PROGRESS_RE.search(joined):
            violations.append("missing postMessage({type:'progress', ...}) hook")
        if not _POST_WON_RE.search(joined):
            violations.append("missing postMessage({type:'won'}) hook")

    # 4. Canvas present.
    if not _CANVAS_RE.search(joined):
        violations.append("no <canvas> element (host samples the canvas for frames)")

    return violations


def semantic_inputs_from(files: dict[str, str]) -> list[str]:
    """Best-effort extraction of the semantic action names for the session.

    Parses the ``semantic_to_key`` object literal to recover the semantic
    keys the game declared, so the play session's ``semantic_inputs``
    vocabulary matches what the game actually binds. Falls back to empty on
    any parse miss (the caller then supplies the build-spec controls).
    """
    joined = "\n".join(files.values())
    m = re.search(r"semantic_to_key\s*:\s*\{([^}]*)\}", joined, re.I | re.S)
    if not m:
        return []
    body = m.group(1)
    return re.findall(r"['\"]([A-Za-z0-9_]+)['\"]\s*:", body)
