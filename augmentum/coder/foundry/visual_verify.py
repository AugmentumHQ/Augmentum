"""Visual verification of Blender renders + game frames for the foundry loop.

Closes the "does it look right" half of the loop, complementing the
play-based score. Reuses Augmentum's existing vision routing rather than
introducing a new model path:

* **Auto** (``coder_visual_verify_model`` empty) → the ``VisionRouter`` with a
  ``background`` workload hint. That hint is deliberate: the router keeps the
  primary chat model's KV cache clean while the user may be mid-conversation —
  exactly the coder-pipeline case. This is the "wire in current behavior" the
  design calls for.
* **Pinned** (an explicit model id) → the user's choice WINS (per the
  never-auto-select rule). The loop supplies a ``captioner`` bound to that
  model via ``resolve_model_for_role(override=...)``.

This module stays framework-free and unit-testable by taking the captioning
call as an injected async callable — the loop wires the real ``VisionRouter`` /
resolved-backend behind it.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable

# An async callable: (image_bytes, prompt) -> caption text ("" on failure).
Captioner = Callable[[bytes, str], Awaitable[str]]

# Phrases a captioner returns when nothing is wrong. Matched case-insensitively
# as a whole-note check so a clean render produces zero defects rather than a
# noisy "looks fine" note fed back to the generator.
_CLEAN_MARKERS = ("looks fine", "no obvious problem", "no issues", "nothing wrong")


def _inspect_prompt(objective: str, *, kind: str) -> str:
    """Build the grounded inspection prompt.

    ``kind`` is "render" (a Blender still of an asset/scene) or "frame" (a
    live game frame). The prompt asks for problems, not praise, and for a
    terse verdict so the note stays actionable.
    """
    subject = (
        "a rendered still of a 3D game asset/scene"
        if kind == "render" else "a frame captured from a running game"
    )
    goal = (objective or "").strip()
    goal_line = f" The intended result: {goal}." if goal else ""
    return (
        f"You are inspecting {subject}.{goal_line} In ONE short sentence, name "
        "any obvious visual problem — untextured/flat surfaces, wrong scale, "
        "clipping/intersecting geometry, an empty or black image, or something "
        "unreadable. If it looks fine, reply exactly 'looks fine'."
    )


async def verify_image(
    image_bytes: bytes,
    *,
    captioner: Captioner,
    objective: str = "",
    kind: str = "render",
) -> list[str]:
    """Return visual-defect notes for one image. Empty list == looks fine.

    A blank/failed caption yields no note (we don't invent problems when the
    verifier is unavailable — the play-based score still gates the loop).
    """
    if not image_bytes:
        return []
    prompt = _inspect_prompt(objective, kind=kind)
    try:
        note = await captioner(image_bytes, prompt)
    except Exception:
        return []
    note = (note or "").strip()
    if not note:
        return []
    low = note.lower()
    if any(m in low for m in _CLEAN_MARKERS):
        return []
    return [note]
