"""Generation stage — dispatch the coder loop to build a contract-bound game.

Binds the real coder machinery (``InternalCoderDriver`` → ``coder_background_run``
job) into a :data:`~augmentum.coder.foundry.loop.GenerateStage` callable. It:

1. Renders a task = concept + the output contract + any prior playtest relay.
2. Dispatches it into a workspace and waits for the run to finish.
3. Reads the emitted bundle back from ``/workspace/generated/<slug>/``.
4. **Validates** it against the contract before returning — a build that
   lacks the control declaration or state postMessages is reported with its
   violations and never handed to the player pretending to be playable
   (the "designed ≠ applied" discipline).

Requires the running stack (job runner + a Docker workspace). The loop's
orchestration is tested separately with fakes; this module is the live seam.
"""
from __future__ import annotations

import asyncio
from typing import Any

from augmentum.coder.coding_driver import InternalCoderDriver, get_run
from augmentum.coder.executors import ContainerExecutor
from augmentum.coder.foundry.contract import (
    GameBuildSpec,
    contract_prompt,
    semantic_inputs_from,
    validate_generated_game,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

# Coder run statuses that mean "no longer running".
_TERMINAL = {"completed", "succeeded", "done", "failed", "error", "cancelled"}
# Text extensions we read back for contract validation (binary assets like
# GLB/PNG are excluded — the validator only inspects source).
_TEXT_EXT = (".html", ".htm", ".js", ".mjs", ".css", ".json", ".txt", ".svg")


def make_generate_stage(
    app_state: Any,
    *,
    user_id: str,
    workspace_id: str,
    model: str,
    poll_interval_s: float = 5.0,
    max_wait_s: float = 1800.0,
):
    """Return an async ``generate(spec) -> dict`` bound to a workspace.

    The returned dict matches the loop's GenerateStage contract:
    ``{"slug", "files", "violations", "run_id"}``.
    """
    driver = InternalCoderDriver(app_state)
    cm = getattr(app_state, "container_manager", None)

    async def generate(spec: GameBuildSpec) -> dict:
        out_dir = f"/workspace/generated/{spec.slug}"
        task = _build_task(spec, out_dir)

        dispatch = await driver.dispatch(
            user_id=user_id, workspace_id=workspace_id, task=task,
            model=model, origin_surface="foundry",
        )
        if not dispatch.get("ok"):
            return {"slug": spec.slug, "files": {}, "run_id": "",
                    "violations": [f"dispatch failed: {dispatch.get('error')}"]}
        run_id = dispatch["run_id"]

        status = await _await_run(app_state, user_id=user_id, run_id=run_id,
                                  poll_interval_s=poll_interval_s, max_wait_s=max_wait_s)
        if status not in ("completed", "succeeded", "done"):
            return {"slug": spec.slug, "files": {}, "run_id": run_id,
                    "violations": [f"coder run ended '{status}' without a usable build"]}

        files = await _read_bundle(cm, workspace_id, out_dir)
        violations = validate_generated_game(files)
        # Recover the game's declared control vocabulary for the play session.
        sem = semantic_inputs_from(files)
        return {"slug": spec.slug, "files": files, "run_id": run_id,
                "violations": violations, "semantic_inputs": sem}

    return generate


def _build_task(spec: GameBuildSpec, out_dir: str) -> str:
    """Compose the coder task prompt: concept + contract + prior relay."""
    parts = [
        f"Build a small, self-contained browser game: {spec.title}.",
        spec.concept.strip(),
        f"The player's objective: {spec.objective.strip()}",
        f"Write ALL game files into the directory {out_dir}/ "
        f"(entry file {out_dir}/index.html).",
        "",
        contract_prompt(spec),
    ]
    if spec.relay.strip():
        parts += ["", "PREVIOUS PLAYTEST FEEDBACK — address this in your build:",
                  spec.relay.strip()]
    return "\n".join(parts)


async def _await_run(
    app_state: Any, *, user_id: str, run_id: str,
    poll_interval_s: float, max_wait_s: float,
) -> str:
    """Poll ``get_run`` until the status is terminal or the budget expires.

    Returns the final status string ('timeout' if the wait budget expired
    before the run reached a terminal state).
    """
    waited = 0.0
    while waited < max_wait_s:
        run = await get_run(app_state, user_id=user_id, run_id=run_id)
        status = (run or {}).get("status", "") if run else ""
        if status in _TERMINAL:
            return status
        await asyncio.sleep(poll_interval_s)
        waited += poll_interval_s
    return "timeout"


async def _read_bundle(cm, workspace_id: str, out_dir: str) -> dict[str, str]:
    """Read the generated text files under ``out_dir`` into {relpath: content}.

    Uses the container executor's shell + read_file. Binary assets (GLB/PNG)
    are intentionally skipped — validation is over source only, and the play
    host composes/serves the bundle directly. Returns {} on any failure so
    the caller sees "no files" (→ contract violations) rather than a crash.
    """
    if cm is None:
        return {}
    ex = ContainerExecutor(cm, workspace_id)
    try:
        listing = await ex.run_command(
            ["bash", "-c", f"find {out_dir} -type f 2>/dev/null | head -200"],
            timeout=15.0,
        )
    except Exception as exc:
        log.warning("foundry_read_bundle_list_failed", error=str(exc))
        return {}

    files: dict[str, str] = {}
    for path in listing.splitlines():
        path = path.strip()
        if not path or not path.lower().endswith(_TEXT_EXT):
            continue
        try:
            content = await ex.read_file(path)
        except Exception:
            continue
        rel = path[len(out_dir):].lstrip("/")
        files[rel] = content
    return files
