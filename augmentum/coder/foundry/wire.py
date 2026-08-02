"""Bind the real implementations behind the foundry loop's injected stages.

``wire_default_stages(request, …)`` returns the concrete generate / asset /
verify / play callables ``run_foundry`` consumes. Kept separate from
``loop.py`` so the loop's control flow stays unit-testable with fakes while
this module carries the live-stack wiring (coder dispatch, Blender, vision,
game-agent session + headless-browser play).
"""
from __future__ import annotations

import asyncio
import base64
import contextlib
import uuid
from pathlib import Path
from typing import Any

from augmentum.coder import browser_sidecar as _sidecar
from augmentum.coder.executors import ContainerExecutor
from augmentum.coder.foundry.blender_asset import make_asset_stage
from augmentum.coder.foundry.contract import GameBuildSpec, semantic_inputs_from
from augmentum.coder.foundry.game_gen import make_generate_stage
from augmentum.config import settings
from augmentum.game_agent.playtest import build_playtest_objective
from augmentum.utils.logging import get_logger

log = get_logger(__name__)

_TEXT_EXT = (".html", ".htm", ".js", ".mjs", ".css", ".json", ".txt", ".svg")
_BIN_EXT = (".glb", ".gltf", ".bin", ".png", ".jpg", ".jpeg", ".gif", ".webp")


# ── Play-host base URL (must be reachable from the headless browser) ───

def _play_host_base() -> str:
    """augmentum base URL as the browser sidecar can reach it.

    Derived from the same operator setting the in-container bridge dialler
    uses (``agent_bridge_base_url``, a ``ws(s)://`` URL), converted to http.
    Falls back to the compose-network default and logs when unset so an
    unconfigured deployment is diagnosable rather than silently broken.
    """
    base = (settings.agent_bridge_base_url or "").strip().rstrip("/")
    if base.startswith("ws://"):
        return "http://" + base[len("ws://"):]
    if base.startswith("wss://"):
        return "https://" + base[len("wss://"):]
    if base:
        return base
    log.warning("foundry_play_host_base_unset",
                hint="set agent_bridge_base_url so the browser can reach augmentum")
    # Compose-network default: the app listens on 6100 internally (EXPOSE 6100
    # in Dockerfile.gpu), reachable by the browser sidecar as host 'augmentum'.
    return "http://augmentum:6100"


# ── Vision captioner (auto = VisionRouter background; pinned = that model) ─

def make_captioner(app_state: Any, *, verify_model: str = ""):
    """Return an async ``(image_bytes, prompt) -> caption`` callable.

    Auto (``verify_model`` empty) routes through the VisionRouter with a
    ``background`` workload hint. A pinned id resolves via the provider
    registry and captions that model directly (the user's choice wins).
    Any failure degrades to '' — the play-based score still gates the loop.
    """
    async def captioner(image_bytes: bytes, prompt: str) -> str:
        # Pinned model — resolve + caption directly.
        if verify_model:
            try:
                registry = getattr(app_state, "provider_registry", None)
                http = getattr(app_state, "http_client", None)
                if registry is not None and http is not None:
                    backend, model = await registry.resolve_model_for_role(
                        "heavyweight", override=verify_model, settings=settings,
                    )
                    base_url = getattr(backend, "base_url", "") or ""
                    if base_url:
                        from augmentum.vision.provider import (
                            _caption_via_openai_endpoint,
                        )
                        return await _caption_via_openai_endpoint(
                            http, base_url, image_bytes, prompt=prompt,
                            max_tokens=120, timeout_s=30.0, model=model,
                        )
            except Exception as exc:
                log.warning("foundry_pinned_verify_failed", model=verify_model,
                            error=str(exc))
                # fall through to router
        # Auto — the VisionRouter (current routing), background workload.
        router = getattr(app_state, "vision_router", None)
        if router is None:
            return ""
        try:
            from augmentum.vision.router import Workload
            return await router.caption(
                image_bytes, prompt=prompt, max_tokens=120,
                workload=Workload.BACKGROUND,
            )
        except Exception as exc:
            log.warning("foundry_auto_verify_failed", error=str(exc))
            return ""

    return captioner


# ── Deploy bundle read (text + binary-as-base64 for the browser) ──────

async def _read_deploy_bundle(ex: ContainerExecutor, gen_dir: str):
    """Read the generated game dir into (entry_html, {relpath: {c, e}}).

    Text files pass through; binary assets (the Blender GLB, images) are
    base64-encoded so ``composeBundle``'s fetch shim can serve them to the
    in-iframe game. Build scripts and the verify render are skipped.
    """
    listing = await ex.run_command(
        ["bash", "-c", f"find {gen_dir} -type f 2>/dev/null | head -300"],
        timeout=15.0,
    )
    html = ""
    files: dict[str, dict] = {}
    for path in listing.splitlines():
        path = path.strip()
        if not path:
            continue
        rel = path[len(gen_dir):].lstrip("/")
        if not rel or rel.startswith("__") or rel.endswith(".render.png"):
            continue
        low = rel.lower()
        try:
            if low.endswith(_TEXT_EXT):
                content = await ex.read_file(path)
                if rel == "index.html":
                    html = content
                files[rel] = {"c": content, "e": "text"}
            elif low.endswith(_BIN_EXT):
                raw = await ex.read_file_bytes(path)
                files[rel] = {"c": base64.b64encode(raw).decode(), "e": "base64"}
        except Exception:
            continue
    return html, files


# ── Play stage (create session → headless play → read score) ──────────

def make_play_stage(request: Any, *, workspace_id: str, user_id: str,
                    on_event: Any = None):
    """Return an async ``play(slug, files, spec, play_seconds) -> progress``.

    Creates a bridged game-agent session in-process, hands the composed game
    to the play-host route, points the browser sidecar at it (the injected
    shim dials the same-origin bridge WS, starting the Orchestrator), bounds
    play to ``play_seconds``, and returns the captured scorecard.

    When ``on_event`` is supplied (the theater feed), the session id + live
    game preview URL are announced and the agent's decisions are streamed as
    ``observation`` events by tailing the session log — this is the "watch the
    agent play" feed.
    """
    # Imported here to avoid a route→foundry import cycle at module load.
    from augmentum.proxy.game_agent_routes import (
        _create_bridged_session,
        _foundry_payloads,
        _log_dir,
        _sessions,
    )

    def _emit(t: str, **d: Any) -> None:
        if on_event is not None:
            with contextlib.suppress(Exception):
                on_event(t, **d)

    app_state = request.app.state
    cm = getattr(app_state, "container_manager", None)
    base = _play_host_base()

    async def play(slug: str, files: dict, spec: GameBuildSpec,
                   play_seconds: int) -> dict | None:
        if cm is None:
            return None
        session_id = f"s_{uuid.uuid4().hex[:10]}"
        log_path = Path(_log_dir(request)) / f"{session_id}.ndjson"
        sem = semantic_inputs_from(files) or list(spec.controls.keys())

        rec = _create_bridged_session(
            request, session_id=session_id, log_path=log_path,
            owner_user_id=user_id, surface="js13k",
            objective=build_playtest_objective(spec.objective),
            semantic_inputs=sem, log_schema="js13k.v1", companion=False,
            persona=None, journal=None, controller_profile=None, game_profile=None,
        )
        # _create_bridged_session returns a JSONResponse on bad input.
        if not hasattr(rec, "bridge_token"):
            log.warning("foundry_session_create_failed", slug=slug)
            return None

        gen_dir = f"/workspace/generated/{spec.slug}"
        ex = ContainerExecutor(cm, workspace_id)
        html, deploy_files = await _read_deploy_bundle(ex, gen_dir)
        if not html:
            log.warning("foundry_no_entry_html", slug=slug)
            _sessions(request).pop(session_id, None)
            return None

        _foundry_payloads(request)[session_id] = {
            "html": html, "entry": "index.html", "files": deploy_files,
            "agentBridge": {
                "wsUrl": f"/api/game-agent/surfaces/js13k/bridge/{session_id}",
                "sessionId": session_id, "token": rec.bridge_token,
                "semanticToKey": dict(spec.controls),
            },
        }
        play_url = (f"/api/game-agent/foundry/play-host/{session_id}"
                    f"?token={rec.bridge_token}")
        # The theater embeds this to show the game as the agent plays it live.
        _emit("play_session", session_id=session_id, play_url=play_url)
        url = f"{base}{play_url}"

        # Tail the session log → observation events (the live decision stream).
        tail_task = asyncio.create_task(_stream_observations(rec.log_path, _emit))
        try:
            await _sidecar.action(cm, workspace_id, url=url, action="open")
            # Wait for the shim to connect (status flips to running), then
            # bound play, then stop and let the orchestrator finalize.
            await _wait_status(rec, ("running", "stopped", "error"), timeout_s=30)
            await asyncio.sleep(max(5, play_seconds))
            if getattr(rec, "orchestrator", None) is not None and rec.status == "running":
                rec.orchestrator.stop("timeout")
            await _wait_result(rec, timeout_s=15)
        finally:
            tail_task.cancel()
            _foundry_payloads(request).pop(session_id, None)

        progress = getattr(rec, "result_progress", None)
        _sessions(request).pop(session_id, None)
        return progress

    return play


async def _stream_observations(log_path: Any, emit) -> None:
    """Tail a game-agent session log, emitting the agent's decisions.

    Renders the human-meaningful entries — screen/state observations, the
    slow-path scratchpad, and button presses — as ``observation`` events so
    the theater shows the agent thinking and acting in real time. Best-effort:
    cancelled when play ends; any error just stops the stream.
    """
    from augmentum.game_agent.log import tail_log
    try:
        async for entry in tail_log(log_path, from_start=True):
            kind = entry.get("kind")
            text = ""
            if kind == "input":
                p = entry.get("payload", {})
                text = f"→ {p.get('semantic', '')}"
            elif kind == "plan":
                p = entry.get("payload", {})
                text = (p.get("scratchpad") or p.get("intent") or "").strip()[:200]
            elif kind == "event":
                p = entry.get("payload", {})
                data = p.get("data", {}) if isinstance(p, dict) else {}
                if isinstance(data, dict) and data.get("screen"):
                    text = f"● {data.get('screen')}"
            if text:
                emit("observation", text=text)
            if kind == "session_end":
                break
    except asyncio.CancelledError:
        raise
    except Exception:
        return


async def _wait_status(rec: Any, wanted: tuple[str, ...], *, timeout_s: float) -> None:
    waited = 0.0
    while waited < timeout_s:
        if getattr(rec, "status", "") in wanted:
            return
        await asyncio.sleep(1.0)
        waited += 1.0


async def _wait_result(rec: Any, *, timeout_s: float) -> None:
    waited = 0.0
    while waited < timeout_s:
        if getattr(rec, "result_progress", None) is not None:
            return
        if getattr(rec, "status", "") in ("stopped", "error"):
            # give finalize a beat to store the scorecard
            await asyncio.sleep(1.0)
            return
        await asyncio.sleep(1.0)
        waited += 1.0


# ── The default binding ───────────────────────────────────────────────

def wire_default_stages(
    request: Any,
    *,
    user_id: str,
    workspace_id: str,
    model: str,
    verify_model: str = "",
    on_event: Any = None,
) -> dict:
    """Return ``{generate, asset, verify, play}`` bound to the live stack.

    Pass the result to ``run_foundry(spec, **wire_default_stages(...))``.
    ``on_event`` (the theater bus emit) is threaded into the play stage so the
    agent's live decisions stream to the viewer.
    """
    app_state = request.app.state
    captioner = make_captioner(app_state, verify_model=verify_model)

    async def verify(image_bytes: bytes, objective: str) -> list:
        from augmentum.coder.foundry.visual_verify import verify_image
        return await verify_image(image_bytes, captioner=captioner,
                                  objective=objective, kind="render")

    return {
        "generate": make_generate_stage(
            app_state, user_id=user_id, workspace_id=workspace_id, model=model),
        "asset": make_asset_stage(app_state, workspace_id=workspace_id),
        "verify": verify,
        "play": make_play_stage(request, workspace_id=workspace_id,
                                user_id=user_id, on_event=on_event),
    }
