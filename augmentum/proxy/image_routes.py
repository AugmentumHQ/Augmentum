"""FastAPI router for image generation endpoints."""

from __future__ import annotations

import asyncio
import json
import os

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from augmentum.auth.guards import require_admin
from augmentum.config import settings
from augmentum.proxy import system_events
from augmentum.utils.secrets import decrypt_api_key, sanitize_error_detail
from augmentum.image.schemas import (
    AspectRatio,
    BatchDeleteRequest,
    CatalogModelInfo,
    GenerateRequest,
    GenerateResponse,
    HardwareInfo,
    HistoryPage,
    Img2ImgRequest,
    InpaintRequest,
    JobStatus,
    JobType,
    ModelInfo,
    ModelPullRequest,
)
from augmentum.utils.logging import get_logger

log = get_logger(__name__)


def _pull_task_done_callback(task: asyncio.Task) -> None:
    """Log exceptions from fire-and-forget model pull tasks."""
    if not task.cancelled() and task.exception():
        log.error("pull_task_background_failed", error=str(task.exception()))


def _monitor_task_done_callback(task: asyncio.Task) -> None:
    """Log exceptions from fire-and-forget disk monitor tasks."""
    if not task.cancelled() and task.exception():
        log.error("disk_monitor_task_failed", error=str(task.exception()))


router = APIRouter(prefix="/api/image", tags=["image"])


def _require_image(request: Request):
    """Raise 503 if no image-generation path can serve this request.

    Called after the cloud-provider and fabric-peer tiers have already had
    a chance to claim the request. Reaching here means all three tiers
    failed — the generic "not enabled" string was misleading because the
    setting may be on (or not the actual gate). Surface what was checked
    so the frontend / devtools shows a useful diagnostic instead of a
    one-liner that points the user at the wrong knob.
    """
    if getattr(request.app.state, "image_queue", None):
        return

    # CPU-only image generation isn't a supported path. Don't mention the
    # AUGMENTUM_IMAGE_ENABLED flag here — flipping it on a CPU-only box
    # produces a queue that then renders unusably slowly, sending the user
    # down the wrong rabbit hole.
    reasons: list[str] = ["local pipeline unavailable"]

    try:
        coordinator = getattr(request.app.state, "fabric_coordinator", None)
        if coordinator is None:
            reasons.append("no fabric coordinator")
        else:
            from augmentum.fabric.capabilities import KIND_IMAGE_GENERATION
            peers = {
                nid for nid, _ in coordinator.find_peers_with_capability(
                    KIND_IMAGE_GENERATION
                )
            }
            if not peers:
                reasons.append(
                    "no fabric peer advertises image.generation"
                )
    except Exception:
        # diagnostic only; never let it mask the real 503
        log.debug("require_image_fabric_probe_failed", exc_info=True)

    raise HTTPException(
        503,
        "no image-generation path available: " + "; ".join(reasons)
        + ". Configure a cloud provider or pair a fabric peer with a GPU.",
    )


def _require_image_info(request: Request):
    """Raise 503 only if the image subsystem was never attempted.

    Softer guard for read-only endpoints (models list, hardware info) that
    don't need the generation queue — they just need ``image_enabled`` to be
    True and for at least some of the subsystem state to exist.
    """
    if not settings.image_enabled:
        raise HTTPException(503, "Image generation is not enabled")


async def _resolve_cloud_provider(
    request: Request, provider_id: str, model: str,
) -> dict | None:
    """Check if the request should be routed to a cloud image provider.

    Returns the provider dict if cloud routing applies, None otherwise.
    Checks: explicit cloud: prefix on provider_id/model, then looks up
    the model name in cloud provider catalogs from the DB.
    """
    from augmentum.proxy.cloud_image_routes import (
        _get_conn,
        _get_default_image_provider,
        _fetch_cloud_models,
    )

    is_cloud_prefix = (
        provider_id.startswith("cloud:") or model.startswith("cloud:")
    )
    local_available = bool(getattr(request.app.state, "image_queue", None))

    conn = _get_conn(request)
    if not conn:
        return None

    # Explicit cloud: prefix
    if is_cloud_prefix:
        pid = provider_id.replace("cloud:", "") or model.replace("cloud:", "")
        if pid:
            from augmentum.proxy.cloud_image_routes import _get_image_provider_by_id
            prov = await _get_image_provider_by_id(conn, pid)
            if prov:
                return prov
        # Fallback to default provider
        return await _get_default_image_provider(conn)

    # Look up model name in all enabled cloud provider catalogs
    try:
        cursor = await conn.execute(
            "SELECT id, name, base_url, api_key, default_model, default_quality "
            "FROM image_providers WHERE is_enabled = 1 "
            "ORDER BY is_default DESC, name"
        )
        rows = await cursor.fetchall()
        for r in rows:
            provider = {
                "id": r[0], "name": r[1], "base_url": r[2],
                "api_key": decrypt_api_key(r[3]), "default_model": r[4], "default_quality": r[5],
            }
            catalog = await _fetch_cloud_models(provider)
            for cm in catalog:
                if cm["name"] == model:
                    return provider
    except Exception as exc:
        log.debug("image_cloud_provider_resolve_failed", error=str(exc))

    # No local GPU available — fall back to default cloud provider
    if not local_available:
        return await _get_default_image_provider(conn)

    return None


async def _maybe_route_image_to_peer(
    req: "GenerateRequest", request: Request, model: str,
) -> JSONResponse | None:
    """If a fabric peer advertises this image model and we don't have it
    locally, run the generation there and return the result.

    Returns ``None`` for stay-local (the common case). When a peer is
    selected, runs the full peer round-trip (POST + bytes fetch), writes
    the bytes into our local image_output store + image_generations row,
    and returns a ``GenerateResponse`` pointing at the local image_id —
    so consumers downstream (chat embeds, library browser, share links)
    can't tell the image was rendered on another box.

    Failures bubble up as HTTPException(502) with the operator-facing
    message from RemoteImageError. We don't silently fall back to local
    because (a) the operator chose a peer model deliberately, (b) the
    local pipeline almost certainly can't serve a model whose weights
    aren't here.
    """
    director = getattr(request.app.state, "fabric_director", None)
    if director is None or not model:
        return None

    # Local-can-serve is the UNION of two authoritative sources:
    #   1) ImagePersistence.list_models() -- SQLite ``image_models`` rows.
    #      Populated only when a model is registered via save_model()
    #      (download flow, civitai import, etc.).
    #   2) ModelManager.list_local_models() -- disk scan of model_dir +
    #      the baked-in system_dir (e.g. DreamShaper 8 in the GPU image).
    #      Catches models the user dropped in by hand and system models
    #      that ship with the image and never went through save_model().
    # Using only (1) means baked-in / hand-dropped models get mistaken
    # for "missing locally" -- the dropdown shows them (it scans disk),
    # so the user picks one, we look in SQLite, find nothing, ask the
    # fabric for a peer that has it, and bounce to a remote render even
    # though the local pipeline could have served it instantly.
    persistence = getattr(request.app.state, "image_persistence", None)
    model_mgr = getattr(request.app.state, "image_model_manager", None)
    local_can_serve = False
    if persistence is not None:
        try:
            local_models = await persistence.list_models()
            local_can_serve = any(
                getattr(m, "name", "") == model for m in local_models
            )
        except Exception as exc:
            log.debug("fabric_image_local_check_failed", error=str(exc))
    if not local_can_serve and model_mgr is not None:
        try:
            disk_models = model_mgr.list_local_models()
            local_can_serve = any(
                m.get("name", "") == model for m in disk_models
            )
        except Exception as exc:
            log.debug("fabric_image_local_check_disk_failed", error=str(exc))

    route = await director.maybe_route_image(
        model_id=model, local_can_serve=local_can_serve,
    )
    if route is None:
        # Two cases collapse here:
        #   (a) local_can_serve=True: stay local, the common path.
        #   (b) local_can_serve=False AND no peer advertises the model:
        #       falling through to local would error confusingly when
        #       the local pipeline tries to load a model it doesn't
        #       have. Raise a clean 400 with the peer diagnostic so the
        #       operator can see what we checked.
        if not local_can_serve:
            connected = []
            advertised: list[str] = []
            try:
                coord = getattr(request.app.state, "fabric_coordinator", None)
                if coord is not None:
                    connected = coord.connected_peer_ids()
                    from augmentum.fabric.capabilities import KIND_IMAGE_GENERATION
                    for _, cap in coord.find_peers_with_capability(KIND_IMAGE_GENERATION):
                        mid = getattr(cap, "model_id", "")
                        if mid:
                            advertised.append(mid)
            except Exception:
                log.debug("fabric_image_diag_collect_failed", exc_info=True)
            log.warning(
                "fabric_image_route_no_peer_match",
                model=model, connected_peers=connected,
                advertised_models=advertised,
            )
            raise HTTPException(
                status_code=400,
                detail=(
                    f"image model {model!r} is not installed locally and no "
                    f"connected fabric peer advertises it "
                    f"(connected_peers={connected}, advertised={advertised})"
                ),
            )
        return None

    peer_node_id, peer_addr = route
    coordinator = getattr(request.app.state, "fabric_coordinator", None)
    fabric_http = getattr(request.app.state, "fabric_http_client", None)
    if coordinator is None or fabric_http is None:
        # Director picked a peer but we lack the infrastructure to call
        # it. Raise rather than fall through to local — local can't serve
        # this model (we already established that above) so the silent
        # fallback would just produce a different confusing error.
        log.warning(
            "fabric_image_route_aborted_missing_dep",
            has_coordinator=coordinator is not None,
            has_http=fabric_http is not None,
        )
        raise HTTPException(
            status_code=503,
            detail=(
                "fabric image dispatch infrastructure not initialised "
                "(coordinator/http_client missing); cannot route to peer"
            ),
        )

    user = request.scope.get("user")
    user_id = user.id if user else ""
    if not user_id:
        raise HTTPException(401, "Unauthorized")

    # Serialise the request the same shape the peer's /api/image/generate
    # expects (it's a GenerateRequest on both sides). Pydantic v2:
    # ``model_dump`` is the canonical serialiser; older v1 callers used
    # ``dict()`` — both still work, prefer model_dump for clarity.
    payload = req.model_dump() if hasattr(req, "model_dump") else req.dict()

    from augmentum.fabric.image_client import (
        RemoteImageError,
        generate_image_via_peer,
    )

    try:
        image_bytes, metadata = await generate_image_via_peer(
            http_client=fabric_http,
            identity=coordinator._identity,
            user_id=user_id,
            peer_addr=peer_addr,
            generate_request_payload=payload,
        )
    except RemoteImageError as exc:
        msg = str(exc)
        log.warning(
            "fabric_image_route_failed",
            peer_node_id=peer_node_id, model=model, error=msg[:200],
        )
        # Resolve a friendly peer label for the operator-facing message.
        peer_host = peer_node_id[:12]
        try:
            st = coordinator.peer_state(peer_node_id)
            if st is not None and getattr(st, "paired", None) is not None:
                peer_host = st.paired.hostname or peer_host
        except Exception:
            pass
        # A peer that advertised this model but can't actually serve it
        # (older build / capability mismatch) is not a transient blip —
        # make the next step obvious instead of leaking a raw "peer
        # returned 501". Local can't serve it either (that's why we
        # routed), so the fix is on the peer or via a different model.
        if "501" in msg or "older Augmentum build" in msg:
            detail = (
                f"The peer '{peer_host}' advertises image model {model!r} but is "
                f"running an older Augmentum build that can't serve cross-peer "
                f"image generation. Rebuild that peer, or choose a model served "
                f"locally or by a cloud provider."
            )
        else:
            detail = f"Image generation on peer '{peer_host}' failed: {msg[:200]}"
        raise HTTPException(502, detail) from None

    # Write to local image_output + register a local image_id so the
    # result lives in our library exactly like a locally-rendered one.
    import os
    import uuid

    image_id = str(uuid.uuid4())
    output_dir = settings.image_output_dir or f"{settings.data_dir}/image_output"
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, f"{image_id}.png")
    try:
        with open(file_path, "wb") as f:
            f.write(image_bytes)
    except OSError as exc:
        log.warning(
            "fabric_image_local_write_failed",
            peer_node_id=peer_node_id, image_id=image_id, error=str(exc),
        )
        raise HTTPException(500, f"could not save peer image: {exc}") from None

    if persistence is not None:
        try:
            await persistence.save_generation(
                image_id=image_id,
                session_id=str(metadata.get("session_id", "") or ""),
                prompt=str(metadata.get("prompt", req.prompt or "")),
                negative_prompt=str(metadata.get("negative_prompt", req.negative_prompt or "")),
                model=str(metadata.get("model", model)),
                seed=int(metadata.get("seed", req.seed or -1)),
                width=int(metadata.get("width", req.width or 0)),
                height=int(metadata.get("height", req.height or 0)),
                steps=int(metadata.get("steps", req.steps or 0)),
                cfg_scale=float(metadata.get("cfg_scale", req.cfg_scale or 0)),
                preset=str(metadata.get("preset", req.preset or "")),
                loras=metadata.get("loras", []) if isinstance(metadata.get("loras"), list) else [],
                file_path=file_path,
                job_type=str(metadata.get("job_type", "txt2img")),
                user_id=user_id,
            )
        except Exception as exc:
            log.warning(
                "fabric_image_save_generation_failed",
                image_id=image_id, error=str(exc),
            )

    log.info(
        "fabric_image_route_completed",
        peer_node_id=peer_node_id, model=model, image_id=image_id,
        bytes=len(image_bytes),
    )

    return JSONResponse(content={
        "image_id": image_id,
        "job_id": "fabric",
        "status": "completed",
        "url": f"/api/image/{image_id}",
        "seed": int(metadata.get("seed", req.seed or -1)),
        "prompt": metadata.get("prompt", req.prompt),
        "negative_prompt": metadata.get("negative_prompt", req.negative_prompt),
        "width": metadata.get("width", req.width),
        "height": metadata.get("height", req.height),
        "steps": metadata.get("steps", req.steps),
        "model": metadata.get("model", model),
        "source": "fabric",
        "augmentum_peer": {"node_id": peer_node_id, "addr": peer_addr},
    })


def _resolve_dimensions(req: GenerateRequest) -> tuple[int, int]:
    """Resolve width/height from request, aspect ratio, or defaults."""
    if req.width and req.height:
        return req.width, req.height

    base_w = settings.image_default_width
    base_h = settings.image_default_height

    if req.aspect == AspectRatio.PORTRAIT:
        return base_w, int(base_h * 1.5)
    if req.aspect == AspectRatio.LANDSCAPE:
        return int(base_w * 1.5), base_h
    return base_w, base_h


# --- Prompt Enhancement ---


class PromptLLMRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10000)
    model: str = ""  # LLM model for the enhancement/generation call
    image_model: str = ""  # target image model (for style-aware prompting)


async def _resolve_llm_backend(request: Request, model: str):
    """Resolve an LLM backend and model name for prompt operations.

    Uses the role-based resolver (utility role) with ``image_prompt_condense_model``
    as the per-feature override so all prompt-LLM calls follow the same fallback
    chain: per-request model → image_prompt_condense_model → utility_model → primary.
    """
    provider_reg = getattr(request.app.state, "provider_registry", None)
    if not provider_reg or not provider_reg.backends:
        raise HTTPException(503, "No LLM backend available")

    try:
        return await provider_reg.resolve_model_for_role(
            "utility",
            override=model or settings.image_prompt_condense_model,
            settings=settings,
        )
    except Exception:
        log.warning("llm_model_resolve_failed", exc_info=True)
        raise HTTPException(503, "Could not resolve LLM model — check backend connection")


class SceneMessageItem(BaseModel):
    role: str
    content: str


class SceneImageRequest(BaseModel):
    """Request to generate a scene image from narrative context."""

    session_id: str = Field(..., min_length=1, max_length=200)
    instruction: str = Field(default="", max_length=2000)
    model: str = ""  # image model override (empty = use UI-selected or default)
    messages: list[SceneMessageItem] = Field(default_factory=list)  # conversation history from UI
    width: int = 0
    height: int = 0
    steps: int = 0
    cfg: float = 0.0
    seed: int = -1
    sampler: str = ""
    # Character data from UI — supplements engine's cached card (which may be
    # stale if the user edited traits after starting the chat).
    character_name: str = ""
    visual_traits: str = ""


@router.post("/generate-scene")
async def generate_scene_image(req: SceneImageRequest, request: Request):
    """Generate a scene image using the narrative distiller pipeline.

    Gathers character card, world state, and conversation context from
    the session's narrative engine, distills a prompt via LLM, and submits
    an image generation job.  Returns the image URL on success.
    """
    # Don't require local image queue — cloud providers can handle it
    local_available = bool(getattr(request.app.state, "image_queue", None))
    if not local_available:
        # Check if cloud providers exist
        cloud_provider = await _resolve_cloud_provider(request, "", req.model or "")
        if not cloud_provider:
            raise HTTPException(503, "Image generation is not enabled (no local GPU or cloud provider)")

    from augmentum.classifier.router import Mode
    from augmentum.image.queue import GenerationJob
    from augmentum.models.base import InternalChatRequest, Message
    from augmentum.proxy.handler_factory import get_handler_for_mode

    queue = getattr(request.app.state, "image_queue", None)
    if not queue:
        raise HTTPException(503, "Image subsystem not available")

    # Build a narrative handler for this session (reuses cached engine)
    registry = getattr(request.app.state, "provider_registry", None)
    if not registry:
        raise HTTPException(503, "No backend available")
    try:
        backend, _model_name = await registry.resolve_model_for_role("utility", settings=settings)
    except Exception as exc:
        raise HTTPException(503, f"No LLM backend available: {exc}")
    img_user = request.scope.get("user")
    img_uid = img_user.id if img_user else ""
    handler = get_handler_for_mode(
        mode=Mode.NARRATIVE,
        backend=backend,
        session_id=req.session_id,
        app_state=request.app.state,
        user_id=img_uid,
    )

    gen_fn = getattr(handler, "_generate_scene_image", None)
    if not gen_fn:
        raise HTTPException(500, "Narrative handler failed to initialize — check backend configuration")

    # Ensure engine state is loaded before generating
    load_fn = getattr(handler, "_ensure_state_loaded", None)
    if load_fn:
        await load_fn()

    # Build request with real conversation history from the UI
    if req.messages:
        msgs = [Message(role=m.role, content=m.content) for m in req.messages]
    else:
        msgs = [Message(role="user", content=req.instruction or "the current scene")]
    context_request = InternalChatRequest(model=_model_name, messages=msgs)

    # Pass UI image panel settings as overrides
    image_overrides = {}
    if req.model:
        image_overrides["model"] = req.model
    if req.width > 0:
        image_overrides["width"] = req.width
    if req.height > 0:
        image_overrides["height"] = req.height
    if req.steps > 0:
        image_overrides["steps"] = req.steps
    if req.cfg > 0:
        image_overrides["cfg"] = req.cfg
    if req.seed >= 0:
        image_overrides["seed"] = req.seed
    if req.sampler:
        image_overrides["sampler"] = req.sampler
    # Pass UI character data so the distiller can use fresh visual traits
    # even when the engine's cached card is stale or missing.
    if req.character_name:
        image_overrides["character_name"] = req.character_name
    if req.visual_traits:
        image_overrides["visual_traits"] = req.visual_traits
    image_url = await gen_fn(req.instruction or "", context_request, image_overrides=image_overrides)

    if not image_url:
        raise HTTPException(500, "Scene image generation failed")

    return {"url": image_url, "image_id": image_url.split("/")[-1]}


@router.post("/enhance-prompt")
async def enhance_prompt_endpoint(req: PromptLLMRequest, request: Request):
    """Enhance an image prompt using the LLM."""
    from augmentum.image.prompt_condenser import enhance_prompt

    backend, model = await _resolve_llm_backend(request, req.model)
    enhanced = await enhance_prompt(
        req.prompt, backend, model=model, image_model=req.image_model,
    )
    return {"prompt": enhanced}


@router.post("/generate-negative")
async def generate_negative_endpoint(req: PromptLLMRequest, request: Request):
    """Generate a negative prompt based on the positive prompt."""
    from augmentum.image.prompt_condenser import generate_negative_prompt

    backend, model = await _resolve_llm_backend(request, req.model)
    negative = await generate_negative_prompt(
        req.prompt, backend, model=model, image_model=req.image_model,
    )
    return {"negative_prompt": negative}


class ExtractTraitsRequest(BaseModel):
    description: str = Field(..., min_length=1, max_length=50000)
    name: str = ""  # Character/card name for context
    scenario: str = ""  # Scenario text to help detect card type
    personality: str = ""  # Personality text (may contain character names)
    model: str = ""  # LLM model to use for extraction


@router.post("/extract-visual-traits")
async def extract_visual_traits_endpoint(req: ExtractTraitsRequest, request: Request):
    """Extract physical/visual traits from a character description using the LLM.

    Detects card type (single character, ensemble, world/narrator) and outputs
    the appropriate format:
    - Single: comma-separated trait list
    - Ensemble: <CharName> traits per character
    - World/RPG: scene descriptors for environments
    """
    from augmentum.models.base import InternalChatRequest, Message

    # Use prompt condense model (same model used for image prompt work)
    registry = getattr(request.app.state, "provider_registry", None)
    if not registry or not registry.backends:
        raise HTTPException(503, "No LLM backend available")
    backend, model = await registry.resolve_model_for_role(
        "utility",
        override=req.model or settings.image_prompt_condense_model,
        settings=settings,
    )

    system = (
        "You are a character card visual trait extractor. Analyze the description "
        "and determine what type of card this is, then extract visual information "
        "in the correct format.\n\n"
        "CARD TYPES:\n"
        "1. SINGLE CHARACTER — one named character with physical traits.\n"
        "   Output: comma-separated visual descriptors.\n"
        "   Example: blonde hair, blue eyes, athletic build, school uniform\n\n"
        "2. ENSEMBLE / MULTI-CHARACTER — multiple named characters.\n"
        "   Output: each character on its own line with <Name> tag prefix.\n"
        "   Example:\n"
        "   <Alice> red hair, green eyes, freckles, petite\n"
        "   <Bob> tall, dark skin, glasses, leather jacket\n\n"
        "3. WORLD / NARRATOR / RPG — no specific character, describes a setting.\n"
        "   Output: comma-separated scene/environment descriptors for image generation.\n"
        "   Example: medieval fantasy, stone castles, enchanted forests, torchlit dungeons\n\n"
        "RULES:\n"
        "- Extract ONLY what is explicitly stated — do NOT invent or assume\n"
        "- Use the exact colors and descriptors from the text\n"
        "- For characters: hair color/style, eye color, skin tone, body type, height, "
        "age appearance, distinguishing marks, species traits, clothing\n"
        "- For worlds: setting genre, key locations, atmosphere, visual style, "
        "architecture, lighting, color palette\n"
        "- Keep it concise — visual facts only, no full sentences\n"
        "- Output ONLY the formatted traits, nothing else — no preamble or explanation"
    )

    # Give context to help detect card type
    user_parts = []
    if req.name:
        user_parts.append(f"Card name: {req.name}")
    if req.scenario:
        user_parts.append(f"Scenario: {req.scenario[:500]}")
    if req.personality:
        user_parts.append(f"Personality: {req.personality[:500]}")
    user_parts.append(f"Description:\n{req.description}")

    chat_request = InternalChatRequest(
        model=model,
        messages=[
            Message(role="system", content=system),
            Message(role="user", content="\n\n".join(user_parts)),
        ],
        stream=False,
        temperature=0.1,
        max_tokens=800,
    )

    try:
        response = await backend.chat(chat_request)
        traits = response.message.content.strip()
        return {"visual_traits": traits}
    except Exception as exc:
        log.warning("extract_visual_traits_failed", error=str(exc))
        raise HTTPException(500, f"Trait extraction failed: {exc}") from exc


# --- Generation ---


@router.post("/generate")
async def generate_image(req: GenerateRequest, request: Request):
    provider_id = getattr(req, "provider_id", "") or ""
    model = req.model or ""

    # Resolve cloud provider: explicit cloud: prefix, or look up model name in DB
    cloud_provider = await _resolve_cloud_provider(request, provider_id, model)

    if cloud_provider:
        from augmentum.proxy.cloud_image_routes import (
            CloudGenerateRequest,
            cloud_generate,
        )
        cloud_req = CloudGenerateRequest(
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            provider_id=cloud_provider["id"],
            model=model,
            width=req.width or settings.image_default_width,
            height=req.height or settings.image_default_height,
            quality=req.quality if hasattr(req, "quality") else "standard",
            seed=req.seed,
        )
        return await cloud_generate(cloud_req, request)

    # Fabric dispatch: if a paired peer advertises this image model and
    # we don't have it locally, route the generation there. Otherwise the
    # local pipeline would try to download/load a model that may not fit
    # this node's VRAM — exactly the failure mode where fabric should be
    # orchestrating transparently.
    #
    # Local-first: when the model IS in our local image_models table,
    # always stay local. Cross-peer transfer of an N-MB image is more
    # expensive than running the diffusion on already-resident weights.
    fabric_response = await _maybe_route_image_to_peer(req, request, model)
    if fabric_response is not None:
        return fabric_response

    _require_image(request)

    from augmentum.image.queue import GenerationJob

    queue = getattr(request.app.state, "image_queue", None)
    preset_manager = getattr(request.app.state, "image_preset_manager", None)
    cache = getattr(request.app.state, "image_cache", None)

    # Apply preset if specified
    prompt = req.prompt
    negative_prompt = req.negative_prompt
    preset_name = req.preset or settings.image_default_preset

    sampler = req.sampler or req.scheduler or ""

    if preset_name:
        preset = preset_manager.get(preset_name)
        if preset:
            prompt, negative_prompt = preset.apply(prompt, negative_prompt)
            if not req.steps:
                req.steps = preset.steps
            if not req.cfg_scale:
                req.cfg_scale = preset.cfg_scale
            # Use preset's sampler as default when user didn't specify one
            if not sampler and getattr(preset, "sampler", ""):
                sampler = preset.sampler

    width, height = _resolve_dimensions(req)
    model = req.model or settings.image_default_model

    # Apply distilled-model-aware defaults for steps / cfg_scale
    from augmentum.image.distilled import apply_distilled_defaults

    steps, cfg_scale = apply_distilled_defaults(model, req.steps, req.cfg_scale)

    # Resolve negative prompt: explicit > preset > config default > pipeline default
    from augmentum.image.defaults import resolve_negative_prompt

    hw = getattr(request.app.state, "image_hardware", None)
    pipeline_type_str = hw.recommended_pipeline if hw else "sd15"
    negative_prompt = resolve_negative_prompt(
        negative_prompt, pipeline_type_str, settings.image_default_negative_prompt,
    )

    _img_user = request.scope.get("user")
    _img_user_id = _img_user.id if _img_user else ""
    if not _img_user_id:
        raise HTTPException(401, "Unauthorized")

    # Check cache
    if cache and req.seed != -1:
        cached_id = await cache.get(
            prompt, negative_prompt, model, req.seed, width, height, steps, cfg_scale,
            user_id=_img_user_id,
        )
        if cached_id:
            return GenerateResponse(
                image_id=cached_id,
                job_id="cached",
                status=JobStatus.COMPLETED,
                url=f"/api/image/{cached_id}",
                seed=req.seed,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                model=model,
            )

    # VRAM safety check runs inside the queue worker (_generate_fn) using
    # live VRAM data and accounting for model swaps.  The route-level check
    # was removed because it used a stale startup snapshot and blocked
    # legitimate model-swap requests where the old model would be unloaded.

    # Resolve per-generation quality settings (request value > config default)
    guidance_rescale = req.guidance_rescale if req.guidance_rescale is not None else settings.image_cfg_rescale
    hires_fix = req.hires_fix if req.hires_fix is not None else settings.image_hires_fix
    hires_scale = req.hires_scale if req.hires_scale is not None else settings.image_hires_scale
    hires_denoise = req.hires_denoise if req.hires_denoise is not None else settings.image_hires_denoise

    job = GenerationJob(
        prompt=prompt,
        negative_prompt=negative_prompt,
        model=model,
        preset=preset_name,
        width=width,
        height=height,
        steps=steps,
        cfg_scale=cfg_scale,
        seed=req.seed,
        sampler=sampler,
        scheduler=req.scheduler or "",
        loras=[lora.model_dump() for lora in req.loras] if req.loras else [],
        condense_model=req.condense_model,
        enhance_prompt=req.enhance_prompt,
        condense_prompt=req.condense_prompt,
        guidance_rescale=guidance_rescale,
        hires_fix=hires_fix,
        hires_scale=hires_scale,
        hires_denoise=hires_denoise,
        clip_skip=req.clip_skip,
        ip_adapter_image=req.ip_adapter_image,
        ip_adapter_scale=req.ip_adapter_scale,
        user_id=_img_user_id,
        # Closed vocabulary — only the architect-dispatched client path
        # may claim companion provenance.
        origin="companion" if req.origin == "companion" else "",
    )

    try:
        job = await queue.submit(job)
    except RuntimeError as exc:
        raise HTTPException(429, str(exc)) from exc

    # Wait for result (with timeout)
    try:
        result = await queue.wait_for_result(job, timeout=settings.image_generation_timeout)
    except TimeoutError as exc:
        raise HTTPException(504, "Image generation timed out") from exc
    except Exception as exc:
        raise HTTPException(500, f"Generation failed: {exc}") from exc

    image_id = result["image_id"]

    # Store in cache
    if cache and result.get("seed", -1) != -1:
        await cache.put(
            prompt, negative_prompt, model, result["seed"],
            width, height, steps, cfg_scale, image_id,
            user_id=_img_user_id,
        )

    return GenerateResponse(
        image_id=image_id,
        job_id=job.job_id,
        status=JobStatus.COMPLETED,
        url=f"/api/image/{image_id}",
        seed=result.get("seed", -1),
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        steps=steps,
        model=model,
    )


@router.post("/img2img")
async def img2img(req: Img2ImgRequest, request: Request):
    # Cloud routing
    model = req.model or ""
    cloud_provider = await _resolve_cloud_provider(request, "", model)
    if cloud_provider:
        from augmentum.proxy.cloud_image_routes import (
            CloudEditRequest,
            cloud_edit,
        )
        cloud_req = CloudEditRequest(
            prompt=req.prompt,
            provider_id=cloud_provider["id"],
            model=model,
            source_image=req.source_image,
            strength=req.strength,
            width=req.width or settings.image_default_width,
            height=req.height or settings.image_default_height,
        )
        return await cloud_edit(cloud_req, request)

    _require_image(request)

    from augmentum.image.queue import GenerationJob

    queue = getattr(request.app.state, "image_queue", None)
    model = req.model or settings.image_default_model
    steps = req.steps or settings.image_default_steps
    cfg_scale = req.cfg_scale or settings.image_default_cfg

    from augmentum.image.distilled import apply_distilled_defaults
    steps, cfg_scale = apply_distilled_defaults(model, steps, cfg_scale)

    _i2i_user = request.scope.get("user")
    _i2i_user_id = _i2i_user.id if _i2i_user else ""
    if not _i2i_user_id:
        raise HTTPException(401, "Unauthorized")

    job = GenerationJob(
        job_type=JobType.IMG2IMG,
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        model=model,
        preset=req.preset,
        width=req.width or 0,
        height=req.height or 0,
        steps=steps,
        cfg_scale=cfg_scale,
        seed=req.seed,
        sampler=req.sampler or "",
        source_image=req.source_image,
        strength=req.strength,
        condense_model=req.condense_model,
        enhance_prompt=req.enhance_prompt,
        condense_prompt=req.condense_prompt,
        user_id=_i2i_user_id,
    )

    try:
        job = await queue.submit(job)
    except RuntimeError as exc:
        raise HTTPException(429, str(exc)) from exc

    try:
        result = await queue.wait_for_result(job, timeout=settings.image_generation_timeout)
    except TimeoutError as exc:
        raise HTTPException(504, "Image generation timed out") from exc
    except Exception as exc:
        raise HTTPException(500, f"img2img failed: {exc}") from exc

    return GenerateResponse(
        image_id=result["image_id"],
        job_id=job.job_id,
        status=JobStatus.COMPLETED,
        url=f"/api/image/{result['image_id']}",
        seed=result.get("seed", -1),
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        width=result.get("width", 0),
        height=result.get("height", 0),
        steps=steps,
        model=model,
    )


@router.post("/inpaint")
async def inpaint(req: InpaintRequest, request: Request):
    # Cloud routing
    model = req.model or ""
    cloud_provider = await _resolve_cloud_provider(request, "", model)
    if cloud_provider:
        from augmentum.proxy.cloud_image_routes import (
            CloudEditRequest,
            cloud_edit,
        )
        cloud_req = CloudEditRequest(
            prompt=req.prompt,
            provider_id=cloud_provider["id"],
            model=model,
            source_image=req.source_image,
            mask_image=req.mask_image,
            strength=req.strength,
            width=req.width or settings.image_default_width,
            height=req.height or settings.image_default_height,
        )
        return await cloud_edit(cloud_req, request)

    _require_image(request)

    from augmentum.image.queue import GenerationJob

    queue = getattr(request.app.state, "image_queue", None)
    model = req.model or settings.image_default_model
    steps = req.steps or settings.image_default_steps
    cfg_scale = req.cfg_scale or settings.image_default_cfg

    from augmentum.image.distilled import apply_distilled_defaults
    steps, cfg_scale = apply_distilled_defaults(model, steps, cfg_scale)

    _inp_user = request.scope.get("user")
    _inp_user_id = _inp_user.id if _inp_user else ""
    if not _inp_user_id:
        raise HTTPException(401, "Unauthorized")

    job = GenerationJob(
        job_type=JobType.INPAINT,
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        model=model,
        preset=req.preset,
        width=req.width or 0,
        height=req.height or 0,
        steps=steps,
        cfg_scale=cfg_scale,
        seed=req.seed,
        sampler=req.sampler or "",
        source_image=req.source_image,
        mask_image=req.mask_image,
        strength=req.strength,
        mask_blur=req.mask_blur,
        inpaint_mode=req.inpaint_mode,
        inpaint_full_res=req.inpaint_full_res,
        inpaint_padding=req.inpaint_padding,
        condense_model=req.condense_model,
        enhance_prompt=req.enhance_prompt,
        condense_prompt=req.condense_prompt,
        user_id=_inp_user_id,
    )

    try:
        job = await queue.submit(job)
    except RuntimeError as exc:
        raise HTTPException(429, str(exc)) from exc

    try:
        result = await queue.wait_for_result(job, timeout=settings.image_generation_timeout)
    except TimeoutError as exc:
        raise HTTPException(504, "Image generation timed out") from exc
    except Exception as exc:
        raise HTTPException(500, f"Inpainting failed: {exc}") from exc

    return GenerateResponse(
        image_id=result["image_id"],
        job_id=job.job_id,
        status=JobStatus.COMPLETED,
        url=f"/api/image/{result['image_id']}",
        seed=result.get("seed", -1),
        prompt=req.prompt,
        negative_prompt=req.negative_prompt,
        width=result.get("width", 0),
        height=result.get("height", 0),
        steps=steps,
        model=model,
    )


# --- Job control ---


@router.post("/cancel")
async def cancel_current(request: Request):
    """Cancel the currently running generation job (if any).

    Only the job's owner (or an admin) can cancel — prevents one user
    from killing another's generation on a shared GPU.
    """
    _require_image(request)
    queue = getattr(request.app.state, "image_queue", None)
    job_id = queue._current_job_id
    if not job_id:
        return {"cancelled": False, "reason": "No job currently running"}
    user = request.scope.get("user")
    uid = user.id if user else ""
    job = queue.get_job(job_id)
    if job and uid and job.user_id and job.user_id != uid:
        is_admin = getattr(user, "is_admin", False)
        if not is_admin:
            return {"cancelled": False, "reason": "Not your job"}
    cancelled = queue.cancel_job(job_id)
    return {"job_id": job_id, "cancelled": cancelled}


@router.post("/unload")
async def unload_model(request: Request):
    """Unload the currently loaded image model to free VRAM."""
    _require_image(request)
    pipeline_reg = getattr(request.app.state, "image_pipeline_registry", None)
    if not pipeline_reg:
        raise HTTPException(503, "Image pipeline registry not available")
    model = pipeline_reg.current_model
    if not pipeline_reg.is_loaded:
        return {"unloaded": False, "reason": "No model currently loaded"}
    await pipeline_reg.unload()
    from augmentum.resource.ledger import invalidate as _invalidate_resource
    _invalidate_resource(request.app.state, "image")
    log.info("manual_model_unload", model=model)
    return {"unloaded": True, "model": model}


# --- Post-Processing (upscale, background removal) ---


@router.post("/{image_id}/upscale")
async def upscale_image_route(image_id: str, request: Request):
    """Upscale an image using spandrel. Saves as a new image."""
    import os
    from augmentum.image.postprocess import upscale_image

    body = {}
    try:
        body = await request.json()
    except Exception as exc:
        log.debug("upscale_body_parse_failed", error=str(exc))
    scale = body.get("scale", 4)

    # Resolve source image path
    file_path = await _resolve_image_path(image_id, request)
    if not file_path:
        raise HTTPException(404, "Source image not found")

    user = request.scope.get("user")
    user_id = user.id if user else ""

    try:
        new_id, new_path, w, h = await upscale_image(file_path, scale=scale)
    except FileNotFoundError as e:
        raise HTTPException(503, sanitize_error_detail(str(e)))
    except Exception:
        log.warning("upscale_failed", image_id=image_id, exc_info=True)
        raise HTTPException(500, "Upscale failed")

    # Persist in DB
    persistence = getattr(request.app.state, "image_persistence", None)
    if persistence and user_id:
        # Get source metadata to carry forward
        source = await persistence.get_generation(image_id, user_id=user_id) or {}
        await persistence.save_generation(
            image_id=new_id,
            session_id=source.get("session_id", ""),
            prompt=source.get("prompt", ""),
            negative_prompt=source.get("negative_prompt", ""),
            model=source.get("model", "upscale"),
            seed=source.get("seed", -1),
            width=w, height=h,
            steps=0, cfg_scale=0,
            preset="", loras=[],
            file_path=new_path,
            job_type="upscale",
            source_image_id=image_id,
            user_id=user_id,
        )

    log.info("image_upscaled", source=image_id, new_id=new_id, scale=scale, size=f"{w}x{h}")
    return {
        "image_id": new_id,
        "url": f"/api/image/{new_id}",
        "width": w, "height": h,
        "source_id": image_id,
    }


@router.post("/{image_id}/remove-bg")
async def remove_bg_route(image_id: str, request: Request):
    """Remove background from an image using rembg. Saves as a new RGBA PNG."""
    from augmentum.image.postprocess import remove_background

    file_path = await _resolve_image_path(image_id, request)
    if not file_path:
        raise HTTPException(404, "Source image not found")

    user = request.scope.get("user")
    user_id = user.id if user else ""

    try:
        new_id, new_path, w, h = await remove_background(file_path)
    except Exception:
        log.warning("remove_bg_failed", image_id=image_id, exc_info=True)
        raise HTTPException(500, "Background removal failed")

    persistence = getattr(request.app.state, "image_persistence", None)
    if persistence and user_id:
        source = await persistence.get_generation(image_id, user_id=user_id) or {}
        await persistence.save_generation(
            image_id=new_id,
            session_id=source.get("session_id", ""),
            prompt=source.get("prompt", ""),
            negative_prompt=source.get("negative_prompt", ""),
            model=source.get("model", "rembg"),
            seed=source.get("seed", -1),
            width=w, height=h,
            steps=0, cfg_scale=0,
            preset="", loras=[],
            file_path=new_path,
            job_type="remove_bg",
            source_image_id=image_id,
            user_id=user_id,
        )

    log.info("background_removed", source=image_id, new_id=new_id)
    return {
        "image_id": new_id,
        "url": f"/api/image/{new_id}",
        "width": w, "height": h,
        "source_id": image_id,
    }


async def _resolve_image_path(image_id: str, request) -> str | None:
    """Find the file path for an image by ID.

    Resolution order:
      1. ``image_generations`` row (AI-generated images, the original
         use case for these edit endpoints)
      2. ``image_output_dir/{id}.png`` (cloud-generated images written
         to disk but never recorded in image_generations)
      3. ``file_index`` row whose id is ``image_id`` AND whose
         mime_type is an image (uploads, chat_images, artifact images,
         any future image-bearing source). The mime_type guard
         prevents the route from accepting a PDF or doc id and trying
         to run rembg on it.

      The file_index fall-through is what lets ``remove-bg`` /
      ``upscale`` / etc. work on uploaded phone photos. Before, those
      endpoints only resolved IDs against the AI-image stores, so
      uploaded images came back 404 and the lightbox edit buttons
      silently no-op'd.
    """
    import os

    user = request.scope.get("user")
    uid = user.id if user else ""

    persistence = getattr(request.app.state, "image_persistence", None)
    if persistence and uid:
        gen = await persistence.get_generation(image_id, user_id=uid)
        if gen and os.path.exists(gen["file_path"]):
            return gen["file_path"]

    output_dir = settings.image_output_dir or f"{settings.data_dir}/image_output"
    fallback = os.path.join(output_dir, f"{image_id}.png")
    if os.path.exists(fallback):
        return fallback

    # File-index fall-through. ``real_path`` is set by every adapter
    # whose source maps to a single file on disk (uploads, generated
    # images, voices). Sources without a real_path (chat_images stores
    # bytes in a BLOB column, documents are pre-chunked) are skipped
    # — the postprocess pipeline needs a path, not bytes, so handling
    # those would require staging to a temp file. Out of scope.
    idx = getattr(request.app.state, "file_index", None)
    if idx and uid:
        entry = await idx.get(image_id, user_id=uid)
        if (entry
                and (entry.mime_type or "").startswith("image/")
                and entry.real_path
                and os.path.exists(entry.real_path)):
            return entry.real_path

    return None


# --- Active Settings (synced from image panel UI) ---


class ImageActiveSettings(BaseModel):
    """Current image panel settings pushed by the UI."""
    model: str = ""
    steps: int = 0
    cfg_scale: float = 0.0
    width: int = 0
    height: int = 0
    seed: int = -1
    sampler: str = ""
    preset: str = ""
    negative_prompt: str = ""
    # Per-generation quality optimizations
    guidance_rescale: float = 0.0
    hires_fix: bool = False
    hires_scale: float = 1.5
    hires_denoise: float = 0.5
    # Cloud provider routing — set when a cloud model is selected
    cloud_provider_id: str = ""
    cloud_quality: str = "standard"


def _job_progress_payload(job, queue) -> dict:
    """Serialize a GenerationJob's progress for UI polling.

    Single source of truth for the shape consumed by every progress
    surface (Image Studio panel, voice camera button, illustrate
    moment, scene-gen button, narrative auto-bg badge). Adding a new
    field here surfaces it to every UI automatically.
    """
    import time
    elapsed_s = (
        time.monotonic() - job.started_at
        if job.started_at > 0
        else 0.0
    )
    return {
        "job_id": job.job_id,
        "status": job.status.value,
        "stage": job.stage,
        "steps_done": job.steps_done,
        "steps_total": job.steps_total,
        "elapsed_s": round(elapsed_s, 1),
        "session_id": job.session_id,
        "job_type": job.job_type.value,
        "category": job.category,
        "error": job.error,
    }


def _pre_queue_phase_for_session(app_state, session_id: str) -> dict | None:
    """Look up a pre-queue phase indicator for a session.

    Narrative scene generation runs a distiller LLM for several seconds
    BEFORE any image job exists. Without surfacing this, the UI loader
    sits silent for the distill window. The handler stamps a record on
    ``app_state.scene_image_pre_queue[session_id]`` before calling the
    distiller; this helper exposes it to /api/image/generation-status
    so polling surfaces show "Composing scene prompt" during the gap.

    Cleared by the handler when the job is submitted (or on error).
    """
    if not session_id:
        return None
    pre = getattr(app_state, "scene_image_pre_queue", None)
    if not pre:
        return None
    rec = pre.get(session_id)
    if not rec:
        return None
    import time
    elapsed = time.monotonic() - rec.get("started_at", 0.0)
    return {
        "phase": rec.get("phase", "pre_queue"),
        "stage": rec.get("stage", "Composing scene prompt"),
        "elapsed_s": round(elapsed, 1),
        "session_id": session_id,
    }


@router.get("/job/{job_id}")
async def get_job_status(job_id: str, request: Request):
    """Get the current status and stage of an image generation job."""
    queue = getattr(request.app.state, "image_queue", None)
    if not queue:
        raise HTTPException(503, "Image generation is not enabled")
    job = queue.get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    user = request.scope.get("user")
    uid = user.id if user else ""
    if uid and job.user_id and job.user_id != uid:
        raise HTTPException(404, "Job not found")
    payload = _job_progress_payload(job, queue)
    payload["position"] = queue.get_position(job_id)
    return payload


@router.get("/generation-status")
async def get_generation_status(
    request: Request,
    session_id: str = "",
    category: str = "",
):
    """Get the current generation stage (for UI polling during generation).

    ``session_id`` (optional) scopes the response to a specific narrative
    session — when supplied, we also report the pre-queue phase
    (distiller running) so the UI doesn't show a silent loader during
    the 3-10s LLM call that precedes the actual image job.

    ``category`` (optional) filters by job source. Surfaces use this to
    avoid showing two progress indicators for the same job:
      - in-message illustrate / scene-gen loaders pass ``user``
      - narrative auto-bg corner badge passes ``auto_bg``
    When omitted, returns the current job regardless of source.
    """
    queue = getattr(request.app.state, "image_queue", None)
    if not queue:
        return {"active": False}
    pre = _pre_queue_phase_for_session(request.app.state, session_id)
    # Apply the same category filter to the pre-queue record so a
    # distill running for a user-initiated illustrate doesn't surface
    # as the auto_bg badge (and vice versa).
    if pre and category and pre.get("category", "user") != category:
        pre = None
    user = request.scope.get("user")
    uid = user.id if user else ""
    current_id = getattr(queue, "_current_job_id", None)
    job = queue.get_job(current_id) if current_id else None
    if job and uid and job.user_id and job.user_id != uid:
        job = None
    if job and category and job.category != category:
        # The currently-running job isn't ours — but the pre-queue
        # phase might still be, so surface that alone if present.
        job = None
    if not job:
        payload = {
            "active": False,
            "queue_size": queue.queue_size,
        }
        if pre:
            payload["active"] = True
            payload["pre_queue"] = pre
            payload["stage"] = pre["stage"]
        return payload
    payload = _job_progress_payload(job, queue)
    payload["active"] = True
    payload["queue_size"] = queue.queue_size
    if pre:
        payload["pre_queue"] = pre
    return payload


@router.post("/reload-pipeline")
async def reload_pipeline(request: Request):
    """Reload the current image pipeline so load-time optimizations take effect."""
    pipeline_reg = getattr(request.app.state, "image_pipeline_registry", None)
    if not pipeline_reg:
        raise HTTPException(503, "Image subsystem not available")
    if not pipeline_reg.is_loaded:
        raise HTTPException(400, "No model currently loaded")
    try:
        await pipeline_reg.reload_current()
    except Exception as exc:
        raise HTTPException(500, f"Reload failed: {exc}") from exc
    return {"ok": True, "model": pipeline_reg.current_model}


@router.put("/active-settings")
async def put_active_settings(req: ImageActiveSettings, request: Request):
    """Store the user's current image panel settings for tool use."""
    user = request.scope.get("user")
    uid = user.id if user else ""
    data = req.model_dump(exclude_defaults=True)

    # Per-user in-memory cache (keyed by uid so tenants don't clobber).
    if not hasattr(request.app.state, "_image_active_settings_by_user"):
        request.app.state._image_active_settings_by_user = {}
    if uid:
        request.app.state._image_active_settings_by_user[uid] = data
    else:
        # Anonymous / single-user-no-auth keeps the global attr; an
        # authenticated save must NOT write the process-global mirror —
        # that's how one tenant's panel leaked into another's (the GET
        # below falls back to it). Multi-tenant fix 2026-06.
        request.app.state.image_active_settings = data

    store = getattr(request.app.state, "settings_store", None)
    if store and data:
        if uid:
            await store.set_user(uid, "image_active_settings", json.dumps(data))
        else:
            await store.set("image_active_settings", json.dumps(data))

    return {"ok": True}


@router.get("/active-settings")
async def get_active_settings(request: Request):
    """Return the user's current image panel settings.

    Single source of truth shared with every backend reader (the
    image_generation tool, narrative scene-gen) via
    ``image/active_settings.resolve_active_settings`` — so the read path can
    never drift from the per-user write path again (that drift is what made
    generation ignore the panel's selected model).
    """
    from augmentum.image.active_settings import resolve_active_settings

    user = request.scope.get("user")
    uid = user.id if user else ""
    return await resolve_active_settings(request.app.state, uid)


# --- Models ---


@router.get("/models")
async def list_models(request: Request):
    # No _require_image_info guard here — cloud models should be listed
    # even when local image generation is disabled.

    result = []

    # Local models (only if image subsystem is initialized)
    model_mgr = getattr(request.app.state, "image_model_manager", None)
    if model_mgr:
        pipeline_reg = getattr(request.app.state, "image_pipeline_registry", None)
        current_model = pipeline_reg.current_model if pipeline_reg else None

        from augmentum.image.distilled import get_recommended_defaults
        from augmentum.image.schemas import ModelCapabilities

        models = model_mgr.list_local_models()
        for m in models:
            caps_dict = m.get("capabilities", {})
            rec = get_recommended_defaults(m["name"])
            result.append(ModelInfo(
                name=m["name"],
                pipeline_type=m["pipeline_type"],
                path=m["path"],
                size_bytes=m["size_bytes"],
                source=m["source"],
                is_loaded=(current_model == m["path"]),
                distilled_type=m.get("distilled_type", ""),
                capabilities=ModelCapabilities(**caps_dict) if caps_dict else ModelCapabilities(),
                recommended_steps=rec.get("steps"),
                recommended_cfg=rec.get("cfg_scale"),
            ))

    # Cloud models from enabled image providers
    from augmentum.proxy.cloud_image_routes import _get_conn, _fetch_cloud_models

    conn = _get_conn(request)
    if conn:
        try:
            cursor = await conn.execute(
                "SELECT id, name, base_url, api_key, default_model "
                "FROM image_providers WHERE is_enabled = 1 "
                "ORDER BY is_default DESC, name"
            )
            rows = await cursor.fetchall()
            for r in rows:
                provider = {
                    "id": r[0], "name": r[1], "base_url": r[2],
                    "api_key": decrypt_api_key(r[3]), "default_model": r[4],
                }
                cloud_models = await _fetch_cloud_models(provider)
                for cm in cloud_models:
                    cloud_caps = cm.get("capabilities", {})
                    result.append(ModelInfo(
                        name=cm["name"],
                        pipeline_type=cm["pipeline_type"],
                        source="cloud",
                        is_loaded=False,
                        distilled_type="",
                        # Stash provider info in path field for UI routing
                        path=f"cloud:{provider['id']}",
                        capabilities=ModelCapabilities(**cloud_caps) if cloud_caps else ModelCapabilities(),
                    ))
        except Exception as exc:
            # image_providers table may not exist yet (pre-migration);
            # cloud models simply absent from list_models response.
            log.debug("image_list_cloud_providers_skipped", error=str(exc))

    # Phase 8 — fabric peer image models. Each connected peer's
    # heartbeat advertises ImageGenerationCapability entries; we
    # surface them in the dropdown so operators can pick a peer-
    # hosted model. Local-first: if the same model name is installed
    # locally, the peer entry is skipped (existing list already has
    # the local one, and the image dispatch hook routes to local
    # anyway). Peer entries get path=peer:<node_id> mirroring the
    # cloud convention path=cloud:<provider_id>.
    try:
        coordinator = getattr(request.app.state, "fabric_coordinator", None)
        if coordinator is not None:
            from augmentum.image.schemas import ModelCapabilities as _MC
            from augmentum.image.schemas import PipelineType as _PT

            existing_names = {m.name for m in result}
            for node_id in coordinator.connected_peer_ids():
                state = coordinator.peer_state(node_id)
                if state is None or state.paired is None:
                    continue
                peer_icon = state.paired.icon or ""
                peer_hostname = state.paired.hostname or node_id[:12]
                for cap in state.capabilities:
                    if getattr(cap, "kind", "") != "image.generation":
                        continue
                    model_id = getattr(cap, "model_id", "") or ""
                    if not model_id or model_id in existing_names:
                        continue
                    existing_names.add(model_id)
                    # Map advertised family string to PipelineType
                    # enum. Default to SD15 if unknown — the local
                    # consumer mostly uses this for filter chips and
                    # the actual dispatch goes through the peer.
                    fam = (getattr(cap, "family", "") or "").lower()
                    pt = _PT.SDXL if "xl" in fam else (
                        _PT.FLUX if "flux" in fam else _PT.SD15
                    )
                    result.append(ModelInfo(
                        name=model_id,
                        pipeline_type=pt,
                        path=f"peer:{node_id}",
                        size_bytes=0,
                        source="peer",
                        is_loaded=bool(getattr(cap, "loaded", False)),
                        distilled_type="",
                        capabilities=_MC(),
                        peer_icon=peer_icon,
                        peer_hostname=peer_hostname,
                    ))
    except Exception as exc:
        log.debug("image_peer_model_merge_failed", error=repr(exc))

    return result


@router.get("/availability")
async def image_availability(request: Request) -> dict:
    """Survey every image-generation path and report what's reachable.

    Used by the frontend image panel to decide between rendering the normal
    UI vs. a "set up image generation" empty state. Distinguishes the three
    dispatch tiers (local pipeline, cloud providers, fabric peers) so the
    panel can suggest the right next step rather than collapsing all failures
    into a generic 503. Read-only and cheap — safe to call on every panel
    open without gating.
    """
    local_pipeline_ready = bool(getattr(request.app.state, "image_queue", None))
    local_enabled_setting = bool(settings.image_enabled)

    cloud_providers = 0
    try:
        from augmentum.proxy.cloud_image_routes import _get_conn

        conn = _get_conn(request)
        if conn:
            cursor = await conn.execute(
                "SELECT COUNT(*) FROM image_providers WHERE is_enabled = 1"
            )
            row = await cursor.fetchone()
            if row:
                cloud_providers = int(row[0] or 0)
    except Exception as exc:
        log.debug("image_availability_cloud_count_failed", error=str(exc))

    fabric_peers_connected = 0
    fabric_peers_with_image = 0
    try:
        coordinator = getattr(request.app.state, "fabric_coordinator", None)
        if coordinator is not None:
            fabric_peers_connected = len(coordinator.connected_peer_ids())
            from augmentum.fabric.capabilities import KIND_IMAGE_GENERATION
            matches = coordinator.find_peers_with_capability(KIND_IMAGE_GENERATION)
            # find_peers_with_capability yields (node_id, capability) per
            # advertised cap; collapse to unique peers.
            fabric_peers_with_image = len({nid for nid, _ in matches})
    except Exception as exc:
        log.debug("image_availability_fabric_count_failed", error=str(exc))

    # Cheap CUDA probe so the frontend can decide whether to surface the
    # "switch to the GPU compose variant" suggestion. Suggesting it to a
    # Mac/AMD user (or anyone whose container can't see an NVIDIA device)
    # would be actively wrong — compose.gpu.yaml requires the NVIDIA
    # runtime. torch is a base dep so this import is always cheap; we
    # skip the nvidia-smi subprocess fallback that detect_hardware() uses
    # to keep this endpoint sub-millisecond.
    host_gpu_detected = False
    try:
        import torch
        host_gpu_detected = bool(torch.cuda.is_available())
    except Exception as exc:
        log.debug("image_availability_cuda_probe_failed", error=str(exc))

    any_path_available = (
        local_pipeline_ready
        or cloud_providers > 0
        or fabric_peers_with_image > 0
    )

    return {
        "local_pipeline_ready": local_pipeline_ready,
        "local_enabled_setting": local_enabled_setting,
        "cloud_providers": cloud_providers,
        "fabric_peers_connected": fabric_peers_connected,
        "fabric_peers_with_image": fabric_peers_with_image,
        "host_gpu_detected": host_gpu_detected,
        "any_path_available": any_path_available,
    }


@router.get("/models/detect")
async def detect_model(source: str, request: Request):
    """Detect available download variants for a HuggingFace repo or CivitAI model.

    Returns the model name, source type, and a list of downloadable variants
    with precision, size, and format info so the user can pick one before
    downloading.
    """
    source = source.strip()
    if not source:
        raise HTTPException(400, "Source is required")

    source_type = _detect_source_type(source)

    if source_type == "civitai":
        return await _detect_civitai(source, request)
    elif source_type == "huggingface":
        return await _detect_huggingface(source, request)
    else:
        raise HTTPException(400, f"Could not detect source type for: {source}")


def _detect_source_type(source: str) -> str:
    """Classify input as 'civitai', 'huggingface', or 'unknown'."""
    import re

    s = source.lower()
    # CivitAI URL
    if "civitai.com" in s:
        return "civitai"
    # Bare numeric ID → CivitAI model ID
    if re.match(r"^\d+$", source):
        return "civitai"
    # HuggingFace URL
    if "huggingface.co" in s or "hf.co" in s:
        return "huggingface"
    # owner/repo format → HuggingFace
    if re.match(r"^[a-zA-Z0-9_.-]+/[a-zA-Z0-9_.-]+$", source):
        return "huggingface"
    return "unknown"


async def _detect_civitai(source: str, request: Request) -> dict:
    """Query CivitAI API for model variants."""
    import re
    import httpx

    # Parse model ID and optional version ID from URL or bare ID
    model_id = ""
    version_id = ""

    if "civitai.com" in source:
        from urllib.parse import parse_qs, urlparse

        parsed = urlparse(source)
        qs = parse_qs(parsed.query)
        version_id = qs.get("modelVersionId", [""])[0]

        # Extract model ID from path: /models/12345 or /models/12345/name
        path_match = re.search(r"/models/(\d+)", parsed.path)
        if path_match:
            model_id = path_match.group(1)

        # Download URL: /api/download/models/12345
        dl_match = re.search(r"/api/download/models/(\d+)", parsed.path)
        if dl_match:
            version_id = dl_match.group(1)
    else:
        # Bare numeric ID
        model_id = source

    if not model_id and not version_id:
        return {"error": "Could not parse CivitAI model ID", "source_type": "civitai", "variants": []}

    api_key = settings.image_civitai_api_key
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    model_type = ""
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            if version_id and not model_id:
                # Direct version lookup
                resp = await client.get(
                    f"https://civitai.com/api/v1/model-versions/{version_id}",
                    headers=headers,
                )
                resp.raise_for_status()
                version_data = resp.json()
                model_name = version_data.get("model", {}).get("name", f"CivitAI #{version_id}")
                model_type = version_data.get("model", {}).get("type", "")
                versions = [version_data]
            else:
                # Full model lookup
                resp = await client.get(
                    f"https://civitai.com/api/v1/models/{model_id}",
                    headers=headers,
                )
                resp.raise_for_status()
                model_data = resp.json()
                model_name = model_data.get("name", f"CivitAI #{model_id}")
                model_type = model_data.get("type", "")
                versions = model_data.get("modelVersions", [])
    except httpx.HTTPStatusError as exc:
        return {"error": f"CivitAI API error: {exc.response.status_code}", "source_type": "civitai", "variants": []}
    except Exception as exc:
        return {"error": f"CivitAI API error: {exc}", "source_type": "civitai", "variants": []}

    # Build variants from all versions' files
    variants = []
    for ver in versions[:5]:  # Cap to 5 most recent versions
        ver_name = ver.get("name", "")
        ver_id = ver.get("id", "")
        base_model = ver.get("baseModel", "")
        for f in ver.get("files", []):
            meta = f.get("metadata", {})
            fp = meta.get("fp", "")
            size_type = meta.get("size", "")  # full or pruned
            fmt = meta.get("format", "")
            size_kb = f.get("sizeKB", 0)
            size_gb = round(size_kb / 1_048_576, 2) if size_kb else 0

            # Build display label
            parts = []
            if ver_name:
                parts.append(ver_name)
            if fp:
                parts.append(fp.upper())
            if size_type:
                parts.append(size_type)
            if fmt:
                parts.append(fmt)
            label = " — ".join(parts) if parts else f.get("name", "Unknown")

            # Build download URL with file-specific query params
            # CivitAI uses ?type=...&format=...&fp=...&size=... to select
            # a specific file when a version has multiple variants
            base_dl = f.get("downloadUrl", "")
            if not base_dl and ver_id:
                base_dl = f"https://civitai.com/api/download/models/{ver_id}"
            dl_params = []
            if fmt:
                dl_params.append(f"format={fmt}")
            if fp:
                dl_params.append(f"fp={fp}")
            if size_type:
                dl_params.append(f"size={size_type}")
            dl_url = base_dl
            if dl_params:
                sep = "&" if "?" in dl_url else "?"
                dl_url += sep + "&".join(dl_params)

            variants.append({
                "label": label,
                "version_id": str(ver_id),
                "file_name": f.get("name", ""),
                "fp": fp,
                "size_type": size_type,
                "format": fmt,
                "size_gb": size_gb,
                "base_model": base_model,
                "download_url": dl_url,
                "primary": f.get("primary", False),
            })

    # Extract trigger words and base model for LoRAs
    trigger_words = []
    base_model_raw = ""
    if versions:
        base_model_raw = versions[0].get("baseModel", "")
        if model_type.upper() == "LORA":
            trigger_words = versions[0].get("trainedWords", [])

    # Normalize CivitAI base model names to our PipelineType values
    _BASE_MODEL_MAP = {
        "sd 1.5": "sd15", "sd1.5": "sd15", "sd 1.4": "sd15",
        "sdxl 1.0": "sdxl", "sdxl": "sdxl", "sdxl turbo": "sdxl",
        "pony": "sdxl",  # Pony is SDXL-based
        "flux.1 d": "flux", "flux.1 s": "flux", "flux": "flux",
    }
    base_model = _BASE_MODEL_MAP.get(base_model_raw.lower(), "")

    return {
        "source_type": "civitai",
        "name": model_name,
        "model_id": model_id,
        "model_type": model_type.lower(),
        "base_model": base_model,
        "base_model_raw": base_model_raw,  # original CivitAI label for display
        "trigger_words": trigger_words,
        "variants": variants,
    }


def _normalize_lora_base_model(raw: str) -> str:
    """Map a HuggingFace base-model string to our sd15/sdxl/flux buckets."""
    s = (raw or "").lower()
    if not s:
        return ""
    if "flux" in s:
        return "flux"
    if "xl" in s:  # sdxl / stable-diffusion-xl
        return "sdxl"
    if any(k in s for k in ("3.5", "sd3", "stable-diffusion-3")):
        return "flux"  # SD3 is transformer-based → shares the flux pipeline bucket
    if any(k in s for k in ("v1-5", "v1.5", "sd_v1", "stable-diffusion-v1", "sd15")):
        return "sd15"
    if any(k in s for k in ("v2", "sd_v2", "stable-diffusion-2")):
        return "sd15"  # SD2.x is UNet-based — closest existing bucket
    return ""


async def _detect_hf_lora(api, repo_id: str, siblings, repo_info) -> dict:
    """Best-effort: is this HuggingFace repo a LoRA adapter (not a full model)?

    Returns ``{"is_lora": bool, "base_model", "base_model_raw", "trigger_words"}``.

    Signals (any one is sufficient):
      * a ``lora`` tag on the repo or its model card, or
      * a single root-level ``.safetensors`` whose header is dominated by
        ``lora_`` / ``.alpha`` tensor keys (the authoritative signal — full
        checkpoints have 0 such keys, LoRAs have ~100%).

    A diffusers ``model_index.json`` rules it OUT (that's a full pipeline).
    Without this, HF LoRAs paste into the textbox as regular models and land
    in ``image_models/`` instead of ``loras/`` with no trigger metadata.
    """
    if any((s.rfilename or "") == "model_index.json" for s in siblings):
        return {"is_lora": False}

    tags = [str(t).lower() for t in (getattr(repo_info, "tags", None) or [])]
    card = getattr(repo_info, "card_data", None)
    card_tags: list[str] = []
    base_raw = ""
    if card is not None:
        try:
            card_tags = [str(t).lower() for t in (getattr(card, "tags", None) or [])]
            base_raw = getattr(card, "base_model", "") or ""
            if isinstance(base_raw, (list, tuple)):
                base_raw = base_raw[0] if base_raw else ""
        except Exception:
            log.debug("hf_lora_card_parse_failed", repo=repo_id, exc_info=True)

    is_lora = "lora" in tags or "lora" in card_tags

    # Root-level safetensors only — weights inside subdirs imply a full
    # component pipeline, not a single-file adapter.
    root_st = [
        s.rfilename for s in siblings
        if (s.rfilename or "").endswith(".safetensors") and "/" not in (s.rfilename or "")
    ]

    ss_base = ""
    if not is_lora and len(root_st) == 1:
        # Tags were silent — inspect the safetensors header (range request,
        # no full download) for lora-shaped tensor keys.
        try:
            meta = await asyncio.to_thread(
                api.parse_safetensors_file_metadata, repo_id, root_st[0],
            )
            keys = list(meta.tensors.keys())
            if keys:
                lora_keys = sum(1 for k in keys if ("lora" in k.lower() or k.endswith(".alpha")))
                if lora_keys / len(keys) >= 0.5:
                    is_lora = True
            ss_base = (meta.metadata or {}).get("ss_base_model_version", "") or ""
        except Exception:
            log.debug("hf_lora_safetensors_probe_failed", repo=repo_id, exc_info=True)

    if not is_lora:
        return {"is_lora": False}

    # Resolve base model: card field → repo tag → safetensors ss_ metadata.
    if not base_raw:
        for t in tags:
            if t.startswith("base_model:") and "adapter" not in t:
                base_raw = t.split("base_model:", 1)[1]
                break
    base_norm = _normalize_lora_base_model(base_raw) or _normalize_lora_base_model(ss_base)
    return {
        "is_lora": True,
        "base_model": base_norm,
        "base_model_raw": base_raw or ss_base,
        "trigger_words": [],
    }


async def _detect_huggingface(source: str, request: Request) -> dict:
    """Query HuggingFace API for model variants."""
    import re
    from collections import defaultdict

    # Extract repo_id from URL or use directly
    repo_id = source
    if "huggingface.co" in source or "hf.co" in source:
        match = re.search(r"(?:huggingface\.co|hf\.co)/([^/]+/[^/]+)", source)
        if match:
            repo_id = match.group(1)

    model_mgr = getattr(request.app.state, "image_model_manager", None)
    if not model_mgr:
        from augmentum.image.model_manager import ModelManager

        model_dir = settings.image_model_dir or f"{settings.data_dir}/image_models"
        model_mgr = ModelManager(model_dir)

    try:
        from huggingface_hub import HfApi

        resolved_token = model_mgr._resolve_hf_token(settings.image_huggingface_token)
        api = HfApi(token=resolved_token)
        # files_metadata=True is required to populate per-file LFS sizes —
        # without it every sibling's `size` is None, so the variant sizes all
        # compute to 0 and the UI shows "Auto — 0 KB". The download path
        # (pull_from_huggingface) already passes this; detect must match.
        repo_info = await asyncio.to_thread(
            lambda: api.repo_info(repo_id, files_metadata=True),
        )
        siblings = repo_info.siblings or []
    except Exception as exc:
        return {"error": f"HuggingFace API error: {exc}", "source_type": "huggingface", "variants": []}

    # Detect available variants by scanning safetensors filenames
    variant_sizes: dict[str, int] = defaultdict(int)
    all_size = 0
    for s in siblings:
        name = s.rfilename or ""
        size = s.size or 0
        all_size += size
        # Check for variant tags in safetensors files
        if name.endswith(".safetensors"):
            for tag in ("fp16", "bf16", "fp8"):
                if f".{tag}." in name or f"/{tag}/" in name or name.endswith(f".{tag}.safetensors"):
                    variant_sizes[tag] += size
            # Check for GGUF
        if name.endswith(".gguf"):
            variant_sizes["gguf"] += size

    # Get filtered sizes per variant
    variants = []

    # Auto variant (our smart filtering)
    filtered = model_mgr._filter_inference_files(siblings)
    auto_size = sum(s.size or 0 for s in filtered)
    auto_variant = model_mgr._select_variant(siblings) or "auto"
    variants.append({
        "label": f"Auto ({auto_variant}) — {_fmt_gb(auto_size)}",
        "variant": "",
        "size_gb": round(auto_size / 1_073_741_824, 2),
        "allow_patterns": None,
    })

    # Explicit variants
    for tag in ("fp16", "bf16", "fp32"):
        filtered_v = model_mgr._filter_inference_files(siblings, variant_override=tag)
        v_size = sum(s.size or 0 for s in filtered_v)
        if v_size > 0 and v_size != auto_size:
            variants.append({
                "label": f"{tag.upper()} — {_fmt_gb(v_size)}",
                "variant": tag,
                "size_gb": round(v_size / 1_073_741_824, 2),
                "allow_patterns": None,
            })

    # Full precision (no variant filtering)
    if auto_variant != "auto":
        full_filtered = model_mgr._filter_inference_files(siblings, variant_override="fp32")
        full_size = sum(s.size or 0 for s in full_filtered)
        if full_size > auto_size:
            variants.append({
                "label": f"Full precision — {_fmt_gb(full_size)}",
                "variant": "fp32",
                "size_gb": round(full_size / 1_073_741_824, 2),
                "allow_patterns": None,
            })

    # GGUF variants
    gguf_files = [s for s in siblings if (s.rfilename or "").endswith(".gguf")]
    for gf in gguf_files:
        variants.append({
            "label": f"GGUF: {gf.rfilename} — {_fmt_gb(gf.size or 0)}",
            "variant": "gguf",
            "size_gb": round((gf.size or 0) / 1_073_741_824, 2),
            "allow_patterns": [gf.rfilename],
        })

    result = {
        "source_type": "huggingface",
        "name": repo_id,
        "repo_id": repo_id,
        "variants": variants,
        "model_type": "model",
    }

    # LoRA adapters need model_type=lora so the frontend sets asset_type=lora
    # and the pull routes the file into loras/ with trigger/base metadata.
    lora = await _detect_hf_lora(api, repo_id, siblings, repo_info)
    if lora.get("is_lora"):
        result["model_type"] = "lora"
        result["base_model"] = lora.get("base_model", "")
        result["base_model_raw"] = lora.get("base_model_raw", "")
        result["trigger_words"] = lora.get("trigger_words", [])

    return result


def _fmt_gb(size_bytes: int) -> str:
    """Format bytes as human-readable GB/MB string."""
    if size_bytes >= 1_073_741_824:
        return f"{size_bytes / 1_073_741_824:.1f} GB"
    if size_bytes >= 1_048_576:
        return f"{size_bytes / 1_048_576:.0f} MB"
    return f"{size_bytes / 1024:.0f} KB"


# --- Background model download tasks ---

import uuid as _uuid
from typing import Any

# In-memory task tracker: task_id -> {status, progress, error, result, ...}
_pull_tasks: dict[str, dict[str, Any]] = {}


def _get_pull_deps(request: Request) -> tuple:
    """Return (model_mgr, persistence) from app state."""
    from augmentum.image.model_manager import ModelManager

    model_mgr = getattr(request.app.state, "image_model_manager", None)
    if not model_mgr:
        model_dir = settings.image_model_dir or f"{settings.data_dir}/image_models"
        model_mgr = ModelManager(model_dir)
    persistence = getattr(request.app.state, "image_persistence", None)
    return model_mgr, persistence


def _write_gguf_meta(
    model_path: str,
    base_repo: str,
    pipeline_class: str,
    transformer_class: str,
) -> bool:
    """Write the three GGUF loader fields into ``gguf_meta.json`` next to the weights.

    Returns True on success. No-op when any field is missing.
    """
    import json

    if not model_path or not os.path.isdir(model_path):
        return False
    if not (base_repo and pipeline_class and transformer_class):
        return False
    meta = {
        "gguf_base_repo": base_repo,
        "gguf_pipeline_class": pipeline_class,
        "gguf_transformer_class": transformer_class,
    }
    meta_path = os.path.join(model_path, "gguf_meta.json")
    try:
        with open(meta_path, "w") as f:
            json.dump(meta, f)
        log.info("gguf_meta_saved", path=meta_path)
        return True
    except Exception:
        log.debug("gguf_meta_save_failed", path=meta_path, exc_info=True)
        return False


_GENERIC_LORA_STEMS = {
    "pytorch_lora_weights", "adapter_model", "lora", "model",
    "diffusion_pytorch_model",
}


def _lora_output_filename(model_name: str, src_filename: str) -> str:
    """Pick the on-disk LoRA filename.

    Keep a descriptive source filename (e.g. CivitAI's ``more_details
    .safetensors``), but when it's generic — HF LoRAs almost always ship as
    ``pytorch_lora_weights.safetensors`` — fall back to the repo/model name
    so two different HF LoRAs don't both land as ``pytorch_lora_weights`` and
    clobber each other in ``loras/``.
    """
    stem, ext = os.path.splitext(src_filename)
    if stem.lower() in _GENERIC_LORA_STEMS:
        safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in (model_name or "lora"))
        return (safe or "lora") + (ext or ".safetensors")
    return src_filename


def _save_gguf_meta_if_needed(source: str, model_path: str) -> None:
    """If *source* matches a GGUF catalog model, write a ``gguf_meta.json``
    into the model directory so the pipeline loader knows how to load it.
    """
    from augmentum.image.hardware import RECOMMENDED_MODELS

    if not model_path:
        return
    for cm in RECOMMENDED_MODELS:
        if cm.repo_id == source and cm.gguf_base_repo:
            _write_gguf_meta(
                model_path,
                cm.gguf_base_repo,
                cm.gguf_pipeline_class,
                cm.gguf_transformer_class,
            )
            break


async def _prefetch_gguf_base_components_for(
    base_repo: str,
    task: dict,
    token: str | None = None,
) -> dict:
    """Core base-component prefetcher — works for any explicit base repo.

    Pre-fetches the diffusers pipeline scaffolding (text encoder, VAE,
    tokenizer, scheduler) so first inference doesn't need network access.
    Skips the transformer/ subdir since the GGUF replaces those weights.

    Returns a status dict so callers can mark the install as failed when the
    base components are unreachable (gated repo without a token, repo info
    404, partial file download). Status values:

    * ``ok``      — all required files fetched (or there was nothing to do).
    * ``skipped`` — no base_repo provided / huggingface_hub missing.
    * ``error``   — see ``stage`` and ``error`` fields.
    """
    if not base_repo:
        return {"status": "skipped", "reason": "no_base_repo"}

    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        log.warning("gguf_prefetch_skip", reason="huggingface_hub not installed")
        return {"status": "skipped", "reason": "no_hf_hub"}

    resolved_token = None
    if token:
        resolved_token = token
    else:
        import os as _os
        resolved_token = _os.environ.get("HUGGING_FACE_HUB_TOKEN") or _os.environ.get("HF_TOKEN")

    log.info(
        "gguf_prefetch_starting",
        base_repo=base_repo,
        hint="Pre-fetching text encoder, VAE, tokenizer, and scheduler from base repo",
    )

    # Update task status so the UI shows what's happening
    progress = task.get("progress", {})
    progress["phase"] = "Downloading base model components (text encoder, VAE, tokenizer)..."
    progress["percent"] = 95  # GGUF file is done, components are the last 5%
    task["progress"] = progress

    api = HfApi(token=resolved_token)
    try:
        repo_info = await asyncio.to_thread(api.repo_info, base_repo)
        siblings = repo_info.siblings or []
    except Exception as exc:
        log.warning("gguf_prefetch_repo_info_failed", base_repo=base_repo, error=str(exc))
        return {
            "status": "error",
            "stage": "repo_info",
            "base_repo": base_repo,
            "error": str(exc),
            "has_token": bool(resolved_token),
        }

    # Filter to only non-weight pipeline components: configs, tokenizer,
    # scheduler, and small model files.  Skip the large transformer weights
    # (we already have those as GGUF) and skip training artifacts.
    # The text encoder weights ARE needed — they're not in the GGUF.
    # NOTE: .txt is intentionally NOT skipped — BPE tokenizers (CLIP for
    # SD/SDXL, Qwen3 for Z-Image/Z-Anime) need merges.txt and vocab.txt.
    # Without them, the tokenizer falls back to network fetch at inference
    # time and breaks offline use.
    skip_prefixes = {"transformer/"}
    skip_extensions = {
        ".ckpt", ".bin", ".pt", ".onnx", ".pb", ".tflite",
        ".png", ".jpg", ".jpeg", ".gif", ".md",
    }

    files_to_fetch = []
    for s in siblings:
        fname = s.rfilename
        # Skip transformer weights (already have GGUF version)
        if any(fname.startswith(p) for p in skip_prefixes):
            continue
        # Skip non-inference files
        ext = "." + fname.rsplit(".", 1)[-1].lower() if "." in fname else ""
        if ext in skip_extensions:
            continue
        files_to_fetch.append(fname)

    fetched = 0
    failed = 0
    first_error = ""
    for fname in files_to_fetch:
        try:
            def _dl(filename=fname):
                return hf_hub_download(
                    base_repo,
                    filename=filename,
                    token=resolved_token,
                )
            await asyncio.to_thread(_dl)
            fetched += 1
        except Exception as exc:
            failed += 1
            if not first_error:
                first_error = str(exc)
            log.debug("gguf_prefetch_file_failed", file=fname, error=str(exc))

    log.info(
        "gguf_prefetch_complete",
        base_repo=base_repo,
        fetched=fetched,
        failed=failed,
        total=len(files_to_fetch),
    )
    if failed > 0:
        log.warning(
            "gguf_prefetch_partial",
            base_repo=base_repo,
            failed=failed,
            first_error=first_error,
        )
        return {
            "status": "error",
            "stage": "file_download",
            "base_repo": base_repo,
            "fetched": fetched,
            "failed": failed,
            "total": len(files_to_fetch),
            "error": first_error,
            "has_token": bool(resolved_token),
        }

    return {"status": "ok", "base_repo": base_repo, "fetched": fetched}


async def _prefetch_gguf_base_components(
    source: str,
    task: dict,
    token: str | None = None,
) -> dict:
    """Catalog-aware wrapper: resolve *source* → base_repo via RECOMMENDED_MODELS,
    then delegate to :func:`_prefetch_gguf_base_components_for`.

    Used by the URL-pull flow; the custom-import flow calls the core directly
    with a known base_repo. Returns the status dict from the core so callers
    can react to a failed prefetch.
    """
    from augmentum.image.hardware import RECOMMENDED_MODELS

    base_repo = ""
    for cm in RECOMMENDED_MODELS:
        if cm.repo_id == source and cm.gguf_base_repo:
            base_repo = cm.gguf_base_repo
            break

    return await _prefetch_gguf_base_components_for(base_repo, task, token=token)


def _mark_install_failed_from_prefetch(task: dict, prefetch_result: dict) -> None:
    """Translate a prefetch error dict into a user-facing install failure.

    Distinguishes gated/auth issues from generic network errors so the user
    knows whether to add an HF token or just retry.
    """
    base_repo = prefetch_result.get("base_repo", "")
    has_token = prefetch_result.get("has_token", False)
    raw_err = str(prefetch_result.get("error", ""))
    err_lower = raw_err.lower()
    is_auth = any(
        s in err_lower for s in ("401", "403", "gated", "unauthorized", "forbidden")
    )

    if is_auth and not has_token:
        msg = (
            f"Base model components from {base_repo} couldn't be downloaded — "
            f"this HuggingFace repo is gated. Add an HF token in "
            f"Settings → Image, accept the license at "
            f"https://huggingface.co/{base_repo}, then re-install."
        )
    elif is_auth:
        msg = (
            f"Base model components from {base_repo} couldn't be downloaded — "
            f"your HuggingFace token doesn't have access. Accept the license at "
            f"https://huggingface.co/{base_repo} and re-install."
        )
    else:
        msg = (
            f"Base model components from {base_repo} couldn't be downloaded: "
            f"{raw_err or 'download failed'}"
        )
    task["status"] = "error"
    task["error"] = msg


async def _disk_size_monitor(task: dict, dest_path: str, total_size: int) -> None:
    """Poll the download directory size and update task progress in real-time.

    Runs alongside the actual download so we get continuous progress even
    while a single large file is being written.  Stops when the task leaves
    the "running" state.

    huggingface_hub downloads to ``<dest>/.cache/huggingface/`` first using
    ``.incomplete`` temp files, then moves/copies to the final path.  We
    scan the entire tree (including ``.cache``) but cap the reported size
    to ``total_size`` so duplicates from the copy phase don't push past
    100%.  We also track the *peak* bytes seen so the progress bar never
    goes backwards during the rename/move operations.

    Xet-backed transfers stage block data outside ``dest_path`` (HF's
    shared Xet cache lives at ``$HF_XET_CACHE`` or ``$HF_HOME/xet/``), so
    a pure ``os.walk(dest_path)`` reads near-zero throughout the pull and
    the progress bar stays at 0%. We snapshot the Xet cache size at
    monitor start and add the delta to the local count, so progress
    reflects whichever path hf actually used. Safe because the job runner
    is single-worker — no other download can pollute the delta while this
    one is running.
    """
    import os

    def _dir_size_sync(path: str) -> int:
        total = 0
        if not path or not os.path.isdir(path):
            return total
        try:
            for dirpath, _dirs, files in os.walk(path):
                for f in files:
                    try:
                        total += os.path.getsize(os.path.join(dirpath, f))
                    except OSError:
                        pass
        except OSError:
            pass
        return total

    xet_cache = os.environ.get("HF_XET_CACHE") or os.path.join(
        os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"),
        "xet",
    )
    xet_baseline = await asyncio.to_thread(_dir_size_sync, xet_cache)
    if xet_baseline:
        log.debug("disk_monitor_xet_baseline", path=xet_cache, baseline_bytes=xet_baseline)

    peak_size = 0
    poll_count = 0
    pct = 0.0

    while task["status"] == "running":
        try:
            current_size = await asyncio.to_thread(_dir_size_sync, dest_path)
            xet_current = await asyncio.to_thread(_dir_size_sync, xet_cache)
            xet_delta = max(0, xet_current - xet_baseline)
            observed = current_size + xet_delta

            # During the copy phase, bytes on disk can temporarily exceed
            # total_size (file exists in both .cache and final location).
            # Cap to total_size and use peak so progress never goes backwards.
            effective = min(observed, total_size)
            peak_size = max(peak_size, effective)

            if total_size > 0:
                pct = round(peak_size / total_size * 100, 1)
                pct = min(pct, 99.9)  # Reserve 100% for the "complete" event
                progress = task.get("progress", {})
                progress["percent"] = pct
                progress["downloaded"] = peak_size
                progress["total"] = total_size
                task["progress"] = progress

            poll_count += 1
            if poll_count % 10 == 1:  # Log every ~15s
                log.debug(
                    "disk_monitor_tick",
                    dest=dest_path,
                    local_bytes=current_size,
                    xet_delta_bytes=xet_delta,
                    peak_bytes=peak_size,
                    total_bytes=total_size,
                    pct=pct if total_size > 0 else 0,
                )
        except Exception:
            log.debug("disk_monitor_tick_failed", dest=dest_path, exc_info=True)
        await asyncio.sleep(1.5)


async def _emit_ctx_progress(ctx, task: dict) -> None:
    """Mirror the rich in-memory progress onto a JobContext so restart-survival
    DB rows stay current. No-op if ctx is None or progress dict is empty."""
    if ctx is None:
        return
    prog = task.get("progress", {})
    if not prog:
        return
    pct = prog.get("percent", 0)
    try:
        progress_f = max(0.0, min(1.0, float(pct) / 100.0))
    except (TypeError, ValueError):
        progress_f = 0.0
    stage = prog.get("phase") or task.get("last_event", {}).get("status", "running")
    try:
        await ctx.update_progress(progress_f, stage=stage)
    except Exception:
        # Persistence is best-effort; don't break the pull on a DB hiccup.
        log.debug("ctx_progress_update_failed", task_id=task.get("task_id", ""), exc_info=True)


async def _run_pull_task(
    task_id: str,
    model_mgr,
    persistence,
    source: str,
    name: str,
    allow_patterns: list[str] | None,
    variant: str = "",
    asset_type: str = "",
    trigger_words: list[str] | None = None,
    base_model: str = "",
    ctx=None,
) -> None:
    """Background coroutine that drives the download and updates task state.

    When *ctx* is a :class:`JobContext` (i.e. invoked via the jobs queue),
    progress is also mirrored to the persistent jobs DB so the pull survives
    a server restart — the runner auto-requeues it, HF's cache resumes the
    bytes, and the UI reconnects via the stable task_id.
    """
    task = _pull_tasks[task_id]
    is_civitai = "civitai.com" in source
    is_lora = asset_type == "lora"
    monitor_task: asyncio.Task | None = None
    try:
        if is_civitai:
            gen = model_mgr.pull_from_civitai(
                source, name=name,
                api_key=settings.image_civitai_api_key,
            )
        else:
            gen = model_mgr.pull_from_huggingface(
                source, name=name,
                token=settings.image_huggingface_token,
                allow_patterns=allow_patterns,
                variant=variant,
            )

        async for event in gen:
            task["last_event"] = event
            if event.get("status") == "progress":
                task["progress"] = event
                await _emit_ctx_progress(ctx, task)
            elif event.get("status") == "downloading":
                # Ensure percent/file counts are present from the start
                event.setdefault("percent", 0)
                event.setdefault("files_done", 0)
                event.setdefault("files_total", event.get("file_count", 0))
                task["progress"] = event
                # Start disk-size monitor for real-time progress — unless the
                # generator already streams real intra-file byte progress
                # (native_progress), in which case the disk monitor is a
                # redundant fallback that can't see Xet's burst staging and
                # would only fight the accurate events.
                dest_path = event.get("_dest_path", "")
                total_size = event.get("total_size", 0)
                if (
                    dest_path and total_size > 0 and not monitor_task
                    and not event.get("native_progress")
                ):
                    monitor_task = asyncio.create_task(
                        _disk_size_monitor(task, dest_path, total_size)
                    )
                    monitor_task.add_done_callback(_monitor_task_done_callback)
                await _emit_ctx_progress(ctx, task)
            elif event.get("status") == "complete":
                task["status"] = "complete"
                task["result"] = event
                # Set final 100% progress
                progress = task.get("progress", {})
                progress["percent"] = 100
                task["progress"] = progress

                if is_lora:
                    # Move LoRA file to the loras/ subdirectory
                    dl_path = event.get("path", "")
                    if dl_path and os.path.exists(dl_path):
                        import shutil
                        lora_dir = os.path.join(model_mgr.model_dir, "loras")
                        os.makedirs(lora_dir, exist_ok=True)
                        model_name = event.get("name") or os.path.basename(dl_path)
                        lora_name = model_name
                        # If downloaded as a directory, move the weight file out
                        # (HF pulls land in a dir alongside a README/config).
                        if os.path.isdir(dl_path):
                            for fn in os.listdir(dl_path):
                                if fn.endswith((".safetensors", ".pt", ".bin")):
                                    src = os.path.join(dl_path, fn)
                                    out_name = _lora_output_filename(model_name, fn)
                                    dst = os.path.join(lora_dir, out_name)
                                    shutil.move(src, dst)
                                    event["path"] = dst
                                    lora_name = os.path.splitext(out_name)[0]
                                    break
                            # Remove the leftover download dir (README, config,
                            # .cache) — rmdir only worked when it was empty.
                            shutil.rmtree(dl_path, ignore_errors=True)
                        else:
                            out_name = _lora_output_filename(
                                model_name, os.path.basename(dl_path))
                            dst = os.path.join(lora_dir, out_name)
                            shutil.move(dl_path, dst)
                            event["path"] = dst
                            lora_name = os.path.splitext(out_name)[0]

                        # Save metadata as companion JSON
                        if trigger_words or base_model:
                            import json as _json
                            meta_path = os.path.join(lora_dir, lora_name + ".json")
                            meta = {}
                            if trigger_words:
                                meta["trigger_words"] = trigger_words
                            if base_model:
                                meta["base_model"] = base_model
                            with open(meta_path, "w", encoding="utf-8") as mf:
                                _json.dump(meta, mf)

                        log.info("lora_downloaded", name=lora_name, path=event["path"],
                                 trigger_words=trigger_words or [])
                elif persistence:
                    await persistence.save_model(
                        name=event["name"],
                        pipeline_type=event.get("pipeline_type", "sd15"),
                        path=event["path"],
                        source="civitai" if is_civitai else "huggingface",
                        size_bytes=event.get("size_bytes", 0),
                    )
                # Save GGUF metadata if this is a catalog GGUF model
                if not is_lora:
                    _save_gguf_meta_if_needed(source, event.get("path", ""))
                    # Pre-fetch base repo components (text encoder, VAE, etc.)
                    # so first inference doesn't need network access
                    prefetch_result = await _prefetch_gguf_base_components(
                        source, task,
                        token=settings.image_huggingface_token,
                    )
                    if prefetch_result.get("status") == "error":
                        _mark_install_failed_from_prefetch(task, prefetch_result)
            elif event.get("status") == "exists":
                task["status"] = "exists"
                task["result"] = event
                # The GGUF was already on disk, but the base components may
                # have failed to prefetch on a prior install (gated repo +
                # no HF token, network blip). Re-attempt so a user who has
                # since configured a token can heal the install just by
                # clicking install again.
                if not is_lora:
                    _save_gguf_meta_if_needed(source, event.get("path", ""))
                    prefetch_result = await _prefetch_gguf_base_components(
                        source, task,
                        token=settings.image_huggingface_token,
                    )
                    if prefetch_result.get("status") == "error":
                        _mark_install_failed_from_prefetch(task, prefetch_result)
            elif event.get("status") == "error":
                task["status"] = "error"
                task["error"] = event.get("error", "Unknown error")

        # If generator finished without explicit status, mark complete
        if task["status"] == "running":
            task["status"] = "complete"
    except Exception as exc:
        task["status"] = "error"
        task["error"] = str(exc)
        log.error("pull_task_failed", task_id=task_id, error=str(exc))
    finally:
        # Stop the disk monitor
        if monitor_task and not monitor_task.done():
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
        # Local image-model list changed once the pull completes. Broadcast
        # (image models live on shared server disk) so an open image-model
        # dropdown refreshes without a page reload. Covers both completion
        # paths above (explicit complete event + generator-finished).
        if task.get("status") == "complete":
            system_events.publish("image.models.changed", {"reason": "pull"})


@router.post("/models/pull")
async def pull_model(req: ModelPullRequest, request: Request):
    """Start a background model download. Returns a task_id for polling."""
    model_mgr, persistence = _get_pull_deps(request)
    source = req.source.strip()

    # Check if the model is in our catalog and if the system can run it
    from augmentum.image.hardware import (
        RECOMMENDED_MODELS,
        get_system_ram_free_mb,
    )

    for cat_model in RECOMMENDED_MODELS:
        if cat_model.repo_id == source:
            # Check system RAM — need enough to hold the download + load
            sys_ram_free = get_system_ram_free_mb()
            required_ram_mb = int(cat_model.size_gb * 1500)  # 1.5x size for overhead
            if sys_ram_free and sys_ram_free < required_ram_mb:
                raise HTTPException(
                    422,
                    f"{cat_model.name} is ~{cat_model.size_gb:.1f}GB but only "
                    f"{sys_ram_free / 1000:.1f}GB system RAM free. "
                    f"This would likely crash the system. "
                    f"Free up RAM or choose a smaller model.",
                )
            break

    # Persist via the jobs queue so the pull survives a server restart —
    # task_id == job_id, so the existing GET /pull/{task_id} polling URL
    # stays stable. Falls back to the legacy fire-and-forget path when the
    # jobs store isn't available (early boot, no user, etc.).
    user = request.scope.get("user")
    user_id = getattr(user, "id", "") if user else ""
    jobs_store = getattr(request.app.state, "jobs_store", None)
    job_runner = getattr(request.app.state, "job_runner", None)

    if jobs_store and user_id:
        task_id = await jobs_store.create(
            user_id=user_id,
            job_type="image_pull",
            payload={
                "source": source,
                "name": req.name,
                "allow_patterns": req.allow_patterns,
                "variant": req.variant,
                "asset_type": req.asset_type,
                "trigger_words": req.trigger_words,
                "base_model": req.base_model,
            },
        )
        # Pre-populate the in-memory poll cache so the UI sees a running
        # task immediately rather than after the runner's next tick.
        _pull_tasks[task_id] = {
            "status": "running",
            "source": source,
            "progress": {},
            "last_event": {},
            "result": None,
            "error": None,
        }
        if job_runner is not None:
            job_runner.wake()
        return {"task_id": task_id, "status": "running"}

    # Legacy fire-and-forget path — no restart survival
    task_id = _uuid.uuid4().hex[:16]
    _pull_tasks[task_id] = {
        "status": "running",
        "source": source,
        "progress": {},
        "last_event": {},
        "result": None,
        "error": None,
    }

    task = asyncio.create_task(
        _run_pull_task(
            task_id, model_mgr, persistence, source, req.name,
            req.allow_patterns, req.variant,
            asset_type=req.asset_type, trigger_words=req.trigger_words,
            base_model=req.base_model,
        )
    )
    task.add_done_callback(_pull_task_done_callback)

    return {"task_id": task_id, "status": "running"}


_JOB_STATUS_TO_PULL: dict[str, str] = {
    "pending": "running",
    "running": "running",
    "completed": "complete",
    "failed": "error",
    "cancelled": "error",
}


def _job_to_pull_status(task_id: str, job: dict) -> dict:
    """Translate a ``background_jobs`` row into the legacy pull-status shape.

    Used when the in-memory ``_pull_tasks`` cache is missing an entry but
    the jobs DB has one (typical case: server was restarted while the pull
    was running and the UI is reconnecting).
    """
    payload = job.get("payload") or {}
    raw_status = job.get("status", "")
    out: dict[str, Any] = {
        "task_id": task_id,
        "status": _JOB_STATUS_TO_PULL.get(raw_status, "running"),
        "source": payload.get("source", ""),
        "percent": int(round(float(job.get("progress", 0.0)) * 100)),
        "files_done": 0,
        "files_total": 0,
        "file_count": 0,
        "downloaded": 0,
        "total": 0,
    }
    stage = job.get("stage", "")
    if stage:
        out["phase"] = stage
    if raw_status == "failed":
        out["error"] = job.get("error") or "Pull failed"
    if raw_status == "completed":
        result = job.get("result") or {}
        if isinstance(result, dict):
            if result.get("name"):
                out["name"] = result["name"]
            if result.get("pipeline_type"):
                out["pipeline_type"] = result["pipeline_type"]
    return out


@router.get("/models/pull/{task_id}")
async def pull_model_status(task_id: str, request: Request):
    """Poll the status of a background model download."""
    task = _pull_tasks.get(task_id)
    if not task:
        # Fall back to the jobs DB — a server restart drains _pull_tasks,
        # but the persistent job survives and is being re-dispatched.
        jobs_store = getattr(request.app.state, "jobs_store", None)
        if jobs_store:
            _u = request.scope.get("user")
            _uid = _u.id if _u else ""
            job = await jobs_store.get(task_id, user_id=_uid)
            if job and job.get("job_type") == "image_pull":
                return _job_to_pull_status(task_id, job)
        raise HTTPException(404, "Download task not found")

    import time

    resp: dict[str, Any] = {
        "task_id": task_id,
        "status": task["status"],
        "source": task.get("source", ""),
    }

    progress = task.get("progress", {})
    if progress:
        resp["percent"] = progress.get("percent", 0)
        resp["files_done"] = progress.get("files_done", 0)
        resp["files_total"] = progress.get("files_total", 0)
        resp["file_count"] = progress.get("file_count", 0)
        # Byte-level progress (written by _disk_size_monitor + the HF generator);
        # the UI uses these to render "<downloaded> / <total>" alongside the percent.
        resp["downloaded"] = progress.get("downloaded", 0)
        resp["total"] = progress.get("total", 0)

    if task["status"] == "complete" and task.get("result"):
        resp["name"] = task["result"].get("name", "")
        resp["pipeline_type"] = task["result"].get("pipeline_type", "")
    elif task["status"] == "exists" and task.get("result"):
        resp["name"] = task["result"].get("name", "")
    elif task["status"] == "error":
        resp["error"] = task["error"]

    # Clean up terminal tasks after a grace period (60s) so UI can
    # reliably poll the final status even with retries / page reloads.
    if task["status"] in ("complete", "exists", "error"):
        finished_at = task.get("_finished_at")
        if not finished_at:
            task["_finished_at"] = time.time()
        elif time.time() - finished_at > 60:
            _pull_tasks.pop(task_id, None)

    return resp


@router.get("/models/pull")
async def list_pull_tasks(request: Request):
    """List all active download tasks (for reconnecting after page reload).

    Includes both in-memory tasks (the live progress source) and persistent
    image_pull jobs from the queue (so a fresh page-load after a server
    restart still surfaces the in-flight downloads).
    """
    result = []
    seen: set[str] = set()
    for tid, task in _pull_tasks.items():
        entry: dict[str, Any] = {"task_id": tid, "status": task["status"], "source": task["source"]}
        progress = task.get("progress", {})
        if progress:
            entry["percent"] = progress.get("percent", 0)
            entry["files_done"] = progress.get("files_done", 0)
            entry["files_total"] = progress.get("files_total", 0)
            entry["downloaded"] = progress.get("downloaded", 0)
            entry["total"] = progress.get("total", 0)
        result.append(entry)
        seen.add(tid)

    # Fall back to the persistent queue for jobs not yet (re-)attached to
    # the in-memory cache — typical right after a restart, before the
    # runner has re-dispatched the requeued job.
    user = request.scope.get("user")
    user_id = getattr(user, "id", "") if user else ""
    jobs_store = getattr(request.app.state, "jobs_store", None)
    if jobs_store and user_id:
        try:
            # Both pending and running count as "in flight" from the UI's pov.
            running_jobs = await jobs_store.list_for_user(
                user_id=user_id, status="running", job_type="image_pull", limit=50,
            )
            pending_jobs = await jobs_store.list_for_user(
                user_id=user_id, status="pending", job_type="image_pull", limit=50,
            )
        except Exception:
            running_jobs, pending_jobs = [], []
            log.debug("list_pull_tasks_jobs_lookup_failed", exc_info=True)
        for job in (*running_jobs, *pending_jobs):
            tid = job.get("id", "")
            if not tid or tid in seen:
                continue
            result.append(_job_to_pull_status(tid, job))
            seen.add(tid)

    return result


@router.post("/models/upload")
async def upload_model(file: UploadFile, request: Request, name: str = ""):
    """Upload a .safetensors file via drag-and-drop or file picker.

    Admin-only — installing model weights is shared-infrastructure mutation,
    and the file lands on disk. The model name and uploaded filename are
    both sanitized against path traversal and verified to live inside the
    configured model_dir via realpath checks.
    """
    import re
    from pathlib import Path

    from augmentum.image.model_manager import ModelManager, _detect_pipeline_type, _get_dir_size

    if (forbidden := require_admin(request)) is not None:
        return forbidden

    raw_filename = file.filename or "model.safetensors"
    # Strip any directory components from the client-supplied filename
    # before any further processing — defense-in-depth against the
    # sanitization regex below missing an edge case.
    filename = os.path.basename(raw_filename)
    if not filename.endswith(".safetensors"):
        raise HTTPException(400, "Only .safetensors files are supported")
    # After basename + extension check, only allow a conservative charset.
    if not re.fullmatch(r"[A-Za-z0-9._-]+", filename):
        raise HTTPException(400, "Invalid filename")

    model_mgr = getattr(request.app.state, "image_model_manager", None)
    if not model_mgr:
        model_dir = settings.image_model_dir or f"{settings.data_dir}/image_models"
        model_mgr = ModelManager(model_dir)

    raw_name = name.strip() or Path(filename).stem
    # Conservative slug: ASCII letters/digits/._- only, no path separators.
    model_name = re.sub(r"[^A-Za-z0-9._-]", "_", raw_name)
    if not model_name or model_name in (".", ".."):
        raise HTTPException(400, "Invalid model name")

    # Resolve the destination through realpath and confirm it stays inside
    # model_dir. Defends against symlinked model_dir entries that could
    # otherwise let the joined path escape.
    model_root = os.path.realpath(model_mgr.model_dir)
    dest_dir = os.path.realpath(os.path.join(model_root, model_name))
    if dest_dir != model_root and not dest_dir.startswith(model_root.rstrip(os.sep) + os.sep):
        raise HTTPException(400, "Resolved path escapes model directory")
    Path(dest_dir).mkdir(parents=True, exist_ok=True)
    dest_file = os.path.join(dest_dir, filename)

    # Stream file to disk in chunks to avoid loading into memory
    total_written = 0
    with open(dest_file, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)  # 1MB chunks
            if not chunk:
                break
            f.write(chunk)
            total_written += len(chunk)

    pipeline_type = _detect_pipeline_type(dest_dir)
    size_bytes = _get_dir_size(dest_dir)

    # Persist metadata if available
    persistence = getattr(request.app.state, "image_persistence", None)
    if persistence:
        await persistence.save_model(
            name=model_name,
            pipeline_type=pipeline_type.value,
            path=dest_dir,
            source="upload",
            size_bytes=size_bytes,
        )

    system_events.publish("image.models.changed", {"reason": "upload"})
    log.info("model_uploaded", name=model_name, filename=filename, size_bytes=total_written)
    return {
        "status": "complete",
        "name": model_name,
        "path": dest_dir,
        "pipeline_type": pipeline_type.value,
        "size_bytes": size_bytes,
    }


def _resolve_gguf_family_preset(
    family_key: str,
    base_repo: str,
    pipeline_class: str,
    transformer_class: str,
) -> tuple[str, str, str]:
    """If *family_key* is set, look it up in get_gguf_families() and override
    the three fields. Returns the effective (base_repo, pipeline_class,
    transformer_class). Raises HTTPException on unknown family.
    """
    if not family_key:
        return base_repo, pipeline_class, transformer_class
    from augmentum.image.hardware import get_gguf_families
    for f in get_gguf_families():
        if f["family"] == family_key or f["base_repo"] == family_key:
            return f["base_repo"], f["pipeline_class"], f["transformer_class"]
    raise HTTPException(400, f"Unknown gguf_family: {family_key}")


async def _run_import_prefetch_task(
    task_id: str,
    base_repo: str,
    model_name: str,
    dest_dir: str,
    pipeline_type: str,
    size_bytes: int,
) -> None:
    """Background-fire: pre-fetch base components after a custom import.

    Writes into the shared ``_pull_tasks`` dict so the existing
    GET /api/image/models/pull/{task_id} polling stack works unchanged.
    """
    task = _pull_tasks[task_id]
    try:
        prefetch_result = await _prefetch_gguf_base_components_for(
            base_repo, task, token=settings.image_huggingface_token,
        )
        if prefetch_result.get("status") == "error":
            _mark_install_failed_from_prefetch(task, prefetch_result)
            log.warning(
                "import_prefetch_failed",
                task_id=task_id,
                stage=prefetch_result.get("stage", ""),
                error=prefetch_result.get("error", ""),
            )
            return
        task["status"] = "complete"
        task["result"] = {
            "name": model_name,
            "path": dest_dir,
            "pipeline_type": pipeline_type,
            "size_bytes": size_bytes,
        }
        progress = task.get("progress", {})
        progress["percent"] = 100
        task["progress"] = progress
        system_events.publish("image.models.changed", {"reason": "import"})
    except Exception as exc:
        task["status"] = "error"
        task["error"] = str(exc)
        log.warning("import_prefetch_failed", task_id=task_id, error=str(exc))


@router.post("/models/import")
async def import_model(
    request: Request,
    file: UploadFile = File(...),
    name: str = Form(...),
    kind: str = Form(...),
    gguf_family: str = Form(""),
    gguf_base_repo: str = Form(""),
    gguf_pipeline_class: str = Form(""),
    gguf_transformer_class: str = Form(""),
    prefetch_base_components: bool = Form(True),
):
    """Import a custom image model from a user-supplied file.

    Supported kinds: ``gguf``, ``safetensors-single``, ``diffusers-zip``. The
    flow mirrors the URL importer: stream upload → write metadata → optionally
    pre-fetch base components in the background using the same ``_pull_tasks``
    polling infrastructure the catalog flow uses.
    """
    import os
    import re
    import shutil
    import uuid
    import zipfile
    from pathlib import Path

    from augmentum.image.model_manager import (
        ModelManager,
        _detect_pipeline_type,
        _get_dir_size,
    )

    # --- Sanitise name ---
    raw_name = (name or "").strip() or Path(file.filename or "model").stem
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", raw_name).strip("-")
    if not safe_name or len(safe_name) > 64 or ".." in safe_name:
        raise HTTPException(400, "Invalid model name (must be 1-64 chars, alnum + . _ -)")

    # --- Validate kind + extension against trust boundary ---
    filename = file.filename or ""
    ext = Path(filename).suffix.lower()
    pickle_exts = {".bin", ".pt", ".pth", ".ckpt"}
    allowed_exts = {".safetensors", ".gguf", ".zip"}
    if ext in pickle_exts:
        if not settings.image_allow_pickle_formats:
            raise HTTPException(
                400,
                f"{ext} format disabled by default (pickle deserialisation risk). "
                "Enable image_allow_pickle_formats to override.",
            )
        allowed_exts.add(ext)
    if ext not in allowed_exts:
        raise HTTPException(400, f"Unsupported file extension: {ext or '(no extension)'}")
    if kind == "gguf" and ext != ".gguf":
        raise HTTPException(400, "kind=gguf requires a .gguf file")
    if kind == "diffusers-zip" and ext != ".zip":
        raise HTTPException(400, "kind=diffusers-zip requires a .zip file")
    if kind == "safetensors-single" and ext not in {".safetensors", *pickle_exts}:
        raise HTTPException(400, "kind=safetensors-single requires a .safetensors (or pickle) file")

    # --- Early size cap via Content-Length (chunked uploads re-checked during stream) ---
    max_bytes = max(1, int(settings.image_upload_max_size_gb)) * 1024 ** 3
    content_length = request.headers.get("content-length")
    if content_length and content_length.isdigit() and int(content_length) > max_bytes:
        raise HTTPException(413, f"Upload exceeds {settings.image_upload_max_size_gb} GB cap")

    # --- Resolve family preset (overrides the three raw fields if set) ---
    base_repo, pipeline_class, transformer_class = _resolve_gguf_family_preset(
        gguf_family.strip(),
        gguf_base_repo.strip(),
        gguf_pipeline_class.strip(),
        gguf_transformer_class.strip(),
    )
    if kind == "gguf" and not (base_repo and pipeline_class and transformer_class):
        raise HTTPException(
            400,
            "GGUF imports need a known family OR all three of "
            "gguf_base_repo, gguf_pipeline_class, gguf_transformer_class.",
        )

    # --- Resolve model manager + destination ---
    model_mgr = getattr(request.app.state, "image_model_manager", None)
    if not model_mgr:
        model_dir = settings.image_model_dir or f"{settings.data_dir}/image_models"
        model_mgr = ModelManager(model_dir)

    dest_dir = os.path.join(model_mgr.model_dir, safe_name)
    if os.path.exists(dest_dir):
        if model_mgr._is_valid_model_dir(dest_dir):
            raise HTTPException(409, f"Model '{safe_name}' already exists; rename or delete first.")
        # Stale partial from a prior failed attempt — clean.
        shutil.rmtree(dest_dir, ignore_errors=True)
    Path(dest_dir).mkdir(parents=True, exist_ok=True)

    # --- Stream the upload to disk, enforcing cap as bytes arrive ---
    out_name = Path(filename).name if filename else f"model{ext}"
    out_path = os.path.join(dest_dir, out_name)
    total_written = 0
    try:
        with open(out_path, "wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                total_written += len(chunk)
                if total_written > max_bytes:
                    raise HTTPException(
                        413, f"Upload exceeds {settings.image_upload_max_size_gb} GB cap",
                    )
                f.write(chunk)
    except HTTPException:
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(dest_dir, ignore_errors=True)
        raise HTTPException(500, f"Upload write failed: {exc}")

    # --- Post-write: extract zip if needed ---
    if kind == "diffusers-zip":
        try:
            extract_dir = os.path.join(dest_dir, "_extract")
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(out_path) as zf:
                # Decompression-bomb guard — declared sizes only, runs
                # before any member inflates.
                from augmentum.utils.safe_archive import (
                    UnsafeArchiveError,
                    ensure_zip_sane,
                )
                try:
                    ensure_zip_sane(zf, source="image_pipeline_import")
                except UnsafeArchiveError as exc:
                    raise HTTPException(400, f"Unsafe zip: {exc}") from exc
                # Path-traversal guard: reject absolute or upward paths
                for member in zf.namelist():
                    if member.startswith("/") or member.startswith("\\") or ".." in Path(member).parts:
                        raise HTTPException(400, f"Zip contains unsafe path: {member}")
                # Reject zips that ship arbitrary Python (custom pipeline code)
                if any(m.endswith(".py") for m in zf.namelist()):
                    raise HTTPException(
                        400,
                        "Zip contains .py files — refusing to import (custom pipeline code would execute at load time).",
                    )
                zf.extractall(extract_dir)
            os.remove(out_path)
            # Locate model_index.json — root or one level down
            inner_root = extract_dir
            if not os.path.exists(os.path.join(inner_root, "model_index.json")):
                subdirs = [
                    d for d in os.listdir(extract_dir)
                    if os.path.isdir(os.path.join(extract_dir, d))
                ]
                match = next(
                    (
                        os.path.join(extract_dir, sd)
                        for sd in subdirs
                        if os.path.exists(os.path.join(extract_dir, sd, "model_index.json"))
                    ),
                    None,
                )
                if not match:
                    raise HTTPException(
                        400,
                        "Zip is not a valid diffusers model (no model_index.json at root or one level deep).",
                    )
                inner_root = match
            for item in os.listdir(inner_root):
                shutil.move(os.path.join(inner_root, item), os.path.join(dest_dir, item))
            shutil.rmtree(extract_dir, ignore_errors=True)
        except HTTPException:
            shutil.rmtree(dest_dir, ignore_errors=True)
            raise
        except Exception as exc:
            shutil.rmtree(dest_dir, ignore_errors=True)
            raise HTTPException(400, f"Zip extraction failed: {exc}")

    # --- Write gguf_meta.json so the loader knows pipeline + transformer classes ---
    if kind == "gguf":
        _write_gguf_meta(dest_dir, base_repo, pipeline_class, transformer_class)

    # --- Detect pipeline_type + persist DB record ---
    pipeline_type = _detect_pipeline_type(dest_dir)
    size_bytes = _get_dir_size(dest_dir)

    persistence = getattr(request.app.state, "image_persistence", None)
    if persistence:
        await persistence.save_model(
            name=safe_name,
            pipeline_type=pipeline_type.value,
            path=dest_dir,
            source="upload",
            size_bytes=size_bytes,
        )

    log.info(
        "model_imported",
        name=safe_name,
        kind=kind,
        size_bytes=total_written,
        base_repo=base_repo or "",
        prefetch=bool(kind == "gguf" and prefetch_base_components and base_repo),
    )

    # --- Optional background prefetch — reuses _pull_tasks polling stack ---
    if kind == "gguf" and prefetch_base_components and base_repo:
        task_id = uuid.uuid4().hex[:16]
        _pull_tasks[task_id] = {
            "status": "running",
            "source": f"import:{safe_name}",
            "progress": {
                "percent": 95,
                "downloaded": total_written,
                "total": total_written,
                "files_done": 1,
                "files_total": 1,
                "phase": "Pre-fetching base components from " + base_repo,
            },
        }
        bg = asyncio.create_task(
            _run_import_prefetch_task(
                task_id, base_repo, safe_name, dest_dir,
                pipeline_type.value, size_bytes,
            )
        )
        bg.add_done_callback(_pull_task_done_callback)
        return {
            "task_id": task_id,
            "status": "running",
            "name": safe_name,
            "path": dest_dir,
            "pipeline_type": pipeline_type.value,
            "size_bytes": size_bytes,
        }

    # No prefetch — install is fully complete
    return {
        "status": "complete",
        "name": safe_name,
        "path": dest_dir,
        "pipeline_type": pipeline_type.value,
        "size_bytes": size_bytes,
    }


@router.post("/models/rename")
async def rename_model(request: Request, body: dict):
    """Rename an image model (changes directory name and DB record)."""
    import os
    import shutil

    _require_image_info(request)

    old_name = body.get("old_name", "").strip()
    new_name = body.get("new_name", "").strip()
    if not old_name or not new_name:
        raise HTTPException(400, "old_name and new_name are required")
    if old_name == new_name:
        return {"status": "unchanged", "name": old_name}

    # Sanitize new name — no path separators
    if "/" in new_name or "\\" in new_name or ".." in new_name:
        raise HTTPException(400, "Invalid model name")

    model_mgr = getattr(request.app.state, "image_model_manager", None)
    if not model_mgr:
        raise HTTPException(503, "Image model manager not available")

    model_dir = model_mgr.model_dir
    old_path = os.path.join(model_dir, old_name)
    new_path = os.path.join(model_dir, new_name)

    if not os.path.isdir(old_path):
        raise HTTPException(404, f"Model '{old_name}' not found")
    if os.path.exists(new_path):
        raise HTTPException(409, f"Model '{new_name}' already exists")

    # Rename on disk
    shutil.move(old_path, new_path)

    # Update DB
    persistence = getattr(request.app.state, "image_persistence", None)
    if persistence:
        # Delete old record and re-save with new name
        await persistence.delete_model(old_name)
        from augmentum.image.model_manager import _detect_pipeline_type, _get_dir_size

        pipeline_type = _detect_pipeline_type(new_path)
        await persistence.save_model(
            name=new_name,
            pipeline_type=pipeline_type,
            path=new_path,
            size_bytes=_get_dir_size(new_path),
        )

    # Update loaded pipeline reference if this model is currently active
    pipeline_reg = getattr(request.app.state, "image_pipeline_registry", None)
    if pipeline_reg and pipeline_reg.current_model == old_path:
        pipeline_reg.current_model = new_path

    # Update image_active_settings so stale model names don't persist
    active_settings = getattr(request.app.state, "image_active_settings", None) or {}
    if active_settings.get("model") in (old_name, old_path):
        active_settings["model"] = new_name
        request.app.state.image_active_settings = active_settings
        store = getattr(request.app.state, "settings_store", None)
        if store:
            await store.set("image_active_settings", json.dumps(active_settings))

    log.info("model_renamed", old_name=old_name, new_name=new_name)
    return {"status": "renamed", "old_name": old_name, "new_name": new_name}


@router.delete("/models/{name:path}")
async def delete_model(name: str, request: Request):
    _require_image_info(request)

    model_mgr = getattr(request.app.state, "image_model_manager", None)
    persistence = getattr(request.app.state, "image_persistence", None)

    deleted = await model_mgr.delete_model(name)
    if not deleted:
        raise HTTPException(404, f"Model '{name}' not found")

    if persistence:
        await persistence.delete_model(name)

    system_events.publish("image.models.changed", {"reason": "delete"})
    return {"status": "deleted", "name": name}


@router.get("/models/variants")
async def model_variants(repo_id: str):
    """Fetch available GGUF quantization variants from a HuggingFace repo.

    Returns a list of variants with name, size, and allow_pattern for each.
    """
    import re

    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise HTTPException(500, "huggingface_hub is not installed")

    api = HfApi()
    try:
        # files_metadata=True is required to populate LFS file sizes —
        # without it, GGUF sizes come back as None and the dropdown shows 0 GB.
        repo_info = await asyncio.to_thread(
            lambda: api.repo_info(repo_id, files_metadata=True),
        )
        siblings = repo_info.siblings or []
    except Exception as exc:
        raise HTTPException(502, f"Failed to access repository: {exc}")

    # Extract GGUF files and parse quant types
    quant_re = re.compile(r"[_-]((?:Q\d[A-Za-z0-9_]*|F(?:16|32)|BF16|IQ\d[A-Za-z0-9_]*))\.gguf$", re.IGNORECASE)
    variants = []
    seen = set()
    for s in siblings:
        name = s.rfilename
        if not name.lower().endswith(".gguf"):
            continue
        m = quant_re.search(name)
        if not m:
            continue
        raw_quant = m.group(1)
        quant = raw_quant.upper()
        if quant in seen:
            continue
        seen.add(quant)
        size_gb = round((s.size or 0) / (1024**3), 2) if s.size else 0
        variants.append({
            "quant": quant,
            "filename": name,
            "size_gb": size_gb,
            # Preserve original casing in the pattern — fnmatch is case-sensitive
            # on Linux/Docker, so uppercasing here breaks repos that ship
            # lowercase filenames (e.g. SeeSee21/Z-Anime → z-anime-base-q4_k_s.gguf).
            "pattern": f"*{raw_quant}*",
        })

    # Sort by a rough quality ordering
    quant_order = {
        "Q2_K": 0, "Q2_K_S": 1,
        "IQ3_XS": 2, "IQ3_S": 3, "IQ3_M": 4, "Q3_K_S": 5, "Q3_K_M": 6, "Q3_K_L": 7,
        "IQ4_XS": 8, "IQ4_NL": 9, "Q4_0": 10, "Q4_K_S": 11, "Q4_K_M": 12,
        "Q5_0": 13, "Q5_K_S": 14, "Q5_K_M": 15,
        "Q6_K": 16, "Q8_0": 17,
        "BF16": 18, "F16": 19, "F32": 20,
    }
    variants.sort(key=lambda v: quant_order.get(v["quant"], 99))

    return {"repo_id": repo_id, "variants": variants}


@router.get("/gguf-families")
async def gguf_families():
    """Return GGUF family presets for the custom-import UI.

    Each family carries the three loader fields (``base_repo``,
    ``pipeline_class``, ``transformer_class``) plus filename hints used to
    auto-suggest a family when the user picks a .gguf file. Derived from the
    catalog — new catalog entries with ``gguf_base_repo`` become available
    presets automatically.
    """
    from augmentum.image.hardware import get_gguf_families

    return {"families": get_gguf_families()}


@router.get("/model-profile")
async def get_model_profile(model: str, pipeline: str = "", request: Request = None):
    """Get the effective profile for a model (defaults + user overrides)."""
    from augmentum.image.model_profiles import get_effective_profile

    store = getattr(request.app.state, "settings_store", None)
    profile = await get_effective_profile(model, pipeline, store)
    return profile


@router.put("/model-profile")
async def set_model_profile_override(request: Request):
    """Save user overrides for a model's profile."""
    from augmentum.image.model_profiles import save_user_overrides

    body = await request.json()
    model = body.get("model", "")
    overrides = body.get("overrides", {})
    if not model:
        raise HTTPException(400, "model is required")

    store = getattr(request.app.state, "settings_store", None)
    await save_user_overrides(model, overrides, store)
    return {"status": "saved", "model": model}


@router.delete("/model-profile")
async def reset_model_profile(request: Request):
    """Reset a model's profile to defaults (clear user overrides)."""
    from augmentum.image.model_profiles import clear_user_overrides

    body = await request.json()
    model = body.get("model", "")
    if not model:
        raise HTTPException(400, "model is required")

    store = getattr(request.app.state, "settings_store", None)
    await clear_user_overrides(model, store)
    return {"status": "reset", "model": model}


@router.get("/loras")
async def list_loras(request: Request):
    """List discovered LoRA adapters."""
    from augmentum.image.lora_manager import LoraManager

    model_dir = settings.image_model_dir or f"{settings.data_dir}/image_models"
    mgr = LoraManager(model_dir)
    loras = mgr.discover()
    return [
        {
            "name": l.name,
            "path": l.path,
            "trigger_words": l.trigger_words,
            "base_model": l.base_model,
            "size_mb": round(l.size_bytes / 1_048_576, 1) if l.size_bytes else 0,
        }
        for l in loras
    ]


@router.get("/loras/catalog")
async def lora_catalog(request: Request):
    """Curated LoRA catalog, sorted by compatibility with installed base models."""
    from augmentum.image.hardware import get_lora_catalog
    from augmentum.image.model_manager import ModelManager

    model_dir = settings.image_model_dir or f"{settings.data_dir}/image_models"
    model_mgr = ModelManager(model_dir)

    # Detect which base model types are installed
    installed_bases = set()
    for m in model_mgr.list_local_models():
        pt = m.get("pipeline_type", "")
        if pt:
            installed_bases.add(pt)
    # Also check the currently loaded pipeline type
    pipeline_reg = getattr(request.app.state, "image_pipeline_registry", None)
    if pipeline_reg and pipeline_reg.is_loaded:
        loaded_type = getattr(pipeline_reg._pipeline, "_detected_type", None)
        if loaded_type:
            installed_bases.add(loaded_type.value)

    # Also check which LoRAs are already installed
    from augmentum.image.lora_manager import LoraManager
    lora_mgr = LoraManager(model_dir)
    installed_lora_names = {l.name.lower() for l in lora_mgr.discover()}

    catalog = get_lora_catalog(list(installed_bases) if installed_bases else None)
    for entry in catalog:
        entry["installed"] = entry["name"].lower().replace(" ", "-") in installed_lora_names or \
                             entry["name"].lower().replace(" ", "_") in installed_lora_names or \
                             entry["name"].lower() in installed_lora_names
    return catalog


@router.get("/models/catalog")
async def model_catalog(request: Request):
    from augmentum.image.hardware import ModelTier, get_catalog_for_tier
    from augmentum.image.schemas import ModelCapabilities

    # Detect hardware tier — use live state if available, else detect fresh.
    # detect_hardware() shells out to nvidia-smi with a 5s timeout; if the
    # cached app.state.image_hardware was cleared (e.g. by a hot reload)
    # this fallback path would block the event loop for several seconds.
    # Hand it to a worker thread to keep the route async-clean.
    hw = getattr(request.app.state, "image_hardware", None)
    if hw:
        tier = hw.tier
    else:
        from augmentum.image.hardware import detect_hardware
        hw = await asyncio.to_thread(detect_hardware)
        tier = hw.tier

    # Check installed models if model manager is available
    model_mgr = getattr(request.app.state, "image_model_manager", None)
    local_names: set[str] = set()
    if model_mgr:
        try:
            local_models = model_mgr.list_local_models()
            local_names = {m["name"] for m in local_models}
        except Exception:
            log.debug("image_model_list_failed", exc_info=True)

    tier_order = {ModelTier.CPU: 0, ModelTier.LOW: 1, ModelTier.MEDIUM: 2, ModelTier.HIGH: 3}
    user_rank = tier_order.get(tier, 0)

    catalog = get_catalog_for_tier(tier)
    result = []
    for cm in catalog:
        # Check if installed by matching repo_id-based name or local name
        safe_name = cm.repo_id.replace("/", "--")
        installed = False
        installed_name = ""
        for candidate in (safe_name, cm.repo_id.split("/")[-1], cm.name):
            if candidate in local_names:
                installed = True
                installed_name = candidate
                break

        model_rank = tier_order.get(cm.min_tier, 0)
        compatible = model_rank <= user_rank

        result.append(CatalogModelInfo(
            repo_id=cm.repo_id,
            name=cm.name,
            description=cm.description,
            pipeline_type=cm.pipeline_type,
            size_gb=cm.size_gb,
            min_vram_mb=cm.min_vram_mb,
            min_tier=cm.min_tier.value,
            cpu_friendly=cm.cpu_friendly,
            speed_note=cm.speed_note,
            compatible=compatible,
            installed=installed,
            installed_name=installed_name,
            allow_patterns=cm.allow_patterns,
            capabilities=ModelCapabilities(
                txt2img=cm.cap_txt2img,
                img2img=cm.cap_img2img,
                inpaint=cm.cap_inpaint,
            ),
            precision_variants=cm.precision_variants,
        ))
    return result


# --- Hardware ---


@router.get("/hardware")
async def hardware_info(request: Request):
    _require_image_info(request)

    hw = getattr(request.app.state, "image_hardware", None)
    if not hw:
        from augmentum.image.hardware import detect_hardware
        # Worker-thread off-load — see the matching note in
        # generate_models_info above for why detect_hardware() can't
        # run inline on an async route.
        hw = await asyncio.to_thread(detect_hardware)

    # Return live VRAM (not stale startup snapshot) + system RAM
    from augmentum.image.hardware import get_system_ram_free_mb, refresh_vram_free

    live_vram = refresh_vram_free() if hw.device == "cuda" else 0
    sys_ram = get_system_ram_free_mb()

    resp = HardwareInfo(
        device=hw.device,
        device_name=hw.device_name,
        vram_total_mb=hw.vram_total_mb,
        vram_free_mb=live_vram or hw.vram_free_mb,
        tier=hw.tier.value,
        recommended_pipeline=hw.recommended_pipeline,
        recommended_model=hw.recommended_model,
    )
    # Add system RAM as extra field (not in the Pydantic model to avoid
    # breaking existing consumers)
    resp_dict = resp.model_dump()
    resp_dict["system_ram_free_mb"] = sys_ram
    return resp_dict


# --- History ---


@router.get("/history")
async def generation_history(
    request: Request,
    limit: int = 50,
    offset: int = 0,
    q: str = "",
    model: str = "",
    preset: str = "",
    sort: str = "newest",
    private: str = "",
    background: str = "",
    origin: str = "",
):
    # History is read-only — allow even when image generation is disabled
    persistence = getattr(request.app.state, "image_persistence", None)
    if not persistence:
        return HistoryPage(total=0, entries=[])

    user = request.scope.get("user")
    uid = user.id if user else ""
    if not uid:
        raise HTTPException(401, "Unauthorized")

    # Parse private filter: "" = all, "true" = private only, "false" = public only
    private_filter: bool | None = None
    if private == "true":
        private_filter = True
    elif private == "false":
        private_filter = False

    # Parse background filter
    background_filter: bool | None = None
    if background == "true":
        background_filter = True
    elif background == "false":
        background_filter = False

    # Provenance filter: '' = all, 'companion' = her generations only.
    # Closed vocabulary — anything else is treated as no filter.
    origin_filter = "companion" if origin == "companion" else ""

    entries = await persistence.list_generations(
        limit=limit, offset=offset, q=q, model=model, preset=preset,
        sort=sort, origin=origin_filter, private=private_filter,
        background=background_filter,
        user_id=uid,
    )
    total = await persistence.count_generations(
        q=q, model=model, preset=preset, origin=origin_filter,
        private=private_filter,
        background=background_filter, user_id=uid,
    )

    for entry in entries:
        entry.url = f"/api/image/{entry.image_id}"
    return HistoryPage(total=total, entries=entries)


@router.delete("/batch")
async def batch_delete_images(req: BatchDeleteRequest, request: Request):

    import os

    persistence = getattr(request.app.state, "image_persistence", None)
    if not persistence:
        raise HTTPException(503, "Image persistence not available")

    user = request.scope.get("user")
    uid = user.id if user else ""
    if not uid:
        raise HTTPException(401, "Unauthorized")

    deleted = []
    failed = []
    for image_id in req.image_ids:
        file_path = await persistence.delete_generation(image_id, user_id=uid)
        if file_path is None:
            failed.append(image_id)
            continue
        if os.path.exists(file_path):
            import contextlib
            with contextlib.suppress(OSError):
                os.remove(file_path)
        deleted.append(image_id)

    return {"deleted": deleted, "failed": failed}


@router.delete("/{image_id}")
async def delete_image(image_id: str, request: Request):

    import os

    persistence = getattr(request.app.state, "image_persistence", None)
    if not persistence:
        raise HTTPException(503, "Image persistence not available")

    user = request.scope.get("user")
    uid = user.id if user else ""
    if not uid:
        raise HTTPException(401, "Unauthorized")

    file_path = await persistence.delete_generation(image_id, user_id=uid)
    if file_path is None:
        raise HTTPException(404, "Image not found")

    if os.path.exists(file_path):
        import contextlib
        with contextlib.suppress(OSError):
            os.remove(file_path)

    return {"status": "deleted"}


# --- Privacy ---


class PrivacyToggleRequest(BaseModel):
    image_ids: list[str] = Field(..., min_length=1)
    is_private: bool


@router.post("/privacy")
async def toggle_privacy(req: PrivacyToggleRequest, request: Request):
    """Move images between public gallery and private section."""
    persistence = getattr(request.app.state, "image_persistence", None)
    if not persistence:
        raise HTTPException(503, "Image persistence not available")

    user = request.scope.get("user")
    uid = user.id if user else ""
    if not uid:
        raise HTTPException(401, "Unauthorized")

    updated = await persistence.set_private_batch(req.image_ids, req.is_private, user_id=uid)
    return {"updated": updated}


# --- Backgrounds collection ---


class BackgroundToggleRequest(BaseModel):
    image_ids: list[str] = Field(..., min_length=1)
    is_background: bool


@router.get("/backgrounds")
async def list_backgrounds(request: Request):
    """List all images tagged as backgrounds for the rotation feature."""
    persistence = getattr(request.app.state, "image_persistence", None)
    if not persistence:
        return {"entries": []}

    user = request.scope.get("user")
    uid = user.id if user else ""
    if not uid:
        raise HTTPException(401, "Unauthorized")

    entries = await persistence.list_backgrounds(user_id=uid)
    return {"entries": [
        {
            "image_id": e.image_id,
            "url": f"/api/image/{e.image_id}",
            "prompt": e.prompt,
            "created_at": e.created_at,
        }
        for e in entries
    ]}


@router.post("/backgrounds/toggle")
async def toggle_background(req: BackgroundToggleRequest, request: Request):
    """Add or remove images from the background rotation collection."""
    persistence = getattr(request.app.state, "image_persistence", None)
    if not persistence:
        raise HTTPException(503, "Image persistence not available")

    user = request.scope.get("user")
    uid = user.id if user else ""
    if not uid:
        raise HTTPException(401, "Unauthorized")

    updated = await persistence.set_background_batch(req.image_ids, req.is_background, user_id=uid)
    return {"updated": updated}


# --- Samplers ---


@router.get("/samplers")
async def list_samplers():
    """Return available sampler/scheduler names and aliases."""
    from augmentum.image.schedulers import get_available_samplers

    return get_available_samplers()


# --- Serve Image (must be last — catch-all path param) ---


@router.get("/{image_id}")
async def serve_image(image_id: str, request: Request):
    import os

    user = request.scope.get("user")
    uid = user.id if user else ""
    if not uid:
        raise HTTPException(401, "Unauthorized")

    # Try persistence DB first (local generations)
    persistence = getattr(request.app.state, "image_persistence", None)
    if persistence:
        gen = await persistence.get_generation(image_id, user_id=uid)
        if gen:
            file_path = gen["file_path"]
            if os.path.exists(file_path):
                return FileResponse(file_path, media_type="image/png")

    # Fallback: check output directory directly (cloud-generated images)
    output_dir = settings.image_output_dir or f"{settings.data_dir}/image_output"
    fallback_path = os.path.join(output_dir, f"{image_id}.png")
    if os.path.exists(fallback_path):
        return FileResponse(fallback_path, media_type="image/png")

    raise HTTPException(404, "Image not found")
